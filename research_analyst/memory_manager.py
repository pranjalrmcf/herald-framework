"""
Memory Manager for the research analyst system.

Provides cross-session episodic memory backed by SQLite.
Stores entities, relationships, and answer summaries from past sessions
and injects relevant context into current query synthesis.

Architecture:
    Session ends → MemoryManager.save_session()
                   stores: entities seen, relationships found, answer summary,
                           query text, quality score, timestamp

    New query → MemoryManager.retrieve_context(query, entities)
                searches: entity overlap + semantic similarity on past queries
                returns: MemoryContext injected into Evidence before synthesis

Tables:
    sessions     - one row per completed pipeline run
    memory_facts - individual entity/relationship facts extracted per session
    query_index  - query text + embedding hash for semantic lookup

Integration:
    orchestrator._execute_pipeline():
        memory_ctx = self.memory_manager.retrieve_context(
            query_text=state.query.text,
            entities=state.normalized_query.entities_mentioned,
            session_id=state.query.session_id,
        )
        state.evidence.summary = memory_ctx.inject_into_evidence_summary(
            state.evidence.summary
        )
        state.metadata['memory_context'] = memory_ctx.to_dict()
        # After pipeline completes:
        self.memory_manager.save_session(state)
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import List, Optional, Dict, Any, Set

from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id, generate_hash


logger = get_logger()

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MemoryFact:
    """A single remembered fact — entity or relationship."""
    fact_id: str
    session_id: str
    fact_type: str          # "entity" | "relationship" | "answer_summary"
    subject: str            # entity name or relationship subject
    predicate: Optional[str] = None   # for relationships
    obj: Optional[str] = None         # for relationships
    confidence: float = 0.5
    source_query: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MemoryContext:
    """
    Context retrieved from memory for a given query.
    Injected into synthesis before answer generation.
    """
    relevant_facts: List[MemoryFact] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    related_answers: List[str] = field(default_factory=list)
    entity_history: Dict[str, List[str]] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return (
            not self.relevant_facts
            and not self.related_queries
            and not self.related_answers
        )

    def inject_into_evidence_summary(self, current_summary: Optional[str]) -> str:
        """
        Prepend memory context to evidence summary so synthesis prompt
        is aware of what the system has seen before.
        """
        if self.is_empty:
            return current_summary or ""

        lines = ["[MEMORY CONTEXT FROM PAST SESSIONS]"]

        if self.related_queries:
            lines.append("Related past queries:")
            for q in self.related_queries[:3]:
                lines.append(f"  - {q}")

        if self.related_answers:
            lines.append("Relevant past answer summaries:")
            for a in self.related_answers[:2]:
                lines.append(f"  - {a[:200]}...")

        if self.relevant_facts:
            lines.append("Known facts from memory:")
            for fact in self.relevant_facts[:10]:
                if fact.fact_type == "entity":
                    lines.append(f"  - Entity known: {fact.subject}")
                elif fact.fact_type == "relationship":
                    lines.append(
                        f"  - Relationship: {fact.subject} "
                        f"--[{fact.predicate}]--> {fact.obj} "
                        f"(conf: {fact.confidence:.2f})"
                    )

        lines.append("[END MEMORY CONTEXT]")
        memory_block = "\n".join(lines)

        if current_summary:
            return f"{memory_block}\n\n{current_summary}"
        return memory_block

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_facts": len(self.relevant_facts),
            "num_related_queries": len(self.related_queries),
            "num_related_answers": len(self.related_answers),
            "retrieval_time_ms": self.retrieval_time_ms,
            "facts_preview": [
                {
                    "type": f.fact_type,
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.obj,
                    "confidence": f.confidence,
                }
                for f in self.relevant_facts[:5]
            ],
        }


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Manages cross-session episodic memory using SQLite.

    Thread-safe via a write lock — reads are lock-free (SQLite WAL mode).
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()

        self.enabled: bool = getattr(self.settings, "memory_enabled", True)
        db_path: str = getattr(self.settings, "memory_db_path", "./data/memory.db")
        self.session_ttl_days: int = getattr(self.settings, "memory_session_ttl_days", 30)
        self.max_facts_per_session: int = 200
        self.max_context_facts: int = 20
        self.min_entity_overlap: int = 1  # at least 1 shared entity to be "related"

        self._write_lock = Lock()

        if self.enabled:
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self.logger.info(
                "MemoryManager initialised",
                db_path=str(self._db_path),
                ttl_days=self.session_ttl_days,
            )
        else:
            self.logger.info("MemoryManager disabled via settings")

    # ------------------------------------------------------------------ #
    #  DB initialisation                                                  #
    # ------------------------------------------------------------------ #

    def _get_conn(self) -> sqlite3.Connection:
        """Create a fresh connection per call (safe for multi-thread)."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._write_lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id      TEXT PRIMARY KEY,
                        query_id        TEXT NOT NULL,
                        query_text      TEXT NOT NULL,
                        query_hash      TEXT NOT NULL,
                        answer_summary  TEXT,
                        composite_score REAL DEFAULT 0.0,
                        execution_path  TEXT,
                        entities_json   TEXT,
                        created_at      TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS memory_facts (
                        fact_id         TEXT PRIMARY KEY,
                        session_id      TEXT NOT NULL,
                        fact_type       TEXT NOT NULL,
                        subject         TEXT NOT NULL,
                        predicate       TEXT,
                        object          TEXT,
                        confidence      REAL DEFAULT 0.5,
                        source_query    TEXT,
                        created_at      TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_sessions_query_hash
                        ON sessions(query_hash);
                    CREATE INDEX IF NOT EXISTS idx_sessions_created
                        ON sessions(created_at);
                    CREATE INDEX IF NOT EXISTS idx_facts_session
                        ON memory_facts(session_id);
                    CREATE INDEX IF NOT EXISTS idx_facts_subject
                        ON memory_facts(subject);
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    #  Save session after pipeline completes                             #
    # ------------------------------------------------------------------ #

    def save_session(self, state) -> bool:
        """
        Persist entities, relationships, and answer summary from a
        completed PipelineState.

        Args:
            state: PipelineState after full pipeline execution.

        Returns:
            True if saved successfully.
        """
        if not self.enabled:
            return False

        if not state.answer:
            return False

        try:
            query_text = state.query.text
            query_id = state.query.metadata.get("query_id", generate_id("q"))
            session_id = state.query.session_id or generate_id("sess")
            query_hash = generate_hash(query_text.lower().strip())

            # Collect entities
            entities: List[str] = []
            if state.normalized_query:
                entities = state.normalized_query.entities_mentioned or []

            # Answer summary (first 500 chars)
            answer_summary = (state.answer.text or "")[:500]

            # Composite score
            composite_score = 0.0
            if state.quality_metrics and state.quality_metrics.composite_score:
                composite_score = state.quality_metrics.composite_score

            # Execution path
            exec_path = ""
            if state.routing_decision:
                ep = state.routing_decision.execution_path
                exec_path = ep.value if hasattr(ep, "value") else str(ep)

            with self._write_lock:
                conn = self._get_conn()
                try:
                    # Upsert session
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO sessions
                            (session_id, query_id, query_text, query_hash,
                             answer_summary, composite_score, execution_path,
                             entities_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id, query_id, query_text, query_hash,
                            answer_summary, composite_score, exec_path,
                            json.dumps(entities),
                            datetime.utcnow().isoformat(),
                        ),
                    )

                    # Save entity facts
                    facts_saved = 0
                    for entity in entities:
                        if facts_saved >= self.max_facts_per_session:
                            break
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO memory_facts
                                (fact_id, session_id, fact_type, subject,
                                 confidence, source_query, created_at)
                            VALUES (?, ?, 'entity', ?, 0.8, ?, ?)
                            """,
                            (
                                generate_id("fact"), session_id, entity,
                                query_text[:200], datetime.utcnow().isoformat(),
                            ),
                        )
                        facts_saved += 1

                    # Save relationship facts from subgraph
                    if state.relevant_subgraph:
                        for rel in state.relevant_subgraph.relationships:
                            if facts_saved >= self.max_facts_per_session:
                                break
                            predicate = (
                                rel.predicate.value
                                if hasattr(rel.predicate, "value")
                                else str(rel.predicate)
                            )
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO memory_facts
                                    (fact_id, session_id, fact_type, subject,
                                     predicate, object, confidence,
                                     source_query, created_at)
                                VALUES (?, ?, 'relationship', ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    generate_id("fact"), session_id,
                                    rel.subject, predicate,
                                    rel.object, rel.confidence,
                                    query_text[:200],
                                    datetime.utcnow().isoformat(),
                                ),
                            )
                            facts_saved += 1

                    conn.commit()
                    self.logger.info(
                        "Memory session saved",
                        session_id=session_id,
                        facts_saved=facts_saved,
                        entities=len(entities),
                    )
                    return True

                finally:
                    conn.close()

        except Exception as e:
            self.logger.error("Memory save_session failed", error=str(e))
            return False

    # ------------------------------------------------------------------ #
    #  Retrieve context for a new query                                  #
    # ------------------------------------------------------------------ #

    def retrieve_context(
        self,
        query_text: str,
        entities: List[str],
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> MemoryContext:
        """
        Retrieve relevant memory context for the current query.

        Matching strategy (in order of priority):
          1. Sessions sharing ≥1 entity with current query
          2. Sessions whose query_hash matches (exact repeat)
          3. Keyword overlap on query text

        Args:
            query_text:  Current query text.
            entities:    Entities extracted from current query.
            session_id:  Current session (excluded from lookup).
            top_k:       Max sessions to pull context from.

        Returns:
            MemoryContext with relevant facts and related queries.
        """
        if not self.enabled:
            return MemoryContext()

        start = time.time()

        try:
            conn = self._get_conn()
            try:
                cutoff = (
                    datetime.utcnow() - timedelta(days=self.session_ttl_days)
                ).isoformat()

                # ---- Find related sessions by entity overlap ----
                related_session_ids: Set[str] = set()

                if entities:
                    for entity in entities:
                        rows = conn.execute(
                            """
                            SELECT DISTINCT session_id FROM memory_facts
                            WHERE fact_type = 'entity'
                              AND LOWER(subject) LIKE ?
                              AND session_id != ?
                              AND created_at > ?
                            LIMIT 20
                            """,
                            (
                                f"%{entity.lower()}%",
                                session_id or "__none__",
                                cutoff,
                            ),
                        ).fetchall()
                        for row in rows:
                            related_session_ids.add(row["session_id"])

                # ---- Also match by query keyword overlap ----
                query_words = [
                    w.lower() for w in query_text.split()
                    if len(w) > 3
                ][:8]

                for word in query_words:
                    rows = conn.execute(
                        """
                        SELECT session_id FROM sessions
                        WHERE LOWER(query_text) LIKE ?
                          AND session_id != ?
                          AND created_at > ?
                        LIMIT 5
                        """,
                        (f"%{word}%", session_id or "__none__", cutoff),
                    ).fetchall()
                    for row in rows:
                        related_session_ids.add(row["session_id"])

                if not related_session_ids:
                    return MemoryContext(
                        retrieval_time_ms=round((time.time() - start) * 1000, 1)
                    )

                # Limit to top_k sessions ordered by composite_score desc
                placeholders = ",".join("?" * len(related_session_ids))
                session_rows = conn.execute(
                    f"""
                    SELECT session_id, query_text, answer_summary, composite_score
                    FROM sessions
                    WHERE session_id IN ({placeholders})
                    ORDER BY composite_score DESC, created_at DESC
                    LIMIT ?
                    """,
                    (*list(related_session_ids), top_k),
                ).fetchall()

                related_queries = [r["query_text"] for r in session_rows]
                related_answers = [
                    r["answer_summary"] for r in session_rows
                    if r["answer_summary"]
                ]

                # ---- Fetch facts from related sessions ----
                if session_rows:
                    top_session_ids = [r["session_id"] for r in session_rows]
                    ph2 = ",".join("?" * len(top_session_ids))
                    fact_rows = conn.execute(
                        f"""
                        SELECT fact_id, session_id, fact_type, subject,
                               predicate, object, confidence, source_query,
                               created_at
                        FROM memory_facts
                        WHERE session_id IN ({ph2})
                          AND confidence >= 0.5
                        ORDER BY confidence DESC
                        LIMIT ?
                        """,
                        (*top_session_ids, self.max_context_facts),
                    ).fetchall()

                    facts = [
                        MemoryFact(
                            fact_id=r["fact_id"],
                            session_id=r["session_id"],
                            fact_type=r["fact_type"],
                            subject=r["subject"],
                            predicate=r["predicate"],
                            obj=r["object"],
                            confidence=r["confidence"],
                            source_query=r["source_query"] or "",
                            created_at=r["created_at"],
                        )
                        for r in fact_rows
                    ]
                else:
                    facts = []

                # ---- Build entity history dict ----
                entity_history: Dict[str, List[str]] = {}
                for fact in facts:
                    if fact.fact_type == "relationship" and fact.predicate:
                        entity_history.setdefault(fact.subject, []).append(
                            f"{fact.predicate} → {fact.obj}"
                        )

                ctx = MemoryContext(
                    relevant_facts=facts,
                    related_queries=related_queries,
                    related_answers=related_answers,
                    entity_history=entity_history,
                    retrieval_time_ms=round((time.time() - start) * 1000, 1),
                )

                self.logger.info(
                    "Memory context retrieved",
                    num_facts=len(facts),
                    num_related_sessions=len(related_session_ids),
                    retrieval_ms=ctx.retrieval_time_ms,
                )

                return ctx

            finally:
                conn.close()

        except Exception as e:
            self.logger.error("Memory retrieve_context failed", error=str(e))
            return MemoryContext(
                retrieval_time_ms=round((time.time() - start) * 1000, 1)
            )

    # ------------------------------------------------------------------ #
    #  Maintenance                                                        #
    # ------------------------------------------------------------------ #

    def cleanup_expired(self) -> int:
        """
        Delete sessions (and their facts via CASCADE) older than TTL.

        Returns:
            Number of sessions deleted.
        """
        if not self.enabled:
            return 0

        cutoff = (
            datetime.utcnow() - timedelta(days=self.session_ttl_days)
        ).isoformat()

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE created_at < ?", (cutoff,)
                )
                deleted = cursor.rowcount
                conn.commit()
                if deleted:
                    self.logger.info(
                        "Memory cleanup: expired sessions deleted",
                        count=deleted,
                        cutoff=cutoff,
                    )
                return deleted
            finally:
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Return memory store statistics."""
        if not self.enabled:
            return {"enabled": False}

        conn = self._get_conn()
        try:
            session_count = conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]
            fact_count = conn.execute(
                "SELECT COUNT(*) FROM memory_facts"
            ).fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(created_at) FROM sessions"
            ).fetchone()[0]
            return {
                "enabled": True,
                "total_sessions": session_count,
                "total_facts": fact_count,
                "oldest_session": oldest,
                "ttl_days": self.session_ttl_days,
            }
        finally:
            conn.close()