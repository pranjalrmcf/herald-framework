"""
Deliberative Multi-Hypothesis Arbitration (DMHA) for HERALD.

Paper terminology alignment:
    "Speculative RAG"                      → DeliberativeMultiHypothesisArbitration
    "Hypothesis-Competitive Retrieval"     → three parallel RetrievalStrategy branches
    "Neural Factuality Arbitration"        → _HypothesisJudge (LLM-as-Judge)
    "Speculation candidate"                → HypothesisCandidate
    "Winning strategy"                     → arbitrated winner

Architecture:
    Query ──► BROAD      strategy ──► Candidate A ──┐
          ──► ENTITY     strategy ──► Candidate B ──┼──► HypothesisJudge ──► Winner
          ──► RELATIONAL strategy ──► Candidate C ──┘

Retrieval Strategies:
    BROAD      – Wide semantic recall; maximises document coverage.
    ENTITY     – Entity-focused; boosts docs mentioning query entities.
    RELATIONAL – Relationship-chain evidence from graph subgraph;
                 maximises multi-hop reasoning depth.

Integration:
    ResearchAnalyst._execute_pipeline() calls this on the RESEARCH path
    as a drop-in replacement for a single evidence→answer call.
    All candidates stored in PipelineState.metadata["speculation_candidates"].
    The winner flows through QualityEvaluator → OutputGuardrails unchanged.

Backward compatibility:
    SpeculativeRAG is an alias for DeliberativeMultiHypothesisArbitration
    so existing orchestrator code needs no changes.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from research_analyst.core.models import (
    Answer,
    Citation,
    Evidence,
    ExecutionPath,
    NormalizedQuery,
    Query,
    RankedDocument,
    Relationship,
    Subgraph,
)
from research_analyst.core.exceptions import AnswerGenerationError, LLMError
from research_analyst.config import get_settings
from research_analyst.utils.helpers import generate_id
from research_analyst.utils.llm_client import get_llm_client
from research_analyst.utils.logger import get_logger


logger = get_logger()

_CANDIDATE_TIMEOUT_S = 90   # per-candidate generation timeout
_JUDGE_TIMEOUT_S     = 60   # per-candidate judge evaluation timeout


# ---------------------------------------------------------------------------
# Strategy enumeration
# ---------------------------------------------------------------------------

class RetrievalStrategy(str, Enum):
    """
    Three hypothesis generation strategies for contrastive arbitration.

    BROAD      — Maximise recall via wide semantic search.
    ENTITY     — Prioritise entity-centric document clusters.
    RELATIONAL — Surface multi-hop relationship chains from subgraph.
    """
    BROAD      = "broad"
    ENTITY     = "entity"
    RELATIONAL = "relational"


# ---------------------------------------------------------------------------
# Hypothesis candidate dataclass
# ---------------------------------------------------------------------------

@dataclass
class HypothesisCandidate:
    """
    A single hypothesis (candidate answer) produced by one retrieval strategy.

    Fields populated in order:
        1. candidate_id, strategy, evidence       — after evidence building
        2. answer, generation_time_ms             — after answer generation
        3. judge_scores, composite_judge_score    — after judge evaluation
        4. selected                               — after arbitration
    """
    candidate_id:          str
    strategy:              RetrievalStrategy
    answer:                Optional[Answer]
    evidence:              Evidence
    judge_scores:          Dict[str, float]  = field(default_factory=dict)
    composite_judge_score: float             = 0.0
    reasoning:             str               = ""
    generation_time_ms:    float             = 0.0
    selected:              bool              = False
    error:                 Optional[str]     = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id":          self.candidate_id,
            "strategy":              self.strategy.value,
            "composite_judge_score": round(self.composite_judge_score, 4),
            "judge_scores":          {k: round(v, 4) for k, v in self.judge_scores.items()},
            "selected":              self.selected,
            "generation_time_ms":    round(self.generation_time_ms, 1),
            "error":                 self.error,
            "num_claims":            len(self.evidence.claims) if self.evidence else 0,
            "answer_length":         len(self.answer.text) if self.answer else 0,
        }


# Alias for backward compatibility with existing orchestrator code
SpeculationCandidate = HypothesisCandidate


# ---------------------------------------------------------------------------
# Evidence builder — one method per strategy
# ---------------------------------------------------------------------------

class _EvidenceBuilder:
    """
    Constructs strategy-specific Evidence objects from ranked documents
    and an optional graph subgraph.
    """

    def __init__(self, settings, log):
        self._settings = settings
        self._log      = log

    def build_broad(
        self,
        ranked_docs:  List[RankedDocument],
        base_evidence:Evidence,
    ) -> Evidence:
        """Use all ranked documents; no entity/graph filtering."""
        from research_analyst.core.models import Claim, Evidence as Ev
        from research_analyst.utils.helpers import generate_id

        docs = [rd.document for rd in ranked_docs[:10]]
        # Reuse claims from base_evidence that are backed by these docs
        doc_ids = {d.doc_id for d in docs}
        claims  = [c for c in base_evidence.claims
                   if any(s in doc_ids for s in c.supporting_sources)]
        return Ev(
            evidence_id          = generate_id("ev"),
            claims               = claims or base_evidence.claims[:],
            supporting_documents = docs,
            counter_arguments    = base_evidence.counter_arguments[:],
            summary              = base_evidence.summary,
        )

    def build_entity(
        self,
        ranked_docs:       List[RankedDocument],
        base_evidence:     Evidence,
        entities_mentioned:List[str],
    ) -> Evidence:
        """Boost documents that mention query entities."""
        from research_analyst.core.models import Evidence as Ev
        from research_analyst.utils.helpers import generate_id

        entity_lower = {e.lower() for e in entities_mentioned}

        def entity_score(rd: RankedDocument) -> float:
            text = (rd.document.title + " " + (rd.document.snippet or "")).lower()
            hits = sum(1 for e in entity_lower if e in text)
            return rd.final_score + hits * 0.15

        sorted_docs = sorted(ranked_docs, key=entity_score, reverse=True)
        docs     = [rd.document for rd in sorted_docs[:8]]
        doc_ids  = {d.doc_id for d in docs}
        claims   = [c for c in base_evidence.claims
                    if any(s in doc_ids for s in c.supporting_sources)]
        return Ev(
            evidence_id          = generate_id("ev"),
            claims               = claims or base_evidence.claims[:],
            supporting_documents = docs,
            counter_arguments    = base_evidence.counter_arguments[:],
            summary              = base_evidence.summary,
        )

    def build_relational(
        self,
        ranked_docs:  List[RankedDocument],
        base_evidence:Evidence,
        subgraph:     Optional[Subgraph],
    ) -> Evidence:
        """
        Prioritise documents that appear in the knowledge graph subgraph.
        If no subgraph is available, falls back to BROAD evidence.
        """
        from research_analyst.core.models import Evidence as Ev
        from research_analyst.utils.helpers import generate_id

        if not subgraph or not subgraph.relationships:
            return self.build_broad(ranked_docs, base_evidence)

        # Gather doc IDs referenced in subgraph relationships
        graph_doc_ids: set = set()
        for rel in subgraph.relationships:
            graph_doc_ids.add(rel.source_doc_id)

        def graph_score(rd: RankedDocument) -> float:
            bonus = 0.30 if rd.document.doc_id in graph_doc_ids else 0.0
            return rd.final_score + bonus

        sorted_docs = sorted(ranked_docs, key=graph_score, reverse=True)
        docs     = [rd.document for rd in sorted_docs[:8]]
        doc_ids  = {d.doc_id for d in docs}
        claims   = [c for c in base_evidence.claims
                    if any(s in doc_ids for s in c.supporting_sources)]

        # Attach relationship chains for multi-hop context
        chains: List[List[Relationship]] = []
        if subgraph.relationships:
            # Group into chains of ≤3 hops by shared entity
            seen: set = set()
            chain: List[Relationship] = []
            for rel in subgraph.relationships[:30]:
                key = (rel.subject, rel.predicate, rel.object)
                if key not in seen:
                    seen.add(key)
                    chain.append(rel)
                    if len(chain) == 3:
                        chains.append(chain)
                        chain = []
            if chain:
                chains.append(chain)

        return Ev(
            evidence_id          = generate_id("ev"),
            claims               = claims or base_evidence.claims[:],
            supporting_documents = docs,
            counter_arguments    = base_evidence.counter_arguments[:],
            relationship_chains  = chains or None,
            summary              = base_evidence.summary,
        )


# ---------------------------------------------------------------------------
# Hypothesis judge — LLM arbitration
# ---------------------------------------------------------------------------

class _HypothesisJudge:
    """
    Neural Factuality Arbitration: scores each hypothesis candidate on
    grounding, coherence, and completeness, then picks the best one.
    """

    def __init__(self):
        self._settings   = get_settings()
        self._llm_client = get_llm_client()
        self._log        = get_logger()

    def score_candidate(
        self,
        query:     str,
        candidate: HypothesisCandidate,
    ) -> HypothesisCandidate:
        """
        Evaluate a single hypothesis candidate with LLM-as-Judge.
        Scores: grounding, coherence, completeness (each 0-1).
        """
        if candidate.error or not candidate.answer:
            candidate.composite_judge_score = 0.0
            return candidate

        evidence_summary = self._summarise_evidence(candidate.evidence)
        prompt = (
            f"You are an expert research evaluator.\n\n"
            f"QUERY: {query}\n\n"
            f"ANSWER:\n{candidate.answer.text[:2000]}\n\n"
            f"EVIDENCE SUMMARY:\n{evidence_summary}\n\n"
            "Evaluate on THREE dimensions (0.0-1.0):\n"
            "  grounding   — every claim is supported by the evidence\n"
            "  coherence   — answer is logically structured and clear\n"
            "  completeness — all important aspects of the query are addressed\n\n"
            "Return JSON ONLY:\n"
            '{"grounding": 0.0, "coherence": 0.0, "completeness": 0.0, "reasoning": ""}'
        )
        import json
        try:
            raw = self._llm_client.generate(
                prompt        = prompt,
                system_prompt = (
                    "You are a precise research answer evaluator. "
                    "Respond with valid JSON only."
                ),
                max_tokens    = 300,
                temperature   = 0.0,
                json_mode     = True,
            )
            # Strip markdown fences if model ignored json_mode
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                cleaned = parts[1].lstrip("json").strip() if len(parts) > 1 else cleaned
            result = json.loads(cleaned)

            g = float(result.get("grounding",    0.5))
            c = float(result.get("coherence",    0.5))
            k = float(result.get("completeness", 0.5))
            candidate.judge_scores = {
                "grounding":    round(g, 3),
                "coherence":    round(c, 3),
                "completeness": round(k, 3),
            }
            candidate.composite_judge_score = round(g * 0.40 + c * 0.30 + k * 0.30, 4)
            candidate.reasoning = result.get("reasoning", "")

        except Exception as e:
            self._log.warning(
                "HypothesisJudge scoring failed",
                candidate_id = candidate.candidate_id,
                strategy     = candidate.strategy.value,
                error        = str(e),
            )
            candidate.composite_judge_score = 0.0

        return candidate

    def pick_winner(
        self,
        candidates: List[HypothesisCandidate],
    ) -> HypothesisCandidate:
        """
        Select the hypothesis with the highest composite judge score.
        Falls back to the first viable candidate if all scores are 0.
        """
        scored = [c for c in candidates if not c.error and c.answer]
        if not scored:
            self._log.warning("No viable DMHA candidates — returning first")
            return candidates[0]

        winner = max(scored, key=lambda c: c.composite_judge_score)
        winner.selected = True
        self._log.info(
            "Hypothesis winner selected",
            strategy = winner.strategy.value,
            score    = winner.composite_judge_score,
        )
        return winner

    @staticmethod
    def _summarise_evidence(evidence: Evidence) -> str:
        lines = ["Claims:"]
        for i, c in enumerate(evidence.claims[:8], 1):
            lines.append(f"  {i}. {c.text[:120]} (conf={c.confidence:.2f})")
        lines.append("Sources:")
        for i, d in enumerate(evidence.supporting_documents[:4], 1):
            lines.append(f"  [{i}] {d.title}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class — Deliberative Multi-Hypothesis Arbitration
# ---------------------------------------------------------------------------

class DeliberativeMultiHypothesisArbitration:
    """
    HERALD's Deliberative Multi-Hypothesis Arbitration (DMHA) module.

    Generates N hypothesis candidates in parallel, each using a different
    retrieval strategy, then uses a Neural Factuality Arbitration judge
    to select the best one.

    Parallelism model:
        - Candidate generation: ThreadPoolExecutor (max_workers = num_strategies)
        - Judge evaluation:      ThreadPoolExecutor (max_workers = num_strategies)
        - Timeouts enforced at both stages
    """

    def __init__(self):
        self._settings       = get_settings()
        self._log            = get_logger()
        self._llm_client     = get_llm_client()
        self._evidence_builder = _EvidenceBuilder(self._settings, self._log)
        self._judge          = _HypothesisJudge()
        self._executor       = ThreadPoolExecutor(
            max_workers     = 3,
            thread_name_prefix = "dmha_candidate",
        )
        self._judge_executor = ThreadPoolExecutor(
            max_workers     = 3,
            thread_name_prefix = "dmha_judge",
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        query:          Query,
        normalized_query:NormalizedQuery,
        base_evidence:  Evidence,
        ranked_docs:    List[RankedDocument],
        subgraph:       Optional[Subgraph],
        answer_generator,
    ) -> Tuple[HypothesisCandidate, List[HypothesisCandidate]]:
        """
        Run the full DMHA pipeline.

        Args:
            query:            Original Query object.
            normalized_query: Normalised query with entities and intent.
            base_evidence:    Evidence from the standard context builder.
            ranked_docs:      All ranked documents for this query.
            subgraph:         Graph subgraph (may be None on FAST path).
            answer_generator: AnswerGenerator instance.

        Returns:
            (winning_candidate, all_candidates)
        """
        enabled = getattr(self._settings, "speculative_rag_enabled", True)
        if not enabled:
            raise RuntimeError("DMHA called but speculative_rag_enabled=False")

        self._log.info(
            "DMHA: generating hypothesis candidates",
            query = query.text[:80],
        )

        # Step 1 — build strategy-specific evidence objects
        strategy_evidences = self._build_strategy_evidences(
            base_evidence     = base_evidence,
            ranked_docs       = ranked_docs,
            subgraph          = subgraph,
            entities_mentioned= normalized_query.entities_mentioned,
        )

        # Step 2 — generate candidates in parallel
        candidates = self._generate_candidates_parallel(
            query             = query,
            normalized_query  = normalized_query,
            strategy_evidences= strategy_evidences,
            answer_generator  = answer_generator,
        )

        # Step 3 — score candidates with judge
        candidates = self._score_candidates_parallel(
            query      = query.text,
            candidates = candidates,
        )

        # Step 4 — arbitrate winner
        winner = self._judge.pick_winner(candidates)

        self._log.info(
            "DMHA: arbitration complete",
            winner_strategy  = winner.strategy.value,
            winner_score     = winner.composite_judge_score,
            num_candidates   = len(candidates),
        )

        return winner, candidates

    # ------------------------------------------------------------------ #
    #  Internal steps                                                     #
    # ------------------------------------------------------------------ #

    def _build_strategy_evidences(
        self,
        base_evidence:      Evidence,
        ranked_docs:        List[RankedDocument],
        subgraph:           Optional[Subgraph],
        entities_mentioned: List[str],
    ) -> Dict[RetrievalStrategy, Evidence]:
        return {
            RetrievalStrategy.BROAD: self._evidence_builder.build_broad(
                ranked_docs, base_evidence,
            ),
            RetrievalStrategy.ENTITY: self._evidence_builder.build_entity(
                ranked_docs, base_evidence, entities_mentioned,
            ),
            RetrievalStrategy.RELATIONAL: self._evidence_builder.build_relational(
                ranked_docs, base_evidence, subgraph,
            ),
        }

    def _generate_candidates_parallel(
        self,
        query:              Query,
        normalized_query:   NormalizedQuery,
        strategy_evidences: Dict[RetrievalStrategy, Evidence],
        answer_generator,
    ) -> List[HypothesisCandidate]:
        futures = {
            self._executor.submit(
                self._generate_single_candidate,
                query, normalized_query, strategy, evidence, answer_generator,
            ): strategy
            for strategy, evidence in strategy_evidences.items()
        }
        candidates: List[HypothesisCandidate] = []
        for fut in as_completed(futures, timeout=_CANDIDATE_TIMEOUT_S + 10):
            strategy = futures[fut]
            try:
                candidates.append(fut.result(timeout=_CANDIDATE_TIMEOUT_S))
            except FutureTimeoutError:
                self._log.warning("DMHA candidate timed out", strategy=strategy.value)
                candidates.append(HypothesisCandidate(
                    candidate_id = generate_id("cand"),
                    strategy     = strategy,
                    answer       = None,
                    evidence     = strategy_evidences[strategy],
                    error        = "timeout",
                ))
            except Exception as e:
                self._log.warning(
                    "DMHA candidate failed",
                    strategy = strategy.value,
                    error    = str(e),
                )
                candidates.append(HypothesisCandidate(
                    candidate_id = generate_id("cand"),
                    strategy     = strategy,
                    answer       = None,
                    evidence     = strategy_evidences[strategy],
                    error        = str(e),
                ))
        return candidates

    def _generate_single_candidate(
        self,
        query:            Query,
        normalized_query: NormalizedQuery,
        strategy:         RetrievalStrategy,
        evidence:         Evidence,
        answer_generator,
    ) -> HypothesisCandidate:
        t0 = time.time()
        candidate_id = generate_id("cand")
        try:
            answer = answer_generator.generate_answer(
                query            = query,
                normalized_query = normalized_query,
                evidence         = evidence,
                execution_path   = ExecutionPath.RESEARCH,
            )
            return HypothesisCandidate(
                candidate_id       = candidate_id,
                strategy           = strategy,
                answer             = answer,
                evidence           = evidence,
                generation_time_ms = (time.time() - t0) * 1000,
            )
        except Exception as e:
            return HypothesisCandidate(
                candidate_id       = candidate_id,
                strategy           = strategy,
                answer             = None,
                evidence           = evidence,
                generation_time_ms = (time.time() - t0) * 1000,
                error              = str(e),
            )

    def _score_candidates_parallel(
        self,
        query:      str,
        candidates: List[HypothesisCandidate],
    ) -> List[HypothesisCandidate]:
        futures = {
            self._judge_executor.submit(
                self._judge.score_candidate, query, cand
            ): cand.candidate_id
            for cand in candidates
        }
        scored: List[HypothesisCandidate] = []
        for fut in as_completed(futures, timeout=_JUDGE_TIMEOUT_S + 10):
            try:
                scored.append(fut.result(timeout=_JUDGE_TIMEOUT_S))
            except Exception as e:
                self._log.warning("DMHA judge scoring failed", error=str(e))
                # Return the unscored candidate so we don't lose it
                for cand in candidates:
                    if cand.candidate_id == futures[fut]:
                        scored.append(cand)
                        break
        return scored

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def is_enabled(self) -> bool:
        return bool(getattr(self._settings, "speculative_rag_enabled", True))

    def shutdown(self):
        self._executor.shutdown(wait=False)
        self._judge_executor.shutdown(wait=False)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backward-compatibility alias — existing orchestrator imports SpeculativeRAG
# ---------------------------------------------------------------------------
SpeculativeRAG = DeliberativeMultiHypothesisArbitration