"""
Upgraded CacheManager for the research analyst system.

Adds on top of the original CacheManager:
  1. Semantic cache layer   — query embedding similarity lookup
  2. Tiered TTL by intent   — SEMANTIC=24h, ENTITY=6h, RELATIONAL=2h, TEMPORAL=15m
  3. Component-level caches — entity/relationship extraction, LLM judge scores
  4. Speculation candidate cache — per-strategy caching
  5. Hit/miss counters for all backends (disk, memory, redis)
  6. Cache warming from evaluation history

Drop-in replacement for the existing CacheManager.
Import as:
    from research_analyst.caching.cache_manager import CacheManager

Schema change: all new cache keys are prefixed to avoid collisions with
the original cache entries.
"""

import json
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Dict, List, Tuple

from research_analyst.core.models import (
    Query,
    NormalizedQuery,
    Answer,
    Document,
    PipelineResponse,
    QueryIntent,
)
from research_analyst.core.exceptions import CacheError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_hash


logger = get_logger()

# ---------------------------------------------------------------------------
# TTL constants per intent (seconds)
# ---------------------------------------------------------------------------

_INTENT_TTL: Dict[str, int] = {
    QueryIntent.SEMANTIC.value:   86_400,   # 24 hours
    QueryIntent.ENTITY.value:     21_600,   # 6 hours
    QueryIntent.RELATIONAL.value:  7_200,   # 2 hours
    QueryIntent.TEMPORAL.value:      900,   # 15 minutes
    QueryIntent.HYBRID.value:      3_600,   # 1 hour
}

_DEFAULT_TTL = 3_600
_SEMANTIC_CACHE_TTL = 86_400       # semantic index entries live 24h
_COMPONENT_TTL = 1_800             # entity/rel extraction: 30 min
_JUDGE_SCORE_TTL = 3_600           # LLM judge scores: 1 hour
_SPECULATION_TTL = 1_800           # Speculation candidates: 30 min


# ---------------------------------------------------------------------------
# SemanticCacheIndex — lightweight in-memory embedding similarity index
# ---------------------------------------------------------------------------

class _SemanticCacheIndex:
    """
    Maintains a list of (query_text, embedding, cache_key) tuples.
    On lookup, computes cosine similarity against all stored embeddings
    and returns the cache_key of the best match above threshold.

    Uses sentence-transformers (already in settings.embedding_model).
    Falls back gracefully if sentence-transformers is not installed.
    """

    def __init__(self, model_name: str, threshold: float, max_entries: int):
        self._threshold = threshold
        self._max_entries = max_entries
        self._model = None
        self._model_name = model_name
        self._entries: List[Tuple[str, Any, str, float]] = []
        # (query_text, embedding, cache_key, expires_at_ts)
        self._lock = Lock()
        self._enabled = self._try_load_model(model_name)

    def _try_load_model(self, model_name: str) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info(
                "Semantic cache index loaded",
                model=model_name,
                threshold=self._threshold,
            )
            return True
        except Exception as e:
            logger.warning(
                "sentence-transformers unavailable — semantic cache disabled",
                error=str(e),
            )
            return False

    def add(self, query_text: str, cache_key: str, ttl: int):
        if not self._enabled or not self._model:
            return
        try:
            emb = self._model.encode(query_text, convert_to_numpy=True)
            expires_at = time.time() + ttl
            with self._lock:
                self._entries.append((query_text, emb, cache_key, expires_at))
                # Evict expired + cap at max_entries
                now = time.time()
                self._entries = [
                    e for e in self._entries if e[3] > now
                ][-self._max_entries:]
        except Exception as e:
            logger.warning("Semantic cache add failed", error=str(e))

    def lookup(self, query_text: str) -> Optional[str]:
        """Return cache_key of best match or None."""
        if not self._enabled or not self._model or not self._entries:
            return None
        try:
            import numpy as np
            query_emb = self._model.encode(query_text, convert_to_numpy=True)

            now = time.time()
            best_score = -1.0
            best_key = None

            with self._lock:
                valid = [e for e in self._entries if e[3] > now]

            for _, emb, key, _ in valid:
                # Cosine similarity
                denom = (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                if denom == 0:
                    continue
                sim = float(np.dot(query_emb, emb) / denom)
                if sim > best_score:
                    best_score = sim
                    best_key = key

            if best_score >= self._threshold:
                logger.debug(
                    "Semantic cache hit",
                    similarity=round(best_score, 4),
                    threshold=self._threshold,
                )
                return best_key
            return None
        except Exception as e:
            logger.warning("Semantic cache lookup failed", error=str(e))
            return None

    def clear(self):
        with self._lock:
            self._entries.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Hit/Miss counter
# ---------------------------------------------------------------------------

class _HitMissCounter:
    def __init__(self):
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        self._lock = Lock()

    def hit(self, semantic: bool = False):
        with self._lock:
            self._hits += 1
            if semantic:
                self._semantic_hits += 1

    def miss(self):
        with self._lock:
            self._misses += 1

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "semantic_hits": self._semantic_hits,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "total_requests": total,
            }

    def reset(self):
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._semantic_hits = 0


# ---------------------------------------------------------------------------
# CacheManager (v2)
# ---------------------------------------------------------------------------

class CacheManager:
    """
    Multi-layer cache manager with semantic similarity lookup and tiered TTLs.
    Drop-in replacement for the original CacheManager.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()

        # --- Backend initialisation (same as original) ---
        if self.settings.cache_backend == "redis":
            self._init_redis()
        elif self.settings.cache_backend == "disk":
            self._init_disk()
        else:
            self._init_memory()

        # --- Semantic cache ---
        self._semantic_enabled: bool = getattr(
            self.settings, "semantic_cache_enabled", True
        )
        semantic_threshold: float = getattr(
            self.settings, "semantic_cache_similarity_threshold", 0.92
        )
        semantic_max_entries: int = getattr(
            self.settings, "semantic_cache_max_entries", 10_000
        )
        self._semantic_index = _SemanticCacheIndex(
            model_name=self.settings.embedding_model,
            threshold=semantic_threshold,
            max_entries=semantic_max_entries,
        )

        # --- Hit/miss counter ---
        self._counter = _HitMissCounter()

        self.logger.info(
            "CacheManager v2 initialised",
            backend=self.backend,
            semantic_cache=self._semantic_enabled and self._semantic_index._enabled,
        )

    # ------------------------------------------------------------------ #
    #  Backend initialisation (unchanged from original)                  #
    # ------------------------------------------------------------------ #

    def _init_redis(self):
        try:
            import redis
            self.backend = "redis"
            self.redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                decode_responses=False,
            )
            self.redis_client.ping()
            self.logger.info("Redis cache initialised")
        except ImportError:
            raise CacheError("Redis not installed", details={"required": "pip install redis"})
        except Exception as e:
            raise CacheError(f"Redis connection failed: {e}")

    def _init_disk(self):
        try:
            from diskcache import Cache
            self.backend = "disk"
            cache_dir = Path(self.settings.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.disk_cache = Cache(str(cache_dir))
            self.logger.info("Disk cache initialised", path=str(cache_dir))
        except ImportError:
            raise CacheError("diskcache not installed", details={"required": "pip install diskcache"})

    def _init_memory(self):
        self.backend = "memory"
        self.memory_cache: Dict[str, Dict] = {}
        self.logger.info("In-memory cache initialised")

    # ------------------------------------------------------------------ #
    #  Core get / set / delete / clear                                   #
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self._backend_get(key)
            if value is not None:
                self._counter.hit()
            else:
                self._counter.miss()
            return value
        except Exception as e:
            self.logger.warning("Cache get failed", key=key, error=str(e))
            self._counter.miss()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if ttl is None:
            ttl = self.settings.cache_ttl_seconds
        try:
            return self._backend_set(key, value, ttl)
        except Exception as e:
            self.logger.warning("Cache set failed", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        try:
            return self._backend_delete(key)
        except Exception as e:
            self.logger.warning("Cache delete failed", key=key, error=str(e))
            return False

    def clear(self) -> bool:
        try:
            self._backend_clear()
            self._semantic_index.clear()
            self._counter.reset()
            self.logger.info("Cache cleared")
            return True
        except Exception as e:
            self.logger.error("Cache clear failed", error=str(e))
            return False

    # ------------------------------------------------------------------ #
    #  Backend-specific helpers                                          #
    # ------------------------------------------------------------------ #

    def _backend_get(self, key: str) -> Optional[Any]:
        if self.backend == "redis":
            raw = self.redis_client.get(key)
            return pickle.loads(raw) if raw else None
        elif self.backend == "disk":
            return self.disk_cache.get(key)
        else:
            entry = self.memory_cache.get(key)
            if entry and entry["expires_at"] > datetime.utcnow():
                return entry["value"]
            if entry:
                del self.memory_cache[key]
            return None

    def _backend_set(self, key: str, value: Any, ttl: int) -> bool:
        if self.backend == "redis":
            self.redis_client.setex(key, ttl, pickle.dumps(value))
        elif self.backend == "disk":
            self.disk_cache.set(key, value, expire=ttl)
        else:
            self.memory_cache[key] = {
                "value": value,
                "expires_at": datetime.utcnow() + timedelta(seconds=ttl),
            }
        return True

    def _backend_delete(self, key: str) -> bool:
        if self.backend == "redis":
            self.redis_client.delete(key)
        elif self.backend == "disk":
            self.disk_cache.delete(key)
        else:
            self.memory_cache.pop(key, None)
        return True

    def _backend_clear(self):
        if self.backend == "redis":
            self.redis_client.flushdb()
        elif self.backend == "disk":
            self.disk_cache.clear()
        else:
            self.memory_cache.clear()

    def _generate_key(self, prefix: str, data: Any) -> str:
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, str):
            data_str = data
        else:
            data_str = str(data)
        return f"{prefix}:{generate_hash(data_str)}"

    # ------------------------------------------------------------------ #
    #  TTL helpers                                                        #
    # ------------------------------------------------------------------ #

    def _ttl_for_query(self, query: Query) -> int:
        """Return appropriate TTL based on query intent if available."""
        intent = None
        # Intent is on NormalizedQuery, not Query — we check metadata
        intent_str = query.metadata.get("intent")
        if intent_str:
            return _INTENT_TTL.get(intent_str, _DEFAULT_TTL)
        return _DEFAULT_TTL

    def _ttl_for_normalized(self, nq: Optional[NormalizedQuery]) -> int:
        if nq is None:
            return _DEFAULT_TTL
        try:
            intent_val = nq.intent.value if hasattr(nq.intent, "value") else str(nq.intent)
            return _INTENT_TTL.get(intent_val, _DEFAULT_TTL)
        except Exception:
            return _DEFAULT_TTL

    # ------------------------------------------------------------------ #
    #  Pipeline response (with semantic cache)                           #
    # ------------------------------------------------------------------ #

    def get_pipeline_response(
        self,
        query: Query,
    ) -> Optional[PipelineResponse]:
        """
        Get cached pipeline response.
        Tries: 1) exact key match, 2) semantic similarity match.
        """
        # 1. Exact match
        key = self._generate_key("pipeline", query.text)
        value = self._backend_get(key)
        if value is not None:
            self._counter.hit()
            self.logger.debug("Pipeline cache: exact hit", query=query.text[:60])
            return value
        self._counter.miss()

        # 2. Semantic match
        if self._semantic_enabled:
            sem_key = self._semantic_index.lookup(query.text)
            if sem_key:
                value = self._backend_get(sem_key)
                if value is not None:
                    self._counter.hit(semantic=True)
                    self.logger.info(
                        "Pipeline cache: semantic hit",
                        query=query.text[:60],
                    )
                    return value

        return None

    def cache_pipeline_response(
        self,
        query: Query,
        response: PipelineResponse,
        custom_ttl: Optional[int] = None,
    ) -> bool:
        ttl = custom_ttl or self._ttl_for_query(query)
        key = self._generate_key("pipeline", query.text)
        ok = self._backend_set(key, response, ttl)
        if ok and self._semantic_enabled:
            self._semantic_index.add(query.text, key, ttl)
        return ok

    # ------------------------------------------------------------------ #
    #  Component-level caches (new)                                      #
    # ------------------------------------------------------------------ #

    # Entity extraction — keyed by document content hash
    def cache_entity_extraction(self, doc_id: str, content_hash: str, entities: list) -> bool:
        if not getattr(self.settings, "cache_entity_extraction", True):
            return False
        key = self._generate_key("entity_ext", f"{doc_id}:{content_hash}")
        return self._backend_set(key, entities, _COMPONENT_TTL)

    def get_entity_extraction(self, doc_id: str, content_hash: str) -> Optional[list]:
        if not getattr(self.settings, "cache_entity_extraction", True):
            return None
        key = self._generate_key("entity_ext", f"{doc_id}:{content_hash}")
        return self._backend_get(key)

    # Relationship extraction
    def cache_relationship_extraction(self, doc_id: str, content_hash: str, rels: list) -> bool:
        if not getattr(self.settings, "cache_relationship_extraction", True):
            return False
        key = self._generate_key("rel_ext", f"{doc_id}:{content_hash}")
        return self._backend_set(key, rels, _COMPONENT_TTL)

    def get_relationship_extraction(self, doc_id: str, content_hash: str) -> Optional[list]:
        if not getattr(self.settings, "cache_relationship_extraction", True):
            return None
        key = self._generate_key("rel_ext", f"{doc_id}:{content_hash}")
        return self._backend_get(key)

    # LLM Judge scores — keyed by answer text hash
    def cache_llm_judge_scores(self, answer_id: str, answer_hash: str, scores) -> bool:
        if not getattr(self.settings, "cache_llm_judge_scores", True):
            return False
        key = self._generate_key("judge", f"{answer_id}:{answer_hash}")
        return self._backend_set(key, scores, _JUDGE_SCORE_TTL)

    def get_llm_judge_scores(self, answer_id: str, answer_hash: str):
        if not getattr(self.settings, "cache_llm_judge_scores", True):
            return None
        key = self._generate_key("judge", f"{answer_id}:{answer_hash}")
        return self._backend_get(key)

    # Speculation candidates — keyed by strategy + query hash
    def cache_speculation_candidate(
        self, query_text: str, strategy: str, candidate
    ) -> bool:
        if not getattr(self.settings, "cache_speculation_candidates", True):
            return False
        key = self._generate_key("spec_cand", f"{strategy}:{query_text}")
        return self._backend_set(key, candidate, _SPECULATION_TTL)

    def get_speculation_candidate(self, query_text: str, strategy: str):
        if not getattr(self.settings, "cache_speculation_candidates", True):
            return None
        key = self._generate_key("spec_cand", f"{strategy}:{query_text}")
        return self._backend_get(key)

    # ------------------------------------------------------------------ #
    #  Original methods preserved (unchanged API)                        #
    # ------------------------------------------------------------------ #

    def cache_normalized_query(self, query, normalized_query) -> bool:
        key = self._generate_key("norm_query", query.text)
        return self.set(key, normalized_query)

    def get_normalized_query(self, query) -> Optional[Any]:
        key = self._generate_key("norm_query", query.text)
        return self.get(key)

    def cache_retrieval_results(self, query: str, documents: list) -> bool:
        key = self._generate_key("retrieval", query)
        ttl = min(self.settings.cache_ttl_seconds, 1800)
        return self.set(key, documents, ttl=ttl)

    def get_retrieval_results(self, query: str) -> Optional[list]:
        key = self._generate_key("retrieval", query)
        return self.get(key)

    def cache_answer(self, query, answer, custom_ttl: Optional[int] = None) -> bool:
        key = self._generate_key("answer", query.text)
        return self.set(key, answer, ttl=custom_ttl)

    def get_answer(self, query) -> Optional[Any]:
        key = self._generate_key("answer", query.text)
        return self.get(key)

    def cache_vector_index(self, query: str, vector_index) -> bool:
        key = self._generate_key("vector", query)
        return self.set(key, vector_index)

    def get_vector_index(self, query: str):
        key = self._generate_key("vector", query)
        return self.get(key)

    def cache_knowledge_graph(self, query: str, graph) -> bool:
        key = self._generate_key("graph", query)
        return self.set(key, graph)

    def get_knowledge_graph(self, query: str):
        key = self._generate_key("graph", query)
        return self.get(key)

    def cache_subgraph(self, query: str, subgraph) -> bool:
        key = self._generate_key("subgraph", query)
        return self.set(key, subgraph)

    def get_subgraph(self, query: str):
        key = self._generate_key("subgraph", query)
        return self.get(key)

    def invalidate_query_cache(self, query) -> bool:
        success = True
        for prefix in ("norm_query", "retrieval", "answer", "pipeline"):
            key = self._generate_key(prefix, query.text)
            success &= self.delete(key)
        return success

    # ------------------------------------------------------------------ #
    #  Cache warming                                                     #
    # ------------------------------------------------------------------ #

    def warm_cache(self, common_queries: list) -> int:
        """
        Pre-warm cache from a list of (query_text, pipeline_response) tuples
        or plain query strings.

        If given plain strings, we can only register them in the semantic index
        (no stored responses yet).

        Returns number of entries warmed.
        """
        warmed = 0
        for item in common_queries:
            try:
                if isinstance(item, tuple) and len(item) == 2:
                    query_text, response = item
                    q = Query(text=query_text)
                    self.cache_pipeline_response(q, response)
                    warmed += 1
                elif isinstance(item, str) and self._semantic_enabled:
                    # Register in semantic index only (no response yet)
                    key = self._generate_key("pipeline", item)
                    self._semantic_index.add(item, key, _DEFAULT_TTL)
                    warmed += 1
            except Exception as e:
                self.logger.warning("Cache warm failed for item", error=str(e))

        self.logger.info("Cache warming complete", warmed=warmed)
        return warmed

    # ------------------------------------------------------------------ #
    #  Statistics                                                        #
    # ------------------------------------------------------------------ #

    def get_cache_stats(self) -> Dict:
        stats: Dict[str, Any] = {
            "backend": self.backend,
            **self._counter.stats,
            "semantic_cache": {
                "enabled": self._semantic_enabled and self._semantic_index._enabled,
                "index_size": self._semantic_index.size,
                "threshold": getattr(
                    self.settings, "semantic_cache_similarity_threshold", 0.92
                ),
            },
        }

        try:
            if self.backend == "redis":
                info = self.redis_client.info("stats")
                stats["redis_hits"] = info.get("keyspace_hits", 0)
                stats["redis_misses"] = info.get("keyspace_misses", 0)
                stats["keys"] = self.redis_client.dbsize()
            elif self.backend == "disk":
                stats["disk_size_bytes"] = self.disk_cache.volume()
                stats["keys"] = len(self.disk_cache)
            else:
                now = datetime.utcnow()
                live = {
                    k: v for k, v in self.memory_cache.items()
                    if v["expires_at"] > now
                }
                stats["keys"] = len(live)
                stats["expired_keys"] = len(self.memory_cache) - len(live)
        except Exception as e:
            self.logger.warning("Failed to collect backend stats", error=str(e))

        return stats

    def cleanup_expired(self) -> int:
        if self.backend != "memory":
            return 0
        now = datetime.utcnow()
        expired = [k for k, v in self.memory_cache.items() if v["expires_at"] <= now]
        for k in expired:
            del self.memory_cache[k]
        if expired:
            self.logger.info("Expired cache entries cleaned", count=len(expired))
        return len(expired)

    def __del__(self):
        if self.backend == "redis" and hasattr(self, "redis_client"):
            self.redis_client.close()