"""
Complexity Estimator for the HERALD routing system.

Scores query complexity on six weighted signals to decide
SIMPLE / MEDIUM / COMPLEX, then uses that to route to
FAST (vector-only) or RESEARCH (graph + vector) path.

Fix from v0.1:
    should_use_graph() had an unreachable code path.  The new version
    uses a clear three-factor OR rule:
        (a) intent_classifier said requires_graph=True
        (b) query is COMPLEX with ≥2 entities
        (c) ≥3 entities mentioned regardless of complexity
"""

from typing import Dict, Tuple

from research_analyst.core.models import NormalizedQuery, QueryComplexity, QueryIntent
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger


logger = get_logger()


class ComplexityEstimator:
    """Estimate query complexity for routing decisions."""

    def __init__(self):
        self.settings = get_settings()
        self.logger   = get_logger()

        # Six signals and their contribution weights (must sum to 1.0)
        self.weights = {
            "word_count":       0.15,
            "entity_count":     0.20,
            "intent_complexity":0.25,
            "requires_graph":   0.20,
            "has_time_range":   0.10,
            "domain_specificity":0.10,
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def estimate(self, normalized_query: NormalizedQuery) -> QueryComplexity:
        """
        Estimate query complexity.

        Args:
            normalized_query: Normalised query with intent classification.

        Returns:
            QueryComplexity (SIMPLE | MEDIUM | COMPLEX)
        """
        scores = self._compute_scores(normalized_query)
        total  = sum(scores[f] * self.weights[f] for f in scores)
        return self._score_to_complexity(total)

    def estimate_with_details(
        self,
        normalized_query: NormalizedQuery,
    ) -> Tuple[QueryComplexity, Dict[str, float]]:
        """
        Estimate complexity and return per-signal breakdown.

        Returns:
            (QueryComplexity, score_breakdown_dict)
        """
        scores = self._compute_scores(normalized_query)
        total  = sum(scores[f] * self.weights[f] for f in scores)
        scores["total"] = round(total, 4)
        return self._score_to_complexity(total), scores

    def should_use_graph(self, normalized_query: NormalizedQuery) -> bool:
        """
        Decide whether the RESEARCH (graph) path is needed.

        Uses three independent criteria (OR logic):
            (a) Intent classifier explicitly set requires_graph=True.
            (b) Query is COMPLEX and mentions ≥ 2 entities.
            (c) Query mentions ≥ 3 entities regardless of complexity.

        This replaces the v0.1 implementation which had an unreachable
        branch (the first 'if not requires_graph: return False' made
        the second 'if requires_graph: return True' unreachable).
        """
        if normalized_query.requires_graph:
            return True  # (a) explicit flag from intent classifier

        n_entities = len(normalized_query.entities_mentioned)
        complexity = (
            normalized_query.complexity
            if isinstance(normalized_query.complexity, str)
            else normalized_query.complexity.value
        )

        if complexity == QueryComplexity.COMPLEX.value and n_entities >= 2:
            return True  # (b) complex multi-entity query

        if n_entities >= 3:
            return True  # (c) entity-heavy query

        return False

    def estimate_cost(
        self,
        complexity: QueryComplexity,
        use_graph:  bool,
    ) -> float:
        """Estimate query processing cost in USD."""
        base = {
            QueryComplexity.SIMPLE:  0.02,
            QueryComplexity.MEDIUM:  0.05,
            QueryComplexity.COMPLEX: 0.10,
        }.get(complexity, 0.05)
        return base * 3.0 if use_graph else base

    def estimate_latency(
        self,
        complexity: QueryComplexity,
        use_graph:  bool,
    ) -> float:
        """Estimate query processing latency in seconds."""
        base = {
            QueryComplexity.SIMPLE:  2.0,
            QueryComplexity.MEDIUM:  5.0,
            QueryComplexity.COMPLEX: 10.0,
        }.get(complexity, 5.0)
        return base * 2.0 if use_graph else base

    # ------------------------------------------------------------------ #
    #  Signal scorers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_scores(self, q: NormalizedQuery) -> Dict[str, float]:
        return {
            "word_count":        self._score_word_count(q),
            "entity_count":      self._score_entity_count(q),
            "intent_complexity": self._score_intent(q),
            "requires_graph":    1.0 if q.requires_graph else 0.0,
            "has_time_range":    0.7 if q.time_range else 0.0,
            "domain_specificity":self._score_domain(q),
        }

    @staticmethod
    def _score_word_count(q: NormalizedQuery) -> float:
        n = len(q.normalized_text.split())
        if n <= 5:  return 0.2
        if n <= 10: return 0.4
        if n <= 20: return 0.7
        return 1.0

    @staticmethod
    def _score_entity_count(q: NormalizedQuery) -> float:
        n = len(q.entities_mentioned)
        if n == 0: return 0.1
        if n == 1: return 0.3
        if n == 2: return 0.6
        if n == 3: return 0.8
        return 1.0

    @staticmethod
    def _score_intent(q: NormalizedQuery) -> float:
        mapping = {
            QueryIntent.SEMANTIC:   0.2,
            QueryIntent.ENTITY:     0.4,
            QueryIntent.RELATIONAL: 0.8,
            QueryIntent.TEMPORAL:   0.7,
            QueryIntent.HYBRID:     1.0,
        }
        intent = q.intent
        if isinstance(intent, str):
            intent = QueryIntent(intent)
        return mapping.get(intent, 0.5)

    @staticmethod
    def _score_domain(q: NormalizedQuery) -> float:
        if not q.domain:
            return 0.2
        complex_domains = {"technology", "science", "health", "finance"}
        return 0.8 if q.domain in complex_domains else 0.5

    @staticmethod
    def _score_to_complexity(score: float) -> QueryComplexity:
        if score <= 0.35: return QueryComplexity.SIMPLE
        if score <= 0.65: return QueryComplexity.MEDIUM
        return QueryComplexity.COMPLEX