"""
Router for the research analyst system.
Determines execution path (fast vs research) based on query characteristics.
"""

from typing import Tuple

from research_analyst.core.models import (
    NormalizedQuery,
    RoutingDecision,
    ExecutionPath,
    QueryComplexity,
)
from research_analyst.routing.complexity_estimator import ComplexityEstimator
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger


logger = get_logger()


# ============================================================================
# Helper functions
# ============================================================================

def _complexity_value(complexity) -> str:
    """
    Normalize QueryComplexity to its string value.
    Works whether complexity is an enum or already a string.
    """
    if hasattr(complexity, "value"):
        return complexity.value
    return str(complexity)


# ============================================================================
# Router
# ============================================================================

class Router:
    """Route queries to appropriate execution path."""

    def __init__(self):
        """Initialize router."""
        self.settings = get_settings()
        self.logger = get_logger()
        self.complexity_estimator = ComplexityEstimator()

        # Thresholds for routing decisions
        self.fast_path_max_complexity = QueryComplexity.SIMPLE.value
        self.force_research_on_graph = True  # Force research path if graph needed

    def route(self, normalized_query: NormalizedQuery) -> RoutingDecision:
        """
        Determine execution path for query.

        Args:
            normalized_query: Normalized query with intent classification

        Returns:
            RoutingDecision with path and reasoning
        """
        complexity = _complexity_value(normalized_query.complexity)

        self.logger.info(
            "Routing query",
            query=normalized_query.normalized_text[:100],
            complexity=complexity,
            requires_graph=normalized_query.requires_graph,
        )

        # Estimate complexity if not already set
        if normalized_query.complexity is None:
            normalized_query.complexity = self.complexity_estimator.estimate(
                normalized_query
            )
            complexity = _complexity_value(normalized_query.complexity)

        # Determine if graph should be used
        should_use_graph = self.complexity_estimator.should_use_graph(
            normalized_query
        )

        # Make routing decision
        decision = self._make_decision(
            normalized_query,
            should_use_graph,
        )

        execution_path = (
            decision.execution_path.value
            if hasattr(decision.execution_path, "value")
            else decision.execution_path
        )

        self.logger.log_routing_decision(
            query_id=normalized_query.original_text[:50],
            execution_path=execution_path,
            reasoning=decision.reasoning,
            confidence=decision.confidence
        )


        return decision

    def _make_decision(
        self,
        query: NormalizedQuery,
        should_use_graph: bool,
    ) -> RoutingDecision:
        """
        Make routing decision based on query characteristics.
        """
        factors = []
        confidence = 0.8

        complexity = _complexity_value(query.complexity)

        # Factor 1: Graph requirement
        if query.requires_graph or should_use_graph:
            factors.append("Requires graph-based reasoning for entity relationships")
            path = ExecutionPath.RESEARCH
            confidence = 0.9

        # Factor 2: Complexity
        elif complexity == QueryComplexity.COMPLEX.value:
            factors.append("High complexity query requires comprehensive research")
            path = ExecutionPath.RESEARCH
            confidence = 0.85

        elif complexity == QueryComplexity.MEDIUM.value:
            if len(query.entities_mentioned) >= 2:
                factors.append("Multiple entities suggest need for comprehensive retrieval")
                path = ExecutionPath.RESEARCH
                confidence = 0.75
            else:
                factors.append("Moderate complexity suitable for fast vector search")
                path = ExecutionPath.FAST
                confidence = 0.7

        else:  # SIMPLE
            factors.append("Simple query suitable for fast vector search")
            path = ExecutionPath.FAST
            confidence = 0.9

        # Factor 3: Domain specificity
        if (
            query.domain in ["science", "health", "technology"]
            and path == ExecutionPath.FAST 
            and query.requires_graph  

        ):
            factors.append(
                f"Technical domain ({query.domain}) may benefit from graph reasoning"
            )
            if complexity != QueryComplexity.SIMPLE.value:
                path = ExecutionPath.RESEARCH
                confidence = 0.75

        # Estimate cost and latency
        estimated_cost = self.complexity_estimator.estimate_cost(
            complexity,
            path == ExecutionPath.RESEARCH,
        )

        estimated_latency = self.complexity_estimator.estimate_latency(
            complexity,
            path == ExecutionPath.RESEARCH,
        )

        reasoning = " | ".join(factors)

        return RoutingDecision(
            execution_path=path,
            reasoning=reasoning,
            confidence=confidence,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
        )

    def should_cache(self, query: NormalizedQuery, decision: RoutingDecision) -> bool:
        """
        Determine if results should be cached for this query.
        """
        complexity = _complexity_value(query.complexity)

        # Don't cache if query has time-sensitive requirements
        if query.time_range:
            time_keywords = ["today", "now", "current", "latest", "recent"]
            if any(kw in query.normalized_text.lower() for kw in time_keywords):
                return False

        # Cache simple and medium complexity queries
        if complexity in [
            QueryComplexity.SIMPLE.value,
            QueryComplexity.MEDIUM.value,
        ]:
            return True

        # Cache complex queries only on fast path
        if (
            complexity == QueryComplexity.COMPLEX.value
            and decision.execution_path == ExecutionPath.FAST
        ):
            return True

        return False

    def get_cache_ttl(self, query: NormalizedQuery) -> int:
        """
        Get appropriate cache TTL for query.
        """
        # Shorter TTL for time-sensitive queries
        if query.time_range:
            return 300  # 5 minutes

        # Domain-based TTL
        dynamic_domains = ["news", "politics", "sports", "entertainment"]
        if query.domain in dynamic_domains:
            return 1800  # 30 minutes

        return self.settings.cache_ttl_seconds

    def route_with_alternatives(
        self,
        normalized_query: NormalizedQuery,
    ) -> Tuple[RoutingDecision, RoutingDecision]:
        """
        Get primary routing decision plus alternative.
        """
        primary = self.route(normalized_query)

        if primary.execution_path == ExecutionPath.FAST:
            alternative_path = ExecutionPath.RESEARCH
            alternative_reasoning = "Alternative: Use research path for higher quality"
            alternative_confidence = 0.6
        else:
            alternative_path = ExecutionPath.FAST
            alternative_reasoning = "Alternative: Use fast path for lower latency"
            alternative_confidence = 0.5

        complexity = _complexity_value(normalized_query.complexity)

        alternative_cost = self.complexity_estimator.estimate_cost(
            complexity,
            alternative_path == ExecutionPath.RESEARCH,
        )

        alternative_latency = self.complexity_estimator.estimate_latency(
            complexity,
            alternative_path == ExecutionPath.RESEARCH,
        )

        alternative = RoutingDecision(
            execution_path=alternative_path,
            reasoning=alternative_reasoning,
            confidence=alternative_confidence,
            estimated_cost=alternative_cost,
            estimated_latency=alternative_latency,
        )

        return primary, alternative

    def explain_decision(self, decision: RoutingDecision) -> str:
        """
        Generate human-readable explanation of routing decision.
        """
        path_name = (
            "Fast Path (Vector Search)"
            if decision.execution_path == ExecutionPath.FAST
            else "Research Path (Graph + Vector)"
        )

        return (
            f"Routing Decision: {path_name}\n"
            f"Confidence: {decision.confidence:.2%}\n"
            f"Estimated Cost: ${decision.estimated_cost:.4f}\n"
            f"Estimated Latency: {decision.estimated_latency:.1f}s\n\n"
            f"Reasoning: {decision.reasoning}"
        )
