"""
Graph Scorer for the research analyst system.

Replaces the word-overlap heuristic in GraphQuerier._score_entity_relevance()
with proper graph-theoretic scoring using NetworkX algorithms:

    - Weighted PageRank       (entity structural importance)
    - Betweenness Centrality  (connector entities for multi-hop reasoning)
    - Freshness-weighted edges (integrates TemporalDecayScorer output)
    - Composite entity score  (blends all signals)

The scorer re-ranks subgraph entities so the synthesiser focuses on
structurally important entities, not just those whose text overlaps
with the query words.

Integration:
    Replace graph_querier._rank_subgraph_elements() body:

        from research_analyst.graph_scorer import GraphScorer
        scorer = GraphScorer()
        subgraph = scorer.score_and_rank(subgraph, normalized_query)
"""

import math
from collections import defaultdict
from typing import List, Dict, Optional, Set, Tuple

import networkx as nx

from research_analyst.core.models import (
    Subgraph,
    Entity,
    Relationship,
    NormalizedQuery,
)
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger


logger = get_logger()

# ---------------------------------------------------------------------------
# Weight constants
# ---------------------------------------------------------------------------

_W_PAGERANK = 0.40       # Structural importance via PageRank
_W_BETWEENNESS = 0.25    # Bridge/connector role
_W_QUERY_MATCH = 0.20    # Query text overlap (kept as a signal, not the only one)
_W_CONFIDENCE = 0.10     # Entity extraction confidence
_W_MENTION = 0.05        # Mention count across documents

_MAX_ENTITIES_RETURNED = 50
_PAGERANK_ALPHA = 0.85   # Standard damping factor


# ---------------------------------------------------------------------------
# GraphScorer
# ---------------------------------------------------------------------------

class GraphScorer:
    """
    Scores and re-ranks subgraph entities using graph-theoretic signals.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def score_and_rank(
        self,
        subgraph: Subgraph,
        normalized_query: NormalizedQuery,
    ) -> Subgraph:
        """
        Score all entities in the subgraph and re-rank them.

        Args:
            subgraph:         Subgraph from GraphQuerier.
            normalized_query: NormalizedQuery for query-match signal.

        Returns:
            Subgraph with entities sorted by composite score descending,
            and subgraph.relevance_score updated to best entity score.
        """
        if not subgraph.entities:
            return subgraph

        # Build a local NetworkX graph from subgraph data
        G = self._build_nx_graph(subgraph)

        if G.number_of_nodes() == 0:
            return subgraph

        # Compute graph-theoretic scores
        pagerank_scores = self._compute_pagerank(G, subgraph)
        betweenness_scores = self._compute_betweenness(G, subgraph)

        # Query match scores (text overlap — kept as partial signal)
        query_words = set(normalized_query.normalized_text.lower().split())
        query_entities = set(e.lower() for e in normalized_query.entities_mentioned)

        # Score each entity
        scored: List[Tuple[float, Entity]] = []

        for entity in subgraph.entities:
            score = self._composite_score(
                entity=entity,
                pagerank=pagerank_scores.get(entity.entity_id, 0.0),
                betweenness=betweenness_scores.get(entity.entity_id, 0.0),
                query_words=query_words,
                query_entities=query_entities,
            )
            # Attach score to entity metadata for downstream use
            entity.attributes["graph_score"] = round(score, 4)
            entity.attributes["pagerank"] = round(
                pagerank_scores.get(entity.entity_id, 0.0), 4
            )
            entity.attributes["betweenness"] = round(
                betweenness_scores.get(entity.entity_id, 0.0), 4
            )
            scored.append((score, entity))

        # Sort descending and trim
        scored.sort(key=lambda x: x[0], reverse=True)
        subgraph.entities = [e for _, e in scored[:_MAX_ENTITIES_RETURNED]]

        # Update subgraph relevance score to top entity's composite score
        if scored:
            subgraph.relevance_score = min(1.0, scored[0][0])

        self.logger.info(
            "GraphScorer: subgraph re-ranked",
            num_entities=len(subgraph.entities),
            top_entity=(
                subgraph.entities[0].text if subgraph.entities else "none"
            ),
            top_score=round(scored[0][0], 4) if scored else 0.0,
        )

        return subgraph

    def score_relationships_by_centrality(
        self,
        subgraph: Subgraph,
    ) -> Subgraph:
        """
        Re-rank relationships by the combined centrality of their endpoints.
        Higher-centrality relationships are surfaced first in synthesis context.

        Args:
            subgraph: Subgraph (score_and_rank should be called first).

        Returns:
            Subgraph with relationships sorted by endpoint centrality descending.
        """
        if not subgraph.relationships or not subgraph.entities:
            return subgraph

        # Build entity_id -> graph_score lookup
        score_map: Dict[str, float] = {
            e.entity_id: e.attributes.get("graph_score", 0.0)
            for e in subgraph.entities
        }

        def rel_score(r: Relationship) -> float:
            s_id = r.metadata.get("subject_entity_id", "")
            o_id = r.metadata.get("object_entity_id", "")
            return (
                score_map.get(s_id, 0.0) + score_map.get(o_id, 0.0)
            ) * r.confidence * r.metadata.get("freshness_score", 0.7)

        subgraph.relationships.sort(key=rel_score, reverse=True)
        return subgraph

    # ------------------------------------------------------------------ #
    #  Graph construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_nx_graph(self, subgraph: Subgraph) -> nx.DiGraph:
        """
        Build a weighted directed NetworkX graph from subgraph data.
        Edge weight = relationship.confidence * freshness_score
        """
        G = nx.DiGraph()

        # Add nodes
        for entity in subgraph.entities:
            G.add_node(
                entity.entity_id,
                text=entity.text,
                confidence=entity.confidence,
                mention_count=entity.attributes.get("mention_count", 1),
            )

        # Add edges with composite weight
        for rel in subgraph.relationships:
            s_id = rel.metadata.get("subject_entity_id")
            o_id = rel.metadata.get("object_entity_id")

            if not s_id or not o_id:
                continue
            if not G.has_node(s_id) or not G.has_node(o_id):
                continue

            freshness = rel.metadata.get("freshness_score", 0.7)
            weight = rel.confidence * freshness

            # MultiDiGraph can have multiple edges; we take the highest weight
            if G.has_edge(s_id, o_id):
                existing_weight = G[s_id][o_id].get("weight", 0.0)
                if weight > existing_weight:
                    G[s_id][o_id]["weight"] = weight
            else:
                G.add_edge(s_id, o_id, weight=weight, predicate=rel.predicate)

        return G

    # ------------------------------------------------------------------ #
    #  Graph-theoretic metrics                                           #
    # ------------------------------------------------------------------ #

    def _compute_pagerank(
        self,
        G: nx.DiGraph,
        subgraph: Subgraph,
    ) -> Dict[str, float]:
        """
        Compute weighted PageRank. Falls back to uniform if graph is too sparse.
        Returns entity_id -> normalised PageRank score [0, 1].
        """
        if G.number_of_edges() == 0:
            # No edges — uniform PageRank
            n = G.number_of_nodes()
            uniform = 1.0 / n if n > 0 else 0.0
            return {node: uniform for node in G.nodes()}

        try:
            pr = nx.pagerank(
                G,
                alpha=_PAGERANK_ALPHA,
                weight="weight",
                max_iter=100,
                tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            self.logger.warning("PageRank failed to converge — using uniform")
            n = G.number_of_nodes()
            uniform = 1.0 / n if n > 0 else 0.0
            return {node: uniform for node in G.nodes()}

        # Normalise to [0, 1]
        max_pr = max(pr.values()) if pr else 1.0
        if max_pr == 0:
            return {k: 0.0 for k in pr}
        return {k: v / max_pr for k, v in pr.items()}

    def _compute_betweenness(
        self,
        G: nx.DiGraph,
        subgraph: Subgraph,
    ) -> Dict[str, float]:
        """
        Compute normalised betweenness centrality.
        For large graphs (>200 nodes) uses approximate k-sample version.
        Returns entity_id -> normalised betweenness [0, 1].
        """
        n = G.number_of_nodes()
        if n < 2:
            return {node: 0.0 for node in G.nodes()}

        try:
            if n > 200:
                # Approximate: sample 100 source nodes
                bc = nx.betweenness_centrality(
                    G, k=min(100, n), weight="weight", normalized=True
                )
            else:
                bc = nx.betweenness_centrality(
                    G, weight="weight", normalized=True
                )
        except Exception as e:
            self.logger.warning(
                "Betweenness centrality failed", error=str(e)
            )
            return {node: 0.0 for node in G.nodes()}

        max_bc = max(bc.values()) if bc else 1.0
        if max_bc == 0:
            return {k: 0.0 for k in bc}
        return {k: v / max_bc for k, v in bc.items()}

    # ------------------------------------------------------------------ #
    #  Composite entity scoring                                          #
    # ------------------------------------------------------------------ #

    def _composite_score(
        self,
        entity: Entity,
        pagerank: float,
        betweenness: float,
        query_words: Set[str],
        query_entities: Set[str],
    ) -> float:
        """
        Blend all signals into a composite entity relevance score.
        """
        # Query match: exact entity name match gets full credit
        entity_lower = entity.text.lower()
        if entity_lower in query_entities:
            query_match = 1.0
        else:
            entity_words = set(entity_lower.split())
            overlap = len(entity_words & query_words)
            query_match = min(1.0, overlap * 0.4)

        # Alias bonus
        for alias in entity.aliases:
            if alias.lower() in query_entities:
                query_match = min(1.0, query_match + 0.3)
                break

        # Mention count signal (log-scaled to avoid domination)
        mention_count = entity.attributes.get("mention_count", 1)
        mention_score = min(1.0, math.log1p(mention_count) / math.log1p(10))

        # Confidence signal
        confidence_score = entity.confidence

        composite = (
            _W_PAGERANK * pagerank
            + _W_BETWEENNESS * betweenness
            + _W_QUERY_MATCH * query_match
            + _W_CONFIDENCE * confidence_score
            + _W_MENTION * mention_score
        )

        return min(1.0, composite)

    # ------------------------------------------------------------------ #
    #  Reporting                                                          #
    # ------------------------------------------------------------------ #

    def generate_scoring_report(self, subgraph: Subgraph) -> str:
        """Human-readable scoring report for debugging."""
        if not subgraph.entities:
            return "No entities in subgraph."

        lines = [
            "=" * 60,
            "GRAPH SCORER REPORT",
            f"Entities: {len(subgraph.entities)} | "
            f"Relationships: {len(subgraph.relationships)}",
            "=" * 60,
            "Top 10 entities by composite score:",
        ]

        for i, entity in enumerate(subgraph.entities[:10], 1):
            gs = entity.attributes.get("graph_score", 0.0)
            pr = entity.attributes.get("pagerank", 0.0)
            bt = entity.attributes.get("betweenness", 0.0)
            lines.append(
                f"  {i:2}. {entity.text:<30} "
                f"score={gs:.3f}  PR={pr:.3f}  BC={bt:.3f}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)