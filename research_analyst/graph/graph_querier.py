"""
Graph querier for the research analyst system.
Queries and reasons over knowledge graphs to extract relevant information.
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from research_analyst.core.models import (
    NormalizedQuery,
    Entity,
    Relationship,
    Subgraph,
    EntityType
)
from research_analyst.core.exceptions import GraphQueryError
from research_analyst.graph.graph_store import GraphStore
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id


logger = get_logger()


class GraphQuerier:
    """Query and reason over knowledge graphs."""
    
    def __init__(self, graph_store: GraphStore):
        """
        Initialize graph querier.
        
        Args:
            graph_store: GraphStore instance
        """
        self.settings = get_settings()
        self.logger = get_logger()
        self.graph_store = graph_store
    
    def query_for_normalized_query(
        self,
        normalized_query: NormalizedQuery
    ) -> Subgraph:
        """
        Query graph based on normalized query.
        
        Args:
            normalized_query: Normalized query with entities
            
        Returns:
            Relevant subgraph
        """
        self.logger.info(
            "Querying graph",
            query=normalized_query.normalized_text[:100],
            num_mentioned_entities=len(normalized_query.entities_mentioned)
        )
        
        try:
            # Find entities matching query
            relevant_entity_ids = self._find_relevant_entities(normalized_query)
            
            if not relevant_entity_ids:
                self.logger.warning("No relevant entities found in graph")
                return self._empty_subgraph()
            
            # Extract subgraph
            subgraph = self.graph_store.get_subgraph(
                central_entities=list(relevant_entity_ids),
                hops=self.settings.max_graph_hops
            )
            
            # Rank entities and relationships by relevance
            subgraph = self._rank_subgraph_elements(subgraph, normalized_query)
            
            self.logger.info(
                "Graph query complete",
                num_entities=len(subgraph.entities),
                num_relationships=len(subgraph.relationships),
                relevance=subgraph.relevance_score
            )
            
            return subgraph
            
        except Exception as e:
            self.logger.error(
                "Graph query failed",
                error=str(e)
            )
            raise GraphQueryError(
                f"Failed to query graph: {str(e)}",
                details={"query": normalized_query.normalized_text}
            )
    
    def _find_relevant_entities(
        self,
        normalized_query: NormalizedQuery
    ) -> Set[str]:
        """
        Find entity IDs relevant to query.
        
        Args:
            normalized_query: Normalized query
            
        Returns:
            Set of entity IDs
        """
        relevant_ids = set()
        
        # Search by mentioned entities
        for entity_text in normalized_query.entities_mentioned:
            entity_ids = self._find_entities_by_text(entity_text)
            relevant_ids.update(entity_ids)
        
        # If no entities found, try keyword search
        if not relevant_ids:
            keywords = normalized_query.normalized_text.split()[:5]  # Top 5 words
            for keyword in keywords:
                entity_ids = self._find_entities_by_text(keyword)
                relevant_ids.update(entity_ids)
        
        return relevant_ids
    
    def _find_entities_by_text(self, text: str) -> Set[str]:
        """
        Find entities matching text.
        
        Args:
            text: Text to search for
            
        Returns:
            Set of entity IDs
        """
        matching_ids = set()
        text_lower = text.lower().strip()
        
        # Search in entity map
        for entity_id, entity in self.graph_store.entity_map.items():
            # Check entity text
            if text_lower in entity.text.lower():
                matching_ids.add(entity_id)
                continue
            
            # Check aliases
            if any(text_lower in alias.lower() for alias in entity.aliases):
                matching_ids.add(entity_id)
        
        return matching_ids
    
    def _rank_subgraph_elements(
        self,
        subgraph: Subgraph,
        normalized_query: NormalizedQuery
    ) -> Subgraph:
        """
        Rank entities and relationships in subgraph by relevance.
        
        Args:
            subgraph: Subgraph to rank
            normalized_query: Query for context
            
        Returns:
            Subgraph with ranked elements
        """
        # Score entities
        query_words = set(normalized_query.normalized_text.lower().split())
        
        scored_entities = []
        for entity in subgraph.entities:
            score = self._score_entity_relevance(entity, query_words)
            scored_entities.append((score, entity))
        
        # Sort and keep top entities
        scored_entities.sort(reverse=True, key=lambda x: x[0])
        subgraph.entities = [e for _, e in scored_entities[:50]]  # Top 50
        
        # Update relevance score
        if subgraph.entities:
            subgraph.relevance_score = scored_entities[0][0]
        
        return subgraph
    
    def _score_entity_relevance(
        self,
        entity: Entity,
        query_words: Set[str]
    ) -> float:
        """
        Score entity relevance to query.
        
        Args:
            entity: Entity to score
            query_words: Set of query words
            
        Returns:
            Relevance score
        """
        score = 0.0
        
        # Check text match
        entity_words = set(entity.text.lower().split())
        matches = len(entity_words & query_words)
        score += matches * 0.3
        
        # Check alias matches
        for alias in entity.aliases:
            alias_words = set(alias.lower().split())
            matches = len(alias_words & query_words)
            score += matches * 0.2
        
        # Boost by confidence
        score *= entity.confidence
        
        # Boost by mention count
        mention_count = entity.attributes.get('mention_count', 1)
        score *= (1 + 0.1 * min(mention_count, 5))
        
        return score
    
    def find_relationship_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_length: int = 3
    ) -> List[List[Relationship]]:
        """
        Find relationship paths between two entities.
        
        Args:
            source_entity: Source entity text
            target_entity: Target entity text
            max_length: Maximum path length
            
        Returns:
            List of relationship paths
        """
        # Find entity IDs
        source_ids = self._find_entities_by_text(source_entity)
        target_ids = self._find_entities_by_text(target_entity)
        
        if not source_ids or not target_ids:
            return []
        
        all_paths = []
        
        # Try all combinations
        for source_id in source_ids:
            for target_id in target_ids:
                # Find paths in graph
                entity_paths = self.graph_store.find_paths(
                    source_id,
                    target_id,
                    max_length=max_length
                )
                
                # Convert entity paths to relationship paths
                for entity_path in entity_paths:
                    rel_path = self._entity_path_to_relationships(entity_path)
                    if rel_path:
                        all_paths.append(rel_path)
        
        return all_paths
    
    def _entity_path_to_relationships(
        self,
        entity_path: List[str]
    ) -> List[Relationship]:
        """
        Convert path of entity IDs to list of relationships.
        
        Args:
            entity_path: List of entity IDs
            
        Returns:
            List of relationships
        """
        relationships = []
        
        for i in range(len(entity_path) - 1):
            source_id = entity_path[i]
            target_id = entity_path[i + 1]
            
            # Find relationships between these entities
            for rel_id, rel in self.graph_store.relationship_map.items():
                subject_id = rel.metadata.get('subject_entity_id')
                object_id = rel.metadata.get('object_entity_id')
                
                if subject_id == source_id and object_id == target_id:
                    relationships.append(rel)
                    break
        
        return relationships
    
    def get_entity_context(
        self,
        entity_text: str,
        include_neighbors: bool = True
    ) -> Dict:
        """
        Get comprehensive context for an entity.
        
        Args:
            entity_text: Entity text to look up
            include_neighbors: Whether to include neighbor information
            
        Returns:
            Context dictionary
        """
        # Find entity
        entity_ids = self._find_entities_by_text(entity_text)
        
        if not entity_ids:
            return {}
        
        # Use first matching entity
        entity_id = list(entity_ids)[0]
        entity = self.graph_store.entity_map.get(entity_id)
        
        if not entity:
            return {}
        
        context = {
            'entity': entity,
            'relationships': self.graph_store.get_relationships_for_entity(entity_id)
        }
        
        if include_neighbors:
            neighbor_ids = self.graph_store.get_neighbors(entity_id, hops=1)
            neighbors = [
                self.graph_store.entity_map[nid]
                for nid in neighbor_ids
                if nid in self.graph_store.entity_map
            ]
            context['neighbors'] = neighbors
        
        return context
    
    def analyze_entity_centrality(self) -> List[Tuple[Entity, float]]:
        """
        Analyze entity centrality in graph.
        
        Returns:
            List of (entity, centrality_score) tuples sorted by centrality
        """
        if self.graph_store.backend != "networkx":
            self.logger.warning("Centrality analysis only available for NetworkX backend")
            return []
        
        import networkx as nx
        
        # Calculate PageRank centrality
        try:
            centrality = nx.pagerank(self.graph_store.graph)
            
            # Convert to entity list
            entity_centrality = [
                (self.graph_store.entity_map[entity_id], score)
                for entity_id, score in centrality.items()
                if entity_id in self.graph_store.entity_map
            ]
            
            # Sort by centrality
            entity_centrality.sort(key=lambda x: x[1], reverse=True)
            
            return entity_centrality
            
        except Exception as e:
            self.logger.error(
                "Centrality calculation failed",
                error=str(e)
            )
            return []
    
    def find_communities(self) -> List[List[Entity]]:
        """
        Find communities (clusters) in graph.
        
        Returns:
            List of communities (each is a list of entities)
        """
        if self.graph_store.backend != "networkx":
            self.logger.warning("Community detection only available for NetworkX backend")
            return []
        
        import networkx as nx
        from networkx.algorithms import community
        
        try:
            # Convert to undirected for community detection
            undirected = self.graph_store.graph.to_undirected()
            
            # Detect communities using Louvain method
            communities = community.louvain_communities(undirected)
            
            # Convert to entity lists
            entity_communities = []
            for comm in communities:
                entities = [
                    self.graph_store.entity_map[entity_id]
                    for entity_id in comm
                    if entity_id in self.graph_store.entity_map
                ]
                if entities:
                    entity_communities.append(entities)
            
            return entity_communities
            
        except Exception as e:
            self.logger.error(
                "Community detection failed",
                error=str(e)
            )
            return []
    
    def summarize_subgraph(self, subgraph: Subgraph) -> str:
        """
        Generate text summary of subgraph.
        
        Args:
            subgraph: Subgraph to summarize
            
        Returns:
            Text summary
        """
        lines = []
        
        # Entity summary
        lines.append(f"Entities ({len(subgraph.entities)}):")
        entity_types = defaultdict(int)
        for entity in subgraph.entities[:10]:  # Top 10
            etype = self._safe_entity_type(entity.entity_type)
            entity_types[etype] += 1
            lines.append(f"  - {entity.text} ({etype})")

        
        if len(subgraph.entities) > 10:
            lines.append(f"  ... and {len(subgraph.entities) - 10} more")
        
        # Relationship summary
        lines.append(f"\nRelationships ({len(subgraph.relationships)}):")
        rel_types = defaultdict(int)
        for rel in subgraph.relationships[:10]:  # Top 10
            rel_types[rel.predicate] += 1
            lines.append(f"  - {rel.subject} --[{rel.predicate}]--> {rel.object}")
        
        if len(subgraph.relationships) > 10:
            lines.append(f"  ... and {len(subgraph.relationships) - 10} more")
        
        # Statistics
        lines.append("\nStatistics:")
        lines.append(f"  Entity types: {dict(entity_types)}")
        lines.append(f"  Relationship types: {dict(rel_types)}")
        lines.append(f"  Relevance score: {subgraph.relevance_score:.2f}")
        
        return "\n".join(lines)
    
    def _empty_subgraph(self) -> Subgraph:
        """Create empty subgraph."""
        return Subgraph(
            subgraph_id=generate_id("subgraph"),
            central_entities=[],
            entities=[],
            relationships=[],
            relevance_score=0.0
        )
    
    def _safe_enum(self, val):
        return val.value if hasattr(val, "value") else str(val)
    
    def _safe_entity_type(self, entity_type):
        return entity_type.value if hasattr(entity_type, "value") else str(entity_type)

