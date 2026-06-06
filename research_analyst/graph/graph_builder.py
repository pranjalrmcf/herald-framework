"""
Graph builder for the research analyst system.
Constructs knowledge graphs from extracted entities and relationships.
"""

from typing import List, Dict, Optional
from datetime import datetime

from research_analyst.core.models import Entity, Relationship, KnowledgeGraph
from research_analyst.core.exceptions import GraphConstructionError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id


logger = get_logger()


class GraphBuilder:
    """Build knowledge graphs from entities and relationships."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.settings = get_settings()
        self.logger = get_logger()


    def _safe_enum(self, val):
        return val.value if hasattr(val, "value") else str(val)

    
    def build_graph(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> KnowledgeGraph:
        """
        Build a knowledge graph from entities and relationships.
        
        Args:
            entities: List of entities
            relationships: List of relationships
            
        Returns:
            KnowledgeGraph object
        """
        self.logger.info(
            "Building knowledge graph",
            num_entities=len(entities),
            num_relationships=len(relationships)
        )
        
        try:
            # Merge duplicate entities
            merged_entities = self._merge_entities(entities)
            
            # Link relationships to merged entities
            linked_relationships = self._link_relationships(
                relationships,
                merged_entities
            )
            
            # Create knowledge graph
            graph = KnowledgeGraph(
                graph_id=generate_id("graph"),
                entities=merged_entities,
                relationships=linked_relationships,
                created_at=datetime.utcnow(),
                metadata={
                    'num_entities': len(merged_entities),
                    'num_relationships': len(linked_relationships),
                    'entity_types': self._count_entity_types(merged_entities),
                    'relationship_types': self._count_relationship_types(linked_relationships)
                }
            )
            
            self.logger.info(
                "Knowledge graph built",
                graph_id=graph.graph_id,
                entities=len(merged_entities),
                relationships=len(linked_relationships)
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(
                "Graph construction failed",
                error=str(e)
            )
            raise GraphConstructionError(
                f"Failed to build graph: {str(e)}",
                details={
                    'num_entities': len(entities),
                    'num_relationships': len(relationships)
                }
            )
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Merge duplicate entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Merged entities
        """
        # Group entities by normalized text and type
        entity_groups = {}
        
        for entity in entities:
            # Create key from text and type
            key = (entity.text.lower().strip(), entity.entity_type)
            
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Merge each group
        merged_entities = []
        for key, group in entity_groups.items():
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                merged = self._merge_entity_group(group)
                merged_entities.append(merged)
        
        self.logger.debug(
            "Entities merged",
            original=len(entities),
            merged=len(merged_entities)
        )
        
        return merged_entities
    
    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """
        Merge a group of similar entities.
        
        Args:
            entities: Entities to merge
            
        Returns:
            Merged entity
        """
        # Use highest confidence entity as base
        base = max(entities, key=lambda e: e.confidence)
        
        # Collect all aliases
        all_aliases = set(base.aliases)
        for entity in entities:
            all_aliases.update(entity.aliases)
            # Add variant spellings as aliases
            if entity.text != base.text:
                all_aliases.add(entity.text)
        
        base.aliases = list(all_aliases)
        
        # Merge attributes
        merged_attrs = {}
        for entity in entities:
            merged_attrs.update(entity.attributes)
        
        # Track source documents
        source_docs = list(set(e.source_doc_id for e in entities))
        merged_attrs['source_documents'] = source_docs
        merged_attrs['mention_count'] = len(entities)
        
        base.attributes = merged_attrs
        
        # Update confidence (higher if mentioned multiple times)
        base.confidence = min(1.0, base.confidence * (1 + 0.05 * len(entities)))
        
        return base
    
    def _link_relationships(
        self,
        relationships: List[Relationship],
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Link relationships to canonical entity IDs.
        
        Args:
            relationships: List of relationships
            entities: Merged entities
            
        Returns:
            Linked relationships
        """
        # Create entity lookup by text
        entity_map = {}
        for entity in entities:
            # Add main text
            entity_map[entity.text.lower().strip()] = entity.entity_id
            # Add aliases
            for alias in entity.aliases:
                entity_map[alias.lower().strip()] = entity.entity_id
        
        # Link relationships
        linked_relationships = []
        
        for rel in relationships:
            # Try to find entity IDs
            subject_id = entity_map.get(rel.subject.lower().strip())
            object_id = entity_map.get(rel.object.lower().strip())
            
            # Only keep relationships where both entities are known
            if subject_id and object_id:
                # Add entity IDs to metadata
                rel.metadata['subject_entity_id'] = subject_id
                rel.metadata['object_entity_id'] = object_id
                linked_relationships.append(rel)
            else:
                self.logger.debug(
                    "Skipping relationship with unknown entities",
                    subject=rel.subject,
                    object=rel.object,
                    subject_found=bool(subject_id),
                    object_found=bool(object_id)
                )
        
        self.logger.debug(
            "Relationships linked",
            original=len(relationships),
            linked=len(linked_relationships)
        )
        
        return linked_relationships
    
    def _count_entity_types(self, entities: List[Entity]) -> Dict[str, int]:
        """Count entities by type."""
        counts = {}
        for entity in entities:
            entity_type = self._safe_enum(entity.entity_type)
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts
    
    def _count_relationship_types(self, relationships: List[Relationship]) -> Dict[str, int]:
        """Count relationships by predicate."""
        counts = {}
        for rel in relationships:
            predicate = self._safe_enum(rel.predicate)
            counts[predicate] = counts.get(predicate, 0) + 1

            # counts[rel.predicate] = counts.get(rel.predicate, 0) + 1
        return counts
    
    def add_to_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: List[Entity],
        new_relationships: List[Relationship]
    ) -> KnowledgeGraph:
        """
        Add new entities and relationships to existing graph.
        
        Args:
            graph: Existing graph
            new_entities: New entities to add
            new_relationships: New relationships to add
            
        Returns:
            Updated graph
        """
        self.logger.info(
            "Adding to existing graph",
            graph_id=graph.graph_id,
            new_entities=len(new_entities),
            new_relationships=len(new_relationships)
        )
        
        # Combine entities and merge
        all_entities = graph.entities + new_entities
        merged_entities = self._merge_entities(all_entities)
        
        # Combine relationships and deduplicate
        all_relationships = graph.relationships + new_relationships
        linked_relationships = self._link_relationships(
            all_relationships,
            merged_entities
        )
        
        # Update graph
        graph.entities = merged_entities
        graph.relationships = linked_relationships
        graph.metadata['num_entities'] = len(merged_entities)
        graph.metadata['num_relationships'] = len(linked_relationships)
        graph.metadata['entity_types'] = self._count_entity_types(merged_entities)
        graph.metadata['relationship_types'] = self._count_relationship_types(linked_relationships)
        
        return graph
    
    def prune_graph(
        self,
        graph: KnowledgeGraph,
        min_entity_confidence: float = 0.3,
        min_relationship_confidence: float = 0.4,
        remove_isolated: bool = True
    ) -> KnowledgeGraph:
        """
        Prune low-quality nodes and edges from graph.
        
        Args:
            graph: Knowledge graph
            min_entity_confidence: Minimum entity confidence
            min_relationship_confidence: Minimum relationship confidence
            remove_isolated: Whether to remove isolated entities
            
        Returns:
            Pruned graph
        """
        self.logger.info(
            "Pruning graph",
            graph_id=graph.graph_id
        )
        
        # Filter entities by confidence
        filtered_entities = [
            e for e in graph.entities
            if e.confidence >= min_entity_confidence
        ]
        
        # Filter relationships by confidence
        filtered_relationships = [
            r for r in graph.relationships
            if r.confidence >= min_relationship_confidence
        ]
        
        # Relink relationships to filtered entities
        filtered_relationships = self._link_relationships(
            filtered_relationships,
            filtered_entities
        )
        
        # Remove isolated entities if requested
        if remove_isolated:
            filtered_entities = self._remove_isolated_entities(
                filtered_entities,
                filtered_relationships
            )
        
        # Update graph
        graph.entities = filtered_entities
        graph.relationships = filtered_relationships
        graph.metadata['num_entities'] = len(filtered_entities)
        graph.metadata['num_relationships'] = len(filtered_relationships)
        
        self.logger.info(
            "Graph pruned",
            entities_removed=len(graph.entities) - len(filtered_entities),
            relationships_removed=len(graph.relationships) - len(filtered_relationships)
        )
        
        return graph
    
    def _remove_isolated_entities(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> List[Entity]:
        """
        Remove entities that have no relationships.
        
        Args:
            entities: List of entities
            relationships: List of relationships
            
        Returns:
            Entities with at least one relationship
        """
        # Find entities mentioned in relationships
        connected_entity_ids = set()
        for rel in relationships:
            subject_id = rel.metadata.get('subject_entity_id')
            object_id = rel.metadata.get('object_entity_id')
            if subject_id:
                connected_entity_ids.add(subject_id)
            if object_id:
                connected_entity_ids.add(object_id)
        
        # Keep only connected entities
        connected_entities = [
            e for e in entities
            if e.entity_id in connected_entity_ids
        ]
        
        return connected_entities
    
    def get_graph_stats(self, graph: KnowledgeGraph) -> Dict:
        """
        Get statistics about the graph.
        
        Args:
            graph: Knowledge graph
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'num_entities': len(graph.entities),
            'num_relationships': len(graph.relationships),
            'entity_types': graph.metadata.get('entity_types', {}),
            'relationship_types': graph.metadata.get('relationship_types', {}),
            'avg_entity_confidence': sum(e.confidence for e in graph.entities) / len(graph.entities) if graph.entities else 0,
            'avg_relationship_confidence': sum(r.confidence for r in graph.relationships) / len(graph.relationships) if graph.relationships else 0,
        }
        
        return stats