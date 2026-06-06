"""
Graph store for the research analyst system.
Stores and manages knowledge graphs using NetworkX (with optional Neo4j support).
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import networkx as nx

from research_analyst.core.models import KnowledgeGraph, Entity, Relationship, Subgraph
from research_analyst.core.exceptions import DatabaseError, GraphError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id


logger = get_logger()


class GraphStore:
    """Store and manage knowledge graphs."""
    
    def __init__(self):
        """Initialize graph store."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Initialize based on backend
        if self.settings.graph_backend == "neo4j":
            self._init_neo4j()
        else:
            self._init_networkx()
    
    def _init_networkx(self):
        """Initialize NetworkX backend."""
        self.backend = "networkx"
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.entity_map = {}  # entity_id -> Entity
        self.relationship_map = {}  # relationship_id -> Relationship
        
        self.logger.info("Initialized NetworkX graph store")
    
    def _init_neo4j(self):
        """Initialize Neo4j backend."""
        try:
            from neo4j import GraphDatabase
            
            self.backend = "neo4j"
            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password)
            )
            
            self.logger.info("Initialized Neo4j graph store")
            
        except ImportError:
            raise DatabaseError(
                "Neo4j library not installed",
                details={"required": "pip install neo4j"}
            )
        except Exception as e:
            raise DatabaseError(
                f"Failed to connect to Neo4j: {str(e)}",
                details={"uri": self.settings.neo4j_uri}
            )
        
    def _safe_enum(self, val):
        return val.value if hasattr(val, "value") else str(val)
    
    def store_graph(self, knowledge_graph: KnowledgeGraph):
        """
        Store a knowledge graph.
        
        Args:
            knowledge_graph: KnowledgeGraph to store
        """
        self.logger.info(
            "Storing knowledge graph",
            graph_id=knowledge_graph.graph_id,
            num_entities=len(knowledge_graph.entities),
            num_relationships=len(knowledge_graph.relationships)
        )
        
        if self.backend == "networkx":
            self._store_networkx(knowledge_graph)
        else:
            self._store_neo4j(knowledge_graph)
    
    def _store_networkx(self, knowledge_graph: KnowledgeGraph):
        """Store graph in NetworkX."""
        # Clear existing graph
        self.graph.clear()
        self.entity_map.clear()
        self.relationship_map.clear()
        
        # Add entities as nodes
        for entity in knowledge_graph.entities:
            self.graph.add_node(
                entity.entity_id,
                text=entity.text,
                type=self._safe_enum(entity.entity_type),
                confidence=entity.confidence,
                aliases=entity.aliases,
                attributes=entity.attributes
            )
            self.entity_map[entity.entity_id] = entity
        
        # Add relationships as edges
        for relationship in knowledge_graph.relationships:
            subject_id = relationship.metadata.get('subject_entity_id')
            object_id = relationship.metadata.get('object_entity_id')
            
            if subject_id and object_id:
                self.graph.add_edge(
                    subject_id,
                    object_id,
                    key=relationship.relationship_id,
                    predicate=relationship.predicate,
                    confidence=relationship.confidence,
                    source_url=str(relationship.source_url),
                    temporal_info=relationship.temporal_info,
                    metadata=relationship.metadata
                )
                self.relationship_map[relationship.relationship_id] = relationship
        
        self.logger.info(
            "Graph stored in NetworkX",
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges()
        )
    
    def _store_neo4j(self, knowledge_graph: KnowledgeGraph):
        """Store graph in Neo4j."""
        with self.driver.session() as session:
            # Clear existing graph (optional - you might want to append instead)
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create entities
            for entity in knowledge_graph.entities:
                session.run(
                    """
                    CREATE (e:Entity {
                        entity_id: $entity_id,
                        text: $text,
                        type: $type,
                        confidence: $confidence,
                        aliases: $aliases
                    })
                    """,
                    entity_id=entity.entity_id,
                    text=entity.text,
                    type=entity.entity_type.value,
                    confidence=entity.confidence,
                    aliases=entity.aliases
                )
            
            # Create relationships
            for rel in knowledge_graph.relationships:
                subject_id = rel.metadata.get('subject_entity_id')
                object_id = rel.metadata.get('object_entity_id')
                
                if subject_id and object_id:
                    session.run(
                        """
                        MATCH (s:Entity {entity_id: $subject_id})
                        MATCH (o:Entity {entity_id: $object_id})
                        CREATE (s)-[r:RELATES {
                            relationship_id: $rel_id,
                            predicate: $predicate,
                            confidence: $confidence
                        }]->(o)
                        """,
                        subject_id=subject_id,
                        object_id=object_id,
                        rel_id=rel.relationship_id,
                        predicate=rel.predicate,
                        confidence=rel.confidence
                    )
        
        self.logger.info("Graph stored in Neo4j")
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity object or None
        """
        if self.backend == "networkx":
            return self.entity_map.get(entity_id)
        else:
            # Neo4j query
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (e:Entity {entity_id: $entity_id}) RETURN e",
                    entity_id=entity_id
                )
                record = result.single()
                if record:
                    # Convert to Entity object
                    # (implementation details omitted for brevity)
                    return None  # Placeholder
            return None
    
    def get_relationships_for_entity(
        self,
        entity_id: str,
        direction: str = "both"
    ) -> List[Relationship]:
        """
        Get all relationships for an entity.
        
        Args:
            entity_id: Entity ID
            direction: "in", "out", or "both"
            
        Returns:
            List of relationships
        """
        if self.backend == "networkx":
            relationships = []
            
            if direction in ["out", "both"]:
                # Outgoing edges
                for _, target, key, data in self.graph.out_edges(entity_id, keys=True, data=True):
                    rel_id = key
                    if rel_id in self.relationship_map:
                        relationships.append(self.relationship_map[rel_id])
            
            if direction in ["in", "both"]:
                # Incoming edges
                for source, _, key, data in self.graph.in_edges(entity_id, keys=True, data=True):
                    rel_id = key
                    if rel_id in self.relationship_map:
                        relationships.append(self.relationship_map[rel_id])
            
            return relationships
        else:
            # Neo4j implementation
            return []
    
    def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
        direction: str = "both"
    ) -> Set[str]:
        """
        Get neighbor entities within N hops.
        
        Args:
            entity_id: Starting entity ID
            hops: Number of hops
            direction: "in", "out", or "both"
            
        Returns:
            Set of entity IDs
        """
        if self.backend == "networkx":
            if not self.graph.has_node(entity_id):
                return set()
            
            neighbors = set()
            current_level = {entity_id}
            
            for _ in range(hops):
                next_level = set()
                
                for node in current_level:
                    if direction in ["out", "both"]:
                        next_level.update(self.graph.successors(node))
                    if direction in ["in", "both"]:
                        next_level.update(self.graph.predecessors(node))
                
                neighbors.update(next_level)
                current_level = next_level
            
            # Remove the starting entity
            neighbors.discard(entity_id)
            
            return neighbors
        else:
            # Neo4j implementation
            return set()
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        if self.backend == "networkx":
            try:
                # Find all simple paths
                paths = list(nx.all_simple_paths(
                    self.graph,
                    source_id,
                    target_id,
                    cutoff=max_length
                ))
                return paths
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return []
        else:
            # Neo4j implementation
            return []
    
    def get_subgraph(
        self,
        central_entities: List[str],
        hops: int = 1
    ) -> Subgraph:
        """
        Extract subgraph around central entities.
        
        Args:
            central_entities: List of central entity IDs
            hops: Number of hops to include
            
        Returns:
            Subgraph object
        """
        self.logger.info(
            "Extracting subgraph",
            num_central_entities=len(central_entities),
            hops=hops
        )
        
        # Get all neighbors within hops
        all_entity_ids = set(central_entities)
        
        for entity_id in central_entities:
            neighbors = self.get_neighbors(entity_id, hops=hops)
            all_entity_ids.update(neighbors)
        
        # Get entities
        entities = [
            self.entity_map[eid]
            for eid in all_entity_ids
            if eid in self.entity_map
        ]
        
        # Get relationships between these entities
        relationships = []
        for rel in self.relationship_map.values():
            subject_id = rel.metadata.get('subject_entity_id')
            object_id = rel.metadata.get('object_entity_id')
            
            if subject_id in all_entity_ids and object_id in all_entity_ids:
                relationships.append(rel)
        
        # Calculate relevance score (simple heuristic)
        relevance_score = len(relationships) / max(len(entities), 1)
        
        subgraph = Subgraph(
            subgraph_id=generate_id("subgraph"),
            central_entities=central_entities,
            entities=entities,
            relationships=relationships,
            relevance_score=min(1.0, relevance_score)
        )
        
        self.logger.info(
            "Subgraph extracted",
            num_entities=len(entities),
            num_relationships=len(relationships)
        )
        
        return subgraph
    
    def save_to_file(self, filepath: str):
        """
        Save graph to file.
        
        Args:
            filepath: Path to save file
        """
        if self.backend == "networkx":
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save NetworkX graph
            graph_path = str(path.with_suffix('.gpickle'))
            nx.write_gpickle(self.graph, graph_path)
            
            # Save entity and relationship maps
            maps_path = str(path.with_suffix('.maps.pkl'))
            with open(maps_path, 'wb') as f:
                pickle.dump({
                    'entity_map': self.entity_map,
                    'relationship_map': self.relationship_map
                }, f)
            
            self.logger.info(
                "Graph saved",
                graph_path=graph_path,
                maps_path=maps_path
            )
        else:
            self.logger.warning("Neo4j graphs are stored in database, not files")
    
    def load_from_file(self, filepath: str):
        """
        Load graph from file.
        
        Args:
            filepath: Path to load from
        """
        if self.backend == "networkx":
            path = Path(filepath)
            
            graph_path = str(path.with_suffix('.gpickle'))
            maps_path = str(path.with_suffix('.maps.pkl'))
            
            if not Path(graph_path).exists() or not Path(maps_path).exists():
                raise GraphError(
                    f"Graph files not found at {filepath}",
                    details={"graph_path": graph_path, "maps_path": maps_path}
                )
            
            # Load graph
            self.graph = nx.read_gpickle(graph_path)
            
            # Load maps
            with open(maps_path, 'rb') as f:
                maps = pickle.load(f)
                self.entity_map = maps['entity_map']
                self.relationship_map = maps['relationship_map']
            
            self.logger.info(
                "Graph loaded",
                nodes=self.graph.number_of_nodes(),
                edges=self.graph.number_of_edges()
            )
        else:
            self.logger.warning("Neo4j graphs are loaded from database")
    
    def get_graph_stats(self) -> Dict:
        """
        Get statistics about stored graph.
        
        Returns:
            Statistics dictionary
        """
        if self.backend == "networkx":
            stats = {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'is_connected': nx.is_weakly_connected(self.graph),
                'num_components': nx.number_weakly_connected_components(self.graph),
            }
            
            # Calculate density (if not too large)
            if stats['num_nodes'] < 1000:
                stats['density'] = nx.density(self.graph)
            
            return stats
        else:
            return {}
    
    def clear(self):
        """Clear all stored graph data."""
        if self.backend == "networkx":
            self.graph.clear()
            self.entity_map.clear()
            self.relationship_map.clear()
        else:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        
        self.logger.info("Graph store cleared")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.backend == "neo4j" and hasattr(self, 'driver'):
            self.driver.close()