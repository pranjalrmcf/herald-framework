"""
Relationship extractor for the research analyst system.
Extracts relationships (triples) between entities using LLM-based methods.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from research_analyst.core.models import Document, Entity, Relationship
from research_analyst.core.exceptions import RelationshipExtractionError
from research_analyst.config import get_settings, prompts
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id
from research_analyst.utils.llm_client import get_llm_client


logger = get_logger()


class RelationshipExtractor:
    """Extract relationships between entities."""
    
    def __init__(self):
        """Initialize relationship extractor."""
        self.settings = get_settings()
        self.logger = get_logger()
        self.llm_client = get_llm_client()
        
        # Common relationship patterns (for rule-based extraction)
        self.relationship_patterns = [
            (r'(.+?)\s+(?:is|was|are|were)\s+(?:the\s+)?([A-Z]+)\s+of\s+(.+)', 'role'),
            (r'(.+?)\s+founded\s+(.+)', 'founded'),
            (r'(.+?)\s+(?:owns|acquired|bought)\s+(.+)', 'owns'),
            (r'(.+?)\s+(?:partnered|collaborated)\s+with\s+(.+)', 'partners_with'),
            (r'(.+?)\s+(?:invested|invests)\s+in\s+(.+)', 'invests_in'),
        ]
    
    def extract_from_document(
        self,
        document: Document,
        entities: List[Entity],
        use_llm: bool = True
    ) -> List[Relationship]:
        """
        Extract relationships from a document.
        
        Args:
            document: Document to process
            entities: Known entities in the document
            use_llm: Whether to use LLM for extraction
            
        Returns:
            List of Relationship objects
        """
        self.logger.debug(
            "Extracting relationships from document",
            doc_id=document.doc_id,
            num_entities=len(entities)
        )
        
        if not document.content:
            self.logger.warning(
                "No content to extract relationships from",
                doc_id=document.doc_id
            )
            return []
        
        try:
            relationships = []
            
            # Rule-based extraction
            rule_relationships = self._extract_with_rules(document, entities)
            relationships.extend(rule_relationships)
            
            # LLM-based extraction (more comprehensive)
            if use_llm and not self.settings.mock_llm_calls:
                llm_relationships = self._extract_with_llm(document, entities)
                relationships.extend(llm_relationships)
            
            # Deduplicate
            relationships = self._deduplicate_relationships(relationships)
            
            self.logger.info(
                "Relationships extracted",
                doc_id=document.doc_id,
                num_relationships=len(relationships)
            )
            
            return relationships
            
        except Exception as e:
            self.logger.error(
                "Relationship extraction failed",
                doc_id=document.doc_id,
                error=str(e)
            )
            raise RelationshipExtractionError(
                f"Failed to extract relationships: {str(e)}",
                details={"doc_id": document.doc_id}
            )
    
    def _extract_with_rules(
        self,
        document: Document,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships using pattern matching.
        
        Args:
            document: Document to process
            entities: Known entities
            
        Returns:
            List of relationships
        """
        relationships = []
        text = document.content[:10000]  # Limit text
        
        # Create entity lookup
        entity_texts = {e.text.lower() for e in entities}
        
        for pattern, rel_type in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    subject = match.group(1).strip()
                    obj = match.group(2).strip() if len(match.groups()) == 2 else match.group(3).strip()
                    predicate = rel_type
                    
                    # Only create relationship if both entities are recognized
                    if subject.lower() in entity_texts and obj.lower() in entity_texts:
                        relationship = Relationship(
                            relationship_id=generate_id("rel"),
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            confidence=0.6,  # Rule-based has medium confidence
                            source_doc_id=document.doc_id,
                            source_url=document.url,
                            metadata={'extraction_method': 'rule-based'}
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _extract_with_llm(
            self,
            document: Document,
            entities: List[Entity]
        ) -> List[Relationship]:

        text_sample = document.content[:3000]

        prompt = prompts.format_prompt(
            prompts.RELATIONSHIP_EXTRACTION,
            text=text_sample,
            source_url=str(document.url)
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are an information extraction system.\n"
                    "Return ONLY valid JSON.\n"
                    "Do not include explanations, markdown, or text outside JSON.\n"
                    "JSON schema:\n"
                    "{ \"relationships\": ["
                    "{ \"subject\": str, \"predicate\": str, \"object\": str, "
                    "\"confidence\": float, \"context\": str } ] }"
                ),
                temperature=0.0,
                max_tokens=1500,
                json_mode=True
            )

            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                self.logger.warning(
                    "Invalid JSON from LLM relationship extraction",
                    raw_response=response[:300]
                )
                return []

            relationships = []
            for rel_data in result.get("relationships", []):
                relationship = Relationship(
                    relationship_id=generate_id("rel"),
                    subject=rel_data["subject"],
                    predicate=rel_data["predicate"],
                    object=rel_data["object"],
                    confidence=rel_data.get("confidence", 0.8),
                    source_doc_id=document.doc_id,
                    source_url=document.url,
                    metadata={
                        "extraction_method": "llm",
                        "context": rel_data.get("context", "")
                    }
                )
                relationships.append(relationship)

            return relationships

        except Exception as e:
            self.logger.warning(
                "LLM relationship extraction failed",
                error=str(e)
            )
            return []

        
    def _deduplicate_relationships(
        self,
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """
        Deduplicate relationships.
        
        Args:
            relationships: List of relationships
            
        Returns:
            Deduplicated relationships
        """
        # Use dict to deduplicate by (subject, predicate, object)
        rel_dict = {}
        
        for rel in relationships:
            # Normalize key
            key = (
                rel.subject.lower().strip(),
                rel.predicate.lower().strip(),
                rel.object.lower().strip()
            )
            
            # Keep relationship with higher confidence
            if key in rel_dict:
                if rel.confidence > rel_dict[key].confidence:
                    rel_dict[key] = rel
            else:
                rel_dict[key] = rel
        
        return list(rel_dict.values())
    
    def extract_from_multiple_documents(
        self,
        documents: List[Document],
        entities: List[Entity],
        use_llm: bool = True
    ) -> List[Relationship]:
        """
        Extract relationships from multiple documents.
        
        Args:
            documents: List of documents
            entities: Known entities across all documents
            use_llm: Whether to use LLM
            
        Returns:
            List of relationships
        """
        self.logger.info(
            "Extracting relationships from multiple documents",
            num_documents=len(documents),
            num_entities=len(entities)
        )
        
        all_relationships = []
        
        for doc in documents:
            try:
                # Filter entities relevant to this document
                doc_entities = [e for e in entities if e.source_doc_id == doc.doc_id]
                
                relationships = self.extract_from_document(
                    doc,
                    doc_entities,
                    use_llm=use_llm
                )
                all_relationships.extend(relationships)
                
            except RelationshipExtractionError as e:
                self.logger.warning(
                    "Failed to extract from document",
                    doc_id=doc.doc_id,
                    error=str(e)
                )
                continue
        
        # Global deduplication
        deduplicated = self._deduplicate_relationships(all_relationships)
        
        self.logger.info(
            "Multi-document relationship extraction complete",
            total_relationships=len(all_relationships),
            unique_relationships=len(deduplicated)
        )
        
        return deduplicated
    
    def enrich_relationships(
        self,
        relationships: List[Relationship],
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Enrich relationships with entity information.
        
        Args:
            relationships: List of relationships
            entities: List of entities
            
        Returns:
            Enriched relationships
        """
        # Create entity lookup
        entity_map = {e.text.lower(): e for e in entities}
        
        for rel in relationships:
            # Find matching entities
            subject_entity = entity_map.get(rel.subject.lower())
            object_entity = entity_map.get(rel.object.lower())
            
            # Add entity type information
            if subject_entity:
                rel.metadata['subject_type'] = subject_entity.entity_type.value
                rel.metadata['subject_id'] = subject_entity.entity_id
            
            if object_entity:
                rel.metadata['object_type'] = object_entity.entity_type.value
                rel.metadata['object_id'] = object_entity.entity_id
        
        return relationships
    
    def filter_relationships(
        self,
        relationships: List[Relationship],
        min_confidence: float = 0.5,
        required_predicates: Optional[List[str]] = None
    ) -> List[Relationship]:
        """
        Filter relationships by confidence and predicate.
        
        Args:
            relationships: List of relationships
            min_confidence: Minimum confidence threshold
            required_predicates: List of allowed predicates
            
        Returns:
            Filtered relationships
        """
        filtered = []
        
        for rel in relationships:
            # Filter by confidence
            if rel.confidence < min_confidence:
                continue
            
            # Filter by predicate
            if required_predicates and rel.predicate not in required_predicates:
                continue
            
            filtered.append(rel)
        
        self.logger.info(
            "Filtered relationships",
            original_count=len(relationships),
            filtered_count=len(filtered)
        )
        
        return filtered
    
    def find_relationship_chains(
        self,
        relationships: List[Relationship],
        start_entity: str,
        end_entity: str,
        max_length: int = 3
    ) -> List[List[Relationship]]:
        """
        Find chains of relationships connecting two entities.
        
        Args:
            relationships: List of relationships
            start_entity: Starting entity
            end_entity: Target entity
            max_length: Maximum chain length
            
        Returns:
            List of relationship chains (each chain is a list of relationships)
        """
        # Build adjacency list
        graph = {}
        for rel in relationships:
            if rel.subject not in graph:
                graph[rel.subject] = []
            graph[rel.subject].append(rel)
        
        # BFS to find paths
        chains = []
        queue = [([rel], rel.object) for rel in graph.get(start_entity, [])]
        
        while queue:
            current_chain, current_entity = queue.pop(0)
            
            # Check if we reached the target
            if current_entity.lower() == end_entity.lower():
                chains.append(current_chain)
                continue
            
            # Check max length
            if len(current_chain) >= max_length:
                continue
            
            # Explore neighbors
            for rel in graph.get(current_entity, []):
                new_chain = current_chain + [rel]
                queue.append((new_chain, rel.object))
        
        return chains