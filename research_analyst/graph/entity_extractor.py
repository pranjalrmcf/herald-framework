"""
Entity extractor for the research analyst system.
Extracts named entities using spaCy and LLM-based methods.
"""

import re
from typing import List, Dict, Set, Optional
from collections import defaultdict

from research_analyst.core.models import Document, Entity, EntityType
from research_analyst.core.exceptions import EntityExtractionError
from research_analyst.config import get_settings, prompts
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id


logger = get_logger()


class EntityExtractor:
    """Extract entities from documents."""
    
    def __init__(self):
        """Initialize entity extractor."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Initialize spaCy model
        self._init_spacy()
        
        # Entity type mapping from spaCy to our types
        self.entity_type_mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'PRODUCT': EntityType.PRODUCT,
            'EVENT': EntityType.EVENT,
            'DATE': EntityType.DATE,
        }
    
    def _init_spacy(self):
        """Initialize spaCy NLP model."""
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_trf")
                self.logger.info("Loaded spaCy transformer model")
            except OSError:
                # Fallback to smaller model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.logger.warning("Using smaller spaCy model (en_core_web_sm)")
                except OSError:
                    raise EntityExtractionError(
                        "No spaCy model found. Run: python -m spacy download en_core_web_trf"
                    )
        except ImportError:
            raise EntityExtractionError(
                "spaCy not installed. Run: pip install spacy"
            )
    
    def extract_from_document(
        self,
        document: Document,
        use_llm: bool = False
    ) -> List[Entity]:
        """
        Extract entities from a document.
        
        Args:
            document: Document to process
            use_llm: Whether to use LLM for enhanced extraction
            
        Returns:
            List of Entity objects
        """
        self.logger.debug(
            "Extracting entities from document",
            doc_id=document.doc_id,
            use_llm=use_llm
        )
        
        if not document.content:
            self.logger.warning(
                "No content to extract entities from",
                doc_id=document.doc_id
            )
            return []
        
        try:
            # Extract using spaCy
            spacy_entities = self._extract_with_spacy(document)
            
            # Optionally enhance with LLM
            if use_llm and not self.settings.mock_llm_calls:
                llm_entities = self._extract_with_llm(document)
                entities = self._merge_entities(spacy_entities, llm_entities)
            else:
                entities = spacy_entities
            
            # Deduplicate and resolve coreferences
            entities = self._deduplicate_entities(entities)
            
            self.logger.info(
                "Entities extracted",
                doc_id=document.doc_id,
                num_entities=len(entities)
            )
            
            return entities
            
        except Exception as e:
            self.logger.error(
                "Entity extraction failed",
                doc_id=document.doc_id,
                error=str(e)
            )
            raise EntityExtractionError(
                f"Failed to extract entities: {str(e)}",
                details={"doc_id": document.doc_id}
            )
    
    def _extract_with_spacy(self, document: Document) -> List[Entity]:
        """
        Extract entities using spaCy NER.
        
        Args:
            document: Document to process
            
        Returns:
            List of entities
        """
        # Process with spaCy (limit text length)
        text = document.content[:100000]  # Limit to avoid memory issues
        doc = self.nlp(text)
        
        entities = []
        seen_entities = set()
        
        for ent in doc.ents:
            # Map spaCy entity type to our types
            entity_type = self.entity_type_mapping.get(
                ent.label_,
                EntityType.UNKNOWN
            )
            
            # Skip if already seen
            entity_key = (ent.text.lower(), entity_type)
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)
            
            # Create Entity object
            entity = Entity(
                entity_id=generate_id("entity"),
                text=ent.text,
                entity_type=entity_type,
                aliases=[],
                confidence=0.8,  # spaCy is generally reliable
                source_doc_id=document.doc_id,
                attributes={
                    'spacy_label': ent.label_,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char
                }
            )
            
            entities.append(entity)
        
        return entities
    
    def _extract_with_llm(self, document: Document) -> List[Entity]:
        """
        Extract entities using LLM (for enhanced accuracy).
        
        Args:
            document: Document to process
            
        Returns:
            List of entities
        """
        # Use first 2000 chars to avoid token limits
        text_sample = document.content[:2000]
        
        # Format prompt
        prompt = prompts.format_prompt(
            prompts.ENTITY_EXTRACTION_FROM_TEXT,
            text=text_sample
        )
        
        try:
            # Call LLM (simplified - would use actual LLM client)
            from research_analyst.query_processing import IntentClassifier
            classifier = IntentClassifier()

            response = classifier._call_llm(prompt)

            
            # if self.settings.default_llm_provider == "openai":
            #     response = classifier._call_openai(prompt)
            # else:
            #     response = classifier._call_anthropic(prompt)
            
            # Parse JSON response
            import json
            result = json.loads(response)
            
            # Convert to Entity objects
            entities = []
            for ent_data in result.get('entities', []):
                entity_type_str = ent_data.get('type', 'UNKNOWN')
                
                # Map string to EntityType
                try:
                    entity_type = EntityType[entity_type_str]
                except KeyError:
                    entity_type = EntityType.UNKNOWN
                
                entity = Entity(
                    entity_id=generate_id("entity"),
                    text=ent_data['text'],
                    entity_type=entity_type,
                    aliases=ent_data.get('aliases', []),
                    confidence=ent_data.get('confidence', 0.7),
                    source_doc_id=document.doc_id,
                    attributes=ent_data.get('attributes', {})
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            self.logger.warning(
                "LLM entity extraction failed",
                error=str(e)
            )
            return []
    
    def _merge_entities(
        self,
        spacy_entities: List[Entity],
        llm_entities: List[Entity]
    ) -> List[Entity]:
        """
        Merge entities from spaCy and LLM.
        
        Args:
            spacy_entities: Entities from spaCy
            llm_entities: Entities from LLM
            
        Returns:
            Merged list of entities
        """
        # Use dict to deduplicate by text
        entity_dict = {}
        
        # Add spaCy entities first
        for entity in spacy_entities:
            key = entity.text.lower()
            entity_dict[key] = entity
        
        # Add LLM entities, preferring higher confidence
        for entity in llm_entities:
            key = entity.text.lower()
            if key in entity_dict:
                # Keep entity with higher confidence
                if entity.confidence > entity_dict[key].confidence:
                    entity_dict[key] = entity
            else:
                entity_dict[key] = entity
        
        return list(entity_dict.values())
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities and merge aliases.
        
        Args:
            entities: List of entities
            
        Returns:
            Deduplicated entities
        """
        # Group by normalized text
        entity_groups = defaultdict(list)
        
        for entity in entities:
            # Normalize key
            key = entity.text.lower().strip()
            entity_groups[key].append(entity)
        
        deduplicated = []
        
        for key, group in entity_groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge group into single entity
                merged = self._merge_entity_group(group)
                deduplicated.append(merged)
        
        return deduplicated
    
    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """
        Merge multiple entities that refer to the same thing.
        
        Args:
            entities: List of entities to merge
            
        Returns:
            Merged entity
        """
        # Use entity with highest confidence as base
        base_entity = max(entities, key=lambda e: e.confidence)
        
        # Collect all aliases
        all_aliases = set()
        for entity in entities:
            all_aliases.update(entity.aliases)
            # Add the entity text itself as an alias if different from base
            if entity.text != base_entity.text:
                all_aliases.add(entity.text)
        
        base_entity.aliases = list(all_aliases)
        
        # Merge attributes
        merged_attributes = {}
        for entity in entities:
            merged_attributes.update(entity.attributes)
        base_entity.attributes = merged_attributes
        
        # Average confidence
        base_entity.confidence = sum(e.confidence for e in entities) / len(entities)
        
        return base_entity
    
    def extract_from_multiple_documents(
        self,
        documents: List[Document],
        use_llm: bool = False
    ) -> List[Entity]:
        """
        Extract entities from multiple documents.
        
        Args:
            documents: List of documents
            use_llm: Whether to use LLM
            
        Returns:
            Deduplicated list of entities across all documents
        """
        self.logger.info(
            "Extracting entities from multiple documents",
            num_documents=len(documents)
        )
        
        all_entities = []
        
        for doc in documents:
            try:
                entities = self.extract_from_document(doc, use_llm=use_llm)
                all_entities.extend(entities)
            except EntityExtractionError as e:
                self.logger.warning(
                    "Failed to extract from document",
                    doc_id=doc.doc_id,
                    error=str(e)
                )
                continue
        
        # Global deduplication across documents
        deduplicated = self._global_deduplicate(all_entities)
        
        self.logger.info(
            "Multi-document extraction complete",
            total_entities=len(all_entities),
            unique_entities=len(deduplicated)
        )
        
        return deduplicated
    
    def _global_deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities across multiple documents.
        
        Args:
            entities: All entities
            
        Returns:
            Deduplicated entities
        """
        # Group by text and type
        entity_map = defaultdict(list)
        
        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)
            entity_map[key].append(entity)
        
        deduplicated = []
        
        for key, group in entity_map.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge entities from different documents
                merged = self._merge_cross_document_entities(group)
                deduplicated.append(merged)
        
        return deduplicated
    
    def _merge_cross_document_entities(self, entities: List[Entity]) -> Entity:
        """
        Merge entities that appear across multiple documents.
        
        Args:
            entities: Entities to merge
            
        Returns:
            Merged entity
        """
        # Use highest confidence entity as base
        base_entity = max(entities, key=lambda e: e.confidence)
        
        # Collect source documents
        source_docs = list(set(e.source_doc_id for e in entities))
        
        # Merge aliases
        all_aliases = set()
        for entity in entities:
            all_aliases.update(entity.aliases)
        
        base_entity.aliases = list(all_aliases)
        base_entity.attributes['source_documents'] = source_docs
        base_entity.attributes['mention_count'] = len(entities)
        
        # Higher confidence if mentioned in multiple documents
        base_entity.confidence = min(1.0, base_entity.confidence * (1 + 0.1 * len(source_docs)))
        
        return base_entity
    
    def filter_entities(
        self,
        entities: List[Entity],
        min_confidence: float = 0.5,
        entity_types: Optional[List[EntityType]] = None
    ) -> List[Entity]:
        """
        Filter entities by confidence and type.
        
        Args:
            entities: List of entities
            min_confidence: Minimum confidence threshold
            entity_types: List of allowed entity types
            
        Returns:
            Filtered entities
        """
        filtered = []
        
        for entity in entities:
            # Filter by confidence
            if entity.confidence < min_confidence:
                continue
            
            # Filter by type
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            filtered.append(entity)
        
        self.logger.info(
            "Filtered entities",
            original_count=len(entities),
            filtered_count=len(filtered)
        )
        
        return filtered