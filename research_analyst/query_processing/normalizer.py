"""
Query normalizer for the research analyst system.
Cleans, structures, and enhances queries before processing.
"""

import re
from typing import List, Dict, Optional
from datetime import datetime
import dateparser

from research_analyst.core.models import Query, NormalizedQuery, QueryComplexity
from research_analyst.core.exceptions import QueryNormalizationError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import clean_text, parse_time_range


logger = get_logger()


class QueryNormalizer:
    """Normalize and structure user queries."""
    
    def __init__(self):
        """Initialize query normalizer."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Common question words to extract intent
        self.question_patterns = {
            'what': ['definition', 'explanation'],
            'who': ['person', 'entity'],
            'where': ['location'],
            'when': ['time', 'temporal'],
            'why': ['reasoning', 'explanation'],
            'how': ['process', 'method'],
            'which': ['comparison', 'choice'],
        }
        
        # Stop words for entity extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
            'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
        }
    
    def normalize(self, query: Query) -> NormalizedQuery:
        """
        Normalize a query.
        
        Args:
            query: Input query
            
        Returns:
            Normalized query
            
        Raises:
            QueryNormalizationError: If normalization fails
        """
        query_id = query.metadata.get("query_id", "unknown")
        
        self.logger.info(
            "Normalizing query",
            query_id=query_id,
            original_length=len(query.text)
        )
        
        try:
            # 1. Clean and normalize text
            normalized_text = self._normalize_text(query.text)
            
            # 2. Detect language (simplified - assume English for now)
            language = self._detect_language(normalized_text)
            
            # 3. Extract mentioned entities (simple keyword extraction)
            entities = self._extract_entities(normalized_text)
            
            # 4. Parse time range if present
            time_range = self._parse_time_range(normalized_text)
            
            # 5. Determine domain (simplified heuristic)
            domain = self._detect_domain(normalized_text, entities)
            
            # 6. Estimate complexity
            complexity = self._estimate_complexity(
                normalized_text,
                entities,
                time_range
            )
            
            # Create normalized query (intent will be set by intent classifier)
            from research_analyst.core.models import QueryIntent
            normalized_query = NormalizedQuery(
                original_text=query.text,
                normalized_text=normalized_text,
                intent=QueryIntent.SEMANTIC,  # Placeholder - will be updated by classifier
                domain=domain,
                time_range=time_range,
                entities_mentioned=entities,
                language=language,
                complexity=complexity,
                requires_graph=False  # Will be updated by intent classifier
            )
            
            self.logger.info(
                "Query normalized",
                query_id=query_id,
                complexity=complexity.value,
                num_entities=len(entities),
                domain=domain
            )
            
            return normalized_query
            
        except Exception as e:
            self.logger.error(
                "Query normalization failed",
                query_id=query_id,
                error=str(e)
            )
            raise QueryNormalizationError(
                f"Failed to normalize query: {str(e)}",
                details={"query_id": query_id, "original_query": query.text}
            )
    
    def _normalize_text(self, text: str) -> str:
        """
        Clean and normalize query text.
        
        Args:
            text: Raw query text
            
        Returns:
            Normalized text
        """
        # Use helper function for basic cleaning
        normalized = clean_text(text)
        
        # Convert to lowercase for processing (but keep original case for entities)
        # We'll use lowercase for analysis but return mixed case
        
        # Remove multiple question marks, exclamation points
        normalized = re.sub(r'[?!]{2,}', '?', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove trailing punctuation except question marks
        if not normalized.endswith('?'):
            normalized = normalized.rstrip('.,!;:')
        
        return normalized
    
    def _detect_language(self, text: str) -> str:
        """
        Detect query language (simplified - assumes English).
        
        Args:
            text: Query text
            
        Returns:
            Language code
        """
        # In production, use langdetect or similar library
        # For now, assume English
        return "en"
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract potential entities from query (simple keyword-based).
        
        Args:
            text: Query text
            
        Returns:
            List of potential entity mentions
        """
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        words = text.split()
        
        # Extract capitalized sequences
        current_entity = []
        for word in words:
            # Check if word is capitalized and not a stop word
            word_clean = re.sub(r'[^\w\s]', '', word)
            if word_clean and word_clean[0].isupper() and word_clean.lower() not in self.stop_words:
                current_entity.append(word_clean)
            else:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
        
        # Add last entity if exists
        if current_entity:
            entities.append(' '.join(current_entity))
        
        # Also extract quoted strings as potential entities
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        
        # Remove duplicates and short entities
        entities = list(set([e for e in entities if len(e) > 2]))
        
        return entities
    
    def _parse_time_range(self, text: str) -> Optional[Dict]:
        """
        Parse time range from query.
        
        Args:
            text: Query text
            
        Returns:
            Time range dict or None
        """
        # Look for time-related keywords
        time_keywords = [
            'yesterday', 'today', 'tomorrow',
            'last week', 'last month', 'last year',
            'this week', 'this month', 'this year',
            'recent', 'recently', 'latest', 'current',
            'past', 'previous',
            r'\d{4}',  # Years like 2023
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # Dates
        ]
        
        text_lower = text.lower()
        
        # Check for time keywords
        time_info = {}
        for keyword in time_keywords:
            if isinstance(keyword, str) and keyword in text_lower:
                # Try to parse with dateparser
                parsed = dateparser.parse(
                    keyword,
                    settings={'RELATIVE_BASE': datetime.utcnow()}
                )
                if parsed:
                    time_info['keyword'] = keyword
                    time_info['parsed_date'] = parsed.isoformat()
                    break
            elif re.search(keyword, text):
                match = re.search(keyword, text)
                if match:
                    time_info['keyword'] = match.group()
                    break
        
        return time_info if time_info else None
    
    def _detect_domain(self, text: str, entities: List[str]) -> Optional[str]:
        """
        Detect query domain (simplified heuristic-based).
        
        Args:
            text: Query text
            entities: Extracted entities
            
        Returns:
            Domain string or None
        """
        text_lower = text.lower()
        
        # Domain keywords
        domains = {
            'technology': ['ai', 'ml', 'software', 'hardware', 'computer', 'tech', 'algorithm', 'code', 'programming'],
            'science': ['research', 'study', 'experiment', 'theory', 'hypothesis', 'scientific', 'biology', 'physics', 'chemistry'],
            'business': ['company', 'corporation', 'startup', 'investment', 'market', 'stock', 'ceo', 'business', 'finance'],
            'politics': ['government', 'policy', 'election', 'president', 'minister', 'law', 'politics', 'senate'],
            'health': ['health', 'medical', 'disease', 'treatment', 'doctor', 'patient', 'medicine', 'clinical'],
            'entertainment': ['movie', 'film', 'actor', 'music', 'game', 'entertainment', 'celebrity'],
        }
        
        # Count domain keyword matches
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return None
    
    def _estimate_complexity(
        self,
        text: str,
        entities: List[str],
        time_range: Optional[Dict]
    ) -> QueryComplexity:
        """
        Estimate query complexity.
        
        Args:
            text: Query text
            entities: Extracted entities
            time_range: Parsed time range
            
        Returns:
            QueryComplexity enum
        """
        complexity_score = 0
        
        # Length-based factors
        word_count = len(text.split())
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Entity-based factors
        if len(entities) > 3:
            complexity_score += 2
        elif len(entities) > 1:
            complexity_score += 1
        
        # Time range adds complexity
        if time_range:
            complexity_score += 1
        
        # Comparison/relationship words
        comparison_words = ['compare', 'difference', 'versus', 'vs', 'relationship', 'connection', 'between']
        if any(word in text.lower() for word in comparison_words):
            complexity_score += 2
        
        # Multi-part questions
        if '?' in text[:-1]:  # Question mark not at the end
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score <= 2:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 5:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.COMPLEX
    
    def expand_query(self, normalized_query: NormalizedQuery) -> List[str]:
        """
        Generate alternative phrasings for query expansion.
        
        Args:
            normalized_query: Normalized query
            
        Returns:
            List of expanded query strings
        """
        expanded_queries = [normalized_query.normalized_text]
        
        # Add version without question words
        query_lower = normalized_query.normalized_text.lower()
        for qw in ['what is', 'who is', 'where is', 'when did', 'why did', 'how does']:
            if query_lower.startswith(qw):
                expanded_queries.append(
                    normalized_query.normalized_text[len(qw):].strip()
                )
                break
        
        # Add version with entities only (if multiple entities)
        if len(normalized_query.entities_mentioned) >= 2:
            entities_query = ' '.join(normalized_query.entities_mentioned)
            if entities_query not in expanded_queries:
                expanded_queries.append(entities_query)
        
        # Add domain-specific version if domain detected
        if normalized_query.domain:
            domain_query = f"{normalized_query.domain} {normalized_query.normalized_text}"
            expanded_queries.append(domain_query)
        
        # Limit to 4 queries max
        return expanded_queries[:4]