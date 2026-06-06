"""
Ranker for the research analyst system.
Ranks documents by relevance, credibility, and recency.
"""

from typing import List, Dict
from datetime import datetime, timedelta
import math

from research_analyst.core.models import Document, RankedDocument, DocumentChunk
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger


logger = get_logger()


class Ranker:
    """Rank documents using multiple factors."""
    
    def __init__(self):
        """Initialize ranker."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Ranking weights (should sum to 1.0)
        self.weights = {
            'relevance': 0.50,      # Semantic similarity
            'credibility': 0.30,    # Source trustworthiness
            'recency': 0.20,        # How recent the content is
        }
    
    def rank_documents(
        self,
        documents: List[Document],
        query: str,
        relevance_scores: Dict[str, float] = None
    ) -> List[RankedDocument]:
        """
        Rank documents by multiple factors.
        
        Args:
            documents: List of documents to rank
            query: Original query for relevance calculation
            relevance_scores: Pre-computed relevance scores (doc_id -> score)
            
        Returns:
            List of RankedDocument objects sorted by score
        """
        self.logger.info(
            "Ranking documents",
            num_documents=len(documents),
            query_length=len(query)
        )
        
        if not documents:
            return []
        
        ranked_documents = []
        
        for doc in documents:
            # Calculate individual scores
            relevance = self._score_relevance(doc, query, relevance_scores)
            credibility = doc.credibility_score  # Already set
            recency = self._score_recency(doc)
            
            # Calculate final weighted score
            final_score = (
                relevance * self.weights['relevance'] +
                credibility * self.weights['credibility'] +
                recency * self.weights['recency']
            )
            
            # Create RankedDocument
            ranked_doc = RankedDocument(
                document=doc,
                relevance_score=relevance,
                credibility_score=credibility,
                recency_score=recency,
                final_score=final_score,
                ranking_factors={
                    'relevance': relevance,
                    'credibility': credibility,
                    'recency': recency
                }
            )
            
            ranked_documents.append(ranked_doc)
        
        # Sort by final score
        ranked_documents.sort(key=lambda x: x.final_score, reverse=True)
        
        self.logger.info(
            "Documents ranked",
            num_ranked=len(ranked_documents),
            top_score=ranked_documents[0].final_score if ranked_documents else 0
        )
        
        return ranked_documents
    
    def _score_relevance(
        self,
        document: Document,
        query: str,
        relevance_scores: Dict[str, float] = None
    ) -> float:
        """
        Score document relevance to query.
        
        Args:
            document: Document to score
            query: Query text
            relevance_scores: Pre-computed scores
            
        Returns:
            Relevance score 0.0-1.0
        """
        # Use pre-computed score if available
        if relevance_scores and document.doc_id in relevance_scores:
            return relevance_scores[document.doc_id]
        
        # Otherwise, use simple keyword matching
        return self._keyword_relevance(document, query)
    
    def _keyword_relevance(self, document: Document, query: str) -> float:
        """
        Calculate relevance based on keyword matching.
        
        Args:
            document: Document
            query: Query text
            
        Returns:
            Relevance score 0.0-1.0
        """
        # Extract keywords from query
        query_words = set(query.lower().split())
        
        # Check title
        title_words = set(document.title.lower().split()) if document.title else set()
        title_matches = len(query_words & title_words)
        
        # Check snippet/content
        content = document.snippet or document.content or ""
        content_words = set(content.lower().split())
        content_matches = len(query_words & content_words)
        
        # Calculate score
        total_query_words = len(query_words)
        if total_query_words == 0:
            return 0.5
        
        # Title matches are more important
        score = (title_matches * 2 + content_matches) / (total_query_words * 3)
        
        # Normalize to 0-1
        return min(1.0, score)
    
    def _score_recency(self, document: Document) -> float:
        """
        Score document recency.
        
        Args:
            document: Document
            
        Returns:
            Recency score 0.0-1.0
        """
        # If no published date, use retrieval time
        if document.published_date:
            doc_date = document.published_date
        else:
            doc_date = document.retrieved_at
        
        # Calculate age in days
        now = datetime.utcnow()
        age_days = (now - doc_date).days
        
        # Decay function: newer is better
        # Score decays over 365 days
        if age_days < 0:
            # Future date (error), treat as very recent
            return 1.0
        elif age_days == 0:
            return 1.0
        elif age_days <= 7:
            return 0.95
        elif age_days <= 30:
            return 0.85
        elif age_days <= 90:
            return 0.70
        elif age_days <= 180:
            return 0.50
        elif age_days <= 365:
            return 0.30
        else:
            # Older than a year
            return 0.10
    
    def rank_chunks(
        self,
        chunks: List[tuple[DocumentChunk, float]],
        diversify: bool = True
    ) -> List[tuple[DocumentChunk, float]]:
        """
        Rank chunks from vector search results.
        
        Args:
            chunks: List of (chunk, similarity_score) tuples
            diversify: Whether to diversify results by source
            
        Returns:
            Ranked list of (chunk, score) tuples
        """
        if not chunks:
            return []
        
        # Sort by similarity score
        ranked_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
        
        # Optionally diversify
        if diversify:
            ranked_chunks = self._diversify_chunks(ranked_chunks)
        
        return ranked_chunks
    
    def _diversify_chunks(
        self,
        chunks: List[tuple[DocumentChunk, float]],
        max_per_doc: int = 3
    ) -> List[tuple[DocumentChunk, float]]:
        """
        Diversify chunks to avoid over-representation of single documents.
        
        Args:
            chunks: List of (chunk, score) tuples
            max_per_doc: Maximum chunks per document
            
        Returns:
            Diversified list
        """
        doc_counts = {}
        diversified = []
        
        for chunk, score in chunks:
            doc_id = chunk.doc_id
            
            # Count chunks from this document
            count = doc_counts.get(doc_id, 0)
            
            if count < max_per_doc:
                diversified.append((chunk, score))
                doc_counts[doc_id] = count + 1
        
        return diversified
    
    def rerank_with_custom_weights(
        self,
        ranked_documents: List[RankedDocument],
        custom_weights: Dict[str, float]
    ) -> List[RankedDocument]:
        """
        Rerank documents with custom weights.
        
        Args:
            ranked_documents: Previously ranked documents
            custom_weights: Custom weight dictionary
            
        Returns:
            Reranked documents
        """
        # Validate weights sum to 1.0
        total = sum(custom_weights.values())
        if abs(total - 1.0) > 0.01:
            self.logger.warning(
                "Custom weights don't sum to 1.0, normalizing",
                total=total
            )
            custom_weights = {k: v/total for k, v in custom_weights.items()}
        
        # Recalculate scores
        for ranked_doc in ranked_documents:
            new_score = (
                ranked_doc.relevance_score * custom_weights.get('relevance', 0) +
                ranked_doc.credibility_score * custom_weights.get('credibility', 0) +
                ranked_doc.recency_score * custom_weights.get('recency', 0)
            )
            ranked_doc.final_score = new_score
            # ranked_doc.ranking_factors['weights'] = custom_weights.copy()
        
        # Re-sort
        ranked_documents.sort(key=lambda x: x.final_score, reverse=True)
        
        return ranked_documents
    
    def filter_by_score_threshold(
        self,
        ranked_documents: List[RankedDocument],
        min_score: float = 0.3
    ) -> List[RankedDocument]:
        """
        Filter documents below score threshold.
        
        Args:
            ranked_documents: Ranked documents
            min_score: Minimum score threshold
            
        Returns:
            Filtered documents
        """
        filtered = [doc for doc in ranked_documents if doc.final_score >= min_score]
        
        self.logger.info(
            "Filtered by score threshold",
            original_count=len(ranked_documents),
            filtered_count=len(filtered),
            threshold=min_score
        )
        
        return filtered
    
    def explain_ranking(self, ranked_doc: RankedDocument) -> str:
        """
        Generate explanation for document ranking.
        
        Args:
            ranked_doc: Ranked document
            
        Returns:
            Explanation string
        """
        explanation = f"""
Document: {ranked_doc.document.title}
Final Score: {ranked_doc.final_score:.3f}

Factor Breakdown:
- Relevance: {ranked_doc.relevance_score:.3f} (weight: {self.weights['relevance']:.2f})
- Credibility: {ranked_doc.credibility_score:.3f} (weight: {self.weights['credibility']:.2f})
- Recency: {ranked_doc.recency_score:.3f} (weight: {self.weights['recency']:.2f})

Source: {ranked_doc.document.url}
Source Type: {ranked_doc.document.source_type.value}
        """.strip()
        
        return explanation