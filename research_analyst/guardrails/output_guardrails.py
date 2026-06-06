"""
Output guardrails for the research analyst system.
Validates generated answers before returning to user.
"""

import re
from typing import List, Optional, Tuple

from research_analyst.core.models import Answer, Citation, QualityMetrics
from research_analyst.core.exceptions import (
    CitationCoverageError,
    ConfidenceThresholdError,
    GroundingError,
)
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import estimate_token_count


logger = get_logger()


class OutputGuardrails:
    """Output validation and quality checks."""
    
    def __init__(self):
        """Initialize output guardrails."""
        self.settings = get_settings()
        self.logger = get_logger()
    
    def check(self, answer: Answer, quality_metrics: QualityMetrics) -> Tuple[bool, Optional[str]]:
        """
        Run all output guardrail checks on an answer.
        
        Args:
            answer: Generated answer
            quality_metrics: Quality metrics for the answer
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Raises:
            CitationCoverageError: If citation coverage is insufficient
            ConfidenceThresholdError: If confidence is too low
            GroundingError: If answer is poorly grounded
        """
        answer_id = answer.answer_id
        
        self.logger.info(
            "Running output guardrails",
            answer_id=answer_id,
            answer_length=len(answer.text)
        )
        
        try:
            # 1. Check citation coverage
            self._check_citation_coverage(answer, quality_metrics, answer_id)
            
            # 2. Check confidence threshold
            self._check_confidence_threshold(answer, answer_id)
            
            # 3. Check grounding
            self._check_grounding(answer, quality_metrics, answer_id)
            
            # 4. Check for unsafe content in answer
            self._check_answer_safety(answer, answer_id)
            
            # 5. Check answer completeness
            self._check_completeness(answer, answer_id)
            
            self.logger.info(
                "Output guardrails passed",
                answer_id=answer_id
            )
            
            return True, None
            
        except (CitationCoverageError, ConfidenceThresholdError, GroundingError) as e:
            self.logger.warning(
                "Output guardrail check failed",
                answer_id=answer_id,
                error=str(e),
                recoverable=e.recoverable
            )
            raise
    
    def _check_citation_coverage(
        self,
        answer: Answer,
        quality_metrics: QualityMetrics,
        answer_id: str
    ):
        """
        Check if answer has adequate citations.
        
        Args:
            answer: Generated answer
            quality_metrics: Quality metrics
            answer_id: Answer identifier
            
        Raises:
            CitationCoverageError: If coverage is insufficient
        """
        min_coverage = (
            0.3 if self.settings.dev_mode else self.settings.min_citation_coverage
        )

        if quality_metrics.citation_coverage < min_coverage:

        # if quality_metrics.citation_coverage < self.settings.min_citation_coverage:
            raise CitationCoverageError(
                f"Citation coverage {quality_metrics.citation_coverage:.2f} below threshold "
                f"{self.settings.min_citation_coverage:.2f}",
                metrics={"citation_coverage": quality_metrics.citation_coverage},
                details={
                    "answer_id": answer_id,
                    "num_citations": len(answer.citations),
                    "answer_length": len(answer.text)
                }
            )
        
        # Check if answer has any citations at all
        if len(answer.citations) == 0:
            raise CitationCoverageError(
                "Answer has no citations",
                metrics={"citation_coverage": 0.0},
                details={"answer_id": answer_id}
            )
        
        # Count citation markers in text
        citation_markers = re.findall(r'\[\d+\]', answer.text)
        if len(citation_markers) == 0:
            raise CitationCoverageError(
                "No citation markers found in answer text",
                metrics={"citation_coverage": 0.0},
                details={
                    "answer_id": answer_id,
                    "citations_provided": len(answer.citations)
                }
            )
    
    def _check_confidence_threshold(self, answer: Answer, answer_id: str):
        """
        Check if answer confidence meets threshold.
        
        Args:
            answer: Generated answer
            answer_id: Answer identifier
            
        Raises:
            ConfidenceThresholdError: If confidence is too low
        """
        if answer.confidence < self.settings.min_confidence_threshold:
            raise ConfidenceThresholdError(
                f"Answer confidence {answer.confidence:.2f} below threshold "
                f"{self.settings.min_confidence_threshold:.2f}",
                metrics={"confidence": answer.confidence},
                details={
                    "answer_id": answer_id,
                    "suggestion": "Consider re-retrieval or query expansion"
                }
            )
    
    def _check_grounding(
        self,
        answer: Answer,
        quality_metrics: QualityMetrics,
        answer_id: str
    ):
        """
        Check if answer is well-grounded in evidence.
        
        Args:
            answer: Generated answer
            quality_metrics: Quality metrics
            answer_id: Answer identifier
            
        Raises:
            GroundingError: If grounding is insufficient
        """
        # Check grounding score (from quality metrics)
        if quality_metrics.grounding_score < 0.6:  # Threshold for grounding
            raise GroundingError(
                f"Answer grounding score {quality_metrics.grounding_score:.2f} is too low",
                metrics={"grounding_score": quality_metrics.grounding_score},
                details={
                    "answer_id": answer_id,
                    "issues": quality_metrics.issues
                }
            )
        
        # Check for hedging language that might indicate uncertainty
        hedging_phrases = [
            "i don't know",
            "i'm not sure",
            "i cannot find",
            "no information available",
            "unable to determine"
        ]
        
        answer_lower = answer.text.lower()
        hedges_found = [phrase for phrase in hedging_phrases if phrase in answer_lower]
        
        if len(hedges_found) > 2:  # Too much uncertainty
            self.logger.warning(
                "Answer contains multiple hedging phrases",
                answer_id=answer_id,
                hedges=hedges_found
            )
    
    def _check_answer_safety(self, answer: Answer, answer_id: str):
        """
        Check for unsafe content in the generated answer.
        
        Args:
            answer: Generated answer
            answer_id: Answer identifier
        """
        # Basic check for leaked system information
        unsafe_patterns = [
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"system:",
            r"assistant:",
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, answer.text, re.IGNORECASE):
                self.logger.warning(
                    "Potential unsafe content in answer",
                    answer_id=answer_id,
                    pattern=pattern
                )
        
        # Check for PII leakage
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, answer.text):
                self.logger.error(
                    "Potential PII in answer - BLOCKING",
                    answer_id=answer_id,
                    pattern=pattern
                )
                raise GroundingError(
                    "Answer contains potentially sensitive information",
                    metrics={},
                    details={"answer_id": answer_id, "issue": "PII detected"}
                )
    
    def _check_completeness(self, answer: Answer, answer_id: str):
        """
        Check if answer is reasonably complete.
        
        Args:
            answer: Generated answer
            answer_id: Answer identifier
        """
        # Check minimum length
        if len(answer.text.strip()) < 50:
            self.logger.warning(
                "Answer is very short",
                answer_id=answer_id,
                length=len(answer.text)
            )
        
        # Check if answer is cut off
        if answer.text.endswith(("...", "…")) and len(answer.text) > 500:
            self.logger.warning(
                "Answer may be truncated",
                answer_id=answer_id
            )
    
    def validate_citations(self, citations: List[Citation]) -> List[Citation]:
        """
        Validate and clean citation list.
        
        Args:
            citations: List of citations
            
        Returns:
            Validated citations
        """
        valid_citations = []
        
        for i, citation in enumerate(citations):
            # Check if URL is valid
            if not str(citation.source_url).startswith(('http://', 'https://')):
                self.logger.warning(
                    "Invalid citation URL",
                    citation_index=i,
                    url=str(citation.source_url)
                )
                continue
            
            # Check if title exists
            if not citation.source_title or len(citation.source_title.strip()) == 0:
                self.logger.warning(
                    "Citation missing title",
                    citation_index=i
                )
                # Use URL as title
                citation.source_title = str(citation.source_url)
            
            valid_citations.append(citation)
        
        return valid_citations
    
    def format_answer_for_display(self, answer: Answer) -> str:
        """
        Format answer text for user display.
        
        Args:
            answer: Answer object
            
        Returns:
            Formatted answer text
        """
        formatted_text = answer.text
        
        # Ensure citation markers are properly formatted
        formatted_text = re.sub(r'\[(\d+)\]', r'[\1]', formatted_text)
        
        # Add sources section if needed
        if answer.citations:
            formatted_text += "\n\n**Sources:**\n"
            for i, citation in enumerate(answer.citations, 1):
                formatted_text += f"{i}. {citation.source_title}\n   {citation.source_url}\n"
        
        return formatted_text
    
    def validate_and_format(
        self,
        answer: Answer,
        quality_metrics: QualityMetrics
    ) -> Tuple[Answer, str]:
        """
        Run full validation and formatting pipeline.
        
        Args:
            answer: Generated answer
            quality_metrics: Quality metrics
            
        Returns:
            Tuple of (validated answer, formatted text)
            
        Raises:
            CitationCoverageError, ConfidenceThresholdError, GroundingError
        """
        # Skip if guardrails disabled
        if not self.settings.enable_output_guardrails:
            self.logger.warning("Output guardrails disabled - skipping validation")
            return answer, answer.text
        
        # Run all checks
        self.check(answer, quality_metrics)
        
        # Validate citations
        validated_citations = self.validate_citations(answer.citations)
        answer.citations = validated_citations
        
        # Format for display
        formatted_text = self.format_answer_for_display(answer)
        
        return answer, formatted_text