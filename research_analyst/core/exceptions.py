"""
Custom exceptions for the research analyst system.
Provides a clear exception hierarchy for different failure modes.
"""

from typing import Optional, Dict


# ============================================================================
# Base Exception
# ============================================================================

class ResearchAnalystException(Exception):
    """Base exception for all research analyst errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict] = None,
        recoverable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable


# ============================================================================
# Input & Validation Errors
# ============================================================================

class ValidationError(ResearchAnalystException):
    """Raised when input validation fails."""
    pass


class GuardrailViolation(ResearchAnalystException):
    """Raised when a guardrail check fails."""

    def __init__(
        self,
        message: str,
        violation_type: str,
        details: Optional[Dict] = None,
    ):
        self.violation_type = violation_type
        super().__init__(message, details, recoverable=False)


class PromptInjectionDetected(GuardrailViolation):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "prompt_injection", details)


class UnsafeContentDetected(GuardrailViolation):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "unsafe_content", details)


class OutOfScopeQuery(GuardrailViolation):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "out_of_scope", details)


# ============================================================================
# Processing Errors
# ============================================================================

class QueryProcessingError(ResearchAnalystException):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details, recoverable=True)


class IntentClassificationError(QueryProcessingError):
    pass


class QueryNormalizationError(QueryProcessingError):
    pass


# ============================================================================
# Retrieval Errors
# ============================================================================

class RetrievalError(ResearchAnalystException):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details, recoverable=True)


class SearchEngineError(RetrievalError):
    pass


class DocumentFetchError(RetrievalError):
    pass


class EmbeddingError(RetrievalError):
    pass


class RankingError(RetrievalError):
    pass


class NoResultsError(RetrievalError):
    def __init__(
        self,
        message: str = "No results found for query",
        details: Optional[Dict] = None,
    ):
        super().__init__(message, details)


# ============================================================================
# Graph Errors
# ============================================================================

class GraphError(ResearchAnalystException):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details, recoverable=True)


class EntityExtractionError(GraphError):
    pass


class RelationshipExtractionError(GraphError):
    pass


class GraphConstructionError(GraphError):
    pass


class GraphQueryError(GraphError):
    pass


class SubgraphExtractionError(GraphError):
    pass


# ============================================================================
# Synthesis Errors (FIXED + UNIFIED)
# ============================================================================

class SynthesisError(ResearchAnalystException):
    def __init__(
        self,
        message: str,
        *,
        recoverable: bool = False,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, details, recoverable)


class ContextBuildingError(SynthesisError):
    pass


class AnswerGenerationError(SynthesisError):
    pass


class LLMError(SynthesisError):
    def __init__(
        self,
        message: str,
        provider: str,
        details: Optional[Dict] = None,
    ):
        self.provider = provider
        super().__init__(
            message,
            recoverable=True,
            details=details,
        )


# ============================================================================
# Quality & Evaluation Errors
# ============================================================================

class EvaluationError(ResearchAnalystException):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details, recoverable=False)


class InsufficientQuality(EvaluationError):
    def __init__(
        self,
        message: str,
        metrics: Dict,
        details: Optional[Dict] = None,
    ):
        self.metrics = metrics
        super().__init__(message, details)


class CitationCoverageError(InsufficientQuality):
    pass


class GroundingError(InsufficientQuality):
    pass


class ConfidenceThresholdError(InsufficientQuality):
    pass


# ============================================================================
# Self-Healing Errors
# ============================================================================

class SelfHealingError(ResearchAnalystException):
    def __init__(
        self,
        message: str,
        attempts: int,
        details: Optional[Dict] = None,
    ):
        self.attempts = attempts
        super().__init__(message, details, recoverable=False)


class MaxRetriesExceeded(SelfHealingError):
    def __init__(self, attempts: int, details: Optional[Dict] = None):
        super().__init__(
            f"Maximum self-healing attempts ({attempts}) exceeded",
            attempts,
            details,
        )


# ============================================================================
# Infrastructure Errors
# ============================================================================

class InfrastructureError(ResearchAnalystException):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details, recoverable=False)


class CacheError(InfrastructureError):
    pass


class DatabaseError(InfrastructureError):
    pass


class ConfigurationError(InfrastructureError):
    pass


class RateLimitError(InfrastructureError):
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        details: Optional[Dict] = None,
    ):
        self.retry_after = retry_after
        super().__init__(message, details)


class TimeoutError(InfrastructureError):
    def __init__(
        self,
        message: str,
        timeout_seconds: int,
        details: Optional[Dict] = None,
    ):
        self.timeout_seconds = timeout_seconds
        super().__init__(message, details)


# ============================================================================
# Utility Functions
# ============================================================================

def get_error_details(exception: Exception) -> Dict:
    if isinstance(exception, ResearchAnalystException):
        return {
            "type": exception.__class__.__name__,
            "message": exception.message,
            "details": exception.details,
            "recoverable": exception.recoverable,
        }
    return {
        "type": exception.__class__.__name__,
        "message": str(exception),
        "details": {},
        "recoverable": False,
    }


def is_recoverable(exception: Exception) -> bool:
    return isinstance(exception, ResearchAnalystException) and exception.recoverable
