"""
Core module for research analyst system.
Provides base models and exceptions.
"""

from .models import (
    # Enums
    QueryIntent,
    QueryComplexity,
    ExecutionPath,
    SourceType,
    EntityType,
    
    # Query models
    Query,
    NormalizedQuery,
    RoutingDecision,
    
    # Document models
    Document,
    DocumentChunk,
    RankedDocument,
    
    # Graph models
    Entity,
    Relationship,
    KnowledgeGraph,
    Subgraph,
    
    # Evidence models
    Claim,
    Evidence,
    Citation,
    Answer,
    
    # Evaluation models
    QualityMetrics,
    SelfHealingAction,
    
    # Pipeline models
    PipelineState,
    PipelineResponse,
    ErrorDetails,
)

from .exceptions import (
    # Base
    ResearchAnalystException,
    
    # Input/Validation
    ValidationError,
    GuardrailViolation,
    PromptInjectionDetected,
    UnsafeContentDetected,
    OutOfScopeQuery,
    
    # Processing
    QueryProcessingError,
    IntentClassificationError,
    QueryNormalizationError,
    
    # Retrieval
    RetrievalError,
    SearchEngineError,
    DocumentFetchError,
    EmbeddingError,
    RankingError,
    NoResultsError,
    
    # Graph
    GraphError,
    EntityExtractionError,
    RelationshipExtractionError,
    GraphConstructionError,
    GraphQueryError,
    SubgraphExtractionError,
    
    # Synthesis
    SynthesisError,
    ContextBuildingError,
    AnswerGenerationError,
    LLMError,
    
    # Quality
    EvaluationError,
    InsufficientQuality,
    CitationCoverageError,
    GroundingError,
    ConfidenceThresholdError,
    
    # Self-healing
    SelfHealingError,
    MaxRetriesExceeded,
    
    # Infrastructure
    InfrastructureError,
    CacheError,
    DatabaseError,
    ConfigurationError,
    RateLimitError,
    TimeoutError,
    
    # Utilities
    get_error_details,
    is_recoverable,
)

__all__ = [
    # Enums
    "QueryIntent",
    "QueryComplexity",
    "ExecutionPath",
    "SourceType",
    "EntityType",
    
    # Models
    "Query",
    "NormalizedQuery",
    "RoutingDecision",
    "Document",
    "DocumentChunk",
    "RankedDocument",
    "Entity",
    "Relationship",
    "KnowledgeGraph",
    "Subgraph",
    "Claim",
    "Evidence",
    "Citation",
    "Answer",
    "QualityMetrics",
    "SelfHealingAction",
    "PipelineState",
    "PipelineResponse",
    "ErrorDetails",
    
    # Exceptions
    "ResearchAnalystException",
    "ValidationError",
    "GuardrailViolation",
    "PromptInjectionDetected",
    "UnsafeContentDetected",
    "OutOfScopeQuery",
    "QueryProcessingError",
    "IntentClassificationError",
    "QueryNormalizationError",
    "RetrievalError",
    "SearchEngineError",
    "DocumentFetchError",
    "EmbeddingError",
    "RankingError",
    "NoResultsError",
    "GraphError",
    "EntityExtractionError",
    "RelationshipExtractionError",
    "GraphConstructionError",
    "GraphQueryError",
    "SubgraphExtractionError",
    "SynthesisError",
    "ContextBuildingError",
    "AnswerGenerationError",
    "LLMError",
    "EvaluationError",
    "InsufficientQuality",
    "CitationCoverageError",
    "GroundingError",
    "ConfidenceThresholdError",
    "SelfHealingError",
    "MaxRetriesExceeded",
    "InfrastructureError",
    "CacheError",
    "DatabaseError",
    "ConfigurationError",
    "RateLimitError",
    "TimeoutError",
    "get_error_details",
    "is_recoverable",
]
