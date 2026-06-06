"""
API schemas for the research analyst system.
Pydantic models for request/response validation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


# ============================================================================
# Request Schemas
# ============================================================================

class QueryRequest(BaseModel):
    """Request schema for query endpoint."""
    
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The research query",
        example="What is the relationship between OpenAI and Microsoft?"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier",
        example="user_123"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Optional session identifier",
        example="session_456"
    )
    
    use_cache: bool = Field(
        True,
        description="Whether to use cached results if available"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the relationship between OpenAI and Microsoft?",
                "user_id": "user_123",
                "use_cache": True
            }
        }


# ============================================================================
# Response Schemas
# ============================================================================

class CitationResponse(BaseModel):
    """Citation in answer response."""
    
    source_url: HttpUrl = Field(..., description="Source URL")
    source_title: str = Field(..., description="Source title")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class AnswerResponse(BaseModel):
    """Answer in query response."""
    
    answer_id: str = Field(..., description="Unique answer identifier")
    text: str = Field(..., description="Answer text with inline citations")
    citations: List[CitationResponse] = Field(default_factory=list, description="List of citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    execution_path: str = Field(..., description="Execution path used (fast or research)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer_id": "answer_abc123",
                "text": "OpenAI and Microsoft have a strategic partnership [1]. Microsoft invested in OpenAI...",
                "citations": [
                    {
                        "source_url": "https://example.com/article",
                        "source_title": "Microsoft and OpenAI Partnership",
                        "relevance": 0.95
                    }
                ],
                "confidence": 0.85,
                "execution_path": "research"
            }
        }


class QualityMetricsResponse(BaseModel):
    """Quality metrics in response."""
    
    citation_coverage: float = Field(..., ge=0.0, le=1.0)
    grounding_score: float = Field(..., ge=0.0, le=1.0)
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    answer_completeness: float = Field(..., ge=0.0, le=1.0)
    source_diversity: float = Field(..., ge=0.0, le=1.0)
    passes_threshold: bool
    issues: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Main response schema for query endpoint."""
    
    success: bool = Field(..., description="Whether query was successful")
    answer: Optional[AnswerResponse] = Field(None, description="Generated answer")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_path: Optional[str] = Field(None, description="Execution path used")
    quality_metrics: Optional[QualityMetricsResponse] = Field(None, description="Quality metrics")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    cost_estimate: float = Field(..., description="Estimated cost in USD")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "answer": {
                    "answer_id": "answer_abc123",
                    "text": "OpenAI and Microsoft have a strategic partnership...",
                    "citations": [],
                    "confidence": 0.85,
                    "execution_path": "research"
                },
                "execution_path": "research",
                "quality_metrics": {
                    "citation_coverage": 0.8,
                    "grounding_score": 0.85,
                    "coherence_score": 0.9,
                    "answer_completeness": 0.85,
                    "source_diversity": 0.7,
                    "passes_threshold": True,
                    "issues": []
                },
                "execution_time_ms": 12500.0,
                "cost_estimate": 0.15,
                "metadata": {}
            }
        }


# ============================================================================
# Health & Stats Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="0.1.0", description="System version")
    components: Dict[str, str] = Field(default_factory=dict, description="Component statuses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-22T12:00:00Z",
                "version": "0.1.0",
                "components": {
                    "cache": "healthy",
                    "graph_store": "healthy",
                    "llm": "healthy"
                }
            }
        }


class StatsResponse(BaseModel):
    """System statistics response."""
    
    cache_stats: Dict[str, Any] = Field(default_factory=dict)
    graph_stats: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "cache_stats": {
                    "backend": "disk",
                    "keys": 42,
                    "size": 1048576
                },
                "graph_stats": {
                    "num_nodes": 150,
                    "num_edges": 300,
                    "is_connected": False
                }
            }
        }


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response schema."""
    
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Query is too short (minimum 3 characters)",
                "error_type": "ValidationError",
                "details": {"query_length": 2},
                "timestamp": "2024-01-22T12:00:00Z"
            }
        }