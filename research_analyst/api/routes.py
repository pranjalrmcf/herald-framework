"""
API routes for the research analyst system.
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from research_analyst.api.schemas import (
    QueryRequest,
    QueryResponse,
    AnswerResponse,
    CitationResponse,
    QualityMetricsResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse
)
from research_analyst.orchestration import ResearchAnalyst
from research_analyst.core.exceptions import (
    ResearchAnalystException,
    GuardrailViolation,
    get_error_details
)
from research_analyst.utils.logger import get_logger


# Initialize router
router = APIRouter()

# Initialize research analyst (singleton)
research_analyst = ResearchAnalyst()

# Logger
logger = get_logger()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process a research query",
    description="Submit a research query and receive a comprehensive answer with citations",
    responses={
        200: {"description": "Successful response with answer"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def process_query(request: QueryRequest):
    """
    Process a research query.
    
    Args:
        request: Query request with query text and optional parameters
        
    Returns:
        Query response with answer and metadata
    """
    logger.info(
        "API query received",
        query=request.query[:100],
        user_id=request.user_id
    )
    
    try:
        # Process query
        pipeline_response = research_analyst.query(
            query_text=request.query,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        # Convert to API response
        if pipeline_response.success and pipeline_response.answer:
            # Convert answer
            answer_response = AnswerResponse(
                answer_id=pipeline_response.answer.answer_id,
                text=pipeline_response.answer.text,
                citations=[
                    CitationResponse(
                        source_url=c.source_url,
                        source_title=c.source_title,
                        excerpt=c.excerpt,
                        relevance=c.relevance
                    )
                    for c in pipeline_response.answer.citations
                ],
                confidence=pipeline_response.answer.confidence,
                execution_path=pipeline_response.answer.execution_path.value
            )
            
            # Convert quality metrics
            quality_response = None
            if pipeline_response.quality_metrics:
                quality_response = QualityMetricsResponse(
                    citation_coverage=pipeline_response.quality_metrics.citation_coverage,
                    grounding_score=pipeline_response.quality_metrics.grounding_score,
                    coherence_score=pipeline_response.quality_metrics.coherence_score,
                    answer_completeness=pipeline_response.quality_metrics.answer_completeness,
                    source_diversity=pipeline_response.quality_metrics.source_diversity,
                    passes_threshold=pipeline_response.quality_metrics.passes_threshold,
                    issues=pipeline_response.quality_metrics.issues
                )
            
            response = QueryResponse(
                success=True,
                answer=answer_response,
                execution_path=pipeline_response.execution_path.value if pipeline_response.execution_path else None,
                quality_metrics=quality_response,
                execution_time_ms=pipeline_response.execution_time_ms,
                cost_estimate=pipeline_response.cost_estimate,
                metadata=pipeline_response.metadata
            )
        else:
            # Error response
            response = QueryResponse(
                success=False,
                error=pipeline_response.error,
                execution_time_ms=pipeline_response.execution_time_ms,
                cost_estimate=pipeline_response.cost_estimate,
                metadata=pipeline_response.metadata
            )
        
        logger.info(
            "API query completed",
            success=response.success,
            execution_time_ms=response.execution_time_ms
        )
        
        return response
        
    except GuardrailViolation as e:
        logger.warning(
            "Guardrail violation",
            violation_type=e.violation_type,
            message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(e),
                "error_type": "GuardrailViolation",
                "details": e.details
            }
        )
    
    except ResearchAnalystException as e:
        logger.error(
            "API query failed",
            error=str(e)
        )
        
        error_details = get_error_details(e)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "error_type": error_details['type'],
                "details": error_details['details']
            }
        )
    
    except Exception as e:
        logger.error(
            "Unexpected error",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "An unexpected error occurred",
                "error_type": "InternalError"
            }
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check system health and component status"
)
async def health_check():
    """
    Check system health.
    
    Returns:
        Health status
    """
    components = {
        "api": "healthy",
        "orchestrator": "healthy"
    }
    
    # Check cache
    try:
        cache_stats = research_analyst.cache_manager.get_cache_stats()
        components["cache"] = "healthy"
    except Exception as e:
        logger.warning("Cache health check failed", error=str(e))
        components["cache"] = "unhealthy"
    
    # Check graph store
    try:
        graph_stats = research_analyst.graph_store.get_graph_stats()
        components["graph_store"] = "healthy"
    except Exception as e:
        logger.warning("Graph store health check failed", error=str(e))
        components["graph_store"] = "unhealthy"
    
    # Overall status
    overall_status = "healthy" if all(
        status == "healthy" for status in components.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        components=components
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="System statistics",
    description="Get system statistics including cache and graph metrics"
)
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        System statistics
    """
    try:
        stats = research_analyst.get_stats()
        
        return StatsResponse(
            cache_stats=stats.get("cache", {}),
            graph_stats=stats.get("graph", {})
        )
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to retrieve statistics"}
        )


@router.post(
    "/clear-cache",
    status_code=status.HTTP_200_OK,
    summary="Clear cache",
    description="Clear all cached data (requires admin access)"
)
async def clear_cache():
    """
    Clear all caches.
    
    Returns:
        Success message
    """
    try:
        research_analyst.clear_caches()
        logger.info("Cache cleared via API")
        return {"success": True, "message": "All caches cleared"}
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to clear cache"}
        )