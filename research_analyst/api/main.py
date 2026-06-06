"""
FastAPI application for the research analyst system.
Main entry point for the REST API.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time

from research_analyst.api.routes import router
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger


# Get settings and logger
settings = get_settings()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the application.
    Handles startup and shutdown.
    """
    # Startup
    logger.info("Starting Research Analyst API")
    logger.info(f"Environment: {'Development' if settings.debug_mode else 'Production'}")
    logger.info(f"LLM Provider: {settings.default_llm_provider}")
    logger.info(f"Cache Backend: {settings.cache_backend}")
    logger.info(f"Graph Backend: {settings.graph_backend}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Research Analyst API")


# Create FastAPI app
app = FastAPI(
    title="Research Analyst API",
    description="""
    **Autonomous Research Analyst System**
    
    A production-grade research analyst using agent-controlled RAG architecture with graph reasoning.
    
    ## Features
    
    * **Intelligent Routing**: Automatically chooses between fast vector search and comprehensive graph-based research
    * **Graph RAG**: Extracts entities and relationships for multi-hop reasoning
    * **Quality Evaluation**: Automatic quality metrics with self-healing capabilities
    * **Multi-layer Caching**: Optimized for cost and latency
    * **Citation-rich Answers**: All answers include proper source attribution
    
    ## Execution Paths
    
    * **Fast Path**: Quick vector-based retrieval for simple queries
    * **Research Path**: Deep analysis with knowledge graph construction for complex queries
    
    ## Usage
    
    Submit a research query to `/query` endpoint and receive:
    - Comprehensive answer with inline citations
    - Quality metrics
    - Source links and excerpts
    - Execution metadata
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # ms
    response.headers["X-Process-Time-Ms"] = str(process_time)
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    logger.info(
        "API request",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None
    )
    
    response = await call_next(request)
    
    logger.info(
        "API response",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code
    )
    
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(
        "Validation error",
        errors=exc.errors()
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "error_type": "ValidationError",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(
        "Unexpected error",
        error=str(exc),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred",
            "error_type": "InternalError"
        }
    )


# Include routers
app.include_router(router, prefix="/api/v1", tags=["Research Analyst"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Research Analyst API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Run with: uvicorn codebase.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "research_analyst.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug_mode,
        log_level=settings.log_level.lower()
    )