"""
Structured logging utility using Loguru.
Provides consistent logging across the application.
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict
from loguru import logger
from datetime import datetime


class StructuredLogger:
    """Wrapper around loguru for structured logging."""
    
    def __init__(self, log_file: str = "./logs/research_analyst.log", 
                 log_level: str = "INFO",
                 rotation: str = "100 MB"):
        """
        Initialize structured logger.
        
        Args:
            log_file: Path to log file
            log_level: Logging level
            rotation: Log rotation size
        """
        self.log_file = log_file
        self.log_level = log_level
        self.rotation = rotation
        
        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Add console handler with colors
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True,
        )
        
        # Add file handler with JSON formatting
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention="30 days",
            compression="zip",
            serialize=False,  # Keep human-readable
        )
        
        self.logger = logger
    
    def _add_context(self, message: str, extra: Dict[str, Any] = None) -> str:
        """Add structured context to log message."""
        if extra:
            context = json.dumps(extra, default=str)
            return f"{message} | context={context}"
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._add_context(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._add_context(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._add_context(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._add_context(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._add_context(message, kwargs))
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(self._add_context(message, kwargs))
    
    # ============================================================================
    # Specialized Logging Methods
    # ============================================================================
    
    def log_query(self, query: str, query_id: str, user_id: str = None):
        """Log incoming query."""
        self.info(
            "Received query",
            query_id=query_id,
            query=query[:100],  # Truncate long queries
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_routing_decision(self, query_id: str, execution_path: str, 
                            reasoning: str, confidence: float):
        """Log routing decision."""
        self.info(
            "Routing decision made",
            query_id=query_id,
            execution_path=execution_path,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def log_retrieval(self, query_id: str, num_results: int, 
                     search_time_ms: float, source: str = "web"):
        """Log retrieval results."""
        self.info(
            "Retrieval completed",
            query_id=query_id,
            num_results=num_results,
            search_time_ms=search_time_ms,
            source=source
        )
    
    def log_graph_construction(self, query_id: str, num_entities: int, 
                              num_relationships: int, construction_time_ms: float):
        """Log graph construction."""
        self.info(
            "Graph constructed",
            query_id=query_id,
            num_entities=num_entities,
            num_relationships=num_relationships,
            construction_time_ms=construction_time_ms
        )
    
    def log_answer_generation(self, query_id: str, answer_length: int, 
                             num_citations: int, confidence: float,
                             generation_time_ms: float):
        """Log answer generation."""
        self.info(
            "Answer generated",
            query_id=query_id,
            answer_length=answer_length,
            num_citations=num_citations,
            confidence=confidence,
            generation_time_ms=generation_time_ms
        )
    
    def log_quality_metrics(self, query_id: str, metrics: Dict[str, float], 
                           passes_threshold: bool):
        """Log quality metrics."""
        self.info(
            "Quality evaluation completed",
            query_id=query_id,
            metrics=metrics,
            passes_threshold=passes_threshold
        )
    
    def log_self_healing(self, query_id: str, action_type: str, 
                        attempt: int, reasoning: str):
        """Log self-healing action."""
        self.warning(
            "Self-healing triggered",
            query_id=query_id,
            action_type=action_type,
            attempt=attempt,
            reasoning=reasoning
        )
    
    def log_pipeline_completion(self, query_id: str, execution_path: str,
                               total_time_ms: float, cost_estimate: float,
                               success: bool):
        """Log pipeline completion."""
        log_func = self.info if success else self.error
        log_func(
            "Pipeline completed",
            query_id=query_id,
            execution_path=execution_path,
            total_time_ms=total_time_ms,
            cost_estimate=cost_estimate,
            success=success
        )
    
    def log_guardrail_violation(self, query_id: str, violation_type: str, 
                               details: str):
        """Log guardrail violation."""
        self.warning(
            "Guardrail violation detected",
            query_id=query_id,
            violation_type=violation_type,
            details=details
        )
    
    def log_cache_hit(self, cache_key: str, cache_type: str):
        """Log cache hit."""
        self.debug(
            "Cache hit",
            cache_key=cache_key,
            cache_type=cache_type
        )
    
    def log_cache_miss(self, cache_key: str, cache_type: str):
        """Log cache miss."""
        self.debug(
            "Cache miss",
            cache_key=cache_key,
            cache_type=cache_type
        )
    
    def log_llm_call(self, provider: str, model: str, prompt_tokens: int,
                    completion_tokens: int, cost_estimate: float,
                    latency_ms: float):
        """Log LLM API call."""
        self.debug(
            "LLM API call",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_estimate=cost_estimate,
            latency_ms=latency_ms
        )
    
    def log_error_with_context(self, error_type: str, error_message: str,
                              query_id: str = None, recoverable: bool = False,
                              **kwargs):
        """Log error with full context."""
        self.error(
            f"Error occurred: {error_type}",
            error_type=error_type,
            error_message=error_message,
            query_id=query_id,
            recoverable=recoverable,
            **kwargs
        )


# Singleton instance
_logger: StructuredLogger = None


def get_logger(log_file: str = "./logs/research_analyst.log",
              log_level: str = "INFO",
              rotation: str = "100 MB") -> StructuredLogger:
    """Get or create logger instance."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger(log_file, log_level, rotation)
    return _logger


def setup_logger_from_settings(settings) -> StructuredLogger:
    """Setup logger from settings object."""
    return get_logger(
        log_file=settings.log_file,
        log_level=settings.log_level,
        rotation=settings.log_rotation
    )
