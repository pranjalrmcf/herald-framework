"""
Autonomous Research Analyst System
A production-grade research analyst using agent-controlled RAG with graph reasoning.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .orchestration.orchestrator import ResearchAnalyst
from .core.models import Query, Answer, PipelineResponse

__all__ = [
    "ResearchAnalyst",
    "Query", 
    "Answer",
    "PipelineResponse",
]
