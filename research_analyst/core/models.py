"""
Core data models for the HERALD research analyst system.
All data structures use Pydantic v2 for validation and type safety.

Changes from v0.1:
  - Added UncertaintyBand Pydantic model (replaces dataclass monkeypatch)
  - Added GEvalScores model (formalises G-Eval chain-of-thought output)
  - Claim.uncertainty_band / Claim.uncertainty_level  (proper fields)
  - QualityMetrics.geval_composite                    (was set via setattr)
  - QualityMetrics.calculate_composite_score blends G-Eval at 20% when present
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Enumerations
# ============================================================================

class QueryIntent(str, Enum):
    SEMANTIC   = "semantic"
    ENTITY     = "entity"
    RELATIONAL = "relational"
    TEMPORAL   = "temporal"
    HYBRID     = "hybrid"


class QueryComplexity(str, Enum):
    SIMPLE  = "simple"
    MEDIUM  = "medium"
    COMPLEX = "complex"


class ExecutionPath(str, Enum):
    FAST     = "fast"
    RESEARCH = "research"


class SourceType(str, Enum):
    WEB          = "web"
    ACADEMIC     = "academic"
    NEWS         = "news"
    SOCIAL_MEDIA = "social_media"
    GOVERNMENT   = "government"
    UNKNOWN      = "unknown"


class EntityType(str, Enum):
    PERSON       = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION     = "GPE"
    DATE         = "DATE"
    EVENT        = "EVENT"
    PRODUCT      = "PRODUCT"
    UNKNOWN      = "UNKNOWN"


# ============================================================================
# Query Models
# ============================================================================

class Query(BaseModel):
    text:       str             = Field(..., min_length=1)
    user_id:    Optional[str]   = None
    session_id: Optional[str]   = None
    timestamp:  datetime        = Field(default_factory=datetime.utcnow)
    metadata:   Dict[str, Any]  = Field(default_factory=dict)

    model_config = {"json_schema_extra": {
        "example": {"text": "What is the relationship between OpenAI and Microsoft?"}
    }}


class NormalizedQuery(BaseModel):
    original_text:      str
    normalized_text:    str
    intent:             QueryIntent
    domain:             Optional[str]           = None
    time_range:         Optional[Dict[str, Any]]= None
    entities_mentioned: List[str]               = Field(default_factory=list)
    language:           str                     = "en"
    complexity:         QueryComplexity
    requires_graph:     bool                    = False

    model_config = {"use_enum_values": True}


class RoutingDecision(BaseModel):
    execution_path:    ExecutionPath
    reasoning:         str
    confidence:        float = Field(..., ge=0.0, le=1.0)
    estimated_cost:    float = Field(..., ge=0.0)
    estimated_latency: float = Field(..., ge=0.0)

    model_config = {"use_enum_values": True}


# ============================================================================
# Document Models
# ============================================================================

class Document(BaseModel):
    doc_id:           str
    url:              str
    title:            str
    content:          str
    snippet:          Optional[str]      = None
    source_type:      SourceType         = SourceType.UNKNOWN
    author:           Optional[str]      = None
    published_date:   Optional[datetime] = None
    retrieved_at:     datetime           = Field(default_factory=datetime.utcnow)
    credibility_score:float              = Field(default=0.5, ge=0.0, le=1.0)
    metadata:         Dict[str, Any]     = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class DocumentChunk(BaseModel):
    chunk_id:   str
    doc_id:     str
    text:       str
    chunk_index:int
    embedding:  Optional[List[float]] = None
    metadata:   Dict[str, Any]        = Field(default_factory=dict)


class RankedDocument(BaseModel):
    document:        Document
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    credibility_score:float= Field(..., ge=0.0, le=1.0)
    recency_score:   float = Field(..., ge=0.0, le=1.0)
    final_score:     float = Field(..., ge=0.0, le=1.0)
    ranking_factors: Dict[str, float] = Field(default_factory=dict)


# ============================================================================
# Graph Models
# ============================================================================

class Entity(BaseModel):
    entity_id:   str
    text:        str
    entity_type: EntityType
    aliases:     List[str]       = Field(default_factory=list)
    confidence:  float           = Field(..., ge=0.0, le=1.0)
    source_doc_id: str
    attributes:  Dict[str, Any]  = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class Relationship(BaseModel):
    relationship_id: str
    subject:         str
    predicate:       str
    object:          str
    confidence:      float           = Field(..., ge=0.0, le=1.0)
    source_doc_id:   str
    source_url:      str
    temporal_info:   Optional[Dict[str, Any]] = None
    metadata:        Dict[str, Any]  = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    graph_id:      str
    entities:      List[Entity]
    relationships: List[Relationship]
    created_at:    datetime        = Field(default_factory=datetime.utcnow)
    metadata:      Dict[str, Any]  = Field(default_factory=dict)


class Subgraph(BaseModel):
    subgraph_id:      str
    central_entities: List[str]
    entities:         List[Entity]
    relationships:    List[Relationship]
    relevance_score:  float = Field(..., ge=0.0, le=1.0)


# ============================================================================
# Uncertainty Models
# ============================================================================

class UncertaintyBand(BaseModel):
    """
    Epistemic credible interval for a claim's confidence score.

    Computed from five signals:
        1. Number of supporting sources   (more → narrower band)
        2. Source domain diversity        (diverse → narrower band)
        3. Claim controversy flag         (controversial → wider band)
        4. LLM judge factuality score     (if available → blended)
        5. Source credibility scores

    Used by UncertaintyQuantifier and surfaced in inline answer markers.
    """
    lower:            float
    upper:            float
    level:            Literal["high", "medium", "low"]
    num_sources:      int
    source_diversity: float
    is_controversial: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower":            round(self.lower, 3),
            "upper":            round(self.upper, 3),
            "level":            self.level,
            "num_sources":      self.num_sources,
            "source_diversity": round(self.source_diversity, 3),
            "is_controversial": self.is_controversial,
        }


# ============================================================================
# Evidence & Synthesis Models
# ============================================================================

class Claim(BaseModel):
    """
    A factual claim extracted from retrieved sources.

    uncertainty_band and uncertainty_level are populated by
    UncertaintyQuantifier after answer generation.
    """
    claim_id:           str
    text:               str
    supporting_sources: List[str]   = Field(default_factory=list)
    confidence:         float       = Field(..., ge=0.0, le=1.0)
    is_controversial:   bool        = False
    # Populated by UncertaintyQuantifier (first or second pass)
    uncertainty_band:   Optional[UncertaintyBand]           = None
    uncertainty_level:  Optional[Literal["high", "medium", "low"]] = None


class Evidence(BaseModel):
    evidence_id:         str
    claims:              List[Claim]
    supporting_documents:List[Document]
    counter_arguments:   List[str]              = Field(default_factory=list)
    relationship_chains: Optional[List[List[Relationship]]] = None
    summary:             Optional[str]          = None


class Citation(BaseModel):
    source_url:   str
    source_title: str
    excerpt:      Optional[str] = None
    relevance:    float         = Field(..., ge=0.0, le=1.0)


class Answer(BaseModel):
    answer_id:     str
    query:         str
    text:          str
    citations:     List[Citation]
    confidence:    float    = Field(..., ge=0.0, le=1.0)
    generated_at:  datetime = Field(default_factory=datetime.utcnow)
    execution_path:ExecutionPath
    metadata:      Dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


# ============================================================================
# Evaluation Models
# ============================================================================

class LLMJudgeScore(BaseModel):
    """Scores from the Neural Factuality Arbitration (LLM-as-Judge) module."""
    grounding_score:    float = Field(..., ge=0.0, le=1.0)
    factuality_score:   float = Field(..., ge=0.0, le=1.0)
    relevance_score:    float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    reasoning:          str
    issues_found:       List[str] = Field(default_factory=list)
    evaluated_at:       datetime  = Field(default_factory=datetime.utcnow)


class GEvalScores(BaseModel):
    """
    Chain-of-thought evaluation scores from G-Eval framework.
    Dimensions are on a 1-5 scale; composite is normalised to [0, 1].
    """
    coherence:    int   = Field(..., ge=1, le=5)
    consistency:  int   = Field(..., ge=1, le=5)
    fluency:      int   = Field(..., ge=1, le=5)
    relevance:    int   = Field(..., ge=1, le=5)
    composite:    float = Field(..., ge=0.0, le=1.0)
    reasoning:    str   = ""
    steps:        List[str] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GEvalScores":
        composite = round(
            (d.get("coherence", 3) + d.get("consistency", 3)
             + d.get("fluency", 3) + d.get("relevance", 3)) / 20.0, 4
        )
        return cls(
            coherence   = min(5, max(1, int(d.get("coherence",   3)))),
            consistency = min(5, max(1, int(d.get("consistency", 3)))),
            fluency     = min(5, max(1, int(d.get("fluency",     3)))),
            relevance   = min(5, max(1, int(d.get("relevance",   3)))),
            composite   = d.get("composite", composite),
            reasoning   = d.get("reasoning", ""),
            steps       = d.get("steps", []),
        )


class QualityMetrics(BaseModel):
    """
    Composite quality assessment for a generated answer.

    Scoring hierarchy:
        heuristic_score = weighted sum of citation_coverage, grounding_score,
                          coherence_score, answer_completeness, source_diversity
        if llm_judge available:
            llm_score = weighted blend of 4 judge dimensions
            base = 0.60 * heuristic + 0.40 * llm_score
        if geval available:
            composite = 0.80 * base + 0.20 * geval_composite
        else:
            composite = base
    """
    # Heuristic metrics
    citation_coverage:  float = Field(..., ge=0.0, le=1.0)
    grounding_score:    float = Field(..., ge=0.0, le=1.0)
    coherence_score:    float = Field(..., ge=0.0, le=1.0)
    answer_completeness:float = Field(..., ge=0.0, le=1.0)
    source_diversity:   float = Field(..., ge=0.0, le=1.0)

    # Neural Factuality Arbitration scores (optional)
    llm_judge_scores:   Optional[LLMJudgeScore] = None

    # G-Eval chain-of-thought scores (optional)
    geval_scores:       Optional[GEvalScores]   = None
    geval_composite:    Optional[float]          = None   # convenience alias

    # Overall
    passes_threshold:   bool
    issues:             List[str] = Field(default_factory=list)
    composite_score:    Optional[float] = None

    def calculate_composite_score(self) -> float:
        """
        Calculate weighted composite score, blending heuristic, LLM judge,
        and G-Eval signals.
        """
        weights = {
            "citation_coverage":  0.15,
            "grounding_score":    0.25,
            "coherence_score":    0.15,
            "answer_completeness":0.20,
            "source_diversity":   0.10,
        }
        heuristic = (
            self.citation_coverage   * weights["citation_coverage"]
            + self.grounding_score   * weights["grounding_score"]
            + self.coherence_score   * weights["coherence_score"]
            + self.answer_completeness * weights["answer_completeness"]
            + self.source_diversity  * weights["source_diversity"]
        )

        if self.llm_judge_scores:
            llm = (
                self.llm_judge_scores.grounding_score    * 0.30
                + self.llm_judge_scores.factuality_score * 0.30
                + self.llm_judge_scores.relevance_score  * 0.20
                + self.llm_judge_scores.completeness_score * 0.20
            )
            base = heuristic * 0.60 + llm * 0.40
        else:
            base = heuristic

        # Blend G-Eval at 20% weight when available
        geval = self.geval_composite
        if geval is None and self.geval_scores is not None:
            geval = self.geval_scores.composite

        if geval is not None:
            composite = round(base * 0.80 + geval * 0.20, 4)
        else:
            composite = round(base, 4)

        return composite


class SelfHealingAction(BaseModel):
    action_type: str
    reasoning:   str
    parameters:  Dict[str, Any] = Field(default_factory=dict)
    timestamp:   datetime       = Field(default_factory=datetime.utcnow)


# ============================================================================
# Persistence & Regression Models
# ============================================================================

class EvaluationRecord(BaseModel):
    evaluation_id:         str
    query_id:              str
    query_text:            str
    answer_id:             str
    quality_metrics:       QualityMetrics
    execution_path:        ExecutionPath
    execution_time_ms:     float
    cost_estimate:         float
    timestamp:             datetime = Field(default_factory=datetime.utcnow)
    self_healing_triggered:bool     = False
    self_healing_attempts: int      = 0

    model_config = {"use_enum_values": True}


class PerformanceBaseline(BaseModel):
    baseline_id:  str
    metric_name:  str
    mean_value:   float
    std_dev:      float
    min_value:    float
    max_value:    float
    sample_size:  int
    calculated_at:datetime = Field(default_factory=datetime.utcnow)
    window_start: datetime
    window_end:   datetime


class RegressionAlert(BaseModel):
    alert_id:          str
    metric_name:       str
    current_value:     float
    baseline_mean:     float
    baseline_std_dev:  float
    z_score:           float
    severity:          Literal["low", "medium", "high", "critical"]
    detected_at:       datetime = Field(default_factory=datetime.utcnow)
    query_id:          str
    recommendation:    str


# ============================================================================
# Pipeline State & Response
# ============================================================================

class PipelineState(BaseModel):
    query:                Query
    normalized_query:     Optional[NormalizedQuery]     = None
    routing_decision:     Optional[RoutingDecision]     = None
    retrieved_documents:  List[Document]                = Field(default_factory=list)
    ranked_documents:     List[RankedDocument]          = Field(default_factory=list)
    knowledge_graph:      Optional[KnowledgeGraph]      = None
    relevant_subgraph:    Optional[Subgraph]            = None
    evidence:             Optional[Evidence]            = None
    answer:               Optional[Answer]              = None
    quality_metrics:      Optional[QualityMetrics]      = None
    self_healing_actions: List[SelfHealingAction]       = Field(default_factory=list)
    execution_time_ms:    Optional[float]               = None
    cost_estimate:        Optional[float]               = None
    metadata:             Dict[str, Any]                = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class PipelineResponse(BaseModel):
    success:          bool
    answer:           Optional[Answer]         = None
    error:            Optional[str]            = None
    execution_path:   Optional[ExecutionPath]  = None
    quality_metrics:  Optional[QualityMetrics] = None
    execution_time_ms:float
    cost_estimate:    float
    metadata:         Dict[str, Any]           = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class ErrorDetails(BaseModel):
    error_code:  str
    message:     str
    details:     Optional[Dict[str, Any]] = None
    timestamp:   datetime = Field(default_factory=datetime.utcnow)
    recoverable: bool     = False