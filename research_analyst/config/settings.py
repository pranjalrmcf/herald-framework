"""
Centralized configuration management using pydantic-settings.
Loads configuration from environment variables with validation.
"""

from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_prefix="RA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ============================================================================
    # API Keys
    # ============================================================================
    
    groq_api_key: Optional[SecretStr] = Field(
        None,
        description="Groq API key"
    )

    serper_api_key: Optional[SecretStr] = Field(
        None,
        description="Serper API key"
    )

    dev_mode: bool = Field(
        False,
        description="Development mode — relaxes some guardrail thresholds"
    )

    # ============================================================================
    # LLM Configuration
    # ============================================================================
    
    default_llm_provider: Literal["groq", "ollama"] = Field(
    "groq",
    description="LLM provider: 'groq' (cloud) or 'ollama' (local)"
    )
    
    default_model: str = Field(
        "llama-3.3-70b-versatile",
        description="Default model name"
    )

    ollama_base_url: str = Field(
    "http://localhost:11434",
    description="Ollama server base URL"
    )
    
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Local model for generating embeddings (sentence-transformers)"
    )
    
    max_tokens: int = Field(
        12000,
        ge=100,
        le=128000,
        description="Maximum tokens for LLM responses"
    )
    
    temperature: float = Field(
        0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )

    # ============================================================================
    # Search Configuration
    # ============================================================================

    mock_search: bool = Field(
        False,
        description="Use mock search instead of real web search"
    )
    
    search_engine: Literal["duckduckgo", "serper", "you"] = Field(
        "duckduckgo",
        description="Search engine to use"
    )
    
    max_search_results: int = Field(
        12,
        ge=1,
        le=50,
        description="Maximum number of search results"
    )
    
    search_timeout: int = Field(
        30,
        ge=5,
        le=120,
        description="Search timeout in seconds"
    )

    # ============================================================================
    # Graph Configuration
    # ============================================================================
    
    graph_backend: Literal["networkx", "neo4j"] = Field(
        "networkx",
        description="Graph storage backend"
    )
    
    neo4j_uri: str = Field(
        "bolt://localhost:7687",
        description="Neo4j connection URI"
    )
    
    neo4j_user: str = Field(
        "neo4j",
        description="Neo4j username"
    )
    
    neo4j_password: str = Field(
        "password",
        description="Neo4j password"
    )
    
    max_graph_hops: int = Field(
        2,
        ge=1,
        le=5,
        description="Maximum hops for graph traversal"
    )

    # ============================================================================
    # Cache Configuration
    # ============================================================================
    
    cache_backend: Literal["disk", "redis", "memory"] = Field(
        "disk",
        description="Cache storage backend"
    )
    
    redis_host: str = Field(
        "localhost",
        description="Redis host"
    )
    
    redis_port: int = Field(
        6379,
        ge=1,
        le=65535,
        description="Redis port"
    )
    
    redis_db: int = Field(
        0,
        ge=0,
        le=15,
        description="Redis database number"
    )
    
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Default cache TTL in seconds"
    )
    
    cache_dir: str = Field(
        "./cache",
        description="Directory for disk cache"
    )

    # ============================================================================
    # Semantic Cache Configuration
    # ============================================================================

    semantic_cache_enabled: bool = Field(
        True,
        description=(
            "Enable semantic similarity cache. Paraphrased queries that are "
            "sufficiently similar to a cached query return the cached answer "
            "without running the full pipeline."
        )
    )

    semantic_cache_similarity_threshold: float = Field(
        0.92,
        ge=0.5,
        le=1.0,
        description="Cosine similarity threshold for semantic cache hit (0.92 = very tight)"
    )

    semantic_cache_max_entries: int = Field(
        10000,
        ge=100,
        le=100000,
        description="Maximum number of entries stored in the semantic cache index"
    )

    cache_entity_extraction: bool = Field(
        True,
        description="Cache entity extraction results per document content hash"
    )

    cache_relationship_extraction: bool = Field(
        True,
        description="Cache relationship extraction results per document content hash"
    )

    cache_llm_judge_scores: bool = Field(
        True,
        description="Cache LLM judge scores keyed by answer text hash"
    )

    cache_speculation_candidates: bool = Field(
        True,
        description="Cache speculative RAG candidates keyed by strategy + query hash"
    )

    cache_warm_on_startup: bool = Field(
        False,
        description="Pre-warm cache with top-N queries from evaluation history on startup"
    )

    cache_warm_top_n: int = Field(
        20,
        ge=1,
        le=100,
        description="Number of top queries to pre-warm on startup"
    )

    # ============================================================================
    # Performance Configuration
    # ============================================================================
    
    max_workers: int = Field(
        4,
        ge=1,
        le=32,
        description="Maximum number of worker threads"
    )
    
    request_timeout: int = Field(
        60,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    
    max_retries: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum number of retries"
    )
    
    rate_limit_per_minute: int = Field(
        60,
        ge=1,
        le=1000,
        description="Rate limit per minute"
    )

    # ============================================================================
    # Logging Configuration
    # ============================================================================
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO",
        description="Logging level"
    )
    
    log_file: str = Field(
        "./logs/research_analyst.log",
        description="Log file path"
    )
    
    log_rotation: str = Field(
        "100 MB",
        description="Log rotation size"
    )

    # ============================================================================
    # Monitoring Configuration
    # ============================================================================
    
    enable_metrics: bool = Field(
        True,
        description="Enable Prometheus metrics"
    )
    
    metrics_port: int = Field(
        9090,
        ge=1024,
        le=65535,
        description="Metrics server port"
    )

    # ============================================================================
    # Safety & Guardrails
    # ============================================================================
    
    enable_input_guardrails: bool = Field(
        True,
        description="Enable input guardrails"
    )
    
    enable_output_guardrails: bool = Field(
        True,
        description="Enable output guardrails"
    )
    
    min_citation_coverage: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Minimum citation coverage threshold"
    )
    
    min_confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    
    max_self_healing_attempts: int = Field(
        2,
        ge=1,
        le=5,
        description="Maximum self-healing attempts"
    )

    # ============================================================================
    # LLM-as-Judge & Regression Detection
    # ============================================================================
    
    enable_llm_judge: bool = Field(
        True,
        description="Enable LLM-as-Judge evaluation"
    )
    
    enable_regression_detection: bool = Field(
        True,
        description="Enable performance regression detection"
    )
    
    evaluation_storage_backend: Literal["sqlite", "json"] = Field(
        "sqlite",
        description="Storage backend for evaluation history"
    )
    
    evaluation_db_path: str = Field(
        "./data/evaluations.db",
        description="Path to evaluation database (SQLite)"
    )
    
    evaluation_json_path: str = Field(
        "./data/evaluations.json",
        description="Path to evaluation JSON file"
    )
    
    baseline_window_size: int = Field(
        100,
        ge=10,
        le=1000,
        description="Number of recent queries to use for baseline calculation"
    )
    
    
    regression_threshold: float = Field(
        2.0,
        ge=0.5,
        le=5.0,
        description="Z-score threshold for regression detection (std devs from mean)"
    )
    
    min_baseline_samples: int = Field(
        20,
        ge=5,
        le=100,
        description="Minimum samples needed before regression detection activates"
    )

    # ============================================================================
    # Speculative RAG Configuration
    # ============================================================================

    speculative_rag_enabled: bool = Field(
        True,
        description=(
            "Enable speculative parallel candidate generation. "
            "Runs 3 retrieval strategies in parallel and uses LLM-Judge "
            "to select the best answer. Only applies on RESEARCH path."
        )
    )

    num_speculative_candidates: int = Field(
        3,
        ge=2,
        le=5,
        description="Number of parallel retrieval strategies (candidates) to generate"
    )

    speculative_candidate_timeout: int = Field(
        30,
        ge=10,
        le=120,
        description="Timeout in seconds per speculative candidate generation"
    )

    speculative_judge_model: Optional[str] = Field(
        None,
        description=(
            "LLM model to use for speculative judge scoring. "
            "None = use default_model. Can be set to a cheaper/faster model."
        )
    )

    # ============================================================================
    # Memory Configuration
    # ============================================================================

    memory_enabled: bool = Field(
        True,
        description=(
            "Enable cross-session episodic memory. Stores entities, "
            "relationships, and answer summaries from past sessions and "
            "injects relevant context into synthesis for follow-up queries."
        )
    )

    memory_db_path: str = Field(
        "./data/memory.db",
        description="SQLite file path for cross-session memory store"
    )

    memory_session_ttl_days: int = Field(
        30,
        ge=1,
        le=365,
        description="Number of days to retain memory sessions before cleanup"
    )

    # ============================================================================
    # Temporal Decay Configuration
    # ============================================================================

    temporal_decay_enabled: bool = Field(
        True,
        description=(
            "Enable temporal decay scoring on knowledge graph edges. "
            "Downweights stale relationships based on source document age "
            "using exponential half-life decay."
        )
    )

    temporal_decay_halflife_days: int = Field(
        90,
        ge=7,
        le=730,
        description=(
            "Half-life in days for relationship freshness decay. "
            "A relationship at halflife_days old scores 0.5. "
            "Default 90 days: 30d=0.79, 90d=0.50, 180d=0.25, 365d=0.06"
        )
    )

    # ============================================================================
    # Uncertainty Quantification
    # ============================================================================

    uncertainty_enabled: bool = Field(
        True,
        description=(
            "Enable uncertainty quantification on claims. "
            "Computes credible intervals per claim and injects inline "
            "uncertainty markers into the answer text for flagged claims."
        )
    )

    uncertainty_low_threshold: float = Field(
        0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Claim confidence below this value is classified as high uncertainty "
            "and will be flagged with an inline marker in the answer."
        )
    )

    # ============================================================================
    # Development Configuration
    # ============================================================================
    
    debug_mode: bool = Field(
        False,
        description="Enable debug mode"
    )
    
    mock_llm_calls: bool = Field(
        False,
        description="Mock LLM calls for testing"
    )

    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration."""
        return {
            "provider": self.default_llm_provider,
            "model": self.default_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
    
    def get_cache_config(self) -> dict:
        """Get cache configuration."""
        if self.cache_backend == "redis":
            return {
                "backend": "redis",
                "host": self.redis_host,
                "port": self.redis_port,
                "db": self.redis_db,
                "ttl": self.cache_ttl_seconds,
            }
        elif self.cache_backend == "disk":
            return {
                "backend": "disk",
                "directory": self.cache_dir,
                "ttl": self.cache_ttl_seconds,
            }
        else:
            return {
                "backend": "memory",
                "ttl": self.cache_ttl_seconds,
            }
    
    def get_graph_config(self) -> dict:
        """Get graph configuration."""
        if self.graph_backend == "neo4j":
            return {
                "backend": "neo4j",
                "uri": self.neo4j_uri,
                "user": self.neo4j_user,
                "password": self.neo4j_password,
                "max_hops": self.max_graph_hops,
            }
        else:
            return {
                "backend": "networkx",
                "max_hops": self.max_graph_hops,
            }

    def get_speculative_rag_config(self) -> dict:
        """Get speculative RAG configuration."""
        return {
            "enabled": self.speculative_rag_enabled,
            "num_candidates": self.num_speculative_candidates,
            "candidate_timeout": self.speculative_candidate_timeout,
            "judge_model": self.speculative_judge_model or self.default_model,
        }

    def get_memory_config(self) -> dict:
        """Get memory configuration."""
        return {
            "enabled": self.memory_enabled,
            "db_path": self.memory_db_path,
            "ttl_days": self.memory_session_ttl_days,
        }

    def get_temporal_decay_config(self) -> dict:
        """Get temporal decay configuration."""
        return {
            "enabled": self.temporal_decay_enabled,
            "halflife_days": self.temporal_decay_halflife_days,
        }

    def get_semantic_cache_config(self) -> dict:
        """Get semantic cache configuration."""
        return {
            "enabled": self.semantic_cache_enabled,
            "threshold": self.semantic_cache_similarity_threshold,
            "max_entries": self.semantic_cache_max_entries,
        }


# ============================================================================
# Singleton instance
# ============================================================================

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings