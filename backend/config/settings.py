"""Application settings and configuration management.

This module provides centralized configuration management using Pydantic Settings.
All environment variables and application settings are defined and validated here.

Usage:
    >>> from backend.config.settings import settings
    >>> print(settings.OPENAI_API_KEY)
    >>> print(settings.MCP_WEATHER_SERVER_URL)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, BeforeValidator
from typing import Literal, Annotated
from functools import lru_cache
import os
from pathlib import Path


def parse_cors_origins(v: str | list[str]) -> list[str]:
    """Parse CORS_ORIGINS from string or list."""
    if isinstance(v, str):
        # Handle empty string
        if not v or v.strip() == "":
            return ["*"]
        # Handle comma-separated string
        return [origin.strip() for origin in v.split(",") if origin.strip()]
    return v if isinstance(v, list) else ["*"]


# Get project root directory (using resolve() for absolute path - more robust)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable loading.

    All settings are loaded from environment variables (.env file or system env).
    Type validation is performed automatically by Pydantic.

    Environment Variables:
        - Load from .env file in project root
        - System environment variables override .env file
        - All variables are type-validated
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra env vars not defined here
        # Disable JSON parsing for list fields - use custom validators instead
        env_parse_none_str="null"
    )

    # ============================================================================
    # APPLICATION SETTINGS
    # ============================================================================

    APP_NAME: str = Field(
        default="Weather AI Agent Service",
        description="Application name"
    )

    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version (Level 1)"
    )

    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )

    DEBUG: bool = Field(
        default=True,
        description="Debug mode (disable in production)"
    )

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # ============================================================================
    # API SERVER SETTINGS
    # ============================================================================

    HOST: str = Field(
        default="0.0.0.0",
        description="FastAPI server host"
    )

    PORT: int = Field(
        default=8000,
        description="FastAPI server port"
    )

    RELOAD: bool = Field(
        default=True,
        description="Enable auto-reload for development"
    )

    WORKERS: int = Field(
        default=1,
        description="Number of worker processes (production only)"
    )

    # ============================================================================
    # LLM API KEYS (REQUIRED)
    # ============================================================================

    OPENAI_API_KEY: str = Field(
        ...,
        description="OpenAI API key (REQUIRED for GPT-4o-mini)",
        min_length=20
    )

    ANTHROPIC_API_KEY: str | None = Field(
        default=None,
        description="Anthropic API key (optional, for future use)",
        min_length=20
    )

    # ============================================================================
    # LLM MODEL CONFIGURATION
    # ============================================================================

    PRIMARY_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Primary LLM model for weather agent"
    )

    MODEL_TEMPERATURE: float = Field(
        default=0.7,
        description="LLM temperature (0.0-2.0)",
        ge=0.0,
        le=2.0
    )

    MODEL_MAX_TOKENS: int | None = Field(
        default=None,
        description="Maximum tokens for LLM response (None = model default)"
    )

    # ============================================================================
    # MCP SERVER CONFIGURATION
    # ============================================================================

    # ============================================================================
    # WEATHER MCP SERVER CONFIGURATION
    # ============================================================================

    MCP_WEATHER_SERVER_URL: str = Field(
        default="http://localhost:8080",
        description="Weather MCP server URL"
    )

    MCP_WEATHER_SERVER_ENABLED: bool = Field(
        default=True,
        description="Enable Weather MCP server"
    )

    MCP_WEATHER_REQUEST_TIMEOUT: int = Field(
        default=90,
        description="Weather MCP request timeout in seconds (default: 90s to accommodate retry cascades)",
        ge=5,
        le=180
    )

    MCP_WEATHER_HEALTH_CHECK_TIMEOUT: int = Field(
        default=500,
        description="Weather MCP health check timeout in milliseconds"
    )

    MCP_WEATHER_MAX_RETRIES: int = Field(
        default=3,
        description="Maximum retries for Weather MCP requests"
    )

    MCP_WEATHER_RETRY_DELAY: int = Field(
        default=1000,
        description="Delay between Weather MCP retries in milliseconds (base delay for exponential backoff)"
    )

    # ============================================================================
    # HURRICANE MCP SERVER CONFIGURATION
    # ============================================================================

    MCP_HURRICANE_SERVER_URL: str = Field(
        default="http://localhost:8081",
        description="Hurricane Tracker MCP server URL"
    )

    MCP_HURRICANE_SERVER_ENABLED: bool = Field(
        default=True,
        description="Enable Hurricane Tracker MCP server"
    )

    MCP_HURRICANE_REQUEST_TIMEOUT: int = Field(
        default=90,
        description="Hurricane MCP request timeout in seconds (default: 90s for complex hurricane queries)",
        ge=5,
        le=180
    )

    MCP_HURRICANE_HEALTH_CHECK_TIMEOUT: int = Field(
        default=500,
        description="Hurricane MCP health check timeout in milliseconds"
    )

    MCP_HURRICANE_MAX_RETRIES: int = Field(
        default=3,
        description="Maximum retries for Hurricane MCP requests"
    )

    MCP_HURRICANE_RETRY_DELAY: int = Field(
        default=1000,
        description="Delay between Hurricane MCP retries in milliseconds (base delay for exponential backoff)"
    )

    # ============================================================================
    # MCP GLOBAL CONFIGURATION
    # ============================================================================

    MCP_ENABLE_FAILOVER: bool = Field(
        default=True,
        description="Enable MCP failover to direct NHC API calls if MCP server is down"
    )

    # ============================================================================
    # AGENT FEATURE FLAGS (Level 2+)
    # ============================================================================

    ENABLE_RAG: bool = Field(
        default=True,
        description="Enable RAG retrieval (8 tools: 3 MCP + 5 RAG). Can be overridden at runtime."
    )

    ENABLE_COT: bool = Field(
        default=False,  # Changed from True to False for explicit Studio control
        description="Enable Chain-of-Thought reasoning (5-step framework). Can be overridden at runtime."
    )

    # ============================================================================
    # WORKFLOW TIMEOUT CONFIGURATION (Level 4+)
    # ============================================================================

    WORKFLOW_TIMEOUT_SECONDS: float = Field(
        default=45.0,
        description="Maximum time (seconds) for agent workflow execution. Prevents hanging queries. Default: 45s (reasonable for multi-agent with LLM calls)."
    )

    WORKFLOW_TIMEOUT_SIMPLE_SECONDS: float = Field(
        default=30.0,
        description="Maximum time (seconds) for SIMPLE tier queries (basic single-agent). Default: 30s (increased from 15s to support CoT reasoning)."
    )

    WORKFLOW_TIMEOUT_EMERGENCY_SECONDS: float = Field(
        default=60.0,
        description="Maximum time (seconds) for EMERGENCY tier queries (critical safety queries). Default: 60s (more time for thorough analysis)."
    )

    WORKFLOW_TIMEOUT_HURRICANE_SECONDS: float = Field(
        default=90.0,
        description="Maximum time (seconds) for HURRICANE-related queries with evaluation. Default: 90s (complex queries need more time)."
    )

    WORKFLOW_TIMEOUT_BATCH_SECONDS: float = Field(
        default=120.0,
        description="Maximum time (seconds) for BATCH evaluation operations. Default: 120s."
    )

    # ============================================================================
    # HYBRID SEARCH CONFIGURATION (Level 2)
    # ============================================================================

    HYBRID_SEARCH_VECTOR_WEIGHT: float = Field(
        default=0.7,
        description="Weight for vector (semantic) search in hybrid search (0.0-1.0). Default: 0.7 (70%)",
        ge=0.0,
        le=1.0
    )

    HYBRID_SEARCH_BM25_WEIGHT: float = Field(
        default=0.3,
        description="Weight for BM25 (keyword) search in hybrid search (0.0-1.0). Default: 0.3 (30%)",
        ge=0.0,
        le=1.0
    )

    # ============================================================================
    # DATABASE CONFIGURATION (for future levels)
    # ============================================================================

    DATABASE_URL: str | None = Field(
        default=None,
        description="PostgreSQL database URL (Level 3+) - DEPRECATED, use POSTGRES_URL"
    )

    # ============================================================================
    # DATABASE CONFIGURATION (Level 3c)
    # ============================================================================

    POSTGRES_URL: str | None = Field(
        default=None,
        description="PostgreSQL connection URL for Layers 5 & 7 (Procedural & Reflective Memory). Format: postgresql://user:password@host:port/database"
    )

    POSTGRES_MAX_CONNECTIONS: int = Field(
        default=20,
        description="Maximum PostgreSQL connection pool size",
        ge=5,
        le=100
    )

    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for caching and short-term memory (Level 3a+)"
    )

    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database URL (Level 2+)"
    )

    QDRANT_API_KEY: str | None = Field(
        default=None,
        description="Qdrant API key (optional, for Qdrant Cloud in Level 5a+)"
    )

    # ============================================================================
    # LANGSMITH CONFIGURATION (Observability)
    # ============================================================================

    LANGCHAIN_TRACING_V2: bool = Field(
        default=False,
        description="Enable LangSmith tracing"
    )

    LANGCHAIN_API_KEY: str | None = Field(
        default=None,
        description="LangSmith API key (optional)"
    )

    LANGCHAIN_PROJECT: str = Field(
        default="weather-ai-agent-level1",
        description="LangSmith project name"
    )

    LANGCHAIN_ENDPOINT: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint"
    )

    # ============================================================================
    # CORS CONFIGURATION
    # ============================================================================

    CORS_ORIGINS: str = Field(
        default="*",
        description="Allowed CORS origins (comma-separated string, e.g., 'http://localhost:3000,http://localhost:8080')"
    )

    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )

    # ============================================================================
    # SECURITY SETTINGS
    # ============================================================================

    SECRET_KEY: str = Field(
        default="development-secret-key-change-in-production",
        description="Secret key for session management (CHANGE IN PRODUCTION!)"
    )

    API_KEY_HEADER: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication (Level 5c+)"
    )

    # ============================================================================
    # RATE LIMITING (Level 5c+)
    # ============================================================================

    RATE_LIMIT_ENABLED: bool = Field(
        default=False,
        description="Enable rate limiting (Level 5c)"
    )

    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        description="Maximum requests per minute"
    )

    # ============================================================================
    # OBSERVABILITY CONFIGURATION (Level 5c)
    # ============================================================================

    NEO4J_BOLT_URL: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt protocol URL for Graphiti"
    )

    NEO4J_USER: str = Field(
        default="neo4j",
        description="Neo4j username"
    )

    NEO4J_PASSWORD: str = Field(
        default="password",
        description="Neo4j password"
    )

    PROMETHEUS_URL: str = Field(
        default="http://localhost:9090",
        description="Prometheus server URL"
    )

    GRAFANA_URL: str = Field(
        default="http://localhost:3001",
        description="Grafana dashboard URL"
    )

    LOKI_URL: str = Field(
        default="http://localhost:3100",
        description="Loki log aggregation URL"
    )

    # ============================================================================
    # VALIDATORS
    # ============================================================================

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is valid."""
        if v not in ["development", "staging", "production"]:
            raise ValueError("ENVIRONMENT must be development, staging, or production")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """Warn if using default secret key in production."""
        environment = info.data.get("ENVIRONMENT", "development")
        if environment == "production" and v == "development-secret-key-change-in-production":
            raise ValueError("MUST change SECRET_KEY in production environment!")
        return v

    @field_validator("MODEL_MAX_TOKENS", mode="before")
    @classmethod
    def validate_model_max_tokens(cls, v) -> int | None:
        """Handle empty string for optional int field."""
        if v == "" or v is None:
            return None
        return int(v) if isinstance(v, str) else v

    @field_validator("HYBRID_SEARCH_BM25_WEIGHT")
    @classmethod
    def validate_hybrid_search_weights(cls, v: float, info) -> float:
        """Ensure hybrid search weights sum to approximately 1.0."""
        vector_weight = info.data.get("HYBRID_SEARCH_VECTOR_WEIGHT", 0.7)
        bm25_weight = v
        total = vector_weight + bm25_weight

        # Allow small floating point errors (within 0.01)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Hybrid search weights must sum to 1.0. "
                f"Current: VECTOR_WEIGHT={vector_weight} + BM25_WEIGHT={bm25_weight} = {total}. "
                f"Please adjust weights so they sum to 1.0 (e.g., 0.7 + 0.3 = 1.0)"
            )
        return v

    # ============================================================================
    # COMPUTED PROPERTIES
    # ============================================================================

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == "production"

    @property
    def api_base_url(self) -> str:
        """Get the full API base URL."""
        return f"http://{self.HOST}:{self.PORT}"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS string to list for FastAPI."""
        return parse_cors_origins(self.CORS_ORIGINS)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Create singleton settings instance
# This is loaded once and reused throughout the application
settings = Settings()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@lru_cache
def get_settings() -> Settings:
    """Get settings instance (for dependency injection).

    Uses lru_cache to prevent re-parsing environment variables on each call.
    This ensures true singleton behavior across the application.

    Example:
        >>> from backend.config.settings import get_settings
        >>> settings = get_settings()
        >>> print(settings.OPENAI_API_KEY)

    Returns:
        Settings: Cached settings instance
    """
    return Settings()


def print_settings_summary():
    """Print a summary of current settings (for debugging).

    WARNING: Does NOT print sensitive values (API keys, secrets).
    """
    print("\n" + "=" * 60)
    print("WEATHER AI AGENT - CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"API Server: {settings.api_base_url}")
    print(f"Primary Model: {settings.PRIMARY_MODEL}")
    print(f"Model Temperature: {settings.MODEL_TEMPERATURE}")
    print(f"MCP Weather Server: {settings.MCP_WEATHER_SERVER_URL}")
    print(f"MCP Hurricane Server: {settings.MCP_HURRICANE_SERVER_URL}")
    print(f"LangSmith Tracing: {settings.LANGCHAIN_TRACING_V2}")
    print(f"OpenAI API Key: {' Configured' if settings.OPENAI_API_KEY else 'L Missing'}")
    print(f"Anthropic API Key: {' Configured' if settings.ANTHROPIC_API_KEY else '� Not configured (optional)'}")
    print("=" * 60 + "\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Print settings summary when run directly
    print_settings_summary()
