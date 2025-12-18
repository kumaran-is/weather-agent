"""Common type aliases for the Weather AI Agent Service.

This module defines type aliases to improve code readability and reduce
complexity in type annotations throughout the codebase.

Usage:
    >>> from backend.src.types import Coordinates, Timestamp, AgentName
    >>> def get_weather(location: Coordinates) -> WeatherData:
    ...     pass
"""

from datetime import datetime
from typing import Any, Literal, TypeAlias

# ============================================================================
# COORDINATE TYPES
# ============================================================================

Coordinates: TypeAlias = tuple[float, float]
"""Geographic coordinates as (latitude, longitude) tuple.

Example:
    >>> miami_coords: Coordinates = (25.7617, -80.1918)
    >>> (lat, lon) = miami_coords
"""

Latitude: TypeAlias = float
"""Latitude coordinate in decimal degrees (-90 to +90)."""

Longitude: TypeAlias = float
"""Longitude coordinate in decimal degrees (-180 to +180)."""


# ============================================================================
# TIME TYPES
# ============================================================================

Timestamp: TypeAlias = str
"""ISO 8601 timestamp string in UTC (e.g., '2025-12-17T20:00:00Z')."""

UnixTimestamp: TypeAlias = int
"""Unix timestamp in seconds since epoch."""

DatetimeUTC: TypeAlias = datetime
"""Datetime object in UTC timezone."""


# ============================================================================
# AGENT TYPES
# ============================================================================

AgentName: TypeAlias = Literal[
    "weather",
    "triage",
    "hurricane_specialist",
    "alert_manager",
    "supervisor",
    "forecaster",
    "historical_analyst",
    "research",
    "personalization",
    "reflection",
    "critique",
    "debate",
    "self_healing",
    "climate_analyst",
    "meta_prompt",
    "emergency_response",
]
"""Valid agent names in the multi-agent system."""

AgentLevel: TypeAlias = Literal["L1", "L4a", "L4b", "L4c", "AUTO"]
"""Agent complexity levels (L1=single, L4a=3-agent, L4b=8-agent, L4c=15-agent, AUTO=automatic)."""

QueryTierName: TypeAlias = Literal["simple", "standard", "complex", "emergency"]
"""Query classification tiers for intelligent routing."""


# ============================================================================
# CACHE TYPES
# ============================================================================

CacheLayer: TypeAlias = Literal["L1", "L2", "Q3", "L3"]
"""Cache layer identifiers (L1=in-memory, L2=Redis, Q3=semantic, L3=Anthropic)."""

CacheKey: TypeAlias = str
"""Cache key string (usually hash of query + context)."""


# ============================================================================
# WEATHER DATA TYPES
# ============================================================================

Temperature: TypeAlias = float
"""Temperature value in specified units (Celsius or Fahrenheit)."""

WindSpeed: TypeAlias = float
"""Wind speed in mph."""

HurricaneCategory: TypeAlias = Literal[1, 2, 3, 4, 5]
"""Saffir-Simpson hurricane category (1-5)."""

Severity: TypeAlias = Literal["low", "medium", "high", "critical"]
"""Alert severity level."""


# ============================================================================
# API TYPES
# ============================================================================

UserId: TypeAlias = str
"""User identifier (UUID or username)."""

SessionId: TypeAlias = str
"""Session identifier (UUID)."""

ThreadId: TypeAlias = str
"""LangGraph thread/conversation identifier (UUID)."""

TraceId: TypeAlias = str
"""Distributed tracing identifier for observability."""


# ============================================================================
# LLM TYPES
# ============================================================================

ModelName: TypeAlias = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]
"""Supported LLM model identifiers."""

PromptTemplate: TypeAlias = str
"""Prompt template string with placeholders (e.g., 'Weather in {location}')."""

TokenCount: TypeAlias = int
"""Number of tokens (for LLM context)."""


# ============================================================================
# MEMORY TYPES
# ============================================================================

MemoryLayer: TypeAlias = Literal[
    "short_term",
    "long_term",
    "semantic",
    "procedural",
    "emotional",
    "reflective",
    "episodic",
]
"""7-layer memory system identifiers."""

MemoryKey: TypeAlias = str
"""Memory retrieval key (usually entity or concept name)."""


# ============================================================================
# HEALTH & STATUS TYPES
# ============================================================================

HealthStatus: TypeAlias = Literal["healthy", "unhealthy", "degraded", "disabled", "unknown"]
"""Service health status values."""

ServiceName: TypeAlias = Literal[
    "redis",
    "neo4j",
    "qdrant",
    "postgres",
    "weather_mcp",
    "hurricane_mcp",
    "prometheus",
    "grafana",
    "loki",
]
"""Monitored service names in the infrastructure."""


# ============================================================================
# DICTIONARY TYPES (for complex nested structures)
# ============================================================================

JSONDict: TypeAlias = dict[str, Any]
"""Generic JSON dictionary type."""

ConfigDict: TypeAlias = dict[str, Any]
"""Configuration dictionary type."""

MetadataDict: TypeAlias = dict[str, Any]
"""Metadata dictionary type."""


# ============================================================================
# VALIDATION TYPES
# ============================================================================

ConfidenceScore: TypeAlias = float
"""Confidence score between 0.0 and 1.0."""

SimilarityScore: TypeAlias = float
"""Similarity score between 0.0 and 1.0 (for semantic search)."""

RelevanceScore: TypeAlias = float
"""Relevance score between 0.0 and 1.0 (for RAG retrieval)."""
