"""Weather Query and Response Models

This module defines Pydantic models for weather-related API requests and responses.

PROGRESSIVE ENHANCEMENT PATTERN (from Level 2 learnings):
- Level 2: Basic query model with user tracking
- Level 3a: Add memory support (use_memory, enable_rag, enable_cot)
- Future: Level 3b/3c will add more fields (enable_tot, enable_got)

CRITICAL RULE: ENHANCE existing models, NEVER create versioned models (WeatherQueryV2)
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class WeatherQuery(BaseModel):
    """Request model for weather query endpoint.

    Progressive enhancement across levels:
    - Level 2: query, user_id, session_id
    - Level 3a: + location, enable_rag, enable_cot, use_memory
    - Level 3b: + enable_tot, enable_got

    NOTE: Feature flags (enable_rag, enable_cot, enable_tot, enable_got, use_memory)
    should be set via QUERY PARAMETERS, not in request body.

    Clean example for Swagger UI (use query params for flags):
        {
            "query": "What's the weather in London?",
            "user_id": "user123",
            "session_id": "session456"
        }
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What's the weather in London?",
                    "user_id": "user123",
                    "session_id": "session456"
                }
            ]
        }
    }

    # Core query (Level 2)
    query: str = Field(
        ...,
        description="Weather question from the user",
        min_length=1,
        max_length=500,
        examples=["What's the weather in London?", "Will it rain tomorrow in Seattle?"],
    )

    # User tracking (Level 2)
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        examples=["user123", "uuid-here"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for conversation context",
        examples=["session456", "uuid-here"],
    )

    # Level 3a enhancements (PROGRESSIVE - added fields)
    location: str | None = Field(
        default=None,
        description="Optional explicit location (if not in query)",
        examples=["London", "Seattle, WA", "90210"],
    )
    enable_rag: bool = Field(
        default=True,
        description="Enable RAG for historical weather knowledge retrieval",
    )
    enable_cot: bool = Field(
        default=False,
        description="Enable Chain-of-Thought reasoning for complex queries",
    )
    # Level 3b enhancements (PROGRESSIVE - added fields)
    enable_tot: bool = Field(
        default=False,
        description="Enable Tree of Thoughts reasoning for multi-path exploration",
    )
    enable_got: bool = Field(
        default=False,
        description="Enable Graph of Thoughts reasoning for shared sub-problems",
    )
    use_memory: bool = Field(
        default=True,
        description="Enable memory features (short-term + long-term)",
    )


class WeatherResponse(BaseModel):
    """Response model for weather query endpoint.

    Progressive enhancement across levels:
    - Level 2: response, user_id, timestamp
    - Level 5a: + cache_hit, cache_layer (for observability)

    Example (cache miss):
        {
            "response": "The weather in London is 15°C and rainy...",
            "user_id": "user123",
            "timestamp": "2025-12-03T16:20:00Z",
            "cache_hit": false,
            "cache_layer": null
        }

    Example (L1 cache hit):
        {
            "response": "The weather in London is 15°C and rainy...",
            "user_id": "user123",
            "timestamp": "2025-12-03T16:20:00Z",
            "cache_hit": true,
            "cache_layer": "L1"
        }
    """

    response: str = Field(
        ...,
        description="Agent's response to the weather query"
    )
    user_id: str = Field(
        ...,
        description="User identifier from the request"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp in UTC"
    )

    # 🆕 L5a: Cache observability fields
    cache_hit: bool = Field(
        default=False,
        description="Whether response was served from cache (L1, L2, or L3)",
    )
    cache_layer: str | None = Field(
        default=None,
        description="Cache layer that served the response (L1, L2, or None if cache miss)",
        examples=["L1", "L2", None],
    )
