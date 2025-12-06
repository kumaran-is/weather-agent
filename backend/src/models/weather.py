"""Weather Query and Response Models

This module defines Pydantic models for weather-related API requests and responses.

Level 2 Implementation:
- WeatherQuery: Request model for weather queries
- WeatherResponse: Response model with agent's answer

Future Levels:
- Level 3+: Add structured weather data models (temperature, humidity, etc.)
- Level 5+: Add validation for weather data ranges
"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone


class WeatherQuery(BaseModel):
    """Request model for weather query endpoint.

    Example:
        {
            "query": "What's the weather in London?",
            "user_id": "user123",
            "session_id": "session456"
        }
    """

    query: str = Field(
        ...,
        description="Weather question from the user",
        min_length=1,
        max_length=500,
        examples=["What's the weather in London?", "Will it rain tomorrow in Seattle?"]
    )
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        examples=["user123", "uuid-here"]
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for conversation context",
        examples=["session456", "uuid-here"]
    )


class WeatherResponse(BaseModel):
    """Response model for weather query endpoint.

    Example:
        {
            "response": "The weather in London is 15°C and rainy...",
            "user_id": "user123",
            "timestamp": "2025-12-03T16:20:00Z"
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
