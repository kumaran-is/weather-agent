"""Health Check Models

This module defines Pydantic models for service health check endpoint.

Level 1 Implementation:
- HealthCheckResponse: Health status with current level and timestamp

Future Levels:
- Level 3+: Add memory system health check
- Level 4+: Add multi-agent system health check
- Level 5+: Add comprehensive health metrics (uptime, latency, error rates)
"""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint.

    Example:
        {
            "status": "healthy",
            "level": "2",
            "timestamp": "2025-12-06T16:20:00Z"
        }
    """

    status: Literal["healthy", "unhealthy"] = Field(
        default="healthy",
        description="Health status of the service"
    )
    level: str = Field(
        default="2",
        description="Current implementation level"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp in UTC"
    )
