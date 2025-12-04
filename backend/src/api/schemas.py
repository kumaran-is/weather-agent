"""Pydantic schemas for FastAPI request/response models.

This module defines all request and response models for the Weather AI Agent API.

Level 1 Implementation:
- Basic request/response models
- Pydantic v2 validation
- Clear field descriptions for API documentation
- NO advanced validation (deferred to L2+)
"""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone
import uuid


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


class HurricaneAlertRequest(BaseModel):
    """Request model for creating hurricane alert.

    Example:
        {
            "category": 4,
            "message": "Category 4 Hurricane Ida approaching with 140 mph winds",
            "thread_id": "alert-uuid"
        }
    """

    category: int = Field(
        ...,
        ge=1,
        le=5,
        description="Hurricane category (1-5) on Saffir-Simpson scale"
    )
    message: str = Field(
        ...,
        description="Hurricane alert message with details",
        min_length=10,
        max_length=1000
    )
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique thread ID for tracking this alert workflow"
    )


class HurricaneAlertResponse(BaseModel):
    """Response model for hurricane alert creation.

    Example:
        {
            "status": "pending_approval",
            "thread_id": "alert-uuid",
            "category": 4,
            "message": "Category 4 Hurricane approaching",
            "timestamp": "2025-12-04T01:32:40.331100+00:00"
        }
    """

    status: Literal["sent", "cancelled", "pending_approval"] = Field(
        ...,
        description="Status of the alert: sent (approved), cancelled (rejected), or pending_approval (awaiting human)"
    )
    thread_id: str = Field(
        ...,
        description="Thread ID for tracking this alert workflow"
    )
    category: int = Field(
        ...,
        ge=1,
        le=5,
        description="Hurricane category (1-5)"
    )
    message: str = Field(
        ...,
        description="Alert message content"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the alert was created (UTC)"
    )


class HurricaneApprovalRequest(BaseModel):
    """Request model for approving/rejecting hurricane alert.

    Example:
        {
            "approved": true
        }
    """

    approved: bool = Field(
        ...,
        description="True to approve and send alert, False to reject and cancel"
    )


class HurricaneApprovalResponse(BaseModel):
    """Response model for hurricane approval action.

    Example:
        {
            "status": "sent",
            "thread_id": "alert-uuid"
        }
    """

    status: Literal["sent", "cancelled"] = Field(
        ...,
        description="Final status after approval: sent (approved) or cancelled (rejected)"
    )
    thread_id: str = Field(
        ...,
        description="Thread ID of the alert workflow"
    )


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint.

    Example:
        {
            "status": "healthy",
            "level": "1",
            "timestamp": "2025-12-03T16:20:00Z"
        }
    """

    status: Literal["healthy", "unhealthy"] = Field(
        default="healthy",
        description="Health status of the service"
    )
    level: str = Field(
        default="1",
        description="Current implementation level"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp in UTC"
    )
