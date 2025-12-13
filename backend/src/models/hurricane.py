"""Hurricane Alert and Approval Models

This module defines Pydantic models for hurricane alert workflow with HITL approval.

Level 1 Implementation:
- HurricaneAlertRequest: Request to create hurricane alert
- HurricaneAlertResponse: Response with alert status (sent/pending/cancelled)
- HurricaneApprovalRequest: Human approval decision
- HurricaneApprovalResponse: Final status after approval

Level 2 Enhancements:
- Added timestamp to HurricaneAlertResponse for tracking

Future Levels:
- Level 3+: Add Saffir-Simpson scale validation (category must match wind speed)
- Level 5+: Add evacuation zone validation and life-safety checks
"""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone
import uuid


class HurricaneAlertRequest(BaseModel):
    """Request model for creating hurricane alert.

    HITL Workflow:
    - Category 1-2: Auto-approved, sent immediately
    - Category 3-5: Requires human approval before sending

    Test Scenarios (from test guide):
    - Scenario 28.1: Cat 1-2 (no HITL)
    - Scenario 28.2: Cat 3-4 (HITL triggered)
    - Scenario 28.3: Cat 5 (maximum alert)
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "category": 2,
                    "message": "Category 2 Hurricane Julia approaching Florida with 95 mph winds. Monitor conditions.",
                    "thread_id": "alert-cat2-001"
                },
                {
                    "category": 4,
                    "message": "Category 4 Hurricane Milton approaching Tampa Bay with 140 mph winds. Evacuation recommended for zones A and B.",
                    "thread_id": "alert-cat4-001"
                },
                {
                    "category": 5,
                    "message": "EXTREME DANGER: Category 5 Hurricane approaching with 165 mph winds. IMMEDIATE evacuation required for all coastal zones.",
                    "thread_id": "alert-cat5-001"
                }
            ]
        }
    }

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

    Used to approve or reject Category 3+ hurricane alerts pending human review.

    Test Scenarios (from test guide):
    - Scenario 28.4: Approve (approved: true)
    - Edge case: Reject (approved: false)
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "approved": True
                },
                {
                    "approved": False
                }
            ]
        }
    }

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
