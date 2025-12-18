"""Hurricane Alert and Approval Models

This module defines Pydantic models for hurricane alert workflow with HITL approval.

Level 1 Implementation:
- HurricaneAlertRequest: Request to create hurricane alert
- HurricaneAlertResponse: Response with alert status (sent/pending/cancelled)
- HurricaneApprovalRequest: Human approval decision
- HurricaneApprovalResponse: Final status after approval

Level 2 Enhancements:
- Added timestamp to HurricaneAlertResponse for tracking

Level 3+ Enhancements (2025-12-14):
- ✅ Added Saffir-Simpson scale validation (category must match wind speed)
- ✅ Added field validator to HurricaneAlertRequest

Future Levels:
- Level 5+: Add evacuation zone validation and life-safety checks
"""

import re
import uuid
from datetime import UTC, datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator


class HurricaneAlertRequest(BaseModel):
    """Request model for creating hurricane alert.

    HITL Workflow:
    - Category 1-2: Auto-approved, sent immediately
    - Category 3-5: Requires human approval before sending

    Saffir-Simpson Scale Validation (CRITICAL):
    - Category 1: 74-95 mph
    - Category 2: 96-110 mph
    - Category 3: 111-129 mph (MAJOR)
    - Category 4: 130-156 mph (MAJOR)
    - Category 5: 157+ mph (CATASTROPHIC)

    Test Scenarios (from test guide):
    - Scenario 28.1: Cat 1-2 (no HITL)
    - Scenario 28.2: Cat 3-4 (HITL triggered)
    - Scenario 28.3: Cat 5 (maximum alert)
    """

    # Saffir-Simpson Scale (wind speeds in mph) - ClassVar to avoid Pydantic field error
    SAFFIR_SIMPSON: ClassVar[dict[int, tuple[int, float]]] = {
        1: (74, 95),
        2: (96, 110),
        3: (111, 129),
        4: (130, 156),
        5: (157, float("inf")),
    }

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

    @field_validator('message')
    @classmethod
    def validate_category_matches_wind_speed(cls, v: str, info) -> str:
        """Validate that stated wind speed matches hurricane category (Saffir-Simpson Scale).

        LIFE-SAFETY CRITICAL: Zero tolerance for category-wind mismatches.

        Args:
            v: Message text
            info: Validation context (contains category field)

        Returns:
            Original message if validation passes

        Raises:
            ValueError: If wind speed doesn't match stated category
        """
        # Get category from model data
        category = info.data.get('category')
        if category is None:
            return v  # Category will be validated by its own Field constraints

        # Extract wind speed from message (if present)
        wind_pattern = r"(\d{2,3})\s*(?:mph|miles per hour)"
        wind_matches = re.findall(wind_pattern, v, re.IGNORECASE)

        if not wind_matches:
            return v  # No wind speed stated, nothing to validate

        # Validate each wind speed mention matches the category
        min_wind, max_wind = cls.SAFFIR_SIMPSON[category]

        for wind_str in wind_matches:
            wind_mph = int(wind_str)

            # Check if wind speed is within category range
            if wind_mph < min_wind:
                raise ValueError(
                    f"Hurricane Category {category} requires {min_wind}-"
                    f"{max_wind if max_wind != float('inf') else '∞'} mph winds, "
                    f"but message states {wind_mph} mph. "
                    f"This violates the Saffir-Simpson Hurricane Wind Scale."
                )
            elif wind_mph > max_wind:
                # Determine correct category
                correct_cat = None
                for cat, (min_w, max_w) in cls.SAFFIR_SIMPSON.items():
                    if min_w <= wind_mph <= max_w:
                        correct_cat = cat
                        break

                raise ValueError(
                    f"Wind speed of {wind_mph} mph indicates Category {correct_cat}, "
                    f"not Category {category}. Message states Category {category} but "
                    f"Category {category} maximum is {max_wind} mph. "
                    f"This violates the Saffir-Simpson Hurricane Wind Scale."
                )

        return v


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
        default_factory=lambda: datetime.now(UTC),
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
