"""Agent state definitions for Weather AI Agent.

This module defines TypedDict classes for agent state management.

Level 1 Implementation:
- Simple state with 7 fields
- NO memory (deferred to L3a)
- NO conversation history (deferred to L3a)
- NO user profiles (deferred to L3a)
"""

from typing import TypedDict, Literal


class WeatherAgentState(TypedDict):
    """Level 1 agent state - SIMPLE, NO MEMORY.

    This is a minimal state definition for Level 1. More sophisticated
    memory and state management will be added in Level 3+.

    Attributes:
        user_id: Unique identifier for the user
        session_id: Unique identifier for the current session
        current_query: The user's current weather query
        current_step: Current workflow step (for routing)
        approved: Whether hurricane alert has been approved (HITL)
        hurricane_category: Hurricane category (1-5) if detected, None otherwise
        alert_message: Hurricane alert message if applicable
    """

    # User context
    user_id: str
    session_id: str

    # Current query
    current_query: str

    # Workflow control
    current_step: Literal["input", "agent", "approval", "response"]

    # HITL (Human-in-the-Loop)
    approved: bool
    hurricane_category: int | None
    alert_message: str | None
