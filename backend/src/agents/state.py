"""Agent state definitions for Weather AI Agent.

This module defines state classes for agent state management using LangGraph v1.x Annotation.

Level 1 Implementation:
- Simple state with messages support
- LangGraph v1.x Annotation (required for StateGraph)
- Message reducer for agent communication
- NO memory (deferred to L3a)
- NO conversation history (deferred to L3a)
- NO user profiles (deferred to L3a)

Level 3+ Enhancements:
- Memory layers (short-term Redis, long-term Graphiti)
- Conversation history
- User profiles and preferences
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class WeatherAgentState(TypedDict):
    """Level 1 agent state - LangGraph v1.x compliant with message support.

    This state definition uses LangGraph v1.x Annotation pattern for proper
    StateGraph integration, Studio visualization, and checkpointing.

    Attributes:
        messages: Conversation messages with automatic add_messages reducer
        user_id: Unique identifier for the user
        session_id: Unique identifier for the current session
        current_query: The user's current weather query
        current_step: Current workflow step (for routing)
        approved: Whether hurricane alert has been approved (HITL)
        hurricane_category: Hurricane category (1-5) if detected, None otherwise
        alert_message: Hurricane alert message if applicable

    Example:
        >>> state: WeatherAgentState = {
        ...     "messages": [],
        ...     "user_id": "user123",
        ...     "session_id": "session456",
        ...     "current_query": "Weather in Miami?",
        ...     "current_step": "agent",
        ...     "approved": False,
        ...     "hurricane_category": None,
        ...     "alert_message": None,
        ... }
    """

    # Messages with reducer (LangGraph v1.x pattern)
    messages: Annotated[list[BaseMessage], add_messages]

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
