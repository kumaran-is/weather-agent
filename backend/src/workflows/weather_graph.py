"""LangGraph workflow for Level 1 weather agent.

This module defines the LangGraph StateGraph workflow that orchestrates
the weather agent with HITL approval for hurricane alerts.

Level 1 Implementation:
- Simple 4-node workflow
- Linear edges: START → detect → approval → send/cancel → END
- InMemorySaver checkpointer (REQUIRED for HITL)
- NO conditional edges (keep simple for L1)
- NO guardrail nodes (deferred to L5)
- NO memory loading nodes (deferred to L3a)

Workflow Flow:
1. START: Begin workflow
2. detect: Detect hurricane from weather data
3. approval: Human approval for Cat 3+ (auto-approve Cat 1-2)
4. send/cancel: Send approved alert OR cancel rejected alert
5. END: Complete workflow
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from backend.src.agents.state import WeatherAgentState
from backend.src.hitl.approval_node import (
    detect_hurricane_node,
    hurricane_approval,
    send_alert_node,
    cancel_alert_node
)
import logging

logger = logging.getLogger(__name__)


def build_weather_hitl_workflow():
    """Build Weather HITL (Human-in-the-Loop) workflow.

    Creates a 4-node workflow that orchestrates hurricane detection
    and human-in-the-loop approval for safety-critical decisions.

    Workflow Architecture:
        START → detect → approval → send/cancel → END

    Nodes:
        - detect: Detect hurricane from weather data
        - approval: HITL approval node (auto-approve Cat 1-2, require approval Cat 3+)
        - send: Send approved hurricane alert
        - cancel: Cancel rejected hurricane alert

    Returns:
        Compiled LangGraph workflow with InMemorySaver checkpointer

    Example:
        >>> graph = build_weather_hitl_workflow()
        >>> config = {"configurable": {"thread_id": "test-hurricane"}}
        >>>
        >>> # Test Category 2 (auto-approve)
        >>> result = graph.invoke({
        ...     "user_id": "user123",
        ...     "session_id": "session456",
        ...     "current_query": "Category 2 hurricane approaching",
        ...     "current_step": "input",
        ...     "approved": False,
        ...     "hurricane_category": 2,
        ...     "alert_message": "Category 2 Hurricane Julia - 95 mph winds"
        ... }, config)
        >>>
        >>> # Test Category 4 (requires approval)
        >>> result = graph.invoke({
        ...     "user_id": "user123",
        ...     "session_id": "session456",
        ...     "current_query": "Category 4 hurricane approaching",
        ...     "current_step": "input",
        ...     "approved": False,
        ...     "hurricane_category": 4,
        ...     "alert_message": "Category 4 Hurricane Ida - 140 mph winds"
        ... }, config)
        >>>
        >>> if "__interrupt__" in result:
        ...     print("⚠️ HUMAN APPROVAL REQUIRED")
        ...     # Simulate human approval
        ...     from langgraph.types import Command
        ...     final = graph.invoke(
        ...         Command(resume={"approved": True}),
        ...         config
        ...     )

    Note:
        - MUST use InMemorySaver checkpointer for HITL to work
        - Thread ID must be consistent across invoke calls for same workflow
        - For production (L3b+), upgrade to PostgresSaver for persistence
    """
    logger.info("Building Weather HITL workflow...")

    # Create StateGraph with WeatherAgentState
    builder = StateGraph(WeatherAgentState)

    # Add nodes
    logger.debug("Adding workflow nodes...")
    builder.add_node("detect", detect_hurricane_node)
    builder.add_node("approval", hurricane_approval)
    builder.add_node("send", send_alert_node)
    builder.add_node("cancel", cancel_alert_node)

    # Add edges (simple linear flow for Level 1)
    logger.debug("Adding workflow edges...")
    builder.add_edge(START, "detect")
    builder.add_edge("detect", "approval")
    # Note: approval node uses Command to route to send or cancel
    # So we only need edges from send/cancel to END
    builder.add_edge("send", END)
    builder.add_edge("cancel", END)

    # Compile with checkpointer (REQUIRED for HITL)
    logger.debug("Compiling workflow with InMemorySaver checkpointer...")
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Weather HITL workflow built successfully")
    return graph


# Pre-build workflow for convenience (singleton pattern)
# In production, this would be managed by dependency injection
_weather_hitl_workflow = None


def get_weather_hitl_workflow():
    """Get or create Weather HITL workflow (singleton pattern).

    Returns:
        Compiled LangGraph workflow

    Note:
        This creates a singleton workflow instance. For concurrent
        requests, use separate config with different thread_id values.
    """
    global _weather_hitl_workflow
    if _weather_hitl_workflow is None:
        _weather_hitl_workflow = build_weather_hitl_workflow()
    return _weather_hitl_workflow
