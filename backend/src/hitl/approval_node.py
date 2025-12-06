"""HITL approval nodes for hurricane alerts.

This module implements Human-in-the-Loop (HITL) approval workflows for
safety-critical hurricane alerts using LangGraph's interrupt and Command patterns.

Level 1 Implementation:
- HITL Pattern 1: Basic Approve/Reject
- Auto-approve: Categories 1-2 (less severe)
- Require approval: Categories 3-5 (severe, life-threatening)
- NO error handling & resilience (deferred to L2)
- NO multi-reviewer thread management (deferred to L3)
- NO confidence-based interrupts (deferred to L2)
- NO edit graph state (deferred to L3)

Safety Requirements:
- Zero tolerance for incorrect evacuation guidance
- Always err on the side of caution
- Category 3+ MUST have human approval
"""

from typing import Literal
from langgraph.types import interrupt, Command
from backend.src.agents.state import WeatherAgentState
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def hurricane_approval(state: WeatherAgentState) -> Command[Literal["send", "cancel"]]:
    """MANDATORY: Human approval for Category 3+ hurricanes.

    This is the core HITL approval node implementing Pattern 1: Basic Approve/Reject.

    Approval Logic:
    - Categories 1-2: Auto-approve (less severe, no evacuation typically needed)
    - Categories 3-5: REQUIRE human approval (life-threatening, evacuation often needed)

    Args:
        state: Current agent state containing hurricane information

    Returns:
        Command directing workflow to either "send" or "cancel" node

    Note:
        This implements HITL Pattern 1 (Basic Approve/Reject).
        More sophisticated patterns (multi-reviewer, confidence-based, etc.)
        will be added in Levels 2-3.
    """
    # Extract hurricane category from state
    category = state.get("hurricane_category")

    # Auto-approve Categories 1-2 (less severe)
    if not category or category < 3:
        logger.info(f"Auto-approving Category {category or 'unknown'} hurricane alert")
        return Command(
            goto="send",
            update={"approved": True}
        )

    # REQUIRE human approval for Category 3+ (life-threatening)
    logger.warning(f"Category {category} hurricane detected - HUMAN APPROVAL REQUIRED")

    # Create interrupt payload with all necessary information
    interrupt_payload = {
        "type": "HURRICANE_ALERT_APPROVAL",
        "priority": "CRITICAL",
        "category": category,
        "message": state.get("alert_message", ""),
        "location": state.get("current_query", "Unknown location"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": state.get("user_id", "unknown"),
        "session_id": state.get("session_id", "unknown")
    }

    # Trigger interrupt - workflow pauses here until human responds
    # The human reviewer will receive the interrupt_payload and must
    # respond with {"approved": True} or {"approved": False}
    decision = interrupt(interrupt_payload)

    # Route based on human decision
    if decision.get("approved"):
        logger.info(f"Human APPROVED Category {category} hurricane alert")
        return Command(
            goto="send",
            update={"approved": True}
        )
    else:
        logger.info(f"Human REJECTED Category {category} hurricane alert")
        return Command(
            goto="cancel",
            update={"approved": False}
        )


def detect_hurricane_node(state: WeatherAgentState) -> dict:
    """Detect hurricane from weather data.

    This is a simplified detection node for Level 1. In production (Level 5+),
    this would integrate with the MCP Hurricane Tracker API for real-time
    hurricane detection and categorization.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with hurricane detection results

    Note:
        Level 1 simplified implementation. Production implementation will:
        - Use MCP Hurricane Tracker API
        - Validate Saffir-Simpson scale compliance
        - Cross-reference with NHC data
        - Include wind speed, pressure, trajectory data
    """
    # For Level 1, we use the category already in state
    # In production, this would call the MCP Hurricane Tracker API
    category = state.get("hurricane_category", 1)
    alert_message = state.get(
        "alert_message",
        f"Category {category} hurricane detected"
    )

    logger.info(f"Hurricane detection: Category {category}")

    return {
        "hurricane_category": category,
        "alert_message": alert_message,
        "current_step": "approval"
    }


def send_alert_node(state: WeatherAgentState) -> dict:
    """Send approved hurricane alert.

    This node is executed when a hurricane alert has been approved
    (either auto-approved for Cat 1-2 or human-approved for Cat 3+).

    In production (Level 5+), this would:
    - Send multi-channel notifications (SMS, email, push, sirens)
    - Log to audit trail
    - Update emergency management systems
    - Trigger evacuation protocols if needed

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with alert status

    Note:
        Level 1 simplified implementation using print statements.
        Production implementation will use proper notification systems.
    """
    category = state.get("hurricane_category", "unknown")
    message = state.get("alert_message", "No message")
    user_id = state.get("user_id", "unknown")

    # Log alert being sent with structured logging
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.critical(
        f"🚨 HURRICANE ALERT SENT | "
        f"Category {category} | "
        f"User: {user_id} | "
        f"Message: {message} | "
        f"Timestamp: {timestamp}"
    )

    # Additional structured logging for visibility (Level 2)
    logger.critical(f"\n{'='*60}")
    logger.critical(f"🚨 HURRICANE ALERT SENT")
    logger.critical(f"{'='*60}")
    logger.critical(f"Category: {category}")
    logger.critical(f"Message: {message}")
    logger.critical(f"User: {user_id}")
    logger.critical(f"Timestamp: {timestamp}")
    logger.critical(f"{'='*60}\n")

    return {
        "status": "sent",
        "current_step": "response",
        "approved": True
    }


def cancel_alert_node(state: WeatherAgentState) -> dict:
    """Cancel rejected hurricane alert.

    This node is executed when a hurricane alert has been rejected by
    a human reviewer. The alert will not be sent to users.

    In production (Level 5+), this would:
    - Log rejection reason to audit trail
    - Notify operations team of cancellation
    - Track false positive patterns for model improvement

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with cancellation status

    Note:
        Level 1 simplified implementation using print statements.
        Production implementation will include proper logging and notifications.
    """
    category = state.get("hurricane_category", "unknown")
    message = state.get("alert_message", "No message")
    user_id = state.get("user_id", "unknown")

    # Log alert cancellation with structured logging
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.warning(
        f"❌ HURRICANE ALERT CANCELLED | "
        f"Category {category} | "
        f"User: {user_id} | "
        f"Message: {message} | "
        f"Reason: Rejected by human reviewer | "
        f"Timestamp: {timestamp}"
    )

    # Additional structured logging for visibility (Level 2)
    logger.warning(f"\n{'='*60}")
    logger.warning(f"❌ HURRICANE ALERT CANCELLED")
    logger.warning(f"{'='*60}")
    logger.warning(f"Category: {category}")
    logger.warning(f"Message: {message}")
    logger.warning(f"User: {user_id}")
    logger.warning(f"Reason: Rejected by human reviewer")
    logger.warning(f"Timestamp: {timestamp}")
    logger.warning(f"{'='*60}\n")

    return {
        "status": "cancelled",
        "current_step": "response",
        "approved": False
    }
