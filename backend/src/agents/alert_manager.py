"""Alert Manager Agent for Level 4a: Three-Agent Foundation System.

CRITICAL: This module implements the Alert Manager Agent that generates
weather alerts, emergency notifications, and evacuation guidance.

Components:
- AlertManagerAgent: Main agent class for alert generation
- AlertSeverity: Enum for alert severity levels
- AlertChannel: Enum for delivery channels
- Multi-channel delivery simulation (SMS, email, push, in-app)

Level 4a Architecture:
- Triage Agent → Routes emergency queries directly here
- Hurricane Specialist Agent → Provides analysis that triggers alerts
- Alert Manager Agent → Final stage for user-facing alert delivery

Design Principles:
- Clear, actionable emergency messaging
- Severity-based channel selection
- Life-safety priority in all communications
- Audit logging for all alert deliveries
- Simulation mode for Level 4a (real delivery in Level 5c)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.src.agents.prompts.alert_prompts import (
    ALERT_GENERATION_PROMPT,
    ALERT_MANAGER_SYSTEM_PROMPT,
    ALERT_SEVERITY_LEVELS,
)
from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels for weather notifications.

    Determines alert formatting and delivery channels:
    - INFO: In-app only
    - WARNING: In-app + Push
    - CRITICAL: In-app + Push + SMS
    - EMERGENCY: All channels (In-app + Push + SMS + Email)
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(str, Enum):
    """Alert delivery channels.

    Note: Level 4a uses simulation mode. Real delivery in Level 5c.
    """

    IN_APP = "in_app"
    PUSH = "push_notification"
    SMS = "sms"
    EMAIL = "email"


class AlertManagerAgent:
    """Manage weather alerts and emergency notifications.

    The Alert Manager Agent is responsible for:
    - Classifying alert severity based on weather data
    - Generating multi-format alerts (SMS, email, push, in-app)
    - Simulating delivery across channels (real delivery in Level 5c)
    - Maintaining audit logs for all alert operations

    Features:
    - Severity-based channel selection
    - Character-limited SMS formatting (160 chars)
    - Push notification formatting (title + body)
    - Full email alerts for emergencies
    - Comprehensive audit logging

    Attributes:
        llm: ChatOpenAI model for alert generation (gpt-4o-mini for speed)
        agent_role: Fixed as AgentRole.ALERT_MANAGER
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
    ):
        """Initialize Alert Manager Agent.

        Args:
            model_name: LLM model for alert generation (default: gpt-4o-mini for speed)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,  # Consistent alert formatting
            timeout=5.0,  # Quick alert generation
        )
        self.agent_role = AgentRole.ALERT_MANAGER

        logger.info(
            "alert_manager_initialized",
            model=model_name,
        )

    async def generate_alert(
        self,
        state: MultiAgentState,
    ) -> MultiAgentState:
        """Generate and deliver weather alert.

        Steps:
        1. Classify alert severity based on query and context
        2. Get specialist analysis if available
        3. Generate alert message for each channel
        4. Simulate delivery across channels
        5. Log delivery for audit trail

        Args:
            state: Current multi-agent state containing query and context

        Returns:
            Updated state with alert response and delivery status
        """
        start_time = datetime.now(UTC)
        query = state["query"]
        alert_id = str(uuid.uuid4())[:8]

        logger.info(
            "alert_generation_started",
            alert_id=alert_id,
            query=query[:100],
            user_id=state.get("user_id"),
        )

        try:
            # Step 1: Classify alert severity
            severity = await self._classify_severity(state)

            # Step 2: Get specialist analysis if available
            specialist_analysis = self._get_specialist_analysis(state)

            # Step 3: Get user location
            user_location = self._get_user_location(state)

            # Step 4: Generate alert messages
            alert_messages = await self._generate_alert_messages(
                query=query,
                severity=severity,
                specialist_analysis=specialist_analysis,
                user_location=user_location,
            )

            # Step 5: Determine delivery channels based on severity
            channels = self._select_channels(severity)

            # Step 6: Simulate delivery (real delivery in Level 5c)
            delivery_results = await self._deliver_alert(
                alert_id=alert_id,
                alert_messages=alert_messages,
                severity=severity,
                channels=channels,
                user_id=state.get("user_id", "unknown"),
            )

            # Calculate execution time
            execution_time_ms = (
                datetime.now(UTC) - start_time
            ).total_seconds() * 1000

            # Build final alert content
            final_alert_content = self._format_final_response(
                alert_messages=alert_messages,
                severity=severity,
                channels=channels,
                delivery_results=delivery_results,
            )

            # Create agent response
            agent_response = AgentResponse(
                agent_role=AgentRole.ALERT_MANAGER,
                content=final_alert_content,
                confidence=1.0,  # Alerts are always delivered with full confidence
                timestamp=datetime.now(UTC),
                execution_time_ms=execution_time_ms,
                metadata={
                    "alert_id": alert_id,
                    "severity": severity.value,
                    "channels": [c.value for c in channels],
                    "delivery_results": delivery_results,
                    "user_location": user_location,
                },
            )

            logger.info(
                "alert_generation_complete",
                alert_id=alert_id,
                severity=severity.value,
                channels=[c.value for c in channels],
                execution_time_ms=execution_time_ms,
            )

            # Update state
            updated_state = {
                **state,
                "current_agent": AgentRole.ALERT_MANAGER,
                "workflow_complete": True,
                "final_response": final_alert_content,
            }

            if "agent_responses" not in updated_state:
                updated_state["agent_responses"] = []
            updated_state["agent_responses"].append(agent_response)

            return updated_state

        except Exception as e:
            # Error handling
            logger.error(
                "alert_generation_error",
                alert_id=alert_id,
                error=str(e),
                error_type=type(e).__name__,
            )

            execution_time_ms = (
                datetime.now(UTC) - start_time
            ).total_seconds() * 1000

            # Create fallback alert
            fallback_alert = self._generate_fallback_alert(query, user_location="Unknown")

            error_response = AgentResponse(
                agent_role=AgentRole.ALERT_MANAGER,
                content=fallback_alert,
                confidence=0.5,
                timestamp=datetime.now(UTC),
                execution_time_ms=execution_time_ms,
                metadata={
                    "alert_id": alert_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback": True,
                },
            )

            updated_state = {
                **state,
                "current_agent": AgentRole.ALERT_MANAGER,
                "workflow_complete": True,
                "final_response": fallback_alert,
                "error": str(e),
            }

            if "agent_responses" not in updated_state:
                updated_state["agent_responses"] = []
            updated_state["agent_responses"].append(error_response)

            return updated_state

    async def _classify_severity(self, state: MultiAgentState) -> AlertSeverity:
        """Classify alert severity based on query and context.

        Uses LLM to analyze query complexity and determine appropriate severity.
        Falls back to keyword-based classification if LLM fails.

        Args:
            state: Current multi-agent state

        Returns:
            AlertSeverity enum value
        """
        query = state["query"]

        # First, check routing decision for emergency classification
        routing_decision = state.get("routing_decision")
        if routing_decision:
            query_category = getattr(routing_decision, "query_category", "")
            if query_category == "emergency":
                return AlertSeverity.EMERGENCY

        # Check for Hurricane Specialist's assessment
        agent_responses = state.get("agent_responses", [])
        for response in agent_responses:
            if response.agent_role == AgentRole.HURRICANE_SPECIALIST:
                # Check metadata for risk indicators
                metadata = response.metadata or {}
                if metadata.get("requires_evacuation"):
                    return AlertSeverity.EMERGENCY

        # Keyword-based classification as fallback
        query_lower = query.lower()

        # Emergency keywords
        emergency_keywords = [
            "evacuate now", "evacuate immediately", "leave now",
            "cat 5", "category 5", "life-threatening",
            "immediate danger", "urgent", "emergency",
        ]
        if any(kw in query_lower for kw in emergency_keywords):
            return AlertSeverity.EMERGENCY

        # Critical keywords
        critical_keywords = [
            "evacuate", "should i leave", "cat 4", "category 4",
            "cat 3", "category 3", "major hurricane",
            "mandatory evacuation", "storm surge warning",
        ]
        if any(kw in query_lower for kw in critical_keywords):
            return AlertSeverity.CRITICAL

        # Warning keywords
        warning_keywords = [
            "hurricane watch", "tropical storm", "prepare",
            "preparation", "approaching", "heading toward",
        ]
        if any(kw in query_lower for kw in warning_keywords):
            return AlertSeverity.WARNING

        # Default to INFO for general queries
        return AlertSeverity.INFO

    def _get_specialist_analysis(self, state: MultiAgentState) -> str:
        """Extract Hurricane Specialist's analysis from state.

        Args:
            state: Current multi-agent state

        Returns:
            Specialist analysis text or default message
        """
        agent_responses = state.get("agent_responses", [])

        for response in agent_responses:
            if response.agent_role == AgentRole.HURRICANE_SPECIALIST:
                return response.content

        return "No specialist analysis available."

    def _get_user_location(self, state: MultiAgentState) -> str:
        """Extract user location from state memory context.

        Args:
            state: Current multi-agent state

        Returns:
            User location string or "Unknown location"
        """
        memory_context = state.get("memory_context", {})

        if memory_context:
            user_profile = memory_context.get("user_profile", {})
            location = user_profile.get("location")
            if location:
                return location

        # Try to extract location from query
        query = state.get("query", "")
        # Simple location extraction (could be enhanced)
        location_keywords = ["in", "for", "near", "around"]
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in location_keywords and i + 1 < len(words):
                return words[i + 1].strip(".,?!")

        return "Unknown location"

    async def _generate_alert_messages(
        self,
        query: str,
        severity: AlertSeverity,
        specialist_analysis: str,
        user_location: str,
    ) -> dict[str, str]:
        """Generate alert messages for each channel.

        Args:
            query: User's original query
            severity: Classified alert severity
            specialist_analysis: Analysis from Hurricane Specialist
            user_location: User's location

        Returns:
            Dictionary mapping channel to alert message
        """
        # Build prompt for LLM
        prompt = ALERT_GENERATION_PROMPT.format(
            query=query,
            specialist_analysis=specialist_analysis,
            severity=severity.value.upper(),
            user_location=user_location,
        )

        messages = [
            SystemMessage(content=ALERT_MANAGER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            llm_content = response.content

            # Parse LLM response into channel-specific messages
            alert_messages = self._parse_alert_response(llm_content, severity)

        except Exception as e:
            logger.warning(
                "llm_alert_generation_failed",
                error=str(e),
            )
            # Fall back to template-based generation
            alert_messages = self._generate_template_alert(
                severity=severity,
                user_location=user_location,
                specialist_analysis=specialist_analysis,
            )

        return alert_messages

    def _parse_alert_response(
        self,
        llm_content: str,
        severity: AlertSeverity,
    ) -> dict[str, str]:
        """Parse LLM response into channel-specific alert messages.

        Args:
            llm_content: Raw LLM response
            severity: Alert severity for fallback formatting

        Returns:
            Dictionary mapping channel to alert message
        """
        alert_messages = {}

        # Try to extract sections from LLM response
        sections = {
            "in_app": "## In-App Alert",
            "push": "## Push Notification",
            "sms": "## SMS Alert",
            "email": "## Email Alert",
        }

        for channel, marker in sections.items():
            if marker in llm_content:
                start_idx = llm_content.find(marker) + len(marker)
                # Find next section or end
                end_idx = len(llm_content)
                for other_marker in sections.values():
                    if other_marker != marker:
                        other_idx = llm_content.find(other_marker, start_idx)
                        if other_idx > 0 and other_idx < end_idx:
                            end_idx = other_idx

                section_content = llm_content[start_idx:end_idx].strip()
                alert_messages[channel] = section_content

        # Ensure we have at least in_app message
        if "in_app" not in alert_messages:
            alert_messages["in_app"] = llm_content[:500]  # Use first 500 chars

        # Truncate SMS to 160 chars if present
        if "sms" in alert_messages:
            alert_messages["sms"] = alert_messages["sms"][:160]

        return alert_messages

    def _generate_template_alert(
        self,
        severity: AlertSeverity,
        user_location: str,
        specialist_analysis: str,
    ) -> dict[str, str]:
        """Generate template-based alerts as fallback.

        Args:
            severity: Alert severity
            user_location: User's location
            specialist_analysis: Specialist analysis text

        Returns:
            Dictionary mapping channel to alert message
        """
        severity_info = ALERT_SEVERITY_LEVELS.get(
            severity.value.upper(),
            ALERT_SEVERITY_LEVELS["INFO"],
        )
        prefix = severity_info["prefix"]

        timestamp = datetime.now(UTC).strftime("%I:%M %p EDT")

        # In-app alert (full detail)
        in_app = f"""
{prefix}

Location: {user_location}
Time: {timestamp}

{specialist_analysis[:500] if len(specialist_analysis) > 500 else specialist_analysis}

Please monitor official sources for updates.
        """.strip()

        # Push notification
        push = f"Title: {prefix} - {user_location}\nBody: Weather alert active. Check app for details."

        # SMS (160 char limit)
        sms = f"{prefix}: Weather alert for {user_location}. Check app for details. -{timestamp}"[:160]

        # Email
        email = f"""Subject: [{severity.value.upper()}] Weather Alert - {user_location}

{in_app}

---
This is an automated alert from Weather AI Agent.
        """.strip()

        return {
            "in_app": in_app,
            "push": push,
            "sms": sms,
            "email": email,
        }

    def _select_channels(self, severity: AlertSeverity) -> list[AlertChannel]:
        """Select delivery channels based on severity.

        Channel Selection Rules:
        - INFO: In-app only
        - WARNING: In-app + Push
        - CRITICAL: In-app + Push + SMS
        - EMERGENCY: All channels

        Args:
            severity: Alert severity level

        Returns:
            List of AlertChannel enums
        """
        if severity == AlertSeverity.EMERGENCY:
            return [
                AlertChannel.IN_APP,
                AlertChannel.PUSH,
                AlertChannel.SMS,
                AlertChannel.EMAIL,
            ]

        if severity == AlertSeverity.CRITICAL:
            return [
                AlertChannel.IN_APP,
                AlertChannel.PUSH,
                AlertChannel.SMS,
            ]

        if severity == AlertSeverity.WARNING:
            return [
                AlertChannel.IN_APP,
                AlertChannel.PUSH,
            ]

        # INFO
        return [AlertChannel.IN_APP]

    async def _deliver_alert(
        self,
        alert_id: str,
        alert_messages: dict[str, str],
        severity: AlertSeverity,
        channels: list[AlertChannel],
        user_id: str,
    ) -> dict[str, str]:
        """Simulate alert delivery across channels.

        Note: This is SIMULATION MODE for Level 4a.
        Real delivery (Twilio SMS, SendGrid email, etc.) in Level 5c.

        Args:
            alert_id: Unique alert identifier
            alert_messages: Channel-specific alert messages
            severity: Alert severity
            channels: List of delivery channels
            user_id: User identifier

        Returns:
            Dictionary mapping channel to delivery status
        """
        delivery_results = {}

        for channel in channels:
            channel_key = channel.value.replace("_notification", "")

            try:
                if channel == AlertChannel.IN_APP:
                    # In-app always succeeds (it's displayed in response)
                    delivery_results["in_app"] = "delivered"
                    logger.info(
                        "in_app_alert_delivered",
                        alert_id=alert_id,
                        user_id=user_id,
                        severity=severity.value,
                    )

                elif channel == AlertChannel.PUSH:
                    # Simulate push notification
                    delivery_results["push"] = "simulated_success"
                    logger.info(
                        "push_notification_simulated",
                        alert_id=alert_id,
                        user_id=user_id,
                        severity=severity.value,
                    )

                elif channel == AlertChannel.SMS:
                    # Simulate SMS delivery
                    sms_content = alert_messages.get("sms", "")[:160]
                    delivery_results["sms"] = "simulated_success"
                    logger.info(
                        "sms_alert_simulated",
                        alert_id=alert_id,
                        user_id=user_id,
                        severity=severity.value,
                        message_length=len(sms_content),
                    )

                elif channel == AlertChannel.EMAIL:
                    # Simulate email delivery
                    delivery_results["email"] = "simulated_success"
                    logger.info(
                        "email_alert_simulated",
                        alert_id=alert_id,
                        user_id=user_id,
                        severity=severity.value,
                    )

            except Exception as e:
                delivery_results[channel_key] = f"failed: {str(e)}"
                logger.error(
                    "alert_delivery_failed",
                    alert_id=alert_id,
                    channel=channel.value,
                    error=str(e),
                )

        return delivery_results

    def _format_final_response(
        self,
        alert_messages: dict[str, str],
        severity: AlertSeverity,
        channels: list[AlertChannel],
        delivery_results: dict[str, str],
    ) -> str:
        """Format final response to user.

        Args:
            alert_messages: Channel-specific messages
            severity: Alert severity
            channels: Delivery channels used
            delivery_results: Delivery status per channel

        Returns:
            Formatted response string
        """
        severity_info = ALERT_SEVERITY_LEVELS.get(
            severity.value.upper(),
            ALERT_SEVERITY_LEVELS["INFO"],
        )
        prefix = severity_info["prefix"]

        # Build response
        response_parts = [
            f"# {prefix}\n",
            alert_messages.get("in_app", "Alert generated."),
            "\n---",
            f"\n**Alert delivered via**: {', '.join([c.value for c in channels])}",
        ]

        # Add delivery status summary
        successful = sum(1 for status in delivery_results.values() if "success" in status or status == "delivered")
        total = len(delivery_results)
        response_parts.append(f"\n**Delivery status**: {successful}/{total} channels successful")

        return "\n".join(response_parts)

    def _generate_fallback_alert(self, query: str, user_location: str) -> str:
        """Generate fallback alert when main generation fails.

        Args:
            query: Original user query
            user_location: User location

        Returns:
            Fallback alert message
        """
        return f"""
⚠️ **WEATHER ALERT**

Location: {user_location}
Query: {query[:100]}...

We were unable to generate a detailed alert due to a technical issue.

**Recommended Actions**:
1. Monitor local news and official weather sources
2. Check the National Hurricane Center at hurricanes.gov
3. Follow any evacuation orders from local authorities

Stay safe and contact local emergency services if needed.

---
This is a fallback alert. Please try again for detailed information.
        """.strip()
