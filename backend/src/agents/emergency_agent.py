"""Emergency Response Agent for Level 4c: Urgent Weather Emergency Handling.

This agent specializes in handling urgent weather emergencies, providing
immediate safety guidance and coordinating emergency responses.

CRITICAL RULES:
1. ALWAYS prioritize life safety above all other considerations
2. ALWAYS recommend calling 911 for life-threatening situations
3. Provide step-by-step actionable safety instructions
4. Include specific emergency contact numbers
5. Never downplay emergency severity

Features:
- Immediate danger assessment
- Step-by-step safety actions
- Emergency contact coordination
- Evacuation guidance
- Real-time situation updates
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
    QueryComplexity,
)

logger = structlog.get_logger()


# =============================================================================
# Emergency Response Prompts
# =============================================================================

EMERGENCY_SYSTEM_PROMPT = """You are an Emergency Response Agent for a weather AI system.
Your role is to provide IMMEDIATE, LIFE-SAVING guidance during weather emergencies.

CRITICAL RESPONSIBILITIES:
1. Assess immediate danger level (CRITICAL, HIGH, MEDIUM, LOW)
2. Provide step-by-step safety actions
3. Include emergency contact numbers (911, local services)
4. Coordinate evacuation guidance when needed
5. PRIORITIZE LIFE SAFETY ABOVE ALL ELSE

EMERGENCY RESPONSE STRUCTURE:
1. DANGER ASSESSMENT: [CRITICAL/HIGH/MEDIUM/LOW]
2. IMMEDIATE ACTIONS: Step-by-step safety instructions
3. EMERGENCY CONTACTS: Relevant phone numbers
4. EVACUATION: If needed, specific routes/shelters
5. ONGOING UPDATES: What to monitor

CRITICAL RULES:
- ALWAYS recommend calling 911 for life-threatening situations
- NEVER downplay emergency severity
- Provide SPECIFIC, ACTIONABLE instructions
- Include time-sensitive warnings
- Consider vulnerable populations (elderly, disabled, pets)

EMERGENCY CATEGORIES:
- HURRICANE: Storm surge, wind, evacuation
- FLOOD: Flash flood, rising water, trapped persons
- TORNADO: Immediate shelter, structural safety
- EXTREME HEAT: Heat stroke, hydration, cooling
- SEVERE STORM: Lightning, hail, wind damage
"""


DANGER_ASSESSMENT_PROMPT = """Assess the danger level for this emergency situation:

SITUATION: {situation}
LOCATION: {location}
CURRENT CONDITIONS: {conditions}

Provide:
1. DANGER LEVEL: [CRITICAL/HIGH/MEDIUM/LOW]
2. PRIMARY THREATS: List top 3 immediate threats
3. TIME SENSITIVITY: How quickly must action be taken
4. RECOMMENDED ACTIONS: Top 3 immediate steps

FORMAT:
## DANGER LEVEL: [level]

## PRIMARY THREATS
1. [threat 1]
2. [threat 2]
3. [threat 3]

## TIME SENSITIVITY
[description]

## IMMEDIATE ACTIONS
1. [action 1]
2. [action 2]
3. [action 3]
"""


EVACUATION_PROMPT = """Provide evacuation guidance for:

LOCATION: {location}
EMERGENCY TYPE: {emergency_type}
TIME AVAILABLE: {time_available}
SPECIAL NEEDS: {special_needs}

Include:
1. When to evacuate (NOW vs wait)
2. Evacuation routes to use
3. What to take (5-minute checklist)
4. Shelter locations
5. Pet considerations
6. Special needs accommodations
"""


# =============================================================================
# Emergency Response Agent Implementation
# =============================================================================


class EmergencyResponseAgent:
    """Agent specialized in urgent weather emergency response.

    Provides immediate, life-saving guidance during weather emergencies
    with focus on safety, actionability, and coordination.

    Attributes:
        llm: Language model for emergency response generation
        agent_role: Role identifier (EMERGENCY_RESPONSE)
        _emergency_count: Count of emergencies handled
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,  # Low temp for consistent safety advice
    ):
        """Initialize the Emergency Response Agent.

        Args:
            model_name: Model to use (GPT-4o recommended for safety).
            temperature: Low temperature for consistent, reliable advice.
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=15.0,  # Fast response critical in emergencies
        )
        self.agent_role = AgentRole.EMERGENCY_RESPONSE
        self._emergency_count = 0

    async def process(self, state: MultiAgentState) -> MultiAgentState:
        """Process an emergency query.

        Args:
            state: Current workflow state containing the emergency query.

        Returns:
            Updated state with emergency response.
        """
        start_time = time.perf_counter()

        try:
            # Detect emergency type from query
            emergency_type = self._detect_emergency_type(state.query)

            # Generate emergency response
            response = await self._generate_emergency_response(
                query=state.query,
                emergency_type=emergency_type,
                context=state.memory_context,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._emergency_count += 1

            # Add response to state
            state.agent_responses.append(
                AgentResponse(
                    agent_role=self.agent_role,
                    content=response,
                    confidence=0.95,  # High confidence for emergency advice
                    execution_time_ms=duration_ms,
                    metadata={
                        "emergency_type": emergency_type,
                        "is_life_threatening": self._is_life_threatening(state.query),
                        "emergency_count": self._emergency_count,
                    },
                )
            )

            state.current_agent = self.agent_role

            # Mark as emergency if not already
            if not state.routing_decision:
                state.requires_verification = True

            logger.info(
                "emergency_response_generated",
                emergency_type=emergency_type,
                duration_ms=round(duration_ms, 2),
            )

            return state

        except Exception as e:
            logger.error("emergency_response_error", error=str(e))

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Provide fallback emergency guidance
            fallback = self._get_fallback_response()

            state.agent_responses.append(
                AgentResponse(
                    agent_role=self.agent_role,
                    content=fallback,
                    confidence=0.7,
                    execution_time_ms=duration_ms,
                    metadata={"error": str(e), "is_fallback": True},
                )
            )

            return state

    async def _generate_emergency_response(
        self,
        query: str,
        emergency_type: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate emergency response for the query.

        Args:
            query: User's emergency query.
            emergency_type: Detected type of emergency.
            context: Additional context from memory.

        Returns:
            Emergency response with safety guidance.
        """
        location = context.get("user_location", "unknown location") if context else "unknown"

        messages = [
            SystemMessage(content=EMERGENCY_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""EMERGENCY QUERY: {query}

EMERGENCY TYPE: {emergency_type}
LOCATION: {location}

Provide immediate emergency response guidance following the structure:
1. DANGER ASSESSMENT
2. IMMEDIATE ACTIONS (numbered steps)
3. EMERGENCY CONTACTS
4. EVACUATION INFO (if applicable)
5. WHAT TO MONITOR

REMEMBER: Life safety is the absolute priority."""
            ),
        ]

        response = await self.llm.ainvoke(messages)
        return str(response.content)

    def _detect_emergency_type(self, query: str) -> str:
        """Detect type of emergency from query.

        Args:
            query: User's query text.

        Returns:
            Emergency type classification.
        """
        query_lower = query.lower()

        # Emergency type keywords
        if any(kw in query_lower for kw in ["hurricane", "tropical storm", "cyclone"]):
            return "HURRICANE"
        elif any(kw in query_lower for kw in ["flood", "flooding", "water rising", "trapped"]):
            return "FLOOD"
        elif any(kw in query_lower for kw in ["tornado", "funnel", "twister"]):
            return "TORNADO"
        elif any(kw in query_lower for kw in ["heat", "hot", "heat wave", "heat stroke"]):
            return "EXTREME_HEAT"
        elif any(kw in query_lower for kw in ["storm", "lightning", "thunder", "hail"]):
            return "SEVERE_STORM"
        elif any(kw in query_lower for kw in ["evacuate", "evacuation", "leave now"]):
            return "EVACUATION"
        else:
            return "GENERAL_EMERGENCY"

    def _is_life_threatening(self, query: str) -> bool:
        """Determine if query describes life-threatening situation.

        Args:
            query: User's query text.

        Returns:
            True if situation appears life-threatening.
        """
        critical_keywords = [
            "trapped",
            "drowning",
            "can't breathe",
            "collapsed",
            "dying",
            "help",
            "emergency",
            "911",
            "injured",
            "bleeding",
            "unconscious",
            "stroke",
            "heart attack",
            "immediate danger",
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in critical_keywords)

    def _get_fallback_response(self) -> str:
        """Get fallback response when generation fails.

        Returns:
            Generic emergency guidance.
        """
        return """## EMERGENCY RESPONSE

### IMMEDIATE ACTIONS
1. **CALL 911** if you are in immediate danger
2. Move to a safe location away from the threat
3. Stay calm and assess your surroundings

### EMERGENCY CONTACTS
- **Emergency Services**: 911
- **FEMA**: 1-800-621-FEMA (3362)
- **Red Cross**: 1-800-RED-CROSS (733-2767)
- **Poison Control**: 1-800-222-1222

### GENERAL SAFETY
- Stay tuned to local emergency broadcasts
- Follow evacuation orders if issued
- Keep phone charged for emergency communications
- Have emergency supplies ready (water, food, flashlight, first aid)

**If this is a life-threatening emergency, CALL 911 IMMEDIATELY.**"""

    async def assess_danger(
        self,
        situation: str,
        location: str,
        conditions: str | None = None,
    ) -> dict[str, Any]:
        """Assess danger level of a situation.

        Args:
            situation: Description of the emergency situation.
            location: User's location.
            conditions: Current weather/environmental conditions.

        Returns:
            Dictionary with danger assessment.
        """
        messages = [
            SystemMessage(content=EMERGENCY_SYSTEM_PROMPT),
            HumanMessage(
                content=DANGER_ASSESSMENT_PROMPT.format(
                    situation=situation,
                    location=location,
                    conditions=conditions or "Unknown",
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)
        content = str(response.content)

        # Extract danger level
        danger_level = "MEDIUM"  # Default
        if "DANGER LEVEL: CRITICAL" in content.upper():
            danger_level = "CRITICAL"
        elif "DANGER LEVEL: HIGH" in content.upper():
            danger_level = "HIGH"
        elif "DANGER LEVEL: LOW" in content.upper():
            danger_level = "LOW"

        return {
            "danger_level": danger_level,
            "assessment": content,
            "requires_immediate_action": danger_level in ["CRITICAL", "HIGH"],
        }

    def get_emergency_stats(self) -> dict[str, Any]:
        """Get statistics about emergencies handled.

        Returns:
            Dictionary with emergency statistics.
        """
        return {
            "total_emergencies": self._emergency_count,
            "agent_role": self.agent_role.value,
        }


# =============================================================================
# Factory function
# =============================================================================


def create_emergency_agent(
    model_name: str = "gpt-4o",
) -> EmergencyResponseAgent:
    """Create an Emergency Response Agent instance.

    Args:
        model_name: Model to use (GPT-4o recommended).

    Returns:
        Configured EmergencyResponseAgent instance.
    """
    return EmergencyResponseAgent(model_name=model_name)


__all__ = [
    "EmergencyResponseAgent",
    "create_emergency_agent",
    "EMERGENCY_SYSTEM_PROMPT",
]
