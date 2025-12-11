"""Triage Agent for Level 4a: Three-Agent Foundation System.

CRITICAL: This module implements the Triage Agent that classifies queries and routes
them to appropriate specialist agents based on complexity and urgency.

Components:
- TriageAgent: Main agent class for query classification and routing
- Helper methods for memory integration and JSON parsing
- Fallback logic for error handling

Level 4a Architecture:
- Triage Agent → Analyzes query, routes to specialist
- Hurricane Specialist Agent → Handles hurricane-specific queries
- Alert Manager Agent → Generates and formats user-facing alerts

Design Principles:
- Fast classification with gpt-4o-mini (deterministic, temperature=0.0)
- Confidence-based routing (≥0.8 threshold)
- Memory context integration (user history, preferences, emotional state)
- Graceful fallback to Hurricane Specialist on errors
- Timeout enforcement (max 3s for classification)
- Structured logging for all routing decisions
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.src.models.multi_agent import (
    AgentRole,
    RoutingDecision,
    MultiAgentState,
    AgentResponse,
)
from backend.src.agents.prompts.triage_prompts import (
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_CLASSIFICATION_PROMPT,
    TRIAGE_FALLBACK_PROMPT,
    USER_CONTEXT_TEMPLATE,
)

logger = structlog.get_logger(__name__)


class TriageAgent:
    """Route queries to appropriate specialist agents.

    The Triage Agent is the entry point for all queries in the three-agent system.
    It analyzes the query complexity, urgency, and domain requirements to route
    to the most appropriate specialist agent.

    Routing Logic:
    - SIMPLE queries → Direct Response (no specialist needed)
    - MODERATE queries → Hurricane Specialist (domain expertise)
    - COMPLEX queries → Hurricane Specialist first, then Alert Manager if needed
    - EMERGENCY queries → Alert Manager (immediate life-safety response)

    Attributes:
        llm: ChatOpenAI model for query classification (gpt-4o-mini)
        memory_manager: Optional memory manager for user context
        agent_role: Fixed as AgentRole.TRIAGE
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        memory_manager: Optional[Any] = None,
    ):
        """Initialize Triage Agent.

        Args:
            model_name: LLM model for classification (default: gpt-4o-mini for speed/cost)
            memory_manager: Optional MemoryManager for user context integration
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,  # Deterministic routing decisions
            timeout=3.0,  # Fast classification (max 3s)
        )
        self.memory_manager = memory_manager
        self.agent_role = AgentRole.TRIAGE

        logger.info(
            "triage_agent_initialized",
            model=model_name,
            has_memory=memory_manager is not None,
        )

    async def classify_and_route(
        self,
        state: MultiAgentState | dict,
    ) -> MultiAgentState | dict:
        """Classify query and route to appropriate specialist agent.

        Process:
        1. Fetch user context from memory (if available)
        2. Format context for LLM prompt
        3. Call LLM with system prompt + classification prompt
        4. Parse JSON response with routing decision
        5. Create RoutingDecision object
        6. Update state with routing decision
        7. Log routing decision for observability

        Fallback Logic:
        - If LLM fails to provide valid JSON → Use fallback prompt
        - If fallback also fails → Route to Hurricane Specialist with confidence=0.5
        - If confidence <0.8 → Route to Hurricane Specialist with original confidence

        Args:
            state: Current multi-agent state containing query and context
                   (supports both Pydantic model and dict for L4a/L4b compatibility)

        Returns:
            Updated state with routing_decision and next_agent populated

        Raises:
            No exceptions raised - all errors handled with fallback logic
        """
        start_time = datetime.now(timezone.utc)

        # Support both Pydantic model and dict state (L4a/L4b compatibility)
        is_pydantic = hasattr(state, 'query') and not isinstance(state, dict)
        query = state.query if is_pydantic else state["query"]
        user_id = state.user_id if is_pydantic else state.get("user_id")
        session_id = state.session_id if is_pydantic else state.get("session_id")

        logger.info(
            "triage_classification_started",
            query=query[:100],  # Log first 100 chars
            user_id=user_id,
            session_id=session_id,
        )

        try:
            # Step 1: Fetch memory context (if available)
            memory_context = await self._fetch_memory_context(state)

            # Step 2: Format user context for LLM
            user_context_str = self._format_user_context(memory_context)

            # Step 3: Call LLM for classification
            classification_prompt = TRIAGE_CLASSIFICATION_PROMPT.format(
                query=query,
                user_context=user_context_str,
            )

            messages = [
                SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
                HumanMessage(content=classification_prompt),
            ]

            llm_response = await self.llm.ainvoke(messages)
            llm_content = llm_response.content

            # Step 4: Parse JSON response
            routing_data = self._parse_routing_decision(llm_content)

            # Fallback if parsing failed
            if routing_data is None:
                logger.warning(
                    "llm_json_parsing_failed",
                    llm_response=llm_content[:200],
                    query=query[:100],
                )
                routing_data = await self._use_fallback_routing(query)

            # Step 5: Create RoutingDecision object
            routing_decision = RoutingDecision(
                next_agent=AgentRole(routing_data["target_agent"]),
                confidence=routing_data["confidence"],
                rationale=routing_data["reasoning"],
                timestamp=datetime.now(timezone.utc),
                query_category=routing_data.get("complexity", "moderate"),
            )

            # Apply confidence threshold rule
            if routing_decision.confidence < 0.8:
                logger.warning(
                    "low_confidence_routing",
                    original_agent=routing_decision.next_agent,
                    original_confidence=routing_decision.confidence,
                    fallback_agent="hurricane_specialist",
                )
                # Fallback to Hurricane Specialist for safety
                routing_decision.next_agent = AgentRole.HURRICANE_SPECIALIST
                routing_decision.rationale += " [Routed to Hurricane Specialist due to low confidence (<0.8)]"

            # Step 6: Update state with routing decision
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Create Triage Agent response
            triage_response = AgentResponse(
                agent_role=AgentRole.TRIAGE,
                content=f"Query classified as {routing_data['complexity']}. Routing to {routing_decision.next_agent.value}.",
                confidence=routing_decision.confidence,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=execution_time_ms,
                metadata={
                    "complexity": routing_data["complexity"],
                    "requires_memory": routing_data.get("requires_memory", False),
                    "requires_tools": routing_data.get("requires_tools", []),
                    "raw_llm_response": llm_content[:500],  # First 500 chars for debugging
                },
            )

            # Step 7: Log routing decision
            logger.info(
                "triage_classification_complete",
                target_agent=routing_decision.next_agent.value,
                complexity=routing_data["complexity"],
                confidence=routing_decision.confidence,
                execution_time_ms=execution_time_ms,
                query=query[:100],
                rationale=routing_decision.rationale[:200],
            )

            # Update state (support both Pydantic and dict)
            if is_pydantic:
                # Pydantic model: update attributes directly
                state.routing_decision = routing_decision
                state.next_agent = routing_decision.next_agent
                state.current_agent = AgentRole.TRIAGE
                state.agent_responses.append(triage_response)
                return state
            else:
                # Dict state: create new dict with updates
                updated_state = {
                    **state,
                    "routing_decision": routing_decision,
                    "next_agent": routing_decision.next_agent,
                    "current_agent": AgentRole.TRIAGE,
                }
                if "agent_responses" not in updated_state:
                    updated_state["agent_responses"] = []
                updated_state["agent_responses"].append(triage_response)
                return updated_state

        except Exception as e:
            # Catch-all error handling with fallback
            logger.error(
                "triage_classification_error",
                error=str(e),
                error_type=type(e).__name__,
                query=query[:100],
            )

            # Fallback to Hurricane Specialist with low confidence
            fallback_routing = RoutingDecision(
                next_agent=AgentRole.HURRICANE_SPECIALIST,
                confidence=0.5,
                rationale=f"Routing to Hurricane Specialist as fallback due to triage error: {type(e).__name__}",
                timestamp=datetime.now(timezone.utc),
                query_category="moderate",
            )

            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Create error response
            error_response = AgentResponse(
                agent_role=AgentRole.TRIAGE,
                content=f"Triage classification error. Routing to Hurricane Specialist as fallback.",
                confidence=0.5,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=execution_time_ms,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback": True,
                },
            )

            # Update state (support both Pydantic and dict)
            if is_pydantic:
                # Pydantic model: update attributes directly
                state.routing_decision = fallback_routing
                state.next_agent = AgentRole.HURRICANE_SPECIALIST
                state.current_agent = AgentRole.TRIAGE
                state.agent_responses.append(error_response)
                return state
            else:
                # Dict state: create new dict with updates
                updated_state = {
                    **state,
                    "routing_decision": fallback_routing,
                    "next_agent": AgentRole.HURRICANE_SPECIALIST,
                    "current_agent": AgentRole.TRIAGE,
                }
                if "agent_responses" not in updated_state:
                    updated_state["agent_responses"] = []
                updated_state["agent_responses"].append(error_response)
                return updated_state

    async def _fetch_memory_context(self, state: MultiAgentState | dict) -> dict[str, Any]:
        """Fetch user context from memory manager.

        Args:
            state: Current multi-agent state (supports both Pydantic and dict)

        Returns:
            Dictionary containing memory context with keys:
            - previous_queries: List of recent user queries
            - user_preferences: User's weather preferences
            - emotional_state: Current emotional state indicators
            - location_history: Recently queried locations
        """
        if self.memory_manager is None:
            return {
                "previous_queries": [],
                "user_preferences": {},
                "emotional_state": "neutral",
                "location_history": [],
            }

        try:
            # TODO: Integrate with actual MemoryManager once implemented
            # For now, return placeholder context
            # Support both Pydantic model and dict state
            is_pydantic = hasattr(state, 'user_id') and not isinstance(state, dict)
            user_id = state.user_id if is_pydantic else state.get("user_id", "unknown")
            session_id = state.session_id if is_pydantic else state.get("session_id", "unknown")

            logger.debug(
                "fetching_memory_context",
                user_id=user_id,
                session_id=session_id,
            )

            # Placeholder - replace with actual memory manager calls
            memory_context = {
                "previous_queries": [],
                "user_preferences": {},
                "emotional_state": "neutral",
                "location_history": [],
            }

            return memory_context

        except Exception as e:
            logger.warning(
                "memory_context_fetch_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return empty context on error
            return {
                "previous_queries": [],
                "user_preferences": {},
                "emotional_state": "neutral",
                "location_history": [],
            }

    def _format_user_context(self, memory_context: dict[str, Any]) -> str:
        """Format memory context for LLM prompt.

        Args:
            memory_context: Dictionary with previous_queries, user_preferences,
                           emotional_state, location_history

        Returns:
            Formatted string suitable for inclusion in LLM prompt
        """
        # Format previous queries
        previous_queries_str = "\n".join(
            [f"- {q}" for q in memory_context.get("previous_queries", [])]
        )
        if not previous_queries_str:
            previous_queries_str = "None"

        # Format user preferences
        preferences = memory_context.get("user_preferences", {})
        if preferences:
            preferences_str = "\n".join([f"- {k}: {v}" for k, v in preferences.items()])
        else:
            preferences_str = "None"

        # Emotional state
        emotional_state = memory_context.get("emotional_state", "neutral")

        # Location history
        location_history = memory_context.get("location_history", [])
        location_history_str = ", ".join(location_history) if location_history else "None"

        # Use template
        formatted_context = USER_CONTEXT_TEMPLATE.format(
            previous_queries=previous_queries_str,
            user_preferences=preferences_str,
            emotional_state=emotional_state,
            location_history=location_history_str,
        )

        return formatted_context

    def _parse_routing_decision(self, llm_response: str) -> dict[str, Any] | None:
        """Parse JSON routing decision from LLM response.

        Args:
            llm_response: Raw LLM response string (should be JSON)

        Returns:
            Dictionary with routing decision fields, or None if parsing failed

        Expected JSON format:
        {
            "target_agent": "hurricane_specialist" | "alert_manager" | "direct_response",
            "complexity": "simple" | "moderate" | "complex" | "emergency",
            "confidence": 0.0-1.0,
            "reasoning": "Clear explanation",
            "requires_memory": true | false,
            "requires_tools": ["tool1", "tool2"]
        }
        """
        try:
            # Try to parse JSON directly
            routing_data = json.loads(llm_response)

            # Validate required fields
            required_fields = ["target_agent", "complexity", "confidence", "reasoning"]
            for field in required_fields:
                if field not in routing_data:
                    logger.error(
                        "json_missing_required_field",
                        field=field,
                        response=llm_response[:200],
                    )
                    return None

            # Validate field types
            if not isinstance(routing_data["confidence"], (int, float)):
                logger.error(
                    "invalid_confidence_type",
                    confidence=routing_data["confidence"],
                )
                return None

            if routing_data["confidence"] < 0.0 or routing_data["confidence"] > 1.0:
                logger.error(
                    "confidence_out_of_range",
                    confidence=routing_data["confidence"],
                )
                return None

            return routing_data

        except json.JSONDecodeError as e:
            logger.error(
                "json_decode_error",
                error=str(e),
                response=llm_response[:200],
            )
            return None
        except Exception as e:
            logger.error(
                "unexpected_parsing_error",
                error=str(e),
                error_type=type(e).__name__,
                response=llm_response[:200],
            )
            return None

    async def _use_fallback_routing(self, query: str) -> dict[str, Any]:
        """Use fallback prompt when primary classification fails.

        Args:
            query: Original user query

        Returns:
            Fallback routing decision (Hurricane Specialist, confidence=0.5)
        """
        try:
            fallback_prompt = TRIAGE_FALLBACK_PROMPT.format(query=query)

            messages = [
                SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
                HumanMessage(content=fallback_prompt),
            ]

            llm_response = await self.llm.ainvoke(messages)
            llm_content = llm_response.content

            # Try to parse fallback response
            routing_data = self._parse_routing_decision(llm_content)

            if routing_data is not None:
                logger.info(
                    "fallback_routing_successful",
                    target_agent=routing_data["target_agent"],
                )
                return routing_data

        except Exception as e:
            logger.error(
                "fallback_routing_error",
                error=str(e),
                error_type=type(e).__name__,
            )

        # Ultimate fallback: Hurricane Specialist with low confidence
        logger.warning(
            "using_ultimate_fallback",
            fallback_agent="hurricane_specialist",
            confidence=0.5,
        )

        return {
            "target_agent": "hurricane_specialist",
            "complexity": "moderate",
            "confidence": 0.5,
            "reasoning": "Routing to Hurricane Specialist as ultimate fallback due to classification failures",
            "requires_memory": True,
            "requires_tools": [],
        }
