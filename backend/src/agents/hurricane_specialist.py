"""Hurricane Specialist Agent for Level 4a: Three-Agent Foundation System.

CRITICAL: This module implements the Hurricane Specialist Agent that provides
expert hurricane forecasts, storm tracking, and evacuation guidance using
NHC data from the Hurricane MCP Server.

Components:
- HurricaneSpecialistAgent: Main agent class for hurricane domain expertise
- Integration with Hurricane MCP Server (http://localhost:8081)
- Optional ToT/GoT advanced reasoning integration
- Saffir-Simpson scale validation

Level 4a Architecture:
- Triage Agent → Routes hurricane queries here
- Hurricane Specialist Agent → Processes with domain expertise
- Alert Manager Agent → Receives high-risk assessments for alert generation

Design Principles:
- NHC data via Hurricane MCP Server as authoritative source
- Saffir-Simpson validation for all category assignments
- Confidence scoring based on data availability and reasoning
- Life-safety priority for all recommendations
- Timeout enforcement (max 30s for complex reasoning)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.config.settings import settings
from backend.src.agents.prompts.hurricane_prompts import (
    HURRICANE_FORECAST_PROMPT,
    HURRICANE_SPECIALIST_SYSTEM_PROMPT,
    NHC_DATA_TEMPLATE,
    SAFFIR_SIMPSON_SCALE,
    USER_CONTEXT_TEMPLATE,
)
from backend.src.mcp.hurricane_client import HurricaneMCPClient
from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)

logger = structlog.get_logger(__name__)


class HurricaneSpecialistAgent:
    """Domain expert agent for hurricane forecasts and storm tracking.

    The Hurricane Specialist Agent provides expert analysis of hurricane data
    from the NHC via the Hurricane MCP Server. It uses advanced reasoning
    (ToT/GoT) for complex queries and validates all responses against the
    Saffir-Simpson scale.

    Features:
    - NHC data integration via Hurricane MCP Server (http://localhost:8081)
    - Optional Tree of Thoughts (ToT) / Graph of Thoughts (GoT) reasoning
    - Saffir-Simpson scale validation for category accuracy
    - Confidence scoring based on data availability and reasoning quality
    - Structured logging for all operations

    Attributes:
        llm: ChatOpenAI model for forecast generation (gpt-4o for accuracy)
        agent_role: Fixed as AgentRole.HURRICANE_SPECIALIST
        mcp_server_url: URL of Hurricane MCP Server
        enable_tot: Whether to use Tree of Thoughts reasoning
        enable_got: Whether to use Graph of Thoughts reasoning
        tot_engine: Optional TreeOfThoughts instance
        got_engine: Optional GraphOfThoughts instance
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        enable_tot: bool = False,
        enable_got: bool = False,
    ):
        """Initialize Hurricane Specialist Agent.

        Args:
            model_name: LLM model for forecast generation (default: gpt-4o for accuracy)
            enable_tot: Enable Tree of Thoughts reasoning for complex queries
            enable_got: Enable Graph of Thoughts reasoning for multi-factor analysis
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # Low temperature for factual accuracy
            timeout=30.0,  # Longer timeout for complex reasoning
        )
        self.agent_role = AgentRole.HURRICANE_SPECIALIST
        self.mcp_server_url = settings.MCP_HURRICANE_SERVER_URL
        self.enable_tot = enable_tot
        self.enable_got = enable_got

        # Initialize Hurricane MCP Client
        self.hurricane_client = HurricaneMCPClient(base_url=self.mcp_server_url)
        self._mcp_initialized = False

        # Initialize reasoning engines if enabled
        self.tot_engine = None
        self.got_engine = None

        if enable_tot:
            try:
                from backend.src.reasoning.tot import TreeOfThoughts
                self.tot_engine = TreeOfThoughts(self.llm, max_depth=3, breadth=3)
                logger.info("tot_engine_initialized", max_depth=3, breadth=3)
            except ImportError:
                logger.warning("tot_import_failed", message="ToT not available")

        if enable_got:
            try:
                from backend.src.reasoning.got import GraphOfThoughts
                self.got_engine = GraphOfThoughts(self.llm, max_iterations=5)
                logger.info("got_engine_initialized", max_iterations=5)
            except ImportError:
                logger.warning("got_import_failed", message="GoT not available")

        logger.info(
            "hurricane_specialist_initialized",
            model=model_name,
            mcp_server_url=self.mcp_server_url,
            enable_tot=enable_tot,
            enable_got=enable_got,
        )

    async def process_query(
        self,
        state: MultiAgentState | dict,
    ) -> MultiAgentState | dict:
        """Process hurricane-related query with domain expertise.

        Steps:
        1. Fetch NHC data from Hurricane MCP Server
        2. Apply advanced reasoning (ToT/GoT) if enabled and query is complex
        3. Generate expert forecast with LLM
        4. Validate response against Saffir-Simpson scale
        5. Calculate confidence score
        6. Update state with agent response

        Args:
            state: Current multi-agent state containing query and context
                   (supports both Pydantic model and dict for L4a/L4b compatibility)

        Returns:
            Updated state with hurricane specialist response
        """
        start_time = datetime.now(UTC)

        # Support both Pydantic model and dict state (L4a/L4b compatibility)
        is_pydantic = hasattr(state, 'query') and not isinstance(state, dict)
        query = state.query if is_pydantic else state["query"]
        user_id = state.user_id if is_pydantic else state.get("user_id")
        session_id = state.session_id if is_pydantic else state.get("session_id")
        tool_calls: list[str] = []

        logger.info(
            "hurricane_specialist_started",
            query=query[:100],
            user_id=user_id,
            session_id=session_id,
        )

        try:
            # Step 1: Fetch NHC data from Hurricane MCP Server
            nhc_data = await self._fetch_nhc_data()
            if nhc_data:
                tool_calls.append("hurricane_mcp_server")

            # Step 2: Apply advanced reasoning if query is complex
            reasoning_context = ""
            reasoning_result = None

            if self._is_complex_query(state, is_pydantic):
                if self.got_engine:
                    try:
                        reasoning_result = await self.got_engine.build_graph(query)
                        reasoning_context = f"Graph of Thoughts Analysis:\n{reasoning_result.final_answer}"
                        tool_calls.append("graph_of_thoughts")
                        logger.info(
                            "got_reasoning_complete",
                            confidence=reasoning_result.confidence_score,
                        )
                    except Exception as e:
                        logger.warning("got_reasoning_failed", error=str(e))

                elif self.tot_engine:
                    try:
                        reasoning_result = await self.tot_engine.explore(query)
                        reasoning_context = f"Tree of Thoughts Analysis:\n{reasoning_result.final_answer}"
                        tool_calls.append("tree_of_thoughts")
                        logger.info(
                            "tot_reasoning_complete",
                            confidence=reasoning_result.confidence_score,
                        )
                    except Exception as e:
                        logger.warning("tot_reasoning_failed", error=str(e))

            # Step 3: Format data for LLM prompt
            nhc_data_str = self._format_nhc_data(nhc_data)
            user_context_str = self._format_user_context(state, is_pydantic)

            # Step 4: Generate expert forecast
            forecast_prompt = HURRICANE_FORECAST_PROMPT.format(
                query=query,
                nhc_data=nhc_data_str,
                reasoning_context=reasoning_context if reasoning_context else "No advanced reasoning applied.",
                user_context=user_context_str,
            )

            messages = [
                SystemMessage(content=HURRICANE_SPECIALIST_SYSTEM_PROMPT),
                HumanMessage(content=forecast_prompt),
            ]

            llm_response = await self.llm.ainvoke(messages)
            forecast_text = llm_response.content

            # Step 5: Validate response against Saffir-Simpson scale
            validation_errors = self._validate_hurricane_data(forecast_text)

            if validation_errors:
                logger.warning(
                    "saffir_simpson_validation_failed",
                    errors=validation_errors,
                )
                # Add validation warnings to response
                forecast_text += f"\n\n⚠️ **Validation Notes**: {', '.join(validation_errors)}"

            # Step 6: Calculate confidence score
            confidence = self._calculate_confidence(
                nhc_data_available=nhc_data is not None,
                reasoning_used=reasoning_result is not None,
                reasoning_confidence=reasoning_result.confidence_score if reasoning_result else 0.0,
                validation_passed=len(validation_errors) == 0,
            )

            # Calculate execution time
            execution_time_ms = (
                datetime.now(UTC) - start_time
            ).total_seconds() * 1000

            # Step 7: Create agent response
            agent_response = AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content=forecast_text,
                confidence=confidence,
                timestamp=datetime.now(UTC),
                execution_time_ms=execution_time_ms,
                metadata={
                    "nhc_data_used": nhc_data is not None,
                    "reasoning_type": reasoning_result.reasoning_type if reasoning_result else None,
                    "reasoning_confidence": reasoning_result.confidence_score if reasoning_result else None,
                    "validation_errors": validation_errors,
                    "tool_calls": tool_calls,
                    "mcp_server_url": self.mcp_server_url,
                },
            )

            logger.info(
                "hurricane_specialist_complete",
                confidence=confidence,
                tool_calls=tool_calls,
                execution_time_ms=execution_time_ms,
                validation_errors_count=len(validation_errors),
            )

            # Update state (support both Pydantic and dict)
            requires_alert = self._requires_alert_manager(state, confidence, nhc_data, is_pydantic)

            if is_pydantic:
                # Pydantic model: update attributes directly
                state.current_agent = AgentRole.HURRICANE_SPECIALIST
                state.agent_responses.append(agent_response)

                if requires_alert:
                    state.next_agent = AgentRole.ALERT_MANAGER
                    state.workflow_complete = False
                    logger.info(
                        "routing_to_alert_manager",
                        reason="high_risk_situation",
                    )
                else:
                    state.workflow_complete = True
                    state.final_response = forecast_text

                return state
            else:
                # Dict state: create new dict with updates
                updated_state = {
                    **state,
                    "current_agent": AgentRole.HURRICANE_SPECIALIST,
                }
                if "agent_responses" not in updated_state:
                    updated_state["agent_responses"] = []
                updated_state["agent_responses"].append(agent_response)

                if requires_alert:
                    updated_state["next_agent"] = AgentRole.ALERT_MANAGER
                    updated_state["workflow_complete"] = False
                    logger.info(
                        "routing_to_alert_manager",
                        reason="high_risk_situation",
                    )
                else:
                    updated_state["workflow_complete"] = True
                    updated_state["final_response"] = forecast_text

                return updated_state

        except Exception as e:
            # Error handling with fallback
            logger.error(
                "hurricane_specialist_error",
                error=str(e),
                error_type=type(e).__name__,
                query=query[:100],
            )

            execution_time_ms = (
                datetime.now(UTC) - start_time
            ).total_seconds() * 1000

            # Create error response
            error_response = AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Unable to process hurricane query due to an error. Please try again or contact support.",
                confidence=0.0,
                timestamp=datetime.now(UTC),
                execution_time_ms=execution_time_ms,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "tool_calls": tool_calls,
                },
            )

            # Update state (support both Pydantic and dict)
            if is_pydantic:
                # Pydantic model: update attributes directly
                state.current_agent = AgentRole.HURRICANE_SPECIALIST
                state.workflow_complete = True
                state.error = str(e)
                state.agent_responses.append(error_response)
                return state
            else:
                # Dict state: create new dict with updates
                updated_state = {
                    **state,
                    "current_agent": AgentRole.HURRICANE_SPECIALIST,
                    "workflow_complete": True,
                    "error": str(e),
                }
                if "agent_responses" not in updated_state:
                    updated_state["agent_responses"] = []
                updated_state["agent_responses"].append(error_response)
                return updated_state

    async def _fetch_nhc_data(self) -> dict[str, Any] | None:
        """Fetch NHC data from Hurricane MCP Server using MCP protocol.

        CRITICAL: Uses HurricaneMCPClient to call MCP tools (NOT REST endpoints).
        The Hurricane MCP Server exposes 5 MCP tools via the /mcp endpoint:
        - get_active_storms: Get currently active storms
        - get_storm_cone: Get forecast cone for specific storm
        - get_storm_track: Get historical and forecast track
        - get_local_hurricane_alerts: Get alerts for specific location
        - search_historical_tracks: Search historical hurricane data

        Implements 3-tier failover (Gap #2):
        1. Primary: Hurricane MCP Server (MCP protocol)
        2. Fallback: Direct NHC API calls (TODO: implement in future)
        3. Last resort: Cached data (TODO: implement in future)

        Returns:
            Dictionary containing NHC data, or None if all tiers fail

        Structured Logging (Gap #3):
            Logs all MCP operations with correlation IDs and performance metrics
        """
        if not settings.MCP_HURRICANE_SERVER_ENABLED:
            logger.info("hurricane_mcp_disabled", tier="disabled")
            return None

        # Generate correlation ID for structured logging (Gap #3)
        correlation_id = f"nhc-{datetime.now(UTC).timestamp()}"
        start_time = datetime.now(UTC)

        logger.info(
            "nhc_data_fetch_started",
            correlation_id=correlation_id,
            tier="mcp_primary",
            mcp_server_url=self.mcp_server_url,
        )

        try:
            # Tier 1: Hurricane MCP Server (PRIMARY)
            # Initialize MCP session if needed
            if not self._mcp_initialized:
                try:
                    await self.hurricane_client.initialize()
                    self._mcp_initialized = True
                    logger.info(
                        "mcp_session_initialized",
                        correlation_id=correlation_id,
                        mcp_server_url=self.mcp_server_url,
                    )
                except Exception as init_error:
                    logger.error(
                        "mcp_session_init_failed",
                        correlation_id=correlation_id,
                        error=str(init_error),
                        error_type=type(init_error).__name__,
                    )
                    # If failover is enabled, try Tier 2 (future implementation)
                    if settings.MCP_ENABLE_FAILOVER:
                        logger.info(
                            "failover_to_tier2",
                            correlation_id=correlation_id,
                            tier="nhc_api_direct",
                        )
                        # TODO: Implement direct NHC API fallback
                        return None
                    return None

            # Call get_active_storms MCP tool
            mcp_start_time = datetime.now(UTC)
            active_storms_response = await self.hurricane_client.get_active_storms()
            mcp_duration_ms = (datetime.now(UTC) - mcp_start_time).total_seconds() * 1000

            logger.info(
                "mcp_tool_called",
                correlation_id=correlation_id,
                tool_name="get_active_storms",
                duration_ms=round(mcp_duration_ms, 2),
                success=True,
            )

            # Extract storm data from MCP response
            # MCP response structure: {"result": {"content": [{"type": "text", "text": "..."}]}}
            active_storms_data = self._parse_mcp_response(active_storms_response)

            # If no storms, return early
            if not active_storms_data or not active_storms_data.get("storms"):
                logger.info(
                    "nhc_data_fetched",
                    correlation_id=correlation_id,
                    active_storms_count=0,
                    has_forecast=False,
                    total_duration_ms=round(
                        (datetime.now(UTC) - start_time).total_seconds() * 1000, 2
                    ),
                )
                return {
                    "active_storms": [],
                    "forecast_data": {},
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "source": "Hurricane MCP Server (MCP Protocol)",
                    "correlation_id": correlation_id,
                }

            # Fetch storm track for first active storm (if available)
            forecast_data = {}
            active_storms = active_storms_data.get("storms", [])
            if active_storms and len(active_storms) > 0:
                first_storm_id = active_storms[0].get("id", "")
                if first_storm_id:
                    try:
                        mcp_start_time = datetime.now(UTC)
                        storm_track_response = await self.hurricane_client.get_storm_track(
                            storm_id=first_storm_id
                        )
                        mcp_duration_ms = (
                            datetime.now(UTC) - mcp_start_time
                        ).total_seconds() * 1000

                        logger.info(
                            "mcp_tool_called",
                            correlation_id=correlation_id,
                            tool_name="get_storm_track",
                            storm_id=first_storm_id,
                            duration_ms=round(mcp_duration_ms, 2),
                            success=True,
                        )

                        forecast_data = self._parse_mcp_response(storm_track_response)

                    except Exception as track_error:
                        logger.warning(
                            "storm_track_fetch_failed",
                            correlation_id=correlation_id,
                            storm_id=first_storm_id,
                            error=str(track_error),
                            error_type=type(track_error).__name__,
                        )

            # Assemble NHC data
            nhc_data = {
                "active_storms": active_storms,
                "forecast_data": forecast_data,
                "fetched_at": datetime.now(UTC).isoformat(),
                "source": "Hurricane MCP Server (MCP Protocol)",
                "correlation_id": correlation_id,
            }

            total_duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            logger.info(
                "nhc_data_fetched",
                correlation_id=correlation_id,
                active_storms_count=len(active_storms),
                has_forecast=bool(forecast_data),
                total_duration_ms=round(total_duration_ms, 2),
                tier="mcp_primary",
            )

            return nhc_data

        except httpx.TimeoutException:
            duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            logger.error(
                "hurricane_mcp_timeout",
                correlation_id=correlation_id,
                url=self.mcp_server_url,
                timeout=settings.MCP_REQUEST_TIMEOUT,
                duration_ms=round(duration_ms, 2),
                tier="mcp_primary",
            )

            # Tier 2: Direct NHC API failover (if enabled)
            if settings.MCP_ENABLE_FAILOVER:
                logger.info(
                    "failover_to_tier2",
                    correlation_id=correlation_id,
                    tier="nhc_api_direct",
                )
                # TODO: Implement direct NHC API fallback
                # return await self._fetch_nhc_api_direct()

            return None

        except httpx.ConnectError:
            duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            logger.error(
                "hurricane_mcp_connection_error",
                correlation_id=correlation_id,
                url=self.mcp_server_url,
                duration_ms=round(duration_ms, 2),
                tier="mcp_primary",
            )

            # Tier 2: Direct NHC API failover (if enabled)
            if settings.MCP_ENABLE_FAILOVER:
                logger.info(
                    "failover_to_tier2",
                    correlation_id=correlation_id,
                    tier="nhc_api_direct",
                )
                # TODO: Implement direct NHC API fallback

            return None

        except Exception as e:
            duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            logger.error(
                "hurricane_mcp_error",
                correlation_id=correlation_id,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(duration_ms, 2),
                tier="mcp_primary",
            )

            # Tier 2: Direct NHC API failover (if enabled)
            if settings.MCP_ENABLE_FAILOVER:
                logger.info(
                    "failover_to_tier2",
                    correlation_id=correlation_id,
                    tier="nhc_api_direct",
                )
                # TODO: Implement direct NHC API fallback

            return None

    def _parse_mcp_response(self, mcp_response: dict[str, Any]) -> dict[str, Any]:
        """Parse MCP tool response to extract actual data.

        MCP responses have structure:
        {
            "result": {
                "content": [
                    {"type": "text", "text": "{...actual JSON data...}"}
                ]
            }
        }

        Args:
            mcp_response: Raw MCP response dictionary

        Returns:
            Parsed data dictionary, or empty dict if parsing fails
        """
        try:
            # Extract content from MCP response structure
            result = mcp_response.get("result", {})
            content = result.get("content", [])

            if not content or not isinstance(content, list):
                logger.warning("mcp_response_no_content", response_keys=list(mcp_response.keys()))
                return {}

            # Get first content item (usually the only one)
            first_content = content[0]
            if not isinstance(first_content, dict):
                logger.warning("mcp_response_invalid_content_type", content_type=type(first_content).__name__)
                return {}

            # Extract text field which contains the JSON data
            text_data = first_content.get("text", "")
            if not text_data:
                logger.warning("mcp_response_no_text_data")
                return {}

            # Parse JSON from text field
            parsed_data = json.loads(text_data) if isinstance(text_data, str) else text_data

            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(
                "mcp_response_json_parse_error",
                error=str(e),
                text_preview=text_data[:200] if text_data else None,
            )
            return {}
        except Exception as e:
            logger.error(
                "mcp_response_parse_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {}

    def _format_nhc_data(self, nhc_data: dict[str, Any] | None) -> str:
        """Format NHC data for LLM prompt.

        Args:
            nhc_data: Dictionary containing NHC data from MCP server

        Returns:
            Formatted string suitable for LLM prompt
        """
        if not nhc_data:
            return "**No NHC data available.** Hurricane MCP Server may be unavailable."

        active_storms = nhc_data.get("active_storms", [])

        if not active_storms:
            return "**No active storms currently in the Atlantic basin.**"

        storms_str = []
        for storm in active_storms:
            storm_info = (
                f"- **{storm.get('name', 'Unknown')}** "
                f"(Category {storm.get('category', '?')}): "
                f"{storm.get('max_wind_mph', '?')} mph winds, "
                f"Location: {storm.get('latitude', '?')}°N, {storm.get('longitude', '?')}°W, "
                f"Movement: {storm.get('movement', 'Unknown')}"
            )
            storms_str.append(storm_info)

        active_storms_str = "\n".join(storms_str)

        # Format forecast data if available
        forecast_str = "No detailed forecast data available."
        forecast_data = nhc_data.get("forecast_data", {})
        if forecast_data:
            forecast_points = forecast_data.get("forecast_points", [])
            if forecast_points:
                forecast_entries = []
                for point in forecast_points[:5]:  # First 5 forecast points
                    forecast_entries.append(
                        f"  - {point.get('time', '?')}: "
                        f"{point.get('latitude', '?')}°N, {point.get('longitude', '?')}°W, "
                        f"{point.get('max_wind_mph', '?')} mph"
                    )
                forecast_str = "Forecast Track:\n" + "\n".join(forecast_entries)

        return NHC_DATA_TEMPLATE.format(
            active_storms=active_storms_str,
            storm_details="See active storms above.",
            forecast_cone=forecast_str,
            evacuation_zones="Contact local emergency management for zone information.",
        )

    def _format_user_context(self, state: MultiAgentState | dict, is_pydantic: bool = False) -> str:
        """Format user context from memory for LLM prompt.

        Args:
            state: Current multi-agent state (supports both Pydantic and dict)
            is_pydantic: Whether state is a Pydantic model

        Returns:
            Formatted user context string
        """
        if is_pydantic:
            memory_context = getattr(state, "memory_context", {}) or {}
        else:
            memory_context = state.get("memory_context", {})

        # Extract user location
        user_location = "Unknown location"
        # Initialize defaults
        previous_queries_str = "No previous queries."
        preferences_str = "No specific preferences."
        emotional_state = "neutral"
        emotional_guidance = ""

        if memory_context:
            user_profile = memory_context.get("user_profile", {})
            user_location = user_profile.get("location", "Unknown location")

            # Extract previous queries
            previous_queries = memory_context.get("previous_queries", [])
            previous_queries_str = "\n".join(
                [f"- {q}" for q in previous_queries[-3:]]  # Last 3 queries
            ) if previous_queries else "No previous queries."

            # Extract preferences
            preferences = memory_context.get("user_preferences", {})
            preferences_str = json.dumps(preferences, indent=2) if preferences else "No specific preferences."

            # Emotional state handling
            emotional_state = memory_context.get("emotional_state", "neutral")
            if emotional_state in ["anxious", "fearful", "worried"]:
                emotional_guidance = "⚠️ User appears anxious. Provide reassuring but factual guidance."
            elif emotional_state in ["urgent", "panicked"]:
                emotional_guidance = "⚠️ User may be in crisis. Provide clear, calm, actionable steps."

        return USER_CONTEXT_TEMPLATE.format(
            user_location=user_location,
            previous_queries=previous_queries_str,
            user_preferences=preferences_str,
            emotional_state=emotional_state,
            emotional_guidance=emotional_guidance,
        )

    def _is_complex_query(self, state: MultiAgentState | dict, is_pydantic: bool = False) -> bool:
        """Determine if query requires advanced reasoning.

        Complex queries include:
        - Multi-day forecasts (>48 hours)
        - Evacuation decisions
        - Risk assessments
        - Multi-location comparisons
        - Rapidly intensifying storms

        Args:
            state: Current multi-agent state (supports both Pydantic and dict)
            is_pydantic: Whether state is a Pydantic model

        Returns:
            True if query is complex and should use ToT/GoT
        """
        query = state.query if is_pydantic else state["query"]
        query_lower = query.lower()

        # Check routing decision complexity
        routing_decision = state.routing_decision if is_pydantic else state.get("routing_decision")
        if routing_decision:
            query_category = getattr(routing_decision, "query_category", "")
            if query_category in ["complex", "emergency"]:
                return True

        # Keyword-based complexity detection
        complex_keywords = [
            "evacuate", "evacuation", "should i leave",
            "prepare", "preparation", "what should i do",
            "category 4", "category 5", "cat 4", "cat 5",
            "major hurricane", "life-threatening",
            "multiple", "compare", "which", "versus",
            "next week", "long-term", "extended",
            "rapidly intensifying", "strengthening quickly",
        ]

        return any(keyword in query_lower for keyword in complex_keywords)

    def _validate_hurricane_data(self, forecast_text: str) -> list[str]:
        """Validate hurricane data against Saffir-Simpson scale.

        Checks:
        - Category assignments match wind speeds
        - No vague timing (soon, later)
        - Proper format for emergency situations

        Args:
            forecast_text: Generated forecast text to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        text_lower = forecast_text.lower()

        # Check for vague timing
        vague_terms = ["soon", "later", "in a while", "sometime", "eventually"]
        for term in vague_terms:
            if term in text_lower:
                errors.append(f"Vague timing used: '{term}' - use specific EDT/UTC times")

        # Check for category/wind speed mismatches
        # Look for patterns like "Category X" with nearby wind speeds
        import re

        # Find all category mentions with nearby wind speeds
        cat_pattern = r"category\s*(\d)|cat\s*(\d)"
        wind_pattern = r"(\d{2,3})\s*mph"

        categories = re.findall(cat_pattern, text_lower)
        wind_speeds = re.findall(wind_pattern, text_lower)

        # Validate each category mention
        for cat_match in categories:
            category = int(cat_match[0] or cat_match[1])

            if category in SAFFIR_SIMPSON_SCALE:
                min_wind, max_wind = SAFFIR_SIMPSON_SCALE[category]["wind_mph"]

                # Check if any mentioned wind speed is inconsistent
                for wind_str in wind_speeds:
                    wind = int(wind_str)

                    # Check if wind speed doesn't match stated category
                    if wind < min_wind and category > 1:
                        # Wind is too low for this category
                        correct_cat = self._get_category_for_wind(wind)
                        if correct_cat != category:
                            errors.append(
                                f"Category {category} stated but {wind} mph wind "
                                f"suggests Category {correct_cat}"
                            )
                        break

        return errors

    def _get_category_for_wind(self, wind_mph: int) -> int:
        """Get correct Saffir-Simpson category for wind speed.

        Args:
            wind_mph: Wind speed in mph

        Returns:
            Saffir-Simpson category (0 for tropical storm, 1-5 for hurricane)
        """
        if wind_mph < 74:
            return 0  # Tropical storm
        elif wind_mph <= 95:
            return 1
        elif wind_mph <= 110:
            return 2
        elif wind_mph <= 129:
            return 3
        elif wind_mph <= 156:
            return 4
        else:
            return 5

    def _calculate_confidence(
        self,
        nhc_data_available: bool,
        reasoning_used: bool,
        reasoning_confidence: float,
        validation_passed: bool,
    ) -> float:
        """Calculate response confidence based on available data and reasoning.

        Confidence Scoring:
        - Base confidence: 0.5
        - NHC data available: +0.3
        - Advanced reasoning used: +0.1 (weighted by reasoning confidence)
        - Validation passed: +0.1

        Args:
            nhc_data_available: Whether NHC data was fetched successfully
            reasoning_used: Whether ToT/GoT reasoning was applied
            reasoning_confidence: Confidence from reasoning engine (0-1)
            validation_passed: Whether Saffir-Simpson validation passed

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.5  # Base confidence

        if nhc_data_available:
            confidence += 0.3  # NHC data adds authority

        if reasoning_used:
            # Add reasoning bonus weighted by reasoning confidence
            confidence += 0.1 * reasoning_confidence

        if validation_passed:
            confidence += 0.1  # Validation adds accuracy

        return min(confidence, 1.0)

    def _requires_alert_manager(
        self,
        state: MultiAgentState | dict,
        confidence: float,
        nhc_data: dict[str, Any] | None,
        is_pydantic: bool = False,
    ) -> bool:
        """Determine if Alert Manager should be invoked for emergency alerts.

        Conditions for Alert Manager routing:
        - Query classified as EMERGENCY complexity
        - Active Category 3+ hurricane
        - Evacuation mentioned in query
        - Time-critical situation detected

        Args:
            state: Current multi-agent state (supports both Pydantic and dict)
            confidence: Hurricane Specialist's confidence score
            nhc_data: NHC data from MCP server
            is_pydantic: Whether state is a Pydantic model

        Returns:
            True if Alert Manager should be invoked
        """
        query = state.query if is_pydantic else state["query"]
        query_lower = query.lower()

        # Check routing decision for emergency classification
        routing_decision = state.routing_decision if is_pydantic else state.get("routing_decision")
        if routing_decision:
            query_category = getattr(routing_decision, "query_category", "")
            if query_category == "emergency":
                return True

        # Check for urgent keywords
        urgent_keywords = [
            "evacuate now", "should i leave now", "immediate",
            "urgent", "emergency", "right now", "asap",
        ]
        if any(keyword in query_lower for keyword in urgent_keywords):
            return True

        # Check for major hurricane (Cat 3+) in NHC data
        if nhc_data:
            active_storms = nhc_data.get("active_storms", [])
            for storm in active_storms:
                category = storm.get("category", 0)
                if isinstance(category, int) and category >= 3:
                    return True

        return False
