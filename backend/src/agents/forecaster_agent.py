"""Forecaster Agent for Level 4b: General Weather Forecasting Specialist.

CRITICAL RULES:
1. Handles non-hurricane weather queries
2. Uses Weather MCP Server for data retrieval
3. Integrates with Supervisor for parallel execution
4. Provides clear, actionable forecasts

Architecture:
- Weather MCP Server integration for real-time data
- Location extraction from queries
- Temperature, precipitation, wind forecasting
- Multi-day forecast support

Design Principles:
- Fast response with gpt-4o-mini
- MCP-first data retrieval
- Clear actionable guidance
- Integration with multi-agent workflow
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.config.settings import settings
from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)

logger = structlog.get_logger(__name__)


class ForecasterAgent:
    """General weather forecasting specialist.

    The Forecaster Agent handles non-hurricane weather queries:
    - Current conditions
    - Temperature forecasts
    - Precipitation forecasts
    - Wind conditions
    - Multi-day outlooks

    Data Sources:
    - Weather MCP Server (primary)
    - LLM knowledge (fallback)

    Attributes:
        llm: ChatOpenAI model for forecast generation
        agent_role: Fixed as AgentRole.FORECASTER
        mcp_server_url: URL of Weather MCP Server
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        """Initialize Forecaster Agent.

        Args:
            model_name: LLM model for forecast generation (default: gpt-4o-mini for speed)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Some variability for natural language
            timeout=15.0,
        )
        self.agent_role = AgentRole.FORECASTER
        self.mcp_server_url = settings.MCP_WEATHER_SERVER_URL

        logger.info(
            "forecaster_agent_initialized",
            model=model_name,
            mcp_server_url=self.mcp_server_url,
        )

    async def generate_forecast(
        self,
        state: MultiAgentState,
    ) -> MultiAgentState:
        """Generate weather forecast for query.

        Steps:
        1. Extract location from query
        2. Fetch weather data from MCP server
        3. Generate forecast using LLM
        4. Return updated state with response

        Args:
            state: Current multi-agent state

        Returns:
            Updated state with forecast response
        """
        start_time = time.perf_counter()
        query = state.query
        tool_calls: list[str] = []

        logger.info(
            "forecaster_started",
            query=query[:100],
            user_id=state.user_id,
        )

        try:
            # Step 1: Extract location from query
            location = self._extract_location(query)

            # Step 2: Fetch weather data from MCP server
            weather_data = await self._fetch_weather_data(location)
            if weather_data:
                tool_calls.append("weather_mcp_server")

            # Step 3: Determine forecast type
            forecast_type = self._determine_forecast_type(query)

            # Step 4: Generate forecast with LLM
            forecast_prompt = self._build_forecast_prompt(
                query=query,
                location=location,
                weather_data=weather_data,
                forecast_type=forecast_type,
            )

            messages = [
                SystemMessage(content=self._get_system_prompt(forecast_type)),
                HumanMessage(content=forecast_prompt),
            ]

            response = await self.llm.ainvoke(messages)
            forecast_text = response.content

            # Calculate confidence
            confidence = self._calculate_confidence(
                has_weather_data=weather_data is not None,
                forecast_type=forecast_type,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Create agent response
            agent_response = AgentResponse(
                agent_role=self.agent_role,
                content=forecast_text,
                confidence=confidence,
                timestamp=datetime.now(UTC),
                execution_time_ms=duration_ms,
                metadata={
                    "location": location,
                    "forecast_type": forecast_type,
                    "weather_data_used": weather_data is not None,
                    "tool_calls": tool_calls,
                    "mcp_server_url": self.mcp_server_url,
                },
            )

            logger.info(
                "forecaster_complete",
                location=location,
                forecast_type=forecast_type,
                confidence=confidence,
                duration_ms=duration_ms,
            )

            # Update state
            if isinstance(state, dict):
                if "agent_responses" not in state:
                    state["agent_responses"] = []
                state["agent_responses"].append(agent_response)
            else:
                state.agent_responses.append(agent_response)

            return state

        except Exception as e:
            logger.error(
                "forecaster_error",
                error=str(e),
                error_type=type(e).__name__,
                query=query[:100],
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            error_response = AgentResponse(
                agent_role=self.agent_role,
                content=f"Unable to generate forecast: {type(e).__name__}",
                confidence=0.0,
                timestamp=datetime.now(UTC),
                execution_time_ms=duration_ms,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "tool_calls": tool_calls,
                },
            )

            if isinstance(state, dict):
                if "agent_responses" not in state:
                    state["agent_responses"] = []
                state["agent_responses"].append(error_response)
            else:
                state.agent_responses.append(error_response)

            return state

    async def _fetch_weather_data(self, location: str) -> dict[str, Any] | None:
        """Fetch weather data from MCP server.

        Args:
            location: Location to fetch weather for

        Returns:
            Weather data dictionary or None if fetch fails
        """
        if not settings.MCP_WEATHER_SERVER_ENABLED:
            logger.info("weather_mcp_disabled")
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Fetch current conditions
                current_response = await client.get(
                    f"{self.mcp_server_url}/api/weather/current",
                    params={"location": location},
                )

                # Fetch forecast
                forecast_response = await client.get(
                    f"{self.mcp_server_url}/api/weather/forecast",
                    params={"location": location, "days": 7},
                )

                weather_data = {
                    "current": current_response.json() if current_response.status_code == 200 else None,
                    "forecast": forecast_response.json() if forecast_response.status_code == 200 else None,
                    "location": location,
                    "fetched_at": datetime.now(UTC).isoformat(),
                }

                logger.info(
                    "weather_data_fetched",
                    location=location,
                    has_current=weather_data["current"] is not None,
                    has_forecast=weather_data["forecast"] is not None,
                )

                return weather_data

        except httpx.TimeoutException:
            logger.warning("weather_mcp_timeout", url=self.mcp_server_url)
            return None
        except httpx.ConnectError:
            logger.warning("weather_mcp_connection_error", url=self.mcp_server_url)
            return None
        except Exception as e:
            logger.warning("weather_mcp_error", error=str(e))
            return None

    def _extract_location(self, query: str) -> str:
        """Extract location from query.

        Simple keyword-based extraction for Level 4b.
        Level 5 will use NER for better extraction.

        Args:
            query: User's query string

        Returns:
            Extracted location or default
        """
        query_lower = query.lower()

        # Common Florida locations
        florida_locations = [
            "Miami", "Tampa", "Orlando", "Jacksonville", "Fort Lauderdale",
            "Sarasota", "Naples", "Key West", "Tallahassee", "Gainesville",
            "Fort Myers", "Pensacola", "Clearwater", "St. Petersburg",
            "Palm Beach", "Boca Raton", "Daytona Beach",
        ]

        # Other common US locations
        other_locations = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "San Antonio", "San Diego", "Dallas", "San Francisco", "Seattle",
            "Boston", "Atlanta", "Denver", "Washington", "Las Vegas",
        ]

        # Check for locations in query
        all_locations = florida_locations + other_locations

        for loc in all_locations:
            if loc.lower() in query_lower:
                return loc

        # Check for state mentions
        if "florida" in query_lower:
            return "Florida"

        # Default to Florida (project focus)
        return "Tampa, FL"

    def _determine_forecast_type(self, query: str) -> str:
        """Determine type of forecast requested.

        Args:
            query: User's query string

        Returns:
            Forecast type: "current", "hourly", "daily", "extended"
        """
        query_lower = query.lower()

        # Extended forecast (5+ days)
        if any(word in query_lower for word in ["week", "extended", "long-term", "7 day", "next week"]):
            return "extended"

        # Daily forecast (2-4 days)
        if any(word in query_lower for word in ["tomorrow", "next few days", "this week", "daily"]):
            return "daily"

        # Hourly forecast
        if any(word in query_lower for word in ["hourly", "hour by hour", "this afternoon", "tonight"]):
            return "hourly"

        # Current conditions
        if any(word in query_lower for word in ["current", "right now", "currently", "at the moment"]):
            return "current"

        # Default to daily
        return "daily"

    def _get_system_prompt(self, forecast_type: str) -> str:
        """Get system prompt based on forecast type.

        Args:
            forecast_type: Type of forecast requested

        Returns:
            System prompt string
        """
        base_prompt = """You are a professional weather forecaster providing accurate, helpful forecasts.

Guidelines:
1. Use specific times (e.g., "3:00 PM EDT", never "later today")
2. Include temperature ranges (high/low)
3. Mention precipitation probability with percentages
4. Note wind conditions if significant
5. Provide actionable recommendations
6. Be clear and concise

"""

        if forecast_type == "extended":
            return base_prompt + """For extended forecasts:
- Provide day-by-day outlook for 5-7 days
- Note any significant weather changes expected
- Mention confidence levels (higher for near-term, lower for later days)
- Include temperature trends
- Highlight any severe weather potential"""

        if forecast_type == "hourly":
            return base_prompt + """For hourly forecasts:
- Provide hour-by-hour breakdown
- Include specific times (e.g., "2:00 PM EDT: 85°F, partly cloudy")
- Note precipitation timing precisely
- Mention when conditions will change"""

        if forecast_type == "current":
            return base_prompt + """For current conditions:
- Report real-time observations
- Include temperature, humidity, wind
- Note any active weather (rain, storms)
- Provide "feels like" temperature if different"""

        # Default daily
        return base_prompt + """For daily forecasts:
- Provide day/night breakdown
- Include high and low temperatures
- Note precipitation chances for each period
- Mention any notable weather features"""

    def _build_forecast_prompt(
        self,
        query: str,
        location: str,
        weather_data: dict[str, Any] | None,
        forecast_type: str,
    ) -> str:
        """Build forecast prompt with available data.

        Args:
            query: User's query
            location: Extracted location
            weather_data: Weather data from MCP (or None)
            forecast_type: Type of forecast

        Returns:
            Formatted prompt string
        """
        # Format weather data if available
        if weather_data:
            current = weather_data.get("current", {})
            forecast = weather_data.get("forecast", {})

            data_str = f"""**Weather Data for {location}**:

Current Conditions:
{self._format_current_conditions(current)}

Forecast Data:
{self._format_forecast_data(forecast)}
"""
        else:
            data_str = f"**Note**: Weather MCP Server data unavailable for {location}. Use general knowledge."

        return f"""Generate a {forecast_type} weather forecast for this query:

**Query**: {query}
**Location**: {location}
**Forecast Type**: {forecast_type}

{data_str}

Provide a clear, actionable forecast addressing the user's query."""

    def _format_current_conditions(self, current: dict[str, Any]) -> str:
        """Format current conditions for prompt.

        Args:
            current: Current conditions data

        Returns:
            Formatted string
        """
        if not current:
            return "No current conditions data available."

        parts = []

        if "temperature" in current:
            parts.append(f"Temperature: {current['temperature']}°F")
        if "feels_like" in current:
            parts.append(f"Feels Like: {current['feels_like']}°F")
        if "humidity" in current:
            parts.append(f"Humidity: {current['humidity']}%")
        if "wind_speed" in current:
            parts.append(f"Wind: {current['wind_speed']} mph {current.get('wind_direction', '')}")
        if "conditions" in current:
            parts.append(f"Conditions: {current['conditions']}")

        return "\n".join(parts) if parts else "Limited current data available."

    def _format_forecast_data(self, forecast: dict[str, Any]) -> str:
        """Format forecast data for prompt.

        Args:
            forecast: Forecast data

        Returns:
            Formatted string
        """
        if not forecast:
            return "No forecast data available."

        days = forecast.get("days", forecast.get("daily", []))

        if not days:
            return "No daily forecast data available."

        parts = []
        for day in days[:7]:  # Max 7 days
            day_str = f"- {day.get('date', 'Unknown')}: "
            day_str += f"High {day.get('high_temp', '?')}°F, Low {day.get('low_temp', '?')}°F"
            if "precip_chance" in day:
                day_str += f", {day['precip_chance']}% precip"
            if "conditions" in day:
                day_str += f" ({day['conditions']})"
            parts.append(day_str)

        return "\n".join(parts)

    def _calculate_confidence(
        self,
        has_weather_data: bool,
        forecast_type: str,
    ) -> float:
        """Calculate forecast confidence.

        Args:
            has_weather_data: Whether MCP data was available
            forecast_type: Type of forecast

        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = 0.6

        # Boost for MCP data
        if has_weather_data:
            base_confidence += 0.25

        # Adjust for forecast type (near-term more confident)
        type_adjustments = {
            "current": 0.1,
            "hourly": 0.05,
            "daily": 0.0,
            "extended": -0.1,  # Less confident for extended forecasts
        }

        adjustment = type_adjustments.get(forecast_type, 0.0)
        return min(0.95, max(0.5, base_confidence + adjustment))
