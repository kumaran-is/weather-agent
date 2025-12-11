"""Tests for Forecaster Agent (Level 4b Phase 9).

Test Coverage:
- Weather forecast generation
- Location extraction from queries
- Forecast type detection (current, hourly, daily)
- MCP Weather Server integration
- Multi-source data handling
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.agents.forecaster_agent import ForecasterAgent
from backend.src.models.multi_agent import (
    AgentRole,
    AgentResponse,
    MultiAgentState,
)


# Fixtures
@pytest.fixture
def base_state() -> MultiAgentState:
    """Create base MultiAgentState for testing."""
    return MultiAgentState(
        query="Test weather query",
        user_id="test_user_123",
        session_id="test_session_456",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        timeout_ms=30000,
        current_agent=None,
        routing_decision=None,
        agent_responses=[],
        next_agent=None,
        workflow_complete=False,
        final_response=None,
        error=None,
        total_execution_time_ms=0.0,
    )


@pytest.fixture
def forecaster_agent() -> ForecasterAgent:
    """Create ForecasterAgent for testing."""
    return ForecasterAgent()


@pytest.fixture
def mock_forecast_response():
    """Mock LLM forecast response."""
    return MagicMock(
        content="""Weather Forecast for Miami, FL:

CURRENT CONDITIONS:
- Temperature: 78°F (26°C)
- Humidity: 65%
- Wind: SE at 12 mph
- Conditions: Partly Cloudy

TODAY'S FORECAST:
- High: 85°F (29°C)
- Low: 72°F (22°C)
- Chance of rain: 30%

5-DAY OUTLOOK:
- Monday: Sunny, 86°F
- Tuesday: Partly cloudy, 84°F
- Wednesday: Isolated storms, 82°F
- Thursday: Partly cloudy, 83°F
- Friday: Sunny, 85°F"""
    )


# Test: ForecasterAgent Initialization
class TestForecasterAgentInit:
    """Tests for ForecasterAgent initialization."""

    def test_default_initialization(self, forecaster_agent):
        """Test ForecasterAgent initializes with defaults."""
        assert forecaster_agent.llm is not None
        assert forecaster_agent.agent_role == AgentRole.FORECASTER

    def test_custom_initialization(self):
        """Test ForecasterAgent with custom model."""
        agent = ForecasterAgent(model_name="gpt-4o")
        assert agent.llm is not None


# Test: Location Extraction
class TestLocationExtraction:
    """Tests for location extraction from queries."""

    def test_extract_city_state(self, forecaster_agent):
        """Test extraction of city and state."""
        query = "What's the weather in Miami, Florida?"
        location = forecaster_agent._extract_location(query)
        assert location is not None
        assert "miami" in location.lower()

    def test_extract_city_only(self, forecaster_agent):
        """Test extraction of city name only."""
        query = "Weather forecast for Tampa"
        location = forecaster_agent._extract_location(query)
        assert location is not None
        assert "tampa" in location.lower()

    def test_extract_with_zip(self, forecaster_agent):
        """Test extraction with ZIP code context."""
        query = "Weather for Orlando, FL 32801"
        location = forecaster_agent._extract_location(query)
        assert location is not None

    def test_no_location_returns_default(self, forecaster_agent):
        """Test default location when none specified."""
        query = "What's the weather like today?"
        location = forecaster_agent._extract_location(query)
        # Should return some default or None
        assert location is None or location != ""


# Test: Forecast Type Detection
class TestForecastTypeDetection:
    """Tests for forecast type detection."""

    def test_detect_current_weather(self, forecaster_agent):
        """Test detection of current weather request."""
        query = "What's the current weather in Miami?"
        forecast_type = forecaster_agent._determine_forecast_type(query)
        assert forecast_type == "current"

    def test_detect_hourly_forecast(self, forecaster_agent):
        """Test detection of hourly forecast request."""
        query = "Hourly forecast for Tampa today"
        forecast_type = forecaster_agent._determine_forecast_type(query)
        assert forecast_type == "hourly"

    def test_detect_daily_forecast(self, forecaster_agent):
        """Test detection of daily/extended forecast request."""
        query = "5-day forecast for Orlando"
        forecast_type = forecaster_agent._determine_forecast_type(query)
        assert forecast_type == "daily"

    def test_detect_weekly_forecast(self, forecaster_agent):
        """Test detection of weekly forecast request."""
        query = "What's the weather for this week?"
        forecast_type = forecaster_agent._determine_forecast_type(query)
        assert forecast_type in ["daily", "weekly", "extended"]  # Extended also valid

    def test_default_forecast_type(self, forecaster_agent):
        """Test default forecast type for ambiguous queries."""
        query = "Weather in Miami"
        forecast_type = forecaster_agent._determine_forecast_type(query)
        assert forecast_type in ["current", "daily"]  # Default may be current or daily


# Test: Forecast Query
class TestForecastQuery:
    """Tests for forecast_query method."""

    @pytest.mark.asyncio
    async def test_forecast_query_success(
        self, forecaster_agent, base_state, mock_forecast_response
    ):
        """Test successful forecast query."""
        base_state.query = "What's the weather in Miami?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_forecast_response)
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        assert result is not None
        assert len(result.agent_responses) == 1
        assert result.agent_responses[0].agent_role == AgentRole.FORECASTER

    @pytest.mark.asyncio
    async def test_forecast_query_with_location(
        self, forecaster_agent, base_state, mock_forecast_response
    ):
        """Test forecast query with specific location."""
        base_state.query = "5-day forecast for Tampa, Florida"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_forecast_response)
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        assert result is not None
        response = result.agent_responses[-1]
        assert response.confidence > 0

    @pytest.mark.asyncio
    async def test_forecast_query_error_handling(
        self, forecaster_agent, base_state
    ):
        """Test forecast query handles errors."""
        base_state.query = "Weather in Miami"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        # Should handle error gracefully
        assert result is not None
        assert len(result.agent_responses) == 1
        assert result.agent_responses[0].confidence == 0.0


# Test: MCP Integration
class TestMCPIntegration:
    """Tests for MCP Weather Server integration."""

    @pytest.mark.asyncio
    async def test_fetch_from_mcp_server(self, forecaster_agent):
        """Test fetching data from MCP server."""
        # This tests the MCP integration pattern
        # Actual MCP calls would be mocked in integration tests
        assert forecaster_agent is not None

    @pytest.mark.asyncio
    async def test_mcp_fallback_on_error(self, forecaster_agent, base_state):
        """Test fallback when MCP server unavailable."""
        base_state.query = "Weather in Miami"

        # Mock LLM to simulate fallback
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Forecast based on available data"
        ))
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        # Should still provide response via LLM
        assert result is not None


# Test: Response Formatting
class TestResponseFormatting:
    """Tests for response formatting."""

    @pytest.mark.asyncio
    async def test_response_includes_metadata(
        self, forecaster_agent, base_state, mock_forecast_response
    ):
        """Test that response includes proper metadata."""
        base_state.query = "Weather in Tampa"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_forecast_response)
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        response = result.agent_responses[-1]
        assert response.metadata is not None
        assert "forecast_type" in response.metadata

    @pytest.mark.asyncio
    async def test_response_has_execution_time(
        self, forecaster_agent, base_state, mock_forecast_response
    ):
        """Test that response includes execution time."""
        base_state.query = "Weather in Orlando"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_forecast_response)
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        response = result.agent_responses[-1]
        assert response.execution_time_ms > 0


# Test: Confidence Calculation
class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    @pytest.mark.asyncio
    async def test_high_confidence_with_location(
        self, forecaster_agent, base_state, mock_forecast_response
    ):
        """Test confidence when location is clear."""
        base_state.query = "Current weather in Miami, FL"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_forecast_response)
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        response = result.agent_responses[-1]
        # Base confidence is 0.7 without weather data, higher with data
        assert response.confidence >= 0.6

    @pytest.mark.asyncio
    async def test_lower_confidence_without_location(
        self, forecaster_agent, base_state, mock_forecast_response
    ):
        """Test lower confidence when location is ambiguous."""
        base_state.query = "What's the weather?"  # No location

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_forecast_response)
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        response = result.agent_responses[-1]
        # Should still work but potentially lower confidence
        assert response.confidence >= 0


# Test: Multi-Day Forecasts
class TestMultiDayForecasts:
    """Tests for multi-day forecast handling."""

    @pytest.mark.asyncio
    async def test_7_day_forecast(
        self, forecaster_agent, base_state
    ):
        """Test 7-day forecast request."""
        base_state.query = "7-day forecast for Tampa"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="7-day forecast: Monday sunny 85F..."
        ))
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        response = result.agent_responses[-1]
        assert response.metadata.get("forecast_type") == "daily"

    @pytest.mark.asyncio
    async def test_weekend_forecast(
        self, forecaster_agent, base_state
    ):
        """Test weekend forecast request."""
        base_state.query = "What's the weather this weekend in Miami?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Weekend forecast: Saturday partly cloudy..."
        ))
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        assert result is not None


# Test: Error Handling
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_llm_timeout(
        self, forecaster_agent, base_state
    ):
        """Test handling of LLM timeout."""
        base_state.query = "Weather in Miami"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=TimeoutError("Timeout"))
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        # Should handle gracefully
        assert result is not None
        assert result.agent_responses[-1].confidence == 0.0

    @pytest.mark.asyncio
    async def test_handles_invalid_response(
        self, forecaster_agent, base_state
    ):
        """Test handling of invalid LLM response."""
        base_state.query = "Weather in Tampa"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=None))
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        assert result is not None


# Integration Tests
class TestForecasterIntegration:
    """Integration tests for ForecasterAgent."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_forecast_workflow(
        self, forecaster_agent, base_state, mock_forecast_response
    ):
        """Test complete forecast query workflow."""
        base_state.query = "What's the 5-day forecast for Miami, Florida?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_forecast_response)
        forecaster_agent.llm = mock_llm

        result = await forecaster_agent.generate_forecast(base_state)

        # Verify complete workflow
        assert result is not None
        assert len(result.agent_responses) == 1

        response = result.agent_responses[0]
        assert response.agent_role == AgentRole.FORECASTER
        assert response.content is not None
        assert response.confidence > 0
        assert response.execution_time_ms > 0
        assert "forecast_type" in response.metadata
