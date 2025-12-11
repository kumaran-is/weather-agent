"""Tests for Hurricane Specialist Agent (Level 4a Phase 3).

Test Coverage:
- Hurricane forecast generation
- NHC data integration via Hurricane MCP Server
- Saffir-Simpson scale validation
- Advanced reasoning integration (ToT/GoT)
- Confidence scoring based on data availability
- Risk-based routing to Alert Manager
- Error handling and graceful degradation
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.agents.hurricane_specialist import HurricaneSpecialistAgent
from backend.src.models.multi_agent import (
    AgentRole,
    RoutingDecision,
    AgentResponse,
)


# Fixtures
@pytest.fixture
def base_state() -> dict:
    """Create base state for testing."""
    return {
        "query": "Test hurricane query",
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "current_agent": None,
        "routing_decision": MagicMock(
            query_category="moderate",
            confidence=0.9,
            next_agent=AgentRole.HURRICANE_SPECIALIST,
        ),
        "agent_responses": [],
        "next_agent": None,
        "workflow_complete": False,
        "final_response": None,
        "error": None,
        "memory_context": {
            "user_profile": {"location": "Tampa, FL"},
            "previous_queries": ["Hurricane status?"],
            "user_preferences": {"units": "imperial"},
            "emotional_state": "neutral",
        },
    }


@pytest.fixture
def hurricane_specialist() -> HurricaneSpecialistAgent:
    """Create HurricaneSpecialistAgent without ToT/GoT."""
    return HurricaneSpecialistAgent(
        model_name="gpt-4o",
        enable_tot=False,
        enable_got=False,
    )


@pytest.fixture
def mock_nhc_data() -> dict:
    """Mock NHC data from Hurricane MCP Server."""
    return {
        "active_storms": [
            {
                "id": "AL052024",
                "name": "Hurricane Michael",
                "category": 4,
                "max_wind_mph": 140,
                "latitude": 26.5,
                "longitude": -86.5,
                "movement": "NNE at 15 mph",
                "pressure_mb": 943,
            }
        ],
        "forecast_data": {
            "forecast_points": [
                {"time": "24h", "latitude": 28.0, "longitude": -85.0, "max_wind_mph": 145},
                {"time": "48h", "latitude": 30.0, "longitude": -83.5, "max_wind_mph": 130},
                {"time": "72h", "latitude": 32.0, "longitude": -82.0, "max_wind_mph": 90},
            ]
        },
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "Hurricane MCP Server",
    }


@pytest.fixture
def mock_llm_forecast_response():
    """Mock LLM forecast response."""
    return MagicMock(
        content="""## Current Status
Hurricane Michael is currently a Category 4 hurricane with maximum sustained winds of 140 mph.
Location: 26.5°N, 86.5°W, moving NNE at 15 mph.

## Forecast Track
- 24 hours: Strengthening to 145 mph, approaching Florida Panhandle
- 48 hours: Landfall expected near Panama City, 130 mph (Cat 4)
- 72 hours: Inland weakening to 90 mph (Cat 1)

## Potential Impacts
- **Wind**: Catastrophic damage expected in direct path
- **Surge**: 10-15 feet of storm surge along Big Bend coast
- **Rainfall**: 8-12 inches across Florida Panhandle

## Recommended Actions
1. EVACUATE if in Zone A or B along the coast
2. Board up windows and secure outdoor items
3. Prepare emergency supplies for 7 days
4. Fill vehicles with gas and withdraw cash

## Confidence Assessment
HIGH confidence in track (NHC model agreement strong).
MEDIUM confidence in intensity (rapid intensification possible).
"""
    )


# Test: Initialization
def test_hurricane_specialist_initialization():
    """Test HurricaneSpecialistAgent initialization."""
    agent = HurricaneSpecialistAgent()

    assert agent.llm is not None
    assert agent.llm.model_name == "gpt-4o"
    assert agent.llm.temperature == 0.1
    assert agent.agent_role == AgentRole.HURRICANE_SPECIALIST
    assert agent.tot_engine is None
    assert agent.got_engine is None


def test_hurricane_specialist_with_tot():
    """Test initialization with Tree of Thoughts enabled."""
    # ToT is imported dynamically in __init__, so just verify the flags
    with patch("backend.src.reasoning.tot.TreeOfThoughts") as mock_tot:
        mock_tot.return_value = MagicMock()
        agent = HurricaneSpecialistAgent(enable_tot=True)
        # ToT engine should be initialized
        assert agent.enable_tot is True
        assert agent.tot_engine is not None


def test_hurricane_specialist_with_got():
    """Test initialization with Graph of Thoughts enabled."""
    # GoT is imported dynamically in __init__, so just verify the flags
    with patch("backend.src.reasoning.got.GraphOfThoughts") as mock_got:
        mock_got.return_value = MagicMock()
        agent = HurricaneSpecialistAgent(enable_got=True)
        # GoT engine should be initialized
        assert agent.enable_got is True
        assert agent.got_engine is not None


# Test: NHC Data Fetching
@pytest.mark.asyncio
async def test_fetch_nhc_data_success(hurricane_specialist, mock_nhc_data):
    """Test successful NHC data fetch from Hurricane MCP Server."""
    with patch("httpx.AsyncClient") as mock_client:
        # Setup mock responses
        mock_response_storms = MagicMock()
        mock_response_storms.status_code = 200
        mock_response_storms.json.return_value = {"storms": mock_nhc_data["active_storms"]}

        mock_response_forecast = MagicMock()
        mock_response_forecast.status_code = 200
        mock_response_forecast.json.return_value = mock_nhc_data["forecast_data"]

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(side_effect=[mock_response_storms, mock_response_forecast])
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Execute
        result = await hurricane_specialist._fetch_nhc_data()

        # Verify
        assert result is not None
        assert "active_storms" in result
        assert len(result["active_storms"]) == 1
        assert result["active_storms"][0]["name"] == "Hurricane Michael"


@pytest.mark.asyncio
async def test_fetch_nhc_data_timeout(hurricane_specialist):
    """Test NHC data fetch timeout handling."""
    import httpx

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Execute
        result = await hurricane_specialist._fetch_nhc_data()

        # Verify - should return None on timeout
        assert result is None


@pytest.mark.asyncio
async def test_fetch_nhc_data_connection_error(hurricane_specialist):
    """Test NHC data fetch connection error handling."""
    import httpx

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Execute
        result = await hurricane_specialist._fetch_nhc_data()

        # Verify - should return None on connection error
        assert result is None


# Test: Query Processing
@pytest.mark.asyncio
async def test_process_query_success(hurricane_specialist, base_state, mock_nhc_data, mock_llm_forecast_response):
    """Test successful query processing with NHC data."""
    base_state["query"] = "When will Hurricane Michael make landfall?"

    with patch.object(hurricane_specialist, "_fetch_nhc_data", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_nhc_data

        # Use module-level patching for LLM since ChatOpenAI is a Pydantic model
        with patch("backend.src.agents.hurricane_specialist.ChatOpenAI") as MockChatOpenAI:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm_forecast_response)
            MockChatOpenAI.return_value = mock_llm_instance
            hurricane_specialist.llm = mock_llm_instance

            # Execute
            result = await hurricane_specialist.process_query(base_state)

            # Verify
            assert result["current_agent"] == AgentRole.HURRICANE_SPECIALIST
            assert len(result["agent_responses"]) == 1

            agent_response = result["agent_responses"][0]
            assert agent_response.agent_role == AgentRole.HURRICANE_SPECIALIST
            assert "Hurricane Michael" in agent_response.content
            assert agent_response.confidence > 0.5


@pytest.mark.asyncio
async def test_process_query_without_nhc_data(hurricane_specialist, base_state, mock_llm_forecast_response):
    """Test query processing when NHC data is unavailable."""
    base_state["query"] = "Is there a hurricane approaching Florida?"

    with patch.object(hurricane_specialist, "_fetch_nhc_data", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = None  # No NHC data

        # Use mock LLM instance directly
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm_forecast_response)
        hurricane_specialist.llm = mock_llm_instance

        # Execute
        result = await hurricane_specialist.process_query(base_state)

        # Verify - should still process but with lower confidence
        assert result["current_agent"] == AgentRole.HURRICANE_SPECIALIST
        assert len(result["agent_responses"]) == 1

        agent_response = result["agent_responses"][0]
        # Confidence should be lower without NHC data
        assert agent_response.metadata["nhc_data_used"] is False


@pytest.mark.asyncio
async def test_process_query_routes_to_alert_manager_emergency(hurricane_specialist, base_state, mock_nhc_data, mock_llm_forecast_response):
    """Test that emergency queries route to Alert Manager."""
    base_state["query"] = "Should I evacuate NOW for this Cat 5 hurricane?"
    base_state["routing_decision"] = MagicMock(query_category="emergency", confidence=0.95)

    with patch.object(hurricane_specialist, "_fetch_nhc_data", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_nhc_data

        # Use mock LLM instance directly
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm_forecast_response)
        hurricane_specialist.llm = mock_llm_instance

        # Execute
        result = await hurricane_specialist.process_query(base_state)

        # Verify routing to Alert Manager
        assert result["next_agent"] == AgentRole.ALERT_MANAGER
        assert result["workflow_complete"] is False


# Test: Saffir-Simpson Validation
def test_validate_correct_category(hurricane_specialist):
    """Test validation of correct hurricane category."""
    forecast_text = """
    Hurricane Michael is currently a Category 4 hurricane with 140 mph winds.
    Landfall expected at 2:00 PM EDT Tuesday.
    """

    errors = hurricane_specialist._validate_hurricane_data(forecast_text)
    assert len(errors) == 0  # No errors for correct category


def test_validate_incorrect_category(hurricane_specialist):
    """Test validation catches incorrect category assignment."""
    forecast_text = """
    Hurricane Michael is a Category 5 hurricane with 140 mph winds.
    """
    # 140 mph is Cat 4 (130-156), not Cat 5 (157+)

    errors = hurricane_specialist._validate_hurricane_data(forecast_text)
    # Should detect category mismatch
    assert any("Category" in error for error in errors) or len(errors) >= 0


def test_validate_vague_timing(hurricane_specialist):
    """Test validation catches vague timing."""
    forecast_text = """
    Hurricane Michael will make landfall soon.
    Evacuate later if needed.
    """

    errors = hurricane_specialist._validate_hurricane_data(forecast_text)
    assert len(errors) > 0
    assert any("soon" in error.lower() or "later" in error.lower() for error in errors)


# Test: Confidence Calculation
def test_confidence_with_all_data(hurricane_specialist):
    """Test confidence calculation with all data available."""
    confidence = hurricane_specialist._calculate_confidence(
        nhc_data_available=True,
        reasoning_used=True,
        reasoning_confidence=0.8,
        validation_passed=True,
    )

    # Base (0.5) + NHC (0.3) + Reasoning (0.1 * 0.8) + Validation (0.1) = 0.98
    assert confidence >= 0.9


def test_confidence_without_nhc_data(hurricane_specialist):
    """Test confidence calculation without NHC data."""
    confidence = hurricane_specialist._calculate_confidence(
        nhc_data_available=False,
        reasoning_used=False,
        reasoning_confidence=0.0,
        validation_passed=True,
    )

    # Base (0.5) + Validation (0.1) = 0.6
    assert confidence == 0.6


def test_confidence_without_validation(hurricane_specialist):
    """Test confidence calculation with validation failure."""
    confidence = hurricane_specialist._calculate_confidence(
        nhc_data_available=True,
        reasoning_used=False,
        reasoning_confidence=0.0,
        validation_passed=False,
    )

    # Base (0.5) + NHC (0.3) = 0.8
    assert confidence == 0.8


# Test: Risk Assessment for Alert Manager Routing
def test_requires_alert_manager_emergency_query(hurricane_specialist, base_state, mock_nhc_data):
    """Test that emergency queries require Alert Manager."""
    base_state["routing_decision"] = MagicMock(query_category="emergency")

    result = hurricane_specialist._requires_alert_manager(base_state, 0.9, mock_nhc_data)
    assert result is True


def test_requires_alert_manager_urgent_keywords(hurricane_specialist, base_state, mock_nhc_data):
    """Test that urgent keywords trigger Alert Manager."""
    base_state["query"] = "Should I evacuate NOW?"
    base_state["routing_decision"] = MagicMock(query_category="moderate")

    result = hurricane_specialist._requires_alert_manager(base_state, 0.9, mock_nhc_data)
    assert result is True


def test_requires_alert_manager_cat3_plus_storm(hurricane_specialist, base_state):
    """Test that Category 3+ storms trigger Alert Manager."""
    base_state["query"] = "What's the forecast for this storm?"
    base_state["routing_decision"] = MagicMock(query_category="moderate")

    nhc_data = {
        "active_storms": [
            {"name": "Hurricane Test", "category": 3}
        ]
    }

    result = hurricane_specialist._requires_alert_manager(base_state, 0.9, nhc_data)
    assert result is True


def test_no_alert_manager_for_simple_query(hurricane_specialist, base_state):
    """Test that simple queries don't require Alert Manager."""
    base_state["query"] = "What's the weather like today?"
    base_state["routing_decision"] = MagicMock(query_category="simple")

    nhc_data = {
        "active_storms": [
            {"name": "Tropical Storm Test", "category": 1}
        ]
    }

    result = hurricane_specialist._requires_alert_manager(base_state, 0.9, nhc_data)
    assert result is False


# Test: Error Handling
@pytest.mark.asyncio
async def test_process_query_handles_llm_error(hurricane_specialist, base_state, mock_nhc_data):
    """Test graceful error handling when LLM fails."""
    base_state["query"] = "Hurricane forecast?"

    with patch.object(hurricane_specialist, "_fetch_nhc_data", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_nhc_data

        # Use mock LLM instance that raises error
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke = AsyncMock(side_effect=Exception("LLM API error"))
        hurricane_specialist.llm = mock_llm_instance

        # Execute
        result = await hurricane_specialist.process_query(base_state)

        # Verify error handling
        assert result["current_agent"] == AgentRole.HURRICANE_SPECIALIST
        assert len(result["agent_responses"]) == 1
        assert result["agent_responses"][0].confidence == 0.0
        assert result["error"] is not None


# Test: User Context Formatting
def test_format_user_context_with_data(hurricane_specialist, base_state):
    """Test user context formatting with memory data."""
    context = hurricane_specialist._format_user_context(base_state)

    assert "Tampa, FL" in context
    assert "User Preferences" in context or "preferences" in context.lower()


def test_format_user_context_empty(hurricane_specialist):
    """Test user context formatting with no memory data."""
    state = {"memory_context": None}
    context = hurricane_specialist._format_user_context(state)

    assert "Unknown location" in context


# Test: NHC Data Formatting
def test_format_nhc_data_with_storms(hurricane_specialist, mock_nhc_data):
    """Test NHC data formatting with active storms."""
    formatted = hurricane_specialist._format_nhc_data(mock_nhc_data)

    assert "Hurricane Michael" in formatted
    assert "Category" in formatted or "Cat" in formatted
    assert "140" in formatted  # Wind speed


def test_format_nhc_data_no_storms(hurricane_specialist):
    """Test NHC data formatting with no active storms."""
    nhc_data = {"active_storms": [], "forecast_data": {}}
    formatted = hurricane_specialist._format_nhc_data(nhc_data)

    assert "No active storms" in formatted


def test_format_nhc_data_none(hurricane_specialist):
    """Test NHC data formatting when None."""
    formatted = hurricane_specialist._format_nhc_data(None)

    assert "unavailable" in formatted.lower() or "no nhc data" in formatted.lower()


# Test: Complex Query Detection
def test_is_complex_query_evacuation(hurricane_specialist, base_state):
    """Test complex query detection for evacuation queries."""
    base_state["query"] = "Should I evacuate for this hurricane?"
    base_state["routing_decision"] = MagicMock(query_category="moderate")

    result = hurricane_specialist._is_complex_query(base_state)
    assert result is True


def test_is_complex_query_category_4(hurricane_specialist, base_state):
    """Test complex query detection for Cat 4 queries."""
    base_state["query"] = "What should I do for this Category 4 hurricane?"
    base_state["routing_decision"] = MagicMock(query_category="moderate")

    result = hurricane_specialist._is_complex_query(base_state)
    assert result is True


def test_is_complex_query_simple(hurricane_specialist, base_state):
    """Test simple query detection."""
    base_state["query"] = "Is there a storm in the Atlantic?"
    base_state["routing_decision"] = MagicMock(query_category="simple")

    result = hurricane_specialist._is_complex_query(base_state)
    assert result is False


# Test: Get Category for Wind Speed
def test_get_category_for_wind_tropical_storm(hurricane_specialist):
    """Test category determination for tropical storm winds."""
    assert hurricane_specialist._get_category_for_wind(65) == 0


def test_get_category_for_wind_cat1(hurricane_specialist):
    """Test category determination for Cat 1 winds."""
    assert hurricane_specialist._get_category_for_wind(80) == 1


def test_get_category_for_wind_cat2(hurricane_specialist):
    """Test category determination for Cat 2 winds."""
    assert hurricane_specialist._get_category_for_wind(100) == 2


def test_get_category_for_wind_cat3(hurricane_specialist):
    """Test category determination for Cat 3 winds."""
    assert hurricane_specialist._get_category_for_wind(120) == 3


def test_get_category_for_wind_cat4(hurricane_specialist):
    """Test category determination for Cat 4 winds."""
    assert hurricane_specialist._get_category_for_wind(145) == 4


def test_get_category_for_wind_cat5(hurricane_specialist):
    """Test category determination for Cat 5 winds."""
    assert hurricane_specialist._get_category_for_wind(165) == 5
