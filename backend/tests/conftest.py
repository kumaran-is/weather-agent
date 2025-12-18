"""Pytest configuration and fixtures for backend tests (Level 4a).

This module provides shared fixtures for multi-agent system tests.
"""

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for all tests.

    This fixture automatically runs for all tests to provide necessary
    environment variables without requiring a .env file.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-testing-only")
    monkeypatch.setenv("MCP_WEATHER_SERVER_URL", "http://localhost:8080")
    monkeypatch.setenv("MCP_HURRICANE_SERVER_URL", "http://localhost:8081")
    monkeypatch.setenv("MCP_HURRICANE_SERVER_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-for-testing")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "testpassword")


@pytest.fixture
def mock_llm():
    """Mock LLM for agent testing.

    Returns:
        Mock LLM with ainvoke method
    """
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Mocked LLM response"))
    return llm


@pytest.fixture
def sample_hurricane_data():
    """Sample hurricane data from MCP server for testing.

    Returns:
        dict: Mock hurricane data
    """
    return {
        "active_storms": [
            {
                "name": "Hurricane Milton",
                "category": 4,
                "wind_speed": 145,
                "wind_unit": "mph",
                "location": "Gulf of Mexico",
                "latitude": 27.5,
                "longitude": -90.2,
                "movement": "NE at 12 mph",
                "pressure": 950,
                "pressure_unit": "mb"
            }
        ],
        "storm_details": {
            "name": "Milton",
            "category": 4,
            "max_sustained_winds": 145,
            "wind_unit": "mph",
            "central_pressure": 950,
            "pressure_unit": "mb",
            "movement_direction": "NE",
            "movement_speed": 12,
            "current_location": {
                "latitude": 27.5,
                "longitude": -90.2
            },
            "forecast_track": [
                {"hours": 24, "latitude": 28.5, "longitude": -88.5, "intensity": 150},
                {"hours": 48, "latitude": 29.8, "longitude": -86.0, "intensity": 145},
                {"hours": 72, "latitude": 31.2, "longitude": -83.5, "intensity": 130}
            ],
            "potential_impacts": {
                "storm_surge": "13-18 feet",
                "rainfall": "8-12 inches",
                "wind_damage": "Catastrophic"
            }
        },
        "evacuation_zones": {
            "zone_a": {
                "status": "mandatory",
                "counties": ["Pinellas", "Hillsborough", "Manatee"],
                "deadline": "2024-10-08 18:00 EDT"
            },
            "zone_b": {
                "status": "voluntary",
                "counties": ["Pinellas", "Hillsborough"],
                "deadline": "2024-10-09 06:00 EDT"
            }
        }
    }


@pytest.fixture
def sample_routing_decision():
    """Sample routing decision from Triage Agent.

    Returns:
        dict: Mock routing decision
    """
    from datetime import datetime

    from backend.src.models.multi_agent import AgentRole, RoutingDecision

    return RoutingDecision(
        next_agent=AgentRole.HURRICANE_SPECIALIST,
        confidence=0.92,
        rationale="Hurricane-specific query requiring domain expertise",
        timestamp=datetime.now(UTC),
        query_category="hurricane_forecast"
    )


@pytest.fixture
def sample_multi_agent_state():
    """Sample MultiAgentState for workflow testing.

    Returns:
        dict: Initial workflow state
    """
    from datetime import datetime

    return {
        "query": "When will Hurricane Milton make landfall in Tampa?",
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "timeout_ms": 3000,
        "current_agent": None,
        "routing_decision": None,
        "agent_responses": [],
        "next_agent": None,
        "workflow_complete": False,
        "final_response": None,
        "error": None,
        "total_execution_time_ms": 0.0,
    }
