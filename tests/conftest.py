"""Pytest configuration and fixtures for Weather AI Agent tests.

This module provides shared fixtures for all test modules.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for all tests.

    This fixture automatically runs for all tests to provide necessary
    environment variables without requiring a .env file.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-testing-only-12345678901234567890")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-for-testing-only-1234567890")
    monkeypatch.setenv("MCP_WEATHER_SERVER_URL", "http://localhost:8080")
    monkeypatch.setenv("MCP_HURRICANE_SERVER_URL", "http://localhost:8081")


@pytest.fixture
def mock_mcp_client(monkeypatch):
    """Mock MCP client for testing without real API calls.

    This fixture mocks the WeatherMCPClient to return predefined
    weather data without making actual HTTP requests to the MCP server.

    Returns:
        Mock WeatherMCPClient with mocked methods
    """
    # Lazy import to avoid circular imports
    from backend.src.mcp.weather_client import WeatherMCPClient

    async def mock_initialize():
        """Mock initialize method."""
        return {"status": "initialized"}

    async def mock_get_current_weather(location: str):
        """Mock get_current_weather method."""
        return {
            "location": location,
            "temperature": 72,
            "condition": "sunny",
            "humidity": 45,
            "wind_speed": 10,
            "unit": "imperial"
        }

    async def mock_get_forecast(location: str, days: int = 7):
        """Mock get_forecast method."""
        return {
            "location": location,
            "forecast_7day": [
                {"day": "Monday", "high": 75, "low": 60, "condition": "sunny"},
                {"day": "Tuesday", "high": 78, "low": 62, "condition": "cloudy"},
                {"day": "Wednesday", "high": 72, "low": 58, "condition": "rainy"},
                {"day": "Thursday", "high": 70, "low": 56, "condition": "sunny"},
                {"day": "Friday", "high": 73, "low": 59, "condition": "partly_cloudy"}
            ]
        }

    # Patch the WeatherMCPClient methods
    monkeypatch.setattr(WeatherMCPClient, "initialize", mock_initialize)
    monkeypatch.setattr(WeatherMCPClient, "get_current_weather", mock_get_current_weather)
    monkeypatch.setattr(WeatherMCPClient, "get_forecast", mock_get_forecast)

    return WeatherMCPClient()


@pytest.fixture
def mock_llm():
    """Mock LLM for agent testing.

    This fixture provides a mocked LLM that can be used for testing
    agents without making actual API calls to OpenAI.

    Returns:
        Mock LLM with ainvoke method
    """
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value="Mocked LLM response")
    return llm


@pytest.fixture
def mock_agent_executor(monkeypatch):
    """Mock AgentExecutor to avoid calling real OpenAI API in tests.

    This fixture mocks the query_weather function to return a predefined
    response without invoking the actual agent or making API calls.

    Returns:
        None (patches query_weather function directly)
    """
    async def mock_query_weather(user_query: str) -> str:
        """Mock query_weather that returns a sample weather response."""
        return f"The weather in {user_query.split('in')[-1].strip().rstrip('?')} is 72°F and sunny."

    # Patch where it's imported in main.py
    monkeypatch.setattr("backend.src.api.main.query_weather", mock_query_weather)


@pytest.fixture
def sample_weather_data():
    """Sample weather data for testing.

    Returns:
        dict: Sample weather response data
    """
    return {
        "location": "London",
        "temperature": 15,
        "condition": "rainy",
        "humidity": 85,
        "wind_speed": 12,
        "unit": "metric"
    }


@pytest.fixture
def sample_forecast_data():
    """Sample forecast data for testing.

    Returns:
        dict: Sample forecast response data
    """
    return {
        "location": "London",
        "forecast_7day": [
            {"day": "Monday", "high": 18, "low": 12, "condition": "rainy"},
            {"day": "Tuesday", "high": 20, "low": 14, "condition": "cloudy"},
            {"day": "Wednesday", "high": 16, "low": 10, "condition": "rainy"},
            {"day": "Thursday", "high": 19, "low": 13, "condition": "sunny"},
            {"day": "Friday", "high": 21, "low": 15, "condition": "partly_cloudy"}
        ]
    }


@pytest.fixture
def sample_hurricane_cat2():
    """Sample Category 2 hurricane data for testing.

    Returns:
        dict: Hurricane alert data
    """
    return {
        "user_id": "test_user",
        "session_id": "test_session",
        "current_query": "Category 2 hurricane approaching",
        "current_step": "input",
        "approved": False,
        "hurricane_category": 2,
        "alert_message": "Category 2 Hurricane Julia - 95 mph winds"
    }


@pytest.fixture
def sample_hurricane_cat4():
    """Sample Category 4 hurricane data for testing.

    Returns:
        dict: Hurricane alert data
    """
    return {
        "user_id": "test_user",
        "session_id": "test_session",
        "current_query": "Category 4 hurricane approaching",
        "current_step": "input",
        "approved": False,
        "hurricane_category": 4,
        "alert_message": "Category 4 Hurricane Ida - 140 mph winds"
    }
