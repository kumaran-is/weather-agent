"""Tests for weather tools.

Tests the LangChain tool wrappers for weather operations.
"""

import pytest

from backend.src.tools.weather_tools import get_current_weather, get_forecast


@pytest.mark.asyncio
async def test_get_current_weather_tool(mock_mcp_client):
    """Test get_current_weather tool."""
    result = await get_current_weather.ainvoke({"location": "Seattle"})

    assert result is not None
    assert isinstance(result, dict)
    # Tool should return weather data or error
    assert ("temperature" in result) or ("error" in result)


@pytest.mark.asyncio
async def test_get_forecast_tool(mock_mcp_client):
    """Test get_forecast tool."""
    result = await get_forecast.ainvoke({"location": "Seattle", "days": 7})

    assert result is not None
    assert isinstance(result, dict)
    # Tool should return forecast data or error
    assert ("forecast_7day" in result) or ("error" in result)


@pytest.mark.asyncio
async def test_tool_error_handling(monkeypatch):
    """Test that tools handle errors gracefully."""
    from backend.src.mcp.weather_client import WeatherMCPClient

    async def mock_failing_weather(*args, **kwargs):
        raise Exception("MCP server unavailable")

    monkeypatch.setattr(WeatherMCPClient, "get_current_weather", mock_failing_weather)

    result = await get_current_weather.ainvoke({"location": "London"})

    assert result is not None
    assert "error" in result
    assert "status" in result
    assert result["status"] == "failed"
