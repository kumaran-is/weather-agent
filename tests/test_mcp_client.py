"""Tests for MCP weather client.

Tests the WeatherMCPClient async HTTP client implementation.
"""

import pytest


@pytest.mark.asyncio
async def test_mcp_client_initialize(mock_mcp_client):
    """Test MCP client initialization."""
    result = await mock_mcp_client.initialize()
    assert result is not None
    assert "status" in result


@pytest.mark.asyncio
async def test_get_current_weather(mock_mcp_client):
    """Test get_current_weather method."""
    result = await mock_mcp_client.get_current_weather("London")

    assert result is not None
    assert "location" in result
    assert result["location"] == "London"
    assert "temperature" in result
    assert "condition" in result
    assert "humidity" in result
    assert "wind_speed" in result


@pytest.mark.asyncio
async def test_get_forecast(mock_mcp_client):
    """Test get_forecast method."""
    result = await mock_mcp_client.get_forecast("London", days=5)

    assert result is not None
    assert "location" in result
    assert result["location"] == "London"
    assert "forecast_7day" in result
    assert isinstance(result["forecast_7day"], list)
    assert len(result["forecast_7day"]) > 0


@pytest.mark.asyncio
async def test_get_current_weather_empty_location(mock_mcp_client):
    """Test get_current_weather with empty location raises ValueError."""
    with pytest.raises(ValueError, match="Location cannot be empty"):
        await mock_mcp_client.get_current_weather("")


@pytest.mark.asyncio
async def test_get_forecast_invalid_days(mock_mcp_client):
    """Test get_forecast with invalid days raises ValueError."""
    with pytest.raises(ValueError, match="Days must be between 1 and 14"):
        await mock_mcp_client.get_forecast("London", days=20)

    with pytest.raises(ValueError, match="Days must be between 1 and 14"):
        await mock_mcp_client.get_forecast("London", days=0)
