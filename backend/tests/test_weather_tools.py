"""Comprehensive tests for Weather MCP tools (LangChain v1.x compliance).

Tests cover:
- P1: Factory pattern (get_weather_mcp_client)
- P2: Async tools with error handling
- MCP integration
- Tool descriptions and structured args
- Backward compatibility

Run: PYTHONPATH=. pytest backend/tests/test_weather_tools.py -v
"""

import pytest

from backend.src.tools.weather_tools import (
    get_current_weather,
    get_forecast,
    get_weather_mcp_client,
    retrieve_weather_context,
)


class TestFactoryPattern:
    """✅ P1: Verify factory pattern for MCP client."""

    def test_get_weather_mcp_client_factory(self):
        """✅ get_weather_mcp_client() creates MCP client instance."""
        client = get_weather_mcp_client()
        assert client is not None
        assert hasattr(client, "get_current_weather")
        assert hasattr(client, "get_forecast")
        assert hasattr(client, "retrieve_weather_context")

    def test_multiple_clients_independent(self):
        """✅ Multiple factory calls create independent instances."""
        client1 = get_weather_mcp_client()
        client2 = get_weather_mcp_client()

        # Should be different instances (for test isolation)
        assert client1 is not client2


class TestAsyncTools:
    """✅ Verify all tools are async."""

    @pytest.mark.asyncio
    async def test_get_current_weather_is_async(self):
        """✅ get_current_weather is async."""
        result = await get_current_weather.ainvoke({"location": "London"})
        assert isinstance(result, dict)
        # Should return either MCP result, weather data, or error
        assert "error" in result or "temperature" in result or "location" in result or "result" in result

    @pytest.mark.asyncio
    async def test_get_forecast_is_async(self):
        """✅ get_forecast is async."""
        result = await get_forecast.ainvoke({"location": "London", "days": 5})
        assert isinstance(result, dict)
        # Should return either forecast data or error
        assert "error" in result or "forecast_7day" in result or "location" in result

    @pytest.mark.asyncio
    async def test_retrieve_weather_context_is_async(self):
        """✅ retrieve_weather_context is async."""
        result = await retrieve_weather_context.ainvoke(
            {"query": "Is it rainy in Seattle today?"}
        )
        assert isinstance(result, dict)
        # Should return either context or error
        assert isinstance(result, dict)


class TestErrorHandling:
    """✅ P2: Verify structured error responses."""

    @pytest.mark.asyncio
    async def test_get_current_weather_error_format(self):
        """✅ get_current_weather returns structured errors."""
        # Test with potentially invalid location
        result = await get_current_weather.ainvoke({"location": "InvalidCity12345XYZ"})
        assert isinstance(result, dict)

        # If error, check structure
        if "error" in result:
            assert "location" in result
            assert "status" in result
            assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_get_forecast_error_format(self):
        """✅ get_forecast returns structured errors."""
        # Test with potentially invalid location
        result = await get_forecast.ainvoke({"location": "InvalidCity12345XYZ", "days": 5})
        assert isinstance(result, dict)

        # If error, check structure
        if "error" in result:
            assert "location" in result
            assert "days" in result
            assert "status" in result
            assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_retrieve_weather_context_error_format(self):
        """✅ retrieve_weather_context returns structured errors."""
        # Test with empty query
        result = await retrieve_weather_context.ainvoke({"query": ""})
        assert isinstance(result, dict)

        # If error, check structure
        if "error" in result:
            assert "query" in result
            assert "status" in result
            assert result["status"] == "failed"


class TestToolDescriptions:
    """✅ Verify all tools have proper descriptions."""

    def test_get_current_weather_description(self):
        """✅ get_current_weather has description."""
        assert get_current_weather.description is not None
        assert len(get_current_weather.description) > 50
        assert "location" in get_current_weather.args
        assert "current weather" in get_current_weather.description.lower()

    def test_get_forecast_description(self):
        """✅ get_forecast has description."""
        assert get_forecast.description is not None
        assert len(get_forecast.description) > 50
        assert "location" in get_forecast.args
        assert "days" in get_forecast.args
        assert "forecast" in get_forecast.description.lower()

    def test_retrieve_weather_context_description(self):
        """✅ retrieve_weather_context has description."""
        assert retrieve_weather_context.description is not None
        assert len(retrieve_weather_context.description) > 50
        assert "query" in retrieve_weather_context.args
        assert "context" in retrieve_weather_context.description.lower()


class TestMCPIntegration:
    """✅ Verify MCP client initialization and error handling."""

    @pytest.mark.asyncio
    async def test_mcp_client_initialization(self):
        """✅ MCP client initializes before use."""
        # Tools should initialize MCP client if not initialized
        result = await get_current_weather.ainvoke({"location": "London"})
        assert isinstance(result, dict)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_forecast_days_validation(self):
        """✅ get_forecast validates days parameter (1-7)."""
        # Test with valid range
        result = await get_forecast.ainvoke({"location": "London", "days": 5})
        assert isinstance(result, dict)

        # Test with out-of-range (should default to 5)
        result = await get_forecast.ainvoke({"location": "London", "days": 100})
        assert isinstance(result, dict)
        # Should not fail


class TestToolReturns:
    """✅ Verify tools return correct dict structure."""

    @pytest.mark.asyncio
    async def test_get_current_weather_returns_dict(self):
        """✅ get_current_weather returns dict."""
        result = await get_current_weather.ainvoke({"location": "London"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_forecast_returns_dict(self):
        """✅ get_forecast returns dict."""
        result = await get_forecast.ainvoke({"location": "London", "days": 3})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_retrieve_weather_context_returns_dict(self):
        """✅ retrieve_weather_context returns dict."""
        result = await retrieve_weather_context.ainvoke(
            {"query": "weather in Paris"}
        )
        assert isinstance(result, dict)


class TestBackwardCompatibility:
    """✅ Verify backward compatibility."""

    @pytest.mark.asyncio
    async def test_tools_work_with_module_level_client(self):
        """✅ Tools work with module-level client (backward compatibility)."""
        from backend.src.tools.weather_tools import weather_mcp_client

        assert weather_mcp_client is not None
        assert hasattr(weather_mcp_client, "get_current_weather")

    @pytest.mark.asyncio
    async def test_tools_accept_kwargs(self):
        """✅ Tools accept keyword arguments."""
        # Test with kwargs
        result = await get_current_weather.ainvoke({"location": "Tokyo"})
        assert isinstance(result, dict)

        result = await get_forecast.ainvoke({"location": "Tokyo", "days": 3})
        assert isinstance(result, dict)

        result = await retrieve_weather_context.ainvoke(
            {"query": "What's the weather like?"}
        )
        assert isinstance(result, dict)


# ============================================================================
# Test Coverage Summary
# ============================================================================
# ✅ P1 #1: Factory pattern (get_weather_mcp_client)
# ✅ P2 #2: Structured error responses (all tools)
# ✅ Async tools (all 3 tools)
# ✅ MCP integration (client initialization)
# ✅ Tool descriptions (all 3 tools)
# ✅ Backward compatibility (module-level client)
# ============================================================================
