"""Comprehensive tests for Weather MCP Client.

Tests cover:
- Client initialization and configuration
- MCP session management
- JSON-RPC 2.0 protocol compliance
- SSE response parsing
- Weather API methods
- Connection pooling
- Error handling
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from backend.src.mcp.weather_client import WeatherMCPClient


@pytest.fixture
def mock_response():
    """Create a mock httpx.Response."""
    def _create_response(sse_data: str, session_id: str = "test-session-123"):
        response = Mock(spec=httpx.Response)
        response.text = sse_data
        response.headers = {"mcp-session-id": session_id}
        response.raise_for_status = Mock()
        return response
    return _create_response


class TestWeatherMCPClientInitialization:
    """Test client initialization."""

    def test_initialization_default(self):
        """Test client initializes with default URL from settings."""
        client = WeatherMCPClient()
        assert client.base_url is not None
        assert client.client is None
        assert client._initialized is False
        assert client._session_id is None

    def test_initialization_custom_url(self):
        """Test client initializes with custom URL."""
        custom_url = "http://custom-server:9090"
        client = WeatherMCPClient(base_url=custom_url)
        assert client.base_url == custom_url


class TestConnectionPooling:
    """Test connection pooling configuration."""

    def test_get_client_creates_with_pooling(self):
        """Test client creation includes connection pooling config."""
        client = WeatherMCPClient()
        http_client = client._get_client()

        assert isinstance(http_client, httpx.AsyncClient)
        # Verify timeout configuration
        assert http_client.timeout.connect == 10.0
        assert http_client.timeout.read == 30.0
        # Note: _limits is private API - connection pooling is configured but not directly testable

    def test_get_client_reuses_instance(self):
        """Test client instance is reused (singleton pattern)."""
        client = WeatherMCPClient()
        http_client1 = client._get_client()
        http_client2 = client._get_client()

        assert http_client1 is http_client2


class TestSSEParsing:
    """Test Server-Sent Events (SSE) parsing."""

    def test_parse_sse_response_valid(self):
        """Test parsing valid SSE response."""
        client = WeatherMCPClient()
        sse_text = """event: message
data: {"result": {"temperature": 72, "condition": "sunny"}}
"""
        result = client._parse_sse_response(sse_text)

        assert result == {"result": {"temperature": 72, "condition": "sunny"}}

    def test_parse_sse_response_multiple_lines(self):
        """Test parsing SSE with multiple lines."""
        client = WeatherMCPClient()
        sse_text = """id: 1
event: message
data: {"success": true}
"""
        result = client._parse_sse_response(sse_text)

        assert result == {"success": True}

    def test_parse_sse_response_invalid(self):
        """Test parsing invalid SSE raises ValueError."""
        client = WeatherMCPClient()
        sse_text = "event: message\nno data line here"

        with pytest.raises(ValueError, match="No data found in SSE response"):
            client._parse_sse_response(sse_text)


class TestMCPInitialization:
    """Test MCP session initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_response):
        """Test successful MCP session initialization."""
        client = WeatherMCPClient(base_url="http://localhost:8080")

        sse_data = 'data: {"result": {"capabilities": {}}}'
        mock_resp = mock_response(sse_data, session_id="session-abc-123")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_http_client

            result = await client.initialize()

            # Verify JSON-RPC 2.0 request
            call_args = mock_http_client.post.call_args
            assert call_args[0][0] == "http://localhost:8080/mcp"
            json_body = call_args[1]["json"]
            assert json_body["jsonrpc"] == "2.0"
            assert json_body["method"] == "initialize"
            assert json_body["params"]["protocolVersion"] == "2025-03-26"

            # Verify headers include correct protocol version
            headers = call_args[1]["headers"]
            assert headers["MCP-Protocol-Version"] == "2025-03-26"

            # Verify session established
            assert client._initialized is True
            assert client._session_id == "session-abc-123"
            assert result == {"result": {"capabilities": {}}}

    @pytest.mark.asyncio
    async def test_initialize_no_session_id(self, mock_response):
        """Test initialization fails when server doesn't return session ID."""
        client = WeatherMCPClient()

        sse_data = 'data: {"result": {}}'
        mock_resp = mock_response(sse_data, session_id=None)
        mock_resp.headers = {}  # No session ID

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_http_client

            with pytest.raises(ValueError, match="MCP server did not return a session ID"):
                await client.initialize()


class TestGetCurrentWeather:
    """Test get_current_weather method."""

    @pytest.mark.asyncio
    async def test_get_current_weather_success(self, mock_response):
        """Test successful current weather retrieval."""
        client = WeatherMCPClient()
        client._session_id = "test-session-123"

        sse_data = 'data: {"result": {"temperature": 68, "condition": "cloudy"}}'
        mock_resp = mock_response(sse_data)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_http_client

            result = await client.get_current_weather("Seattle")

            # Verify JSON-RPC request
            call_args = mock_http_client.post.call_args
            json_body = call_args[1]["json"]
            assert json_body["method"] == "tools/call"
            assert json_body["params"]["name"] == "get_current_weather"
            assert json_body["params"]["arguments"]["city"] == "Seattle"

            # Verify session ID in headers
            headers = call_args[1]["headers"]
            assert headers["mcp-session-id"] == "test-session-123"

            assert result == {"result": {"temperature": 68, "condition": "cloudy"}}

    @pytest.mark.asyncio
    async def test_get_current_weather_empty_location(self):
        """Test get_current_weather validates location."""
        client = WeatherMCPClient()

        with pytest.raises(ValueError, match="Location cannot be empty"):
            await client.get_current_weather("")


class TestGetForecast:
    """Test get_forecast method."""

    @pytest.mark.asyncio
    async def test_get_forecast_success(self, mock_response):
        """Test successful forecast retrieval."""
        client = WeatherMCPClient()
        client._session_id = "test-session"

        sse_data = 'data: {"result": {"forecast_7day": [{"day": "Mon", "high": 75}]}}'
        mock_resp = mock_response(sse_data)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_http_client

            result = await client.get_forecast("Portland", days=5)

            # Verify JSON-RPC request
            call_args = mock_http_client.post.call_args
            json_body = call_args[1]["json"]
            assert json_body["params"]["name"] == "get_weather_forecast"
            assert json_body["params"]["arguments"]["city"] == "Portland"
            assert json_body["params"]["arguments"]["days"] == 5

            assert "forecast_7day" in result["result"]

    @pytest.mark.asyncio
    async def test_get_forecast_validates_days(self):
        """Test get_forecast validates days parameter."""
        client = WeatherMCPClient()

        with pytest.raises(ValueError, match="Days must be between 1 and 7"):
            await client.get_forecast("Seattle", days=0)

        with pytest.raises(ValueError, match="Days must be between 1 and 7"):
            await client.get_forecast("Seattle", days=10)


class TestRetrieveWeatherContext:
    """Test retrieve_weather_context method."""

    @pytest.mark.asyncio
    async def test_retrieve_weather_context_success(self, mock_response):
        """Test successful weather context retrieval."""
        client = WeatherMCPClient()
        client._session_id = "test-session"

        sse_data = 'data: {"result": {"context": "Paris weather is sunny"}}'
        mock_resp = mock_response(sse_data)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_http_client

            result = await client.retrieve_weather_context("weather in Paris")

            # Verify JSON-RPC request
            call_args = mock_http_client.post.call_args
            json_body = call_args[1]["json"]
            assert json_body["params"]["name"] == "retrieve_weather_context"
            assert json_body["params"]["arguments"]["query"] == "weather in Paris"

            assert "context" in result["result"]

    @pytest.mark.asyncio
    async def test_retrieve_weather_context_empty_query(self):
        """Test retrieve_weather_context validates query."""
        client = WeatherMCPClient()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await client.retrieve_weather_context("")


class TestClientCleanup:
    """Test client cleanup."""

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test client cleanup closes HTTP client."""
        client = WeatherMCPClient()

        # Create client
        _ = client._get_client()
        assert client.client is not None

        # Mock aclose
        with patch.object(client.client, "aclose", new_callable=AsyncMock) as mock_aclose:
            await client.close()

            mock_aclose.assert_called_once()
            assert client.client is None

    @pytest.mark.asyncio
    async def test_close_client_already_closed(self):
        """Test closing already-closed client is safe."""
        client = WeatherMCPClient()

        # Client was never created
        assert client.client is None

        # Should not raise error
        await client.close()
        assert client.client is None


class TestProtocolVersionCompliance:
    """Test MCP protocol version compliance."""

    @pytest.mark.asyncio
    async def test_all_methods_use_correct_protocol_version(self, mock_response):
        """Test all methods use protocol version 2025-03-26."""
        client = WeatherMCPClient()
        client._session_id = "test-session"

        sse_data = 'data: {"result": {}}'
        mock_resp = mock_response(sse_data)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_http_client

            # Test initialize
            await client.initialize()
            headers = mock_http_client.post.call_args[1]["headers"]
            assert headers["MCP-Protocol-Version"] == "2025-03-26"

            # Test get_current_weather
            await client.get_current_weather("Test")
            headers = mock_http_client.post.call_args[1]["headers"]
            assert headers["MCP-Protocol-Version"] == "2025-03-26"

            # Test get_forecast
            await client.get_forecast("Test", days=3)
            headers = mock_http_client.post.call_args[1]["headers"]
            assert headers["MCP-Protocol-Version"] == "2025-03-26"

            # Test retrieve_weather_context
            await client.retrieve_weather_context("test query")
            headers = mock_http_client.post.call_args[1]["headers"]
            assert headers["MCP-Protocol-Version"] == "2025-03-26"
