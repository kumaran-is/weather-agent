"""Weather MCP Client - Async HTTP client for Weather MCP Server.

This module provides an async HTTP client that communicates with the Weather MCP
Server using the Model Context Protocol (MCP) over HTTP transport with JSON-RPC 2.0.

✅ Level 5c Production Features:
- Retry logic with exponential backoff (1s, 2s, 4s delays)
- Circuit breaker integration for failover
- Structured logging with correlation IDs
- Health monitoring support
- Configurable timeouts from settings
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

import httpx

from backend.config.settings import settings
from backend.src.mcp.failover import MCPFailoverHandler
from backend.src.mcp.logger import MCPLogger

logger = logging.getLogger(__name__)


class WeatherMCPClient:
    """Async HTTP client for Weather MCP Server.

    Implements JSON-RPC 2.0 protocol over HTTP transport to communicate
    with the Weather MCP Server. Handles SSE (Server-Sent Events) responses.

    Attributes:
        base_url: Base URL of the MCP server (default: http://localhost:8080)
        client: Persistent httpx.AsyncClient that maintains session cookies

    Example:
        >>> client = WeatherMCPClient(base_url="http://localhost:8080")
        >>> await client.initialize()
        >>> weather = await client.get_current_weather("London")
        >>> print(weather)
    """

    def __init__(self, base_url: str | None = None, enable_failover: bool | None = None):
        """Initialize Weather MCP client.

        Args:
            base_url: Base URL of the MCP server. If None, uses MCP_WEATHER_SERVER_URL
                     from environment or defaults to http://localhost:8080
            enable_failover: Enable failover to direct NHC API. If None, uses MCP_ENABLE_FAILOVER
        """
        self.base_url = base_url or settings.MCP_WEATHER_SERVER_URL
        self.client: httpx.AsyncClient | None = None
        self._initialized: bool = False
        self._session_id: str | None = None

        # MCP Integration (Level 5c)
        self.mcp_logger = MCPLogger(server_name="weather")
        self.failover_handler = MCPFailoverHandler(
            server_name="weather",
            mcp_logger=self.mcp_logger,
            enable_failover=enable_failover if enable_failover is not None else settings.MCP_ENABLE_FAILOVER
        )

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the persistent HTTP client with connection pooling.

        Returns:
            Persistent httpx.AsyncClient instance with session cookie support
                and optimized connection pooling
        """
        if self.client is None:
            # Use configurable timeout from settings (default: 30s)
            timeout_seconds = float(settings.MCP_WEATHER_REQUEST_TIMEOUT)
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds, connect=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=100
                ),
            )
        return self.client

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _call_with_retry(
        self,
        tool_name: str,
        request_func,
        correlation_id: str | None = None,
    ) -> dict[str, any]:
        """Call MCP tool with retry logic and exponential backoff.

        Args:
            tool_name: Name of the MCP tool being called
            request_func: Async function that makes the HTTP request
            correlation_id: Optional correlation ID (generated if None)

        Returns:
            Parsed MCP response

        Raises:
            httpx.HTTPError: If all retries fail

        Retry Strategy:
            - Max retries: 3 (from settings.MCP_WEATHER_MAX_RETRIES)
            - Exponential backoff: 1s, 2s, 4s (from settings.MCP_WEATHER_RETRY_DELAY)
            - Retry on: TimeoutError, ConnectError, 5xx errors
            - Don't retry on: 4xx errors (client errors)
        """
        if correlation_id is None:
            correlation_id = self.mcp_logger.generate_correlation_id()

        max_retries = settings.MCP_WEATHER_MAX_RETRIES
        base_delay_ms = settings.MCP_WEATHER_RETRY_DELAY

        # Log call start
        self.mcp_logger.log_mcp_call_start(
            tool=tool_name,
            correlation_id=correlation_id,
        )

        # Check circuit breaker before attempting call
        if self.failover_handler.should_failover(correlation_id):
            # Circuit breaker OPEN - use failover
            logger.warning(
                f"Circuit breaker OPEN for weather MCP, failover triggered | "
                f"correlation_id={correlation_id}"
            )
            raise NotImplementedError(
                "MCP failover to direct NHC API not yet implemented. "
                "Circuit breaker is OPEN - MCP server is unavailable."
            )

        # Retry loop
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Make the request
                response = await request_func()

                # Check HTTP status
                response.raise_for_status()

                # Parse SSE response
                result = self._parse_sse_response(response.text)

                # Success - record and log
                self.failover_handler.record_success()
                self.mcp_logger.log_mcp_call_success(
                    correlation_id=correlation_id,
                    tool=tool_name,
                    response_size=len(response.text),
                )

                return result

            except httpx.TimeoutException as e:
                last_exception = e
                self.failover_handler.record_failure()

                will_retry = (attempt < max_retries - 1)
                self.mcp_logger.log_mcp_call_timeout(
                    correlation_id=correlation_id,
                    tool=tool_name,
                    timeout_seconds=settings.MCP_WEATHER_REQUEST_TIMEOUT,
                    will_retry=will_retry,
                )

                if will_retry:
                    # Exponential backoff: 1s, 2s, 4s
                    delay_ms = base_delay_ms * (2 ** attempt)
                    await asyncio.sleep(delay_ms / 1000)
                else:
                    raise

            except httpx.ConnectError as e:
                last_exception = e
                self.failover_handler.record_failure()

                will_retry = (attempt < max_retries - 1)
                self.mcp_logger.log_mcp_call_failure(
                    correlation_id=correlation_id,
                    tool=tool_name,
                    error_type="ConnectError",
                    error_message=str(e),
                    will_retry=will_retry,
                )

                if will_retry:
                    delay_ms = base_delay_ms * (2 ** attempt)
                    await asyncio.sleep(delay_ms / 1000)
                else:
                    raise

            except httpx.HTTPStatusError as e:
                last_exception = e
                self.failover_handler.record_failure()

                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    self.mcp_logger.log_mcp_call_failure(
                        correlation_id=correlation_id,
                        tool=tool_name,
                        error_type=f"HTTPError{e.response.status_code}",
                        error_message=str(e),
                        will_retry=False,
                    )
                    raise

                # Retry on 5xx errors (server errors)
                will_retry = (attempt < max_retries - 1)
                self.mcp_logger.log_mcp_call_failure(
                    correlation_id=correlation_id,
                    tool=tool_name,
                    error_type=f"HTTPError{e.response.status_code}",
                    error_message=str(e),
                    will_retry=will_retry,
                )

                if will_retry:
                    delay_ms = base_delay_ms * (2 ** attempt)
                    await asyncio.sleep(delay_ms / 1000)
                else:
                    raise

        # All retries exhausted
        if last_exception:
            raise last_exception

    def _parse_sse_response(self, sse_text: str) -> dict[str, any]:
        """Parse Server-Sent Events (SSE) response from MCP server.

        Args:
            sse_text: Raw SSE response text

        Returns:
            Parsed JSON data from the SSE message

        Example SSE format:
            event: message
            data: {"result": {...}}
        """
        lines = sse_text.strip().split('\n')
        for line in lines:
            if line.startswith('data: '):
                data_json = line[6:]  # Remove 'data: ' prefix
                return json.loads(data_json)
        raise ValueError(f"No data found in SSE response: {sse_text}")

    async def initialize(self) -> dict[str, any]:
        """Initialize MCP session with the server.

        Establishes a session with the MCP server using the MCP initialization protocol.
        The session cookie is stored in the persistent client for subsequent requests.

        Returns:
            dict containing the initialization response from the server

        Raises:
            httpx.HTTPError: If the HTTP request fails

        Example:
            >>> client = WeatherMCPClient()
            >>> response = await client.initialize()
            >>> print(response)
        """
        client = self._get_client()
        response = await client.post(
            f"{self.base_url}/mcp",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-03-26"
            },
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "weather-agent",
                        "version": "1.0.0"
                    }
                }
            }
        )
        response.raise_for_status()

        # Extract session ID from response headers (MCP uses header-based sessions, not cookies)
        self._session_id = response.headers.get("mcp-session-id")
        if not self._session_id:
            raise ValueError("MCP server did not return a session ID")

        # Mark as initialized
        self._initialized = True

        # Parse SSE response
        return self._parse_sse_response(response.text)

    async def get_current_weather(self, location: str) -> dict[str, any]:
        """Get current weather conditions for a location.

        ✅ With retry logic, circuit breaker, and structured logging.

        Args:
            location: City name or coordinates (e.g., 'Seattle' or '47.6062,-122.3321')

        Returns:
            dict containing current weather data:
                - temperature: Current temperature
                - condition: Weather condition (e.g., 'sunny', 'rainy')
                - humidity: Humidity percentage
                - wind_speed: Wind speed

        Raises:
            httpx.HTTPError: If the HTTP request fails after retries
            ValueError: If location is empty or invalid

        Example:
            >>> client = WeatherMCPClient()
            >>> await client.initialize()
            >>> weather = await client.get_current_weather("London")
            >>> print(f"Temperature: {weather['temperature']}°C")
        """
        if not location:
            raise ValueError("Location cannot be empty")

        client = self._get_client()

        # Prepare headers with session ID
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26"
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        # Define request function
        async def make_request():
            return await client.post(
                f"{self.base_url}/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": str(datetime.now(UTC).timestamp()),
                    "method": "tools/call",
                    "params": {
                        "name": "get_current_weather",
                        "arguments": {"city": location}
                    }
                }
            )

        # Call with retry logic
        return await self._call_with_retry(
            tool_name="get_current_weather",
            request_func=make_request,
        )

    async def get_forecast(self, location: str, days: int = 5) -> dict[str, any]:
        """Get weather forecast for a location.

        ✅ With retry logic, circuit breaker, and structured logging.

        Args:
            location: City name or coordinates (e.g., 'Seattle' or '47.6062,-122.3321')
            days: Number of days to forecast (1-7, default 5)

        Returns:
            dict containing forecast data with daily predictions:
                - forecast_7day: List of daily forecasts
                - Each forecast includes: day, high, low, condition

        Raises:
            httpx.HTTPError: If the HTTP request fails after retries
            ValueError: If location is empty or days is out of range

        Example:
            >>> client = WeatherMCPClient()
            >>> await client.initialize()
            >>> forecast = await client.get_forecast("London", days=5)
            >>> for day in forecast['forecast_7day']:
            ...     print(f"{day['day']}: {day['high']}°C")
        """
        if not location:
            raise ValueError("Location cannot be empty")
        if not 1 <= days <= 7:
            raise ValueError("Days must be between 1 and 7")

        client = self._get_client()

        # Prepare headers with session ID
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26"
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        # Define request function
        async def make_request():
            return await client.post(
                f"{self.base_url}/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": str(datetime.now(UTC).timestamp()),
                    "method": "tools/call",
                    "params": {
                        "name": "get_weather_forecast",
                        "arguments": {
                            "city": location,
                            "days": days
                        }
                    }
                }
            )

        # Call with retry logic
        return await self._call_with_retry(
            tool_name="get_weather_forecast",
            request_func=make_request,
        )

    async def retrieve_weather_context(self, query: str) -> dict[str, any]:
        """Retrieve weather context for AI agent queries.

        ✅ With retry logic, circuit breaker, and structured logging.

        This method extracts weather information from natural language queries
        and returns relevant context for the AI agent. Useful when the query
        contains implicit location references.

        Args:
            query: Natural language query containing city reference
                  (e.g., "weather in Paris for travel", "Should I bring an umbrella to London?")

        Returns:
            dict containing weather context extracted from the query

        Raises:
            httpx.HTTPError: If the HTTP request fails after retries
            ValueError: If query is empty

        Example:
            >>> client = WeatherMCPClient()
            >>> await client.initialize()
            >>> context = await client.retrieve_weather_context("weather in Paris for travel")
            >>> print(context)
        """
        if not query:
            raise ValueError("Query cannot be empty")

        client = self._get_client()

        # Prepare headers with session ID
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26"
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        # Define request function
        async def make_request():
            return await client.post(
                f"{self.base_url}/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": str(datetime.now(UTC).timestamp()),
                    "method": "tools/call",
                    "params": {
                        "name": "retrieve_weather_context",
                        "arguments": {
                            "query": query
                        }
                    }
                }
            )

        # Call with retry logic
        return await self._call_with_retry(
            tool_name="retrieve_weather_context",
            request_func=make_request,
        )
