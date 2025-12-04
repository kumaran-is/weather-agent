"""Weather MCP Client - Async HTTP client for Weather MCP Server.

This module provides an async HTTP client that communicates with the Weather MCP
Server using the Model Context Protocol (MCP) over HTTP transport with JSON-RPC 2.0.

Level 1 Implementation:
- Simple HTTP MCP client with session management
- 2 weather methods: get_current_weather, get_forecast
- Basic try/except error handling
- SSE (Server-Sent Events) response parsing
- NO health monitoring, failover, or structured logging (deferred to L5c)
"""

from typing import Dict, Any
import httpx
from datetime import datetime, timezone
import os
import json


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

    def __init__(self, base_url: str | None = None):
        """Initialize Weather MCP client.

        Args:
            base_url: Base URL of the MCP server. If None, uses MCP_WEATHER_SERVER_URL
                     from environment or defaults to http://localhost:8080
        """
        self.base_url = base_url or os.getenv("MCP_WEATHER_SERVER_URL", "http://localhost:8080")
        self.client: httpx.AsyncClient | None = None
        self._initialized: bool = False
        self._session_id: str | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the persistent HTTP client.

        Returns:
            Persistent httpx.AsyncClient instance with session cookie support
        """
        if self.client is None:
            self.client = httpx.AsyncClient()
        return self.client

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    def _parse_sse_response(self, sse_text: str) -> Dict[str, Any]:
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

    async def initialize(self) -> Dict[str, Any]:
        """Initialize MCP session with the server.

        Establishes a session with the MCP server using the MCP initialization protocol.
        The session cookie is stored in the persistent client for subsequent requests.

        Returns:
            Dict containing the initialization response from the server

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
                "MCP-Protocol-Version": "2024-11-05"
            },
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
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

    async def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather conditions for a location.

        Args:
            location: City name or coordinates (e.g., 'Seattle' or '47.6062,-122.3321')

        Returns:
            Dict containing current weather data:
                - temperature: Current temperature
                - condition: Weather condition (e.g., 'sunny', 'rainy')
                - humidity: Humidity percentage
                - wind_speed: Wind speed

        Raises:
            httpx.HTTPError: If the HTTP request fails
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
            "MCP-Protocol-Version": "2024-11-05"
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        response = await client.post(
            f"{self.base_url}/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": str(datetime.now(timezone.utc).timestamp()),
                "method": "tools/call",
                "params": {
                    "name": "get_current_weather",
                    "arguments": {"city": location}
                }
            }
        )
        response.raise_for_status()
        # Parse SSE response
        return self._parse_sse_response(response.text)

    async def get_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        """Get weather forecast for a location.

        Args:
            location: City name or coordinates (e.g., 'Seattle' or '47.6062,-122.3321')
            days: Number of days to forecast (1-7, default 5)

        Returns:
            Dict containing forecast data with daily predictions:
                - forecast_7day: List of daily forecasts
                - Each forecast includes: day, high, low, condition

        Raises:
            httpx.HTTPError: If the HTTP request fails
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
            "MCP-Protocol-Version": "2024-11-05"
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        response = await client.post(
            f"{self.base_url}/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": str(datetime.now(timezone.utc).timestamp()),
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
        response.raise_for_status()
        # Parse SSE response
        return self._parse_sse_response(response.text)

    async def retrieve_weather_context(self, query: str) -> Dict[str, Any]:
        """Retrieve weather context for AI agent queries.

        This method extracts weather information from natural language queries
        and returns relevant context for the AI agent. Useful when the query
        contains implicit location references.

        Args:
            query: Natural language query containing city reference
                  (e.g., "weather in Paris for travel", "Should I bring an umbrella to London?")

        Returns:
            Dict containing weather context extracted from the query

        Raises:
            httpx.HTTPError: If the HTTP request fails
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
            "MCP-Protocol-Version": "2024-11-05"
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        response = await client.post(
            f"{self.base_url}/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": str(datetime.now(timezone.utc).timestamp()),
                "method": "tools/call",
                "params": {
                    "name": "retrieve_weather_context",
                    "arguments": {
                        "query": query
                    }
                }
            }
        )
        response.raise_for_status()
        # Parse SSE response
        return self._parse_sse_response(response.text)
