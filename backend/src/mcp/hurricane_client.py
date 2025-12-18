"""Hurricane MCP Client - Async HTTP client for Hurricane Tracker MCP Server.

This module provides an async HTTP client that communicates with the Hurricane MCP
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


class HurricaneMCPClient:
    """Async HTTP client for Hurricane Tracker MCP Server.

    Implements JSON-RPC 2.0 protocol over HTTP transport to communicate
    with the Hurricane MCP Server. Handles SSE (Server-Sent Events) responses.

    Attributes:
        base_url: Base URL of the MCP server (default: http://localhost:8081)
        client: Persistent httpx.AsyncClient that maintains session cookies

    Example:
        >>> client = HurricaneMCPClient(base_url="http://localhost:8081")
        >>> await client.initialize()
        >>> storms = await client.get_active_storms()
        >>> print(storms)
    """

    def __init__(self, base_url: str | None = None, enable_failover: bool | None = None):
        """Initialize Hurricane MCP client.

        Args:
            base_url: Base URL of the MCP server. If None, uses MCP_HURRICANE_SERVER_URL
                     from environment or defaults to http://localhost:8081
            enable_failover: Enable failover to direct NHC API. If None, uses MCP_ENABLE_FAILOVER
        """
        self.base_url = base_url or settings.MCP_HURRICANE_SERVER_URL
        self.client: httpx.AsyncClient | None = None
        self._initialized: bool = False
        self._session_id: str | None = None

        # MCP Integration (Level 5c)
        self.mcp_logger = MCPLogger(server_name="hurricane")
        self.failover_handler = MCPFailoverHandler(
            server_name="hurricane",
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
            timeout_seconds = float(settings.MCP_HURRICANE_REQUEST_TIMEOUT)
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
            - Max retries: 3 (from settings.MCP_HURRICANE_MAX_RETRIES)
            - Exponential backoff: 1s, 2s, 4s (from settings.MCP_HURRICANE_RETRY_DELAY)
            - Retry on: TimeoutError, ConnectError, 5xx errors
            - Don't retry on: 4xx errors (client errors)
        """
        if correlation_id is None:
            correlation_id = self.mcp_logger.generate_correlation_id()

        max_retries = settings.MCP_HURRICANE_MAX_RETRIES
        base_delay_ms = settings.MCP_HURRICANE_RETRY_DELAY

        # Log call start
        self.mcp_logger.log_mcp_call_start(
            tool=tool_name,
            correlation_id=correlation_id,
        )

        # Check circuit breaker before attempting call
        if self.failover_handler.should_failover(correlation_id):
            # Circuit breaker OPEN - use failover
            logger.warning(
                f"Circuit breaker OPEN for hurricane MCP, failover triggered | "
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
                    timeout_seconds=settings.MCP_HURRICANE_REQUEST_TIMEOUT,
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
            >>> client = HurricaneMCPClient()
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

    async def get_active_storms(self, basin: str | None = None) -> dict[str, any]:
        """Get currently active tropical storms and hurricanes.

        ✅ With retry logic, circuit breaker, and structured logging.

        Args:
            basin: Optional 2-letter basin code (e.g., 'AL' for Atlantic, 'EP' for East Pacific)
                  If None, returns storms from all basins

        Returns:
            dict containing active storm data:
                - storms: List of active storms with details
                - Each storm includes: name, basin, category, wind_speed, etc.

        Raises:
            httpx.HTTPError: If the HTTP request fails after retries
            ValueError: If basin code is invalid

        Example:
            >>> client = HurricaneMCPClient()
            >>> await client.initialize()
            >>> storms = await client.get_active_storms(basin="AL")
            >>> for storm in storms.get('storms', []):
            ...     print(f"{storm['name']}: Category {storm['category']}")
        """
        if basin and len(basin) != 2:
            raise ValueError("Basin code must be 2 letters (e.g., 'AL', 'EP')")

        client = self._get_client()

        # Prepare headers with session ID
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26"
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        # Prepare arguments
        arguments = {}
        if basin:
            arguments["basin"] = basin.upper()

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
                        "name": "get_active_storms",
                        "arguments": arguments
                    }
                }
            )

        # Call with retry logic
        return await self._call_with_retry(
            tool_name="get_active_storms",
            request_func=make_request,
        )

    async def get_storm_cone(self, storm_id: str) -> dict[str, any]:
        """Get forecast cone (cone of uncertainty) for a specific storm.

        ✅ With retry logic, circuit breaker, and structured logging.

        Args:
            storm_id: Storm identifier (e.g., 'AL092023' for Hurricane Ian)

        Returns:
            dict containing forecast cone data:
                - cone_coordinates: GeoJSON coordinates for forecast cone
                - forecast_track: Predicted storm path
                - uncertainty: Cone of uncertainty parameters

        Raises:
            httpx.HTTPError: If the HTTP request fails after retries
            ValueError: If storm_id is empty or invalid

        Example:
            >>> client = HurricaneMCPClient()
            >>> await client.initialize()
            >>> cone = await client.get_storm_cone("AL092023")
            >>> print(cone['forecast_track'])
        """
        if not storm_id:
            raise ValueError("Storm ID cannot be empty")

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
                        "name": "get_storm_cone",
                        "arguments": {"stormId": storm_id}
                    }
                }
            )

        # Call with retry logic
        return await self._call_with_retry(
            tool_name="get_storm_cone",
            request_func=make_request,
        )

    async def get_storm_track(self, storm_id: str) -> dict[str, any]:
        """Get historical and forecast track for a specific storm.

        ✅ With retry logic, circuit breaker, and structured logging.

        Args:
            storm_id: Storm identifier (e.g., 'AL092023' for Hurricane Ian)

        Returns:
            dict containing storm track data:
                - historical_positions: Past storm positions
                - forecast_positions: Predicted future positions
                - Each position includes: time, lat, lon, wind_speed, pressure

        Raises:
            httpx.HTTPError: If the HTTP request fails after retries
            ValueError: If storm_id is empty or invalid

        Example:
            >>> client = HurricaneMCPClient()
            >>> await client.initialize()
            >>> track = await client.get_storm_track("AL092023")
            >>> for pos in track['historical_positions']:
            ...     print(f"{pos['time']}: {pos['wind_speed']} mph")
        """
        if not storm_id:
            raise ValueError("Storm ID cannot be empty")

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
                        "name": "get_storm_track",
                        "arguments": {"stormId": storm_id}
                    }
                }
            )

        # Call with retry logic
        return await self._call_with_retry(
            tool_name="get_storm_track",
            request_func=make_request,
        )

    async def get_local_hurricane_alerts(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, any]:
        """Get hurricane alerts and warnings for a specific location.

        ✅ With retry logic, circuit breaker, and structured logging.

        CRITICAL: Hurricane MCP server uses 'lat' and 'lon' parameter names (not 'latitude' and 'longitude').

        Args:
            latitude: Latitude in decimal degrees (-90 to 90)
            longitude: Longitude in decimal degrees (-180 to 180)

        Returns:
            dict containing local hurricane alerts:
                - alerts: List of active alerts for the area
                - Each alert includes: type, severity, message, expires_at

        Raises:
            httpx.HTTPError: If the HTTP request fails after retries
            ValueError: If coordinates are out of range

        Example:
            >>> client = HurricaneMCPClient()
            >>> await client.initialize()
            >>> alerts = await client.get_local_hurricane_alerts(
            ...     latitude=25.7617,
            ...     longitude=-80.1918,
            ... )
            >>> for alert in alerts.get('alerts', []):
            ...     print(f"{alert['type']}: {alert['message']}")
        """
        if not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180")

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
                        "name": "get_local_hurricane_alerts",
                        "arguments": {
                            "lat": latitude,  # Server expects 'lat' not 'latitude'
                            "lon": longitude   # Server expects 'lon' not 'longitude'
                        }
                    }
                }
            )

        # Call with retry logic
        return await self._call_with_retry(
            tool_name="get_local_hurricane_alerts",
            request_func=make_request,
        )

    async def search_historical_tracks(
        self,
        aoi: dict[str, any],
        start: str,
        end: str,
        basin: str | None = None
    ) -> dict[str, any]:
        """Search historical hurricane tracks with filters.

        ✅ With retry logic, circuit breaker, and structured logging.

        CRITICAL: Hurricane MCP server uses 'aoi', 'start', 'end' parameters (not 'year', 'minCategory').

        Args:
            aoi: Area of Interest as GeoJSON Polygon object
                 Format: {"type": "Polygon", "coordinates": [[[lon, lat], [lon, lat], ...]]}
            start: Start date in YYYY-MM-DD format (e.g., "2023-01-01")
            end: End date in YYYY-MM-DD format (e.g., "2023-12-31")
            basin: Optional 2-letter basin code (e.g., 'AL' for Atlantic)

        Returns:
            dict containing historical storm tracks:
                - tracks: List of matching historical storms
                - Each track includes: storm_id, name, year, max_category, track_data

        Raises:
            httpx.HTTPError: If the HTTP request fails after retries
            ValueError: If parameters are invalid

        Example:
            >>> client = HurricaneMCPClient()
            >>> await client.initialize()
            >>> # Search Atlantic basin for 2023
            >>> aoi = {
            ...     "type": "Polygon",
            ...     "coordinates": [[
            ...         [-100.0, 0.0], [-100.0, 50.0],
            ...         [-20.0, 50.0], [-20.0, 0.0], [-100.0, 0.0]
            ...     ]]
            ... }
            >>> tracks = await client.search_historical_tracks(
            ...     aoi=aoi,
            ...     start="2023-01-01",
            ...     end="2023-12-31",
            ...     basin="AL"
            ... )
            >>> for track in tracks.get('tracks', []):
            ...     print(f"{track['name']} ({track['year']}): Cat {track['max_category']}")
        """
        # Validate date format (YYYY-MM-DD)
        import re
        date_pattern = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
        if not date_pattern.match(start):
            raise ValueError("Start date must be in YYYY-MM-DD format")
        if not date_pattern.match(end):
            raise ValueError("End date must be in YYYY-MM-DD format")

        # Validate basin code if provided
        if basin and len(basin) != 2:
            raise ValueError("Basin code must be 2 letters (e.g., 'AL', 'EP')")

        # Validate aoi structure
        if not isinstance(aoi, dict):
            raise ValueError("aoi must be a dictionary")
        if aoi.get("type") != "Polygon":
            raise ValueError("aoi type must be 'Polygon'")
        if "coordinates" not in aoi:
            raise ValueError("aoi must have 'coordinates' field")

        client = self._get_client()

        # Prepare headers with session ID
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26"
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        # Prepare arguments (server expects 'aoi', 'start', 'end')
        arguments = {
            "aoi": aoi,
            "start": start,
            "end": end
        }
        if basin:
            arguments["basin"] = basin.upper()

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
                        "name": "search_historical_tracks",
                        "arguments": arguments
                    }
                }
            )

        # Call with retry logic
        return await self._call_with_retry(
            tool_name="search_historical_tracks",
            request_func=make_request,
        )
