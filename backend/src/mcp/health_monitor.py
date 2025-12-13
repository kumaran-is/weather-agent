"""MCP server health monitoring with background polling.

This module provides health monitoring capabilities for MCP servers:
- Background health checks every 30 seconds
- <500ms timeout for health checks
- Status tracking (UP/DOWN/DEGRADED)
- Last check timestamp
- Failure count tracking
- Automatic recovery detection

Usage:
    >>> from backend.src.mcp.health_monitor import MCPHealthMonitor
    >>>
    >>> monitor = MCPHealthMonitor(
    ...     server_name="weather",
    ...     server_url="http://localhost:8080",
    ...     health_check_timeout_ms=500
    ... )
    >>> await monitor.start()  # Starts background polling
    >>> status = monitor.get_status()  # Get current health status
    >>> await monitor.stop()  # Stop background polling
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

import httpx

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """MCP server health status.

    Attributes:
        UP: Server is healthy and responding
        DOWN: Server is not responding or returning errors
        DEGRADED: Server is responding but with high latency or intermittent failures
    """

    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"


@dataclass
class HealthCheckResult:
    """Result of a health check.

    Attributes:
        status: Health status (UP/DOWN/DEGRADED)
        latency_ms: Response latency in milliseconds (None if timed out)
        error: Error message if check failed (None if successful)
        timestamp: When this check was performed
    """

    status: HealthStatus
    latency_ms: float | None
    error: str | None
    timestamp: datetime


class MCPHealthMonitor:
    """Background health monitor for MCP servers.

    Polls MCP server health endpoint every 30 seconds with <500ms timeout.
    Tracks status, latency, and failure counts.

    Attributes:
        server_name: Name of the MCP server ("weather" or "hurricane")
        server_url: Base URL of the MCP server
        health_check_timeout_ms: Timeout for health checks in milliseconds

    Example:
        >>> monitor = MCPHealthMonitor(
        ...     server_name="weather",
        ...     server_url="http://localhost:8080",
        ...     health_check_timeout_ms=500
        ... )
        >>> await monitor.start()
        >>> # ... later ...
        >>> status = monitor.get_status()
        >>> print(f"Weather MCP is {status.status}")
        >>> await monitor.stop()
    """

    def __init__(
        self,
        server_name: Literal["weather", "hurricane"],
        server_url: str,
        health_check_timeout_ms: int = 500,
        check_interval_seconds: int = 30,
    ):
        """Initialize MCP health monitor.

        Args:
            server_name: Name of the MCP server ("weather" or "hurricane")
            server_url: Base URL of the MCP server
            health_check_timeout_ms: Timeout for health checks in milliseconds (default: 500ms)
            check_interval_seconds: Interval between health checks in seconds (default: 30s)
        """
        self.server_name = server_name
        self.server_url = server_url.rstrip("/")
        self.health_check_timeout_ms = health_check_timeout_ms
        self.check_interval_seconds = check_interval_seconds

        # Health status tracking
        self._current_status: HealthStatus = HealthStatus.DOWN  # Start pessimistic
        self._last_check: datetime | None = None
        self._last_success: datetime | None = None
        self._consecutive_failures: int = 0
        self._last_error: str | None = None
        self._last_latency_ms: float | None = None

        # Background task
        self._monitor_task: asyncio.Task | None = None
        self._running: bool = False

        # HTTP client (reused for all checks)
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Start background health monitoring.

        Launches background task that polls server health every 30 seconds.

        Example:
            >>> monitor = MCPHealthMonitor("weather", "http://localhost:8080")
            >>> await monitor.start()
        """
        if self._running:
            logger.warning(
                f"Health monitor for {self.server_name} MCP is already running"
            )
            return

        self._running = True

        # Create HTTP client with timeout
        timeout = httpx.Timeout(
            timeout=self.health_check_timeout_ms / 1000,  # Convert ms to seconds
            connect=self.health_check_timeout_ms / 1000,
        )
        self._http_client = httpx.AsyncClient(timeout=timeout)

        # Start background task
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info(
            f"✅ Health monitor started for {self.server_name} MCP | "
            f"url={self.server_url} | "
            f"interval={self.check_interval_seconds}s | "
            f"timeout={self.health_check_timeout_ms}ms"
        )

    async def stop(self) -> None:
        """Stop background health monitoring.

        Cancels background task and closes HTTP client.

        Example:
            >>> await monitor.stop()
        """
        if not self._running:
            return

        self._running = False

        # Cancel background task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info(f"Health monitor stopped for {self.server_name} MCP")

    async def _monitor_loop(self) -> None:
        """Background health check loop.

        Runs continuously, performing health checks every check_interval_seconds.
        """
        while self._running:
            try:
                # Perform health check
                result = await self._perform_health_check()

                # Update status
                self._update_status(result)

                # Wait for next check
                await asyncio.sleep(self.check_interval_seconds)

            except asyncio.CancelledError:
                # Task cancelled, exit loop
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error in health monitor loop for {self.server_name} MCP: {e}"
                )
                # Continue monitoring despite errors
                await asyncio.sleep(self.check_interval_seconds)

    async def _perform_health_check(self) -> HealthCheckResult:
        """Perform a single health check.

        Returns:
            HealthCheckResult with status, latency, and error information
        """
        start_time = time.time()
        timestamp = datetime.now()

        try:
            # Health check endpoint (MCP servers typically use /health or /)
            response = await self._http_client.get(f"{self.server_url}/health")

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Check response status
            if response.status_code == 200:
                # Determine status based on latency
                if latency_ms < 200:
                    status = HealthStatus.UP
                elif latency_ms < 1000:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.DOWN

                return HealthCheckResult(
                    status=status,
                    latency_ms=latency_ms,
                    error=None,
                    timestamp=timestamp,
                )
            else:
                # Non-200 response
                return HealthCheckResult(
                    status=HealthStatus.DOWN,
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}",
                    timestamp=timestamp,
                )

        except httpx.TimeoutException:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"Timeout after {self.health_check_timeout_ms}ms",
                timestamp=timestamp,
            )
        except httpx.ConnectError as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"Connection error: {str(e)}",
                timestamp=timestamp,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"Unexpected error: {str(e)}",
                timestamp=timestamp,
            )

    def _update_status(self, result: HealthCheckResult) -> None:
        """Update health status based on check result.

        Args:
            result: Health check result
        """
        previous_status = self._current_status
        self._current_status = result.status
        self._last_check = result.timestamp
        self._last_latency_ms = result.latency_ms

        if result.status == HealthStatus.UP:
            # Success - reset failure count
            if self._consecutive_failures > 0:
                logger.info(
                    f"✅ {self.server_name} MCP recovered | "
                    f"status={result.status} | "
                    f"latency={result.latency_ms:.1f}ms | "
                    f"previous_failures={self._consecutive_failures}"
                )
            self._consecutive_failures = 0
            self._last_success = result.timestamp
            self._last_error = None
        else:
            # Failure - increment failure count
            self._consecutive_failures += 1
            self._last_error = result.error

            # Log status change or persistent failures
            if previous_status != result.status:
                logger.warning(
                    f"⚠️ {self.server_name} MCP status changed | "
                    f"from={previous_status} | "
                    f"to={result.status} | "
                    f"error={result.error} | "
                    f"consecutive_failures={self._consecutive_failures}"
                )
            elif self._consecutive_failures % 5 == 0:
                # Log every 5th consecutive failure
                logger.error(
                    f"❌ {self.server_name} MCP still down | "
                    f"status={result.status} | "
                    f"consecutive_failures={self._consecutive_failures} | "
                    f"error={result.error}"
                )

    def get_status(self) -> dict:
        """Get current health status.

        Returns:
            Dictionary with status, timestamps, latency, and error information

        Example:
            >>> status = monitor.get_status()
            >>> print(status)
            {
                'server_name': 'weather',
                'status': 'UP',
                'last_check': '2025-12-13T10:30:15',
                'last_success': '2025-12-13T10:30:15',
                'consecutive_failures': 0,
                'last_latency_ms': 123.4,
                'last_error': None
            }
        """
        return {
            "server_name": self.server_name,
            "status": self._current_status.value,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "last_success": (
                self._last_success.isoformat() if self._last_success else None
            ),
            "consecutive_failures": self._consecutive_failures,
            "last_latency_ms": self._last_latency_ms,
            "last_error": self._last_error,
        }

    def is_healthy(self) -> bool:
        """Check if server is currently healthy.

        Returns:
            True if status is UP, False otherwise

        Example:
            >>> if monitor.is_healthy():
            ...     print("Server is up!")
        """
        return self._current_status == HealthStatus.UP

    def is_degraded(self) -> bool:
        """Check if server is degraded.

        Returns:
            True if status is DEGRADED, False otherwise
        """
        return self._current_status == HealthStatus.DEGRADED

    def is_down(self) -> bool:
        """Check if server is down.

        Returns:
            True if status is DOWN, False otherwise
        """
        return self._current_status == HealthStatus.DOWN
