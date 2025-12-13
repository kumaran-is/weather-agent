"""MCP-specific structured logging with correlation IDs.

This module provides structured logging capabilities for MCP server interactions,
including:
- Correlation ID generation and tracking
- Latency measurement
- Error categorization
- JSON-formatted structured logs
- MCP-specific log methods

Usage:
    >>> from backend.src.mcp.logger import MCPLogger
    >>>
    >>> logger = MCPLogger(server_name="weather")
    >>> correlation_id = logger.log_mcp_call_start(
    ...     tool="get-forecast",
    ...     params={"location": "Miami, FL"}
    ... )
    >>> # ... make MCP call ...
    >>> logger.log_mcp_call_success(
    ...     correlation_id=correlation_id,
    ...     response_size=1024,
    ...     latency_ms=234.5
    ... )
"""

import logging
import time
import uuid
from typing import Any, Literal

logger = logging.getLogger(__name__)


class MCPLogger:
    """Structured logger for MCP server interactions.

    Provides correlation ID tracking, latency measurement, and structured
    logging for all MCP server calls (Weather and Hurricane).

    Attributes:
        server_name: Name of the MCP server ("weather" or "hurricane")

    Example:
        >>> mcp_logger = MCPLogger(server_name="weather")
        >>> correlation_id = mcp_logger.log_mcp_call_start(
        ...     tool="get-forecast",
        ...     params={"location": "Miami, FL"}
        ... )
        >>> # ... HTTP call to MCP server ...
        >>> mcp_logger.log_mcp_call_success(
        ...     correlation_id=correlation_id,
        ...     response_size=2048,
        ...     latency_ms=187.3
        ... )
    """

    def __init__(self, server_name: Literal["weather", "hurricane"]):
        """Initialize MCP logger.

        Args:
            server_name: Name of the MCP server ("weather" or "hurricane")
        """
        self.server_name = server_name
        self._call_timestamps: dict[str, float] = {}

    def generate_correlation_id(self) -> str:
        """Generate a unique correlation ID for request tracing.

        Returns:
            UUID correlation ID string (e.g., "a3b4c5d6-e7f8-9012-3456-789abcdef012")
        """
        return str(uuid.uuid4())

    def log_mcp_call_start(
        self,
        tool: str,
        params: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Log the start of an MCP server call.

        Args:
            tool: MCP tool name (e.g., "get-forecast", "list-active-storms")
            params: Tool parameters (optional, logged at DEBUG level)
            correlation_id: Existing correlation ID (optional, generates new if None)

        Returns:
            Correlation ID for this call

        Example:
            >>> logger = MCPLogger("weather")
            >>> correlation_id = logger.log_mcp_call_start(
            ...     tool="get-forecast",
            ...     params={"location": "Miami, FL"}
            ... )
        """
        if correlation_id is None:
            correlation_id = self.generate_correlation_id()

        # Store timestamp for latency calculation
        self._call_timestamps[correlation_id] = time.time()

        # Structured log with correlation ID
        logger.info(
            f"🔵 MCP call started | "
            f"server={self.server_name} | "
            f"tool={tool} | "
            f"correlation_id={correlation_id}",
            extra={
                "mcp_server": self.server_name,
                "mcp_tool": tool,
                "correlation_id": correlation_id,
                "event": "mcp_call_start",
            }
        )

        # Log parameters at DEBUG level (may contain PII)
        if params:
            logger.debug(
                f"MCP call parameters | correlation_id={correlation_id} | params={params}",
                extra={
                    "correlation_id": correlation_id,
                    "params": params,
                }
            )

        return correlation_id

    def log_mcp_call_success(
        self,
        correlation_id: str,
        tool: str | None = None,
        response_size: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Log successful MCP server call.

        Args:
            correlation_id: Correlation ID from log_mcp_call_start()
            tool: MCP tool name (optional, for context)
            response_size: Response size in bytes (optional)
            latency_ms: Latency in milliseconds (optional, auto-calculated if None)

        Example:
            >>> logger.log_mcp_call_success(
            ...     correlation_id="a3b4c5d6-e7f8-9012-3456-789abcdef012",
            ...     tool="get-forecast",
            ...     response_size=2048,
            ...     latency_ms=187.3
            ... )
        """
        # Calculate latency if not provided
        if latency_ms is None and correlation_id in self._call_timestamps:
            start_time = self._call_timestamps[correlation_id]
            latency_ms = (time.time() - start_time) * 1000  # Convert to ms
            del self._call_timestamps[correlation_id]  # Cleanup

        # Build log message
        msg_parts = [
            "✅ MCP call succeeded",
            f"server={self.server_name}",
            f"correlation_id={correlation_id}",
        ]

        if tool:
            msg_parts.append(f"tool={tool}")
        if latency_ms is not None:
            msg_parts.append(f"latency={latency_ms:.1f}ms")
        if response_size is not None:
            msg_parts.append(f"response_size={response_size}B")

        logger.info(
            " | ".join(msg_parts),
            extra={
                "mcp_server": self.server_name,
                "mcp_tool": tool,
                "correlation_id": correlation_id,
                "latency_ms": latency_ms,
                "response_size": response_size,
                "event": "mcp_call_success",
            }
        )

    def log_mcp_call_failure(
        self,
        correlation_id: str,
        tool: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        latency_ms: float | None = None,
        will_retry: bool = False,
    ) -> None:
        """Log failed MCP server call.

        Args:
            correlation_id: Correlation ID from log_mcp_call_start()
            tool: MCP tool name (optional, for context)
            error_type: Error type (e.g., "TimeoutError", "HTTPError")
            error_message: Error message
            latency_ms: Latency in milliseconds (optional, auto-calculated if None)
            will_retry: Whether this call will be retried

        Example:
            >>> logger.log_mcp_call_failure(
            ...     correlation_id="a3b4c5d6-e7f8-9012-3456-789abcdef012",
            ...     tool="get-forecast",
            ...     error_type="TimeoutError",
            ...     error_message="Request timed out after 30s",
            ...     will_retry=True
            ... )
        """
        # Calculate latency if not provided
        if latency_ms is None and correlation_id in self._call_timestamps:
            start_time = self._call_timestamps[correlation_id]
            latency_ms = (time.time() - start_time) * 1000  # Convert to ms
            del self._call_timestamps[correlation_id]  # Cleanup

        # Build log message
        msg_parts = [
            "❌ MCP call failed",
            f"server={self.server_name}",
            f"correlation_id={correlation_id}",
        ]

        if tool:
            msg_parts.append(f"tool={tool}")
        if error_type:
            msg_parts.append(f"error_type={error_type}")
        if error_message:
            msg_parts.append(f"error={error_message}")
        if latency_ms is not None:
            msg_parts.append(f"latency={latency_ms:.1f}ms")
        if will_retry:
            msg_parts.append("will_retry=True")

        logger.error(
            " | ".join(msg_parts),
            extra={
                "mcp_server": self.server_name,
                "mcp_tool": tool,
                "correlation_id": correlation_id,
                "error_type": error_type,
                "error_message": error_message,
                "latency_ms": latency_ms,
                "will_retry": will_retry,
                "event": "mcp_call_failure",
            }
        )

    def log_mcp_call_timeout(
        self,
        correlation_id: str,
        tool: str | None = None,
        timeout_seconds: float | None = None,
        will_retry: bool = False,
    ) -> None:
        """Log MCP server call timeout.

        Args:
            correlation_id: Correlation ID from log_mcp_call_start()
            tool: MCP tool name (optional, for context)
            timeout_seconds: Timeout value in seconds
            will_retry: Whether this call will be retried

        Example:
            >>> logger.log_mcp_call_timeout(
            ...     correlation_id="a3b4c5d6-e7f8-9012-3456-789abcdef012",
            ...     tool="get-forecast",
            ...     timeout_seconds=30.0,
            ...     will_retry=True
            ... )
        """
        # Calculate actual latency (should be close to timeout_seconds)
        latency_ms = None
        if correlation_id in self._call_timestamps:
            start_time = self._call_timestamps[correlation_id]
            latency_ms = (time.time() - start_time) * 1000  # Convert to ms
            del self._call_timestamps[correlation_id]  # Cleanup

        # Build log message
        msg_parts = [
            "⏱️ MCP call timed out",
            f"server={self.server_name}",
            f"correlation_id={correlation_id}",
        ]

        if tool:
            msg_parts.append(f"tool={tool}")
        if timeout_seconds is not None:
            msg_parts.append(f"timeout={timeout_seconds}s")
        if latency_ms is not None:
            msg_parts.append(f"latency={latency_ms:.1f}ms")
        if will_retry:
            msg_parts.append("will_retry=True")

        logger.warning(
            " | ".join(msg_parts),
            extra={
                "mcp_server": self.server_name,
                "mcp_tool": tool,
                "correlation_id": correlation_id,
                "timeout_seconds": timeout_seconds,
                "latency_ms": latency_ms,
                "will_retry": will_retry,
                "event": "mcp_call_timeout",
            }
        )

    def log_mcp_failover(
        self,
        correlation_id: str,
        from_source: str,
        to_source: str,
        reason: str,
    ) -> None:
        """Log MCP failover to alternative data source.

        Args:
            correlation_id: Correlation ID from log_mcp_call_start()
            from_source: Original data source (e.g., "MCP Server")
            to_source: Failover target (e.g., "Direct NHC API")
            reason: Reason for failover (e.g., "MCP server circuit breaker OPEN")

        Example:
            >>> logger.log_mcp_failover(
            ...     correlation_id="a3b4c5d6-e7f8-9012-3456-789abcdef012",
            ...     from_source="MCP Server",
            ...     to_source="Direct NHC API",
            ...     reason="MCP server circuit breaker OPEN"
            ... )
        """
        logger.warning(
            f"🔄 MCP failover | "
            f"server={self.server_name} | "
            f"correlation_id={correlation_id} | "
            f"from={from_source} | "
            f"to={to_source} | "
            f"reason={reason}",
            extra={
                "mcp_server": self.server_name,
                "correlation_id": correlation_id,
                "from_source": from_source,
                "to_source": to_source,
                "failover_reason": reason,
                "event": "mcp_failover",
            }
        )
