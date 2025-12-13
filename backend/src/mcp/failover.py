"""MCP failover handler with circuit breaker integration.

This module provides failover capabilities for MCP server calls:
- Circuit breaker integration (reuses existing CircuitBreakerRegistry)
- 3-tier failover chain: MCP Server → Direct NHC API → Cached Data
- Automatic failover on circuit breaker OPEN state
- Failover event tracking
- Support for Weather and Hurricane MCP servers

Usage:
    >>> from backend.src.mcp.failover import MCPFailoverHandler
    >>> from backend.src.mcp.logger import MCPLogger
    >>>
    >>> logger = MCPLogger(server_name="weather")
    >>> failover = MCPFailoverHandler(
    ...     server_name="weather",
    ...     mcp_logger=logger,
    ...     enable_failover=True
    ... )
    >>>
    >>> # Check if MCP call is allowed (circuit breaker check)
    >>> if failover.can_call_mcp():
    ...     # Attempt MCP call
    ...     try:
    ...         response = await mcp_client.call_tool(...)
    ...         failover.record_success()
    ...     except Exception as e:
    ...         failover.record_failure()
    ...         # Failover to NHC API
    ...         response = await failover.call_nhc_api_direct(...)
"""

import logging
from typing import Any, Literal

from backend.src.agents.circuit_breaker import circuit_registry
from backend.src.mcp.logger import MCPLogger

logger = logging.getLogger(__name__)


class MCPFailoverHandler:
    """Failover handler for MCP server calls with circuit breaker integration.

    Provides 3-tier failover chain:
    1. MCP Server (primary)
    2. Direct NHC API (secondary)
    3. Cached data (tertiary, not implemented yet)

    Integrates with existing CircuitBreakerRegistry to prevent cascade failures.

    Attributes:
        server_name: Name of the MCP server ("weather" or "hurricane")
        mcp_logger: MCPLogger instance for structured logging
        enable_failover: Whether to enable failover to NHC API

    Example:
        >>> logger = MCPLogger("weather")
        >>> failover = MCPFailoverHandler(
        ...     server_name="weather",
        ...     mcp_logger=logger,
        ...     enable_failover=True
        ... )
        >>> if failover.can_call_mcp():
        ...     # MCP call allowed
        ...     pass
        ... else:
        ...     # Circuit breaker OPEN, use failover
        ...     pass
    """

    def __init__(
        self,
        server_name: Literal["weather", "hurricane"],
        mcp_logger: MCPLogger,
        enable_failover: bool = True,
    ):
        """Initialize MCP failover handler.

        Args:
            server_name: Name of the MCP server ("weather" or "hurricane")
            mcp_logger: MCPLogger instance for structured logging
            enable_failover: Whether to enable failover to NHC API (default: True)
        """
        self.server_name = server_name
        self.mcp_logger = mcp_logger
        self.enable_failover = enable_failover

        # Circuit breaker name (global registry)
        self._circuit_breaker_name = f"mcp_{server_name}_server"

        # Get or create circuit breaker from global registry
        self._circuit_breaker = circuit_registry.get_or_create(
            agent_name=self._circuit_breaker_name,
            failure_threshold=3,  # Open after 3 consecutive failures
            success_threshold=2,  # Close after 2 successes in HALF_OPEN
            recovery_timeout=30,  # Wait 30s before attempting recovery
        )

        # Failover statistics
        self._failover_count: int = 0
        self._mcp_success_count: int = 0
        self._nhc_api_count: int = 0
        self._cache_count: int = 0

        logger.info(
            f"✅ MCP failover handler initialized | "
            f"server={server_name} | "
            f"circuit_breaker={self._circuit_breaker_name} | "
            f"failover_enabled={enable_failover}"
        )

    def can_call_mcp(self) -> bool:
        """Check if MCP call is allowed by circuit breaker.

        Returns:
            True if circuit breaker is CLOSED or HALF_OPEN, False if OPEN

        Example:
            >>> if failover.can_call_mcp():
            ...     # Attempt MCP call
            ...     pass
            ... else:
            ...     # Circuit breaker OPEN, use failover
            ...     pass
        """
        return self._circuit_breaker.can_execute()

    def record_success(self) -> None:
        """Record successful MCP call.

        Updates circuit breaker state and increments success counter.

        Example:
            >>> try:
            ...     response = await mcp_client.call_tool(...)
            ...     failover.record_success()
            ... except Exception as e:
            ...     failover.record_failure()
        """
        self._circuit_breaker.record_success()
        self._mcp_success_count += 1

    def record_failure(self) -> None:
        """Record failed MCP call.

        Updates circuit breaker state. May open circuit after threshold failures.

        Example:
            >>> try:
            ...     response = await mcp_client.call_tool(...)
            ...     failover.record_success()
            ... except Exception as e:
            ...     failover.record_failure()
            ...     # Use failover
        """
        self._circuit_breaker.record_failure()

    def should_failover(self, correlation_id: str) -> bool:
        """Determine if failover should be used.

        Args:
            correlation_id: Correlation ID for logging

        Returns:
            True if should use failover (circuit breaker OPEN), False otherwise

        Example:
            >>> if failover.should_failover(correlation_id):
            ...     response = await failover.call_nhc_api_direct(...)
        """
        if not self.enable_failover:
            return False

        # Check circuit breaker state
        if not self._circuit_breaker.can_execute():
            # Circuit breaker OPEN - use failover
            self._failover_count += 1

            self.mcp_logger.log_mcp_failover(
                correlation_id=correlation_id,
                from_source=f"{self.server_name.upper()} MCP Server",
                to_source="Direct NHC API",
                reason=f"Circuit breaker {self._circuit_breaker.state.value}",
            )

            return True

        return False

    async def call_nhc_api_direct(
        self,
        correlation_id: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call NHC API directly (failover tier 2).

        This is a placeholder for direct NHC API calls.
        Actual implementation will depend on NHC API endpoints.

        Args:
            correlation_id: Correlation ID for logging
            endpoint: NHC API endpoint (e.g., "/active-storms")
            params: Query parameters

        Returns:
            API response data

        Raises:
            NotImplementedError: Direct NHC API calls not yet implemented

        Example:
            >>> if failover.should_failover(correlation_id):
            ...     response = await failover.call_nhc_api_direct(
            ...         correlation_id=correlation_id,
            ...         endpoint="/active-storms",
            ...         params={"basin": "AL"}
            ...     )
        """
        self._nhc_api_count += 1

        logger.info(
            f"🔄 Direct NHC API call (failover tier 2) | "
            f"correlation_id={correlation_id} | "
            f"endpoint={endpoint} | "
            f"params={params}"
        )

        # TODO: Implement direct NHC API calls
        # This will require:
        # 1. NHC API client (similar to MCP client but direct HTTP)
        # 2. Response parsing to match MCP response format
        # 3. Error handling and retry logic

        raise NotImplementedError(
            "Direct NHC API failover not yet implemented. "
            "This requires NHC API client implementation. "
            "For now, MCP calls will fail if circuit breaker is OPEN."
        )

    async def get_cached_data(
        self,
        correlation_id: str,
        cache_key: str,
    ) -> dict[str, Any] | None:
        """Get cached data (failover tier 3).

        This is a placeholder for cached data retrieval.
        Actual implementation will use Redis or similar.

        Args:
            correlation_id: Correlation ID for logging
            cache_key: Cache key for data lookup

        Returns:
            Cached data if available, None otherwise

        Example:
            >>> cached = await failover.get_cached_data(
            ...     correlation_id=correlation_id,
            ...     cache_key="hurricane:active_storms"
            ... )
            >>> if cached:
            ...     return cached
        """
        self._cache_count += 1

        logger.info(
            f"🔄 Cached data retrieval (failover tier 3) | "
            f"correlation_id={correlation_id} | "
            f"cache_key={cache_key}"
        )

        # TODO: Implement cached data retrieval
        # This will require:
        # 1. Redis integration (or similar cache)
        # 2. Cache key strategy (e.g., "mcp:weather:forecast:miami")
        # 3. TTL management
        # 4. Stale data handling

        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get failover statistics.

        Returns:
            Dictionary with failover counts and circuit breaker state

        Example:
            >>> stats = failover.get_statistics()
            >>> print(stats)
            {
                'server_name': 'weather',
                'circuit_breaker_state': 'CLOSED',
                'mcp_success_count': 100,
                'failover_count': 5,
                'nhc_api_count': 5,
                'cache_count': 0
            }
        """
        return {
            "server_name": self.server_name,
            "circuit_breaker_name": self._circuit_breaker_name,
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "circuit_breaker_failure_count": self._circuit_breaker.failure_count,
            "circuit_breaker_success_count": self._circuit_breaker.success_count,
            "mcp_success_count": self._mcp_success_count,
            "failover_count": self._failover_count,
            "nhc_api_count": self._nhc_api_count,
            "cache_count": self._cache_count,
        }

    def reset_statistics(self) -> None:
        """Reset failover statistics (for testing).

        Example:
            >>> failover.reset_statistics()
        """
        self._failover_count = 0
        self._mcp_success_count = 0
        self._nhc_api_count = 0
        self._cache_count = 0

        logger.info(f"Failover statistics reset for {self.server_name} MCP")
