"""MCP response normalization layer.

This module provides response normalization to ensure consistent output format
regardless of data source (MCP server, direct NHC API, cached data, or errors).

Purpose:
- Normalize MCP server responses to standard schema
- Normalize direct NHC API responses to standard schema (when implemented)
- Normalize error responses to standard schema
- Enable consistent evaluation comparisons

Usage:
    >>> from backend.src.mcp.response_normalizer import MCPResponseNormalizer
    >>>
    >>> normalizer = MCPResponseNormalizer()
    >>>
    >>> # Normalize MCP response
    >>> mcp_response = {"content": [...], "isError": False}
    >>> normalized = normalizer.normalize_mcp_response(mcp_response, "get_active_hurricanes")
    >>>
    >>> # Normalize error
    >>> error_response = normalizer.normalize_error("Connection timeout", "get_active_hurricanes")
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NormalizedResponse(BaseModel):
    """Normalized response schema for all MCP/API calls.

    This ensures consistent format regardless of data source.

    Attributes:
        success: Whether the operation succeeded
        source: Data source (mcp_server, direct_api, cache, error)
        tool_name: Name of the tool/endpoint called
        data: Response data (normalized structure)
        error: Error message if success=False
        metadata: Additional metadata (latency, retry count, etc.)
    """

    success: bool = Field(description="Whether the operation succeeded")
    source: Literal["mcp_server", "direct_api", "cache", "error"] = Field(
        description="Data source"
    )
    tool_name: str = Field(description="Name of the tool/endpoint called")
    data: dict[str, Any] | list[Any] | None = Field(
        default=None,
        description="Response data in normalized format"
    )
    error: str | None = Field(
        default=None,
        description="Error message if success=False"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (latency, retry count, etc.)"
    )


class MCPResponseNormalizer:
    """Response normalizer for MCP server and API calls.

    Ensures consistent response format across all data sources:
    - MCP server responses
    - Direct NHC API responses (when implemented)
    - Cached data responses
    - Error responses

    Benefits:
    - Consistent evaluation comparisons
    - Easier agent processing
    - Clear error handling
    - Source tracking for debugging

    Example:
        >>> normalizer = MCPResponseNormalizer()
        >>>
        >>> # Normalize successful MCP response
        >>> mcp_response = {
        ...     "content": [{"type": "text", "text": "Hurricane data..."}],
        ...     "isError": False
        ... }
        >>> result = normalizer.normalize_mcp_response(
        ...     mcp_response,
        ...     tool_name="get_active_hurricanes"
        ... )
        >>> assert result.success is True
        >>> assert result.source == "mcp_server"
        >>>
        >>> # Normalize error
        >>> error_result = normalizer.normalize_error(
        ...     error_message="Connection timeout",
        ...     tool_name="get_active_hurricanes"
        ... )
        >>> assert error_result.success is False
        >>> assert error_result.source == "error"
    """

    def __init__(self):
        """Initialize response normalizer."""
        self._normalization_count = 0
        self._error_count = 0

        logger.info("✅ MCP response normalizer initialized")

    def normalize_mcp_response(
        self,
        mcp_response: dict[str, Any],
        tool_name: str,
        latency_ms: float | None = None,
        retry_count: int | None = None,
    ) -> NormalizedResponse:
        """Normalize MCP server response to standard schema.

        Args:
            mcp_response: Raw MCP server response
            tool_name: Name of the tool called
            latency_ms: Request latency in milliseconds
            retry_count: Number of retries attempted

        Returns:
            NormalizedResponse with standardized format

        Example:
            >>> mcp_response = {
            ...     "content": [
            ...         {"type": "text", "text": '{"storms": [...]}'}
            ...     ],
            ...     "isError": False
            ... }
            >>> result = normalizer.normalize_mcp_response(
            ...     mcp_response,
            ...     tool_name="get_active_hurricanes",
            ...     latency_ms=250.5,
            ...     retry_count=0
            ... )
        """
        self._normalization_count += 1

        # Extract data from MCP response
        try:
            # MCP responses have format: {"content": [...], "isError": bool}
            content_list = mcp_response.get("content", [])
            is_error = mcp_response.get("isError", False)

            # Extract text from content items
            data = None
            if content_list:
                # Usually content[0] contains the actual data
                first_content = content_list[0]
                if isinstance(first_content, dict):
                    data = first_content.get("text") or first_content
                else:
                    data = first_content

            # Build metadata
            metadata = {}
            if latency_ms is not None:
                metadata["latency_ms"] = latency_ms
            if retry_count is not None:
                metadata["retry_count"] = retry_count
            metadata["original_format"] = "mcp_server"

            return NormalizedResponse(
                success=not is_error,
                source="mcp_server",
                tool_name=tool_name,
                data=data,
                error=None if not is_error else "MCP server returned error",
                metadata=metadata
            )

        except Exception as e:
            logger.error(
                f"❌ Failed to normalize MCP response | "
                f"tool={tool_name} | error={str(e)}"
            )
            return self.normalize_error(
                error_message=f"Response normalization failed: {str(e)}",
                tool_name=tool_name
            )

    def normalize_direct_api_response(
        self,
        api_response: dict[str, Any],
        tool_name: str,
        latency_ms: float | None = None,
    ) -> NormalizedResponse:
        """Normalize direct NHC API response to standard schema.

        Args:
            api_response: Raw NHC API response
            tool_name: Name of the endpoint called
            latency_ms: Request latency in milliseconds

        Returns:
            NormalizedResponse with standardized format

        Example:
            >>> api_response = {
            ...     "activeStorms": [...],
            ...     "lastUpdate": "2024-01-15T12:00:00Z"
            ... }
            >>> result = normalizer.normalize_direct_api_response(
            ...     api_response,
            ...     tool_name="get_active_hurricanes",
            ...     latency_ms=180.2
            ... )
        """
        self._normalization_count += 1

        try:
            # Build metadata
            metadata = {}
            if latency_ms is not None:
                metadata["latency_ms"] = latency_ms
            metadata["original_format"] = "direct_nhc_api"

            # Direct API responses are already dict format
            # Just wrap in normalized schema
            return NormalizedResponse(
                success=True,
                source="direct_api",
                tool_name=tool_name,
                data=api_response,
                error=None,
                metadata=metadata
            )

        except Exception as e:
            logger.error(
                f"❌ Failed to normalize Direct API response | "
                f"tool={tool_name} | error={str(e)}"
            )
            return self.normalize_error(
                error_message=f"Direct API normalization failed: {str(e)}",
                tool_name=tool_name
            )

    def normalize_cached_response(
        self,
        cached_data: dict[str, Any],
        tool_name: str,
        cache_age_seconds: float | None = None,
    ) -> NormalizedResponse:
        """Normalize cached data response to standard schema.

        Args:
            cached_data: Cached data
            tool_name: Name of the tool/endpoint
            cache_age_seconds: Age of cached data in seconds

        Returns:
            NormalizedResponse with standardized format

        Example:
            >>> cached_data = {"storms": [...], "cached_at": "..."}
            >>> result = normalizer.normalize_cached_response(
            ...     cached_data,
            ...     tool_name="get_active_hurricanes",
            ...     cache_age_seconds=120.5
            ... )
        """
        self._normalization_count += 1

        # Build metadata
        metadata = {
            "original_format": "cache",
            "latency_ms": 0.0  # Cache hits are instant
        }
        if cache_age_seconds is not None:
            metadata["cache_age_seconds"] = cache_age_seconds

        return NormalizedResponse(
            success=True,
            source="cache",
            tool_name=tool_name,
            data=cached_data,
            error=None,
            metadata=metadata
        )

    def normalize_error(
        self,
        error_message: str,
        tool_name: str,
        error_type: str | None = None,
    ) -> NormalizedResponse:
        """Normalize error to standard schema.

        Args:
            error_message: Error message
            tool_name: Name of the tool/endpoint that failed
            error_type: Type of error (timeout, connection, etc.)

        Returns:
            NormalizedResponse with standardized error format

        Example:
            >>> result = normalizer.normalize_error(
            ...     error_message="Connection timeout after 60s",
            ...     tool_name="get_active_hurricanes",
            ...     error_type="timeout"
            ... )
            >>> assert result.success is False
            >>> assert result.source == "error"
        """
        self._error_count += 1

        # Build metadata
        metadata = {"original_format": "error"}
        if error_type:
            metadata["error_type"] = error_type

        return NormalizedResponse(
            success=False,
            source="error",
            tool_name=tool_name,
            data=None,
            error=error_message,
            metadata=metadata
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get normalization statistics.

        Returns:
            Dictionary with normalization counts

        Example:
            >>> stats = normalizer.get_statistics()
            >>> print(stats)
            {
                'total_normalizations': 100,
                'error_normalizations': 5,
                'success_rate': 0.95
            }
        """
        return {
            "total_normalizations": self._normalization_count,
            "error_normalizations": self._error_count,
            "success_rate": (
                (self._normalization_count - self._error_count) / self._normalization_count
                if self._normalization_count > 0
                else 0.0
            )
        }

    def reset_statistics(self) -> None:
        """Reset normalization statistics (for testing).

        Example:
            >>> normalizer.reset_statistics()
        """
        self._normalization_count = 0
        self._error_count = 0

        logger.info("Normalization statistics reset")
