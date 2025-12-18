"""Cached Tool Decorator for transparent MCP caching.

Level 9b: Wrap any tool function with two-tier caching.

Benefits:
    - Zero code changes to existing tool implementations
    - Automatic cache key generation from function arguments
    - Per-tool TTL and threshold from ToolCacheConfig
    - Force refresh support via function parameter
    - Metrics tracking per tool

Usage:
    from backend.src.cache.tool_cache import cached_tool

    @cached_tool("get_forecast")
    async def get_forecast(location: str, days: int = 7) -> dict:
        '''Get weather forecast with automatic caching.'''
        return await weather_mcp_client.call("get_forecast", {
            "location": location,
            "days": days,
        })

    @cached_tool("get_current_weather")
    async def get_current_weather(location: str) -> dict:
        '''Get current weather with automatic caching.'''
        return await weather_mcp_client.call("get_current_weather", {
            "location": location,
        })

    # NO CACHING for life-safety tools
    async def get_hurricane_alerts(location: str) -> dict:
        '''Get hurricane alerts - NEVER CACHED (life-safety).'''
        return await hurricane_mcp_client.call("get_hurricane_alerts", {
            "location": location,
        })

    # Usage
    forecast = await get_forecast("Miami", days=7)
    forecast_cached = await get_forecast("Miami", days=7)  # Cache hit!

    # Force refresh
    forecast_fresh = await get_forecast("Miami", days=7, force_refresh=True)
"""

import inspect
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from backend.src.cache.tool_cache.tool_result_cache import ToolResultCache

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Global tool cache instance
_tool_cache: ToolResultCache | None = None


def get_tool_cache() -> ToolResultCache | None:
    """Get the global ToolResultCache instance.

    Returns:
        ToolResultCache instance if initialized, None otherwise
    """
    return _tool_cache


def set_tool_cache(cache: ToolResultCache) -> None:
    """Set the global ToolResultCache instance.

    Called during application startup to configure caching.

    Args:
        cache: ToolResultCache instance
    """
    global _tool_cache
    _tool_cache = cache
    logger.info("✅ Global tool cache configured")


def cached_tool(
    tool_name: str,
    cache: ToolResultCache | None = None,
    force_refresh_param: str = "force_refresh",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to add two-tier caching to tool functions.

    Wraps async tool functions with automatic cache lookup and storage.
    Uses ToolResultCache for T1 (Redis exact) + T2 (Qdrant semantic) caching.

    Args:
        tool_name: Name of the tool for cache config lookup
        cache: ToolResultCache instance (uses global if None)
        force_refresh_param: Parameter name for cache bypass (default: "force_refresh")

    Returns:
        Decorated function with caching

    Example:
        @cached_tool("get_forecast")
        async def get_forecast(location: str, days: int = 7) -> dict:
            return await mcp.call("get_forecast", {"location": location, "days": days})

        # Normal call (uses cache)
        result = await get_forecast("Miami", days=7)

        # Force refresh (bypasses cache)
        result = await get_forecast("Miami", days=7, force_refresh=True)

    Notes:
        - Only works with async functions
        - force_refresh parameter is consumed by decorator, not passed to function
        - Life-safety tools (get_hurricane_alerts) are never cached
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Get cache instance
            tool_cache = cache or get_tool_cache()

            # Extract force_refresh if present
            force_refresh = kwargs.pop(force_refresh_param, False)

            # Build tool args from function signature
            tool_args = _build_tool_args(func, args, kwargs)

            # Try cache first (if cache is configured)
            if tool_cache:
                try:
                    result = await tool_cache.get_tool_result(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        force_refresh=force_refresh,
                    )

                    if result.hit:
                        logger.debug(
                            f"✅ Tool cache {result.tier} HIT | tool={tool_name} | "
                            f"similarity={result.similarity_score}"
                        )
                        return result.value  # type: ignore
                except Exception as e:
                    logger.warning(f"⚠️ Tool cache lookup error: {e}")

            # Cache miss - execute function
            logger.debug(f"❌ Tool cache MISS | tool={tool_name}")
            value = await func(*args, **kwargs)

            # Store in cache (if cache is configured)
            if tool_cache and value is not None:
                try:
                    await tool_cache.set_tool_result(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        result=value if isinstance(value, dict) else {"result": value},
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Tool cache store error: {e}")

            return value

        return wrapper  # type: ignore

    return decorator


def _build_tool_args(
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> dict[str, Any]:
    """Build tool arguments dict from function call.

    Uses inspect to map positional args to parameter names.

    Args:
        func: The decorated function
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dict mapping parameter names to values

    Example:
        >>> def get_forecast(location: str, days: int = 7): ...
        >>> _build_tool_args(get_forecast, ("Miami",), {"days": 5})
        {'location': 'Miami', 'days': 5}
    """
    # Get function signature
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Build args dict
    tool_args = {}

    # Map positional args
    for i, arg in enumerate(args):
        if i < len(params):
            tool_args[params[i]] = arg

    # Merge keyword args
    tool_args.update(kwargs)

    return tool_args


# Context manager for tool cache scopes
class ToolCacheContext:
    """Context manager for scoped tool caching.

    Useful for testing or temporary cache configuration.

    Usage:
        async with ToolCacheContext(custom_cache) as cache:
            result = await get_forecast("Miami")
            # Uses custom_cache
        # Global cache restored
    """

    def __init__(self, cache: ToolResultCache | None):
        """Initialize context.

        Args:
            cache: ToolResultCache to use within context
        """
        self.cache = cache
        self.previous_cache: ToolResultCache | None = None

    async def __aenter__(self) -> ToolResultCache | None:
        """Enter context, setting temporary cache."""
        global _tool_cache
        self.previous_cache = _tool_cache
        _tool_cache = self.cache
        return self.cache

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context, restoring previous cache."""
        global _tool_cache
        _tool_cache = self.previous_cache
