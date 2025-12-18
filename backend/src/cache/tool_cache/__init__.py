"""Tool Result Cache module for caching MCP tool call results.

Level 9b: Two-tier caching for tool results (T1 Redis exact → T2 Qdrant semantic).

Benefits:
    - Reduce MCP server calls by 50%+
    - Per-tool TTL configuration (weather: 30min, geocode: 7 days)
    - Automatic bypass for life-safety tools (hurricane alerts)
    - @cached_tool decorator for easy integration

Architecture:
    T1 (Redis): Exact match on tool name + arguments hash
    T2 (Qdrant): Semantic match for similar tool calls

Usage:
    from backend.src.cache.tool_cache import (
        ToolResultCache,
        ToolCacheConfig,
        ToolCacheSettings,
        cached_tool,
        get_tool_cache,
    )

    # Using decorator
    @cached_tool("get_forecast")
    async def get_forecast(location: str, days: int = 7) -> dict:
        return await mcp_client.call("get_forecast", {"location": location, "days": days})

    # Direct cache access
    cache = get_tool_cache()
    result = await cache.get_tool_result("get_forecast", {"location": "Miami"})
    if result.hit:
        return result.value
"""

from backend.src.cache.tool_cache.cached_tool_decorator import cached_tool, get_tool_cache
from backend.src.cache.tool_cache.tool_cache_config import ToolCacheConfig, ToolCacheSettings
from backend.src.cache.tool_cache.tool_result_cache import ToolResultCache

__all__ = [
    "ToolResultCache",
    "ToolCacheConfig",
    "ToolCacheSettings",
    "cached_tool",
    "get_tool_cache",
]
