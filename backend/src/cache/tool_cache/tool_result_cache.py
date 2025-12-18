"""Tool Result Cache: T1 (Redis Exact) → T2 (Qdrant Semantic).

Level 9b: Caches MCP tool call results to reduce API calls.

Architecture:
    T1 (Redis): Exact match on tool_name + args hash (fast, <5ms)
    T2 (Qdrant): Semantic match on natural language tool description (<50ms)

Benefits:
    - 50%+ reduction in MCP API calls
    - Per-tool TTL configuration
    - Automatic bypass for life-safety tools
    - Cache promotion (T2 hit → T1 backfill)

CRITICAL SAFETY RULE:
    get_hurricane_alerts is NEVER cached because:
    - Hurricane alerts are time-sensitive (minutes can save lives)
    - Stale alert data could lead to improper evacuation decisions
    - The cost of an API call is negligible vs. the risk of stale data

Usage:
    from backend.src.cache.tool_cache import ToolResultCache

    cache = ToolResultCache(redis_client, semantic_matcher, cache_promoter, tool_config)

    # Lookup
    result = await cache.get_tool_result("get_forecast", {"location": "Miami"})
    if result.hit:
        return result.value  # Cached tool response

    # Execute and cache
    response = await mcp_client.call("get_forecast", {"location": "Miami"})
    await cache.set_tool_result("get_forecast", {"location": "Miami"}, response)
"""

import json
import logging
from typing import Any, ClassVar

import redis.asyncio as redis

from backend.src.cache.common.cache_key_generator import CacheKeyGenerator
from backend.src.cache.common.cache_promoter import CachePromoter
from backend.src.cache.common.semantic_matcher import SemanticMatcher
from backend.src.cache.common.two_tier_cache import TwoTierCache, TwoTierCacheResult
from backend.src.cache.query_normalizer import QueryNormalizer, get_query_normalizer
from backend.src.cache.tool_cache.tool_cache_config import ToolCacheConfig, get_tool_cache_config

logger = logging.getLogger(__name__)


class ToolResultCache(TwoTierCache[dict[str, Any]]):
    """Two-tier cache for MCP tool results.

    Extends TwoTierCache with tool-specific logic:
    - Per-tool TTL and threshold from ToolCacheConfig
    - Bypass for life-safety tools (get_hurricane_alerts)
    - Natural language embedding for semantic matching

    Key Features:
        - T1 (Redis): Exact match on tool_name + normalized args
        - T2 (Qdrant): Semantic match on tool description
        - Automatic bypass for bypass_tools
        - Force refresh support for real-time data

    Attributes:
        tool_config: ToolCacheConfig instance
        normalizer: QueryNormalizer for location normalization
        BYPASS_TOOLS: Tools that should never be cached
    """

    COLLECTION_NAME = "tool_result_cache"

    # Tools that should NEVER be cached (life-safety critical)
    BYPASS_TOOLS: ClassVar[set[str]] = {
        "get_hurricane_alerts",
        "get_emergency_alerts",
        "get_evacuation_orders",
        "get_shelter_locations",
    }

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        semantic_matcher: SemanticMatcher | None = None,
        cache_promoter: CachePromoter | None = None,
        tool_config: ToolCacheConfig | None = None,
        normalizer: QueryNormalizer | None = None,
    ):
        """Initialize tool result cache.

        Args:
            redis_client: Async Redis client for T1 exact match
            semantic_matcher: Qdrant semantic matcher for T2
            cache_promoter: Handles T2 → T1 promotion
            tool_config: Per-tool cache configuration
            normalizer: Query normalizer for location args
        """
        # Get default config (will be overridden per-tool)
        self.tool_config = tool_config or get_tool_cache_config()
        default_config = self.tool_config.get_config("default")

        super().__init__(
            redis_client=redis_client,
            semantic_matcher=semantic_matcher,
            cache_promoter=cache_promoter,
            collection_name=self.COLLECTION_NAME,
            default_ttl=default_config.ttl,
            similarity_threshold=default_config.threshold,
        )

        self.normalizer = normalizer or get_query_normalizer()

        # Tool-specific metrics
        self.tool_hits: dict[str, int] = {}
        self.tool_misses: dict[str, int] = {}
        self.bypasses = 0

        logger.info(
            f"✅ ToolResultCache initialized | collection={self.COLLECTION_NAME} | "
            f"bypass_tools={len(self.BYPASS_TOOLS)}"
        )

    async def get_tool_result(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        force_refresh: bool = False,
    ) -> TwoTierCacheResult[dict[str, Any]]:
        """Get cached tool result or indicate cache miss.

        Args:
            tool_name: Name of the tool (e.g., "get_forecast")
            tool_args: Tool arguments (e.g., {"location": "Miami"})
            force_refresh: Bypass cache for real-time data

        Returns:
            TwoTierCacheResult with hit/miss and cached value

        Example:
            >>> result = await cache.get_tool_result("get_forecast", {"location": "Miami"})
            >>> if result.hit:
            ...     print(f"Cache {result.tier} hit: {result.value}")
        """
        # CRITICAL: Never cache life-safety tools
        if tool_name in self.BYPASS_TOOLS:
            self.bypasses += 1
            logger.debug(f"⚠️ BYPASS (life-safety) | tool={tool_name}")
            return TwoTierCacheResult(
                hit=False,
                tier=None,
                value=None,
                similarity_score=None,
                lookup_ms=0.0,
            )

        # Check if tool should bypass cache
        if self.tool_config.should_bypass(tool_name):
            self.bypasses += 1
            logger.debug(f"⚠️ BYPASS (config) | tool={tool_name}")
            return TwoTierCacheResult(
                hit=False,
                tier=None,
                value=None,
                similarity_score=None,
                lookup_ms=0.0,
            )

        # Allow force refresh for real-time critical data
        if force_refresh:
            logger.debug(f"⚠️ BYPASS (force_refresh) | tool={tool_name}")
            return TwoTierCacheResult(
                hit=False,
                tier=None,
                value=None,
                similarity_score=None,
                lookup_ms=0.0,
            )

        # Get tool-specific config
        config = self.tool_config.get_config(tool_name)
        self.ttl = config.ttl
        self.threshold = config.threshold

        # Generate key input from tool name + normalized args
        key_input = self._build_key_input(tool_name, tool_args)

        # Use parent's get method
        result = await self.get(
            key_input,
            tool_name=tool_name,
            tool_args=tool_args,
        )

        # Track per-tool metrics
        if result.hit:
            self.tool_hits[tool_name] = self.tool_hits.get(tool_name, 0) + 1
        else:
            self.tool_misses[tool_name] = self.tool_misses.get(tool_name, 0) + 1

        return result

    async def set_tool_result(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Cache tool result.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            result: Tool result to cache
        """
        # Don't cache bypass tools
        if tool_name in self.BYPASS_TOOLS or self.tool_config.should_bypass(tool_name):
            logger.debug(f"⚠️ NOT CACHING (bypass) | tool={tool_name}")
            return

        # Get tool-specific config
        config = self.tool_config.get_config(tool_name)
        self.ttl = config.ttl
        self.threshold = config.threshold

        # Generate key input
        key_input = self._build_key_input(tool_name, tool_args)

        # Use parent's set method
        await self.set(
            key_input,
            result,
            tool_name=tool_name,
            tool_args=tool_args,
        )

    def _build_key_input(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str:
        """Build key input from tool name and normalized arguments.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments

        Returns:
            Key input string for cache operations
        """
        # Normalize location arguments if present
        normalized_args = dict(tool_args)
        for key in ["location", "city", "place"]:
            if key in normalized_args and isinstance(normalized_args[key], str):
                normalized_args[key] = self.normalizer.normalize(normalized_args[key])

        # Sort args for deterministic key
        sorted_args = json.dumps(normalized_args, sort_keys=True, default=str)

        return f"{tool_name}:{sorted_args}"

    def _generate_cache_key(self, key_input: str, **kwargs) -> str:
        """Generate exact match cache key from tool+args.

        Args:
            key_input: Combined tool name and args string

        Returns:
            Hash-based cache key with tool: prefix
        """
        return CacheKeyGenerator.generate(key_input, prefix="tool:")

    def _generate_embedding_input(self, key_input: str, **kwargs) -> str:
        """Generate semantic embedding input.

        Creates natural language description for semantic matching:
        - "get_forecast for miami" instead of "get_forecast:{"location":"miami"}"

        Args:
            key_input: Combined tool name and args string

        Returns:
            Natural language description for embedding
        """
        tool_name = kwargs.get("tool_name", "")
        tool_args = kwargs.get("tool_args", {})

        # Build natural language description
        if "location" in tool_args:
            location = self.normalizer.normalize(str(tool_args["location"]))
            return f"{tool_name} for {location}"
        elif "city" in tool_args:
            city = self.normalizer.normalize(str(tool_args["city"]))
            return f"{tool_name} for {city}"
        elif "storm_name" in tool_args:
            return f"{tool_name} for hurricane {tool_args['storm_name']}"

        # Fallback to key_input
        return key_input

    def _serialize(self, value: dict[str, Any]) -> str:
        """Serialize tool result dict to JSON string.

        Args:
            value: Tool result dictionary

        Returns:
            JSON string
        """
        return json.dumps(value, default=str)

    def _deserialize(self, data: str) -> dict[str, Any]:
        """Deserialize JSON string to tool result dict.

        Args:
            data: JSON string

        Returns:
            Tool result dictionary
        """
        return json.loads(data)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dict with overall stats and per-tool breakdown
        """
        base_stats = super().get_stats()

        # Add tool-specific stats
        base_stats["bypasses"] = self.bypasses
        base_stats["per_tool"] = {
            tool_name: {
                "hits": self.tool_hits.get(tool_name, 0),
                "misses": self.tool_misses.get(tool_name, 0),
                "config": self.tool_config.get_config(tool_name).model_dump(),
            }
            for tool_name in set(self.tool_hits.keys()) | set(self.tool_misses.keys())
        }

        return base_stats

    def reset_stats(self) -> None:
        """Reset all cache statistics."""
        super().reset_stats()
        self.tool_hits.clear()
        self.tool_misses.clear()
        self.bypasses = 0
