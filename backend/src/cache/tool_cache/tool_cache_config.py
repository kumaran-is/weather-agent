"""Per-Tool Cache Configuration.

Level 9b: Different tools have different caching strategies.

Configuration Philosophy:
    - Weather data: Short TTL (5-30 min) - changes frequently
    - Hurricane alerts: NEVER cached (life-safety critical)
    - Geocoding: Long TTL (7 days) - cities don't move
    - Historical data: Very long TTL (24 hours) - doesn't change

Per-Tool Settings:
    - ttl: Time-to-live in seconds
    - threshold: Semantic similarity threshold (0.0-1.0)
    - enabled: Can be disabled per-tool
    - bypass_semantic: Skip T2 semantic search (for exact-match-only tools)

Safety Rules:
    CRITICAL: get_hurricane_alerts is NEVER cached because:
    1. Hurricane data is time-sensitive (minutes matter)
    2. Incorrect/stale alerts could endanger lives
    3. Users expect real-time information for emergencies
    4. Cost of API call << cost of providing stale emergency info

Usage:
    from backend.src.cache.tool_cache import ToolCacheConfig, ToolCacheSettings

    config = ToolCacheConfig()

    # Get tool-specific settings
    settings = config.get_config("get_forecast")
    print(settings.ttl)  # 1800 (30 minutes)

    # Check if tool should bypass cache
    if config.should_bypass("get_hurricane_alerts"):
        # Call MCP directly, never cache
        result = await mcp.call("get_hurricane_alerts", args)
"""

import logging
from typing import ClassVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCacheSettings(BaseModel):
    """Settings for a specific tool's cache behavior.

    Attributes:
        ttl: Time-to-live in seconds
        threshold: Semantic similarity threshold (0.0-1.0)
        enabled: Whether caching is enabled for this tool
        bypass_semantic: Skip T2 semantic search (exact match only)
    """

    ttl: int = Field(default=300, description="Cache TTL in seconds")
    threshold: float = Field(default=0.85, description="Semantic similarity threshold")
    enabled: bool = Field(default=True, description="Cache enabled flag")
    bypass_semantic: bool = Field(default=False, description="Skip semantic search")


class ToolCacheConfig:
    """Per-tool cache configuration management.

    Provides tool-specific caching settings based on data freshness requirements
    and safety considerations.

    CRITICAL SAFETY RULE:
        get_hurricane_alerts is ALWAYS disabled because:
        - Hurricane alerts are time-sensitive (minutes can save lives)
        - Stale alert data could lead to improper evacuation decisions
        - The cost of an API call is negligible vs. the risk of stale data

    Tool Categories:
        1. Real-time Critical (NEVER cache): hurricane_alerts
        2. Frequently Updated (5 min): current_weather, active_storms
        3. Moderately Updated (30 min): forecast, storm_forecast
        4. Rarely Updated (7 days): geocode, historical data
    """

    # Default configurations per tool
    TOOL_CONFIGS: ClassVar[dict[str, ToolCacheSettings]] = {
        # Weather tools (Weather MCP)
        "get_current_weather": ToolCacheSettings(
            ttl=300,  # 5 minutes
            threshold=0.90,
            enabled=True,
        ),
        "get_forecast": ToolCacheSettings(
            ttl=1800,  # 30 minutes
            threshold=0.85,
            enabled=True,
        ),
        "get_hourly_forecast": ToolCacheSettings(
            ttl=900,  # 15 minutes
            threshold=0.88,
            enabled=True,
        ),
        "geocode_location": ToolCacheSettings(
            ttl=604800,  # 7 days (cities don't move)
            threshold=0.95,
            enabled=True,
            bypass_semantic=True,  # Exact match only for geocoding
        ),
        # Hurricane tools (Hurricane MCP)
        "get_active_storms": ToolCacheSettings(
            ttl=120,  # 2 minutes (storms move fast)
            threshold=0.95,
            enabled=True,
        ),
        "get_storm_forecast": ToolCacheSettings(
            ttl=300,  # 5 minutes
            threshold=0.90,
            enabled=True,
        ),
        "get_storm_track": ToolCacheSettings(
            ttl=300,  # 5 minutes
            threshold=0.90,
            enabled=True,
        ),
        "get_storm_history": ToolCacheSettings(
            ttl=86400,  # 24 hours (historical data doesn't change)
            threshold=0.80,
            enabled=True,
        ),
        # CRITICAL: Hurricane alerts - NEVER CACHED
        "get_hurricane_alerts": ToolCacheSettings(
            ttl=0,  # No TTL (never cached)
            threshold=1.0,  # Exact match only (irrelevant since disabled)
            enabled=False,  # CRITICAL: Never cache life-safety data
        ),
        # Default for unknown tools
        "default": ToolCacheSettings(
            ttl=300,  # 5 minutes
            threshold=0.85,
            enabled=True,
        ),
    }

    # Tools that should NEVER be cached (life-safety critical)
    BYPASS_TOOLS: ClassVar[set[str]] = {
        "get_hurricane_alerts",
        "get_emergency_alerts",
        "get_evacuation_orders",
        "get_shelter_locations",
    }

    def __init__(
        self,
        custom_configs: dict[str, ToolCacheSettings] | None = None,
        custom_bypass_tools: set[str] | None = None,
    ):
        """Initialize tool cache configuration.

        Args:
            custom_configs: Additional tool configurations to merge
            custom_bypass_tools: Additional tools to bypass caching
        """
        self.configs = dict(self.TOOL_CONFIGS)
        if custom_configs:
            self.configs.update(custom_configs)

        self.bypass_tools = set(self.BYPASS_TOOLS)
        if custom_bypass_tools:
            self.bypass_tools.update(custom_bypass_tools)

        logger.info(
            f"✅ ToolCacheConfig initialized | "
            f"tools={len(self.configs)} | bypass={len(self.bypass_tools)}"
        )

    def get_config(self, tool_name: str) -> ToolCacheSettings:
        """Get cache configuration for a specific tool.

        Args:
            tool_name: Name of the tool (e.g., "get_forecast")

        Returns:
            ToolCacheSettings for the tool (or default if not configured)
        """
        return self.configs.get(tool_name, self.configs["default"])

    def should_bypass(self, tool_name: str) -> bool:
        """Check if tool should bypass cache entirely.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool should never be cached (life-safety critical)
        """
        # Check explicit bypass list
        if tool_name in self.bypass_tools:
            return True

        # Check if tool is disabled in config
        config = self.get_config(tool_name)
        return not config.enabled

    def should_use_semantic(self, tool_name: str) -> bool:
        """Check if tool should use semantic (T2) caching.

        Args:
            tool_name: Name of the tool

        Returns:
            True if semantic caching should be used
        """
        config = self.get_config(tool_name)
        return config.enabled and not config.bypass_semantic

    def get_ttl(self, tool_name: str) -> int:
        """Get TTL for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            TTL in seconds
        """
        return self.get_config(tool_name).ttl

    def get_threshold(self, tool_name: str) -> float:
        """Get semantic similarity threshold for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Similarity threshold (0.0-1.0)
        """
        return self.get_config(tool_name).threshold

    def list_cacheable_tools(self) -> list[str]:
        """List all tools with caching enabled.

        Returns:
            List of tool names that can be cached
        """
        return [
            name for name, config in self.configs.items()
            if config.enabled and name != "default"
        ]

    def list_bypass_tools(self) -> list[str]:
        """List all tools that bypass caching.

        Returns:
            List of tool names that should never be cached
        """
        return list(self.bypass_tools)


# Singleton instance
_tool_cache_config: ToolCacheConfig | None = None


def get_tool_cache_config() -> ToolCacheConfig:
    """Get singleton ToolCacheConfig instance.

    Returns:
        Default ToolCacheConfig instance
    """
    global _tool_cache_config
    if _tool_cache_config is None:
        _tool_cache_config = ToolCacheConfig()
    return _tool_cache_config
