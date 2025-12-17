"""LangChain tools for Weather AI Agent.

This module provides LangChain tool wrappers that integrate with:
- Weather MCP server (real-time weather data)
- Hurricane MCP server (tropical storm tracking, conditional on MCP_HURRICANE_SERVER_ENABLED)
- RAG system (historical weather knowledge, 600+ documents)
- LangGraph-bigtool (semantic tool discovery, ~50% context reduction)

MIGRATION: VectorToolStore → langgraph-bigtool (v1.5.0)
- Before: Custom VectorToolStore with InMemoryStore (37.5% token reduction)
- After: langgraph-bigtool with embeddings (~50% token reduction, scales to 1000s)

Level 1 Tools (MCP - Weather):
- get_current_weather: Current weather for a location
- get_forecast: Weather forecast (1-7 days)

Level 1 Tools (MCP - Hurricane, conditional):
- get_active_storms: Currently active tropical storms
- get_storm_forecast: Storm forecast cone and track
- get_hurricane_alerts: Hurricane alerts for a location
- get_storm_history: Historical hurricane search

Level 2 Tools (RAG):
- analyze_trends: Historical weather pattern analysis
- identify_patterns: Anomaly and pattern detection
- compare_conditions: Cross-location weather comparison
- retrieve_weather_knowledge_tool: Semantic knowledge retrieval

Level 7 Tools (Bigtool Discovery):
- get_bigtool_registry: Get the bigtool registry singleton
- BigtoolRegistry: LangGraph-bigtool based tool registry

References:
- PyPI: https://pypi.org/project/langgraph-bigtool/
- GitHub: https://github.com/langchain-ai/langgraph-bigtool
"""

from backend.config.settings import settings
from backend.src.tools.rag_tools import (
    analyze_trends,
    compare_conditions,
    get_rag_tools,
    identify_patterns,
    retrieve_weather_knowledge_tool,
)
from backend.src.tools.weather_tools import (
    get_current_weather,
    get_forecast,
    weather_mcp_client,
)

# Import bigtool registry (replaces VectorToolStore)
from backend.src.registry import (
    BigtoolRegistry,
    get_bigtool_registry,
    reset_bigtool_registry,
)

__all__ = [
    # Level 1: MCP tools (Weather)
    "get_current_weather",
    "get_forecast",
    "weather_mcp_client",
    # Level 2: RAG tools
    "analyze_trends",
    "identify_patterns",
    "compare_conditions",
    "retrieve_weather_knowledge_tool",
    "get_rag_tools",
    # Level 7: Bigtool registry (replaces VectorToolStore)
    "BigtoolRegistry",
    "get_bigtool_registry",
    "reset_bigtool_registry",
]

# Conditionally export hurricane tools if enabled
if settings.MCP_HURRICANE_SERVER_ENABLED:
    from backend.src.tools.hurricane_tools import (
        get_active_storms,
        get_hurricane_alerts,
        get_storm_forecast,
        get_storm_history,
    )

    __all__.extend([
        # Level 1: MCP tools (Hurricane)
        "get_active_storms",
        "get_storm_forecast",
        "get_hurricane_alerts",
        "get_storm_history",
    ])


# =============================================================================
# DEPRECATED: Backward-compatible aliases for VectorToolStore
# =============================================================================
# These are kept for backward compatibility. Use BigtoolRegistry instead.


class VectorToolStore:
    """DEPRECATED: Use BigtoolRegistry from backend.src.registry instead.

    This class is a shim for backward compatibility.
    All functionality has been migrated to langgraph-bigtool.

    Example (old - deprecated):
        >>> store = VectorToolStore()
        >>> tools = store.search_tools("weather forecast", limit=3)

    Example (new - recommended):
        >>> from backend.src.registry import get_bigtool_registry
        >>> registry = get_bigtool_registry()
        >>> tools = registry.search_tools("weather forecast", limit=3)
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize with BigtoolRegistry."""
        self._registry = get_bigtool_registry()

    async def initialize(self) -> None:
        """No-op for backward compatibility."""
        pass

    def search_tools(self, query: str, limit: int = 3):
        """Search tools using bigtool registry."""
        return self._registry.search_tools(query, limit=limit)

    async def search_tools_async(self, query: str, limit: int = 3):
        """Async search (delegates to sync)."""
        return self.search_tools(query, limit=limit)

    def get_all_tools(self):
        """Get all tools."""
        return self._registry.get_all_tools()

    def get_tool_by_name(self, name: str):
        """Get tool by name."""
        return self._registry.get_tool(name)

    def get_tools_by_category(self, category: str):
        """Get tools by category."""
        return self._registry.get_tools_by_category(category)


def get_tool_store() -> VectorToolStore:
    """DEPRECATED: Use get_bigtool_registry() instead."""
    return VectorToolStore()


def create_tool_store(*args, **kwargs) -> VectorToolStore:
    """DEPRECATED: Use get_bigtool_registry() instead."""
    return VectorToolStore()


# Add deprecated exports
__all__.extend([
    "VectorToolStore",  # → BigtoolRegistry
    "get_tool_store",  # → get_bigtool_registry
    "create_tool_store",  # → get_bigtool_registry
])
