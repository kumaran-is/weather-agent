"""LangGraph-bigtool Tool Registry System for Level 7.

MIGRATION: Custom VectorToolStore → langgraph-bigtool
- Before: Custom semantic search with InMemoryStore (37.5% token reduction)
- After: LangGraph-bigtool with embeddings (~50% token reduction, scales to 1000s of tools)

This module provides centralized tool management with:
- BigtoolRegistry: LangGraph-bigtool based semantic tool discovery
- Backward-compatible aliases for old ToolRegistry/SemanticToolDiscovery

Usage:
    >>> # New API (recommended)
    >>> from backend.src.registry import get_bigtool_registry
    >>> registry = get_bigtool_registry()
    >>> tools = registry.search_tools("hurricane forecast Miami", limit=3)
    >>> print([t.name for t in tools])

    >>> # Backward-compatible API (deprecated)
    >>> from backend.src.registry import get_tool_registry
    >>> registry = get_tool_registry()  # Returns BigtoolRegistry
    >>> tools = registry.get_all_tools()

References:
- PyPI: https://pypi.org/project/langgraph-bigtool/
- GitHub: https://github.com/langchain-ai/langgraph-bigtool
"""

from backend.src.registry.bigtool_registry import (
    BigtoolRegistry,
    BigtoolStats,
    ToolCategory,
    ToolMetadata,
    get_bigtool_registry,
    reset_bigtool_registry,
    retrieve_tools_for_query,
)

# =============================================================================
# Backward-Compatible Aliases (DEPRECATED)
# =============================================================================
# These aliases allow existing code to continue working during migration.
# New code should use BigtoolRegistry and get_bigtool_registry() directly.

# Alias ToolRegistry → BigtoolRegistry
ToolRegistry = BigtoolRegistry

# Alias get_tool_registry → get_bigtool_registry
get_tool_registry = get_bigtool_registry

# Alias initialize_registry → get_bigtool_registry
initialize_registry = get_bigtool_registry


# Backward-compatible SemanticToolDiscovery (now integrated into BigtoolRegistry)
class SemanticToolDiscovery:
    """DEPRECATED: Use BigtoolRegistry.search_tools() instead.

    This class is kept for backward compatibility only.
    All functionality is now integrated into BigtoolRegistry.

    Example (old - deprecated):
        >>> discovery = get_semantic_discovery()
        >>> recommendations = await discovery.discover_tools("weather forecast")

    Example (new - recommended):
        >>> registry = get_bigtool_registry()
        >>> tools = registry.search_tools("weather forecast", limit=3)
    """

    def __init__(self) -> None:
        """Initialize with BigtoolRegistry."""
        self._registry = get_bigtool_registry()

    async def discover_tools(
        self,
        query: str,
        max_recommendations: int | None = None,
    ) -> list:
        """Discover tools using semantic search.

        DEPRECATED: Use BigtoolRegistry.search_tools() instead.
        """
        limit = max_recommendations or 3
        tools = self._registry.search_tools(query, limit=limit)

        # Return in old ToolRecommendation format for compatibility
        return [
            {
                "tool_name": tool.name,
                "tool": tool,
                "confidence_score": 0.8,  # Placeholder
                "reasoning": "Semantic match via langgraph-bigtool",
            }
            for tool in tools
        ]

    def discover_tools_sync(
        self,
        query: str,
        max_recommendations: int | None = None,
    ) -> list:
        """Synchronous version of discover_tools.

        DEPRECATED: Use BigtoolRegistry.search_tools() instead.
        """
        limit = max_recommendations or 3
        tools = self._registry.search_tools(query, limit=limit)

        return [
            {
                "tool_name": tool.name,
                "tool": tool,
                "confidence_score": 0.8,
                "reasoning": "Semantic match via langgraph-bigtool",
            }
            for tool in tools
        ]

    def search_tools(self, query: str, limit: int = 3) -> list:
        """Search tools using semantic search.

        DEPRECATED: Use BigtoolRegistry.search_tools() directly instead.

        Args:
            query: Search query
            limit: Maximum number of tools to return

        Returns:
            List of LangChain tools
        """
        return self._registry.search_tools(query, limit=limit)


_semantic_discovery_instance: SemanticToolDiscovery | None = None


def get_semantic_discovery() -> SemanticToolDiscovery:
    """Get SemanticToolDiscovery instance.

    DEPRECATED: Use get_bigtool_registry() instead.
    """
    global _semantic_discovery_instance
    if _semantic_discovery_instance is None:
        _semantic_discovery_instance = SemanticToolDiscovery()
    return _semantic_discovery_instance


def reset_semantic_discovery() -> None:
    """Reset semantic discovery instance.

    DEPRECATED: Use reset_bigtool_registry() instead.
    """
    global _semantic_discovery_instance
    _semantic_discovery_instance = None
    reset_bigtool_registry()


__all__ = [
    # New API (recommended)
    "BigtoolRegistry",
    "BigtoolStats",
    "ToolCategory",
    "ToolMetadata",
    "get_bigtool_registry",
    "reset_bigtool_registry",
    "retrieve_tools_for_query",
    # Backward-compatible aliases (deprecated)
    "ToolRegistry",  # → BigtoolRegistry
    "get_tool_registry",  # → get_bigtool_registry
    "initialize_registry",  # → get_bigtool_registry
    "SemanticToolDiscovery",  # → BigtoolRegistry.search_tools()
    "get_semantic_discovery",  # → get_bigtool_registry()
    "reset_semantic_discovery",  # → reset_bigtool_registry()
]
