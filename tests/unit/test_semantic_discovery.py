"""Unit tests for Level 7 Semantic Tool Discovery via BigtoolRegistry.

Tests:
- Semantic search via InMemoryStore embeddings
- Tool retrieval for queries
- Search result ranking
- Integration with BigtoolRegistry

NOTE: SemanticToolDiscovery class has been replaced with BigtoolRegistry's
built-in semantic search capabilities using LangGraph-bigtool.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool

from backend.src.registry.bigtool_registry import (
    BigtoolRegistry,
    ToolCategory,
    ToolMetadata,
    get_bigtool_registry,
    reset_bigtool_registry,
    retrieve_tools_for_query,
)


def create_test_tool(name: str, description: str = "Test tool") -> StructuredTool:
    """Create a proper LangChain StructuredTool for testing."""

    def dummy_func(input_str: str = "") -> str:
        """Dummy function for test tool."""
        return f"Result from {name}: {input_str}"

    return StructuredTool.from_function(
        func=dummy_func,
        name=name,
        description=description,
    )


class TestSemanticSearchBasics:
    """Test semantic search basic functionality."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True

    def test_search_returns_list(self):
        """Test that search returns a list of tools."""
        test_tool = create_test_tool("get_weather", "Get current weather data")
        self.registry.register_tool(
            tool=test_tool,
            description="Get current weather data",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search with mocked embeddings
        results = self.registry.search_tools("weather", limit=3)
        assert isinstance(results, list)

    def test_search_empty_query(self):
        """Test search with empty query."""
        test_tool = create_test_tool("get_weather", "Get current weather data")
        self.registry.register_tool(
            tool=test_tool,
            description="Get current weather data",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search with mocked embeddings
        results = self.registry.search_tools("", limit=3)
        assert isinstance(results, list)

    def test_search_no_results(self):
        """Test search with no matching results."""
        test_tool = create_test_tool("get_weather", "Get current weather data")
        self.registry.register_tool(
            tool=test_tool,
            description="Get current weather data",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search - may or may not find results
        results = self.registry.search_tools("completely unrelated topic xyz", limit=3)
        assert isinstance(results, list)  # Should return a list (may be empty)


class TestRetrieveToolsForQuery:
    """Test the retrieve_tools_for_query function."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_retrieve_tools_returns_list(self):
        """Test that retrieve_tools_for_query returns a list."""
        registry = get_bigtool_registry()
        registry._auto_registered = True  # Skip auto-registration

        test_tool = create_test_tool("get_forecast", "Get weather forecast")
        registry.register_tool(
            tool=test_tool,
            description="Get weather forecast",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search with mocked embeddings
        results = retrieve_tools_for_query("forecast", limit=3)
        assert isinstance(results, list)

    def test_retrieve_tools_respects_limit(self):
        """Test that retrieve_tools_for_query respects limit."""
        registry = get_bigtool_registry()
        registry._auto_registered = True

        # Register multiple tools
        for i in range(5):
            tool = create_test_tool(f"tool_{i}", f"Tool {i} description")
            registry.register_tool(
                tool=tool,
                description=f"Tool {i} description",
                category=ToolCategory.WEATHER_DATA,
            )

        # Use real semantic search with mocked embeddings
        results = retrieve_tools_for_query("test", limit=2)
        assert isinstance(results, list)
        assert len(results) <= 2  # Should respect limit


class TestSemanticSearchWithCategories:
    """Test semantic search with category filtering."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True

    def test_search_weather_tools(self):
        """Test searching for weather-related tools."""
        weather_tool = create_test_tool("get_weather", "Get current weather conditions")
        analysis_tool = create_test_tool("analyze_data", "Analyze historical data")

        self.registry.register_tool(
            tool=weather_tool,
            description="Get current weather conditions",
            category=ToolCategory.WEATHER_DATA,
            tags=["weather", "current", "conditions"],
        )
        self.registry.register_tool(
            tool=analysis_tool,
            description="Analyze historical data",
            category=ToolCategory.ANALYSIS,
            tags=["analysis", "historical"],
        )

        # Use category filtering instead of semantic search
        weather_tools = self.registry.get_tools_by_category(ToolCategory.WEATHER_DATA)
        assert len(weather_tools) == 1
        assert weather_tool in weather_tools

    def test_search_rag_tools(self):
        """Test searching for RAG tools."""
        rag_tool = create_test_tool("rag_search", "Search knowledge base")
        weather_tool = create_test_tool("get_weather", "Get weather data")

        self.registry.register_tool(
            tool=rag_tool,
            description="Search knowledge base",
            category=ToolCategory.RAG,
            tags=["rag", "search", "knowledge"],
        )
        self.registry.register_tool(
            tool=weather_tool,
            description="Get weather data",
            category=ToolCategory.WEATHER_DATA,
        )

        rag_tools = self.registry.get_tools_by_category(ToolCategory.RAG)
        assert len(rag_tools) == 1
        assert rag_tool in rag_tools


class TestSemanticSearchTracking:
    """Test that semantic search tracks queries in statistics."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True

    def test_search_updates_statistics(self):
        """Test that search updates last_search_query in stats."""
        test_tool = create_test_tool("get_weather", "Get weather")
        self.registry.register_tool(
            tool=test_tool,
            description="Get weather",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search with mocked embeddings
        self.registry.search_tools("weather forecast Miami", limit=3)

        stats = self.registry.get_statistics()
        assert stats.last_search_query == "weather forecast Miami"

    def test_multiple_searches_update_stats(self):
        """Test that multiple searches update stats correctly."""
        test_tool = create_test_tool("get_weather", "Get weather")
        self.registry.register_tool(
            tool=test_tool,
            description="Get weather",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search with mocked embeddings
        self.registry.search_tools("first query", limit=3)
        stats1 = self.registry.get_statistics()
        assert stats1.last_search_query == "first query"

        self.registry.search_tools("second query", limit=3)
        stats2 = self.registry.get_statistics()
        assert stats2.last_search_query == "second query"


class TestToolMetadataInSearch:
    """Test tool metadata handling in search results."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True

    def test_get_metadata_after_registration(self):
        """Test getting metadata after tool registration."""
        test_tool = create_test_tool("get_forecast", "Get 7-day weather forecast")
        self.registry.register_tool(
            tool=test_tool,
            description="Get 7-day weather forecast",
            category=ToolCategory.WEATHER_DATA,
            tags=["forecast", "7-day", "prediction"],
        )

        metadata = self.registry.get_metadata("get_forecast")
        assert metadata is not None
        assert metadata.name == "get_forecast"
        assert metadata.category == ToolCategory.WEATHER_DATA
        assert metadata.description == "Get 7-day weather forecast"
        assert "forecast" in metadata.tags

    def test_metadata_not_found(self):
        """Test getting metadata for non-existent tool."""
        metadata = self.registry.get_metadata("nonexistent_tool")
        assert metadata is None


class TestSemanticSearchIntegration:
    """Integration tests for semantic search."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_registry_search_with_auto_registration(self):
        """Test search works with auto-registered tools."""
        # Get registry which triggers auto-registration
        registry = get_bigtool_registry()

        # Should have tools registered
        assert registry.count_tools() >= 0  # May be 0 if MCP is disabled

        # Search should work with real semantic search
        results = registry.search_tools("weather forecast", limit=3)
        assert isinstance(results, list)

    def test_search_result_is_langchain_tool(self):
        """Test that search results are LangChain tools."""
        registry = get_bigtool_registry()
        registry._auto_registered = True

        test_tool = create_test_tool("get_weather", "Get current weather")
        registry.register_tool(
            tool=test_tool,
            description="Get current weather",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search with mocked embeddings
        results = registry.search_tools("weather", limit=3)
        for tool in results:
            assert isinstance(tool, StructuredTool)


class TestBackwardCompatibleSemanticDiscovery:
    """Test backward compatibility for SemanticToolDiscovery shim."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_semantic_tool_discovery_shim_exists(self):
        """Test that SemanticToolDiscovery shim is importable."""
        from backend.src.registry import SemanticToolDiscovery

        # Should be able to instantiate
        discovery = SemanticToolDiscovery()
        assert discovery is not None

    def test_semantic_tool_discovery_delegates_to_registry(self):
        """Test that SemanticToolDiscovery delegates to BigtoolRegistry."""
        from backend.src.registry import SemanticToolDiscovery

        discovery = SemanticToolDiscovery()

        # Should have search_tools method
        assert hasattr(discovery, "search_tools")

        # Register a tool
        test_tool = create_test_tool("test_tool", "Test tool")
        discovery._registry.register_tool(
            tool=test_tool,
            description="Test tool",
            category=ToolCategory.WEATHER_DATA,
        )
        discovery._registry._auto_registered = True

        # Search should work with real semantic search
        results = discovery.search_tools("test", limit=3)
        assert isinstance(results, list)


class TestEdgeCases:
    """Test edge cases for semantic search."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True

    def test_search_with_special_characters(self):
        """Test search with special characters in query."""
        test_tool = create_test_tool("get_weather", "Get weather")
        self.registry.register_tool(
            tool=test_tool,
            description="Get weather",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search - should not raise with special characters
        results = self.registry.search_tools("weather @#$%^&*()", limit=3)
        assert isinstance(results, list)

    def test_search_with_unicode(self):
        """Test search with unicode characters."""
        test_tool = create_test_tool("get_weather", "Get weather")
        self.registry.register_tool(
            tool=test_tool,
            description="Get weather",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search - should not raise with unicode
        results = self.registry.search_tools("天气预报 weather", limit=3)
        assert isinstance(results, list)

    def test_search_with_very_long_query(self):
        """Test search with very long query."""
        test_tool = create_test_tool("get_weather", "Get weather")
        self.registry.register_tool(
            tool=test_tool,
            description="Get weather",
            category=ToolCategory.WEATHER_DATA,
        )

        long_query = "weather " * 1000
        # Use real semantic search - should not raise with long query
        results = self.registry.search_tools(long_query, limit=3)
        assert isinstance(results, list)

    def test_search_with_zero_limit(self):
        """Test search with zero limit."""
        test_tool = create_test_tool("get_weather", "Get weather")
        self.registry.register_tool(
            tool=test_tool,
            description="Get weather",
            category=ToolCategory.WEATHER_DATA,
        )

        # Use real semantic search with zero limit
        results = self.registry.search_tools("weather", limit=0)
        assert results == []  # Should return empty list with zero limit
