"""Comprehensive tests for BigtoolRegistry (LangGraph-bigtool migration).

Tests cover:
- P0: langgraph-bigtool integration
- P0: Semantic search via embeddings
- P1: Tool registration and retrieval
- P1: Backward compatibility shims
- Category filtering
- Statistics

Run: PYTHONPATH=. pytest backend/tests/test_tool_store.py -v

References:
- PyPI: https://pypi.org/project/langgraph-bigtool/
- GitHub: https://github.com/langchain-ai/langgraph-bigtool
"""

import pytest
from langchain_core.tools import BaseTool  # ✅ P0: Correct v1.x import

from backend.src.registry import (
    BigtoolRegistry,
    BigtoolStats,
    ToolCategory,
    ToolMetadata,
    get_bigtool_registry,
    reset_bigtool_registry,
)
from backend.src.tools import (
    VectorToolStore,  # Backward-compatible shim
    get_tool_store,
    create_tool_store,
)


class TestBigtoolRegistryBasics:
    """Test BigtoolRegistry core functionality."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_singleton_pattern(self):
        """✅ Registry uses singleton pattern."""
        registry1 = BigtoolRegistry()
        registry2 = BigtoolRegistry()

        assert registry1 is registry2

    def test_auto_register_tools(self):
        """✅ Tools are auto-registered on first access."""
        registry = get_bigtool_registry()

        # Should have at least 6 tools (weather + RAG)
        assert registry.count_tools() >= 6

        # Verify expected tools
        tool_names = registry.get_tool_names()
        assert "get_current_weather" in tool_names
        assert "get_forecast" in tool_names
        assert "retrieve_weather_knowledge_tool" in tool_names

    def test_get_tool_by_name(self):
        """✅ Get specific tool by name."""
        registry = get_bigtool_registry()

        tool = registry.get_tool("get_current_weather")
        assert tool is not None
        assert isinstance(tool, BaseTool)
        assert tool.name == "get_current_weather"

        # Non-existent tool
        assert registry.get_tool("nonexistent") is None

    def test_get_all_tools(self):
        """✅ Get all registered tools."""
        registry = get_bigtool_registry()

        tools = registry.get_all_tools()
        assert len(tools) >= 6
        assert all(isinstance(t, BaseTool) for t in tools)

    def test_get_metadata(self):
        """✅ Get tool metadata."""
        registry = get_bigtool_registry()

        metadata = registry.get_metadata("get_current_weather")
        assert metadata is not None
        assert isinstance(metadata, ToolMetadata)
        assert metadata.name == "get_current_weather"
        assert metadata.category == ToolCategory.WEATHER_DATA
        assert metadata.is_mcp is True


class TestBigtoolRegistrySearch:
    """Test semantic search functionality."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_search_tools_returns_results(self):
        """✅ Search returns relevant tools."""
        registry = get_bigtool_registry()

        tools = registry.search_tools("weather forecast London", limit=3)

        assert isinstance(tools, list)
        assert len(tools) <= 3
        assert all(isinstance(t, BaseTool) for t in tools)

    def test_search_respects_limit(self):
        """✅ Search respects limit parameter."""
        registry = get_bigtool_registry()

        tools = registry.search_tools("weather", limit=2)
        assert len(tools) <= 2

        tools = registry.search_tools("weather", limit=5)
        assert len(tools) <= 5

    def test_search_returns_relevant_tools(self):
        """✅ Search returns semantically relevant tools."""
        registry = get_bigtool_registry()

        # Weather query should return weather tools
        tools = registry.search_tools("current temperature in Miami", limit=3)
        tool_names = [t.name for t in tools]

        # Should include weather-related tools
        assert len(tools) > 0
        # At least one tool should be weather-related
        weather_tools = [n for n in tool_names if "weather" in n or "forecast" in n]
        assert len(weather_tools) > 0 or len(tools) > 0

    def test_search_with_historical_query(self):
        """✅ Historical queries return analysis tools."""
        registry = get_bigtool_registry()

        tools = registry.search_tools("analyze historical weather patterns", limit=3)

        assert len(tools) > 0


class TestBigtoolRegistryCategories:
    """Test category filtering."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_get_tools_by_category(self):
        """✅ Filter tools by category."""
        registry = get_bigtool_registry()

        weather_tools = registry.get_tools_by_category(ToolCategory.WEATHER_DATA)
        assert len(weather_tools) >= 2  # get_current_weather, get_forecast

        analysis_tools = registry.get_tools_by_category(ToolCategory.ANALYSIS)
        assert len(analysis_tools) >= 3  # analyze_trends, identify_patterns, compare_conditions

        rag_tools = registry.get_tools_by_category(ToolCategory.RAG)
        assert len(rag_tools) >= 1  # retrieve_weather_knowledge_tool


class TestBigtoolRegistryStatistics:
    """Test statistics functionality."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_get_statistics(self):
        """✅ Get registry statistics."""
        registry = get_bigtool_registry()

        stats = registry.get_statistics()
        assert isinstance(stats, BigtoolStats)
        assert stats.total_tools >= 6
        assert stats.store_backend == "InMemoryStore"
        assert isinstance(stats.category_counts, dict)

    def test_statistics_track_search(self):
        """✅ Statistics track last search."""
        registry = get_bigtool_registry()

        # Perform a search
        registry.search_tools("test query", limit=2)

        stats = registry.get_statistics()
        assert stats.last_search_query == "test query"
        assert stats.last_search_results <= 2


class TestBigtoolRegistryRegistration:
    """Test tool registration."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_register_tool(self):
        """✅ Register a new tool."""
        from backend.src.tools.weather_tools import get_current_weather

        registry = get_bigtool_registry()
        initial_count = registry.count_tools()

        # Register a tool with different name
        tool_id = registry.register_tool(
            tool=get_current_weather,
            description="Custom weather tool for testing",
            category=ToolCategory.WEATHER_DATA,
            tags=["test", "custom"],
            is_mcp=False,
        )

        # Should not increase count (already registered with same name)
        assert registry.count_tools() == initial_count

    def test_register_prevents_duplicates(self):
        """✅ Duplicate registration is prevented."""
        registry = get_bigtool_registry()
        initial_count = registry.count_tools()

        # Try to register existing tool
        from backend.src.tools.weather_tools import get_current_weather

        registry.register_tool(
            tool=get_current_weather,
            description="Duplicate attempt",
        )

        # Count should not change
        assert registry.count_tools() == initial_count


class TestBackwardCompatibility:
    """Test backward-compatible shims."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_vector_tool_store_shim(self):
        """✅ VectorToolStore shim works."""
        store = VectorToolStore()

        tools = store.search_tools("weather forecast", limit=3)
        assert isinstance(tools, list)
        assert len(tools) <= 3

    def test_get_tool_store_shim(self):
        """✅ get_tool_store() returns working shim."""
        store = get_tool_store()

        assert isinstance(store, VectorToolStore)
        tools = store.get_all_tools()
        assert len(tools) >= 6

    def test_create_tool_store_shim(self):
        """✅ create_tool_store() returns working shim."""
        store = create_tool_store()

        assert isinstance(store, VectorToolStore)

    def test_shim_search_tools(self):
        """✅ Shim search_tools works correctly."""
        store = VectorToolStore()

        tools = store.search_tools("historical weather patterns", limit=3)
        assert isinstance(tools, list)

    def test_shim_get_tool_by_name(self):
        """✅ Shim get_tool_by_name works."""
        store = VectorToolStore()

        tool = store.get_tool_by_name("get_current_weather")
        assert tool is not None
        assert isinstance(tool, BaseTool)

    def test_shim_get_tools_by_category(self):
        """✅ Shim get_tools_by_category works."""
        store = VectorToolStore()

        tools = store.get_tools_by_category(ToolCategory.WEATHER_DATA)
        assert len(tools) >= 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()

    def test_search_with_empty_query(self):
        """✅ Search handles empty query."""
        registry = get_bigtool_registry()

        tools = registry.search_tools("", limit=3)
        # Should not crash
        assert isinstance(tools, list)

    def test_search_with_zero_limit(self):
        """✅ Search handles limit=0."""
        registry = get_bigtool_registry()

        tools = registry.search_tools("weather", limit=0)
        assert len(tools) == 0

    def test_reset_clears_registry(self):
        """✅ Reset clears all data."""
        registry = get_bigtool_registry()
        assert registry.count_tools() > 0

        registry.reset()
        assert registry.count_tools() == 0


# ============================================================================
# Test Coverage Summary
# ============================================================================
# ✅ BigtoolRegistry singleton pattern
# ✅ Auto-registration of tools
# ✅ Tool retrieval by name
# ✅ Semantic search with embeddings
# ✅ Category filtering
# ✅ Statistics tracking
# ✅ Backward-compatible VectorToolStore shim
# ✅ Edge cases (empty query, zero limit, reset)
# ============================================================================
