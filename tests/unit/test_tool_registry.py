"""Unit tests for Level 7 BigtoolRegistry (LangGraph-bigtool based).

Tests:
- BigtoolRegistry singleton pattern
- Thread-safe operations
- Semantic search via InMemoryStore
- Tool registration and retrieval
- Category filtering
- Statistics aggregation
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool

from backend.src.registry.bigtool_registry import (
    BigtoolRegistry,
    BigtoolStats,
    ToolCategory,
    ToolMetadata,
    get_bigtool_registry,
    reset_bigtool_registry,
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


class TestBigtoolRegistrySingleton:
    """Test BigtoolRegistry singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_bigtool_registry()

    def test_singleton_returns_same_instance(self):
        """Test that BigtoolRegistry returns the same instance."""
        registry1 = BigtoolRegistry()
        registry2 = BigtoolRegistry()

        assert registry1 is registry2

    def test_get_bigtool_registry_returns_singleton(self):
        """Test that get_bigtool_registry returns singleton."""
        registry1 = get_bigtool_registry()
        registry2 = get_bigtool_registry()

        assert registry1 is registry2

    def test_singleton_thread_safe(self):
        """Test that singleton is thread-safe."""
        instances = []

        def get_instance():
            instances.append(BigtoolRegistry())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)


class TestBigtoolRegistryCRUD:
    """Test basic CRUD operations."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True  # Skip auto-registration

    def test_register_tool(self):
        """Test tool registration."""
        test_tool = create_test_tool("test_tool")

        self.registry.register_tool(
            tool=test_tool,
            description="Test tool",
            category=ToolCategory.WEATHER_DATA,
            tags=["test"],
        )

        assert self.registry.count_tools() == 1
        assert "test_tool" in self.registry.get_tool_names()

    def test_register_duplicate_tool_skips(self):
        """Test that registering duplicate tool is skipped."""
        test_tool = create_test_tool("test_tool")

        self.registry.register_tool(
            tool=test_tool,
            description="Test tool",
            category=ToolCategory.WEATHER_DATA,
        )

        # Try to register again
        self.registry.register_tool(
            tool=test_tool,
            description="Test tool v2",
            category=ToolCategory.WEATHER_DATA,
        )

        # Should still be 1 tool
        assert self.registry.count_tools() == 1

    def test_get_tool(self):
        """Test getting tool by name."""
        test_tool = create_test_tool("test_tool")

        self.registry.register_tool(
            tool=test_tool,
            description="Test tool",
            category=ToolCategory.WEATHER_DATA,
        )

        tool = self.registry.get_tool("test_tool")
        assert tool is test_tool

    def test_get_tool_not_found(self):
        """Test getting non-existent tool returns None."""
        tool = self.registry.get_tool("nonexistent")
        assert tool is None

    def test_get_all_tools(self):
        """Test getting all tools."""
        test_tool1 = create_test_tool("tool1")
        test_tool2 = create_test_tool("tool2")

        self.registry.register_tool(
            tool=test_tool1,
            description="Tool 1",
            category=ToolCategory.WEATHER_DATA,
        )
        self.registry.register_tool(
            tool=test_tool2,
            description="Tool 2",
            category=ToolCategory.RAG,
        )

        tools = self.registry.get_all_tools()
        assert len(tools) == 2
        assert test_tool1 in tools
        assert test_tool2 in tools

    def test_get_tool_names(self):
        """Test getting all tool names."""
        test_tool = create_test_tool("tool1")
        test_tool2 = create_test_tool("tool2")

        self.registry.register_tool(
            tool=test_tool,
            description="Tool 1",
            category=ToolCategory.WEATHER_DATA,
        )
        self.registry.register_tool(
            tool=test_tool2,
            description="Tool 2",
            category=ToolCategory.RAG,
        )

        names = self.registry.get_tool_names()
        assert set(names) == {"tool1", "tool2"}


class TestBigtoolRegistrySearch:
    """Test semantic search functionality."""

    def setup_method(self):
        """Reset registry and setup tools."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True

    @patch("backend.src.registry.bigtool_registry.OpenAIEmbeddings")
    def test_search_tools_returns_results(self, mock_embeddings):
        """Test that search_tools returns results."""
        # Mock embeddings to avoid API calls
        mock_embeddings.return_value = MagicMock()

        test_tool = create_test_tool("get_weather", "Get current weather data")
        self.registry.register_tool(
            tool=test_tool,
            description="Get current weather data",
            category=ToolCategory.WEATHER_DATA,
        )

        # Search using real semantic search with mocked embeddings
        results = self.registry.search_tools("weather forecast", limit=3)
        assert isinstance(results, list)

    def test_search_tools_respects_limit(self):
        """Test that search respects limit parameter."""
        # Register multiple tools
        for i in range(5):
            tool = create_test_tool(f"tool_{i}", f"Tool description {i}")
            self.registry.register_tool(
                tool=tool,
                description=f"Tool description {i}",
                category=ToolCategory.WEATHER_DATA,
            )

        # Search with limit using real semantic search
        results = self.registry.search_tools("test", limit=2)
        assert isinstance(results, list)
        assert len(results) <= 2  # Should respect limit


class TestBigtoolRegistryCategories:
    """Test category-based filtering."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True

    def test_get_tools_by_category(self):
        """Test filtering tools by category."""
        test_tool1 = create_test_tool("weather_tool")
        test_tool2 = create_test_tool("analysis_tool")
        test_tool3 = create_test_tool("rag_tool")

        self.registry.register_tool(
            tool=test_tool1,
            description="Weather tool",
            category=ToolCategory.WEATHER_DATA,
        )
        self.registry.register_tool(
            tool=test_tool2,
            description="Analysis tool",
            category=ToolCategory.ANALYSIS,
        )
        self.registry.register_tool(
            tool=test_tool3,
            description="RAG tool",
            category=ToolCategory.RAG,
        )

        weather_tools = self.registry.get_tools_by_category(ToolCategory.WEATHER_DATA)
        assert len(weather_tools) == 1
        assert test_tool1 in weather_tools

    def test_get_tools_by_category_empty(self):
        """Test filtering by category with no matches."""
        test_tool = create_test_tool("weather_tool")

        self.registry.register_tool(
            tool=test_tool,
            description="Weather tool",
            category=ToolCategory.WEATHER_DATA,
        )

        # No utility tools registered
        utility_tools = self.registry.get_tools_by_category(ToolCategory.UTILITY)
        assert len(utility_tools) == 0


class TestBigtoolRegistryStatistics:
    """Test statistics aggregation."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True

    def test_get_statistics(self):
        """Test getting registry statistics."""
        test_tool1 = create_test_tool("tool1")
        test_tool2 = create_test_tool("tool2")

        self.registry.register_tool(
            tool=test_tool1,
            description="Tool 1",
            category=ToolCategory.WEATHER_DATA,
        )
        self.registry.register_tool(
            tool=test_tool2,
            description="Tool 2",
            category=ToolCategory.ANALYSIS,
        )

        stats = self.registry.get_statistics()

        assert isinstance(stats, BigtoolStats)
        assert stats.total_tools == 2
        assert "weather_data" in stats.category_counts
        assert "analysis" in stats.category_counts
        assert stats.store_backend == "InMemoryStore"

    def test_statistics_track_search(self):
        """Test that statistics track search queries."""
        test_tool = create_test_tool("test_tool")

        self.registry.register_tool(
            tool=test_tool,
            description="Test tool",
            category=ToolCategory.WEATHER_DATA,
        )

        # Perform a search using real semantic search
        self.registry.search_tools("test query", limit=3)

        stats = self.registry.get_statistics()
        assert stats.last_search_query == "test query"


class TestBigtoolRegistryThreadSafety:
    """Test thread safety of registry operations."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_bigtool_registry()
        self.registry = BigtoolRegistry()
        self.registry._auto_registered = True

    def test_concurrent_registration(self):
        """Test concurrent tool registration."""

        def register_tool(tool_id):
            test_tool = create_test_tool(f"tool_{tool_id}")
            self.registry.register_tool(
                tool=test_tool,
                description=f"Tool {tool_id}",
                category=ToolCategory.UTILITY,
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(register_tool, range(50)))

        assert self.registry.count_tools() == 50

    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        test_tool = create_test_tool("test_tool")

        self.registry.register_tool(
            tool=test_tool,
            description="Test tool",
            category=ToolCategory.WEATHER_DATA,
        )

        errors = []

        def read_operation(_):
            try:
                self.registry.get_tool("test_tool")
                self.registry.list_tools()
                self.registry.get_statistics()
            except Exception as e:
                errors.append(e)

        def write_operation(i):
            try:
                new_tool = create_test_tool(f"new_tool_{i}")
                self.registry.register_tool(
                    tool=new_tool,
                    description=f"New tool {i}",
                    category=ToolCategory.UTILITY,
                )
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            read_futures = [executor.submit(read_operation, i) for i in range(50)]
            write_futures = [executor.submit(write_operation, i) for i in range(20)]

            for f in read_futures + write_futures:
                f.result()

        assert len(errors) == 0, f"Concurrent operations failed: {errors}"


class TestBigtoolRegistryReset:
    """Test registry reset functionality."""

    def test_reset_clears_tools(self):
        """Test that reset clears all tools."""
        reset_bigtool_registry()
        registry = BigtoolRegistry()
        registry._auto_registered = True

        test_tool = create_test_tool("test_tool")
        registry.register_tool(
            tool=test_tool,
            description="Test tool",
            category=ToolCategory.WEATHER_DATA,
        )

        assert registry.count_tools() == 1

        registry.reset()

        assert registry.count_tools() == 0
        assert registry._auto_registered is False

    def test_reset_bigtool_registry_clears_singleton(self):
        """Test that reset_bigtool_registry clears singleton."""
        registry1 = get_bigtool_registry()
        reset_bigtool_registry()
        registry2 = get_bigtool_registry()

        # Should be different instances after reset
        assert registry1 is not registry2


class TestToolMetadataModel:
    """Test ToolMetadata model."""

    def test_default_values(self):
        """Test default values."""
        metadata = ToolMetadata(
            name="test_tool",
            category=ToolCategory.WEATHER_DATA,
            description="Test tool",
        )

        assert metadata.name == "test_tool"
        assert metadata.category == ToolCategory.WEATHER_DATA
        assert metadata.description == "Test tool"
        assert metadata.tags == []
        assert metadata.is_available is True

    def test_full_initialization(self):
        """Test full initialization."""
        metadata = ToolMetadata(
            name="test_tool",
            category=ToolCategory.WEATHER_DATA,
            description="Test tool",
            tags=["weather", "forecast"],
            is_available=True,
        )

        assert metadata.tags == ["weather", "forecast"]
        assert metadata.is_available is True


class TestBigtoolStatsModel:
    """Test BigtoolStats model."""

    def test_default_values(self):
        """Test default values."""
        stats = BigtoolStats(
            total_tools=0,
            category_counts={},
            store_backend="InMemoryStore",
        )

        assert stats.total_tools == 0
        assert stats.category_counts == {}
        assert stats.store_backend == "InMemoryStore"
        assert stats.last_search_query is None
        assert stats.last_search_results == 0

    def test_full_initialization(self):
        """Test full initialization."""
        stats = BigtoolStats(
            total_tools=10,
            category_counts={"weather_data": 5, "rag": 3, "analysis": 2},
            store_backend="InMemoryStore",
            last_search_query="test query",
            last_search_results=3,
        )

        assert stats.total_tools == 10
        assert stats.category_counts["weather_data"] == 5
        assert stats.last_search_query == "test query"
        assert stats.last_search_results == 3


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""

    def setup_method(self):
        """Reset before each test."""
        reset_bigtool_registry()

    def test_tool_registry_alias(self):
        """Test ToolRegistry alias points to BigtoolRegistry."""
        from backend.src.registry import ToolRegistry

        assert ToolRegistry is BigtoolRegistry

    def test_get_tool_registry_alias(self):
        """Test get_tool_registry alias works."""
        from backend.src.registry import get_tool_registry

        registry = get_tool_registry()
        assert isinstance(registry, BigtoolRegistry)

    def test_list_tools_returns_metadata(self):
        """Test list_tools returns metadata list."""
        reset_bigtool_registry()
        registry = BigtoolRegistry()
        registry._auto_registered = True

        test_tool = create_test_tool("test_tool")
        registry.register_tool(
            tool=test_tool,
            description="Test tool",
            category=ToolCategory.WEATHER_DATA,
            tags=["test"],
        )

        metadata_list = registry.list_tools()
        assert len(metadata_list) == 1

        metadata = metadata_list[0]
        assert metadata.name == "test_tool"
        assert metadata.category == ToolCategory.WEATHER_DATA
        assert metadata.description == "Test tool"
        assert metadata.tags == ["test"]
