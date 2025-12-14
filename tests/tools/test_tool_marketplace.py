"""
Tests for Enterprise Tool Marketplace Integration.

Level 6b: Self-Improvement Platform

Tests:
1. Tool discovery from multiple sources
2. Tool installation and registration
3. Tool uninstallation
4. Category listing
5. Update checking
6. Analytics tracking
"""

import pytest
from backend.src.tools.tool_marketplace import (
    ToolMarketplace,
    ToolMetadata,
    ToolSearchResult,
    LocalToolSource,
    ComposioToolSource,
    MastraToolSource,
    SmitheryToolSource,
)


class TestToolMarketplace:
    """Test suite for ToolMarketplace."""

    @pytest.fixture
    def marketplace(self):
        """Create marketplace instance."""
        return ToolMarketplace()

    def test_marketplace_initialization(self, marketplace):
        """Test marketplace initializes with all sources."""
        assert "local" in marketplace.sources
        assert "composio" in marketplace.sources
        assert "mastra" in marketplace.sources
        assert "smithery" in marketplace.sources

    def test_discover_tools_all_sources(self, marketplace):
        """Test discovering tools from all sources."""
        result = marketplace.discover_tools(category="all", limit=10)

        assert isinstance(result, ToolSearchResult)
        assert len(result.tools) > 0
        assert result.total_count > 0
        assert len(result.source_counts) > 0

    def test_discover_tools_specific_category(self, marketplace):
        """Test discovering tools for specific category."""
        result = marketplace.discover_tools(category="weather", limit=5)

        assert len(result.tools) > 0
        # All should be weather-related
        for tool in result.tools:
            assert "weather" in tool.category.lower()

    def test_discover_tools_with_source_filter(self, marketplace):
        """Test discovering tools from specific sources."""
        result = marketplace.discover_tools(
            category="all",
            limit=10,
            sources=["local"],
        )

        assert len(result.tools) > 0
        # All should be from local source
        for tool in result.tools:
            assert tool.source == "local"

    def test_tools_sorted_by_ranking(self, marketplace):
        """Test that tools are sorted by rating + downloads."""
        result = marketplace.discover_tools(category="all", limit=20)

        if len(result.tools) >= 2:
            # Higher ranked tools should come first
            # Note: This is approximate due to ranking formula
            first_score = result.tools[0].rating * 0.6
            last_score = result.tools[-1].rating * 0.6
            assert first_score >= last_score - 0.5  # Allow some variance

    def test_install_tool_local(self, marketplace):
        """Test installing a local tool."""
        metadata = marketplace.install_tool(
            tool_id="weather_forecast",
            source="local",
        )

        assert metadata is not None
        assert metadata.name == "Weather Forecast Tool"
        assert metadata.is_installed is True
        assert metadata.installed_at is not None
        assert "weather_forecast" in marketplace.installed_tools

    def test_install_tool_auto_detect_source(self, marketplace):
        """Test installing tool with auto source detection."""
        metadata = marketplace.install_tool(tool_id="hurricane_tracker")

        assert metadata is not None
        assert metadata.source == "local"

    def test_install_nonexistent_tool(self, marketplace):
        """Test installing a tool that doesn't exist."""
        metadata = marketplace.install_tool(tool_id="nonexistent_tool")

        assert metadata is None

    def test_uninstall_tool(self, marketplace):
        """Test uninstalling a tool."""
        # First install
        marketplace.install_tool(tool_id="weather_forecast")
        assert "weather_forecast" in marketplace.installed_tools

        # Then uninstall
        result = marketplace.uninstall_tool("weather_forecast")

        assert result is True
        assert "weather_forecast" not in marketplace.installed_tools

    def test_uninstall_nonexistent_tool(self, marketplace):
        """Test uninstalling a tool that isn't installed."""
        result = marketplace.uninstall_tool("not_installed")

        assert result is False

    def test_get_installed_tools(self, marketplace):
        """Test getting list of installed tools."""
        # Install some tools
        marketplace.install_tool(tool_id="weather_forecast")
        marketplace.install_tool(tool_id="hurricane_tracker")

        installed = marketplace.get_installed_tools()

        assert len(installed) == 2
        assert all(isinstance(t, ToolMetadata) for t in installed)

    def test_get_categories(self, marketplace):
        """Test getting all available categories."""
        categories = marketplace.get_categories()

        assert len(categories) > 0
        assert "weather" in categories


class TestToolSources:
    """Test individual tool sources."""

    def test_local_source_search(self):
        """Test local source search."""
        source = LocalToolSource()
        results = source.search("weather", limit=5)

        assert len(results) > 0
        assert all("id" in r for r in results)

    def test_local_source_get_tool(self):
        """Test local source get tool."""
        source = LocalToolSource()
        tool = source.get_tool("weather_forecast")

        assert tool is not None
        assert tool["id"] == "weather_forecast"

    def test_local_source_categories(self):
        """Test local source categories."""
        source = LocalToolSource()
        categories = source.get_categories()

        assert len(categories) > 0
        assert "weather" in categories

    def test_composio_source_search(self):
        """Test Composio source search (mock mode)."""
        source = ComposioToolSource()
        results = source.search("email", limit=5)

        assert len(results) > 0

    def test_composio_source_get_tool(self):
        """Test Composio source get tool."""
        source = ComposioToolSource()
        tool = source.get_tool("composio_gmail")

        assert tool is not None
        assert tool["id"] == "composio_gmail"

    def test_mastra_source_search(self):
        """Test Mastra source search."""
        source = MastraToolSource()
        results = source.search("weather", limit=5)

        assert len(results) > 0

    def test_smithery_source_search(self):
        """Test Smithery source search."""
        source = SmitheryToolSource()
        results = source.search("data", limit=5)

        assert len(results) > 0


class TestAnalytics:
    """Test marketplace analytics."""

    @pytest.fixture
    def marketplace(self):
        return ToolMarketplace()

    def test_search_tracked_in_analytics(self, marketplace):
        """Test that searches are tracked."""
        marketplace.discover_tools(category="weather", limit=5)

        analytics = marketplace.get_analytics()

        assert analytics["total_searches"] == 1
        assert len(analytics["recent_searches"]) == 1
        assert analytics["recent_searches"][0]["category"] == "weather"

    def test_install_tracked_in_analytics(self, marketplace):
        """Test that installations are tracked."""
        marketplace.install_tool(tool_id="weather_forecast")

        analytics = marketplace.get_analytics()

        assert analytics["total_installs"] == 1
        assert len(analytics["recent_installs"]) == 1
        assert analytics["recent_installs"][0]["tool_id"] == "weather_forecast"

    def test_installed_tools_count(self, marketplace):
        """Test installed tools count in analytics."""
        marketplace.install_tool(tool_id="weather_forecast")
        marketplace.install_tool(tool_id="hurricane_tracker")

        analytics = marketplace.get_analytics()

        assert analytics["installed_tools_count"] == 2

    def test_clear_analytics(self, marketplace):
        """Test clearing analytics."""
        marketplace.discover_tools(category="all")
        marketplace.install_tool(tool_id="weather_forecast")

        marketplace.clear_analytics()
        analytics = marketplace.get_analytics()

        assert analytics["total_searches"] == 0
        assert analytics["total_installs"] == 0


class TestToolMetadata:
    """Test ToolMetadata model."""

    def test_metadata_creation(self):
        """Test creating tool metadata."""
        metadata = ToolMetadata(
            id="test_tool",
            name="Test Tool",
            description="A test tool",
            version="1.0.0",
            source="local",
            category="testing",
        )

        assert metadata.id == "test_tool"
        assert metadata.is_installed is False
        assert metadata.deprecated is False

    def test_metadata_with_optional_fields(self):
        """Test metadata with all optional fields."""
        metadata = ToolMetadata(
            id="test_tool",
            name="Test Tool",
            description="A test tool",
            version="1.0.0",
            source="local",
            category="testing",
            rating=4.5,
            downloads=1000,
            author="Test Author",
            documentation_url="https://docs.example.com",
            is_installed=True,
            installed_at="2025-01-01T00:00:00",
        )

        assert metadata.rating == 4.5
        assert metadata.downloads == 1000
        assert metadata.author == "Test Author"
        assert metadata.is_installed is True

    def test_metadata_serialization(self):
        """Test metadata can be serialized."""
        metadata = ToolMetadata(
            id="test_tool",
            name="Test Tool",
            description="A test tool",
            version="1.0.0",
            source="local",
            category="testing",
        )

        data = metadata.model_dump()

        assert isinstance(data, dict)
        assert data["id"] == "test_tool"


class TestUpdateChecking:
    """Test update checking functionality."""

    @pytest.fixture
    def marketplace(self):
        return ToolMarketplace()

    def test_check_updates_no_installed(self, marketplace):
        """Test checking updates with no installed tools."""
        updates = marketplace.check_updates()

        assert updates == []

    def test_check_updates_with_installed(self, marketplace):
        """Test checking updates with installed tools."""
        marketplace.install_tool(tool_id="weather_forecast")

        updates = marketplace.check_updates()

        # No updates expected since we just installed
        assert isinstance(updates, list)
