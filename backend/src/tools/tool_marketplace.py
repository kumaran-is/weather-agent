"""
Enterprise Tool Marketplace Integration.

Level 6b: Self-Improvement Platform

Sources:
1. Composio: 150+ tools (calendar, email, GitHub, Slack, etc.)
2. Mastra MCP Registry: MCP server marketplace
3. Smithery: Community tool marketplace
4. Local: Custom tools

Capabilities:
- Discover tools from multiple sources
- Install and register tools
- Version management
- Deprecation handling
- Analytics-driven recommendations

Target: Unlimited tool access, safe upgrades, data-driven lifecycle
"""

import logging
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolMetadata(BaseModel):
    """Metadata for a tool from marketplace."""

    id: str
    name: str
    description: str
    version: str = "1.0.0"
    source: str
    category: str
    rating: float = Field(ge=0.0, le=5.0, default=0.0)
    downloads: int = Field(ge=0, default=0)
    author: str | None = None
    documentation_url: str | None = None
    is_installed: bool = False
    installed_at: str | None = None
    deprecated: bool = False
    deprecation_message: str | None = None


class ToolSearchResult(BaseModel):
    """Result from tool search."""

    tools: list[ToolMetadata]
    total_count: int
    source_counts: dict[str, int] = Field(default_factory=dict)


class ToolSource(Protocol):
    """Protocol for tool sources."""

    def search(self, category: str, limit: int) -> list[dict[str, Any]]: ...
    def get_tool(self, tool_id: str) -> dict[str, Any]: ...
    def get_categories(self) -> list[str]: ...


class LocalToolSource:
    """Local/custom tools source."""

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {
            "weather_forecast": {
                "id": "weather_forecast",
                "name": "Weather Forecast Tool",
                "description": "Get weather forecasts for locations",
                "version": "1.0.0",
                "category": "weather",
                "rating": 4.5,
                "downloads": 1000,
            },
            "hurricane_tracker": {
                "id": "hurricane_tracker",
                "name": "Hurricane Tracker Tool",
                "description": "Track active hurricanes and tropical storms",
                "version": "1.0.0",
                "category": "weather",
                "rating": 4.8,
                "downloads": 800,
            },
            "evacuation_planner": {
                "id": "evacuation_planner",
                "name": "Evacuation Planner Tool",
                "description": "Plan evacuation routes and shelter locations",
                "version": "1.0.0",
                "category": "safety",
                "rating": 4.7,
                "downloads": 600,
            },
        }

    def search(self, category: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search local tools by category."""
        results = []
        for tool in self.tools.values():
            if category.lower() in tool["category"].lower() or category.lower() == "all":
                results.append(tool)
                if len(results) >= limit:
                    break
        return results

    def get_tool(self, tool_id: str) -> dict[str, Any]:
        """Get tool by ID."""
        return self.tools.get(tool_id, {})

    def get_categories(self) -> list[str]:
        """Get available categories."""
        return list(set(t["category"] for t in self.tools.values()))


class ComposioToolSource:
    """
    Composio integration (150+ tools).

    Composio provides pre-built tools for:
    - Calendar (Google, Outlook)
    - Email (Gmail, Outlook)
    - GitHub operations
    - Slack messaging
    - And 150+ more integrations
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._composio_available = False

        try:
            from composio import Composio
            self._Composio = Composio
            self._composio_available = True
            logger.info("Composio library loaded successfully")
        except ImportError:
            logger.warning("Composio not available, using mock catalog")

        # Mock catalog for when Composio is not available
        self._mock_catalog = [
            {
                "id": "composio_gmail",
                "name": "Gmail Integration",
                "description": "Send and read emails via Gmail",
                "version": "2.1.0",
                "category": "email",
                "rating": 4.8,
                "downloads": 50000,
            },
            {
                "id": "composio_github",
                "name": "GitHub Integration",
                "description": "Manage repos, issues, and PRs",
                "version": "3.0.0",
                "category": "developer",
                "rating": 4.9,
                "downloads": 75000,
            },
            {
                "id": "composio_slack",
                "name": "Slack Integration",
                "description": "Send messages and manage channels",
                "version": "2.5.0",
                "category": "communication",
                "rating": 4.7,
                "downloads": 60000,
            },
            {
                "id": "composio_calendar",
                "name": "Google Calendar",
                "description": "Manage calendar events",
                "version": "1.8.0",
                "category": "calendar",
                "rating": 4.6,
                "downloads": 45000,
            },
            {
                "id": "composio_notion",
                "name": "Notion Integration",
                "description": "Manage Notion pages and databases",
                "version": "1.5.0",
                "category": "productivity",
                "rating": 4.5,
                "downloads": 30000,
            },
        ]

    def search(self, category: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search Composio catalog."""
        if self._composio_available and self.api_key:
            try:
                client = self._Composio(api_key=self.api_key)
                # Note: Actual Composio API may differ
                tools = client.get_tools(category=category, limit=limit)
                return [t.to_dict() for t in tools]
            except Exception as e:
                logger.warning(f"Composio search failed: {e}")

        # Use mock catalog
        results = []
        for tool in self._mock_catalog:
            if category.lower() in tool["category"].lower() or category.lower() == "all":
                results.append(tool)
                if len(results) >= limit:
                    break
        return results

    def get_tool(self, tool_id: str) -> dict[str, Any]:
        """Get tool by ID."""
        for tool in self._mock_catalog:
            if tool["id"] == tool_id:
                return tool
        return {}

    def get_categories(self) -> list[str]:
        """Get available categories."""
        return ["email", "developer", "communication", "calendar", "productivity"]


class MastraToolSource:
    """
    Mastra MCP Registry - MCP server marketplace.

    Provides access to MCP servers for various capabilities.
    """

    def __init__(self):
        # Mock MCP tool catalog
        self._catalog = [
            {
                "id": "mcp_weather_server",
                "name": "Weather MCP Server",
                "description": "MCP server for weather data retrieval",
                "version": "1.0.0",
                "category": "weather",
                "rating": 4.6,
                "downloads": 5000,
            },
            {
                "id": "mcp_database",
                "name": "Database MCP Server",
                "description": "MCP server for database queries",
                "version": "1.2.0",
                "category": "data",
                "rating": 4.4,
                "downloads": 8000,
            },
            {
                "id": "mcp_filesystem",
                "name": "Filesystem MCP Server",
                "description": "MCP server for file operations",
                "version": "1.1.0",
                "category": "filesystem",
                "rating": 4.5,
                "downloads": 6000,
            },
        ]

    def search(self, category: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search Mastra catalog."""
        results = []
        for tool in self._catalog:
            if category.lower() in tool["category"].lower() or category.lower() == "all":
                results.append(tool)
                if len(results) >= limit:
                    break
        return results

    def get_tool(self, tool_id: str) -> dict[str, Any]:
        """Get tool by ID."""
        for tool in self._catalog:
            if tool["id"] == tool_id:
                return tool
        return {}

    def get_categories(self) -> list[str]:
        """Get available categories."""
        return ["weather", "data", "filesystem"]


class SmitheryToolSource:
    """
    Smithery - Community tool marketplace.

    Community-contributed tools and extensions.
    """

    def __init__(self):
        # Mock Smithery catalog
        self._catalog = [
            {
                "id": "smithery_text_analysis",
                "name": "Text Analysis Tool",
                "description": "Analyze text sentiment and entities",
                "version": "1.0.0",
                "category": "nlp",
                "rating": 4.2,
                "downloads": 2000,
            },
            {
                "id": "smithery_web_scraper",
                "name": "Web Scraper Tool",
                "description": "Extract data from web pages",
                "version": "1.3.0",
                "category": "data",
                "rating": 4.3,
                "downloads": 3500,
            },
        ]

    def search(self, category: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search Smithery catalog."""
        results = []
        for tool in self._catalog:
            if category.lower() in tool["category"].lower() or category.lower() == "all":
                results.append(tool)
                if len(results) >= limit:
                    break
        return results

    def get_tool(self, tool_id: str) -> dict[str, Any]:
        """Get tool by ID."""
        for tool in self._catalog:
            if tool["id"] == tool_id:
                return tool
        return {}

    def get_categories(self) -> list[str]:
        """Get available categories."""
        return ["nlp", "data"]


class ToolMarketplace:
    """
    Multi-source tool discovery and installation.

    Provides unified access to tools from:
    - Local/custom tools
    - Composio (150+ tools)
    - Mastra MCP Registry
    - Smithery community tools
    """

    def __init__(
        self,
        composio_api_key: str | None = None,
    ):
        """
        Initialize Tool Marketplace.

        Args:
            composio_api_key: Optional API key for Composio
        """
        self.sources: dict[str, Any] = {
            "local": LocalToolSource(),
            "composio": ComposioToolSource(api_key=composio_api_key),
            "mastra": MastraToolSource(),
            "smithery": SmitheryToolSource(),
        }

        # Installed tools registry
        self.installed_tools: dict[str, ToolMetadata] = {}

        # Analytics
        self.search_history: list[dict[str, Any]] = []
        self.install_history: list[dict[str, Any]] = []

        logger.info("Tool Marketplace initialized with sources: " + ", ".join(self.sources.keys()))

    def discover_tools(
        self,
        category: str,
        limit: int = 10,
        sources: list[str] | None = None,
    ) -> ToolSearchResult:
        """
        Discover tools from multiple marketplaces.

        Args:
            category: Tool category (e.g., "calendar", "email", "data", "all")
            limit: Max tools to return
            sources: Optional list of sources to search (defaults to all)

        Returns:
            ToolSearchResult with tools sorted by rating + downloads
        """
        search_sources = sources or list(self.sources.keys())
        discovered = []
        source_counts: dict[str, int] = {}

        for source_name in search_sources:
            source = self.sources.get(source_name)
            if source is None:
                continue

            try:
                tools = source.search(category, limit=5)
                source_counts[source_name] = len(tools)

                for tool in tools:
                    discovered.append(
                        ToolMetadata(
                            id=tool.get("id", ""),
                            name=tool.get("name", ""),
                            description=tool.get("description", ""),
                            version=tool.get("version", "1.0.0"),
                            source=source_name,
                            category=tool.get("category", category),
                            rating=tool.get("rating", 0.0),
                            downloads=tool.get("downloads", 0),
                            author=tool.get("author"),
                            documentation_url=tool.get("documentation_url"),
                            is_installed=tool.get("id", "") in self.installed_tools,
                        )
                    )
            except Exception as e:
                logger.warning(f"Search failed for {source_name}: {e}")
                source_counts[source_name] = 0

        # Rank by rating (60%) + downloads (40%)
        discovered.sort(
            key=lambda x: (x.rating * 0.6 + min(x.downloads / 1000, 100) * 0.004),
            reverse=True,
        )

        # Track search
        self.search_history.append({
            "category": category,
            "sources": search_sources,
            "results_count": len(discovered),
            "timestamp": datetime.now().isoformat(),
        })

        return ToolSearchResult(
            tools=discovered[:limit],
            total_count=len(discovered),
            source_counts=source_counts,
        )

    def install_tool(
        self,
        tool_id: str,
        source: str | None = None,
    ) -> ToolMetadata | None:
        """
        Install tool from marketplace.

        Args:
            tool_id: Tool ID to install
            source: Optional source name (searches all if not specified)

        Returns:
            ToolMetadata if successful, None otherwise
        """
        # Find tool across sources
        tool_data = None
        tool_source = source

        if source:
            src = self.sources.get(source)
            if src:
                tool_data = src.get_tool(tool_id)
        else:
            # Search all sources
            for src_name, src in self.sources.items():
                tool_data = src.get_tool(tool_id)
                if tool_data:
                    tool_source = src_name
                    break

        if not tool_data:
            logger.warning(f"Tool not found: {tool_id}")
            return None

        # Create metadata
        metadata = ToolMetadata(
            id=tool_data.get("id", tool_id),
            name=tool_data.get("name", tool_id),
            description=tool_data.get("description", ""),
            version=tool_data.get("version", "1.0.0"),
            source=tool_source or "unknown",
            category=tool_data.get("category", "general"),
            rating=tool_data.get("rating", 0.0),
            downloads=tool_data.get("downloads", 0),
            is_installed=True,
            installed_at=datetime.now().isoformat(),
        )

        # Register installed tool
        self.installed_tools[tool_id] = metadata

        # Track installation
        self.install_history.append({
            "tool_id": tool_id,
            "source": tool_source,
            "version": metadata.version,
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(f"Installed tool: {metadata.name} v{metadata.version} from {tool_source}")
        return metadata

    def uninstall_tool(self, tool_id: str) -> bool:
        """
        Uninstall a tool.

        Args:
            tool_id: Tool ID to uninstall

        Returns:
            True if successful, False otherwise
        """
        if tool_id in self.installed_tools:
            del self.installed_tools[tool_id]
            logger.info(f"Uninstalled tool: {tool_id}")
            return True
        return False

    def get_installed_tools(self) -> list[ToolMetadata]:
        """Get all installed tools."""
        return list(self.installed_tools.values())

    def get_categories(self) -> list[str]:
        """Get all available categories across all sources."""
        categories = set()
        for source in self.sources.values():
            categories.update(source.get_categories())
        return sorted(list(categories))

    def check_updates(self) -> list[dict[str, Any]]:
        """
        Check for updates to installed tools.

        Returns:
            List of tools with available updates
        """
        updates = []

        for tool_id, installed in self.installed_tools.items():
            source = self.sources.get(installed.source)
            if source is None:
                continue

            try:
                latest = source.get_tool(tool_id)
                if latest and latest.get("version") != installed.version:
                    updates.append({
                        "tool_id": tool_id,
                        "name": installed.name,
                        "current_version": installed.version,
                        "latest_version": latest.get("version"),
                        "source": installed.source,
                    })
            except Exception as e:
                logger.warning(f"Update check failed for {tool_id}: {e}")

        return updates

    def get_analytics(self) -> dict[str, Any]:
        """Get marketplace analytics."""
        return {
            "installed_tools_count": len(self.installed_tools),
            "total_searches": len(self.search_history),
            "total_installs": len(self.install_history),
            "sources_available": list(self.sources.keys()),
            "recent_searches": self.search_history[-10:],
            "recent_installs": self.install_history[-10:],
        }

    def clear_analytics(self) -> None:
        """Clear analytics data."""
        self.search_history = []
        self.install_history = []
        logger.info("Analytics cleared")
