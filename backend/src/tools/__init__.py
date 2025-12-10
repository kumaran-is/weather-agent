"""LangChain tools for Weather AI Agent.

This module provides LangChain tool wrappers that integrate with:
- Weather MCP server (real-time weather data)
- RAG system (historical weather knowledge, 600+ documents)
- Semantic tool discovery (vector-based, 37.5% context reduction)

Level 1 Tools (MCP):
- get_current_weather: Current weather for a location
- get_forecast: Weather forecast (1-7 days)

Level 2 Tools (RAG):
- analyze_trends: Historical weather pattern analysis
- identify_patterns: Anomaly and pattern detection
- compare_conditions: Cross-location weather comparison
- retrieve_weather_knowledge_tool: Semantic knowledge retrieval

Level 3 Tools (Semantic Discovery):
- VectorToolStore: Semantic tool discovery via vector search
- get_tool_store: Get global tool store instance
"""

from backend.src.tools.rag_tools import (
    analyze_trends,
    compare_conditions,
    get_rag_tools,
    identify_patterns,
    retrieve_weather_knowledge_tool,
)
from backend.src.tools.tool_store import VectorToolStore, get_tool_store
from backend.src.tools.weather_tools import (
    get_current_weather,
    get_forecast,
    weather_mcp_client,
)

__all__ = [
    # Level 1: MCP tools
    "get_current_weather",
    "get_forecast",
    "weather_mcp_client",
    # Level 2: RAG tools
    "analyze_trends",
    "identify_patterns",
    "compare_conditions",
    "retrieve_weather_knowledge_tool",
    "get_rag_tools",
    # Level 3: Semantic tool discovery
    "VectorToolStore",
    "get_tool_store",
]
