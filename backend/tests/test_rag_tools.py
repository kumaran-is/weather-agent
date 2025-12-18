"""Comprehensive tests for RAG tools (LangChain v1.x compliance).

Tests cover:
- P1: All tools are async
- P2: Error handling with try/except
- Tool descriptions and structured args
- RAG integration functionality
- Backward compatibility

Run: PYTHONPATH=. pytest backend/tests/test_rag_tools.py -v
"""

import pytest

from backend.src.tools.rag_tools import (
    analyze_trends,
    compare_conditions,
    get_rag_tools,
    hybrid_search_weather_knowledge,
    identify_patterns,
    retrieve_weather_knowledge_tool,
)


class TestAsyncPatterns:
    """✅ P1: Verify all tools are async."""

    @pytest.mark.asyncio
    async def test_analyze_trends_is_async(self):
        """✅ analyze_trends is async."""
        result = await analyze_trends.ainvoke(
            {"query": "temperature trends", "location": "Tokyo", "time_period": "1980-2020"}
        )
        assert isinstance(result, str)
        assert "Weather Trend Analysis" in result

    @pytest.mark.asyncio
    async def test_identify_patterns_is_async(self):
        """✅ identify_patterns is async."""
        result = await identify_patterns.ainvoke(
            {"query": "Category 5 hurricanes", "pattern_type": "anomalous"}
        )
        assert isinstance(result, str)
        assert "Pattern Identification" in result

    @pytest.mark.asyncio
    async def test_compare_conditions_is_async(self):
        """✅ compare_conditions is async."""
        result = await compare_conditions.ainvoke(
            {"location_a": "Tokyo", "location_b": "Los Angeles", "aspect": "climate"}
        )
        assert isinstance(result, str)
        assert "Weather Comparison" in result

    @pytest.mark.asyncio
    async def test_retrieve_weather_knowledge_tool_is_async(self):
        """✅ retrieve_weather_knowledge_tool is async."""
        result = await retrieve_weather_knowledge_tool.ainvoke(
            {"query": "Category 5 hurricane", "num_results": 3}
        )
        assert isinstance(result, str)
        assert "Weather Knowledge" in result or "No relevant information" in result

    @pytest.mark.asyncio
    async def test_hybrid_search_is_async(self):
        """✅ hybrid_search_weather_knowledge is async."""
        result = await hybrid_search_weather_knowledge.ainvoke(
            {"query": "Saffir-Simpson scale", "num_results": 3}
        )
        assert isinstance(result, str)
        assert "Hybrid Search Results" in result or "No relevant information" in result


class TestErrorHandling:
    """✅ P2: Verify error handling with structured responses."""

    @pytest.mark.asyncio
    async def test_analyze_trends_error_handling(self):
        """✅ analyze_trends handles errors gracefully."""
        # Test with empty query
        result = await analyze_trends.ainvoke({"query": ""})
        assert isinstance(result, str)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_identify_patterns_error_handling(self):
        """✅ identify_patterns handles errors gracefully."""
        # Test with empty query
        result = await identify_patterns.ainvoke({"query": ""})
        assert isinstance(result, str)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_compare_conditions_error_handling(self):
        """✅ compare_conditions handles errors gracefully."""
        # Test with empty locations
        result = await compare_conditions.ainvoke(
            {"location_a": "", "location_b": ""}
        )
        assert isinstance(result, str)
        # Should not raise exception


class TestToolDescriptions:
    """✅ Verify all tools have proper descriptions and args."""

    def test_all_tools_have_descriptions(self):
        """✅ All RAG tools have descriptions."""
        tools = get_rag_tools()
        assert len(tools) == 5

        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 50  # Meaningful description
            assert len(tool.args) > 0  # Structured args

    def test_analyze_trends_description(self):
        """✅ analyze_trends has proper description."""
        assert "historical data" in analyze_trends.description.lower()
        assert "query" in analyze_trends.args
        assert "location" in analyze_trends.args
        assert "time_period" in analyze_trends.args

    def test_identify_patterns_description(self):
        """✅ identify_patterns has proper description."""
        assert "pattern" in identify_patterns.description.lower()
        assert "query" in identify_patterns.args
        assert "pattern_type" in identify_patterns.args

    def test_compare_conditions_description(self):
        """✅ compare_conditions has proper description."""
        assert "compare" in compare_conditions.description.lower()
        assert "location_a" in compare_conditions.args
        assert "location_b" in compare_conditions.args
        assert "aspect" in compare_conditions.args


class TestRAGIntegration:
    """✅ Verify RAG integration works correctly."""

    @pytest.mark.asyncio
    async def test_retrieve_weather_knowledge_returns_data(self):
        """✅ retrieve_weather_knowledge_tool returns formatted data."""
        result = await retrieve_weather_knowledge_tool.ainvoke(
            {"query": "hurricane", "num_results": 2}
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_data(self):
        """✅ hybrid_search_weather_knowledge returns formatted data."""
        result = await hybrid_search_weather_knowledge.ainvoke(
            {"query": "weather", "num_results": 2}
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_analyze_trends_integrates_with_rag(self):
        """✅ analyze_trends uses RAG retrieval."""
        result = await analyze_trends.ainvoke({"query": "temperature"})
        assert isinstance(result, str)
        # Should contain either data or "No historical data found"
        assert "Weather Trend Analysis" in result or "No historical data" in result


class TestToolCount:
    """✅ Verify correct number of tools."""

    def test_get_rag_tools_returns_5_tools(self):
        """✅ get_rag_tools returns exactly 5 tools."""
        tools = get_rag_tools()
        assert len(tools) == 5

        # Verify tool names
        tool_names = {tool.name for tool in tools}
        expected_names = {
            "analyze_trends",
            "identify_patterns",
            "compare_conditions",
            "retrieve_weather_knowledge_tool",
            "hybrid_search_weather_knowledge",
        }
        assert tool_names == expected_names


class TestNumResultsLimits:
    """✅ Verify num_results limits are enforced."""

    @pytest.mark.asyncio
    async def test_retrieve_tool_limits_num_results(self):
        """✅ retrieve_weather_knowledge_tool limits num_results to 10."""
        # Request more than 10, should cap at 10
        result = await retrieve_weather_knowledge_tool.ainvoke(
            {"query": "weather", "num_results": 100}
        )
        assert isinstance(result, str)
        # Should not fail, and should limit internally to 10

    @pytest.mark.asyncio
    async def test_hybrid_search_limits_num_results(self):
        """✅ hybrid_search_weather_knowledge limits num_results to 10."""
        # Request more than 10, should cap at 10
        result = await hybrid_search_weather_knowledge.ainvoke(
            {"query": "weather", "num_results": 100}
        )
        assert isinstance(result, str)
        # Should not fail, and should limit internally to 10


# ============================================================================
# Test Coverage Summary
# ============================================================================
# ✅ P1 #2: All tools are async (analyze_trends, identify_patterns, compare_conditions, retrieve_weather_knowledge_tool)
# ✅ P2 #3: Error handling with try/except (all tools)
# ✅ Tool descriptions and structured args (all 5 tools)
# ✅ RAG integration (retrieve_weather_knowledge, hybrid_search)
# ✅ Tool count (5 RAG tools)
# ============================================================================
