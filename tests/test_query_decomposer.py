"""Unit tests for L5a: Query Decomposer.

Tests validate the query decomposition for:
- Simple query detection (no decomposition)
- Complex query detection and decomposition
- Multi-location query handling
- Multi-topic query handling (weather + hurricane)
- Time-based query handling (today + tomorrow)
- Heuristic decomposition fallback
- Statistics tracking
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.src.rag.query_decomposer import (
    DecomposedQuery,
    QueryDecomposer,
    decompose_query,
    get_query_decomposer,
)


class TestQueryDecomposer:
    """Test L5a: Query decomposer functionality."""

    @pytest.fixture
    def decomposer_no_llm(self):
        """Create decomposer without LLM (heuristics only)."""
        return QueryDecomposer(
            llm=None,
            enable_llm_decomposition=False,
        )

    def test_simple_query_no_decomposition(self, decomposer_no_llm):
        """Test that simple queries are not decomposed."""
        # Simple weather query (no multi-location, no multi-topic keywords)
        score = decomposer_no_llm._calculate_complexity_score(
            "What is the current temperature?"
        )
        assert score < 2, "Simple query should have low complexity score"

    def test_complex_query_detection(self, decomposer_no_llm):
        """Test that complex queries are detected."""
        # Multi-location query
        score = decomposer_no_llm._calculate_complexity_score(
            "What's the weather in Miami and Tampa?"
        )
        assert score >= 2, "Multi-location query should have high complexity score"

        # Multi-topic query
        score = decomposer_no_llm._calculate_complexity_score(
            "What's the weather and hurricane status for Florida?"
        )
        assert score >= 2, "Multi-topic query should have high complexity score"

        # Time-based multi-query
        score = decomposer_no_llm._calculate_complexity_score(
            "What's the weather today and tomorrow in Miami?"
        )
        assert score >= 2, "Time-based query should have high complexity score"

    @pytest.mark.asyncio
    async def test_heuristic_multi_location_decomposition(self, decomposer_no_llm):
        """Test heuristic decomposition for multi-location queries."""
        result = await decomposer_no_llm.decompose(
            "What's the weather in Miami and Tampa?"
        )

        assert result.is_complex is True
        assert len(result.sub_queries) >= 2
        assert any("Miami" in q.title() for q in result.sub_queries)
        assert any("Tampa" in q.title() for q in result.sub_queries)

    @pytest.mark.asyncio
    async def test_heuristic_weather_hurricane_decomposition(self, decomposer_no_llm):
        """Test heuristic decomposition for weather + hurricane queries."""
        result = await decomposer_no_llm.decompose(
            "What's the weather forecast and hurricane status for Florida?"
        )

        assert result.is_complex is True
        assert len(result.sub_queries) == 2
        # Should have one weather and one hurricane sub-query
        sub_queries_lower = [q.lower() for q in result.sub_queries]
        assert any("weather" in q for q in sub_queries_lower)
        assert any("hurricane" in q for q in sub_queries_lower)

    @pytest.mark.asyncio
    async def test_heuristic_time_decomposition(self, decomposer_no_llm):
        """Test heuristic decomposition for today + tomorrow queries."""
        result = await decomposer_no_llm.decompose(
            "What's the weather today and tomorrow in Miami?"
        )

        assert result.is_complex is True
        assert len(result.sub_queries) == 2
        sub_queries_lower = [q.lower() for q in result.sub_queries]
        assert any("today" in q for q in sub_queries_lower)
        assert any("tomorrow" in q for q in sub_queries_lower)

    @pytest.mark.asyncio
    async def test_simple_query_returns_original(self, decomposer_no_llm):
        """Test that simple queries return original query unchanged."""
        result = await decomposer_no_llm.decompose(
            "What's the weather in Miami?"
        )

        assert result.is_complex is False
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What's the weather in Miami?"

    @pytest.mark.asyncio
    async def test_decomposition_statistics(self, decomposer_no_llm):
        """Test that statistics are tracked correctly."""
        # Simple query
        await decomposer_no_llm.decompose("What's the weather?")

        # Complex query
        await decomposer_no_llm.decompose("Weather in Miami and Tampa?")

        stats = decomposer_no_llm.get_stats()

        assert stats["total_queries"] == 2
        assert stats["complex_queries"] == 1
        assert stats["complex_rate"] == 0.5  # 1/2

    def test_decomposed_query_dataclass(self):
        """Test DecomposedQuery dataclass."""
        result = DecomposedQuery(
            original_query="Test query",
            sub_queries=["Sub 1", "Sub 2"],
            is_complex=True,
            reasoning="Multiple locations detected",
            decomposition_time_ms=15.5,
        )

        assert result.original_query == "Test query"
        assert len(result.sub_queries) == 2
        assert result.is_complex is True
        assert result.reasoning == "Multiple locations detected"
        assert result.decomposition_time_ms == 15.5

    @pytest.mark.asyncio
    async def test_three_city_query(self, decomposer_no_llm):
        """Test decomposition limits to 3 sub-queries for locations."""
        result = await decomposer_no_llm.decompose(
            "What's the weather in Miami, Tampa, Orlando, and Jacksonville?"
        )

        assert result.is_complex is True
        # Should limit to 3 sub-queries
        assert len(result.sub_queries) <= 3

    @pytest.mark.asyncio
    async def test_comparison_query(self, decomposer_no_llm):
        """Test that comparison queries are detected as complex."""
        score = decomposer_no_llm._calculate_complexity_score(
            "Compare the weather between Miami and Tampa"
        )
        assert score >= 2, "Comparison query should have high complexity"

    @pytest.mark.asyncio
    async def test_weekend_query(self, decomposer_no_llm):
        """Test that weekend queries are detected."""
        score = decomposer_no_llm._calculate_complexity_score(
            "What's the weather this weekend in Florida?"
        )
        assert score >= 1, "Weekend query should have some complexity"

    @pytest.mark.asyncio
    async def test_long_query_complexity(self, decomposer_no_llm):
        """Test that very long queries get complexity bonus."""
        short_query = "Weather in Miami?"
        long_query = "I'm planning a trip to Florida next week and I want to know " \
                    "what the weather will be like in Miami, Tampa, and Orlando, " \
                    "and also if there are any hurricane warnings I should be aware of."

        short_score = decomposer_no_llm._calculate_complexity_score(short_query)
        long_score = decomposer_no_llm._calculate_complexity_score(long_query)

        assert long_score > short_score, "Long queries should have higher complexity"

    @pytest.mark.asyncio
    async def test_evacuation_query(self, decomposer_no_llm):
        """Test that evacuation queries are detected as complex."""
        score = decomposer_no_llm._calculate_complexity_score(
            "Should I evacuate from Miami for the hurricane?"
        )
        # Evacuation is a safety-critical keyword
        assert score >= 1, "Evacuation query should be flagged"


class TestQueryDecomposerSingleton:
    """Test get_query_decomposer singleton pattern."""

    def test_get_decomposer_returns_instance(self):
        """Test that get_query_decomposer returns an instance."""
        # Reset singleton for test
        import backend.src.rag.query_decomposer as module
        module._decomposer = None

        decomposer = get_query_decomposer(enable_llm=False)

        assert decomposer is not None
        assert isinstance(decomposer, QueryDecomposer)

    def test_get_decomposer_singleton(self):
        """Test that get_query_decomposer returns same instance."""
        # Reset singleton for test
        import backend.src.rag.query_decomposer as module
        module._decomposer = None

        decomposer1 = get_query_decomposer(enable_llm=False)
        decomposer2 = get_query_decomposer(enable_llm=False)

        assert decomposer1 is decomposer2, "Should return same singleton instance"


class TestConvenienceFunction:
    """Test decompose_query convenience function."""

    @pytest.mark.asyncio
    async def test_decompose_query_function(self):
        """Test that decompose_query convenience function works."""
        # Reset singleton for test
        import backend.src.rag.query_decomposer as module
        module._decomposer = None

        # Mock the singleton creation to avoid needing LLM
        with patch.object(module, 'get_query_decomposer') as mock_get:
            mock_decomposer = AsyncMock()
            mock_decomposer.decompose.return_value = DecomposedQuery(
                original_query="Test",
                sub_queries=["Test"],
                is_complex=False,
                reasoning="Test",
            )
            mock_get.return_value = mock_decomposer

            result = await decompose_query("Test query")

            assert result is not None
            mock_decomposer.decompose.assert_called_once_with("Test query")
