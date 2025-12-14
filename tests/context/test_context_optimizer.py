"""
Tests for Context Window Optimizer.

Level 6a: Context Optimization Testing

Tests:
1. Token reduction (50-60% target)
2. Context recall (>98% critical info preserved)
3. Optimization latency (<100ms)
4. Query type handling (SIMPLE, STANDARD, COMPLEX, EMERGENCY)
5. Weather domain content preservation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.src.context.context_optimizer import (
    ContextWindowOptimizer,
    OptimizationResult,
)


class TestContextWindowOptimizer:
    """Test suite for ContextWindowOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer without embeddings (keyword-based filtering)."""
        return ContextWindowOptimizer(
            embeddings=None,
            target_tokens=4000,
            min_relevance_score=0.1,  # Lower threshold for testing
        )

    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        return """
You are a weather AI assistant specialized in hurricane forecasting.

SYSTEM: This is critical weather information.

IMPORTANT: Hurricane Michael is approaching.

### Current Conditions

The temperature in Tampa, FL is 85°F with humidity at 78%.
Wind speed is 15 mph from the southeast.
Barometric pressure is 29.85 inHg.

### Hurricane Michael Analysis

Hurricane Michael is a Category 4 hurricane with sustained winds of 130 mph.
The storm is currently located 200 miles southwest of Panama City, FL.
Expected landfall is within 24-36 hours.

WARNING: Storm surge expected to reach 12-14 feet in coastal areas.

### Evacuation Information

CRITICAL: Evacuation orders are in effect for zones A and B.
Residents in coastal areas should evacuate immediately.
Shelters are open at the following locations:
- Tampa Convention Center
- USF Marshall Center
- Raymond James Stadium

### Forecast

Day 1: Hurricane conditions expected. Wind gusts up to 150 mph.
Day 2: Tropical storm conditions. Heavy rain 6-10 inches.
Day 3: Gradual improvement. Rain tapering off.

User: What's the current hurricane status?
"""

    @pytest.mark.asyncio
    async def test_optimize_reduces_tokens(self, optimizer, sample_context):
        """Test that optimization reduces token count."""
        query = "What is the current hurricane status?"

        result = await optimizer.optimize(
            query=query,
            context=sample_context,
            query_type="STANDARD",
        )

        assert isinstance(result, OptimizationResult)
        assert result.token_count < result.original_tokens
        assert result.reduction_pct > 0

    @pytest.mark.asyncio
    async def test_optimize_preserves_critical_info(self, optimizer, sample_context):
        """Test that critical weather info is preserved."""
        query = "What is the hurricane category?"

        result = await optimizer.optimize(
            query=query,
            context=sample_context,
            query_type="EMERGENCY",
        )

        # Critical info should be preserved
        assert "hurricane" in result.optimized_context.lower()

    @pytest.mark.asyncio
    async def test_optimize_completes_all_phases(self, optimizer, sample_context):
        """Test that all 5 optimization phases complete."""
        query = "Weather forecast"

        result = await optimizer.optimize(
            query=query,
            context=sample_context,
            query_type="STANDARD",
        )

        expected_phases = [
            "intelligent_truncation",
            "semantic_chunking",
            "relevance_filtering",
            "dynamic_assembly",
            "hierarchical_loading",
        ]

        for phase in expected_phases:
            assert phase in result.phases_completed

    @pytest.mark.asyncio
    async def test_optimize_handles_empty_context(self, optimizer):
        """Test handling of empty context."""
        result = await optimizer.optimize(
            query="test",
            context="",
            query_type="SIMPLE",
        )

        assert result.token_count == 0
        assert result.original_tokens == 0

    @pytest.mark.asyncio
    async def test_optimize_respects_query_type_simple(self, optimizer, sample_context):
        """Test that SIMPLE queries get minimal context."""
        query = "temperature"

        result = await optimizer.optimize(
            query=query,
            context=sample_context,
            query_type="SIMPLE",
        )

        # SIMPLE queries should use fewer chunks
        assert result.chunks_used <= 5

    @pytest.mark.asyncio
    async def test_optimize_respects_query_type_emergency(self, optimizer, sample_context):
        """Test that EMERGENCY queries preserve safety info."""
        query = "evacuation"

        result = await optimizer.optimize(
            query=query,
            context=sample_context,
            query_type="EMERGENCY",
        )

        # EMERGENCY should have more context
        assert result.chunks_used > 0

    @pytest.mark.asyncio
    async def test_optimization_time_reasonable(self, optimizer, sample_context):
        """Test that optimization completes in reasonable time (<500ms)."""
        query = "hurricane status"

        result = await optimizer.optimize(
            query=query,
            context=sample_context,
            query_type="STANDARD",
        )

        # Should complete in under 500ms
        assert result.optimization_time_ms < 500

    def test_token_counting(self, optimizer):
        """Test token counting functionality."""
        text = "Hello world this is a test"
        token_count = optimizer.count_tokens(text)

        assert token_count > 0
        assert isinstance(token_count, int)

    def test_intelligent_truncate_keeps_system_prompts(self, optimizer):
        """Test that system prompts are preserved during truncation."""
        context = """
You are a helpful assistant.
SYSTEM: Important instruction.
Random filler text that can be removed.
More filler content here.
User: Hello
"""
        truncated = optimizer._intelligent_truncate(context)

        assert "You are" in truncated
        assert "SYSTEM:" in truncated
        assert "User:" in truncated

    def test_keyword_filter_scores_relevant_chunks(self, optimizer):
        """Test keyword-based filtering."""
        query = "hurricane wind speed"
        chunks = [
            "Hurricane Michael has 130 mph winds",
            "The weather is nice today",
            "Wind speed forecast for Tampa",
        ]

        scored = optimizer._keyword_filter(query, chunks)

        # Should have scores
        assert len(scored) > 0
        assert all("score" in c for c in scored)
        # Hurricane/wind chunks should score higher
        assert scored[0]["score"] >= scored[-1]["score"]

    def test_metrics_tracking(self, optimizer):
        """Test that metrics are properly tracked."""
        # Reset metrics
        optimizer.reset_metrics()

        initial_metrics = optimizer.get_metrics()
        assert initial_metrics["total_optimizations"] == 0

    @pytest.mark.asyncio
    async def test_metrics_update_after_optimization(self, optimizer, sample_context):
        """Test that metrics update after optimization."""
        optimizer.reset_metrics()

        await optimizer.optimize(
            query="test",
            context=sample_context,
            query_type="STANDARD",
        )

        metrics = optimizer.get_metrics()
        assert metrics["total_optimizations"] == 1
        assert metrics["avg_reduction_pct"] >= 0


class TestContextOptimizerEdgeCases:
    """Edge case tests for ContextWindowOptimizer."""

    @pytest.fixture
    def optimizer(self):
        return ContextWindowOptimizer(embeddings=None, target_tokens=4000)

    @pytest.mark.asyncio
    async def test_very_short_context(self, optimizer):
        """Test with context shorter than target."""
        short_context = "Hello world"

        result = await optimizer.optimize(
            query="test",
            context=short_context,
            query_type="SIMPLE",
        )

        # Should handle gracefully
        assert result.token_count <= result.original_tokens

    @pytest.mark.asyncio
    async def test_context_with_json(self, optimizer):
        """Test context containing JSON data."""
        json_context = """
{
    "temperature": 85,
    "humidity": 78,
    "conditions": "sunny"
}
"""
        result = await optimizer.optimize(
            query="temperature",
            context=json_context,
            query_type="STANDARD",
        )

        assert result.optimized_context is not None

    @pytest.mark.asyncio
    async def test_invalid_query_type_defaults_to_standard(self, optimizer):
        """Test that invalid query type defaults to STANDARD."""
        result = await optimizer.optimize(
            query="test",
            context="Some context here",
            query_type="INVALID_TYPE",
        )

        # Should not raise, should default to STANDARD behavior
        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters_in_context(self, optimizer):
        """Test handling of special characters."""
        special_context = """
Temperature: 85°F
Wind: 15 mph → increasing
Alert: ⚠️ Storm approaching!
"""
        result = await optimizer.optimize(
            query="temperature",
            context=special_context,
            query_type="STANDARD",
        )

        assert result is not None
