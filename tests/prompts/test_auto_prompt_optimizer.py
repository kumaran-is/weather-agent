"""
Tests for Auto-Prompt Optimizer.

Level 6b: Self-Improvement Platform

Tests:
1. Variation generation
2. Prompt optimization
3. Best prompt selection
4. History tracking
"""

import pytest

from backend.src.prompts.auto_prompt_optimizer import (
    AutoPromptOptimizer,
    OptimizationResult,
    OptimizationStatus,
    VariationResult,
)


class TestAutoPromptOptimizer:
    """Test suite for AutoPromptOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer without external dependencies."""
        return AutoPromptOptimizer(
            llm=None,
            evaluator=None,
        )

    @pytest.fixture
    def base_prompt(self):
        """Sample base prompt for testing."""
        return "You are a helpful weather assistant. Provide accurate weather information."

    @pytest.fixture
    def test_queries(self):
        """Sample test queries."""
        return [
            "What's the weather in Miami?",
            "Will it rain tomorrow?",
            "Is there a hurricane warning?",
        ]

    @pytest.mark.asyncio
    async def test_generate_variations(self, optimizer, base_prompt):
        """Test generating variations."""
        variations = await optimizer.generate_variations(
            base_prompt=base_prompt,
            num_variations=3,
        )

        # Original + 3 variations = 4 total
        assert len(variations) == 4
        for variation in variations:
            assert isinstance(variation, VariationResult)
            assert variation.prompt_text != ""
            assert variation.technique != ""

    @pytest.mark.asyncio
    async def test_generate_variations_with_techniques(self, optimizer, base_prompt):
        """Test generating variations with specific techniques."""
        techniques = ["formal_tone", "step_by_step"]

        variations = await optimizer.generate_variations(
            base_prompt=base_prompt,
            num_variations=2,
            techniques=techniques,
        )

        # Original + 2 variations = 3 total
        assert len(variations) == 3
        # Should use specified techniques (excluding original)
        used_techniques = [v.technique for v in variations if v.technique != "original"]
        assert any(t in used_techniques for t in techniques)

    @pytest.mark.asyncio
    async def test_optimize_prompt_returns_result(
        self, optimizer, base_prompt, test_queries
    ):
        """Test that optimize_prompt returns OptimizationResult."""
        result = await optimizer.optimize_prompt(
            base_prompt=base_prompt,
            test_queries=test_queries,
            num_variations=3,
        )

        assert isinstance(result, OptimizationResult)
        assert result.original_prompt == base_prompt
        assert result.best_prompt != ""
        assert result.improvement_pct is not None
        assert result.variations_tested > 0

    @pytest.mark.asyncio
    async def test_optimize_prompt_selects_best(
        self, optimizer, base_prompt, test_queries
    ):
        """Test that optimization selects best prompt."""
        result = await optimizer.optimize_prompt(
            base_prompt=base_prompt,
            test_queries=test_queries,
            num_variations=5,
        )

        # Best score should be the highest in all_scores
        assert result.best_score >= result.original_score or result.best_variation_id == "original"

    @pytest.mark.asyncio
    async def test_optimize_prompt_tracks_metrics(
        self, optimizer, base_prompt, test_queries
    ):
        """Test that optimization tracks metrics."""
        result = await optimizer.optimize_prompt(
            base_prompt=base_prompt,
            test_queries=test_queries,
            num_variations=3,
        )

        assert result.queries_per_variation > 0
        assert result.optimization_time_ms > 0

    @pytest.mark.asyncio
    async def test_optimization_status(self, optimizer, base_prompt, test_queries):
        """Test optimization status updates."""
        assert optimizer.get_status() == OptimizationStatus.PENDING

        await optimizer.optimize_prompt(
            base_prompt=base_prompt,
            test_queries=test_queries,
            num_variations=2,
        )

        assert optimizer.get_status() == OptimizationStatus.COMPLETED


class TestVariationResult:
    """Test VariationResult model."""

    def test_creation(self):
        """Test creating VariationResult."""
        result = VariationResult(
            variation_id="test_1",
            prompt_text="Test prompt",
            technique="formal_tone",
            avg_score=0.85,
            scores=[0.8, 0.9, 0.85],
        )

        assert result.prompt_text == "Test prompt"
        assert result.technique == "formal_tone"
        assert result.avg_score == 0.85
        assert len(result.scores) == 3

    def test_default_values(self):
        """Test default values."""
        result = VariationResult(
            variation_id="test_2",
            prompt_text="Test",
            technique="test",
        )

        assert result.avg_score == 0.0
        assert result.scores == []
        assert result.latency_avg_ms == 0.0


class TestOptimizationResult:
    """Test OptimizationResult model."""

    def test_creation(self):
        """Test creating OptimizationResult."""
        result = OptimizationResult(
            original_prompt="Original",
            best_prompt="Best",
            best_variation_id="variation_1",
            improvement_pct=14.3,
            original_score=0.7,
            best_score=0.8,
            variations_tested=3,
            queries_per_variation=5,
            optimization_time_ms=150.5,
        )

        assert result.best_prompt == "Best"
        assert result.improvement_pct == 14.3
        assert result.variations_tested == 3

    def test_all_scores_tracking(self):
        """Test that all_scores is tracked."""
        result = OptimizationResult(
            original_prompt="Original",
            best_prompt="Best",
            best_variation_id="v1",
            improvement_pct=10.0,
            original_score=0.6,
            best_score=0.66,
            variations_tested=2,
            queries_per_variation=3,
            optimization_time_ms=100.0,
            all_scores={"original": 0.6, "v1": 0.66},
        )

        assert "original" in result.all_scores
        assert "v1" in result.all_scores


class TestOptimizerWithMockLLM:
    """Test optimizer with mock LLM."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""

        class MockLLM:
            async def ainvoke(self, prompt: str) -> object:
                class Response:
                    content = "You are an expert assistant. Provide detailed responses."

                return Response()

        return MockLLM()

    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator."""

        class MockEvaluator:
            async def evaluate_query(self, query: str, answer: str, **kwargs):
                class Result:
                    overall_score = 0.85
                    passed = True

                return Result()

        return MockEvaluator()

    @pytest.mark.asyncio
    async def test_with_llm(self, mock_llm):
        """Test optimizer with LLM."""
        optimizer = AutoPromptOptimizer(llm=mock_llm)

        variations = await optimizer.generate_variations(
            base_prompt="Test prompt",
            num_variations=2,
        )

        assert len(variations) >= 1

    @pytest.mark.asyncio
    async def test_with_evaluator(self, mock_evaluator):
        """Test optimizer with evaluator."""
        optimizer = AutoPromptOptimizer(evaluator=mock_evaluator)

        result = await optimizer.optimize_prompt(
            base_prompt="Test prompt",
            test_queries=["Query 1", "Query 2"],
            num_variations=2,
        )

        # Should have completed optimization
        assert result.variations_tested > 0


class TestOptimizerHistory:
    """Test optimizer history tracking."""

    @pytest.fixture
    def optimizer(self):
        return AutoPromptOptimizer()

    @pytest.mark.asyncio
    async def test_history_tracking(self, optimizer):
        """Test that optimization history is tracked."""
        await optimizer.optimize_prompt(
            base_prompt="Test",
            test_queries=["Q1", "Q2"],
            num_variations=2,
        )

        history = optimizer.get_optimization_history()

        assert len(history) == 1
        assert "original_prompt" in history[0]
        assert "best_prompt" in history[0]

    @pytest.mark.asyncio
    async def test_clear_history(self, optimizer):
        """Test clearing history."""
        await optimizer.optimize_prompt(
            base_prompt="Test",
            test_queries=["Q1"],
            num_variations=2,
        )

        optimizer.clear_history()
        history = optimizer.get_optimization_history()

        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_multiple_optimizations(self, optimizer):
        """Test multiple optimizations tracked."""
        for i in range(3):
            await optimizer.optimize_prompt(
                base_prompt=f"Test prompt {i}",
                test_queries=["Q1", "Q2"],
                num_variations=2,
            )

        history = optimizer.get_optimization_history()
        assert len(history) == 3


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def optimizer(self):
        return AutoPromptOptimizer()

    @pytest.mark.asyncio
    async def test_empty_queries(self, optimizer):
        """Test with empty queries list."""
        result = await optimizer.optimize_prompt(
            base_prompt="Test",
            test_queries=[],
            num_variations=2,
        )

        # Should handle gracefully
        assert result.best_prompt != ""

    @pytest.mark.asyncio
    async def test_single_variation(self, optimizer):
        """Test with single variation."""
        result = await optimizer.optimize_prompt(
            base_prompt="Test",
            test_queries=["Q1"],
            num_variations=1,
        )

        # Original + 1 variation = 2
        assert result.variations_tested >= 1

    @pytest.mark.asyncio
    async def test_short_prompt(self, optimizer):
        """Test with very short prompt."""
        variations = await optimizer.generate_variations(
            base_prompt="Hi",
            num_variations=3,
        )

        # Original + 3 variations = 4
        assert len(variations) == 4

    @pytest.mark.asyncio
    async def test_long_prompt(self, optimizer):
        """Test with long prompt."""
        long_prompt = "You are a helpful assistant. " * 50

        variations = await optimizer.generate_variations(
            base_prompt=long_prompt,
            num_variations=2,
        )

        # Original + 2 variations = 3
        assert len(variations) == 3
