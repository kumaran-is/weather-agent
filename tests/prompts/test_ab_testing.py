"""
Tests for A/B Testing Framework.

Level 6b: Self-Improvement Platform

Tests:
1. Test creation and management
2. Test execution with simulated scores
3. Statistical significance
4. Multi-armed bandit selection
5. Results and recommendations
"""

import pytest

from backend.src.prompts.ab_testing import (
    ABTestResult,
    ABTestRunner,
    MultiArmedBandit,
    TestStatus,
    VariantStats,
)


class TestABTestRunner:
    """Test suite for ABTestRunner."""

    @pytest.fixture
    def runner(self):
        """Create A/B test runner."""
        return ABTestRunner(
            evaluator=None,
            min_trials_per_variant=10,
            confidence_threshold=0.95,
        )

    @pytest.fixture
    def variants(self):
        """Sample variants for testing."""
        return {
            "control": "You are a helpful assistant.",
            "variant_a": "You are an expert weather assistant.",
            "variant_b": "Hey! I'm here to help with weather questions.",
        }

    @pytest.mark.asyncio
    async def test_create_test(self, runner, variants):
        """Test creating an A/B test."""
        test_id = await runner.create_test(
            test_id="test_001",
            variants=variants,
        )

        assert test_id == "test_001"
        assert test_id in runner.active_tests
        assert len(runner.active_tests[test_id]["variants"]) == 3

    @pytest.mark.asyncio
    async def test_create_test_with_queries(self, runner, variants):
        """Test creating test with custom queries."""
        queries = ["What's the weather?", "Will it rain?"]

        await runner.create_test(
            test_id="test_002",
            variants=variants,
            test_queries=queries,
        )

        test = runner.active_tests["test_002"]
        assert test["test_queries"] == queries

    @pytest.mark.asyncio
    async def test_run_test_returns_result(self, runner, variants):
        """Test that running test returns ABTestResult."""
        await runner.create_test(test_id="test_003", variants=variants)

        result = await runner.run_test(
            test_id="test_003",
            agent_fn=None,
            max_trials=20,
        )

        assert isinstance(result, ABTestResult)
        assert result.test_id == "test_003"
        assert result.status == TestStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_test_tracks_trials(self, runner, variants):
        """Test that trials are tracked correctly."""
        await runner.create_test(test_id="test_004", variants=variants)

        result = await runner.run_test(
            test_id="test_004",
            max_trials=15,
        )

        # Each variant should have some trials
        for variant_name, stats in result.variants.items():
            assert stats["trials"] > 0

        # Total trials should be max_trials * num_variants
        assert result.total_trials == 15 * 3

    @pytest.mark.asyncio
    async def test_run_test_determines_winner(self, runner, variants):
        """Test that a winner is determined."""
        await runner.create_test(test_id="test_005", variants=variants)

        result = await runner.run_test(
            test_id="test_005",
            max_trials=30,
        )

        # Winner should be one of the variants
        assert result.winner in variants or result.winner is None
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_run_test_provides_recommendation(self, runner, variants):
        """Test that recommendation is provided."""
        await runner.create_test(test_id="test_006", variants=variants)

        result = await runner.run_test(
            test_id="test_006",
            max_trials=30,
        )

        assert result.recommendation != ""
        assert any(
            word in result.recommendation
            for word in ["RECOMMEND", "TENTATIVE", "Insufficient"]
        )

    @pytest.mark.asyncio
    async def test_run_test_not_found_raises(self, runner):
        """Test that running unknown test raises."""
        with pytest.raises(ValueError, match="not found"):
            await runner.run_test(test_id="unknown")

    @pytest.mark.asyncio
    async def test_get_active_tests(self, runner, variants):
        """Test getting active test list."""
        await runner.create_test(test_id="test_a", variants=variants)
        await runner.create_test(test_id="test_b", variants=variants)

        active = runner.get_active_tests()

        assert "test_a" in active
        assert "test_b" in active
        assert len(active) == 2

    @pytest.mark.asyncio
    async def test_get_test_status(self, runner, variants):
        """Test getting test status."""
        await runner.create_test(test_id="test_007", variants=variants)

        status = runner.get_test_status("test_007")

        assert status is not None
        assert status["test_id"] == "test_007"
        assert status["status"] == "pending"
        assert len(status["variants"]) == 3

    def test_get_test_status_not_found(self, runner):
        """Test status of unknown test."""
        status = runner.get_test_status("unknown")
        assert status is None


class TestABTestWithAgent:
    """Test A/B testing with agent functions."""

    @pytest.fixture
    def runner(self):
        return ABTestRunner(min_trials_per_variant=5)

    @pytest.mark.asyncio
    async def test_with_sync_agent(self, runner):
        """Test with synchronous agent function."""

        def agent_fn(prompt: str, query: str) -> str:
            return f"Response about {query}. Temperature is 75°F."

        variants = {"control": "Prompt A", "variant": "Prompt B"}
        await runner.create_test(test_id="sync_test", variants=variants)

        result = await runner.run_test(
            test_id="sync_test",
            agent_fn=agent_fn,
            max_trials=10,
        )

        assert result.total_trials == 20  # 10 * 2 variants

    @pytest.mark.asyncio
    async def test_with_async_agent(self, runner):
        """Test with asynchronous agent function."""

        async def agent_fn(prompt: str, query: str) -> str:
            return f"Async response about {query}. Sunny weather."

        variants = {"control": "Prompt A", "variant": "Prompt B"}
        await runner.create_test(test_id="async_test", variants=variants)

        result = await runner.run_test(
            test_id="async_test",
            agent_fn=agent_fn,
            max_trials=10,
        )

        assert result.total_trials == 20

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, runner):
        """Test that agent errors are handled gracefully."""

        def failing_agent(prompt: str, query: str) -> str:
            raise Exception("Agent failed")

        variants = {"control": "Prompt A"}
        await runner.create_test(test_id="error_test", variants=variants)

        # Should not raise, should handle gracefully
        result = await runner.run_test(
            test_id="error_test",
            agent_fn=failing_agent,
            max_trials=5,
        )

        assert result.status == TestStatus.COMPLETED


class TestStatisticalSignificance:
    """Test statistical significance testing."""

    @pytest.fixture
    def runner(self):
        return ABTestRunner(
            min_trials_per_variant=10,
            confidence_threshold=0.95,
        )

    def test_check_significance_insufficient_data(self, runner):
        """Test significance with insufficient data."""
        variants = {
            "a": VariantStats(name="a", trials=5, successes=4, failures=1),
            "b": VariantStats(name="b", trials=5, successes=3, failures=2),
        }

        is_sig, p_value = runner._check_significance(variants)

        # Not enough trials
        assert is_sig is False
        assert p_value is None

    def test_check_significance_with_data(self, runner):
        """Test significance with sufficient data."""
        variants = {
            "a": VariantStats(name="a", trials=50, successes=45, failures=5),
            "b": VariantStats(name="b", trials=50, successes=25, failures=25),
        }

        is_sig, p_value = runner._check_significance(variants)

        # Large difference should be significant
        assert p_value is not None

    def test_determine_winner_clear_margin(self, runner):
        """Test determining winner with clear margin."""
        variants = {
            "control": VariantStats(
                name="control",
                trials=30,
                successes=15,
                failures=15,
                success_rate=0.5,
            ),
            "winner": VariantStats(
                name="winner",
                trials=30,
                successes=27,
                failures=3,
                success_rate=0.9,
            ),
        }

        winner, confidence, p_value, is_sig = runner._determine_winner(variants)

        assert winner == "winner"
        assert confidence > 0.5

    def test_determine_winner_close_race(self, runner):
        """Test determining winner in close race."""
        variants = {
            "a": VariantStats(
                name="a",
                trials=20,
                successes=10,
                failures=10,
                success_rate=0.5,
            ),
            "b": VariantStats(
                name="b",
                trials=20,
                successes=11,
                failures=9,
                success_rate=0.55,
            ),
        }

        winner, confidence, p_value, is_sig = runner._determine_winner(variants)

        # Should pick b but with lower confidence
        assert winner == "b"
        # Not likely to be significant
        assert is_sig is False or confidence < 0.9


class TestStopTest:
    """Test stopping tests early."""

    @pytest.fixture
    def runner(self):
        return ABTestRunner(min_trials_per_variant=10)

    @pytest.mark.asyncio
    async def test_stop_running_test(self, runner):
        """Test stopping a running test."""
        variants = {"a": "Prompt A", "b": "Prompt B"}
        await runner.create_test(test_id="stop_test", variants=variants)

        # Manually set to running
        runner.active_tests["stop_test"]["status"] = TestStatus.RUNNING
        runner.active_tests["stop_test"]["variants"]["a"].trials = 10
        runner.active_tests["stop_test"]["variants"]["a"].successes = 7
        runner.active_tests["stop_test"]["variants"]["a"].success_rate = 0.7
        runner.active_tests["stop_test"]["variants"]["b"].trials = 10
        runner.active_tests["stop_test"]["variants"]["b"].successes = 5
        runner.active_tests["stop_test"]["variants"]["b"].success_rate = 0.5

        result = runner.stop_test("stop_test")

        assert result is not None
        assert result.status == TestStatus.STOPPED
        assert result.winner is not None

    def test_stop_pending_test_returns_none(self, runner):
        """Test stopping a pending test returns None."""
        # No running tests
        result = runner.stop_test("nonexistent")
        assert result is None


class TestTestHistory:
    """Test history tracking."""

    @pytest.fixture
    def runner(self):
        return ABTestRunner(min_trials_per_variant=5)

    @pytest.mark.asyncio
    async def test_history_tracking(self, runner):
        """Test that completed tests are added to history."""
        variants = {"a": "Prompt A"}
        await runner.create_test(test_id="hist_test", variants=variants)
        await runner.run_test(test_id="hist_test", max_trials=10)

        history = runner.get_history()

        assert len(history) == 1
        assert history[0].test_id == "hist_test"

    @pytest.mark.asyncio
    async def test_history_limit(self, runner):
        """Test history limit."""
        variants = {"a": "Prompt A"}

        for i in range(5):
            await runner.create_test(test_id=f"test_{i}", variants=variants)
            await runner.run_test(test_id=f"test_{i}", max_trials=10)

        history = runner.get_history(limit=3)
        assert len(history) == 3

    def test_clear_history(self, runner):
        """Test clearing history."""
        runner.test_history = [ABTestResult(test_id="old")]

        runner.clear_history()

        assert len(runner.test_history) == 0


class TestMultiArmedBandit:
    """Test MultiArmedBandit for adaptive selection."""

    @pytest.fixture
    def bandit(self):
        return MultiArmedBandit(variants=["a", "b", "c"])

    def test_initialization(self, bandit):
        """Test bandit initialization."""
        assert len(bandit.variants) == 3
        assert bandit.total_pulls == 0

        # Check priors
        for v in bandit.variants:
            assert bandit.alpha[v] == 1
            assert bandit.beta[v] == 1

    def test_select_variant(self, bandit):
        """Test variant selection."""
        selected = bandit.select_variant()

        assert selected in bandit.variants
        assert bandit.total_pulls == 1
        assert len(bandit.selection_history) == 1

    def test_update_success(self, bandit):
        """Test updating with success."""
        bandit.update("a", success=True)

        assert bandit.alpha["a"] == 2  # 1 prior + 1 success
        assert bandit.beta["a"] == 1   # unchanged

    def test_update_failure(self, bandit):
        """Test updating with failure."""
        bandit.update("a", success=False)

        assert bandit.alpha["a"] == 1  # unchanged
        assert bandit.beta["a"] == 2   # 1 prior + 1 failure

    def test_update_unknown_variant(self, bandit):
        """Test updating unknown variant does nothing."""
        bandit.update("unknown", success=True)

        # Should not affect anything
        for v in bandit.variants:
            assert bandit.alpha[v] == 1
            assert bandit.beta[v] == 1

    def test_exploitation_after_learning(self, bandit):
        """Test that bandit exploits best arm after learning."""
        # Train variant "a" to be clearly better
        for _ in range(20):
            bandit.update("a", success=True)

        for _ in range(20):
            bandit.update("b", success=False)

        # Now selection should favor "a" most of the time
        selections = [bandit.select_variant() for _ in range(100)]
        a_count = selections.count("a")

        # "a" should be selected most of the time
        assert a_count > 70

    def test_get_statistics(self, bandit):
        """Test getting statistics."""
        bandit.update("a", success=True)
        bandit.update("a", success=True)
        bandit.update("b", success=False)
        bandit.total_pulls = 3

        stats = bandit.get_statistics()

        assert "variants" in stats
        assert "total_pulls" in stats
        assert "recommendation" in stats
        assert stats["total_pulls"] == 3

        # Check variant stats
        assert "a" in stats["variants"]
        assert "expected_value" in stats["variants"]["a"]
        assert "confidence_interval" in stats["variants"]["a"]

    def test_get_recommendation_insufficient_data(self, bandit):
        """Test recommendation with insufficient data."""
        stats = bandit.get_statistics()

        assert "Continue exploration" in stats["recommendation"]

    def test_get_recommendation_with_data(self, bandit):
        """Test recommendation with sufficient data."""
        # Simulate many trials
        for _ in range(60):
            bandit.update("a", success=True)
            bandit.total_pulls += 1

        stats = bandit.get_statistics()

        assert any(
            word in stats["recommendation"]
            for word in ["EXPLOIT", "EXPLORE", "INVESTIGATE"]
        )

    def test_reset(self, bandit):
        """Test resetting bandit."""
        bandit.update("a", success=True)
        bandit.total_pulls = 10
        bandit.selection_history = ["a", "b", "c"]

        bandit.reset()

        assert bandit.total_pulls == 0
        assert len(bandit.selection_history) == 0
        assert bandit.alpha["a"] == 1
        assert bandit.beta["a"] == 1


class TestVariantStats:
    """Test VariantStats model."""

    def test_default_values(self):
        """Test default values."""
        stats = VariantStats(name="test")

        assert stats.name == "test"
        assert stats.trials == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.success_rate == 0.0
        assert stats.avg_score == 0.0
        assert stats.scores == []

    def test_with_values(self):
        """Test with values."""
        stats = VariantStats(
            name="test",
            trials=10,
            successes=7,
            failures=3,
            success_rate=0.7,
            avg_score=0.75,
            scores=[0.8, 0.7, 0.75],
        )

        assert stats.trials == 10
        assert stats.success_rate == 0.7
        assert len(stats.scores) == 3
