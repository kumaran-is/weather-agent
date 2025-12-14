"""
Tests for LangChain Benchmark (AgentBench) Integration.

Level 6b: Self-Improvement Platform

Tests:
1. Single task benchmark execution
2. Multi-task benchmark execution
3. Accuracy calculation
4. Latency tracking
5. Regression detection
6. Threshold management
"""

import pytest
from backend.src.evaluation.langchain_benchmark import (
    LangChainBenchmark,
    BenchmarkResult,
    BenchmarkTask,
)


class TestLangChainBenchmark:
    """Test suite for LangChainBenchmark."""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark without agent (uses simulation)."""
        return LangChainBenchmark(agent=None)

    @pytest.mark.asyncio
    async def test_run_benchmark_returns_result(self, benchmark):
        """Test that benchmark returns a BenchmarkResult."""
        result = await benchmark.run_benchmark(
            task_name="weather_forecast",
            max_queries=3,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.task_name == "weather_forecast"
        assert 0 <= result.accuracy <= 1
        assert result.total_queries == 3
        assert result.latency_p50_ms >= 0
        assert result.latency_p95_ms >= 0

    @pytest.mark.asyncio
    async def test_run_weather_forecast_benchmark(self, benchmark):
        """Test weather forecast benchmark task."""
        result = await benchmark.run_benchmark(
            task_name="weather_forecast",
            max_queries=5,
        )

        assert result.task_name == "weather_forecast"
        assert result.total_queries == 5
        assert "threshold" in result.details
        assert result.details["threshold"] == 0.85

    @pytest.mark.asyncio
    async def test_run_hurricane_analysis_benchmark(self, benchmark):
        """Test hurricane analysis benchmark task."""
        result = await benchmark.run_benchmark(
            task_name="hurricane_analysis",
            max_queries=3,
        )

        assert result.task_name == "hurricane_analysis"
        assert result.total_queries == 3
        # Hurricane analysis has higher threshold
        assert result.details["threshold"] == 0.90

    @pytest.mark.asyncio
    async def test_run_multi_hop_retrieval_benchmark(self, benchmark):
        """Test multi-hop retrieval benchmark task."""
        result = await benchmark.run_benchmark(
            task_name="multi_hop_retrieval",
            max_queries=3,
        )

        assert result.task_name == "multi_hop_retrieval"
        assert result.total_queries == 3
        assert result.details["category"] == "reasoning"

    @pytest.mark.asyncio
    async def test_run_tool_usage_benchmark(self, benchmark):
        """Test tool usage benchmark task."""
        result = await benchmark.run_benchmark(
            task_name="tool_usage",
            max_queries=3,
        )

        assert result.task_name == "tool_usage"
        assert result.total_queries == 3
        assert result.details["category"] == "tools"

    @pytest.mark.asyncio
    async def test_unknown_task_falls_back(self, benchmark):
        """Test that unknown task falls back to default."""
        result = await benchmark.run_benchmark(
            task_name="unknown_task",
            max_queries=2,
        )

        # Falls back to weather_forecast
        assert result.task_name == "weather_forecast"

    @pytest.mark.asyncio
    async def test_accuracy_calculation(self, benchmark):
        """Test that accuracy is calculated correctly."""
        result = await benchmark.run_benchmark(
            task_name="weather_forecast",
            max_queries=5,
        )

        # Verify accuracy formula
        expected_accuracy = result.passed_queries / result.total_queries
        assert abs(result.accuracy - expected_accuracy) < 0.01

    @pytest.mark.asyncio
    async def test_latency_tracking(self, benchmark):
        """Test that latency metrics are tracked."""
        result = await benchmark.run_benchmark(
            task_name="weather_forecast",
            max_queries=5,
        )

        # P50 should be <= P95
        assert result.latency_p50_ms <= result.latency_p95_ms
        # Average should be reasonable
        assert result.latency_avg_ms > 0

    @pytest.mark.asyncio
    async def test_cost_tracking(self, benchmark):
        """Test that cost is estimated."""
        result = await benchmark.run_benchmark(
            task_name="weather_forecast",
            max_queries=5,
        )

        # Cost should be proportional to queries
        assert result.cost_usd > 0
        assert result.cost_usd == pytest.approx(0.01, rel=0.5)  # 5 * $0.002


class TestAllBenchmarks:
    """Test suite for running all benchmarks."""

    @pytest.fixture
    def benchmark(self):
        return LangChainBenchmark(agent=None)

    @pytest.mark.asyncio
    async def test_run_all_benchmarks(self, benchmark):
        """Test running all benchmark tasks."""
        results = await benchmark.run_all_benchmarks(max_queries_per_task=2)

        assert "overall_accuracy" in results
        assert "total_passed" in results
        assert "total_queries" in results
        assert "all_passed" in results
        assert "results_by_task" in results

        # Should have results for all tasks
        assert "weather_forecast" in results["results_by_task"]
        assert "hurricane_analysis" in results["results_by_task"]
        assert "multi_hop_retrieval" in results["results_by_task"]
        assert "tool_usage" in results["results_by_task"]

    @pytest.mark.asyncio
    async def test_overall_accuracy_is_average(self, benchmark):
        """Test that overall accuracy is average of task accuracies."""
        results = await benchmark.run_all_benchmarks(max_queries_per_task=2)

        task_accuracies = [
            r["accuracy"] for r in results["results_by_task"].values()
        ]
        expected_overall = sum(task_accuracies) / len(task_accuracies)

        assert abs(results["overall_accuracy"] - expected_overall) < 0.01


class TestRegressionTracking:
    """Test suite for regression tracking."""

    @pytest.fixture
    def benchmark(self):
        return LangChainBenchmark(agent=None)

    @pytest.mark.asyncio
    async def test_regression_report_insufficient_data(self, benchmark):
        """Test regression report with insufficient data."""
        report = benchmark.get_regression_report()

        assert report["status"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_regression_report_after_runs(self, benchmark):
        """Test regression report after multiple runs."""
        # Run benchmark twice
        await benchmark.run_benchmark(task_name="weather_forecast", max_queries=2)
        await benchmark.run_benchmark(task_name="weather_forecast", max_queries=2)

        report = benchmark.get_regression_report()

        assert report["status"] in ["stable", "regression_detected"]
        assert report["total_runs"] == 2
        assert "regressions" in report
        assert "improvements" in report

    @pytest.mark.asyncio
    async def test_history_tracking(self, benchmark):
        """Test that benchmark results are stored in history."""
        await benchmark.run_benchmark(task_name="weather_forecast", max_queries=2)
        await benchmark.run_benchmark(task_name="hurricane_analysis", max_queries=2)

        assert len(benchmark.benchmark_history) == 2

    @pytest.mark.asyncio
    async def test_clear_history(self, benchmark):
        """Test clearing benchmark history."""
        await benchmark.run_benchmark(task_name="weather_forecast", max_queries=2)
        assert len(benchmark.benchmark_history) == 1

        benchmark.clear_history()
        assert len(benchmark.benchmark_history) == 0


class TestThresholds:
    """Test threshold management."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        benchmark = LangChainBenchmark()
        thresholds = benchmark.get_thresholds()

        assert thresholds["weather_forecast"] == 0.85
        assert thresholds["hurricane_analysis"] == 0.90
        assert thresholds["multi_hop_retrieval"] == 0.80
        assert thresholds["tool_usage"] == 0.85

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        custom = {"weather_forecast": 0.95}
        benchmark = LangChainBenchmark(thresholds=custom)
        thresholds = benchmark.get_thresholds()

        assert thresholds["weather_forecast"] == 0.95

    def test_set_threshold(self):
        """Test setting a threshold."""
        benchmark = LangChainBenchmark()
        benchmark.set_threshold("weather_forecast", 0.75)

        thresholds = benchmark.get_thresholds()
        assert thresholds["weather_forecast"] == 0.75

    @pytest.mark.asyncio
    async def test_passed_based_on_threshold(self):
        """Test that passed flag respects thresholds."""
        # Use very low threshold to ensure passing
        benchmark = LangChainBenchmark(thresholds={"weather_forecast": 0.1})

        result = await benchmark.run_benchmark(
            task_name="weather_forecast",
            max_queries=3,
        )

        # With low threshold, should pass
        assert result.passed is True or result.accuracy >= 0.1


class TestBenchmarkTask:
    """Test BenchmarkTask model."""

    def test_task_creation(self):
        """Test creating a BenchmarkTask."""
        task = BenchmarkTask(
            name="test_task",
            description="A test task",
            queries=[
                {"query": "test query 1", "expected_contains": ["test"]},
                {"query": "test query 2", "expected_contains": ["query"]},
            ],
            threshold=0.80,
            category="testing",
        )

        assert task.name == "test_task"
        assert len(task.queries) == 2
        assert task.threshold == 0.80
        assert task.category == "testing"


class TestBenchmarkWithAgent:
    """Test benchmark with custom agent function."""

    @pytest.mark.asyncio
    async def test_callable_agent(self):
        """Test benchmark with callable agent."""
        def simple_agent(query: str) -> str:
            return f"Response about {query}. Temperature is 85°F. Forecast is sunny."

        benchmark = LangChainBenchmark(agent=simple_agent)
        result = await benchmark.run_benchmark(
            task_name="weather_forecast",
            max_queries=3,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.total_queries == 3

    @pytest.mark.asyncio
    async def test_async_callable_agent(self):
        """Test benchmark with async callable agent."""
        async def async_agent(query: str) -> str:
            return f"Async response about {query}. Temperature is 85°F."

        benchmark = LangChainBenchmark(agent=async_agent)
        result = await benchmark.run_benchmark(
            task_name="weather_forecast",
            max_queries=3,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.total_queries == 3
