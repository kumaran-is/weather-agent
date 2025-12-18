"""Tests for Parallel Executor (Level 4b Phase 10).

Test Coverage:
- Parallel task execution
- Semaphore-based concurrency control
- Fallback execution patterns
- Error handling and timeout management
- Performance metrics collection
"""

import asyncio

import pytest

from backend.src.orchestration.parallel_executor import (
    ParallelExecutor,
    ParallelResult,
    TaskResult,
    execute_parallel,
    execute_with_fallback,
)


# Fixtures
@pytest.fixture
def parallel_executor() -> ParallelExecutor:
    """Create ParallelExecutor for testing."""
    return ParallelExecutor(max_concurrent=4)


# Test Helper Functions
async def successful_task(value: str, delay: float = 0.01) -> str:
    """Helper task that succeeds."""
    await asyncio.sleep(delay)
    return f"Result: {value}"


async def failing_task(value: str) -> str:
    """Helper task that fails."""
    raise ValueError(f"Task failed for {value}")


async def slow_task(value: str, delay: float = 1.0) -> str:
    """Helper task that is slow."""
    await asyncio.sleep(delay)
    return f"Slow result: {value}"


# Test: TaskResult
class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_success(self):
        """Test successful TaskResult."""
        result = TaskResult(
            task_id="task_1",
            success=True,
            result="Test result",
            error=None,
            execution_time_ms=100.0,
        )

        assert result.success is True
        assert result.result == "Test result"
        assert result.error is None

    def test_task_result_failure(self):
        """Test failed TaskResult."""
        result = TaskResult(
            task_id="task_2",
            success=False,
            result=None,
            error="Task error",
            execution_time_ms=50.0,
        )

        assert result.success is False
        assert result.result is None
        assert result.error == "Task error"


# Test: ParallelResult
class TestParallelResult:
    """Tests for ParallelResult dataclass."""

    def test_parallel_result_all_success(self):
        """Test ParallelResult with all successes."""
        results = [
            TaskResult("t1", True, "r1", None, 100),
            TaskResult("t2", True, "r2", None, 150),
        ]
        parallel = ParallelResult(
            results=results,
            total_time_ms=150,
            successful_count=2,
            failed_count=0,
        )

        assert parallel.successful_count == 2
        assert parallel.failed_count == 0

    def test_parallel_result_with_failures(self):
        """Test ParallelResult with some failures."""
        results = [
            TaskResult("t1", True, "r1", None, 100),
            TaskResult("t2", False, None, "error", 50),
        ]
        parallel = ParallelResult(
            results=results,
            total_time_ms=100,
            successful_count=1,
            failed_count=1,
        )

        assert parallel.successful_count == 1
        assert parallel.failed_count == 1


# Test: execute_parallel
class TestExecuteParallel:
    """Tests for execute_parallel function."""

    @pytest.mark.asyncio
    async def test_execute_parallel_success(self):
        """Test successful parallel execution."""
        tasks = {
            "task1": successful_task("a"),
            "task2": successful_task("b"),
            "task3": successful_task("c"),
        }

        result = await execute_parallel(tasks, max_concurrent=3)

        assert result.successful_count == 3
        assert result.failed_count == 0
        assert len(result.results) == 3

    @pytest.mark.asyncio
    async def test_execute_parallel_with_failures(self):
        """Test parallel execution with some failures."""
        tasks = {
            "task1": successful_task("a"),
            "task2": failing_task("b"),
            "task3": successful_task("c"),
        }

        result = await execute_parallel(tasks, max_concurrent=3)

        assert result.successful_count == 2
        assert result.failed_count == 1

    @pytest.mark.asyncio
    async def test_execute_parallel_concurrency_limit(self):
        """Test that concurrency limit is enforced."""
        # Track concurrent executions
        concurrent_count = 0
        max_concurrent_seen = 0

        async def tracking_task(value: str) -> str:
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return f"Result: {value}"

        tasks = {
            f"task{i}": tracking_task(str(i))
            for i in range(10)
        }

        result = await execute_parallel(tasks, max_concurrent=3)

        assert result.successful_count == 10
        assert max_concurrent_seen <= 3

    @pytest.mark.asyncio
    async def test_execute_parallel_empty_tasks(self):
        """Test parallel execution with empty task dict."""
        tasks = {}

        result = await execute_parallel(tasks, max_concurrent=3)

        assert result.successful_count == 0
        assert result.failed_count == 0
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_execute_parallel_single_task(self):
        """Test parallel execution with single task."""
        tasks = {
            "only_task": successful_task("single"),
        }

        result = await execute_parallel(tasks, max_concurrent=3)

        assert result.successful_count == 1
        assert result.results[0].result == "Result: single"


# Test: execute_with_fallback
class TestExecuteWithFallback:
    """Tests for execute_with_fallback function."""

    @pytest.mark.asyncio
    async def test_fallback_not_needed(self):
        """Test that fallback is not called when primary succeeds."""
        primary_called = False
        fallback_called = False

        async def primary():
            nonlocal primary_called
            primary_called = True
            return "primary result"

        async def fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback result"

        result = await execute_with_fallback(primary(), fallback())

        assert result == "primary result"
        assert primary_called is True
        assert fallback_called is False

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        """Test that fallback is called when primary fails."""
        async def primary():
            raise ValueError("Primary failed")

        async def fallback():
            return "fallback result"

        result = await execute_with_fallback(primary(), fallback())

        assert result == "fallback result"

    @pytest.mark.asyncio
    async def test_fallback_also_fails(self):
        """Test behavior when both primary and fallback fail."""
        async def primary():
            raise ValueError("Primary failed")

        async def fallback():
            raise RuntimeError("Fallback also failed")

        with pytest.raises(RuntimeError):
            await execute_with_fallback(primary(), fallback())


# Test: ParallelExecutor Class
class TestParallelExecutorClass:
    """Tests for ParallelExecutor class."""

    def test_executor_initialization(self, parallel_executor):
        """Test ParallelExecutor initialization."""
        assert parallel_executor.max_concurrent == 4
        assert parallel_executor._metrics == {}

    def test_executor_custom_concurrency(self):
        """Test ParallelExecutor with custom concurrency."""
        executor = ParallelExecutor(max_concurrent=8)
        assert executor.max_concurrent == 8

    @pytest.mark.asyncio
    async def test_executor_execute(self, parallel_executor):
        """Test ParallelExecutor execute method."""
        tasks = {
            "t1": successful_task("a"),
            "t2": successful_task("b"),
        }

        result = await parallel_executor.execute(tasks)

        assert result.successful_count == 2

    @pytest.mark.asyncio
    async def test_executor_tracks_metrics(self, parallel_executor):
        """Test that ParallelExecutor tracks execution metrics."""
        tasks = {
            "t1": successful_task("a"),
        }

        await parallel_executor.execute(tasks)

        # Should have updated metrics
        metrics = parallel_executor.get_metrics()
        assert metrics is not None

    @pytest.mark.asyncio
    async def test_executor_multiple_executions(self, parallel_executor):
        """Test multiple executions with same executor."""
        tasks1 = {"t1": successful_task("a")}
        tasks2 = {"t2": successful_task("b"), "t3": successful_task("c")}

        result1 = await parallel_executor.execute(tasks1)
        result2 = await parallel_executor.execute(tasks2)

        assert result1.successful_count == 1
        assert result2.successful_count == 2


# Test: Timeout Handling
class TestTimeoutHandling:
    """Tests for timeout handling in parallel execution."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that slow tasks are handled properly."""
        tasks = {
            "fast": successful_task("fast", delay=0.01),
            "slow": slow_task("slow", delay=0.5),
        }

        # With a reasonable timeout, both should complete
        result = await execute_parallel(tasks, max_concurrent=2)

        # Both tasks should complete (no timeout enforced in basic execute_parallel)
        assert len(result.results) == 2


# Test: Error Recovery
class TestErrorRecovery:
    """Tests for error recovery patterns."""

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self):
        """Test that successful tasks complete even when some fail."""
        tasks = {
            "success1": successful_task("a"),
            "fail1": failing_task("b"),
            "success2": successful_task("c"),
            "fail2": failing_task("d"),
        }

        result = await execute_parallel(tasks, max_concurrent=4)

        # Should have partial success
        assert result.successful_count == 2
        assert result.failed_count == 2

        # Verify we can identify which succeeded
        successful_ids = [r.task_id for r in result.results if r.success]
        assert "success1" in successful_ids
        assert "success2" in successful_ids

    @pytest.mark.asyncio
    async def test_all_tasks_fail(self):
        """Test handling when all tasks fail."""
        tasks = {
            "fail1": failing_task("a"),
            "fail2": failing_task("b"),
        }

        result = await execute_parallel(tasks, max_concurrent=2)

        assert result.successful_count == 0
        assert result.failed_count == 2


# Test: Performance Characteristics
class TestPerformanceCharacteristics:
    """Tests for performance characteristics."""

    @pytest.mark.asyncio
    async def test_parallel_is_faster_than_sequential(self):
        """Test that parallel execution is faster than sequential."""
        delay = 0.1
        num_tasks = 4

        tasks = {
            f"task{i}": successful_task(str(i), delay=delay)
            for i in range(num_tasks)
        }

        # Execute in parallel
        result = await execute_parallel(tasks, max_concurrent=num_tasks)

        # Total time should be close to single task time (parallel)
        # Not num_tasks * delay (sequential)
        assert result.total_time_ms < (num_tasks * delay * 1000 * 0.8)  # Allow 80% of sequential time

    @pytest.mark.asyncio
    async def test_execution_time_tracked(self):
        """Test that execution time is tracked for each task."""
        tasks = {
            "task1": successful_task("a", delay=0.05),
            "task2": successful_task("b", delay=0.1),
        }

        result = await execute_parallel(tasks, max_concurrent=2)

        for task_result in result.results:
            assert task_result.execution_time_ms > 0


# Integration Tests
class TestParallelExecutorIntegration:
    """Integration tests for parallel execution."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_realistic_agent_simulation(self, parallel_executor):
        """Test simulating multiple agent executions."""
        async def mock_hurricane_agent():
            await asyncio.sleep(0.1)
            return {"agent": "hurricane", "content": "Hurricane analysis"}

        async def mock_forecast_agent():
            await asyncio.sleep(0.08)
            return {"agent": "forecast", "content": "Weather forecast"}

        async def mock_historical_agent():
            await asyncio.sleep(0.12)
            return {"agent": "historical", "content": "Historical patterns"}

        tasks = {
            "hurricane": mock_hurricane_agent(),
            "forecast": mock_forecast_agent(),
            "historical": mock_historical_agent(),
        }

        result = await parallel_executor.execute(tasks)

        assert result.successful_count == 3
        assert result.failed_count == 0

        # Verify all results are present
        results_by_id = {r.task_id: r for r in result.results}
        assert "hurricane" in results_by_id
        assert "forecast" in results_by_id
        assert "historical" in results_by_id

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_high_concurrency_stress(self, parallel_executor):
        """Test high concurrency execution."""
        num_tasks = 50

        tasks = {
            f"task{i}": successful_task(str(i), delay=0.01)
            for i in range(num_tasks)
        }

        result = await execute_parallel(tasks, max_concurrent=10)

        assert result.successful_count == num_tasks
        assert result.failed_count == 0
