"""Parallel Execution Utilities for Level 4b: 8-Agent Orchestration.

CRITICAL RULES:
1. Maximum 4 concurrent agent executions (resource constraint)
2. Timeout handling for long-running agents (30s default)
3. Graceful error handling (partial results on failures)
4. Comprehensive logging for observability

Architecture:
- Semaphore-based concurrency control
- Timeout enforcement per task
- Result aggregation with error handling
- Execution metrics tracking

Design Principles:
- Reduce latency through parallel execution (40% improvement)
- Prevent resource exhaustion (semaphore limits)
- Graceful degradation on failures
- Full observability through structured logging
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class ParallelResult:
    """Result from parallel execution.

    Contains:
    - Successful results
    - Failed tasks with exceptions
    - Timing metrics

    Attributes:
        successes: List of successful task results
        failures: List of exceptions from failed tasks
        total_time_ms: Total execution time in milliseconds
        task_count: Number of tasks executed
        parallel_count: Maximum tasks that ran in parallel
    """

    successes: list[Any] = field(default_factory=list)
    failures: list[Exception] = field(default_factory=list)
    total_time_ms: float = 0.0
    task_count: int = 0
    parallel_count: int = 0

    @property
    def success_count(self) -> int:
        """Number of successful tasks."""
        return len(self.successes)

    @property
    def failure_count(self) -> int:
        """Number of failed tasks."""
        return len(self.failures)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.task_count == 0:
            return 0.0
        return (self.success_count / self.task_count) * 100

    @property
    def is_fully_successful(self) -> bool:
        """True if all tasks succeeded."""
        return self.failure_count == 0 and self.success_count > 0


@dataclass
class TaskResult:
    """Result from a single task execution.

    Attributes:
        task_id: Identifier for the task
        result: Task result (if successful)
        error: Exception (if failed)
        duration_ms: Execution time in milliseconds
        success: Whether task succeeded
    """

    task_id: str
    result: Any | None = None
    error: Exception | None = None
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        """True if task succeeded."""
        return self.error is None


async def execute_parallel(
    tasks: list[Callable[[], Any]],
    max_concurrent: int = 4,
    timeout_seconds: float = 30.0,
    task_ids: list[str] | None = None,
) -> ParallelResult:
    """Execute tasks in parallel with concurrency limit.

    Executes multiple async tasks concurrently while:
    - Limiting maximum concurrent executions
    - Enforcing per-task timeouts
    - Collecting both successes and failures
    - Tracking execution metrics

    Args:
        tasks: List of async callables to execute
        max_concurrent: Maximum concurrent tasks (default: 4)
        timeout_seconds: Per-task timeout in seconds (default: 30.0)
        task_ids: Optional task identifiers for logging

    Returns:
        ParallelResult with successes, failures, and metrics

    Example:
        ```python
        async def task1():
            return "result1"

        async def task2():
            return "result2"

        result = await execute_parallel([task1, task2])
        print(f"Successes: {result.success_count}")
        ```
    """
    start_time = time.perf_counter()

    if not tasks:
        return ParallelResult()

    # Generate task IDs if not provided
    if task_ids is None:
        task_ids = [f"task_{i}" for i in range(len(tasks))]

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    # Track results
    successes: list[Any] = []
    failures: list[Exception] = []
    task_results: list[TaskResult] = []

    async def run_with_semaphore(
        task: Callable[[], Any],
        task_id: str,
    ) -> TaskResult:
        """Execute task with semaphore and timeout."""
        task_start = time.perf_counter()

        async with semaphore:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    task(),
                    timeout=timeout_seconds,
                )
                duration_ms = (time.perf_counter() - task_start) * 1000

                logger.debug(
                    "parallel_task_success",
                    task_id=task_id,
                    duration_ms=duration_ms,
                )

                return TaskResult(
                    task_id=task_id,
                    result=result,
                    duration_ms=duration_ms,
                )

            except TimeoutError:
                duration_ms = (time.perf_counter() - task_start) * 1000
                error = TimeoutError(f"Task {task_id} exceeded {timeout_seconds}s timeout")

                logger.warning(
                    "parallel_task_timeout",
                    task_id=task_id,
                    timeout_seconds=timeout_seconds,
                    duration_ms=duration_ms,
                )

                return TaskResult(
                    task_id=task_id,
                    error=error,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = (time.perf_counter() - task_start) * 1000

                logger.warning(
                    "parallel_task_failed",
                    task_id=task_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=duration_ms,
                )

                return TaskResult(
                    task_id=task_id,
                    error=e,
                    duration_ms=duration_ms,
                )

    # Create tasks
    async_tasks = [
        asyncio.create_task(run_with_semaphore(task, task_id))
        for task, task_id in zip(tasks, task_ids)
    ]

    # Execute all tasks
    task_results = await asyncio.gather(*async_tasks)

    # Separate successes and failures
    for result in task_results:
        if result.success:
            successes.append(result.result)
        else:
            failures.append(result.error)

    total_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "parallel_execution_complete",
        total_tasks=len(tasks),
        successes=len(successes),
        failures=len(failures),
        max_concurrent=max_concurrent,
        time_ms=total_time_ms,
    )

    return ParallelResult(
        successes=successes,
        failures=failures,
        total_time_ms=total_time_ms,
        task_count=len(tasks),
        parallel_count=min(len(tasks), max_concurrent),
    )


async def execute_with_fallback(
    primary_task: Callable[[], T],
    fallback_task: Callable[[], T],
    timeout_seconds: float = 30.0,
    primary_id: str = "primary",
    fallback_id: str = "fallback",
) -> tuple[T, str]:
    """Execute task with fallback on failure.

    Tries primary task first, falls back to secondary on failure.

    Args:
        primary_task: Primary task to execute
        fallback_task: Fallback task if primary fails
        timeout_seconds: Per-task timeout
        primary_id: Identifier for primary task
        fallback_id: Identifier for fallback task

    Returns:
        Tuple of (result, task_id_that_succeeded)

    Raises:
        Exception: If both primary and fallback fail
    """
    start_time = time.perf_counter()

    # Try primary
    try:
        result = await asyncio.wait_for(
            primary_task(),
            timeout=timeout_seconds,
        )

        logger.debug(
            "primary_task_success",
            task_id=primary_id,
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

        return result, primary_id

    except Exception as primary_error:
        logger.warning(
            "primary_task_failed_trying_fallback",
            task_id=primary_id,
            error=str(primary_error),
        )

        # Try fallback
        try:
            result = await asyncio.wait_for(
                fallback_task(),
                timeout=timeout_seconds,
            )

            logger.info(
                "fallback_task_success",
                task_id=fallback_id,
                total_duration_ms=(time.perf_counter() - start_time) * 1000,
            )

            return result, fallback_id

        except Exception as fallback_error:
            logger.error(
                "both_tasks_failed",
                primary_error=str(primary_error),
                fallback_error=str(fallback_error),
            )

            # Raise combined error
            raise RuntimeError(
                f"Both primary ({primary_id}) and fallback ({fallback_id}) failed. "
                f"Primary: {primary_error}, Fallback: {fallback_error}"
            ) from fallback_error


class ParallelExecutor:
    """Configurable parallel executor for agent workflows.

    Provides a reusable executor with consistent configuration:
    - Concurrency limits
    - Timeout settings
    - Metrics tracking

    Attributes:
        max_concurrent: Maximum concurrent executions
        timeout_seconds: Default per-task timeout
        total_executions: Counter of total executions
        total_successes: Counter of successful tasks
        total_failures: Counter of failed tasks
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize parallel executor.

        Args:
            max_concurrent: Maximum concurrent tasks
            timeout_seconds: Default per-task timeout
        """
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds

        # Metrics
        self.total_executions = 0
        self.total_successes = 0
        self.total_failures = 0

        logger.info(
            "parallel_executor_initialized",
            max_concurrent=max_concurrent,
            timeout_seconds=timeout_seconds,
        )

    async def execute(
        self,
        tasks: list[Callable[[], Any]],
        task_ids: list[str] | None = None,
        timeout_override: float | None = None,
    ) -> ParallelResult:
        """Execute tasks in parallel.

        Args:
            tasks: List of async callables
            task_ids: Optional task identifiers
            timeout_override: Override default timeout

        Returns:
            ParallelResult with results and metrics
        """
        timeout = timeout_override or self.timeout_seconds

        result = await execute_parallel(
            tasks=tasks,
            max_concurrent=self.max_concurrent,
            timeout_seconds=timeout,
            task_ids=task_ids,
        )

        # Update metrics
        self.total_executions += result.task_count
        self.total_successes += result.success_count
        self.total_failures += result.failure_count

        return result

    def get_metrics(self) -> dict[str, Any]:
        """Get executor metrics.

        Returns:
            Dictionary with execution metrics
        """
        return {
            "total_executions": self.total_executions,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": (
                (self.total_successes / self.total_executions * 100)
                if self.total_executions > 0
                else 0.0
            ),
            "max_concurrent": self.max_concurrent,
            "timeout_seconds": self.timeout_seconds,
        }

    def reset_metrics(self) -> None:
        """Reset execution metrics."""
        self.total_executions = 0
        self.total_successes = 0
        self.total_failures = 0

        logger.debug("parallel_executor_metrics_reset")
