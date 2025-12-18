"""Procedural memory backend for Level 3c (Layer 5): Workflow optimization.

This module implements Layer 5 (Procedural Memory) which learns and optimizes
workflows based on historical performance. It tracks:
- Which tool sequences work best for specific tasks
- Success rates and response times for workflows
- Tool usage patterns and optimization opportunities

Storage: PostgreSQL (workflows, tool_usage tables)
Retention: Indefinite (cleaned by consolidation pipeline based on usage)

Key Capabilities:
1. **Workflow Learning**: Automatically learns successful tool sequences
2. **Intelligent Tool Selection**: Recommends optimal tools based on history
3. **Performance Analytics**: Tracks success rates and response times
4. **Workflow Optimization**: Identifies bottlenecks and suggests improvements

Architecture:
```
Agent Tool Call → ProceduralMemory.record_tool_usage() → PostgreSQL
                                                              ↓
Agent Next Tool → ProceduralMemory.recommend_workflow() ← Analytics
```

Example:
    >>> from backend.src.memory.procedural import ProceduralMemoryBackend
    >>> memory = ProceduralMemoryBackend()
    >>>
    >>> # Record tool usage
    >>> await memory.record_tool_usage(
    ...     tool_name="get_weather_forecast",
    ...     task_context="forecast for Miami",
    ...     execution_time=0.5,
    ...     success=True
    ... )
    >>>
    >>> # Learn workflow
    >>> await memory.learn_workflow(
    ...     task_name="hurricane_forecast",
    ...     steps=["get_current_hurricanes", "get_forecast", "analyze_safety"],
    ...     success=True,
    ...     response_time=1.2
    ... )
    >>>
    >>> # Get recommended workflow
    >>> workflow = await memory.recommend_workflow("hurricane_forecast")
    >>> print(workflow.steps)
    ['get_current_hurricanes', 'get_forecast', 'analyze_safety']
    >>> print(f"Expected success rate: {workflow.success_rate:.1%}")
    Expected success rate: 95.0%
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any

import asyncpg
from pydantic import BaseModel, Field

from backend.config.settings import Settings
from backend.src.models.memory import ProceduralMemory

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================


class WorkflowRecommendation(BaseModel):
    """Recommended workflow for a task."""

    task_name: str
    steps: list[str] = Field(..., description="Ordered list of tool names")
    success_rate: float = Field(..., description="Expected success rate", ge=0.0, le=1.0)
    avg_response_time: float = Field(
        ..., description="Expected response time in seconds", ge=0.0
    )
    usage_count: int = Field(
        ..., description="Number of times this workflow was used", ge=0
    )
    confidence: float = Field(
        ..., description="Recommendation confidence (based on usage_count)", ge=0.0, le=1.0
    )
    alternatives: list[dict[str, Any]] = Field(
        default_factory=list, description="Alternative workflows"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task_name": "hurricane_forecast",
                    "steps": [
                        "get_current_hurricanes",
                        "get_forecast",
                        "analyze_safety",
                    ],
                    "success_rate": 0.95,
                    "avg_response_time": 1.2,
                    "usage_count": 42,
                    "confidence": 0.9,
                    "alternatives": [
                        {
                            "steps": ["get_current_hurricanes", "analyze_safety"],
                            "success_rate": 0.88,
                        }
                    ],
                }
            ]
        }
    }


class ToolPerformance(BaseModel):
    """Performance metrics for a specific tool."""

    tool_name: str
    total_invocations: int = Field(..., ge=0)
    successful_invocations: int = Field(..., ge=0)
    failed_invocations: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    avg_execution_time: float = Field(..., ge=0.0)
    last_used: datetime

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tool_name": "get_weather_forecast",
                    "total_invocations": 150,
                    "successful_invocations": 145,
                    "failed_invocations": 5,
                    "success_rate": 0.967,
                    "avg_execution_time": 0.45,
                    "last_used": "2025-12-07T22:00:00Z",
                }
            ]
        }
    }


# ============================================================================
# Procedural Memory Backend
# ============================================================================


class ProceduralMemoryBackend:
    """PostgreSQL backend for procedural memory (Layer 5).

    This class handles storage and retrieval of workflow patterns and
    tool usage data for intelligent workflow optimization.

    Attributes:
        settings: Application settings
        pool: PostgreSQL connection pool
    """

    def __init__(
        self, settings: Settings | None = None, pool: asyncpg.Pool | None = None
    ) -> None:
        """Initialize procedural memory backend.

        Args:
            settings: Application settings (auto-loaded if None)
            pool: PostgreSQL connection pool (optional, for testing)
        """
        self.settings = settings or Settings()
        self.pool = pool
        self._initialized = False

        logger.info("Initialized ProceduralMemoryBackend")

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool."""
        if self._initialized:
            return

        if not self.pool and self.settings.POSTGRES_URL:
            self.pool = await asyncpg.create_pool(
                str(self.settings.POSTGRES_URL),
                min_size=2,
                max_size=self.settings.POSTGRES_MAX_CONNECTIONS,
            )
            logger.info(
                f"Created PostgreSQL pool (max_size={self.settings.POSTGRES_MAX_CONNECTIONS})"
            )

        self._initialized = True

    async def close(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Closed PostgreSQL pool")

    # ========================================================================
    # Tool Usage Tracking
    # ========================================================================

    async def record_tool_usage(
        self,
        tool_name: str,
        task_context: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        execution_time: float = 0.0,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Record individual tool invocation for analytics.

        Args:
            tool_name: Name of the tool used
            task_context: Optional task description
            user_id: Optional user identifier
            session_id: Optional session identifier
            execution_time: Execution time in seconds
            success: Whether the tool call succeeded
            error_message: Optional error message if failed

        Example:
            >>> await memory.record_tool_usage(
            ...     tool_name="get_weather_forecast",
            ...     task_context="forecast for Miami",
            ...     execution_time=0.5,
            ...     success=True
            ... )
        """
        if not self.pool:
            logger.warning("PostgreSQL pool not initialized, skipping tool usage recording")
            return

        await self.initialize()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tool_usage (
                    tool_name, task_context, user_id, session_id,
                    execution_time, success, error_message
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                tool_name,
                task_context,
                user_id,
                session_id,
                execution_time,
                success,
                error_message,
            )

        logger.debug(
            f"Recorded tool usage: {tool_name} (success={success}, time={execution_time:.2f}s)"
        )

    async def get_tool_performance(
        self, tool_name: str, days: int = 7
    ) -> ToolPerformance | None:
        """Get performance metrics for a specific tool.

        Args:
            tool_name: Tool name
            days: Number of days to analyze (default: 7)

        Returns:
            ToolPerformance object or None if no data

        Example:
            >>> perf = await memory.get_tool_performance("get_weather_forecast")
            >>> print(f"Success rate: {perf.success_rate:.1%}")
            Success rate: 96.7%
        """
        if not self.pool:
            return None

        await self.initialize()

        cutoff_time = datetime.now() - timedelta(days=days)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    tool_name,
                    COUNT(*) AS total_invocations,
                    COUNT(*) FILTER (WHERE success = TRUE) AS successful_invocations,
                    COUNT(*) FILTER (WHERE success = FALSE) AS failed_invocations,
                    AVG(execution_time) AS avg_execution_time,
                    MAX(timestamp) AS last_used
                FROM tool_usage
                WHERE tool_name = $1 AND timestamp > $2
                GROUP BY tool_name
                """,
                tool_name,
                cutoff_time,
            )

        if not row:
            return None

        success_rate = (
            row["successful_invocations"] / row["total_invocations"]
            if row["total_invocations"] > 0
            else 0.0
        )

        return ToolPerformance(
            tool_name=row["tool_name"],
            total_invocations=row["total_invocations"],
            successful_invocations=row["successful_invocations"],
            failed_invocations=row["failed_invocations"],
            success_rate=success_rate,
            avg_execution_time=row["avg_execution_time"] or 0.0,
            last_used=row["last_used"],
        )

    # ========================================================================
    # Workflow Learning
    # ========================================================================

    async def learn_workflow(
        self,
        task_name: str,
        steps: list[str],
        success: bool,
        response_time: float,
        feedback_score: float | None = None,
    ) -> ProceduralMemory:
        """Learn or update a workflow pattern.

        If workflow already exists, updates success_rate and avg_response_time
        using running average. If new, creates new workflow.

        Args:
            task_name: Task name (e.g., "hurricane_forecast")
            steps: Ordered list of tool names
            success: Whether this execution succeeded
            response_time: Total response time in seconds
            feedback_score: Optional user feedback (0.0-5.0)

        Returns:
            ProceduralMemory object

        Example:
            >>> workflow = await memory.learn_workflow(
            ...     task_name="hurricane_forecast",
            ...     steps=["get_current_hurricanes", "get_forecast", "analyze_safety"],
            ...     success=True,
            ...     response_time=1.2
            ... )
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        await self.initialize()

        # Calculate workflow hash for deduplication
        workflow_hash = self._hash_workflow(steps)

        async with self.pool.acquire() as conn:
            # Check if workflow exists
            existing = await conn.fetchrow(
                """
                SELECT * FROM workflows
                WHERE workflow_hash = $1
                """,
                workflow_hash,
            )

            if existing:
                # Update existing workflow with running average
                new_usage_count = existing["usage_count"] + 1
                new_success_rate = (
                    existing["success_rate"] * existing["usage_count"]
                    + (1.0 if success else 0.0)
                ) / new_usage_count

                new_avg_response_time = (
                    existing["avg_response_time"] * existing["usage_count"]
                    + response_time
                ) / new_usage_count

                # Update feedback if provided
                if feedback_score is not None:
                    if existing["feedback_score"] is not None:
                        new_feedback_score = (
                            existing["feedback_score"] * (new_usage_count - 1)
                            + feedback_score
                        ) / new_usage_count
                    else:
                        new_feedback_score = feedback_score
                else:
                    new_feedback_score = existing["feedback_score"]

                await conn.execute(
                    """
                    UPDATE workflows
                    SET usage_count = $1,
                        success_rate = $2,
                        avg_response_time = $3,
                        feedback_score = $4,
                        last_used = NOW()
                    WHERE workflow_hash = $5
                    """,
                    new_usage_count,
                    new_success_rate,
                    new_avg_response_time,
                    new_feedback_score,
                    workflow_hash,
                )

                logger.info(
                    f"Updated workflow: {task_name} (usage={new_usage_count}, success={new_success_rate:.1%})"
                )

                return ProceduralMemory(
                    task_name=task_name,
                    steps=steps,
                    success_rate=new_success_rate,
                    avg_response_time=new_avg_response_time,
                    usage_count=new_usage_count,
                    last_used=datetime.now(),
                    feedback_score=new_feedback_score,
                )

            else:
                # Insert new workflow
                await conn.execute(
                    """
                    INSERT INTO workflows (
                        task_name, workflow_hash, steps,
                        success_rate, avg_response_time, usage_count,
                        feedback_score
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    task_name,
                    workflow_hash,
                    steps,
                    1.0 if success else 0.0,
                    response_time,
                    1,
                    feedback_score,
                )

                logger.info(f"Learned new workflow: {task_name} ({len(steps)} steps)")

                return ProceduralMemory(
                    task_name=task_name,
                    steps=steps,
                    success_rate=1.0 if success else 0.0,
                    avg_response_time=response_time,
                    usage_count=1,
                    last_used=datetime.now(),
                    feedback_score=feedback_score,
                )

    def _hash_workflow(self, steps: list[str]) -> str:
        """Calculate SHA256 hash of workflow steps for deduplication.

        Args:
            steps: List of tool names

        Returns:
            SHA256 hex digest
        """
        workflow_str = ",".join(steps)
        return hashlib.sha256(workflow_str.encode()).hexdigest()

    # ========================================================================
    # Workflow Recommendation
    # ========================================================================

    async def recommend_workflow(
        self, task_name: str, min_usage_count: int = 3
    ) -> WorkflowRecommendation | None:
        """Recommend best workflow for a task based on historical performance.

        Args:
            task_name: Task name
            min_usage_count: Minimum usage count to consider (default: 3)

        Returns:
            WorkflowRecommendation or None if no suitable workflows

        Example:
            >>> rec = await memory.recommend_workflow("hurricane_forecast")
            >>> if rec.confidence > 0.8:
            ...     print(f"High confidence recommendation: {rec.steps}")
        """
        if not self.pool:
            return None

        await self.initialize()

        async with self.pool.acquire() as conn:
            # Get best workflow by success_rate, then avg_response_time
            rows = await conn.fetch(
                """
                SELECT * FROM workflows
                WHERE task_name = $1 AND usage_count >= $2
                ORDER BY success_rate DESC, avg_response_time ASC
                LIMIT 5
                """,
                task_name,
                min_usage_count,
            )

        if not rows:
            return None

        # Best workflow
        best = rows[0]

        # Calculate confidence (based on usage_count)
        confidence = min(1.0, best["usage_count"] / 50.0)  # Max confidence at 50 uses

        # Alternatives
        alternatives = [
            {
                "steps": row["steps"],
                "success_rate": row["success_rate"],
                "avg_response_time": row["avg_response_time"],
            }
            for row in rows[1:]
        ]

        return WorkflowRecommendation(
            task_name=task_name,
            steps=best["steps"],
            success_rate=best["success_rate"],
            avg_response_time=best["avg_response_time"],
            usage_count=best["usage_count"],
            confidence=confidence,
            alternatives=alternatives,
        )

    # ========================================================================
    # Analytics
    # ========================================================================

    async def get_top_workflows(self, limit: int = 10) -> list[ProceduralMemory]:
        """Get top workflows by usage count.

        Args:
            limit: Maximum number of workflows to return

        Returns:
            List of ProceduralMemory objects

        Example:
            >>> workflows = await memory.get_top_workflows(limit=5)
            >>> for w in workflows:
            ...     print(f"{w.task_name}: {w.usage_count} uses, {w.success_rate:.1%} success")
        """
        if not self.pool:
            return []

        await self.initialize()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM workflows
                ORDER BY usage_count DESC
                LIMIT $1
                """,
                limit,
            )

        return [
            ProceduralMemory(
                task_name=row["task_name"],
                steps=row["steps"],
                success_rate=row["success_rate"],
                avg_response_time=row["avg_response_time"],
                usage_count=row["usage_count"],
                last_used=row["last_used"],
                feedback_score=row["feedback_score"],
            )
            for row in rows
        ]

    async def cleanup_low_usage_workflows(self, min_usage: int = 3, days_old: int = 90) -> int:
        """Clean up workflows with low usage and old last_used date.

        Args:
            min_usage: Minimum usage count threshold (default: 3)
            days_old: Minimum days since last use (default: 90)

        Returns:
            Number of workflows deleted

        Example:
            >>> deleted = await memory.cleanup_low_usage_workflows()
            >>> print(f"Deleted {deleted} low-usage workflows")
        """
        if not self.pool:
            return 0

        await self.initialize()

        cutoff_date = datetime.now() - timedelta(days=days_old)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM workflows
                WHERE usage_count < $1 AND last_used < $2
                """,
                min_usage,
                cutoff_date,
            )

        # Parse result (format: "DELETE N")
        deleted_count = int(result.split()[-1]) if result else 0

        logger.info(
            f"Cleaned up {deleted_count} workflows (usage<{min_usage}, last_used>{days_old}d ago)"
        )

        return deleted_count


# ============================================================================
# Helper Functions
# ============================================================================


async def initialize_procedural_memory() -> ProceduralMemoryBackend:
    """Initialize and return procedural memory backend.

    Returns:
        Initialized ProceduralMemoryBackend

    Example:
        >>> memory = await initialize_procedural_memory()
        >>> await memory.record_tool_usage("get_weather", execution_time=0.5)
    """
    backend = ProceduralMemoryBackend()
    await backend.initialize()
    return backend
