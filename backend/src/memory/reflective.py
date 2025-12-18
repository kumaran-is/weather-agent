"""Reflective memory backend for Level 3c (Layer 7): Self-improvement.

This module implements Layer 7 (Reflective Memory) which enables meta-cognitive
self-improvement through reflection on past experiences. It tracks:
- What went wrong and why (error analysis)
- What could be improved (insights)
- How improvements were applied (action tracking)
- Whether improvements worked (effectiveness)

Storage: PostgreSQL (reflections table)
Retention: Indefinite (critical for long-term improvement)

Key Capabilities:
1. **Error Pattern Detection**: Identifies recurring mistakes
2. **Insight Generation**: Extracts learnings from experiences
3. **Improvement Tracking**: Monitors application of learnings
4. **Effectiveness Measurement**: Validates improvements work
5. **Meta-Learning**: Learns from learning process itself

Reflection Triggers:
- user_feedback_negative: User explicitly reports dissatisfaction
- error_pattern: Same error occurs 3+ times
- quality_degradation: Success rate drops below threshold
- novel_situation: Encountered unexpected scenario
- goal_misalignment: Action didn't achieve intended goal

Architecture:
```
Event → ReflectiveMemory.trigger_reflection() → Analysis
                                                    ↓
                                                 Insight
                                                    ↓
                                            Store in PostgreSQL
                                                    ↓
                                              Apply Learning
                                                    ↓
                                          Track Effectiveness
```

Example:
    >>> from backend.src.memory.reflective import ReflectiveMemoryBackend
    >>> memory = ReflectiveMemoryBackend()
    >>>
    >>> # Trigger reflection on error
    >>> reflection = await memory.create_reflection(
    ...     trigger="error_pattern",
    ...     context="Failed to parse hurricane data 3 times",
    ...     analysis="Data format changed but parser wasn't updated",
    ...     insight="Implement robust schema validation with fallback parsing",
    ...     learned_from="episode_456"
    ... )
    >>>
    >>> # Mark reflection as applied
    >>> await memory.mark_applied(reflection.reflection_id, action_taken="Added schema validator")
    >>>
    >>> # Track effectiveness
    >>> await memory.update_effectiveness(reflection.reflection_id, improvement_score=0.95)
    >>> print("Improvement successful!")
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, Field

from backend.config.settings import Settings
from backend.src.models.memory import ReflectiveMemory

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================


class ReflectionInsight(BaseModel):
    """Single reflection insight with context."""

    reflection_id: UUID
    trigger: str
    insight: str
    context: str
    effectiveness: float = Field(..., ge=0.0, le=1.0)
    applied: bool
    timestamp: datetime

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "reflection_id": "550e8400-e29b-41d4-a716-446655440000",
                    "trigger": "error_pattern",
                    "insight": "Implement robust schema validation with fallback parsing",
                    "context": "Failed to parse hurricane data 3 times",
                    "effectiveness": 0.95,
                    "applied": True,
                    "timestamp": "2025-12-07T22:00:00Z",
                }
            ]
        }
    }


class ReflectionSummary(BaseModel):
    """Summary of reflections for a trigger type."""

    trigger: str
    total_reflections: int = Field(..., ge=0)
    applied_reflections: int = Field(..., ge=0)
    avg_effectiveness: float = Field(..., ge=0.0, le=1.0)
    avg_improvement: float = Field(..., ge=0.0, le=1.0)
    most_recent: datetime | None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trigger": "error_pattern",
                    "total_reflections": 15,
                    "applied_reflections": 12,
                    "avg_effectiveness": 0.85,
                    "avg_improvement": 0.75,
                    "most_recent": "2025-12-07T22:00:00Z",
                }
            ]
        }
    }


# ============================================================================
# Reflective Memory Backend
# ============================================================================


class ReflectiveMemoryBackend:
    """PostgreSQL backend for reflective memory (Layer 7).

    This class handles storage and retrieval of self-improvement reflections,
    enabling meta-cognitive learning from experience.

    Attributes:
        settings: Application settings
        pool: PostgreSQL connection pool
    """

    def __init__(
        self, settings: Settings | None = None, pool: asyncpg.Pool | None = None
    ) -> None:
        """Initialize reflective memory backend.

        Args:
            settings: Application settings (auto-loaded if None)
            pool: PostgreSQL connection pool (optional, for testing)
        """
        self.settings = settings or Settings()
        self.pool = pool
        self._initialized = False

        logger.info("Initialized ReflectiveMemoryBackend")

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
    # Reflection Creation
    # ========================================================================

    async def create_reflection(
        self,
        trigger: str,
        context: str,
        analysis: str,
        insight: str,
        learned_from: str | None = None,
    ) -> ReflectiveMemory:
        """Create a new reflection from an experience.

        Args:
            trigger: Reflection trigger (e.g., "error_pattern", "user_feedback_negative")
            context: Situation that led to this learning
            analysis: Analysis of what went wrong
            insight: Key learning or takeaway
            learned_from: Optional episode ID or pattern description

        Returns:
            ReflectiveMemory object

        Example:
            >>> reflection = await memory.create_reflection(
            ...     trigger="error_pattern",
            ...     context="Failed to parse hurricane data 3 times",
            ...     analysis="Data format changed but parser wasn't updated",
            ...     insight="Implement robust schema validation with fallback parsing",
            ...     learned_from="episode_456"
            ... )
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        await self.initialize()

        reflection_id = uuid4()
        timestamp = datetime.now()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO reflections (
                    reflection_id, timestamp, trigger, context,
                    analysis, insight, learned_from
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                reflection_id,
                timestamp,
                trigger,
                context,
                analysis,
                insight,
                learned_from,
            )

        logger.info(f"Created reflection: trigger={trigger}, id={reflection_id}")

        return ReflectiveMemory(
            reflection_id=str(reflection_id),
            timestamp=timestamp,
            trigger=trigger,
            context=context,
            analysis=analysis,
            insight=insight,
            learned_from=learned_from,
            action_taken=None,
            applied_count=0,
            effectiveness=0.0,
            improvement_score=None,
            applied=False,
        )

    # ========================================================================
    # Reflection Application
    # ========================================================================

    async def mark_applied(
        self, reflection_id: str | UUID, action_taken: str
    ) -> None:
        """Mark a reflection as applied with the action taken.

        Args:
            reflection_id: Reflection identifier
            action_taken: Description of action taken

        Example:
            >>> await memory.mark_applied(
            ...     reflection_id="550e8400-e29b-41d4-a716-446655440000",
            ...     action_taken="Added schema validator with fallback parsing"
            ... )
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        await self.initialize()

        if isinstance(reflection_id, str):
            reflection_id = UUID(reflection_id)

        async with self.pool.acquire() as conn:
            # Increment applied_count
            await conn.execute(
                """
                UPDATE reflections
                SET applied = TRUE,
                    action_taken = $1,
                    applied_count = applied_count + 1
                WHERE reflection_id = $2
                """,
                action_taken,
                reflection_id,
            )

        logger.info(f"Marked reflection as applied: {reflection_id}")

    async def update_effectiveness(
        self,
        reflection_id: str | UUID,
        improvement_score: float,
        effectiveness: float | None = None,
    ) -> None:
        """Update effectiveness and improvement score for a reflection.

        Called after applying a reflection to measure its impact.

        Args:
            reflection_id: Reflection identifier
            improvement_score: Measured improvement (0.0-1.0)
            effectiveness: Optional effectiveness override (auto-calculated if None)

        Example:
            >>> await memory.update_effectiveness(
            ...     reflection_id="550e8400-e29b-41d4-a716-446655440000",
            ...     improvement_score=0.95  # 95% improvement
            ... )
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        await self.initialize()

        if isinstance(reflection_id, str):
            reflection_id = UUID(reflection_id)

        # Auto-calculate effectiveness if not provided
        if effectiveness is None:
            # Effectiveness = weighted average of improvement_score and applied_count
            # More applications = higher confidence
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT applied_count FROM reflections WHERE reflection_id = $1",
                    reflection_id,
                )
                applied_count = row["applied_count"] if row else 1

            # Confidence factor (caps at 1.0 after 10 applications)
            confidence = min(1.0, applied_count / 10.0)
            effectiveness = improvement_score * confidence

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE reflections
                SET improvement_score = $1,
                    effectiveness = $2
                WHERE reflection_id = $3
                """,
                improvement_score,
                effectiveness,
                reflection_id,
            )

        logger.info(
            f"Updated effectiveness: {reflection_id} (improvement={improvement_score:.2f}, effectiveness={effectiveness:.2f})"
        )

    # ========================================================================
    # Reflection Retrieval
    # ========================================================================

    async def get_reflection(self, reflection_id: str | UUID) -> ReflectiveMemory | None:
        """Get a specific reflection by ID.

        Args:
            reflection_id: Reflection identifier

        Returns:
            ReflectiveMemory object or None if not found
        """
        if not self.pool:
            return None

        await self.initialize()

        if isinstance(reflection_id, str):
            reflection_id = UUID(reflection_id)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM reflections WHERE reflection_id = $1", reflection_id
            )

        if not row:
            return None

        return ReflectiveMemory(
            reflection_id=str(row["reflection_id"]),
            timestamp=row["timestamp"],
            trigger=row["trigger"],
            context=row["context"],
            analysis=row["analysis"],
            insight=row["insight"],
            learned_from=row["learned_from"],
            action_taken=row["action_taken"],
            applied_count=row["applied_count"],
            effectiveness=row["effectiveness"],
            improvement_score=row["improvement_score"],
            applied=row["applied"],
        )

    async def get_reflections_by_trigger(
        self, trigger: str, limit: int = 10, only_applied: bool = False
    ) -> list[ReflectiveMemory]:
        """Get reflections for a specific trigger type.

        Args:
            trigger: Trigger type (e.g., "error_pattern")
            limit: Maximum number of reflections to return
            only_applied: Only return applied reflections

        Returns:
            List of ReflectiveMemory objects (most recent first)

        Example:
            >>> reflections = await memory.get_reflections_by_trigger("error_pattern")
            >>> for r in reflections:
            ...     print(f"{r.insight} (effectiveness={r.effectiveness:.1%})")
        """
        if not self.pool:
            return []

        await self.initialize()

        query = """
            SELECT * FROM reflections
            WHERE trigger = $1
        """
        if only_applied:
            query += " AND applied = TRUE"
        query += " ORDER BY timestamp DESC LIMIT $2"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, trigger, limit)

        return [
            ReflectiveMemory(
                reflection_id=str(row["reflection_id"]),
                timestamp=row["timestamp"],
                trigger=row["trigger"],
                context=row["context"],
                analysis=row["analysis"],
                insight=row["insight"],
                learned_from=row["learned_from"],
                action_taken=row["action_taken"],
                applied_count=row["applied_count"],
                effectiveness=row["effectiveness"],
                improvement_score=row["improvement_score"],
                applied=row["applied"],
            )
            for row in rows
        ]

    async def get_top_insights(
        self, min_effectiveness: float = 0.7, limit: int = 10
    ) -> list[ReflectionInsight]:
        """Get top insights by effectiveness.

        Args:
            min_effectiveness: Minimum effectiveness threshold
            limit: Maximum number of insights to return

        Returns:
            List of ReflectionInsight objects

        Example:
            >>> insights = await memory.get_top_insights(min_effectiveness=0.8)
            >>> for insight in insights:
            ...     print(f"{insight.insight} (effectiveness={insight.effectiveness:.1%})")
        """
        if not self.pool:
            return []

        await self.initialize()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT reflection_id, trigger, insight, context,
                       effectiveness, applied, timestamp
                FROM reflections
                WHERE effectiveness >= $1 AND applied = TRUE
                ORDER BY effectiveness DESC
                LIMIT $2
                """,
                min_effectiveness,
                limit,
            )

        return [
            ReflectionInsight(
                reflection_id=row["reflection_id"],
                trigger=row["trigger"],
                insight=row["insight"],
                context=row["context"],
                effectiveness=row["effectiveness"],
                applied=row["applied"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    # ========================================================================
    # Analytics
    # ========================================================================

    async def get_reflection_summary(self) -> list[ReflectionSummary]:
        """Get summary of reflections by trigger type.

        Returns:
            List of ReflectionSummary objects

        Example:
            >>> summaries = await memory.get_reflection_summary()
            >>> for s in summaries:
            ...     print(f"{s.trigger}: {s.applied_reflections}/{s.total_reflections} applied")
        """
        if not self.pool:
            return []

        await self.initialize()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    trigger,
                    COUNT(*) AS total_reflections,
                    COUNT(*) FILTER (WHERE applied = TRUE) AS applied_reflections,
                    AVG(effectiveness) AS avg_effectiveness,
                    AVG(improvement_score) AS avg_improvement,
                    MAX(timestamp) AS most_recent
                FROM reflections
                GROUP BY trigger
                ORDER BY total_reflections DESC
                """
            )

        return [
            ReflectionSummary(
                trigger=row["trigger"],
                total_reflections=row["total_reflections"],
                applied_reflections=row["applied_reflections"],
                avg_effectiveness=row["avg_effectiveness"] or 0.0,
                avg_improvement=row["avg_improvement"] or 0.0,
                most_recent=row["most_recent"],
            )
            for row in rows
        ]

    async def search_insights(
        self, query: str, limit: int = 10
    ) -> list[ReflectionInsight]:
        """Full-text search for insights.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of ReflectionInsight objects

        Example:
            >>> insights = await memory.search_insights("schema validation")
            >>> for insight in insights:
            ...     print(insight.insight)
        """
        if not self.pool:
            return []

        await self.initialize()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT reflection_id, trigger, insight, context,
                       effectiveness, applied, timestamp
                FROM reflections
                WHERE to_tsvector('english', insight) @@ plainto_tsquery('english', $1)
                   OR to_tsvector('english', analysis) @@ plainto_tsquery('english', $1)
                ORDER BY effectiveness DESC
                LIMIT $2
                """,
                query,
                limit,
            )

        return [
            ReflectionInsight(
                reflection_id=row["reflection_id"],
                trigger=row["trigger"],
                insight=row["insight"],
                context=row["context"],
                effectiveness=row["effectiveness"],
                applied=row["applied"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    # ========================================================================
    # Meta-Learning
    # ========================================================================

    async def detect_reflection_patterns(self) -> dict[str, Any]:
        """Detect patterns in reflections (meta-learning).

        Analyzes:
        - Which triggers lead to most effective improvements
        - Which insights get applied most often
        - Which reflection types have highest success rate

        Returns:
            Dictionary with pattern analysis

        Example:
            >>> patterns = await memory.detect_reflection_patterns()
            >>> print(f"Most effective trigger: {patterns['best_trigger']}")
        """
        if not self.pool:
            return {}

        await self.initialize()

        # Get summary data
        summaries = await self.get_reflection_summary()

        if not summaries:
            return {"status": "no_data"}

        # Find best trigger (highest avg effectiveness)
        best_trigger = max(summaries, key=lambda s: s.avg_effectiveness)

        # Find most actionable trigger (highest application rate)
        most_actionable = max(
            summaries,
            key=lambda s: s.applied_reflections / max(1, s.total_reflections),
        )

        return {
            "status": "success",
            "best_trigger": {
                "name": best_trigger.trigger,
                "avg_effectiveness": best_trigger.avg_effectiveness,
                "total_reflections": best_trigger.total_reflections,
            },
            "most_actionable_trigger": {
                "name": most_actionable.trigger,
                "application_rate": most_actionable.applied_reflections
                / max(1, most_actionable.total_reflections),
                "total_reflections": most_actionable.total_reflections,
            },
            "total_triggers": len(summaries),
            "overall_application_rate": sum(s.applied_reflections for s in summaries)
            / max(1, sum(s.total_reflections for s in summaries)),
        }


# ============================================================================
# Helper Functions
# ============================================================================


async def initialize_reflective_memory() -> ReflectiveMemoryBackend:
    """Initialize and return reflective memory backend.

    Returns:
        Initialized ReflectiveMemoryBackend

    Example:
        >>> memory = await initialize_reflective_memory()
        >>> reflection = await memory.create_reflection(
        ...     trigger="error_pattern",
        ...     context="Recurring parse error",
        ...     analysis="Schema validation missing",
        ...     insight="Add schema validation layer"
        ... )
    """
    backend = ReflectiveMemoryBackend()
    await backend.initialize()
    return backend
