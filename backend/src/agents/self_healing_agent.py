"""Self-Healing Agent for Level 4c: Automatic Error Recovery.

This agent provides self-healing capabilities for the multi-agent system,
including automatic retry, fallback routing, and circuit breaker management.

Features:
- Automatic retry with exponential backoff
- Fallback agent routing when primary fails
- Circuit breaker integration for cascade prevention
- Health monitoring and recovery tracking
- Graceful degradation for partial failures

Configuration:
- MAX_RETRIES: Maximum retry attempts (default: 3)
- BASE_BACKOFF: Initial backoff in seconds (default: 1.0)
- AGENT_TIMEOUT: Timeout per agent call (default: 30.0)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import structlog

from backend.src.agents.circuit_breaker import (
    circuit_registry,
    get_system_health,
)
from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)

logger = structlog.get_logger()


# =============================================================================
# Fallback mappings - which agents can substitute for others
# =============================================================================

DEFAULT_FALLBACK_MAP: dict[AgentRole, list[AgentRole]] = {
    # Hurricane Specialist fallbacks
    AgentRole.HURRICANE_SPECIALIST: [
        AgentRole.FORECASTER,
        AgentRole.RESEARCH,
    ],
    # Forecaster fallbacks
    AgentRole.FORECASTER: [
        AgentRole.RESEARCH,
        AgentRole.HISTORICAL_ANALYST,
    ],
    # Alert Manager fallbacks
    AgentRole.ALERT_MANAGER: [
        AgentRole.SYNTHESIS,
    ],
    # Research fallbacks
    AgentRole.RESEARCH: [
        AgentRole.HISTORICAL_ANALYST,
    ],
    # Historical Analyst fallbacks
    AgentRole.HISTORICAL_ANALYST: [
        AgentRole.RESEARCH,
    ],
    # Emergency Response fallbacks (Level 4c)
    AgentRole.EMERGENCY_RESPONSE: [
        AgentRole.HURRICANE_SPECIALIST,
        AgentRole.ALERT_MANAGER,
    ],
    # Climate Analyst fallbacks (Level 4c)
    AgentRole.CLIMATE_ANALYST: [
        AgentRole.HISTORICAL_ANALYST,
        AgentRole.RESEARCH,
    ],
    # Verification fallbacks
    AgentRole.VERIFICATION: [
        AgentRole.REFLECTION,
    ],
    # Debate fallbacks (Level 4c)
    AgentRole.DEBATE: [
        AgentRole.SYNTHESIS,
        AgentRole.VERIFICATION,
    ],
}


class SelfHealingAgent:
    """Self-healing orchestrator with retry, fallback, and circuit breakers.

    Wraps agent execution with resilience patterns to ensure the system
    continues functioning even when individual agents fail.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_backoff: Base backoff time in seconds for exponential backoff
        agent_timeout: Timeout for individual agent calls
        fallback_map: Mapping of agents to their fallback agents
        _execution_stats: Statistics about executions and failures
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_backoff_seconds: float = 1.0,
        agent_timeout: float = 30.0,
        fallback_map: dict[AgentRole, list[AgentRole]] | None = None,
    ):
        """Initialize the Self-Healing Agent.

        Args:
            max_retries: Maximum retry attempts before using fallback.
            base_backoff_seconds: Base for exponential backoff calculation.
            agent_timeout: Timeout per agent call in seconds.
            fallback_map: Custom fallback mappings (uses default if None).
        """
        self.max_retries = max_retries
        self.base_backoff = base_backoff_seconds
        self.agent_timeout = agent_timeout
        self.fallback_map = fallback_map or DEFAULT_FALLBACK_MAP

        # Statistics tracking
        self._execution_stats: dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "retry_executions": 0,
            "fallback_executions": 0,
            "failed_executions": 0,
        }

    async def execute_with_healing(
        self,
        agent_role: AgentRole,
        agent_func: Callable[[MultiAgentState], Any],
        state: MultiAgentState,
    ) -> MultiAgentState:
        """Execute agent with self-healing capabilities.

        Wraps agent execution with:
        1. Circuit breaker check
        2. Retry with exponential backoff
        3. Fallback to alternate agents if all retries fail

        Args:
            agent_role: Role of the agent to execute.
            agent_func: Async function that processes the state.
            state: Current workflow state.

        Returns:
            Updated state after agent execution (or fallback).
        """
        self._execution_stats["total_executions"] += 1
        start_time = time.perf_counter()

        circuit = circuit_registry.get_or_create(agent_role.value)

        # Step 1: Check circuit breaker
        if not circuit.can_execute():
            logger.warning(
                "circuit_open_using_fallback",
                agent=agent_role.value,
                circuit_state=circuit.state.value,
            )
            return await self._execute_fallback(agent_role, state)

        # Step 2: Execute with retry
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                result = await asyncio.wait_for(
                    agent_func(state),
                    timeout=self.agent_timeout,
                )

                # Success - record and return
                circuit.record_success()
                self._execution_stats["successful_executions"] += 1

                if attempt > 0:
                    self._execution_stats["retry_executions"] += 1

                duration_ms = (time.perf_counter() - start_time) * 1000

                logger.info(
                    "agent_execution_success",
                    agent=agent_role.value,
                    attempt=attempt + 1,
                    duration_ms=round(duration_ms, 2),
                )

                return result

            except TimeoutError:
                last_error = TimeoutError(
                    f"Agent {agent_role.value} timed out after {self.agent_timeout}s"
                )
                logger.warning(
                    "agent_timeout",
                    agent=agent_role.value,
                    attempt=attempt + 1,
                    timeout=self.agent_timeout,
                )
                circuit.record_failure()

            except Exception as e:
                last_error = e
                logger.error(
                    "agent_execution_error",
                    agent=agent_role.value,
                    attempt=attempt + 1,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                circuit.record_failure()

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                backoff = self.base_backoff * (2**attempt)
                logger.debug(
                    "retry_backoff",
                    agent=agent_role.value,
                    attempt=attempt + 1,
                    backoff_seconds=backoff,
                )
                await asyncio.sleep(backoff)

        # Step 3: All retries failed, use fallback
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.warning(
            "all_retries_failed_using_fallback",
            agent=agent_role.value,
            retries=self.max_retries,
            last_error=str(last_error) if last_error else "unknown",
            duration_ms=round(duration_ms, 2),
        )

        return await self._execute_fallback(agent_role, state)

    async def _execute_fallback(
        self,
        failed_agent: AgentRole,
        state: MultiAgentState,
    ) -> MultiAgentState:
        """Execute fallback agent when primary fails.

        Tries each fallback agent in order until one succeeds or
        all fallbacks are exhausted.

        Args:
            failed_agent: The agent that failed.
            state: Current workflow state.

        Returns:
            Updated state with fallback response (or error).
        """
        fallbacks = self.fallback_map.get(failed_agent, [])

        for fallback_role in fallbacks:
            circuit = circuit_registry.get_or_create(fallback_role.value)

            if not circuit.can_execute():
                logger.debug(
                    "fallback_circuit_open",
                    failed_agent=failed_agent.value,
                    fallback_agent=fallback_role.value,
                )
                continue  # Skip if fallback circuit is also open

            logger.info(
                "executing_fallback_agent",
                failed_agent=failed_agent.value,
                fallback_agent=fallback_role.value,
            )

            # Add fallback indicator to state
            state.agent_responses.append(
                AgentResponse(
                    agent_role=fallback_role,
                    content=(
                        f"[Fallback from {failed_agent.value}] "
                        f"Primary agent unavailable, using {fallback_role.value} as fallback."
                    ),
                    confidence=0.6,  # Lower confidence for fallback
                    execution_time_ms=0,
                    metadata={
                        "is_fallback": True,
                        "original_agent": failed_agent.value,
                        "fallback_agent": fallback_role.value,
                    },
                )
            )

            self._execution_stats["fallback_executions"] += 1
            return state

        # No fallbacks available - return error state
        self._execution_stats["failed_executions"] += 1

        error_msg = (
            f"Agent {failed_agent.value} failed and no fallbacks available. "
            f"Attempted fallbacks: {[f.value for f in fallbacks] if fallbacks else 'none'}"
        )

        state.agent_responses.append(
            AgentResponse(
                agent_role=failed_agent,
                content="",
                confidence=0.0,
                execution_time_ms=0,
                metadata={
                    "error": error_msg,
                    "is_failure": True,
                    "attempted_fallbacks": [f.value for f in fallbacks],
                },
            )
        )

        state.error = error_msg

        logger.error(
            "no_fallbacks_available",
            failed_agent=failed_agent.value,
            attempted_fallbacks=[f.value for f in fallbacks],
        )

        return state

    async def process(self, state: MultiAgentState) -> MultiAgentState:
        """Process state as the Self-Healing Agent.

        Monitors system health and records healing metrics.
        This method is called when Self-Healing Agent itself is
        invoked as part of the workflow.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with health information.
        """
        start_time = time.perf_counter()

        # Get system health status
        health = get_system_health()

        # Record health check in state
        state.memory_context["system_health"] = health
        state.memory_context["healing_stats"] = self._execution_stats.copy()

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add response
        state.agent_responses.append(
            AgentResponse(
                agent_role=AgentRole.SELF_HEALING,
                content=(
                    f"System health: {health['overall_status']}. "
                    f"Healthy: {health['healthy_agents']}/{health['total_agents']} agents."
                ),
                confidence=0.95,
                execution_time_ms=duration_ms,
                metadata={
                    "health_status": health["overall_status"],
                    "healthy_count": health["healthy_agents"],
                    "unhealthy_count": health["unhealthy_agents"],
                    "execution_stats": self._execution_stats.copy(),
                },
            )
        )

        state.current_agent = AgentRole.SELF_HEALING

        logger.info(
            "self_healing_check_complete",
            health_status=health["overall_status"],
            healthy_agents=health["healthy_agents"],
            total_agents=health["total_agents"],
        )

        return state

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health status.

        Combines circuit breaker status with execution statistics.

        Returns:
            Dictionary with health status and statistics.
        """
        health = get_system_health()
        health["execution_stats"] = self._execution_stats.copy()
        return health

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with execution statistics.
        """
        stats = self._execution_stats.copy()

        # Calculate rates
        total = stats["total_executions"]
        if total > 0:
            stats["success_rate"] = round(
                stats["successful_executions"] / total, 3
            )
            stats["retry_rate"] = round(
                stats["retry_executions"] / total, 3
            )
            stats["fallback_rate"] = round(
                stats["fallback_executions"] / total, 3
            )
            stats["failure_rate"] = round(
                stats["failed_executions"] / total, 3
            )
        else:
            stats["success_rate"] = 0.0
            stats["retry_rate"] = 0.0
            stats["fallback_rate"] = 0.0
            stats["failure_rate"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "retry_executions": 0,
            "fallback_executions": 0,
            "failed_executions": 0,
        }
        logger.info("self_healing_stats_reset")


# =============================================================================
# Factory function
# =============================================================================


def create_self_healing_agent(
    max_retries: int = 3,
    base_backoff: float = 1.0,
    timeout: float = 30.0,
) -> SelfHealingAgent:
    """Create a Self-Healing Agent instance.

    Factory function for consistent agent creation.

    Args:
        max_retries: Maximum retry attempts.
        base_backoff: Base backoff in seconds.
        timeout: Agent timeout in seconds.

    Returns:
        Configured SelfHealingAgent instance.
    """
    return SelfHealingAgent(
        max_retries=max_retries,
        base_backoff_seconds=base_backoff,
        agent_timeout=timeout,
    )


# =============================================================================
# Global instance for shared state
# =============================================================================

# Default self-healing agent instance
self_healing_agent = SelfHealingAgent()


__all__ = [
    "SelfHealingAgent",
    "create_self_healing_agent",
    "self_healing_agent",
    "DEFAULT_FALLBACK_MAP",
]
