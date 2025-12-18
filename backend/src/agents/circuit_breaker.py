"""Circuit Breaker Implementation for Level 4c: Resilient Agent Execution.

This module implements the Circuit Breaker pattern for agent resilience,
preventing cascade failures when agents are unhealthy.

Circuit Breaker States:
- CLOSED: Normal operation, all calls go through
- OPEN: Agent failing, all calls rejected immediately
- HALF_OPEN: Testing recovery, limited calls allowed

State Transitions:
- CLOSED → OPEN: After failure_threshold consecutive failures
- OPEN → HALF_OPEN: After recovery_timeout
- HALF_OPEN → CLOSED: After success_threshold successes
- HALF_OPEN → OPEN: On any failure

Configuration via environment variables:
- CIRCUIT_BREAKER_FAILURE_THRESHOLD: Failures before opening (default: 3)
- CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Seconds before half-open (default: 30)
- CIRCUIT_BREAKER_SUCCESS_THRESHOLD: Successes to close (default: 2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CircuitState(str, Enum):
    """Circuit breaker states.

    CLOSED: Normal operation - all requests pass through
    OPEN: Failing - requests rejected to prevent cascade failures
    HALF_OPEN: Testing recovery - limited requests to check health
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for agent resilience.

    Implements the Circuit Breaker pattern to protect the system from
    cascade failures when agents become unhealthy.

    Attributes:
        name: Identifier for this circuit breaker (usually agent name)
        failure_threshold: Consecutive failures before opening circuit
        success_threshold: Successes in HALF_OPEN to close circuit
        recovery_timeout_seconds: Wait time before attempting recovery
        state: Current circuit state
        failure_count: Consecutive failure count
        success_count: Success count (in HALF_OPEN state)
        last_failure_time: Timestamp of most recent failure
        last_state_change: Timestamp of last state transition
        total_requests: Total requests through this circuit
        total_failures: Total failures recorded
    """

    name: str
    failure_threshold: int = 3
    success_threshold: int = 2
    recovery_timeout_seconds: int = 30

    # State tracking
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: datetime | None = field(default=None)
    last_state_change: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    # Statistics
    total_requests: int = field(default=0)
    total_failures: int = field(default=0)

    def can_execute(self) -> bool:
        """Check if circuit allows execution.

        Returns:
            True if request should proceed, False if rejected.
        """
        self.total_requests += 1

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = (
                    datetime.now(UTC) - self.last_failure_time
                )
                if time_since_failure > timedelta(
                    seconds=self.recovery_timeout_seconds
                ):
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self) -> None:
        """Record successful execution.

        In HALF_OPEN state, increments success count and may close circuit.
        In CLOSED state, resets failure count.
        """
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                logger.info(
                    "circuit_breaker_recovered",
                    name=self.name,
                    success_count=self.success_count,
                )
        else:
            # Reset failure count on success in CLOSED state
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed execution.

        Increments failure count and may open circuit.
        """
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = datetime.now(UTC)

        if self.state == CircuitState.HALF_OPEN:
            # Any failure in HALF_OPEN immediately opens circuit
            self._transition_to(CircuitState.OPEN)
            logger.warning(
                "circuit_breaker_reopened",
                name=self.name,
                reason="failure_during_recovery",
            )
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "circuit_breaker_opened",
                    name=self.name,
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold,
                )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state.

        Args:
            new_state: Target state to transition to.
        """
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now(UTC)

        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0

        logger.info(
            "circuit_breaker_transition",
            name=self.name,
            old_state=old_state.value,
            new_state=new_state.value,
        )

    def reset(self) -> None:
        """Reset circuit breaker to initial state.

        Useful for testing or administrative reset.
        """
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now(UTC)

        logger.info("circuit_breaker_reset", name=self.name)

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker status and statistics.
        """
        failure_rate = (
            self.total_failures / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "failure_rate": round(failure_rate, 3),
            "last_failure": (
                self.last_failure_time.isoformat()
                if self.last_failure_time
                else None
            ),
            "last_state_change": self.last_state_change.isoformat(),
        }


class CircuitBreakerRegistry:
    """Registry of circuit breakers for all agents.

    Provides centralized management of circuit breakers, ensuring
    each agent has a dedicated circuit breaker instance.

    Attributes:
        _breakers: Dictionary mapping agent names to circuit breakers
        _default_failure_threshold: Default failures before opening
        _default_success_threshold: Default successes to close
        _default_recovery_timeout: Default recovery wait time
    """

    def __init__(
        self,
        default_failure_threshold: int = 3,
        default_success_threshold: int = 2,
        default_recovery_timeout: int = 30,
    ):
        """Initialize the registry.

        Args:
            default_failure_threshold: Failures before circuit opens.
            default_success_threshold: Successes to close circuit.
            default_recovery_timeout: Seconds before recovery attempt.
        """
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_failure_threshold = default_failure_threshold
        self._default_success_threshold = default_success_threshold
        self._default_recovery_timeout = default_recovery_timeout

    def get_or_create(
        self,
        agent_name: str,
        failure_threshold: int | None = None,
        success_threshold: int | None = None,
        recovery_timeout: int | None = None,
    ) -> CircuitBreaker:
        """Get or create circuit breaker for agent.

        Args:
            agent_name: Agent identifier.
            failure_threshold: Custom failure threshold (optional).
            success_threshold: Custom success threshold (optional).
            recovery_timeout: Custom recovery timeout (optional).

        Returns:
            CircuitBreaker instance for the agent.
        """
        if agent_name not in self._breakers:
            self._breakers[agent_name] = CircuitBreaker(
                name=agent_name,
                failure_threshold=(
                    failure_threshold or self._default_failure_threshold
                ),
                success_threshold=(
                    success_threshold or self._default_success_threshold
                ),
                recovery_timeout_seconds=(
                    recovery_timeout or self._default_recovery_timeout
                ),
            )
            logger.debug(
                "circuit_breaker_created",
                agent_name=agent_name,
            )

        return self._breakers[agent_name]

    def get_status(self) -> dict[str, str]:
        """Get status of all circuit breakers.

        Returns:
            Dictionary mapping agent names to circuit states.
        """
        return {
            name: breaker.state.value
            for name, breaker in self._breakers.items()
        }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers.

        Returns:
            Dictionary mapping agent names to statistics.
        """
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }

    def get_unhealthy_agents(self) -> list[str]:
        """Get list of agents with open circuits.

        Returns:
            List of agent names with OPEN circuit state.
        """
        return [
            name
            for name, breaker in self._breakers.items()
            if breaker.state == CircuitState.OPEN
        ]

    def get_recovering_agents(self) -> list[str]:
        """Get list of agents in recovery (HALF_OPEN).

        Returns:
            List of agent names in HALF_OPEN state.
        """
        return [
            name
            for name, breaker in self._breakers.items()
            if breaker.state == CircuitState.HALF_OPEN
        ]

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        for breaker in self._breakers.values():
            breaker.reset()

        logger.info(
            "circuit_breakers_reset_all",
            count=len(self._breakers),
        )

    def reset_agent(self, agent_name: str) -> bool:
        """Reset specific agent's circuit breaker.

        Args:
            agent_name: Agent to reset.

        Returns:
            True if reset successful, False if agent not found.
        """
        if agent_name in self._breakers:
            self._breakers[agent_name].reset()
            return True
        return False


# =============================================================================
# Global registry instance
# =============================================================================

# Default registry with standard configuration
# Can be overridden with custom settings via environment variables
circuit_registry = CircuitBreakerRegistry(
    default_failure_threshold=3,
    default_success_threshold=2,
    default_recovery_timeout=30,
)


# =============================================================================
# Helper functions
# =============================================================================


def get_circuit_status() -> dict[str, str]:
    """Get status of all circuits.

    Convenience function for external access.

    Returns:
        Dictionary mapping agent names to states.
    """
    return circuit_registry.get_status()


def get_system_health() -> dict[str, Any]:
    """Get overall system health from circuit breaker perspective.

    Returns:
        Dictionary with health summary and details.
    """
    status = circuit_registry.get_status()
    stats = circuit_registry.get_all_stats()

    healthy_count = sum(
        1 for s in status.values() if s == CircuitState.CLOSED.value
    )
    recovering_count = sum(
        1 for s in status.values() if s == CircuitState.HALF_OPEN.value
    )
    unhealthy_count = sum(
        1 for s in status.values() if s == CircuitState.OPEN.value
    )
    total_count = len(status)

    if total_count == 0:
        overall_status = "unknown"
    elif unhealthy_count == 0 and recovering_count == 0:
        overall_status = "healthy"
    elif unhealthy_count == 0:
        overall_status = "recovering"
    elif unhealthy_count < total_count:
        overall_status = "degraded"
    else:
        overall_status = "critical"

    return {
        "overall_status": overall_status,
        "healthy_agents": healthy_count,
        "recovering_agents": recovering_count,
        "unhealthy_agents": unhealthy_count,
        "total_agents": total_count,
        "circuit_breakers": stats,
    }


__all__ = [
    "CircuitState",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "circuit_registry",
    "get_circuit_status",
    "get_system_health",
]
