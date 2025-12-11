"""Load-Aware Router for Level 4c: Intelligent Agent Selection.

This module implements intelligent routing based on cost, latency,
and availability for optimal agent selection.

Features:
- Cost-tier routing (cheap agents first for simple queries)
- Latency-aware selection
- Load balancing across agents
- Real-time performance tracking
- Adaptive routing based on agent health

Configuration:
- MAX_CONCURRENT_AGENTS: Maximum parallel agents (default: 4)
- DEFAULT_AGENT_TIMEOUT: Timeout in seconds (default: 30)
- COST_OPTIMIZATION_ENABLED: Enable cost-aware routing (default: true)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from backend.src.agents.circuit_breaker import circuit_registry
from backend.src.models.multi_agent import AgentRole, QueryComplexity

logger = structlog.get_logger()


class CostTier(str, Enum):
    """Agent cost tiers based on resource consumption.

    LOW: Fast, cheap models (gpt-4o-mini, simple processing)
    MEDIUM: Balanced models (gpt-4o, moderate processing)
    HIGH: Premium models (gpt-4o + tools + verification)
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AgentMetrics:
    """Real-time agent performance metrics.

    Tracks agent performance for intelligent routing decisions.

    Attributes:
        agent_role: The agent these metrics are for
        cost_tier: Cost classification of the agent
        avg_latency_ms: Moving average of response latency
        success_rate: Success rate (0.0 to 1.0)
        current_load: Current number of active requests
        max_load: Maximum concurrent requests allowed
        total_requests: Total requests processed
        last_updated: When metrics were last updated
    """

    agent_role: AgentRole
    cost_tier: CostTier
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    current_load: int = 0
    max_load: int = 10
    total_requests: int = 0
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# Default agent metrics configuration
# =============================================================================

DEFAULT_AGENT_METRICS: dict[AgentRole, dict[str, Any]] = {
    # Level 4a agents
    AgentRole.TRIAGE: {
        "cost_tier": CostTier.LOW,
        "avg_latency_ms": 500,
        "max_load": 20,
    },
    AgentRole.HURRICANE_SPECIALIST: {
        "cost_tier": CostTier.HIGH,
        "avg_latency_ms": 5000,
        "max_load": 5,
    },
    AgentRole.ALERT_MANAGER: {
        "cost_tier": CostTier.LOW,
        "avg_latency_ms": 1000,
        "max_load": 15,
    },
    # Level 4b agents
    AgentRole.SUPERVISOR: {
        "cost_tier": CostTier.MEDIUM,
        "avg_latency_ms": 2000,
        "max_load": 10,
    },
    AgentRole.FORECASTER: {
        "cost_tier": CostTier.MEDIUM,
        "avg_latency_ms": 2000,
        "max_load": 10,
    },
    AgentRole.HISTORICAL_ANALYST: {
        "cost_tier": CostTier.MEDIUM,
        "avg_latency_ms": 3000,
        "max_load": 8,
    },
    AgentRole.VERIFICATION: {
        "cost_tier": CostTier.MEDIUM,
        "avg_latency_ms": 2000,
        "max_load": 10,
    },
    AgentRole.RESEARCH: {
        "cost_tier": CostTier.HIGH,
        "avg_latency_ms": 4000,
        "max_load": 5,
    },
    AgentRole.SYNTHESIS: {
        "cost_tier": CostTier.MEDIUM,
        "avg_latency_ms": 1500,
        "max_load": 10,
    },
    # Level 4c agents
    AgentRole.META_PROMPT: {
        "cost_tier": CostTier.MEDIUM,
        "avg_latency_ms": 1500,
        "max_load": 10,
    },
    AgentRole.SELF_HEALING: {
        "cost_tier": CostTier.LOW,
        "avg_latency_ms": 500,
        "max_load": 20,
    },
    AgentRole.DEBATE: {
        "cost_tier": CostTier.HIGH,
        "avg_latency_ms": 8000,
        "max_load": 3,
    },
    AgentRole.EMERGENCY_RESPONSE: {
        "cost_tier": CostTier.HIGH,
        "avg_latency_ms": 2000,
        "max_load": 10,
    },
    AgentRole.CLIMATE_ANALYST: {
        "cost_tier": CostTier.MEDIUM,
        "avg_latency_ms": 3500,
        "max_load": 6,
    },
    AgentRole.PERSONALIZATION: {
        "cost_tier": CostTier.LOW,
        "avg_latency_ms": 1000,
        "max_load": 15,
    },
    AgentRole.REFLECTION: {
        "cost_tier": CostTier.MEDIUM,
        "avg_latency_ms": 2500,
        "max_load": 8,
    },
}


class LoadAwareRouter:
    """Route queries based on cost, latency, and load.

    Implements intelligent routing that considers:
    - Query complexity
    - Agent availability (circuit breaker status)
    - Cost tier preferences
    - Current load on each agent
    - Historical performance metrics

    Attributes:
        _metrics: Performance metrics for each agent
        _cost_optimization_enabled: Whether to prefer cheaper agents
        _routing_history: History of routing decisions
    """

    def __init__(
        self,
        cost_optimization_enabled: bool = True,
    ):
        """Initialize the Load-Aware Router.

        Args:
            cost_optimization_enabled: Whether to prefer cheaper agents
                for simple queries.
        """
        self._metrics: dict[AgentRole, AgentMetrics] = self._initialize_metrics()
        self._cost_optimization_enabled = cost_optimization_enabled
        self._routing_history: list[dict[str, Any]] = []

    def _initialize_metrics(self) -> dict[AgentRole, AgentMetrics]:
        """Initialize agent metrics from defaults.

        Returns:
            Dictionary mapping agent roles to their metrics.
        """
        metrics: dict[AgentRole, AgentMetrics] = {}

        for role, config in DEFAULT_AGENT_METRICS.items():
            metrics[role] = AgentMetrics(
                agent_role=role,
                cost_tier=config["cost_tier"],
                avg_latency_ms=config["avg_latency_ms"],
                max_load=config.get("max_load", 10),
            )

        return metrics

    def select_agents(
        self,
        query_complexity: QueryComplexity,
        required_capabilities: list[str] | None = None,
        max_cost_tier: CostTier = CostTier.HIGH,
        max_latency_ms: float = 10000,
        excluded_agents: list[AgentRole] | None = None,
    ) -> list[AgentRole]:
        """Select optimal agents based on constraints.

        Analyzes query requirements and agent availability to select
        the best agents for the query.

        Args:
            query_complexity: Query complexity level.
            required_capabilities: Required agent capabilities (optional).
            max_cost_tier: Maximum acceptable cost tier.
            max_latency_ms: Maximum acceptable latency.
            excluded_agents: Agents to exclude from selection.

        Returns:
            Ordered list of agents to invoke.
        """
        excluded = set(excluded_agents or [])
        candidates: list[tuple[AgentRole, AgentMetrics]] = []

        for role, metrics in self._metrics.items():
            if role in excluded:
                continue

            # Check circuit breaker
            circuit = circuit_registry.get_or_create(role.value)
            if not circuit.can_execute():
                continue

            # Check cost tier (if optimization enabled)
            if self._cost_optimization_enabled:
                if self._cost_tier_value(metrics.cost_tier) > self._cost_tier_value(max_cost_tier):
                    continue

            # Check latency
            if metrics.avg_latency_ms > max_latency_ms:
                continue

            # Check load
            if metrics.current_load >= metrics.max_load:
                continue

            candidates.append((role, metrics))

        # Sort by: cost tier (if optimizing), then latency
        if self._cost_optimization_enabled:
            sorted_candidates = sorted(
                candidates,
                key=lambda x: (
                    self._cost_tier_value(x[1].cost_tier),
                    x[1].avg_latency_ms,
                ),
            )
        else:
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x[1].avg_latency_ms,
            )

        selected = [c[0] for c in sorted_candidates]

        # Ensure triage is always first if present
        if AgentRole.TRIAGE in selected:
            selected.remove(AgentRole.TRIAGE)
            selected.insert(0, AgentRole.TRIAGE)

        # Add complexity-based agents
        selected = self._add_complexity_agents(
            selected, query_complexity
        )

        # Record routing decision
        self._routing_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "complexity": query_complexity.value,
            "selected": [s.value for s in selected],
            "excluded": [e.value for e in excluded],
        })

        logger.info(
            "agents_selected",
            complexity=query_complexity.value,
            selected=[s.value for s in selected],
            count=len(selected),
            cost_optimization=self._cost_optimization_enabled,
        )

        return selected

    def _add_complexity_agents(
        self,
        selected: list[AgentRole],
        complexity: QueryComplexity,
    ) -> list[AgentRole]:
        """Add agents based on query complexity.

        Args:
            selected: Currently selected agents.
            complexity: Query complexity level.

        Returns:
            Updated list with complexity-appropriate agents.
        """
        # Simple queries: Triage + single specialist
        if complexity == QueryComplexity.SIMPLE:
            # Keep triage, limit to one specialist
            if len(selected) > 2:
                triage = selected[0] if selected else None
                specialists = [s for s in selected[1:] if s != AgentRole.TRIAGE]
                selected = [triage, specialists[0]] if triage and specialists else selected[:2]

        # Moderate: Triage + specialist + verification
        elif complexity == QueryComplexity.MODERATE:
            if AgentRole.VERIFICATION not in selected:
                selected.append(AgentRole.VERIFICATION)

        # Complex: Include reflection and synthesis
        elif complexity == QueryComplexity.COMPLEX:
            if AgentRole.REFLECTION not in selected:
                selected.append(AgentRole.REFLECTION)
            if AgentRole.SYNTHESIS not in selected:
                selected.append(AgentRole.SYNTHESIS)

        # Emergency: Prioritize emergency and alert agents
        elif complexity == QueryComplexity.EMERGENCY:
            emergency_agents = [
                AgentRole.EMERGENCY_RESPONSE,
                AgentRole.ALERT_MANAGER,
                AgentRole.HURRICANE_SPECIALIST,
            ]
            for agent in emergency_agents:
                if agent not in selected:
                    selected.insert(1, agent)  # After triage

        return selected

    def _cost_tier_value(self, tier: CostTier) -> int:
        """Convert cost tier to numeric value.

        Args:
            tier: Cost tier to convert.

        Returns:
            Numeric value (1=LOW, 2=MEDIUM, 3=HIGH).
        """
        return {CostTier.LOW: 1, CostTier.MEDIUM: 2, CostTier.HIGH: 3}[tier]

    def update_metrics(
        self,
        agent_role: AgentRole,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Update agent metrics after execution.

        Uses exponential moving average for latency and success rate.

        Args:
            agent_role: Agent to update metrics for.
            latency_ms: Execution latency in milliseconds.
            success: Whether execution was successful.
        """
        if agent_role not in self._metrics:
            return

        metrics = self._metrics[agent_role]
        metrics.total_requests += 1

        # Update latency (exponential moving average)
        alpha = 0.2
        metrics.avg_latency_ms = (
            (1 - alpha) * metrics.avg_latency_ms + alpha * latency_ms
        )

        # Update success rate
        if success:
            metrics.success_rate = min(1.0, metrics.success_rate + 0.01)
        else:
            metrics.success_rate = max(0.5, metrics.success_rate - 0.05)

        metrics.last_updated = datetime.now(timezone.utc)

        logger.debug(
            "agent_metrics_updated",
            agent=agent_role.value,
            avg_latency_ms=round(metrics.avg_latency_ms, 2),
            success_rate=round(metrics.success_rate, 3),
        )

    def increment_load(self, agent_role: AgentRole) -> None:
        """Increment agent load counter.

        Args:
            agent_role: Agent to increment load for.
        """
        if agent_role in self._metrics:
            self._metrics[agent_role].current_load += 1

    def decrement_load(self, agent_role: AgentRole) -> None:
        """Decrement agent load counter.

        Args:
            agent_role: Agent to decrement load for.
        """
        if agent_role in self._metrics:
            self._metrics[agent_role].current_load = max(
                0, self._metrics[agent_role].current_load - 1
            )

    def get_routing_report(self) -> dict[str, Any]:
        """Get routing statistics report.

        Returns:
            Dictionary with agent metrics and routing stats.
        """
        return {
            role.value: {
                "cost_tier": metrics.cost_tier.value,
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "success_rate": round(metrics.success_rate, 3),
                "current_load": metrics.current_load,
                "max_load": metrics.max_load,
                "total_requests": metrics.total_requests,
                "utilization": round(
                    metrics.current_load / metrics.max_load * 100, 1
                ) if metrics.max_load > 0 else 0,
            }
            for role, metrics in self._metrics.items()
        }

    def get_available_agents(self) -> list[AgentRole]:
        """Get list of currently available agents.

        Returns:
            List of agents that can accept requests.
        """
        available: list[AgentRole] = []

        for role, metrics in self._metrics.items():
            # Check circuit breaker
            circuit = circuit_registry.get_or_create(role.value)
            if not circuit.can_execute():
                continue

            # Check load
            if metrics.current_load >= metrics.max_load:
                continue

            available.append(role)

        return available

    def get_agent_status(self, agent_role: AgentRole) -> dict[str, Any] | None:
        """Get status of a specific agent.

        Args:
            agent_role: Agent to get status for.

        Returns:
            Status dictionary or None if agent not found.
        """
        if agent_role not in self._metrics:
            return None

        metrics = self._metrics[agent_role]
        circuit = circuit_registry.get_or_create(agent_role.value)

        return {
            "role": agent_role.value,
            "cost_tier": metrics.cost_tier.value,
            "avg_latency_ms": round(metrics.avg_latency_ms, 2),
            "success_rate": round(metrics.success_rate, 3),
            "current_load": metrics.current_load,
            "max_load": metrics.max_load,
            "circuit_state": circuit.state.value,
            "available": circuit.can_execute() and metrics.current_load < metrics.max_load,
        }


# =============================================================================
# Global router instance
# =============================================================================

load_aware_router = LoadAwareRouter(cost_optimization_enabled=True)


__all__ = [
    "CostTier",
    "AgentMetrics",
    "LoadAwareRouter",
    "load_aware_router",
    "DEFAULT_AGENT_METRICS",
]
