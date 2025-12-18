"""Cost Optimizer for Level 4c: Agent Cost Management.

This module implements cost-aware routing and optimization strategies
to minimize API costs while maintaining response quality.

Features:
- Per-agent cost tracking
- Query-to-agent cost estimation
- Budget-based routing decisions
- Cost alerts and thresholds
- Token usage optimization

Configuration:
- COST_OPTIMIZATION_ENABLED: Enable cost-aware routing (default: true)
- DAILY_BUDGET_LIMIT: Maximum daily cost in dollars (optional)
- COST_ALERT_THRESHOLD: Percentage of budget triggering alert (default: 80)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from backend.src.models.multi_agent import AgentRole, QueryComplexity
from backend.src.orchestration.load_aware_router import CostTier

logger = structlog.get_logger()


# =============================================================================
# Cost configuration per agent/model
# =============================================================================

# Estimated cost per 1000 tokens (input + output)
TOKEN_COSTS: dict[str, float] = {
    "gpt-4o-mini": 0.00015,  # $0.15 per million tokens
    "gpt-4o": 0.0025,  # $2.50 per million tokens
    "gpt-4": 0.03,  # $30 per million tokens
    "claude-3-sonnet": 0.003,  # $3 per million tokens
    "claude-3-opus": 0.015,  # $15 per million tokens
}

# Estimated tokens per agent call
AGENT_TOKEN_ESTIMATES: dict[AgentRole, dict[str, int]] = {
    # Level 4a agents
    AgentRole.TRIAGE: {
        "input_tokens": 500,
        "output_tokens": 200,
        "model": "gpt-4o-mini",
    },
    AgentRole.HURRICANE_SPECIALIST: {
        "input_tokens": 2000,
        "output_tokens": 1000,
        "model": "gpt-4o",
    },
    AgentRole.ALERT_MANAGER: {
        "input_tokens": 800,
        "output_tokens": 400,
        "model": "gpt-4o-mini",
    },
    # Level 4b agents
    AgentRole.SUPERVISOR: {
        "input_tokens": 1500,
        "output_tokens": 500,
        "model": "gpt-4o",
    },
    AgentRole.FORECASTER: {
        "input_tokens": 1200,
        "output_tokens": 600,
        "model": "gpt-4o",
    },
    AgentRole.HISTORICAL_ANALYST: {
        "input_tokens": 1500,
        "output_tokens": 800,
        "model": "gpt-4o",
    },
    AgentRole.VERIFICATION: {
        "input_tokens": 1000,
        "output_tokens": 400,
        "model": "gpt-4o",
    },
    AgentRole.RESEARCH: {
        "input_tokens": 2000,
        "output_tokens": 1200,
        "model": "gpt-4o",
    },
    AgentRole.SYNTHESIS: {
        "input_tokens": 1500,
        "output_tokens": 600,
        "model": "gpt-4o",
    },
    # Level 4c agents
    AgentRole.META_PROMPT: {
        "input_tokens": 1000,
        "output_tokens": 500,
        "model": "gpt-4o",
    },
    AgentRole.SELF_HEALING: {
        "input_tokens": 200,
        "output_tokens": 100,
        "model": "gpt-4o-mini",
    },
    AgentRole.DEBATE: {
        "input_tokens": 3000,
        "output_tokens": 1500,
        "model": "gpt-4o",
    },
    AgentRole.EMERGENCY_RESPONSE: {
        "input_tokens": 1500,
        "output_tokens": 800,
        "model": "gpt-4o",
    },
    AgentRole.CLIMATE_ANALYST: {
        "input_tokens": 1800,
        "output_tokens": 900,
        "model": "gpt-4o",
    },
    AgentRole.PERSONALIZATION: {
        "input_tokens": 800,
        "output_tokens": 400,
        "model": "gpt-4o-mini",
    },
    AgentRole.REFLECTION: {
        "input_tokens": 1200,
        "output_tokens": 600,
        "model": "gpt-4o",
    },
}


@dataclass
class CostRecord:
    """Record of a single cost event.

    Attributes:
        agent_role: Agent that incurred the cost
        timestamp: When the cost was incurred
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        cost_usd: Total cost in USD
        query_id: Associated query ID (optional)
    """

    agent_role: AgentRole
    timestamp: datetime
    input_tokens: int
    output_tokens: int
    cost_usd: float
    query_id: str | None = None


@dataclass
class CostBudget:
    """Budget configuration and tracking.

    Attributes:
        daily_limit_usd: Maximum daily spend in USD
        alert_threshold: Percentage triggering alerts (0-100)
        current_spend_usd: Current daily spend
        last_reset: When the daily counter was last reset
        alerts_sent: Number of alerts sent today
    """

    daily_limit_usd: float = 100.0
    alert_threshold: float = 80.0
    current_spend_usd: float = 0.0
    last_reset: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    alerts_sent: int = 0


class CostOptimizer:
    """Optimize agent selection based on cost.

    Tracks costs, enforces budgets, and provides cost-aware
    routing recommendations.

    Attributes:
        _cost_history: History of cost records
        _budget: Budget configuration
        _agent_costs: Accumulated costs per agent
        _query_costs: Costs per query
    """

    def __init__(
        self,
        daily_budget_usd: float = 100.0,
        alert_threshold: float = 80.0,
    ):
        """Initialize the Cost Optimizer.

        Args:
            daily_budget_usd: Daily budget limit in USD.
            alert_threshold: Percentage of budget triggering alert.
        """
        self._cost_history: list[CostRecord] = []
        self._budget = CostBudget(
            daily_limit_usd=daily_budget_usd,
            alert_threshold=alert_threshold,
        )
        self._agent_costs: dict[AgentRole, float] = {}
        self._query_costs: dict[str, float] = {}

    def estimate_agent_cost(self, agent_role: AgentRole) -> float:
        """Estimate cost for a single agent call.

        Args:
            agent_role: Agent to estimate cost for.

        Returns:
            Estimated cost in USD.
        """
        if agent_role not in AGENT_TOKEN_ESTIMATES:
            return 0.01  # Default estimate

        estimate = AGENT_TOKEN_ESTIMATES[agent_role]
        model = estimate["model"]
        total_tokens = estimate["input_tokens"] + estimate["output_tokens"]

        token_cost = TOKEN_COSTS.get(model, 0.001)  # Default cost
        cost = (total_tokens / 1000) * token_cost

        return round(cost, 6)

    def estimate_workflow_cost(
        self,
        agents: list[AgentRole],
        complexity: QueryComplexity,
    ) -> dict[str, Any]:
        """Estimate total cost for a workflow.

        Args:
            agents: List of agents to be invoked.
            complexity: Query complexity level.

        Returns:
            Dictionary with cost breakdown.
        """
        agent_costs: dict[str, float] = {}
        total_cost = 0.0

        for agent in agents:
            cost = self.estimate_agent_cost(agent)
            agent_costs[agent.value] = cost
            total_cost += cost

        # Apply complexity multiplier
        multipliers = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 1.2,
            QueryComplexity.COMPLEX: 1.5,
            QueryComplexity.EMERGENCY: 1.3,
        }
        multiplier = multipliers.get(complexity, 1.0)
        total_cost *= multiplier

        return {
            "agent_costs": agent_costs,
            "subtotal": sum(agent_costs.values()),
            "complexity_multiplier": multiplier,
            "total_estimate_usd": round(total_cost, 6),
        }

    def record_cost(
        self,
        agent_role: AgentRole,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4o",
        query_id: str | None = None,
    ) -> CostRecord:
        """Record actual cost from agent execution.

        Args:
            agent_role: Agent that incurred the cost.
            input_tokens: Actual input tokens used.
            output_tokens: Actual output tokens used.
            model: Model used for the call.
            query_id: Associated query ID.

        Returns:
            Created CostRecord.
        """
        # Check if we need to reset daily counters
        self._check_daily_reset()

        # Calculate cost
        total_tokens = input_tokens + output_tokens
        token_cost = TOKEN_COSTS.get(model, 0.001)
        cost_usd = (total_tokens / 1000) * token_cost

        # Create record
        record = CostRecord(
            agent_role=agent_role,
            timestamp=datetime.now(UTC),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            query_id=query_id,
        )

        # Update tracking
        self._cost_history.append(record)
        self._budget.current_spend_usd += cost_usd

        if agent_role not in self._agent_costs:
            self._agent_costs[agent_role] = 0.0
        self._agent_costs[agent_role] += cost_usd

        if query_id:
            if query_id not in self._query_costs:
                self._query_costs[query_id] = 0.0
            self._query_costs[query_id] += cost_usd

        # Check budget alerts
        self._check_budget_alerts()

        logger.debug(
            "cost_recorded",
            agent=agent_role.value,
            cost_usd=round(cost_usd, 6),
            total_tokens=total_tokens,
            daily_spend=round(self._budget.current_spend_usd, 4),
        )

        return record

    def _check_daily_reset(self) -> None:
        """Reset daily counters if needed."""
        now = datetime.now(UTC)
        if now.date() > self._budget.last_reset.date():
            self._budget.current_spend_usd = 0.0
            self._budget.alerts_sent = 0
            self._budget.last_reset = now

            logger.info("cost_daily_reset")

    def _check_budget_alerts(self) -> None:
        """Check and send budget alerts if threshold exceeded."""
        spend_percentage = (
            self._budget.current_spend_usd / self._budget.daily_limit_usd * 100
        )

        if (
            spend_percentage >= self._budget.alert_threshold
            and self._budget.alerts_sent == 0
        ):
            self._budget.alerts_sent += 1
            logger.warning(
                "cost_budget_alert",
                current_spend=round(self._budget.current_spend_usd, 2),
                daily_limit=self._budget.daily_limit_usd,
                percentage=round(spend_percentage, 1),
                threshold=self._budget.alert_threshold,
            )

    def is_within_budget(
        self,
        estimated_cost: float,
    ) -> bool:
        """Check if estimated cost is within budget.

        Args:
            estimated_cost: Estimated cost in USD.

        Returns:
            True if cost is within budget.
        """
        return (
            self._budget.current_spend_usd + estimated_cost
            <= self._budget.daily_limit_usd
        )

    def get_cost_tier_limit(self) -> CostTier:
        """Get recommended cost tier based on budget status.

        Returns lower cost tier as budget is consumed.

        Returns:
            Recommended maximum cost tier.
        """
        spend_percentage = (
            self._budget.current_spend_usd / self._budget.daily_limit_usd * 100
        )

        if spend_percentage < 50:
            return CostTier.HIGH
        elif spend_percentage < 80:
            return CostTier.MEDIUM
        else:
            return CostTier.LOW

    def get_cost_report(self) -> dict[str, Any]:
        """Get comprehensive cost report.

        Returns:
            Dictionary with cost statistics.
        """
        # Calculate daily costs
        today = datetime.now(UTC).date()
        today_records = [
            r for r in self._cost_history
            if r.timestamp.date() == today
        ]
        today_cost = sum(r.cost_usd for r in today_records)

        # Calculate agent breakdown
        agent_breakdown: dict[str, float] = {}
        for role, cost in self._agent_costs.items():
            agent_breakdown[role.value] = round(cost, 4)

        # Calculate average query cost
        avg_query_cost = (
            sum(self._query_costs.values()) / len(self._query_costs)
            if self._query_costs
            else 0.0
        )

        return {
            "budget": {
                "daily_limit_usd": self._budget.daily_limit_usd,
                "current_spend_usd": round(self._budget.current_spend_usd, 4),
                "remaining_usd": round(
                    self._budget.daily_limit_usd - self._budget.current_spend_usd, 4
                ),
                "spend_percentage": round(
                    self._budget.current_spend_usd / self._budget.daily_limit_usd * 100, 1
                ),
            },
            "today": {
                "total_cost_usd": round(today_cost, 4),
                "request_count": len(today_records),
                "avg_cost_per_request": round(
                    today_cost / len(today_records) if today_records else 0, 6
                ),
            },
            "by_agent": agent_breakdown,
            "queries": {
                "total_queries": len(self._query_costs),
                "avg_cost_per_query": round(avg_query_cost, 6),
            },
            "recommended_tier": self.get_cost_tier_limit().value,
        }

    def get_cheapest_agents(
        self,
        count: int = 5,
    ) -> list[tuple[AgentRole, float]]:
        """Get the cheapest agents by estimated cost.

        Args:
            count: Number of agents to return.

        Returns:
            List of (agent_role, estimated_cost) tuples.
        """
        agent_costs = [
            (role, self.estimate_agent_cost(role))
            for role in AgentRole
            if role in AGENT_TOKEN_ESTIMATES
        ]

        sorted_agents = sorted(agent_costs, key=lambda x: x[1])
        return sorted_agents[:count]

    def optimize_workflow(
        self,
        agents: list[AgentRole],
        max_budget: float | None = None,
    ) -> list[AgentRole]:
        """Optimize workflow by removing expensive agents if over budget.

        Args:
            agents: Proposed list of agents.
            max_budget: Maximum budget for this workflow (optional).

        Returns:
            Optimized list of agents.
        """
        if not max_budget:
            max_budget = (
                self._budget.daily_limit_usd - self._budget.current_spend_usd
            )

        # Sort agents by cost (keep essential ones)
        essential_agents = {AgentRole.TRIAGE, AgentRole.ALERT_MANAGER}
        optional_agents = [a for a in agents if a not in essential_agents]

        # Sort optional by cost
        optional_with_cost = [
            (a, self.estimate_agent_cost(a)) for a in optional_agents
        ]
        optional_with_cost.sort(key=lambda x: x[1])

        # Build optimized list
        optimized: list[AgentRole] = []
        total_cost = 0.0

        # Add essential agents first
        for agent in agents:
            if agent in essential_agents:
                cost = self.estimate_agent_cost(agent)
                if total_cost + cost <= max_budget:
                    optimized.append(agent)
                    total_cost += cost

        # Add optional agents by cost
        for agent, cost in optional_with_cost:
            if total_cost + cost <= max_budget:
                optimized.append(agent)
                total_cost += cost
            else:
                logger.info(
                    "agent_excluded_budget",
                    agent=agent.value,
                    cost=cost,
                    remaining_budget=max_budget - total_cost,
                )

        return optimized


# =============================================================================
# Global instance
# =============================================================================

cost_optimizer = CostOptimizer(
    daily_budget_usd=100.0,
    alert_threshold=80.0,
)


__all__ = [
    "CostOptimizer",
    "CostRecord",
    "CostBudget",
    "cost_optimizer",
    "TOKEN_COSTS",
    "AGENT_TOKEN_ESTIMATES",
]
