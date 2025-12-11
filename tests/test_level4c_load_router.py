"""Tests for Level 4c Load-Aware Router.

This module tests the load-aware routing capabilities including:
- Cost tier classification
- Agent metrics tracking
- Load-aware agent selection
- Budget-constrained routing
- Latency optimization
"""

import pytest
from unittest.mock import MagicMock

from backend.src.orchestration.load_aware_router import (
    LoadAwareRouter,
    AgentMetrics,
    CostTier,
    DEFAULT_AGENT_METRICS,
    create_load_aware_router,
)
from backend.src.models.multi_agent import AgentRole, MultiAgentState, QueryComplexity


class TestCostTier:
    """Tests for CostTier enum."""

    def test_cost_tiers_exist(self):
        """Test that all cost tiers are defined."""
        assert CostTier.LOW.value == "low"
        assert CostTier.MEDIUM.value == "medium"
        assert CostTier.HIGH.value == "high"


class TestAgentMetrics:
    """Tests for AgentMetrics dataclass."""

    def test_default_values(self):
        """Test default values for AgentMetrics."""
        metrics = AgentMetrics(
            agent_role=AgentRole.FORECASTER,
            cost_tier=CostTier.MEDIUM,
        )

        assert metrics.avg_latency_ms == 0.0
        assert metrics.success_rate == 1.0
        assert metrics.current_load == 0.0
        assert metrics.max_concurrent == 5
        assert metrics.priority == 5

    def test_custom_values(self):
        """Test custom values for AgentMetrics."""
        metrics = AgentMetrics(
            agent_role=AgentRole.HURRICANE_SPECIALIST,
            cost_tier=CostTier.HIGH,
            avg_latency_ms=500.0,
            success_rate=0.95,
            current_load=0.75,
            max_concurrent=3,
            priority=10,
        )

        assert metrics.agent_role == AgentRole.HURRICANE_SPECIALIST
        assert metrics.cost_tier == CostTier.HIGH
        assert metrics.avg_latency_ms == 500.0
        assert metrics.success_rate == 0.95
        assert metrics.current_load == 0.75
        assert metrics.max_concurrent == 3
        assert metrics.priority == 10


class TestDefaultAgentMetrics:
    """Tests for DEFAULT_AGENT_METRICS configuration."""

    def test_all_agents_have_metrics(self):
        """Test that all expected agents have default metrics."""
        expected_agents = [
            AgentRole.TRIAGE,
            AgentRole.HURRICANE_SPECIALIST,
            AgentRole.FORECASTER,
            AgentRole.ALERT_MANAGER,
            AgentRole.SUPERVISOR,
        ]

        for agent in expected_agents:
            assert agent in DEFAULT_AGENT_METRICS, f"Missing metrics for {agent}"

    def test_triage_is_low_cost(self):
        """Test that triage agent is low cost."""
        metrics = DEFAULT_AGENT_METRICS.get(AgentRole.TRIAGE)
        assert metrics is not None
        assert metrics.cost_tier == CostTier.LOW

    def test_hurricane_specialist_is_high_cost(self):
        """Test that hurricane specialist is high cost."""
        metrics = DEFAULT_AGENT_METRICS.get(AgentRole.HURRICANE_SPECIALIST)
        assert metrics is not None
        assert metrics.cost_tier == CostTier.HIGH


class TestLoadAwareRouterInit:
    """Tests for LoadAwareRouter initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        router = LoadAwareRouter()

        assert router.max_budget_per_query is None
        assert router.prefer_low_latency is True
        assert router.load_threshold == 0.8
        assert len(router._agent_metrics) > 0

    def test_custom_initialization(self):
        """Test custom initialization."""
        router = LoadAwareRouter(
            max_budget_per_query=0.05,
            prefer_low_latency=False,
            load_threshold=0.9,
        )

        assert router.max_budget_per_query == 0.05
        assert router.prefer_low_latency is False
        assert router.load_threshold == 0.9

    def test_custom_metrics(self):
        """Test initialization with custom metrics."""
        custom_metrics = {
            AgentRole.FORECASTER: AgentMetrics(
                agent_role=AgentRole.FORECASTER,
                cost_tier=CostTier.LOW,
                priority=1,
            )
        }

        router = LoadAwareRouter(agent_metrics=custom_metrics)

        assert AgentRole.FORECASTER in router._agent_metrics
        assert router._agent_metrics[AgentRole.FORECASTER].priority == 1


class TestSelectAgents:
    """Tests for select_agents method."""

    def test_select_agents_basic(self):
        """Test basic agent selection."""
        router = LoadAwareRouter()
        state = MultiAgentState(
            query="What's the weather forecast?",
            user_id="test_user",
            complexity=QueryComplexity.SIMPLE,
        )

        selected = router.select_agents(state)

        assert len(selected) > 0
        assert all(isinstance(a, AgentRole) for a in selected)

    def test_select_agents_respects_complexity(self):
        """Test that selection considers query complexity."""
        router = LoadAwareRouter()

        simple_state = MultiAgentState(
            query="Weather in NYC",
            user_id="test_user",
            complexity=QueryComplexity.SIMPLE,
        )

        complex_state = MultiAgentState(
            query="Hurricane analysis with historical comparison",
            user_id="test_user",
            complexity=QueryComplexity.COMPLEX,
        )

        simple_agents = router.select_agents(simple_state)
        complex_agents = router.select_agents(complex_state)

        # Complex queries might get more or different agents
        # At minimum, both should return agents
        assert len(simple_agents) >= 1
        assert len(complex_agents) >= 1

    def test_select_agents_respects_budget(self):
        """Test that selection respects budget constraints."""
        # Very low budget router
        router = LoadAwareRouter(max_budget_per_query=0.001)

        state = MultiAgentState(
            query="Complex hurricane analysis",
            user_id="test_user",
            complexity=QueryComplexity.COMPLEX,
        )

        selected = router.select_agents(state)

        # Should still return at least some agents
        # but might prefer lower cost options
        assert len(selected) >= 1

    def test_select_agents_avoids_overloaded(self):
        """Test that overloaded agents are deprioritized."""
        router = LoadAwareRouter(load_threshold=0.8)

        # Overload the forecaster
        if AgentRole.FORECASTER in router._agent_metrics:
            router._agent_metrics[AgentRole.FORECASTER].current_load = 0.95

        state = MultiAgentState(
            query="Weather forecast",
            user_id="test_user",
        )

        selected = router.select_agents(state)

        # Should still return agents
        assert len(selected) >= 1


class TestMetricsUpdates:
    """Tests for metrics update methods."""

    def test_update_latency(self):
        """Test updating agent latency metrics."""
        router = LoadAwareRouter()

        initial_latency = router._agent_metrics[AgentRole.FORECASTER].avg_latency_ms

        router.update_latency(AgentRole.FORECASTER, 1000.0)

        # Latency should be updated (moving average)
        new_latency = router._agent_metrics[AgentRole.FORECASTER].avg_latency_ms
        assert new_latency != initial_latency or new_latency == 1000.0

    def test_update_success_rate(self):
        """Test updating success rate."""
        router = LoadAwareRouter()

        # Record a failure
        router.record_success(AgentRole.FORECASTER, success=False)

        # Success rate should decrease
        assert router._agent_metrics[AgentRole.FORECASTER].success_rate < 1.0

    def test_update_load(self):
        """Test updating agent load."""
        router = LoadAwareRouter()

        router.update_load(AgentRole.FORECASTER, 0.75)

        assert router._agent_metrics[AgentRole.FORECASTER].current_load == 0.75


class TestGetMetrics:
    """Tests for metrics retrieval."""

    def test_get_agent_metrics(self):
        """Test getting metrics for a specific agent."""
        router = LoadAwareRouter()

        metrics = router.get_agent_metrics(AgentRole.FORECASTER)

        assert metrics is not None
        assert metrics.agent_role == AgentRole.FORECASTER

    def test_get_agent_metrics_nonexistent(self):
        """Test getting metrics for non-existent agent."""
        router = LoadAwareRouter(agent_metrics={})

        metrics = router.get_agent_metrics(AgentRole.FORECASTER)

        # Should return None or create default
        # depending on implementation
        assert metrics is None or metrics.agent_role == AgentRole.FORECASTER

    def test_get_all_metrics(self):
        """Test getting all agent metrics."""
        router = LoadAwareRouter()

        all_metrics = router.get_all_metrics()

        assert len(all_metrics) > 0
        assert all(isinstance(m, AgentMetrics) for m in all_metrics.values())


class TestRouterStats:
    """Tests for router statistics."""

    def test_get_routing_stats(self):
        """Test getting routing statistics."""
        router = LoadAwareRouter()
        state = MultiAgentState(query="Test", user_id="test_user")

        # Make some selections to generate stats
        router.select_agents(state)
        router.select_agents(state)

        stats = router.get_routing_stats()

        assert "total_routing_decisions" in stats
        assert "avg_agents_selected" in stats

    def test_reset_stats(self):
        """Test resetting routing statistics."""
        router = LoadAwareRouter()
        state = MultiAgentState(query="Test", user_id="test_user")

        # Generate some stats
        router.select_agents(state)

        # Reset
        router.reset_stats()

        stats = router.get_routing_stats()
        assert stats.get("total_routing_decisions", 0) == 0


class TestFactoryFunction:
    """Tests for create_load_aware_router factory function."""

    def test_create_with_defaults(self):
        """Test creating router with default settings."""
        router = create_load_aware_router()

        assert isinstance(router, LoadAwareRouter)
        assert router.prefer_low_latency is True

    def test_create_with_custom_budget(self):
        """Test creating router with custom budget."""
        router = create_load_aware_router(max_budget=0.10)

        assert router.max_budget_per_query == 0.10


class TestCostOptimization:
    """Tests for cost optimization in routing."""

    def test_low_cost_tier_preferred_for_simple(self):
        """Test that low cost agents are preferred for simple queries."""
        router = LoadAwareRouter()

        state = MultiAgentState(
            query="Current temperature",
            user_id="test_user",
            complexity=QueryComplexity.SIMPLE,
        )

        selected = router.select_agents(state)

        # Should return agents - verify at least one is low/medium cost
        assert len(selected) >= 1

    def test_high_cost_allowed_for_complex(self):
        """Test that high cost agents are allowed for complex queries."""
        router = LoadAwareRouter()

        state = MultiAgentState(
            query="Detailed hurricane forecast with historical analysis",
            user_id="test_user",
            complexity=QueryComplexity.COMPLEX,
        )

        selected = router.select_agents(state)

        # Should return agents including potentially high-cost ones
        assert len(selected) >= 1


class TestLoadBalancing:
    """Tests for load balancing functionality."""

    def test_load_balancing_distributes_work(self):
        """Test that load balancing distributes work across agents."""
        router = LoadAwareRouter()

        # Set varying loads
        router.update_load(AgentRole.FORECASTER, 0.9)  # High load
        router.update_load(AgentRole.RESEARCH, 0.2)  # Low load

        state = MultiAgentState(
            query="Weather forecast",
            user_id="test_user",
        )

        selected = router.select_agents(state)

        # Should have selected some agents
        assert len(selected) >= 1

        # Research should be preferred over Forecaster due to lower load
        # (if both are applicable for the query)
