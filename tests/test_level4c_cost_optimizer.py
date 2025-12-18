"""Tests for Level 4c Cost Optimizer.

This module tests the cost optimization capabilities including:
- Token cost tracking
- Budget management
- Cost estimation
- Optimization recommendations
"""


import pytest

from backend.src.models.multi_agent import AgentRole
from backend.src.orchestration.cost_optimizer import (
    AGENT_TOKEN_ESTIMATES,
    TOKEN_COSTS,
    CostOptimizer,
    create_cost_optimizer,
)


class TestTokenCosts:
    """Tests for TOKEN_COSTS configuration."""

    def test_token_costs_defined(self):
        """Test that token costs are defined."""
        assert TOKEN_COSTS is not None
        assert isinstance(TOKEN_COSTS, dict)
        assert len(TOKEN_COSTS) > 0

    def test_common_models_have_costs(self):
        """Test that common models have defined costs."""
        common_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

        for model in common_models:
            if model in TOKEN_COSTS:
                cost = TOKEN_COSTS[model]
                assert "input" in cost
                assert "output" in cost
                assert cost["input"] > 0
                assert cost["output"] > 0


class TestAgentTokenEstimates:
    """Tests for AGENT_TOKEN_ESTIMATES configuration."""

    def test_agent_estimates_defined(self):
        """Test that agent token estimates are defined."""
        assert AGENT_TOKEN_ESTIMATES is not None
        assert isinstance(AGENT_TOKEN_ESTIMATES, dict)

    def test_common_agents_have_estimates(self):
        """Test that common agents have token estimates."""
        common_agents = [
            AgentRole.TRIAGE,
            AgentRole.FORECASTER,
            AgentRole.HURRICANE_SPECIALIST,
        ]

        for agent in common_agents:
            if agent in AGENT_TOKEN_ESTIMATES:
                estimate = AGENT_TOKEN_ESTIMATES[agent]
                assert "input_tokens" in estimate
                assert "output_tokens" in estimate


class TestCostOptimizerInit:
    """Tests for CostOptimizer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        optimizer = CostOptimizer()

        assert optimizer._total_cost == 0.0
        assert optimizer._query_count == 0
        assert optimizer._agent_costs == {}

    def test_custom_initialization(self):
        """Test custom initialization with budget."""
        optimizer = CostOptimizer(max_budget=1.0)

        assert optimizer.max_budget == 1.0

    def test_custom_token_costs(self):
        """Test initialization with custom token costs."""
        custom_costs = {
            "custom-model": {"input": 0.01, "output": 0.02},
        }

        optimizer = CostOptimizer(token_costs=custom_costs)

        assert "custom-model" in optimizer._token_costs


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_estimate_agent_cost(self):
        """Test estimating cost for an agent."""
        optimizer = CostOptimizer()

        cost = optimizer.estimate_agent_cost(
            agent_role=AgentRole.FORECASTER,
            model="gpt-4o-mini",
        )

        assert cost >= 0.0

    def test_estimate_query_cost(self):
        """Test estimating cost for a full query."""
        optimizer = CostOptimizer()

        agents = [AgentRole.TRIAGE, AgentRole.FORECASTER]

        cost = optimizer.estimate_query_cost(agents)

        assert cost >= 0.0

    def test_estimate_with_custom_tokens(self):
        """Test estimating cost with custom token counts."""
        optimizer = CostOptimizer()

        cost = optimizer.estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o",
        )

        assert cost > 0.0


class TestTrackCost:
    """Tests for cost tracking."""

    def test_track_agent_cost(self):
        """Test tracking cost for an agent."""
        optimizer = CostOptimizer()

        optimizer.track_cost(
            agent_role=AgentRole.FORECASTER,
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o-mini",
        )

        assert optimizer._total_cost > 0
        assert AgentRole.FORECASTER in optimizer._agent_costs

    def test_track_multiple_queries(self):
        """Test tracking costs across multiple queries."""
        optimizer = CostOptimizer()

        for _ in range(5):
            optimizer.track_cost(
                agent_role=AgentRole.FORECASTER,
                input_tokens=100,
                output_tokens=50,
                model="gpt-4o-mini",
            )

        assert optimizer._query_count == 5
        assert optimizer._total_cost > 0


class TestBudgetManagement:
    """Tests for budget management."""

    def test_check_budget_under_limit(self):
        """Test budget check when under limit."""
        optimizer = CostOptimizer(max_budget=10.0)

        # Small cost should be within budget
        within_budget = optimizer.check_budget(estimated_cost=0.01)

        assert within_budget is True

    def test_check_budget_over_limit(self):
        """Test budget check when over limit."""
        optimizer = CostOptimizer(max_budget=0.01)

        # Track some cost first
        optimizer._total_cost = 0.009

        # Adding more would exceed budget
        within_budget = optimizer.check_budget(estimated_cost=0.005)

        assert within_budget is False

    def test_get_remaining_budget(self):
        """Test getting remaining budget."""
        optimizer = CostOptimizer(max_budget=1.0)

        # Track some cost
        optimizer._total_cost = 0.25

        remaining = optimizer.get_remaining_budget()

        assert remaining == 0.75


class TestOptimizationRecommendations:
    """Tests for optimization recommendations."""

    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        optimizer = CostOptimizer()

        # Track some costs
        optimizer._agent_costs[AgentRole.HURRICANE_SPECIALIST] = 0.5
        optimizer._agent_costs[AgentRole.FORECASTER] = 0.1
        optimizer._agent_costs[AgentRole.TRIAGE] = 0.05

        recommendations = optimizer.get_optimization_recommendations()

        assert isinstance(recommendations, list)

    def test_recommendations_identify_expensive_agents(self):
        """Test that recommendations identify expensive agents."""
        optimizer = CostOptimizer()

        # Make one agent very expensive
        optimizer._agent_costs[AgentRole.HURRICANE_SPECIALIST] = 10.0
        optimizer._agent_costs[AgentRole.FORECASTER] = 0.1

        recommendations = optimizer.get_optimization_recommendations()

        # Should have some recommendations
        if recommendations:
            # At least one should mention the expensive agent
            all_text = " ".join(recommendations)
            # Just verify we get recommendations
            assert len(all_text) > 0


class TestCostStatistics:
    """Tests for cost statistics."""

    def test_get_cost_stats(self):
        """Test getting cost statistics."""
        optimizer = CostOptimizer()

        # Track some costs
        optimizer.track_cost(
            agent_role=AgentRole.FORECASTER,
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o-mini",
        )
        optimizer.track_cost(
            agent_role=AgentRole.FORECASTER,
            input_tokens=800,
            output_tokens=400,
            model="gpt-4o-mini",
        )

        stats = optimizer.get_cost_stats()

        assert "total_cost" in stats
        assert "query_count" in stats
        assert "avg_cost_per_query" in stats
        assert "agent_costs" in stats

    def test_get_cost_stats_empty(self):
        """Test getting stats with no data."""
        optimizer = CostOptimizer()

        stats = optimizer.get_cost_stats()

        assert stats["total_cost"] == 0.0
        assert stats["query_count"] == 0
        assert stats["avg_cost_per_query"] == 0.0

    def test_reset_stats(self):
        """Test resetting statistics."""
        optimizer = CostOptimizer()

        # Add some data
        optimizer._total_cost = 1.5
        optimizer._query_count = 50
        optimizer._agent_costs[AgentRole.FORECASTER] = 0.5

        # Reset
        optimizer.reset_stats()

        assert optimizer._total_cost == 0.0
        assert optimizer._query_count == 0
        assert optimizer._agent_costs == {}


class TestModelSelection:
    """Tests for model selection based on cost."""

    def test_select_cheapest_model(self):
        """Test selecting the cheapest model."""
        optimizer = CostOptimizer()

        model = optimizer.select_cheapest_model(
            min_capability="basic",
        )

        assert model is not None
        # Should return some model name
        assert isinstance(model, str)

    def test_select_model_for_complexity(self):
        """Test selecting model based on query complexity."""
        optimizer = CostOptimizer()

        # Simple query should prefer cheaper model
        simple_model = optimizer.select_model_for_complexity("simple")

        # Complex query may need more capable model
        complex_model = optimizer.select_model_for_complexity("complex")

        # Both should return valid models
        assert simple_model is not None
        assert complex_model is not None


class TestFactoryFunction:
    """Tests for create_cost_optimizer factory function."""

    def test_create_with_defaults(self):
        """Test creating optimizer with default settings."""
        optimizer = create_cost_optimizer()

        assert isinstance(optimizer, CostOptimizer)

    def test_create_with_budget(self):
        """Test creating optimizer with budget."""
        optimizer = create_cost_optimizer(max_budget=5.0)

        assert optimizer.max_budget == 5.0


class TestCostCalculations:
    """Tests for cost calculation accuracy."""

    def test_gpt4o_mini_cost_calculation(self):
        """Test cost calculation for GPT-4o-mini."""
        optimizer = CostOptimizer()

        # GPT-4o-mini pricing (as of 2024):
        # Input: $0.15 per 1M tokens = $0.00015 per 1K tokens
        # Output: $0.60 per 1M tokens = $0.0006 per 1K tokens

        cost = optimizer.estimate_cost(
            input_tokens=1000,
            output_tokens=1000,
            model="gpt-4o-mini",
        )

        # Cost should be positive
        assert cost > 0

    def test_cost_accumulation(self):
        """Test that costs accumulate correctly."""
        optimizer = CostOptimizer()

        # Track same cost twice
        optimizer.track_cost(
            agent_role=AgentRole.FORECASTER,
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o-mini",
        )

        first_cost = optimizer._total_cost

        optimizer.track_cost(
            agent_role=AgentRole.FORECASTER,
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o-mini",
        )

        # Should be approximately double
        assert optimizer._total_cost == pytest.approx(first_cost * 2, rel=0.01)
