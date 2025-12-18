"""Tests for Level 4c Self-Healing Agent.

This module tests the self-healing capabilities including:
- Retry with exponential backoff
- Fallback agent routing
- Circuit breaker integration
- Health monitoring
- Graceful degradation
"""

import asyncio

import pytest

from backend.src.agents.circuit_breaker import CircuitState
from backend.src.agents.self_healing_agent import (
    DEFAULT_FALLBACK_MAP,
    SelfHealingAgent,
    create_self_healing_agent,
    self_healing_agent,
)
from backend.src.models.multi_agent import AgentResponse, AgentRole, MultiAgentState


class TestDefaultFallbackMap:
    """Tests for DEFAULT_FALLBACK_MAP configuration."""

    def test_hurricane_specialist_has_fallbacks(self):
        """Test that Hurricane Specialist has defined fallbacks."""
        assert AgentRole.HURRICANE_SPECIALIST in DEFAULT_FALLBACK_MAP
        fallbacks = DEFAULT_FALLBACK_MAP[AgentRole.HURRICANE_SPECIALIST]
        assert AgentRole.FORECASTER in fallbacks
        assert AgentRole.RESEARCH in fallbacks

    def test_forecaster_has_fallbacks(self):
        """Test that Forecaster has defined fallbacks."""
        assert AgentRole.FORECASTER in DEFAULT_FALLBACK_MAP
        fallbacks = DEFAULT_FALLBACK_MAP[AgentRole.FORECASTER]
        assert AgentRole.RESEARCH in fallbacks

    def test_emergency_response_has_fallbacks(self):
        """Test that Emergency Response has defined fallbacks."""
        assert AgentRole.EMERGENCY_RESPONSE in DEFAULT_FALLBACK_MAP
        fallbacks = DEFAULT_FALLBACK_MAP[AgentRole.EMERGENCY_RESPONSE]
        assert AgentRole.HURRICANE_SPECIALIST in fallbacks

    def test_climate_analyst_has_fallbacks(self):
        """Test that Climate Analyst has defined fallbacks."""
        assert AgentRole.CLIMATE_ANALYST in DEFAULT_FALLBACK_MAP
        fallbacks = DEFAULT_FALLBACK_MAP[AgentRole.CLIMATE_ANALYST]
        assert AgentRole.HISTORICAL_ANALYST in fallbacks


class TestSelfHealingAgentInit:
    """Tests for SelfHealingAgent initialization."""

    def test_default_initialization(self):
        """Test default initialization of SelfHealingAgent."""
        agent = SelfHealingAgent()

        assert agent.max_retries == 3
        assert agent.base_backoff == 1.0
        assert agent.agent_timeout == 30.0
        assert agent.fallback_map is not None

    def test_custom_initialization(self):
        """Test custom initialization of SelfHealingAgent."""
        agent = SelfHealingAgent(
            max_retries=5,
            base_backoff_seconds=2.0,
            agent_timeout=60.0,
        )

        assert agent.max_retries == 5
        assert agent.base_backoff == 2.0
        assert agent.agent_timeout == 60.0

    def test_custom_fallback_map(self):
        """Test initialization with custom fallback map."""
        custom_map = {
            AgentRole.FORECASTER: [AgentRole.RESEARCH],
        }

        agent = SelfHealingAgent(fallback_map=custom_map)

        assert agent.fallback_map == custom_map

    def test_initial_stats_are_zero(self):
        """Test that initial execution stats are zero."""
        agent = SelfHealingAgent()
        stats = agent.get_execution_stats()

        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["retry_executions"] == 0
        assert stats["fallback_executions"] == 0
        assert stats["failed_executions"] == 0


class TestExecuteWithHealing:
    """Tests for execute_with_healing method."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful execution without retries."""
        agent = SelfHealingAgent()
        state = MultiAgentState(
            query="Test query",
            user_id="test_user",
        )

        async def success_func(s):
            s.agent_responses.append(
                AgentResponse(
                    agent_role=AgentRole.FORECASTER,
                    content="Success response",
                    confidence=0.9,
                    execution_time_ms=100,
                )
            )
            return s

        result = await agent.execute_with_healing(
            agent_role=AgentRole.FORECASTER,
            agent_func=success_func,
            state=state,
        )

        assert len(result.agent_responses) == 1
        assert result.agent_responses[0].content == "Success response"

        stats = agent.get_execution_stats()
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 1
        assert stats["retry_executions"] == 0

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that failures trigger retries."""
        agent = SelfHealingAgent(
            max_retries=3,
            base_backoff_seconds=0.01,  # Fast for testing
            agent_timeout=5.0,
        )
        state = MultiAgentState(query="Test query", user_id="test_user")

        call_count = 0

        async def failing_then_success(s):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Simulated failure")
            s.agent_responses.append(
                AgentResponse(
                    agent_role=AgentRole.FORECASTER,
                    content="Success after retries",
                    confidence=0.9,
                    execution_time_ms=100,
                )
            )
            return s

        result = await agent.execute_with_healing(
            agent_role=AgentRole.FORECASTER,
            agent_func=failing_then_success,
            state=state,
        )

        assert call_count == 3  # Called 3 times (2 failures + 1 success)
        assert len(result.agent_responses) == 1
        assert result.agent_responses[0].content == "Success after retries"

    @pytest.mark.asyncio
    async def test_fallback_after_all_retries_fail(self):
        """Test fallback execution after all retries fail."""
        agent = SelfHealingAgent(
            max_retries=2,
            base_backoff_seconds=0.01,
            agent_timeout=5.0,
        )
        state = MultiAgentState(query="Test query", user_id="test_user")

        async def always_failing(s):
            raise RuntimeError("Always fails")

        result = await agent.execute_with_healing(
            agent_role=AgentRole.HURRICANE_SPECIALIST,
            agent_func=always_failing,
            state=state,
        )

        # Should have a fallback response
        assert len(result.agent_responses) >= 1
        last_response = result.agent_responses[-1]
        assert last_response.metadata.get("is_fallback") is True

        stats = agent.get_execution_stats()
        assert stats["fallback_executions"] >= 1

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self):
        """Test that timeouts trigger retries."""
        agent = SelfHealingAgent(
            max_retries=2,
            base_backoff_seconds=0.01,
            agent_timeout=0.1,  # Very short timeout
        )
        state = MultiAgentState(query="Test query", user_id="test_user")

        call_count = 0

        async def slow_then_fast(s):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                await asyncio.sleep(1.0)  # Will timeout
            s.agent_responses.append(
                AgentResponse(
                    agent_role=AgentRole.FORECASTER,
                    content="Fast response",
                    confidence=0.9,
                    execution_time_ms=50,
                )
            )
            return s

        result = await agent.execute_with_healing(
            agent_role=AgentRole.FORECASTER,
            agent_func=slow_then_fast,
            state=state,
        )

        assert call_count >= 2  # At least 2 calls (1 timeout + 1 success)


class TestFallbackExecution:
    """Tests for fallback agent execution."""

    @pytest.mark.asyncio
    async def test_fallback_with_open_circuit(self):
        """Test fallback execution when primary circuit is open."""
        # Create agent with fresh registry to avoid interference
        agent = SelfHealingAgent(max_retries=2, base_backoff_seconds=0.01)

        # Get the circuit and open it
        from backend.src.agents.circuit_breaker import circuit_registry

        circuit = circuit_registry.get_or_create(AgentRole.HURRICANE_SPECIALIST.value)
        circuit._state = CircuitState.OPEN

        state = MultiAgentState(query="Test query", user_id="test_user")

        async def should_not_be_called(s):
            raise AssertionError("Should not be called - circuit is open")

        result = await agent.execute_with_healing(
            agent_role=AgentRole.HURRICANE_SPECIALIST,
            agent_func=should_not_be_called,
            state=state,
        )

        # Should use fallback
        assert len(result.agent_responses) >= 1

        # Reset circuit for other tests
        circuit.reset()

    @pytest.mark.asyncio
    async def test_no_fallback_available(self):
        """Test behavior when no fallbacks are available."""
        custom_map = {}  # Empty fallback map

        agent = SelfHealingAgent(
            max_retries=1,
            base_backoff_seconds=0.01,
            fallback_map=custom_map,
        )
        state = MultiAgentState(query="Test query", user_id="test_user")

        async def always_failing(s):
            raise RuntimeError("Always fails")

        result = await agent.execute_with_healing(
            agent_role=AgentRole.FORECASTER,
            agent_func=always_failing,
            state=state,
        )

        # Should have error state
        assert result.error is not None or any(
            r.metadata.get("is_failure") for r in result.agent_responses
        )


class TestSelfHealingProcess:
    """Tests for the process method (health check)."""

    @pytest.mark.asyncio
    async def test_process_returns_health_status(self):
        """Test that process returns system health status."""
        agent = SelfHealingAgent()
        state = MultiAgentState(query="Health check", user_id="test_user")

        result = await agent.process(state)

        assert result.current_agent == AgentRole.SELF_HEALING
        assert len(result.agent_responses) >= 1

        health_response = result.agent_responses[-1]
        assert health_response.agent_role == AgentRole.SELF_HEALING
        assert "health" in health_response.content.lower()

    @pytest.mark.asyncio
    async def test_process_stores_health_in_memory_context(self):
        """Test that health data is stored in memory context."""
        agent = SelfHealingAgent()
        state = MultiAgentState(query="Health check", user_id="test_user")

        result = await agent.process(state)

        assert "system_health" in result.memory_context
        assert "healing_stats" in result.memory_context


class TestExecutionStats:
    """Tests for execution statistics."""

    def test_get_execution_stats_with_data(self):
        """Test execution stats with some data."""
        agent = SelfHealingAgent()

        # Manually modify stats
        agent._execution_stats["total_executions"] = 100
        agent._execution_stats["successful_executions"] = 90
        agent._execution_stats["retry_executions"] = 5
        agent._execution_stats["fallback_executions"] = 3
        agent._execution_stats["failed_executions"] = 2

        stats = agent.get_execution_stats()

        assert stats["success_rate"] == 0.9
        assert stats["retry_rate"] == 0.05
        assert stats["fallback_rate"] == 0.03
        assert stats["failure_rate"] == 0.02

    def test_get_execution_stats_empty(self):
        """Test execution stats when no executions have occurred."""
        agent = SelfHealingAgent()
        stats = agent.get_execution_stats()

        assert stats["success_rate"] == 0.0
        assert stats["retry_rate"] == 0.0
        assert stats["fallback_rate"] == 0.0
        assert stats["failure_rate"] == 0.0

    def test_reset_stats(self):
        """Test resetting execution statistics."""
        agent = SelfHealingAgent()

        # Add some stats
        agent._execution_stats["total_executions"] = 50
        agent._execution_stats["successful_executions"] = 40

        # Reset
        agent.reset_stats()

        stats = agent.get_execution_stats()
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0


class TestFactoryFunction:
    """Tests for create_self_healing_agent factory function."""

    def test_create_with_defaults(self):
        """Test creating agent with default settings."""
        agent = create_self_healing_agent()

        assert isinstance(agent, SelfHealingAgent)
        assert agent.max_retries == 3
        assert agent.base_backoff == 1.0
        assert agent.agent_timeout == 30.0

    def test_create_with_custom_settings(self):
        """Test creating agent with custom settings."""
        agent = create_self_healing_agent(
            max_retries=5,
            base_backoff=0.5,
            timeout=60.0,
        )

        assert agent.max_retries == 5
        assert agent.base_backoff == 0.5
        assert agent.agent_timeout == 60.0


class TestGlobalInstance:
    """Tests for the global self_healing_agent instance."""

    def test_global_instance_exists(self):
        """Test that the global instance exists."""
        assert self_healing_agent is not None
        assert isinstance(self_healing_agent, SelfHealingAgent)


class TestSystemHealthIntegration:
    """Integration tests for system health reporting."""

    @pytest.mark.asyncio
    async def test_get_system_health_includes_circuit_info(self):
        """Test that system health includes circuit breaker information."""
        agent = SelfHealingAgent()

        health = agent.get_system_health()

        assert "overall_status" in health
        assert "total_agents" in health
        assert "healthy_agents" in health
        assert "unhealthy_agents" in health
        assert "execution_stats" in health
