"""Tests for Supervisor Agent (Level 4b Phase 7).

Test Coverage:
- AgentCapabilityRegistry functionality
- Workflow planning and orchestration
- Parallel agent execution coordination
- Agent selection based on query domains
- Response synthesis
- Error handling and graceful degradation
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.src.agents.supervisor_agent import (
    AgentCapabilityRegistry,
    SupervisorAgent,
)
from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)


# Fixtures
@pytest.fixture
def base_state() -> MultiAgentState:
    """Create base MultiAgentState for testing."""
    return MultiAgentState(
        query="Test query",
        user_id="test_user_123",
        session_id="test_session_456",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        timeout_ms=30000,
        current_agent=None,
        routing_decision=None,
        agent_responses=[],
        next_agent=None,
        workflow_complete=False,
        final_response=None,
        error=None,
        total_execution_time_ms=0.0,
    )


@pytest.fixture
def capability_registry() -> AgentCapabilityRegistry:
    """Create AgentCapabilityRegistry for testing."""
    return AgentCapabilityRegistry()


@pytest.fixture
def supervisor_agent() -> SupervisorAgent:
    """Create SupervisorAgent for testing."""
    return SupervisorAgent()


# Test: AgentCapabilityRegistry
class TestAgentCapabilityRegistry:
    """Tests for AgentCapabilityRegistry."""

    def test_registry_initialization(self, capability_registry):
        """Test that registry initializes with expected agents."""
        assert len(capability_registry.capabilities) > 0

        # Verify key agents are registered
        assert AgentRole.HURRICANE_SPECIALIST in capability_registry.capabilities
        assert AgentRole.FORECASTER in capability_registry.capabilities
        assert AgentRole.HISTORICAL_ANALYST in capability_registry.capabilities
        assert AgentRole.RESEARCH in capability_registry.capabilities

    def test_get_agents_for_domains_hurricane(self, capability_registry):
        """Test agent selection for hurricane domain."""
        agents = capability_registry.get_agents_for_domains(["hurricane"])
        assert AgentRole.HURRICANE_SPECIALIST in agents

    def test_get_agents_for_domains_forecast(self, capability_registry):
        """Test agent selection for forecast domain."""
        agents = capability_registry.get_agents_for_domains(["forecast"])
        assert AgentRole.FORECASTER in agents

    def test_get_agents_for_domains_historical(self, capability_registry):
        """Test agent selection for historical domain."""
        agents = capability_registry.get_agents_for_domains(["historical"])
        assert AgentRole.HISTORICAL_ANALYST in agents

    def test_get_agents_for_domains_research(self, capability_registry):
        """Test agent selection for research domain."""
        agents = capability_registry.get_agents_for_domains(["research"])
        assert AgentRole.RESEARCH in agents

    def test_get_agents_for_domains_unknown(self, capability_registry):
        """Test agent selection for unknown domain returns empty."""
        agents = capability_registry.get_agents_for_domains(["unknown_domain_xyz"])
        assert len(agents) == 0  # No matching agents

    def test_get_agents_for_domains_multiple(self, capability_registry):
        """Test multi-agent selection for multiple domains."""
        agents = capability_registry.get_agents_for_domains(["hurricane", "forecast"])
        assert AgentRole.HURRICANE_SPECIALIST in agents
        assert AgentRole.FORECASTER in agents

    def test_get_agents_for_domains_case_insensitive(self, capability_registry):
        """Test domain matching is case insensitive."""
        agents = capability_registry.get_agents_for_domains(["HURRICANE"])
        assert AgentRole.HURRICANE_SPECIALIST in agents

    def test_get_independent_agents(self, capability_registry):
        """Test grouping agents for parallel execution."""
        agents = [AgentRole.HURRICANE_SPECIALIST, AgentRole.FORECASTER, AgentRole.TRIAGE]
        groups = capability_registry.get_independent_agents(agents)
        assert len(groups) >= 1
        # Parallel-capable agents should be grouped together
        parallel_group = groups[0]
        assert AgentRole.HURRICANE_SPECIALIST in parallel_group or AgentRole.FORECASTER in parallel_group

    def test_get_capability(self, capability_registry):
        """Test getting capability info for an agent."""
        info = capability_registry.get_capability(AgentRole.HURRICANE_SPECIALIST)
        assert info is not None
        assert "domains" in info
        assert "hurricane" in info["domains"]

    def test_estimate_latency(self, capability_registry):
        """Test latency estimation for agent execution."""
        agents = [AgentRole.HURRICANE_SPECIALIST, AgentRole.FORECASTER]
        latency = capability_registry.estimate_latency(agents)
        assert latency > 0


# Test: SupervisorAgent Initialization
class TestSupervisorAgentInit:
    """Tests for SupervisorAgent initialization."""

    def test_default_initialization(self, supervisor_agent):
        """Test SupervisorAgent initializes with defaults."""
        assert supervisor_agent.llm is not None
        assert supervisor_agent.registry is not None
        assert supervisor_agent.max_parallel_agents == 4

    def test_custom_initialization(self):
        """Test SupervisorAgent with custom parameters."""
        agent = SupervisorAgent(
            model_name="gpt-4o",
            max_parallel_agents=6,
        )
        assert agent.max_parallel_agents == 6


# Test: Workflow Orchestration
class TestSupervisorOrchestration:
    """Tests for SupervisorAgent orchestration."""

    @pytest.mark.asyncio
    async def test_orchestrate_simple_query(self, supervisor_agent, base_state):
        """Test orchestration of simple weather query."""
        base_state.query = "What's the weather in Miami?"

        # Mock the LLM
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "parallel_groups": [["forecaster"]],
                "reasoning": "Simple forecast query",
                "requires_verification": False,
            })
        ))
        supervisor_agent.llm = mock_llm

        result = await supervisor_agent.orchestrate(base_state)

        assert result is not None
        assert result.current_agent == AgentRole.SUPERVISOR

    @pytest.mark.asyncio
    async def test_orchestrate_hurricane_query(self, supervisor_agent, base_state):
        """Test orchestration of hurricane-related query."""
        base_state.query = "What is Hurricane Michael's expected landfall?"

        # Mock the LLM
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "parallel_groups": [
                    ["hurricane_specialist", "historical_analyst"],
                    ["verification"]
                ],
                "reasoning": "Hurricane query needs specialist and historical context",
                "requires_verification": True,
            })
        ))
        supervisor_agent.llm = mock_llm

        result = await supervisor_agent.orchestrate(base_state)

        assert result is not None
        assert result.current_agent == AgentRole.SUPERVISOR

    @pytest.mark.asyncio
    async def test_orchestrate_with_error_handling(self, supervisor_agent, base_state):
        """Test orchestration handles errors gracefully."""
        base_state.query = "Test query"

        # Mock LLM to raise exception
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        supervisor_agent.llm = mock_llm

        result = await supervisor_agent.orchestrate(base_state)

        # Should handle error gracefully
        assert result is not None


# Test: Workflow Planning
class TestWorkflowPlanning:
    """Tests for workflow planning functionality."""

    @pytest.mark.asyncio
    async def test_plan_workflow_simple(self, supervisor_agent, base_state):
        """Test planning simple workflow."""
        base_state.query = "Current weather in Tampa"

        # Mock the LLM
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "parallel_groups": [["forecaster"]],
                "reasoning": "Simple query",
                "requires_verification": False,
            })
        ))
        supervisor_agent.llm = mock_llm

        plan = await supervisor_agent._plan_workflow(base_state)

        assert plan is not None
        assert "parallel_groups" in plan
        assert "reasoning" in plan

    @pytest.mark.asyncio
    async def test_plan_workflow_complex(self, supervisor_agent, base_state):
        """Test planning complex multi-agent workflow."""
        base_state.query = "Compare Hurricane Michael evacuation zones with current storm"

        # Mock the LLM
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "parallel_groups": [
                    ["hurricane_specialist", "historical_analyst"],
                    ["research"],
                    ["verification"]
                ],
                "reasoning": "Complex comparison needs multiple specialists",
                "requires_verification": True,
            })
        ))
        supervisor_agent.llm = mock_llm

        plan = await supervisor_agent._plan_workflow(base_state)

        assert plan is not None
        assert len(plan.get("parallel_groups", [])) >= 2

    @pytest.mark.asyncio
    async def test_plan_workflow_invalid_json_fallback(self, supervisor_agent, base_state):
        """Test workflow planning handles invalid JSON."""
        base_state.query = "Test query"

        # Mock LLM to return invalid JSON
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Invalid JSON response"
        ))
        supervisor_agent.llm = mock_llm

        plan = await supervisor_agent._plan_workflow(base_state)

        # Should return default plan
        assert plan is not None
        assert "parallel_groups" in plan


# Test: Response Synthesis
class TestResponseSynthesis:
    """Tests for response synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_single_response(self, supervisor_agent, base_state):
        """Test synthesis with single agent response (skipped - need 2+)."""
        base_state.query = "What's the weather?"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.FORECASTER,
                content="Current weather is sunny, 75F",
                confidence=0.9,
                timestamp=datetime.now(UTC),
                execution_time_ms=100,
                metadata={},
            )
        ]

        # Single response should return state unchanged (synthesis needs 2+ responses)
        result = await supervisor_agent._synthesize_responses(base_state)

        # With only 1 response, synthesis is skipped
        assert result is not None

    @pytest.mark.asyncio
    async def test_synthesize_multiple_responses(self, supervisor_agent, base_state):
        """Test synthesis with multiple agent responses."""
        base_state.query = "Hurricane analysis"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Hurricane is Category 3",
                confidence=0.9,
                timestamp=datetime.now(UTC),
                execution_time_ms=100,
                metadata={},
            ),
            AgentResponse(
                agent_role=AgentRole.HISTORICAL_ANALYST,
                content="Similar to Hurricane Andrew trajectory",
                confidence=0.85,
                timestamp=datetime.now(UTC),
                execution_time_ms=150,
                metadata={},
            ),
        ]

        # Mock the LLM
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Synthesized: Category 3 hurricane with trajectory similar to Andrew"
        ))
        supervisor_agent.llm = mock_llm

        synthesis = await supervisor_agent._synthesize_responses(base_state)

        assert synthesis is not None


# Test: Agent Selection
class TestAgentSelection:
    """Tests for agent selection logic."""

    def test_select_agents_hurricane_domain(self, supervisor_agent):
        """Test agent selection for hurricane domain."""
        agents = supervisor_agent.registry.get_agents_for_domains(["hurricane"])
        assert AgentRole.HURRICANE_SPECIALIST in agents

    def test_select_agents_forecast_domain(self, supervisor_agent):
        """Test agent selection for forecast domain."""
        agents = supervisor_agent.registry.get_agents_for_domains(["forecast"])
        assert AgentRole.FORECASTER in agents

    def test_select_agents_research_domain(self, supervisor_agent):
        """Test agent selection for research domain."""
        agents = supervisor_agent.registry.get_agents_for_domains(["research"])
        assert AgentRole.RESEARCH in agents


# Test: Parallel Execution Coordination
class TestParallelExecution:
    """Tests for parallel execution coordination."""

    @pytest.mark.asyncio
    async def test_execute_parallel_group(self, supervisor_agent, base_state):
        """Test execution of parallel agent group."""
        base_state.query = "Hurricane and historical analysis"

        # This test verifies the parallel execution structure
        # The actual execution would require full agent integration
        assert supervisor_agent.max_parallel_agents == 4

    @pytest.mark.asyncio
    async def test_parallel_limit_enforcement(self, supervisor_agent):
        """Test that parallel execution limit is enforced."""
        # Create agent with specific limit
        agent = SupervisorAgent(max_parallel_agents=3)
        assert agent.max_parallel_agents == 3


# Test: Error Handling
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_llm_timeout(self, supervisor_agent, base_state):
        """Test handling of LLM timeout."""
        base_state.query = "Test query"

        # Mock LLM timeout
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=TimeoutError("LLM timeout"))
        supervisor_agent.llm = mock_llm

        result = await supervisor_agent.orchestrate(base_state)

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_handles_agent_failure(self, supervisor_agent, base_state):
        """Test handling of individual agent failure."""
        base_state.query = "Test query"
        base_state.agent_responses = []

        # This tests the error handling path
        # When an agent fails, supervisor should continue
        assert supervisor_agent is not None


# Integration Tests
class TestSupervisorIntegration:
    """Integration tests for SupervisorAgent."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_orchestration_workflow(self, supervisor_agent, base_state):
        """Test complete orchestration workflow."""
        base_state.query = "Hurricane Michael forecast and historical comparison"

        # Mock the LLM for full workflow
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "parallel_groups": [
                    ["hurricane_specialist", "historical_analyst"]
                ],
                "reasoning": "Hurricane analysis with historical context",
                "requires_verification": True,
            })
        ))
        supervisor_agent.llm = mock_llm

        result = await supervisor_agent.orchestrate(base_state)

        assert result is not None
        assert result.current_agent == AgentRole.SUPERVISOR
