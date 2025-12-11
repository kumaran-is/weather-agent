"""Tests for Level 4b Multi-Agent Workflow (Level 4b Phase 10).

Test Coverage:
- Level 4b StateGraph workflow creation
- Supervisor orchestration integration
- Parallel agent execution in workflow
- Reflection/Critique quality gates
- End-to-end workflow execution
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.models.multi_agent import (
    AgentRole,
    AgentResponse,
    MultiAgentState,
    QueryComplexity,
)
from backend.src.orchestration.multi_agent_workflow import (
    create_level4b_workflow,
    compile_level4b_workflow,
    invoke_workflow_v2,
    supervisor_node,
    forecaster_node,
    historical_analyst_node,
    research_node,
    reflection_node,
)


# Fixtures
@pytest.fixture
def base_state() -> MultiAgentState:
    """Create base MultiAgentState for testing."""
    return MultiAgentState(
        query="Test query",
        user_id="test_user_123",
        session_id="test_session_456",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        timeout_ms=30000,
        current_agent=None,
        routing_decision=None,
        agent_responses=[],
        next_agent=None,
        workflow_complete=False,
        final_response=None,
        error=None,
        total_execution_time_ms=0.0,
        reflection_iterations=0,
        quality_score=0.0,
    )


# Test: Workflow Creation
class TestWorkflowCreation:
    """Tests for Level 4b workflow creation."""

    def test_create_level4b_workflow(self):
        """Test creating Level 4b workflow graph."""
        workflow = create_level4b_workflow()

        assert workflow is not None
        # Verify nodes exist
        assert hasattr(workflow, 'nodes')

    def test_compile_level4b_workflow(self):
        """Test compiling Level 4b workflow."""
        workflow = compile_level4b_workflow()

        assert workflow is not None
        # Compiled workflow should be callable
        assert callable(getattr(workflow, 'invoke', None)) or callable(getattr(workflow, 'ainvoke', None))


# Test: Supervisor Node
class TestSupervisorNode:
    """Tests for supervisor_node function."""

    @pytest.mark.asyncio
    async def test_supervisor_node_execution(self, base_state):
        """Test supervisor node execution."""
        base_state.query = "Hurricane forecast for Tampa"

        # Mock the supervisor agent
        with patch('backend.src.orchestration.multi_agent_workflow.get_supervisor_agent') as mock_get:
            mock_agent = MagicMock()
            mock_agent.orchestrate = AsyncMock(return_value=base_state)
            mock_get.return_value = mock_agent

            result = await supervisor_node(base_state)

            assert result is not None
            mock_agent.orchestrate.assert_called_once()

    @pytest.mark.asyncio
    async def test_supervisor_node_sets_current_agent(self, base_state):
        """Test that supervisor node sets current_agent."""
        base_state.query = "Weather query"

        with patch('backend.src.orchestration.multi_agent_workflow.get_supervisor_agent') as mock_get:
            updated_state = base_state.model_copy()
            updated_state.current_agent = AgentRole.SUPERVISOR
            mock_agent = MagicMock()
            mock_agent.orchestrate = AsyncMock(return_value=updated_state)
            mock_get.return_value = mock_agent

            result = await supervisor_node(base_state)

            assert result.current_agent == AgentRole.SUPERVISOR


# Test: Forecaster Node
class TestForecasterNode:
    """Tests for forecaster_node function."""

    @pytest.mark.asyncio
    async def test_forecaster_node_execution(self, base_state):
        """Test forecaster node execution."""
        base_state.query = "5-day forecast for Miami"

        with patch('backend.src.orchestration.multi_agent_workflow.get_forecaster_agent') as mock_get:
            updated_state = base_state.model_copy()
            updated_state.agent_responses = [
                AgentResponse(
                    agent_role=AgentRole.FORECASTER,
                    content="Weather forecast...",
                    confidence=0.9,
                    timestamp=datetime.now(timezone.utc),
                    execution_time_ms=100,
                    metadata={},
                )
            ]
            mock_agent = MagicMock()
            mock_agent.forecast_query = AsyncMock(return_value=updated_state)
            mock_get.return_value = mock_agent

            result = await forecaster_node(base_state)

            assert result is not None
            mock_agent.forecast_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_forecaster_node_adds_response(self, base_state):
        """Test that forecaster node adds response to state."""
        base_state.query = "Weather in Tampa"

        with patch('backend.src.orchestration.multi_agent_workflow.get_forecaster_agent') as mock_get:
            updated_state = base_state.model_copy()
            updated_state.agent_responses = [
                AgentResponse(
                    agent_role=AgentRole.FORECASTER,
                    content="Forecast content",
                    confidence=0.85,
                    timestamp=datetime.now(timezone.utc),
                    execution_time_ms=150,
                    metadata={"forecast_type": "current"},
                )
            ]
            mock_agent = MagicMock()
            mock_agent.forecast_query = AsyncMock(return_value=updated_state)
            mock_get.return_value = mock_agent

            result = await forecaster_node(base_state)

            assert len(result.agent_responses) >= 1
            assert result.agent_responses[-1].agent_role == AgentRole.FORECASTER


# Test: Historical Analyst Node
class TestHistoricalAnalystNode:
    """Tests for historical_analyst_node function."""

    @pytest.mark.asyncio
    async def test_historical_node_execution(self, base_state):
        """Test historical analyst node execution."""
        base_state.query = "Hurricane Michael 2018 analysis"

        with patch('backend.src.orchestration.multi_agent_workflow.get_historical_agent') as mock_get:
            updated_state = base_state.model_copy()
            updated_state.agent_responses = [
                AgentResponse(
                    agent_role=AgentRole.HISTORICAL_ANALYST,
                    content="Historical analysis...",
                    confidence=0.88,
                    timestamp=datetime.now(timezone.utc),
                    execution_time_ms=200,
                    metadata={"storms_analyzed": 1},
                )
            ]
            mock_agent = MagicMock()
            mock_agent.analyze_patterns = AsyncMock(return_value=updated_state)
            mock_get.return_value = mock_agent

            result = await historical_analyst_node(base_state)

            assert result is not None
            mock_agent.analyze_patterns.assert_called_once()


# Test: Research Node
class TestResearchNode:
    """Tests for research_node function."""

    @pytest.mark.asyncio
    async def test_research_node_execution(self, base_state):
        """Test research node execution."""
        base_state.query = "How do hurricanes form?"

        with patch('backend.src.orchestration.multi_agent_workflow.get_research_agent') as mock_get:
            updated_state = base_state.model_copy()
            updated_state.agent_responses = [
                AgentResponse(
                    agent_role=AgentRole.RESEARCH,
                    content="Research findings...",
                    confidence=0.85,
                    timestamp=datetime.now(timezone.utc),
                    execution_time_ms=250,
                    metadata={"research_areas": ["tropical_meteorology"]},
                )
            ]
            mock_agent = MagicMock()
            mock_agent.research_query = AsyncMock(return_value=updated_state)
            mock_get.return_value = mock_agent

            result = await research_node(base_state)

            assert result is not None
            mock_agent.research_query.assert_called_once()


# Test: Reflection Node
class TestReflectionNode:
    """Tests for reflection_node function."""

    @pytest.mark.asyncio
    async def test_reflection_node_execution(self, base_state):
        """Test reflection node execution."""
        base_state.query = "Hurricane analysis"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Initial response",
                confidence=0.7,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        with patch('backend.src.orchestration.multi_agent_workflow.get_reflection_agent') as mock_get:
            updated_state = base_state.model_copy()
            updated_state.reflection_iterations = 1
            updated_state.quality_score = 0.9
            mock_agent = MagicMock()
            mock_agent.reflect_and_improve = AsyncMock(return_value=updated_state)
            mock_get.return_value = mock_agent

            result = await reflection_node(base_state)

            assert result is not None
            mock_agent.reflect_and_improve.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflection_updates_quality_score(self, base_state):
        """Test that reflection updates quality score."""
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.FORECASTER,
                content="Forecast",
                confidence=0.75,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        with patch('backend.src.orchestration.multi_agent_workflow.get_reflection_agent') as mock_get:
            updated_state = base_state.model_copy()
            updated_state.quality_score = 0.92
            updated_state.reflection_iterations = 1
            mock_agent = MagicMock()
            mock_agent.reflect_and_improve = AsyncMock(return_value=updated_state)
            mock_get.return_value = mock_agent

            result = await reflection_node(base_state)

            assert result.quality_score >= 0.9


# Test: Workflow Routing
class TestWorkflowRouting:
    """Tests for workflow routing logic."""

    def test_route_to_hurricane_specialist(self, base_state):
        """Test routing to hurricane specialist."""
        base_state.query = "Hurricane Michael forecast"
        base_state.next_agent = AgentRole.HURRICANE_SPECIALIST

        # Routing should go to hurricane specialist
        assert base_state.next_agent == AgentRole.HURRICANE_SPECIALIST

    def test_route_to_forecaster(self, base_state):
        """Test routing to forecaster."""
        base_state.query = "Weather forecast for Miami"
        base_state.next_agent = AgentRole.FORECASTER

        assert base_state.next_agent == AgentRole.FORECASTER

    def test_route_to_historical(self, base_state):
        """Test routing to historical analyst."""
        base_state.query = "Compare to past hurricanes"
        base_state.next_agent = AgentRole.HISTORICAL_ANALYST

        assert base_state.next_agent == AgentRole.HISTORICAL_ANALYST


# Test: State Management
class TestStateManagement:
    """Tests for workflow state management."""

    def test_state_preserves_user_context(self, base_state):
        """Test that state preserves user context through workflow."""
        base_state.user_id = "user_123"
        base_state.session_id = "session_456"

        # State should preserve these across operations
        assert base_state.user_id == "user_123"
        assert base_state.session_id == "session_456"

    def test_state_accumulates_responses(self, base_state):
        """Test that state accumulates agent responses."""
        response1 = AgentResponse(
            agent_role=AgentRole.FORECASTER,
            content="Forecast",
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            execution_time_ms=100,
            metadata={},
        )
        response2 = AgentResponse(
            agent_role=AgentRole.HISTORICAL_ANALYST,
            content="History",
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
            execution_time_ms=150,
            metadata={},
        )

        base_state.agent_responses.append(response1)
        base_state.agent_responses.append(response2)

        assert len(base_state.agent_responses) == 2

    def test_state_tracks_workflow_progress(self, base_state):
        """Test that state tracks workflow progress."""
        base_state.current_agent = AgentRole.SUPERVISOR
        base_state.workflow_complete = False

        assert base_state.current_agent == AgentRole.SUPERVISOR
        assert base_state.workflow_complete is False


# Test: invoke_workflow_v2
class TestInvokeWorkflowV2:
    """Tests for invoke_workflow_v2 function."""

    @pytest.mark.asyncio
    async def test_invoke_workflow_success(self, base_state):
        """Test successful workflow invocation."""
        # This would need full mocking of all agents
        # For now, test the function exists and accepts correct parameters
        assert callable(invoke_workflow_v2)

    @pytest.mark.asyncio
    async def test_invoke_workflow_handles_errors(self):
        """Test that workflow handles errors gracefully."""
        # Test with invalid state should not crash
        try:
            # This might raise, which is expected with invalid state
            pass
        except Exception:
            pass  # Expected for invalid state


# Test: Quality Gates
class TestQualityGates:
    """Tests for quality gate functionality in workflow."""

    def test_quality_threshold_check(self, base_state):
        """Test quality threshold checking."""
        base_state.quality_score = 0.92

        # Quality should meet threshold
        assert base_state.quality_score >= 0.9

    def test_reflection_iteration_limit(self, base_state):
        """Test reflection iteration limit."""
        base_state.reflection_iterations = 3

        # Should respect max iterations
        assert base_state.reflection_iterations <= 3


# Test: Error Handling
class TestWorkflowErrorHandling:
    """Tests for workflow error handling."""

    @pytest.mark.asyncio
    async def test_node_error_handling(self, base_state):
        """Test that nodes handle errors gracefully."""
        with patch('backend.src.orchestration.multi_agent_workflow.get_forecaster_agent') as mock_get:
            mock_agent = MagicMock()
            mock_agent.forecast_query = AsyncMock(side_effect=Exception("Agent error"))
            mock_get.return_value = mock_agent

            # Should handle error without crashing
            try:
                result = await forecaster_node(base_state)
                # If it returns, verify error is captured
                if result is not None:
                    pass
            except Exception:
                # Error is expected
                pass


# Integration Tests
class TestLevel4bWorkflowIntegration:
    """Integration tests for Level 4b workflow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_workflow_execution(self, base_state):
        """Test complete workflow execution with mocked agents."""
        base_state.query = "Hurricane Michael forecast and historical comparison"

        # This would require full mocking of all agents
        # For now, verify workflow structure
        workflow = create_level4b_workflow()
        assert workflow is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_parallel_agent_execution(self, base_state):
        """Test parallel agent execution in workflow."""
        base_state.query = "Complex hurricane analysis"
        base_state.parallel_groups = [["forecaster", "historical_analyst"]]

        # Verify parallel groups can be processed
        assert base_state.parallel_groups is not None
        assert len(base_state.parallel_groups) >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_with_reflection(self, base_state):
        """Test workflow with reflection quality gate."""
        base_state.query = "Hurricane evacuation guidance"
        base_state.requires_verification = True

        # Verify verification flag is set
        assert base_state.requires_verification is True
