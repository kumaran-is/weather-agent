"""Tests for Multi-Agent Workflow Orchestration (Level 4a Phase 5).

Test Coverage:
- StateGraph workflow creation
- Routing logic (route_after_triage, route_after_specialist)
- Agent node functions
- End-to-end workflow execution
- Confidence threshold enforcement
- Risk-based alert triggering
- Error handling and recovery
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.orchestration.multi_agent_workflow import (
    create_multi_agent_workflow,
    compile_workflow,
    route_after_triage,
    route_after_specialist,
    triage_agent_node,
    hurricane_specialist_node,
    alert_manager_node,
    invoke_workflow,
    WorkflowState,
    _assess_risk_level,
)
from backend.src.models.multi_agent import (
    AgentRole,
    RoutingDecision,
    AgentResponse,
)


# Fixtures
@pytest.fixture
def base_workflow_state() -> WorkflowState:
    """Create base workflow state for testing."""
    return {
        "query": "Test weather query",
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "current_agent": None,
        "routing_decision": None,
        "agent_responses": [],
        "next_agent": None,
        "workflow_complete": False,
        "final_response": None,
        "error": None,
        "total_execution_time_ms": 0.0,
        "memory_context": None,
        "query_complexity": "simple",
        "triage_confidence": 0.0,
        "risk_level": "LOW",
        "agents_invoked": [],
    }


@pytest.fixture
def triage_state_emergency() -> WorkflowState:
    """Create state after triage with EMERGENCY classification."""
    return {
        "query": "Hurricane Cat 5 approaching - evacuate NOW?",
        "user_id": "test_user",
        "session_id": "test_session",
        "current_agent": AgentRole.TRIAGE,
        "routing_decision": MagicMock(
            query_category="emergency",
            confidence=0.95,
            next_agent=AgentRole.HURRICANE_SPECIALIST,
        ),
        "agent_responses": [],
        "next_agent": AgentRole.HURRICANE_SPECIALIST,
        "workflow_complete": False,
        "query_complexity": "emergency",
        "triage_confidence": 0.95,
        "risk_level": "LOW",
        "agents_invoked": ["triage"],
    }


@pytest.fixture
def triage_state_simple() -> WorkflowState:
    """Create state after triage with SIMPLE classification."""
    return {
        "query": "What's the weather in Miami?",
        "user_id": "test_user",
        "session_id": "test_session",
        "current_agent": AgentRole.TRIAGE,
        "routing_decision": MagicMock(
            query_category="simple",
            confidence=0.92,
            next_agent=None,
        ),
        "agent_responses": [],
        "next_agent": None,
        "workflow_complete": False,
        "query_complexity": "simple",
        "triage_confidence": 0.92,
        "risk_level": "LOW",
        "agents_invoked": ["triage"],
    }


@pytest.fixture
def specialist_state_high_risk() -> WorkflowState:
    """Create state after specialist with HIGH risk."""
    return {
        "query": "Hurricane Cat 4 approaching Tampa",
        "user_id": "test_user",
        "session_id": "test_session",
        "current_agent": AgentRole.HURRICANE_SPECIALIST,
        "agent_responses": [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Category 4 Hurricane Michael with 140 mph winds. Evacuate zones A & B.",
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
            )
        ],
        "next_agent": None,
        "workflow_complete": False,
        "query_complexity": "complex",
        "risk_level": "HIGH",
        "agents_invoked": ["triage", "hurricane_specialist"],
    }


@pytest.fixture
def specialist_state_low_risk() -> WorkflowState:
    """Create state after specialist with LOW risk."""
    return {
        "query": "What's the status of tropical storms?",
        "user_id": "test_user",
        "session_id": "test_session",
        "current_agent": AgentRole.HURRICANE_SPECIALIST,
        "agent_responses": [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Tropical Storm Amy in the Atlantic, Category 1, no threat to US.",
                confidence=0.85,
                timestamp=datetime.now(timezone.utc),
            )
        ],
        "next_agent": None,
        "workflow_complete": True,
        "final_response": "Tropical Storm Amy in the Atlantic...",
        "query_complexity": "moderate",
        "risk_level": "LOW",
        "agents_invoked": ["triage", "hurricane_specialist"],
    }


# Test: Workflow Creation
def test_create_workflow():
    """Test multi-agent workflow creation."""
    workflow = create_multi_agent_workflow()

    assert workflow is not None
    # Verify nodes exist
    assert "triage" in workflow.nodes
    assert "hurricane_specialist" in workflow.nodes
    assert "alert_manager" in workflow.nodes
    assert "direct_response" in workflow.nodes
    assert "end_workflow" in workflow.nodes


def test_compile_workflow_no_checkpointer():
    """Test workflow compilation without checkpointer."""
    compiled = compile_workflow(checkpointer=None)

    assert compiled is not None
    # Compiled graph should be invokable
    assert hasattr(compiled, "ainvoke")


def test_compile_workflow_with_mock_checkpointer():
    """Test workflow compilation with mock checkpointer."""
    mock_checkpointer = MagicMock()
    compiled = compile_workflow(checkpointer=mock_checkpointer)

    assert compiled is not None


# Test: Routing Logic - route_after_triage
def test_route_after_triage_emergency(triage_state_emergency):
    """Test emergency queries always route to hurricane_specialist."""
    result = route_after_triage(triage_state_emergency)
    assert result == "hurricane_specialist"


def test_route_after_triage_high_confidence_simple(triage_state_simple):
    """Test high confidence simple queries route to direct_response."""
    result = route_after_triage(triage_state_simple)
    assert result == "direct_response"


def test_route_after_triage_hurricane_keywords(base_workflow_state):
    """Test hurricane keywords trigger specialist routing."""
    base_workflow_state["query"] = "Is the hurricane approaching Florida?"
    base_workflow_state["triage_confidence"] = 0.6
    base_workflow_state["query_complexity"] = "moderate"

    result = route_after_triage(base_workflow_state)
    assert result == "hurricane_specialist"


def test_route_after_triage_storm_keywords(base_workflow_state):
    """Test storm keywords trigger specialist routing."""
    base_workflow_state["query"] = "When will the tropical storm hit?"
    base_workflow_state["triage_confidence"] = 0.7
    base_workflow_state["query_complexity"] = "moderate"

    result = route_after_triage(base_workflow_state)
    assert result == "hurricane_specialist"


def test_route_after_triage_routing_decision_hurricane(base_workflow_state):
    """Test routing decision to hurricane specialist."""
    base_workflow_state["routing_decision"] = MagicMock(
        next_agent=AgentRole.HURRICANE_SPECIALIST,
        query_category="moderate",
    )
    base_workflow_state["next_agent"] = AgentRole.HURRICANE_SPECIALIST

    result = route_after_triage(base_workflow_state)
    assert result == "hurricane_specialist"


def test_route_after_triage_routing_decision_alert(base_workflow_state):
    """Test routing decision to alert manager (via specialist)."""
    base_workflow_state["routing_decision"] = MagicMock(
        next_agent=AgentRole.ALERT_MANAGER,
        query_category="emergency",
    )

    result = route_after_triage(base_workflow_state)
    assert result == "hurricane_specialist"  # Goes through specialist first


def test_route_after_triage_default_safety(base_workflow_state):
    """Test default routing to specialist for safety."""
    base_workflow_state["query"] = "Unclear weather question"
    base_workflow_state["triage_confidence"] = 0.3
    base_workflow_state["query_complexity"] = "unknown"

    result = route_after_triage(base_workflow_state)
    # Should route to specialist by default for safety
    assert result == "hurricane_specialist"


# Test: Routing Logic - route_after_specialist
def test_route_after_specialist_high_risk(specialist_state_high_risk):
    """Test high risk routes to alert_manager."""
    result = route_after_specialist(specialist_state_high_risk)
    assert result == "alert_manager"


def test_route_after_specialist_low_risk(specialist_state_low_risk):
    """Test low risk ends workflow."""
    result = route_after_specialist(specialist_state_low_risk)
    assert result == "end_workflow"


def test_route_after_specialist_extreme_risk(base_workflow_state):
    """Test extreme risk routes to alert_manager."""
    base_workflow_state["risk_level"] = "EXTREME"
    base_workflow_state["workflow_complete"] = False

    result = route_after_specialist(base_workflow_state)
    assert result == "alert_manager"


def test_route_after_specialist_medium_risk(base_workflow_state):
    """Test medium risk routes to alert_manager."""
    base_workflow_state["risk_level"] = "MEDIUM"
    base_workflow_state["workflow_complete"] = False

    result = route_after_specialist(base_workflow_state)
    assert result == "alert_manager"


def test_route_after_specialist_emergency_query(base_workflow_state):
    """Test emergency queries always get alerts."""
    base_workflow_state["risk_level"] = "LOW"
    base_workflow_state["query_complexity"] = "emergency"
    base_workflow_state["workflow_complete"] = False

    result = route_after_specialist(base_workflow_state)
    assert result == "alert_manager"


def test_route_after_specialist_next_agent_alert(base_workflow_state):
    """Test specialist routing to alert manager."""
    base_workflow_state["next_agent"] = AgentRole.ALERT_MANAGER
    base_workflow_state["workflow_complete"] = False

    result = route_after_specialist(base_workflow_state)
    assert result == "alert_manager"


def test_route_after_specialist_workflow_complete(specialist_state_low_risk):
    """Test completed workflow ends."""
    specialist_state_low_risk["workflow_complete"] = True
    specialist_state_low_risk["next_agent"] = None

    result = route_after_specialist(specialist_state_low_risk)
    assert result == "end_workflow"


# Test: Risk Assessment
def test_assess_risk_level_extreme():
    """Test extreme risk assessment."""
    result = {
        "agent_responses": [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Category 5 hurricane with catastrophic damage potential",
                confidence=0.95,
                timestamp=datetime.now(timezone.utc),
            )
        ]
    }

    risk = _assess_risk_level(result)
    assert risk == "EXTREME"


def test_assess_risk_level_high():
    """Test high risk assessment."""
    result = {
        "agent_responses": [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Category 4 hurricane - evacuate zones A and B",
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
            )
        ]
    }

    risk = _assess_risk_level(result)
    assert risk == "HIGH"


def test_assess_risk_level_medium():
    """Test medium risk assessment."""
    result = {
        "agent_responses": [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Category 3 major hurricane approaching. Prepare for impact.",
                confidence=0.85,
                timestamp=datetime.now(timezone.utc),
            )
        ]
    }

    risk = _assess_risk_level(result)
    assert risk == "MEDIUM"


def test_assess_risk_level_low():
    """Test low risk assessment."""
    result = {
        "agent_responses": [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Tropical storm in the Atlantic, no threat expected",
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
            )
        ]
    }

    risk = _assess_risk_level(result)
    assert risk == "LOW"


def test_assess_risk_level_next_agent_alert():
    """Test risk assessment when next_agent is ALERT_MANAGER."""
    result = {
        "agent_responses": [],
        "next_agent": AgentRole.ALERT_MANAGER,
    }

    risk = _assess_risk_level(result)
    assert risk == "MEDIUM"


# Test: Agent Node Functions (Integration)
@pytest.mark.asyncio
async def test_triage_agent_node(base_workflow_state):
    """Test triage agent node invocation."""
    base_workflow_state["query"] = "Hurricane forecast for Tampa"

    # Mock the TriageAgent
    with patch("backend.src.orchestration.multi_agent_workflow.get_triage_agent") as mock_get:
        mock_agent = AsyncMock()
        mock_agent.classify_and_route = AsyncMock(return_value={
            **base_workflow_state,
            "routing_decision": MagicMock(
                query_category="moderate",
                confidence=0.85,
                next_agent=AgentRole.HURRICANE_SPECIALIST,
            ),
            "next_agent": AgentRole.HURRICANE_SPECIALIST,
        })
        mock_get.return_value = mock_agent

        result = await triage_agent_node(base_workflow_state)

        assert "triage" in result["agents_invoked"]
        mock_agent.classify_and_route.assert_called_once()


@pytest.mark.asyncio
async def test_hurricane_specialist_node(triage_state_emergency):
    """Test hurricane specialist node invocation."""
    with patch("backend.src.orchestration.multi_agent_workflow.get_hurricane_specialist") as mock_get:
        mock_agent = AsyncMock()
        mock_agent.process_query = AsyncMock(return_value={
            **triage_state_emergency,
            "agent_responses": [
                AgentResponse(
                    agent_role=AgentRole.HURRICANE_SPECIALIST,
                    content="Hurricane analysis...",
                    confidence=0.9,
                    timestamp=datetime.now(timezone.utc),
                )
            ],
        })
        mock_get.return_value = mock_agent

        result = await hurricane_specialist_node(triage_state_emergency)

        assert "hurricane_specialist" in result["agents_invoked"]
        mock_agent.process_query.assert_called_once()


@pytest.mark.asyncio
async def test_alert_manager_node(specialist_state_high_risk):
    """Test alert manager node invocation."""
    with patch("backend.src.orchestration.multi_agent_workflow.get_alert_manager") as mock_get:
        mock_agent = AsyncMock()
        mock_agent.generate_alert = AsyncMock(return_value={
            **specialist_state_high_risk,
            "workflow_complete": True,
            "final_response": "Alert generated",
        })
        mock_get.return_value = mock_agent

        result = await alert_manager_node(specialist_state_high_risk)

        assert "alert_manager" in result["agents_invoked"]
        mock_agent.generate_alert.assert_called_once()


# Test: invoke_workflow Helper
@pytest.mark.asyncio
async def test_invoke_workflow_generates_ids():
    """Test that invoke_workflow generates user_id and session_id if not provided."""
    with patch("backend.src.orchestration.multi_agent_workflow.compile_workflow") as mock_compile:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "query": "Test",
            "agents_invoked": ["triage"],
            "workflow_complete": True,
        })
        mock_compile.return_value = mock_graph

        result = await invoke_workflow(query="Test query")

        # Verify ainvoke was called
        mock_graph.ainvoke.assert_called_once()

        # Check initial state has user_id and session_id
        call_args = mock_graph.ainvoke.call_args
        initial_state = call_args[0][0]

        assert "user_id" in initial_state
        assert initial_state["user_id"].startswith("user_")
        assert "session_id" in initial_state
        assert initial_state["session_id"].startswith("session_")


@pytest.mark.asyncio
async def test_invoke_workflow_uses_provided_ids():
    """Test that invoke_workflow uses provided user_id and session_id."""
    with patch("backend.src.orchestration.multi_agent_workflow.compile_workflow") as mock_compile:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "query": "Test",
            "user_id": "custom_user",
            "session_id": "custom_session",
        })
        mock_compile.return_value = mock_graph

        result = await invoke_workflow(
            query="Test query",
            user_id="custom_user",
            session_id="custom_session",
        )

        call_args = mock_graph.ainvoke.call_args
        initial_state = call_args[0][0]

        assert initial_state["user_id"] == "custom_user"
        assert initial_state["session_id"] == "custom_session"


@pytest.mark.asyncio
async def test_invoke_workflow_passes_memory_context():
    """Test that invoke_workflow passes memory context."""
    memory_context = {
        "user_profile": {"location": "Tampa, FL"},
        "previous_queries": ["Hurricane update?"],
    }

    with patch("backend.src.orchestration.multi_agent_workflow.compile_workflow") as mock_compile:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "query": "Test",
            "memory_context": memory_context,
        })
        mock_compile.return_value = mock_graph

        result = await invoke_workflow(
            query="Test query",
            memory_context=memory_context,
        )

        call_args = mock_graph.ainvoke.call_args
        initial_state = call_args[0][0]

        assert initial_state["memory_context"] == memory_context


@pytest.mark.asyncio
async def test_invoke_workflow_error_handling():
    """Test invoke_workflow error handling."""
    with patch("backend.src.orchestration.multi_agent_workflow.compile_workflow") as mock_compile:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("Workflow failed"))
        mock_compile.return_value = mock_graph

        with pytest.raises(Exception) as exc_info:
            await invoke_workflow(query="Test query")

        assert "Workflow failed" in str(exc_info.value)


# Test: End-to-End Workflow Scenarios
@pytest.mark.asyncio
@pytest.mark.integration
async def test_workflow_simple_query_ends_at_direct_response():
    """Test simple queries end at direct_response node."""
    with patch("backend.src.orchestration.multi_agent_workflow.get_triage_agent") as mock_triage:
        mock_agent = AsyncMock()
        mock_agent.classify_and_route = AsyncMock(return_value={
            "query": "What's the weather?",
            "user_id": "test",
            "session_id": "test",
            "routing_decision": MagicMock(
                query_category="simple",
                confidence=0.95,
                next_agent=None,
            ),
            "next_agent": None,
            "query_complexity": "simple",
            "triage_confidence": 0.95,
            "agents_invoked": ["triage"],
            "agent_responses": [
                AgentResponse(
                    agent_role=AgentRole.TRIAGE,
                    content="Simple weather query",
                    confidence=0.95,
                    timestamp=datetime.now(timezone.utc),
                )
            ],
        })
        mock_triage.return_value = mock_agent

        workflow = create_multi_agent_workflow()
        compiled = workflow.compile()

        result = await compiled.ainvoke({
            "query": "What's the weather in Miami?",
            "user_id": "test",
            "session_id": "test",
            "agents_invoked": [],
            "agent_responses": [],
        })

        # Should only invoke triage
        assert "triage" in result["agents_invoked"]
        assert "hurricane_specialist" not in result.get("agents_invoked", [])


@pytest.mark.asyncio
@pytest.mark.integration
async def test_workflow_hurricane_query_invokes_specialist():
    """Test hurricane queries invoke hurricane_specialist."""
    with patch("backend.src.orchestration.multi_agent_workflow.get_triage_agent") as mock_triage, \
         patch("backend.src.orchestration.multi_agent_workflow.get_hurricane_specialist") as mock_specialist:

        # Mock triage
        triage_agent = AsyncMock()
        triage_agent.classify_and_route = AsyncMock(return_value={
            "query": "Hurricane forecast?",
            "user_id": "test",
            "session_id": "test",
            "routing_decision": MagicMock(
                query_category="moderate",
                confidence=0.9,
                next_agent=AgentRole.HURRICANE_SPECIALIST,
            ),
            "next_agent": AgentRole.HURRICANE_SPECIALIST,
            "query_complexity": "moderate",
            "triage_confidence": 0.9,
            "agents_invoked": ["triage"],
            "agent_responses": [],
        })
        mock_triage.return_value = triage_agent

        # Mock specialist
        specialist_agent = AsyncMock()
        specialist_agent.process_query = AsyncMock(return_value={
            "query": "Hurricane forecast?",
            "current_agent": AgentRole.HURRICANE_SPECIALIST,
            "workflow_complete": True,
            "final_response": "Hurricane forecast...",
            "risk_level": "LOW",
            "agents_invoked": ["triage", "hurricane_specialist"],
            "agent_responses": [
                AgentResponse(
                    agent_role=AgentRole.HURRICANE_SPECIALIST,
                    content="Forecast details",
                    confidence=0.85,
                    timestamp=datetime.now(timezone.utc),
                )
            ],
        })
        mock_specialist.return_value = specialist_agent

        workflow = create_multi_agent_workflow()
        compiled = workflow.compile()

        result = await compiled.ainvoke({
            "query": "Hurricane forecast for Tampa?",
            "user_id": "test",
            "session_id": "test",
            "agents_invoked": [],
            "agent_responses": [],
        })

        # Should invoke triage and specialist
        assert "triage" in result["agents_invoked"]
        assert "hurricane_specialist" in result["agents_invoked"]
