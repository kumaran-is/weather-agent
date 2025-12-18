"""Comprehensive tests for WeatherAgentState (LangGraph v1.x compliant).

Tests cover:
- State creation and field validation
- Messages field with add_messages reducer
- LangGraph v1.x Annotation compliance
- StateGraph integration
- Type safety and edge cases
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from backend.src.agents.state import WeatherAgentState


class TestStateCreation:
    """Test state creation and basic field operations."""

    def test_create_empty_state(self):
        """Test creating an empty state with minimal fields."""
        state: WeatherAgentState = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "What's the weather?",
            "current_step": "input",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        assert state["user_id"] == "user123"
        assert state["session_id"] == "session456"
        assert state["current_query"] == "What's the weather?"
        assert state["current_step"] == "input"
        assert state["approved"] is False
        assert state["hurricane_category"] is None
        assert state["alert_message"] is None
        assert len(state["messages"]) == 0

    def test_create_state_with_messages(self):
        """Test creating a state with initial messages."""
        state: WeatherAgentState = {
            "messages": [
                HumanMessage(content="What's the weather in Miami?"),
                AIMessage(content="Let me check that for you."),
            ],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "What's the weather in Miami?",
            "current_step": "agent",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        assert len(state["messages"]) == 2
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
        assert state["messages"][0].content == "What's the weather in Miami?"

    def test_create_state_with_hurricane_alert(self):
        """Test creating a state with hurricane alert information."""
        state: WeatherAgentState = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "Hurricane Milton approaching Miami",
            "current_step": "approval",
            "approved": False,
            "hurricane_category": 4,
            "alert_message": "Category 4 hurricane detected. Evacuate zones A, B, C immediately.",
        }

        assert state["hurricane_category"] == 4
        assert state["alert_message"] is not None
        assert "Category 4" in state["alert_message"]
        assert state["current_step"] == "approval"


class TestWorkflowSteps:
    """Test workflow step transitions."""

    def test_all_valid_workflow_steps(self):
        """Test all valid workflow step literals."""
        valid_steps = ["input", "agent", "approval", "response"]

        for step in valid_steps:
            state: WeatherAgentState = {
                "messages": [],
                "user_id": "user123",
                "session_id": "session456",
                "current_query": "Weather?",
                "current_step": step,  # type: ignore
                "approved": False,
                "hurricane_category": None,
                "alert_message": None,
            }
            assert state["current_step"] == step

    def test_step_progression(self):
        """Test typical workflow step progression."""
        state: WeatherAgentState = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "Weather?",
            "current_step": "input",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        # Progress through workflow
        state["current_step"] = "agent"
        assert state["current_step"] == "agent"

        state["current_step"] = "approval"
        assert state["current_step"] == "approval"

        state["approved"] = True
        state["current_step"] = "response"
        assert state["current_step"] == "response"
        assert state["approved"] is True


class TestHurricaneHITL:
    """Test hurricane HITL (Human-in-the-Loop) workflow."""

    def test_hurricane_detection(self):
        """Test detecting hurricane and setting category."""
        state: WeatherAgentState = {
            "messages": [HumanMessage(content="What's a Category 5 hurricane?")],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "What's a Category 5 hurricane?",
            "current_step": "agent",
            "approved": False,
            "hurricane_category": 5,
            "alert_message": "Category 5 hurricane: Catastrophic damage expected. 157+ mph winds.",
        }

        assert state["hurricane_category"] == 5
        assert "Category 5" in state["alert_message"]
        assert "157+" in state["alert_message"]

    def test_approval_workflow(self):
        """Test hurricane alert approval workflow."""
        state: WeatherAgentState = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "Hurricane approaching",
            "current_step": "approval",
            "approved": False,
            "hurricane_category": 3,
            "alert_message": "Category 3 hurricane approaching.",
        }

        # Initially not approved
        assert state["approved"] is False
        assert state["current_step"] == "approval"

        # Approve and move to response
        state["approved"] = True
        state["current_step"] = "response"

        assert state["approved"] is True
        assert state["current_step"] == "response"

    def test_all_hurricane_categories(self):
        """Test all hurricane categories (1-5)."""
        for category in range(1, 6):
            state: WeatherAgentState = {
                "messages": [],
                "user_id": "user123",
                "session_id": "session456",
                "current_query": f"Category {category} hurricane",
                "current_step": "approval",
                "approved": False,
                "hurricane_category": category,
                "alert_message": f"Category {category} hurricane detected.",
            }

            assert state["hurricane_category"] == category
            assert str(category) in state["alert_message"]


class TestMessagesReducer:
    """Test messages field with add_messages reducer."""

    def test_add_messages_to_empty_state(self):
        """Test adding messages to an initially empty state."""
        state: WeatherAgentState = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "Weather?",
            "current_step": "input",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        # Add human message
        new_messages = [HumanMessage(content="What's the weather in Boston?")]
        state["messages"] = state["messages"] + new_messages

        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)

    def test_messages_conversation_flow(self):
        """Test typical conversation flow with messages."""
        state: WeatherAgentState = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "Weather in NYC?",
            "current_step": "input",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        # User asks
        state["messages"].append(HumanMessage(content="What's the weather in NYC?"))
        assert len(state["messages"]) == 1

        # Agent responds
        state["messages"].append(AIMessage(content="Let me check the current weather in NYC."))
        assert len(state["messages"]) == 2

        # Agent uses tool
        state["messages"].append(AIMessage(content="Tool call: get_current_weather(NYC)"))
        assert len(state["messages"]) == 3

        # Agent provides final answer
        state["messages"].append(
            AIMessage(content="The current weather in NYC is 72°F and sunny.")
        )
        assert len(state["messages"]) == 4

    def test_multiple_message_types(self):
        """Test state with different message types."""
        state: WeatherAgentState = {
            "messages": [
                SystemMessage(content="You are a weather assistant."),
                HumanMessage(content="What's the weather?"),
                AIMessage(content="I'll check that."),
            ],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "What's the weather?",
            "current_step": "agent",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], SystemMessage)
        assert isinstance(state["messages"][1], HumanMessage)
        assert isinstance(state["messages"][2], AIMessage)


class TestLangGraphIntegration:
    """Test LangGraph v1.x StateGraph integration."""

    def test_state_with_langgraph_stategraph(self):
        """Test that WeatherAgentState works with StateGraph."""

        def agent_node(state: WeatherAgentState) -> dict:
            """Mock agent node that processes a query."""
            return {
                "messages": state["messages"]
                + [AIMessage(content="Processing your weather query...")],
                "current_step": "response",
            }

        # Create StateGraph with WeatherAgentState
        workflow = StateGraph(WeatherAgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)

        graph = workflow.compile()

        # Test invocation
        initial_state: WeatherAgentState = {
            "messages": [HumanMessage(content="What's the weather?")],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "What's the weather?",
            "current_step": "input",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        result = graph.invoke(initial_state)

        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + agent response
        assert result["current_step"] == "response"

    def test_state_with_conditional_routing(self):
        """Test state with conditional routing based on hurricane detection."""

        def detect_hurricane(state: WeatherAgentState) -> dict:
            """Mock node that detects hurricanes."""
            if "hurricane" in state["current_query"].lower():
                return {
                    "hurricane_category": 3,
                    "alert_message": "Category 3 hurricane detected.",
                    "current_step": "approval",
                }
            return {"current_step": "response"}

        def should_approve(state: WeatherAgentState) -> str:
            """Route to approval if hurricane detected."""
            if state.get("hurricane_category") is not None:
                return "approval"
            return "response"

        workflow = StateGraph(WeatherAgentState)
        workflow.add_node("detect", detect_hurricane)
        workflow.add_node("approval", lambda s: {"approved": True})
        workflow.add_node("response", lambda s: {"current_step": "response"})

        workflow.add_edge(START, "detect")
        workflow.add_conditional_edges("detect", should_approve)
        workflow.add_edge("approval", END)
        workflow.add_edge("response", END)

        graph = workflow.compile()

        # Test hurricane query (should go through approval)
        hurricane_state: WeatherAgentState = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "Hurricane Milton approaching",
            "current_step": "input",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        result = graph.invoke(hurricane_state)
        assert result["hurricane_category"] == 3
        assert result["current_step"] == "approval"
        assert result["approved"] is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_strings(self):
        """Test state with empty string values."""
        state: WeatherAgentState = {
            "messages": [],
            "user_id": "",
            "session_id": "",
            "current_query": "",
            "current_step": "input",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        assert state["user_id"] == ""
        assert state["session_id"] == ""
        assert state["current_query"] == ""

    def test_very_long_query(self):
        """Test state with very long query string."""
        long_query = "What's the weather in " + "New York " * 100
        state: WeatherAgentState = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": long_query,
            "current_step": "input",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        assert len(state["current_query"]) > 1000
        assert "New York" in state["current_query"]

    def test_state_with_many_messages(self):
        """Test state with large message history."""
        messages = [
            HumanMessage(content=f"Query {i}") if i % 2 == 0 else AIMessage(content=f"Response {i}")
            for i in range(100)
        ]

        state: WeatherAgentState = {
            "messages": messages,
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "Latest query",
            "current_step": "agent",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        assert len(state["messages"]) == 100
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)

    def test_hurricane_category_boundary_values(self):
        """Test hurricane categories at boundary values."""
        for category in [0, 1, 5, 6, 10]:
            state: WeatherAgentState = {
                "messages": [],
                "user_id": "user123",
                "session_id": "session456",
                "current_query": f"Category {category}",
                "current_step": "approval",
                "approved": False,
                "hurricane_category": category,
                "alert_message": f"Category {category}",
            }

            assert state["hurricane_category"] == category

    def test_state_field_updates(self):
        """Test updating individual state fields."""
        state: WeatherAgentState = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "current_query": "Initial query",
            "current_step": "input",
            "approved": False,
            "hurricane_category": None,
            "alert_message": None,
        }

        # Update query
        state["current_query"] = "Updated query"
        assert state["current_query"] == "Updated query"

        # Update step
        state["current_step"] = "agent"
        assert state["current_step"] == "agent"

        # Update hurricane info
        state["hurricane_category"] = 2
        state["alert_message"] = "Category 2 hurricane"
        assert state["hurricane_category"] == 2
        assert state["alert_message"] == "Category 2 hurricane"

        # Approve
        state["approved"] = True
        assert state["approved"] is True
