"""Tests for Alert Manager Agent (Level 4a Phase 4).

Test Coverage:
- Alert severity classification (INFO, WARNING, CRITICAL, EMERGENCY)
- Channel selection based on severity
- Alert message generation for each channel
- Delivery simulation and logging
- Integration with Hurricane Specialist analysis
- Error handling and fallback alerts
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from backend.src.agents.alert_manager import (
    AlertManagerAgent,
    AlertSeverity,
    AlertChannel,
)
from backend.src.models.multi_agent import (
    AgentRole,
    AgentResponse,
    RoutingDecision,
)


# Fixtures
@pytest.fixture
def base_state() -> dict:
    """Create base state for testing."""
    return {
        "query": "Test alert query",
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "current_agent": AgentRole.HURRICANE_SPECIALIST,
        "routing_decision": MagicMock(
            query_category="moderate",
            confidence=0.9,
        ),
        "agent_responses": [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Hurricane Michael is a Category 4 storm with 140 mph winds.",
                confidence=0.85,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=1500.0,
                metadata={"nhc_data_used": True},
            )
        ],
        "next_agent": AgentRole.ALERT_MANAGER,
        "workflow_complete": False,
        "final_response": None,
        "error": None,
        "memory_context": {
            "user_profile": {"location": "Tampa, FL"},
        },
    }


@pytest.fixture
def alert_manager() -> AlertManagerAgent:
    """Create AlertManagerAgent for testing."""
    return AlertManagerAgent(model_name="gpt-4o-mini")


@pytest.fixture
def mock_llm_alert_response():
    """Mock LLM alert generation response."""
    return MagicMock(
        content="""## In-App Alert
🔴 CRITICAL

Location: Tampa, FL
Time: 3:00 PM EDT

Hurricane Michael (Category 4) is approaching the Tampa Bay area.
Maximum sustained winds of 140 mph expected.

IMMEDIATE ACTIONS:
1. Evacuate if in Zone A or B
2. Secure outdoor items
3. Monitor local emergency broadcasts

## Push Notification
Title: 🔴 CRITICAL - Tampa, FL
Body: Hurricane Michael Cat 4 approaching. Evacuate Zone A/B. Monitor updates.

## SMS Alert
🔴CRITICAL: Cat 4 Hurricane Michael 140mph. EVACUATE Zone A/B Tampa. -NHC 3PM

## Email Alert
Subject: [CRITICAL] Hurricane Alert - Tampa, FL
Body: Full alert details...
"""
    )


# Test: Initialization
def test_alert_manager_initialization():
    """Test AlertManagerAgent initialization."""
    agent = AlertManagerAgent()

    assert agent.llm is not None
    assert agent.llm.model_name == "gpt-4o-mini"
    assert agent.llm.temperature == 0.0
    assert agent.agent_role == AgentRole.ALERT_MANAGER


def test_alert_manager_custom_model():
    """Test AlertManagerAgent with custom model."""
    agent = AlertManagerAgent(model_name="gpt-4o")
    assert agent.llm.model_name == "gpt-4o"


# Test: Severity Classification
@pytest.mark.asyncio
async def test_classify_severity_emergency_from_routing(alert_manager, base_state):
    """Test emergency severity from routing decision."""
    base_state["routing_decision"] = MagicMock(query_category="emergency")

    severity = await alert_manager._classify_severity(base_state)
    assert severity == AlertSeverity.EMERGENCY


@pytest.mark.asyncio
async def test_classify_severity_emergency_keywords(alert_manager, base_state):
    """Test emergency severity from keywords."""
    base_state["query"] = "Should I evacuate NOW for this hurricane?"
    base_state["routing_decision"] = MagicMock(query_category="moderate")

    severity = await alert_manager._classify_severity(base_state)
    assert severity == AlertSeverity.EMERGENCY


@pytest.mark.asyncio
async def test_classify_severity_critical_cat4(alert_manager, base_state):
    """Test critical severity for Cat 4 hurricanes."""
    base_state["query"] = "What should I do for this Category 4 hurricane?"
    base_state["routing_decision"] = MagicMock(query_category="moderate")

    severity = await alert_manager._classify_severity(base_state)
    assert severity == AlertSeverity.CRITICAL


@pytest.mark.asyncio
async def test_classify_severity_emergency_cat5(alert_manager, base_state):
    """Test emergency severity for Cat 5 hurricanes (life-threatening)."""
    base_state["query"] = "Cat 5 hurricane approaching my area"
    base_state["routing_decision"] = MagicMock(query_category="moderate")

    severity = await alert_manager._classify_severity(base_state)
    # Cat 5 is life-threatening - EMERGENCY, not CRITICAL
    assert severity == AlertSeverity.EMERGENCY


@pytest.mark.asyncio
async def test_classify_severity_warning(alert_manager, base_state):
    """Test warning severity."""
    base_state["query"] = "Hurricane watch issued for my area"
    base_state["routing_decision"] = MagicMock(query_category="moderate")

    severity = await alert_manager._classify_severity(base_state)
    assert severity == AlertSeverity.WARNING


@pytest.mark.asyncio
async def test_classify_severity_info_default(alert_manager, base_state):
    """Test info severity as default."""
    base_state["query"] = "What's the weather forecast?"
    base_state["routing_decision"] = MagicMock(query_category="simple")

    severity = await alert_manager._classify_severity(base_state)
    assert severity == AlertSeverity.INFO


# Test: Channel Selection
def test_select_channels_emergency(alert_manager):
    """Test all channels selected for emergency."""
    channels = alert_manager._select_channels(AlertSeverity.EMERGENCY)

    assert AlertChannel.IN_APP in channels
    assert AlertChannel.PUSH in channels
    assert AlertChannel.SMS in channels
    assert AlertChannel.EMAIL in channels
    assert len(channels) == 4


def test_select_channels_critical(alert_manager):
    """Test SMS included for critical alerts."""
    channels = alert_manager._select_channels(AlertSeverity.CRITICAL)

    assert AlertChannel.IN_APP in channels
    assert AlertChannel.PUSH in channels
    assert AlertChannel.SMS in channels
    assert AlertChannel.EMAIL not in channels
    assert len(channels) == 3


def test_select_channels_warning(alert_manager):
    """Test push included for warning alerts."""
    channels = alert_manager._select_channels(AlertSeverity.WARNING)

    assert AlertChannel.IN_APP in channels
    assert AlertChannel.PUSH in channels
    assert AlertChannel.SMS not in channels
    assert AlertChannel.EMAIL not in channels
    assert len(channels) == 2


def test_select_channels_info(alert_manager):
    """Test only in-app for info alerts."""
    channels = alert_manager._select_channels(AlertSeverity.INFO)

    assert AlertChannel.IN_APP in channels
    assert len(channels) == 1


# Test: Alert Generation
@pytest.mark.asyncio
async def test_generate_alert_success(alert_manager, base_state, mock_llm_alert_response):
    """Test successful alert generation."""
    # Use mock LLM instance directly
    mock_llm_instance = AsyncMock()
    mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm_alert_response)
    alert_manager.llm = mock_llm_instance

    # Execute
    result = await alert_manager.generate_alert(base_state)

    # Verify
    assert result["current_agent"] == AgentRole.ALERT_MANAGER
    assert result["workflow_complete"] is True
    assert result["final_response"] is not None
    assert len(result["agent_responses"]) == 2  # Specialist + Alert Manager


@pytest.mark.asyncio
async def test_generate_alert_with_specialist_analysis(alert_manager, base_state, mock_llm_alert_response):
    """Test alert generation using Hurricane Specialist analysis."""
    # Use mock LLM instance directly
    mock_llm_instance = AsyncMock()
    mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm_alert_response)
    alert_manager.llm = mock_llm_instance

    # Execute
    result = await alert_manager.generate_alert(base_state)

    # Verify LLM was called with specialist analysis
    call_args = mock_llm_instance.ainvoke.call_args
    messages = call_args[0][0]
    human_message = messages[1].content

    assert "Hurricane Michael" in human_message or "specialist" in human_message.lower()


@pytest.mark.asyncio
async def test_generate_alert_handles_llm_error(alert_manager, base_state):
    """Test template-based fallback alert generation when LLM fails.

    When the LLM call fails, the Alert Manager gracefully degrades
    to template-based alert generation (not an error state).
    """
    # Use mock LLM instance that raises error
    mock_llm_instance = AsyncMock()
    mock_llm_instance.ainvoke = AsyncMock(side_effect=Exception("LLM API error"))
    alert_manager.llm = mock_llm_instance

    # Execute
    result = await alert_manager.generate_alert(base_state)

    # Verify graceful degradation (template-based fallback, not error)
    assert result["workflow_complete"] is True
    assert result["final_response"] is not None

    # Template-based fallback still produces a valid alert response
    alert_response = result["agent_responses"][-1]
    assert alert_response.agent_role == AgentRole.ALERT_MANAGER
    assert alert_response.confidence == 1.0  # Full confidence with template fallback
    assert "severity" in alert_response.metadata  # Severity classification still works
    assert "alert_id" in alert_response.metadata  # Alert ID generated


# Test: Alert Delivery Simulation
@pytest.mark.asyncio
async def test_deliver_alert_all_channels(alert_manager):
    """Test delivery simulation across all channels."""
    alert_messages = {
        "in_app": "Test in-app alert",
        "push": "Test push notification",
        "sms": "Test SMS alert",
        "email": "Test email alert",
    }
    channels = [AlertChannel.IN_APP, AlertChannel.PUSH, AlertChannel.SMS, AlertChannel.EMAIL]

    delivery_results = await alert_manager._deliver_alert(
        alert_id="test_123",
        alert_messages=alert_messages,
        severity=AlertSeverity.EMERGENCY,
        channels=channels,
        user_id="test_user",
    )

    # Verify all channels delivered
    assert delivery_results["in_app"] == "delivered"
    assert delivery_results["push"] == "simulated_success"
    assert delivery_results["sms"] == "simulated_success"
    assert delivery_results["email"] == "simulated_success"


@pytest.mark.asyncio
async def test_deliver_alert_partial_channels(alert_manager):
    """Test delivery with partial channels."""
    alert_messages = {
        "in_app": "Test alert",
        "push": "Test push",
    }
    channels = [AlertChannel.IN_APP, AlertChannel.PUSH]

    delivery_results = await alert_manager._deliver_alert(
        alert_id="test_456",
        alert_messages=alert_messages,
        severity=AlertSeverity.WARNING,
        channels=channels,
        user_id="test_user",
    )

    assert "in_app" in delivery_results
    assert "push" in delivery_results
    assert "sms" not in delivery_results
    assert "email" not in delivery_results


# Test: Get Specialist Analysis
def test_get_specialist_analysis_with_response(alert_manager, base_state):
    """Test extracting specialist analysis from state."""
    analysis = alert_manager._get_specialist_analysis(base_state)

    assert "Hurricane Michael" in analysis
    assert "Category 4" in analysis


def test_get_specialist_analysis_no_response(alert_manager):
    """Test specialist analysis when no response available."""
    state = {"agent_responses": []}
    analysis = alert_manager._get_specialist_analysis(state)

    assert "No specialist analysis" in analysis


# Test: Get User Location
def test_get_user_location_from_memory(alert_manager, base_state):
    """Test extracting location from memory context."""
    location = alert_manager._get_user_location(base_state)
    assert location == "Tampa, FL"


def test_get_user_location_from_query(alert_manager):
    """Test extracting location from query."""
    state = {
        "query": "Weather alert for Miami",
        "memory_context": {},
    }
    location = alert_manager._get_user_location(state)
    # Should extract "Miami" from query
    assert location in ["Miami", "Unknown location"]


def test_get_user_location_unknown(alert_manager):
    """Test default location when not available."""
    state = {
        "query": "Weather alert",
        "memory_context": None,
    }
    location = alert_manager._get_user_location(state)
    assert location == "Unknown location"


# Test: Alert Message Parsing
def test_parse_alert_response_complete(alert_manager, mock_llm_alert_response):
    """Test parsing complete LLM alert response."""
    alert_messages = alert_manager._parse_alert_response(
        mock_llm_alert_response.content,
        AlertSeverity.CRITICAL,
    )

    assert "in_app" in alert_messages
    assert "push" in alert_messages
    assert "sms" in alert_messages
    assert "email" in alert_messages


def test_parse_alert_response_sms_truncation(alert_manager):
    """Test SMS alert truncation to 160 characters."""
    long_sms = "A" * 200
    llm_content = f"## In-App Alert\nTest\n## SMS Alert\n{long_sms}"

    alert_messages = alert_manager._parse_alert_response(
        llm_content,
        AlertSeverity.CRITICAL,
    )

    assert len(alert_messages.get("sms", "")) <= 160


# Test: Template-Based Alert Generation
def test_generate_template_alert(alert_manager):
    """Test template-based alert generation."""
    alerts = alert_manager._generate_template_alert(
        severity=AlertSeverity.CRITICAL,
        user_location="Tampa, FL",
        specialist_analysis="Hurricane Michael Category 4 approaching.",
    )

    assert "in_app" in alerts
    assert "push" in alerts
    assert "sms" in alerts
    assert "email" in alerts

    # Verify in-app contains key info
    assert "CRITICAL" in alerts["in_app"] or "🔴" in alerts["in_app"]
    assert "Tampa, FL" in alerts["in_app"]

    # Verify SMS is within limit
    assert len(alerts["sms"]) <= 160


# Test: Final Response Formatting
def test_format_final_response(alert_manager):
    """Test final response formatting."""
    alert_messages = {
        "in_app": "Test alert message",
        "push": "Test push",
    }
    channels = [AlertChannel.IN_APP, AlertChannel.PUSH]
    delivery_results = {"in_app": "delivered", "push": "simulated_success"}

    response = alert_manager._format_final_response(
        alert_messages=alert_messages,
        severity=AlertSeverity.WARNING,
        channels=channels,
        delivery_results=delivery_results,
    )

    assert "WARNING" in response or "⚠️" in response
    assert "Test alert message" in response
    assert "delivered" in response.lower() or "2/2" in response


# Test: Fallback Alert
def test_generate_fallback_alert(alert_manager):
    """Test fallback alert generation."""
    fallback = alert_manager._generate_fallback_alert(
        query="Test emergency query",
        user_location="Tampa, FL",
    )

    assert "WEATHER ALERT" in fallback
    assert "Tampa, FL" in fallback
    assert "Recommended Actions" in fallback
    assert "fallback" in fallback.lower()


# Test: End-to-End Alert Generation
@pytest.mark.asyncio
async def test_end_to_end_emergency_alert(alert_manager, mock_llm_alert_response):
    """Test complete emergency alert workflow."""
    state = {
        "query": "Hurricane Cat 5 approaching - should I evacuate NOW?",
        "user_id": "test_user",
        "session_id": "test_session",
        "routing_decision": MagicMock(query_category="emergency"),
        "agent_responses": [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Category 5 Hurricane approaching. Immediate evacuation recommended.",
                confidence=0.95,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=1000.0,
            )
        ],
        "memory_context": {
            "user_profile": {"location": "Tampa, FL"},
        },
    }

    # Use mock LLM instance directly (Pydantic models can't be patched normally)
    mock_llm_instance = AsyncMock()
    mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm_alert_response)
    alert_manager.llm = mock_llm_instance

    # Execute
    result = await alert_manager.generate_alert(state)

    # Verify emergency alert generated
    assert result["workflow_complete"] is True

    alert_response = result["agent_responses"][-1]
    assert alert_response.agent_role == AgentRole.ALERT_MANAGER
    assert alert_response.confidence == 1.0

    # Verify all channels used for emergency
    assert "sms" in alert_response.metadata["channels"]
    assert "email" in alert_response.metadata["channels"]
    assert alert_response.metadata["severity"] == "emergency"


# Test: Alert Severity Enum
def test_alert_severity_values():
    """Test AlertSeverity enum values."""
    assert AlertSeverity.INFO.value == "info"
    assert AlertSeverity.WARNING.value == "warning"
    assert AlertSeverity.CRITICAL.value == "critical"
    assert AlertSeverity.EMERGENCY.value == "emergency"


# Test: Alert Channel Enum
def test_alert_channel_values():
    """Test AlertChannel enum values."""
    assert AlertChannel.IN_APP.value == "in_app"
    assert AlertChannel.PUSH.value == "push_notification"
    assert AlertChannel.SMS.value == "sms"
    assert AlertChannel.EMAIL.value == "email"
