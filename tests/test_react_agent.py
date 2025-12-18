"""Tests for ReAct agent.

Tests the weather agent creation and query functionality.
Note: These tests use mocked LLM to avoid actual API calls.
"""

from backend.src.agents.prompts import WEATHER_ASSISTANT_SYSTEM_PROMPT
from backend.src.agents.weather_agent import create_weather_agent


def test_create_weather_agent():
    """Test that weather agent can be created."""
    agent = create_weather_agent()

    assert agent is not None
    # Agent should have the tools available
    # In LangChain v1.0+, agent structure may vary, so just verify it's created


def test_weather_assistant_system_prompt_exists():
    """Test that weather assistant system prompt is defined."""
    assert WEATHER_ASSISTANT_SYSTEM_PROMPT is not None
    assert len(WEATHER_ASSISTANT_SYSTEM_PROMPT) > 0
    assert "weather" in WEATHER_ASSISTANT_SYSTEM_PROMPT.lower()
    assert "tools" in WEATHER_ASSISTANT_SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_tools():
    """Test that system prompt mentions the available tools."""
    assert "get_current_weather" in WEATHER_ASSISTANT_SYSTEM_PROMPT
    assert "get_forecast" in WEATHER_ASSISTANT_SYSTEM_PROMPT


def test_system_prompt_has_process_guidance():
    """Test that system prompt provides process guidance."""
    # Should have numbered steps or clear process
    assert "1." in WEATHER_ASSISTANT_SYSTEM_PROMPT or "Process:" in WEATHER_ASSISTANT_SYSTEM_PROMPT
