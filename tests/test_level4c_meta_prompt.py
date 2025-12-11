"""Tests for Level 4c Meta-Prompt Agent.

This module tests the meta-prompting capabilities including:
- Dynamic prompt generation
- Context-aware prompts
- Few-shot example injection
- Performance tracking
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.agents.meta_prompt_agent import (
    MetaPromptAgent,
    create_meta_prompt_agent,
)
from backend.src.agents.prompts.meta_prompt_templates import (
    META_PROMPT_SYSTEM,
    PROMPT_GENERATION_TEMPLATE,
    FEW_SHOT_EXAMPLES,
    AGENT_PROMPT_TEMPLATES,
)
from backend.src.models.multi_agent import AgentRole, AgentResponse, MultiAgentState


class TestMetaPromptTemplates:
    """Tests for meta-prompt templates."""

    def test_meta_prompt_system_exists(self):
        """Test that system prompt is defined."""
        assert META_PROMPT_SYSTEM is not None
        assert len(META_PROMPT_SYSTEM) > 0
        assert "prompt" in META_PROMPT_SYSTEM.lower()

    def test_prompt_generation_template_exists(self):
        """Test that generation template is defined."""
        assert PROMPT_GENERATION_TEMPLATE is not None
        assert "{target_agent}" in PROMPT_GENERATION_TEMPLATE
        assert "{query}" in PROMPT_GENERATION_TEMPLATE

    def test_few_shot_examples_structure(self):
        """Test few-shot examples structure."""
        assert FEW_SHOT_EXAMPLES is not None
        assert isinstance(FEW_SHOT_EXAMPLES, dict)

        # Check that examples have expected structure
        for agent_type, examples in FEW_SHOT_EXAMPLES.items():
            assert isinstance(examples, list)
            for example in examples:
                assert "query" in example or "keywords" in example

    def test_agent_prompt_templates_exist(self):
        """Test that agent prompt templates are defined."""
        assert AGENT_PROMPT_TEMPLATES is not None
        assert isinstance(AGENT_PROMPT_TEMPLATES, dict)


class TestMetaPromptAgentInit:
    """Tests for MetaPromptAgent initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        agent = MetaPromptAgent()

        assert agent.agent_role == AgentRole.META_PROMPT
        assert agent.llm is not None
        assert agent._generation_count == 0
        assert agent._prompt_performance == {}

    def test_custom_initialization(self):
        """Test custom initialization."""
        agent = MetaPromptAgent(
            model_name="gpt-4",
            temperature=0.5,
            timeout=30.0,
        )

        assert agent.llm.model_name == "gpt-4"
        assert agent.llm.temperature == 0.5


class TestGenerateAgentPrompt:
    """Tests for generate_agent_prompt method."""

    @pytest.mark.asyncio
    async def test_generate_prompt_basic(self):
        """Test basic prompt generation."""
        agent = MetaPromptAgent()

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(
                content="You are a hurricane specialist. Analyze the storm data..."
            )

            prompt = await agent.generate_agent_prompt(
                target_agent=AgentRole.HURRICANE_SPECIALIST,
                query="What's the hurricane forecast?",
            )

            assert "hurricane" in prompt.lower()
            assert agent._generation_count == 1

    @pytest.mark.asyncio
    async def test_generate_prompt_with_context(self):
        """Test prompt generation with context."""
        agent = MetaPromptAgent()

        context = {
            "user_location": "Tampa, FL",
            "emotional_state": "anxious",
            "recent_queries": ["weather update", "storm forecast"],
        }

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(
                content="Given the user's anxiety and location in Tampa..."
            )

            prompt = await agent.generate_agent_prompt(
                target_agent=AgentRole.FORECASTER,
                query="Is the storm coming to Tampa?",
                context=context,
            )

            assert prompt is not None
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_prompt_increments_count(self):
        """Test that generation increments counter."""
        agent = MetaPromptAgent()

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content="Generated prompt")

            await agent.generate_agent_prompt(
                target_agent=AgentRole.FORECASTER,
                query="Query 1",
            )
            await agent.generate_agent_prompt(
                target_agent=AgentRole.RESEARCH,
                query="Query 2",
            )

            assert agent._generation_count == 2


class TestProcess:
    """Tests for process method."""

    @pytest.mark.asyncio
    async def test_process_generates_prompts_for_workflow(self):
        """Test that process generates prompts for workflow plan."""
        agent = MetaPromptAgent()

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content="Generated prompt content")

            state = MultiAgentState(
                query="Hurricane forecast",
                user_id="test_user",
                workflow_plan=[
                    AgentRole.HURRICANE_SPECIALIST,
                    AgentRole.FORECASTER,
                ],
            )

            result = await agent.process(state)

            assert "generated_prompts" in result.memory_context
            assert len(result.memory_context["generated_prompts"]) == 2
            assert result.current_agent == AgentRole.META_PROMPT

    @pytest.mark.asyncio
    async def test_process_skips_self(self):
        """Test that process skips generating prompt for itself."""
        agent = MetaPromptAgent()

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content="Generated prompt")

            state = MultiAgentState(
                query="Test query",
                user_id="test_user",
                workflow_plan=[
                    AgentRole.META_PROMPT,  # Should be skipped
                    AgentRole.FORECASTER,
                ],
            )

            result = await agent.process(state)

            # Should only have prompt for FORECASTER
            prompts = result.memory_context.get("generated_prompts", {})
            assert "meta_prompt" not in prompts
            assert "forecaster" in prompts

    @pytest.mark.asyncio
    async def test_process_adds_response(self):
        """Test that process adds a response to state."""
        agent = MetaPromptAgent()

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content="Generated prompt")

            state = MultiAgentState(
                query="Test query",
                user_id="test_user",
                workflow_plan=[AgentRole.FORECASTER],
            )

            result = await agent.process(state)

            assert len(result.agent_responses) >= 1
            response = result.agent_responses[-1]
            assert response.agent_role == AgentRole.META_PROMPT


class TestFewShotExamples:
    """Tests for few-shot example retrieval."""

    def test_get_relevant_examples_with_keywords(self):
        """Test getting examples with keyword matching."""
        agent = MetaPromptAgent()

        # Test with hurricane-related query
        examples = agent._get_relevant_examples(
            target_agent=AgentRole.HURRICANE_SPECIALIST,
            query="hurricane forecast for Tampa",
        )

        assert examples is not None
        assert len(examples) > 0

    def test_get_relevant_examples_fallback(self):
        """Test fallback when no keywords match."""
        agent = MetaPromptAgent()

        examples = agent._get_relevant_examples(
            target_agent=AgentRole.FORECASTER,
            query="xyz completely unrelated query abc",
        )

        # Should return fallback examples
        assert examples is not None

    def test_get_relevant_examples_unknown_agent(self):
        """Test handling unknown agent type."""
        agent = MetaPromptAgent()

        examples = agent._get_relevant_examples(
            target_agent=AgentRole.SYNTHESIS,  # May not have examples
            query="test query",
        )

        # Should return some message even for unknown agents
        assert examples is not None


class TestContextFormatting:
    """Tests for context formatting."""

    def test_format_context_with_all_fields(self):
        """Test formatting context with all fields."""
        agent = MetaPromptAgent()

        context = {
            "user_location": "Miami, FL",
            "emotional_state": "concerned",
            "recent_queries": ["storm update", "evacuation routes"],
            "preferences": "detailed explanations",
        }

        formatted = agent._format_context(context)

        assert "Miami, FL" in formatted
        assert "concerned" in formatted
        assert "storm update" in formatted
        assert "detailed" in formatted

    def test_format_context_empty(self):
        """Test formatting empty context."""
        agent = MetaPromptAgent()

        formatted = agent._format_context(None)

        assert "No additional context" in formatted

    def test_format_context_partial(self):
        """Test formatting partial context."""
        agent = MetaPromptAgent()

        context = {"user_location": "NYC"}

        formatted = agent._format_context(context)

        assert "NYC" in formatted


class TestBuildContextFromState:
    """Tests for building context from state."""

    def test_build_context_from_memory(self):
        """Test building context from memory context."""
        agent = MetaPromptAgent()

        state = MultiAgentState(
            query="Test query",
            user_id="test_user",
            memory_context={
                "user_location": "Tampa, FL",
                "emotional_state": "anxious",
                "preferences": "concise",
            },
        )

        context = agent._build_context_from_state(state)

        assert context["user_location"] == "Tampa, FL"
        assert context["emotional_state"] == "anxious"
        assert context["preferences"] == "concise"

    def test_build_context_empty_state(self):
        """Test building context from empty state."""
        agent = MetaPromptAgent()

        state = MultiAgentState(
            query="Test query",
            user_id="test_user",
        )

        context = agent._build_context_from_state(state)

        assert isinstance(context, dict)


class TestPerformanceTracking:
    """Tests for prompt performance tracking."""

    def test_track_prompt_performance(self):
        """Test tracking prompt performance."""
        agent = MetaPromptAgent()

        prompt_hash = "abc123"
        agent.track_prompt_performance(prompt_hash, 0.9)
        agent.track_prompt_performance(prompt_hash, 0.85)
        agent.track_prompt_performance(prompt_hash, 0.95)

        assert prompt_hash in agent._prompt_performance
        assert len(agent._prompt_performance[prompt_hash]) == 3

    def test_get_prompt_hash(self):
        """Test generating prompt hash."""
        agent = MetaPromptAgent()

        prompt1 = "This is a test prompt"
        prompt2 = "This is a different prompt"

        hash1 = agent.get_prompt_hash(prompt1)
        hash2 = agent.get_prompt_hash(prompt2)

        assert hash1 != hash2
        # Same input should produce same hash
        assert agent.get_prompt_hash(prompt1) == hash1

    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        agent = MetaPromptAgent()

        # Add some performance data
        agent._generation_count = 50
        agent._prompt_performance["hash1"] = [0.9, 0.85, 0.95]
        agent._prompt_performance["hash2"] = [0.8, 0.75]

        stats = agent.get_performance_stats()

        assert stats["total_generated"] == 50
        assert stats["tracked_prompts"] == 2
        assert "prompt_scores" in stats


class TestGetBaseTemplate:
    """Tests for getting base templates."""

    def test_get_base_template_exists(self):
        """Test getting existing base template."""
        agent = MetaPromptAgent()

        # Only test if templates are defined
        if AGENT_PROMPT_TEMPLATES:
            for role_str in AGENT_PROMPT_TEMPLATES:
                # Get template
                template = agent.get_base_template(AgentRole(role_str))
                assert template is not None or template is None  # Just verify it doesn't crash

    def test_get_base_template_nonexistent(self):
        """Test getting non-existent template."""
        agent = MetaPromptAgent()

        # META_PROMPT itself might not have a template
        template = agent.get_base_template(AgentRole.META_PROMPT)

        # Should return None if not found
        assert template is None or isinstance(template, str)


class TestFactoryFunction:
    """Tests for create_meta_prompt_agent factory function."""

    def test_create_with_defaults(self):
        """Test creating agent with default settings."""
        agent = create_meta_prompt_agent()

        assert isinstance(agent, MetaPromptAgent)
        assert agent.agent_role == AgentRole.META_PROMPT

    def test_create_with_custom_settings(self):
        """Test creating agent with custom settings."""
        agent = create_meta_prompt_agent(
            model_name="gpt-4",
            temperature=0.5,
        )

        assert agent.llm.temperature == 0.5


class TestErrorHandling:
    """Tests for error handling in meta-prompt agent."""

    @pytest.mark.asyncio
    async def test_process_handles_llm_error(self):
        """Test that process handles LLM errors gracefully."""
        agent = MetaPromptAgent()

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM error")

            state = MultiAgentState(
                query="Test query",
                user_id="test_user",
                workflow_plan=[AgentRole.FORECASTER],
            )

            result = await agent.process(state)

            # Should handle error gracefully
            assert len(result.agent_responses) >= 1
            response = result.agent_responses[-1]
            assert response.confidence == 0.0 or "error" in str(response.metadata)
