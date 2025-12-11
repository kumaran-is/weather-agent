"""Tests for Critique Agent (Level 4b Phase 8).

Test Coverage:
- Generator -> Critic -> Refiner workflow
- Two-model approach (expensive generator, cheap critic)
- Conditional refinement based on threshold
- Weather-specific evaluation criteria
- Saffir-Simpson scale validation in critique
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from backend.src.agents.critique_agent import CritiqueAgent
from backend.src.models.multi_agent import (
    AgentRole,
    AgentResponse,
    MultiAgentState,
)


# Fixtures
@pytest.fixture
def base_state() -> MultiAgentState:
    """Create base MultiAgentState for testing."""
    return MultiAgentState(
        query="Test hurricane query",
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
        quality_score=0.0,
    )


@pytest.fixture
def critique_agent() -> CritiqueAgent:
    """Create CritiqueAgent for testing."""
    return CritiqueAgent()


@pytest.fixture
def mock_good_critique():
    """Mock critique response with high score."""
    return MagicMock(
        content=json.dumps({
            "score": 0.90,
            "feedback": "Response is accurate and comprehensive",
            "strengths": [
                "Correct Saffir-Simpson category",
                "Specific wind speeds provided",
                "Clear evacuation guidance",
            ],
            "weaknesses": [],
            "critical_errors": [],
            "suggested_additions": [],
            "suggested_removals": [],
        })
    )


@pytest.fixture
def mock_poor_critique():
    """Mock critique response with low score."""
    return MagicMock(
        content=json.dumps({
            "score": 0.60,
            "feedback": "Response needs significant improvement",
            "strengths": ["Mentions hurricane category"],
            "weaknesses": [
                "Wind speed doesn't match category",
                "Missing evacuation zone guidance",
                "No specific timeline",
            ],
            "critical_errors": [
                "Category 5 stated but wind speed shows 145 mph (should be 157+ for Cat 5)",
            ],
            "suggested_additions": [
                "Add specific evacuation zones (A, B, C)",
                "Include landfall time in EDT/UTC",
                "Storm surge estimates",
            ],
            "suggested_removals": [
                "Remove vague 'soon' language",
            ],
        })
    )


@pytest.fixture
def mock_generated_response():
    """Mock initial generated response."""
    return MagicMock(
        content="""Hurricane Michael is currently a Category 4 hurricane with sustained winds of 145 mph.
The storm is expected to make landfall near Panama City Beach, Florida on October 10 at 2:00 PM EDT.

EVACUATION GUIDANCE:
- Zone A: Mandatory evacuation in effect
- Zone B: Evacuation recommended
- Zone C: Shelter in place, monitor updates

Expected impacts:
- Storm surge: 9-12 feet
- Rainfall: 4-8 inches
- Wind gusts up to 165 mph"""
    )


@pytest.fixture
def mock_refined_response():
    """Mock refined response after critique."""
    return MagicMock(
        content="""URGENT: Hurricane Michael - Category 4

Hurricane Michael has intensified to a Category 4 hurricane with maximum sustained winds of 145 mph.

LANDFALL: Panama City Beach, FL
TIME: October 10, 2024 at 2:00 PM EDT (18:00 UTC)

EVACUATION ORDERS:
- Zone A (coastal): MANDATORY EVACUATION - Leave immediately
- Zone B: EVACUATION RECOMMENDED - Leave within 6 hours
- Zone C: VOLUNTARY EVACUATION - Shelter in place if well-built structure

EXPECTED IMPACTS:
- Storm surge: 9-12 feet along coast
- Rainfall: 4-8 inches, potential for flash flooding
- Wind gusts: Up to 165 mph

LIFE-SAFETY: If in Zone A or B, evacuate NOW. Do not wait."""
    )


# Test: CritiqueAgent Initialization
class TestCritiqueAgentInit:
    """Tests for CritiqueAgent initialization."""

    def test_default_initialization(self, critique_agent):
        """Test CritiqueAgent initializes with defaults."""
        assert critique_agent.generator_llm is not None
        assert critique_agent.critic_llm is not None
        assert critique_agent.refinement_threshold == 0.85
        assert critique_agent.agent_role == AgentRole.VERIFICATION

    def test_two_model_approach(self, critique_agent):
        """Test that two different models are used."""
        # Generator uses gpt-4o (expensive)
        assert critique_agent.generator_llm.model_name == "gpt-4o"
        # Critic uses gpt-4o-mini (cheap)
        assert critique_agent.critic_llm.model_name == "gpt-4o-mini"

    def test_custom_initialization(self):
        """Test CritiqueAgent with custom parameters."""
        agent = CritiqueAgent(
            generator_model="gpt-4o",
            critic_model="gpt-4o-mini",
            refinement_threshold=0.80,
        )
        assert agent.refinement_threshold == 0.80


# Test: Generation Phase
class TestGenerationPhase:
    """Tests for response generation phase."""

    @pytest.mark.asyncio
    async def test_generate_response(
        self, critique_agent, mock_generated_response
    ):
        """Test initial response generation."""
        query = "What is Hurricane Michael's status?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_generated_response)
        critique_agent.generator_llm = mock_llm

        response = await critique_agent._generate(query)

        assert response is not None
        assert "Hurricane Michael" in response

    @pytest.mark.asyncio
    async def test_uses_existing_response(
        self, critique_agent, base_state, mock_good_critique
    ):
        """Test that existing response is used when available."""
        base_state.query = "Hurricane query"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Existing hurricane response with details",
                confidence=0.85,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        # Mock critic
        mock_critic = AsyncMock()
        mock_critic.ainvoke = AsyncMock(return_value=mock_good_critique)
        critique_agent.critic_llm = mock_critic

        result = await critique_agent.critique_and_refine(base_state)

        # Should use existing response, not generate new
        assert result is not None


# Test: Critique Phase
class TestCritiquePhase:
    """Tests for critique evaluation phase."""

    @pytest.mark.asyncio
    async def test_critique_good_response(
        self, critique_agent, mock_good_critique
    ):
        """Test critique of high-quality response."""
        response = "Hurricane Michael is Category 4 with 145 mph winds..."
        query = "Hurricane status?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_good_critique)
        critique_agent.critic_llm = mock_llm

        critique = await critique_agent._critique(response, query)

        assert critique["score"] >= 0.85
        assert len(critique["critical_errors"]) == 0

    @pytest.mark.asyncio
    async def test_critique_poor_response(
        self, critique_agent, mock_poor_critique
    ):
        """Test critique of low-quality response."""
        response = "Hurricane is Category 5."  # Missing details
        query = "Full hurricane analysis?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_poor_critique)
        critique_agent.critic_llm = mock_llm

        critique = await critique_agent._critique(response, query)

        assert critique["score"] < 0.85
        assert len(critique["critical_errors"]) > 0
        assert len(critique["weaknesses"]) > 0

    @pytest.mark.asyncio
    async def test_critique_handles_invalid_json(self, critique_agent):
        """Test critique handles invalid JSON gracefully."""
        response = "Test response"
        query = "Test query"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Not valid JSON response"
        ))
        critique_agent.critic_llm = mock_llm

        critique = await critique_agent._critique(response, query)

        # Should return default critique
        assert critique["score"] == 0.7
        assert "feedback" in critique


# Test: Refinement Phase
class TestRefinementPhase:
    """Tests for response refinement phase."""

    @pytest.mark.asyncio
    async def test_refine_response(
        self, critique_agent, mock_refined_response
    ):
        """Test response refinement based on feedback."""
        original = "Hurricane is approaching."
        feedback = "Add specific details"
        critical_errors = ["Missing wind speed"]
        additions = ["Add evacuation zones"]
        query = "Hurricane status?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_refined_response)
        critique_agent.generator_llm = mock_llm

        refined = await critique_agent._refine(
            original, feedback, critical_errors, additions, query
        )

        assert refined is not None
        assert len(refined) > len(original)

    @pytest.mark.asyncio
    async def test_refine_handles_error(self, critique_agent):
        """Test refinement returns original on error."""
        original = "Original response"
        feedback = "Some feedback"
        query = "Test query"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        critique_agent.generator_llm = mock_llm

        refined = await critique_agent._refine(
            original, feedback, [], [], query
        )

        # Should return original on error
        assert refined == original


# Test: Full Critique and Refine Workflow
class TestCritiqueAndRefineWorkflow:
    """Tests for full critique_and_refine workflow."""

    @pytest.mark.asyncio
    async def test_workflow_no_refinement_needed(
        self, critique_agent, base_state, mock_good_critique
    ):
        """Test workflow skips refinement for high-quality response."""
        base_state.query = "Hurricane query"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="High quality hurricane response",
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        mock_critic = AsyncMock()
        mock_critic.ainvoke = AsyncMock(return_value=mock_good_critique)
        critique_agent.critic_llm = mock_critic

        result = await critique_agent.critique_and_refine(base_state)

        # Check no refinement occurred
        last_response = result.agent_responses[-1]
        assert last_response.metadata.get("was_refined") is False

    @pytest.mark.asyncio
    async def test_workflow_refinement_triggered(
        self, critique_agent, base_state,
        mock_poor_critique, mock_refined_response
    ):
        """Test workflow triggers refinement for low-quality response."""
        base_state.query = "Hurricane query"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Brief hurricane response",
                confidence=0.6,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        mock_critic = AsyncMock()
        mock_critic.ainvoke = AsyncMock(return_value=mock_poor_critique)
        critique_agent.critic_llm = mock_critic

        mock_generator = AsyncMock()
        mock_generator.ainvoke = AsyncMock(return_value=mock_refined_response)
        critique_agent.generator_llm = mock_generator

        result = await critique_agent.critique_and_refine(base_state)

        # Check refinement occurred
        last_response = result.agent_responses[-1]
        assert last_response.metadata.get("was_refined") is True

    @pytest.mark.asyncio
    async def test_workflow_generates_when_no_responses(
        self, critique_agent, base_state,
        mock_generated_response, mock_good_critique
    ):
        """Test workflow generates response when none exist."""
        base_state.query = "Hurricane query"
        base_state.agent_responses = []

        mock_generator = AsyncMock()
        mock_generator.ainvoke = AsyncMock(return_value=mock_generated_response)
        critique_agent.generator_llm = mock_generator

        mock_critic = AsyncMock()
        mock_critic.ainvoke = AsyncMock(return_value=mock_good_critique)
        critique_agent.critic_llm = mock_critic

        result = await critique_agent.critique_and_refine(base_state)

        # Should have generated and critiqued
        assert len(result.agent_responses) >= 1


# Test: Quick Critique
class TestQuickCritique:
    """Tests for quick_critique standalone method."""

    @pytest.mark.asyncio
    async def test_quick_critique(
        self, critique_agent, mock_good_critique
    ):
        """Test standalone quick critique."""
        response = "Hurricane response"
        query = "Hurricane query"

        mock_critic = AsyncMock()
        mock_critic.ainvoke = AsyncMock(return_value=mock_good_critique)
        critique_agent.critic_llm = mock_critic

        result = await critique_agent.quick_critique(response, query)

        assert "score" in result
        assert "feedback" in result
        assert "strengths" in result


# Test: Weather-Specific Evaluation
class TestWeatherSpecificEvaluation:
    """Tests for weather-specific evaluation criteria."""

    @pytest.mark.asyncio
    async def test_detects_saffir_simpson_errors(
        self, critique_agent
    ):
        """Test critique detects Saffir-Simpson scale errors."""
        # Response with incorrect category/wind speed
        response = "Hurricane is Category 5 with winds of 145 mph."  # Should be 157+ for Cat 5
        query = "Hurricane status?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "score": 0.50,
                "feedback": "Critical error in Saffir-Simpson classification",
                "strengths": [],
                "weaknesses": ["Incorrect hurricane category"],
                "critical_errors": [
                    "Category 5 requires 157+ mph winds, but response shows 145 mph"
                ],
                "suggested_additions": ["Correct to Category 4 or update wind speed"],
                "suggested_removals": [],
            })
        ))
        critique_agent.critic_llm = mock_llm

        critique = await critique_agent._critique(response, query)

        assert critique["score"] < 0.85
        assert len(critique["critical_errors"]) > 0

    @pytest.mark.asyncio
    async def test_evaluates_evacuation_guidance(
        self, critique_agent
    ):
        """Test critique evaluates evacuation guidance quality."""
        response = "You should evacuate soon."  # Vague
        query = "Should I evacuate?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "score": 0.55,
                "feedback": "Evacuation guidance is vague",
                "strengths": [],
                "weaknesses": [
                    "No specific evacuation zones",
                    "'Soon' is vague - needs specific time",
                ],
                "critical_errors": [],
                "suggested_additions": [
                    "Specify evacuation zone (A, B, C)",
                    "Provide specific evacuation deadline",
                ],
                "suggested_removals": ["Remove 'soon' - use specific time"],
            })
        ))
        critique_agent.critic_llm = mock_llm

        critique = await critique_agent._critique(response, query)

        assert len(critique["weaknesses"]) > 0


# Test: Error Handling
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_generator_failure(
        self, critique_agent, base_state
    ):
        """Test handling of generator failure."""
        base_state.query = "Hurricane query"
        base_state.agent_responses = []

        mock_generator = AsyncMock()
        mock_generator.ainvoke = AsyncMock(
            side_effect=Exception("Generator failed")
        )
        critique_agent.generator_llm = mock_generator

        result = await critique_agent.critique_and_refine(base_state)

        # Should handle error
        assert result is not None
        assert len(result.agent_responses) == 1
        assert "failed" in result.agent_responses[-1].content.lower()

    @pytest.mark.asyncio
    async def test_handles_critic_failure(
        self, critique_agent, base_state, mock_generated_response
    ):
        """Test handling of critic failure."""
        base_state.query = "Hurricane query"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Hurricane response",
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        mock_critic = AsyncMock()
        mock_critic.ainvoke = AsyncMock(side_effect=Exception("Critic failed"))
        critique_agent.critic_llm = mock_critic

        result = await critique_agent.critique_and_refine(base_state)

        # Should handle error, return default critique
        assert result is not None


# Integration Tests
class TestCritiqueIntegration:
    """Integration tests for CritiqueAgent."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_critique_workflow(
        self, critique_agent, base_state,
        mock_poor_critique, mock_refined_response, mock_good_critique
    ):
        """Test complete Generator -> Critic -> Refiner workflow."""
        base_state.query = "What should I do about Hurricane Michael?"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Hurricane Michael is approaching Florida.",
                confidence=0.7,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=150,
                metadata={},
            )
        ]

        # Setup mocks for full cycle
        mock_critic = AsyncMock()
        mock_critic.ainvoke = AsyncMock(return_value=mock_poor_critique)
        critique_agent.critic_llm = mock_critic

        mock_generator = AsyncMock()
        mock_generator.ainvoke = AsyncMock(return_value=mock_refined_response)
        critique_agent.generator_llm = mock_generator

        result = await critique_agent.critique_and_refine(base_state)

        # Verify workflow completed
        assert result is not None
        assert result.quality_score is not None
        assert len(result.agent_responses) >= 2

        # Verify refinement metadata
        last_response = result.agent_responses[-1]
        assert last_response.agent_role == AgentRole.VERIFICATION
        assert "critique_score" in last_response.metadata
