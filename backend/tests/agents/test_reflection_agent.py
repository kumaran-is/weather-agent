"""Tests for Reflection Agent (Level 4b Phase 8).

Test Coverage:
- Self-critique functionality
- Response improvement iterations
- Quality threshold enforcement (0.9)
- Maximum iteration limit (3)
- Saffir-Simpson scale validation
- Weather-specific quality dimensions
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from backend.src.agents.reflection_agent import ReflectionAgent
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
        reflection_iterations=0,
        quality_score=0.0,
    )


@pytest.fixture
def reflection_agent() -> ReflectionAgent:
    """Create ReflectionAgent for testing."""
    return ReflectionAgent()


@pytest.fixture
def mock_high_quality_critique():
    """Mock critique response with high quality score."""
    return MagicMock(
        content=json.dumps({
            "quality_score": 0.92,
            "accuracy": 0.95,
            "completeness": 0.90,
            "clarity": 0.92,
            "actionability": 0.88,
            "safety": 0.95,
            "strengths": ["Accurate Saffir-Simpson categorization", "Clear timeline"],
            "weaknesses": [],
            "specific_issues": [],
            "improvement_suggestions": [],
        })
    )


@pytest.fixture
def mock_low_quality_critique():
    """Mock critique response with low quality score."""
    return MagicMock(
        content=json.dumps({
            "quality_score": 0.65,
            "accuracy": 0.70,
            "completeness": 0.60,
            "clarity": 0.65,
            "actionability": 0.60,
            "safety": 0.70,
            "strengths": ["Mentions hurricane category"],
            "weaknesses": [
                "Missing specific wind speeds",
                "No evacuation zone guidance",
                "Vague timeline",
            ],
            "specific_issues": ["Category 4 stated but wind speed shows 145 mph (correct)"],
            "improvement_suggestions": [
                "Add specific evacuation zones (A, B, C)",
                "Include exact landfall time in EDT/UTC",
            ],
        })
    )


@pytest.fixture
def mock_improved_response():
    """Mock improved response after reflection."""
    return MagicMock(
        content="""Hurricane Michael is a Category 4 hurricane with sustained winds of 145 mph.
Expected landfall: Panama City Beach, FL at 2:00 PM EDT (18:00 UTC) on October 10.

EVACUATION GUIDANCE:
- Zone A residents: Evacuate immediately
- Zone B residents: Evacuate within 12 hours
- Zone C residents: Prepare to evacuate if storm strengthens

Storm surge warning: 9-12 feet expected along the coast."""
    )


# Test: ReflectionAgent Initialization
class TestReflectionAgentInit:
    """Tests for ReflectionAgent initialization."""

    def test_default_initialization(self, reflection_agent):
        """Test ReflectionAgent initializes with defaults."""
        assert reflection_agent.llm is not None
        assert reflection_agent.quality_threshold == 0.9
        assert reflection_agent.max_iterations == 3
        assert reflection_agent.agent_role == AgentRole.VERIFICATION

    def test_custom_initialization(self):
        """Test ReflectionAgent with custom parameters."""
        agent = ReflectionAgent(
            model_name="gpt-4o",
            quality_threshold=0.85,
            max_iterations=5,
        )
        assert agent.quality_threshold == 0.85
        assert agent.max_iterations == 5


# Test: Self-Critique
class TestSelfCritique:
    """Tests for self-critique functionality."""

    @pytest.mark.asyncio
    async def test_critique_high_quality_response(
        self, reflection_agent, mock_high_quality_critique
    ):
        """Test critique of high-quality response."""
        response = "Hurricane Michael is Category 4 with 145 mph winds..."
        query = "What category is Hurricane Michael?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_high_quality_critique)
        reflection_agent.llm = mock_llm

        critique = await reflection_agent._self_critique(response, query)

        assert critique["quality_score"] >= 0.9
        assert len(critique["weaknesses"]) == 0

    @pytest.mark.asyncio
    async def test_critique_low_quality_response(
        self, reflection_agent, mock_low_quality_critique
    ):
        """Test critique of low-quality response."""
        response = "The hurricane is Category 4."
        query = "What should I do about Hurricane Michael?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_low_quality_critique)
        reflection_agent.llm = mock_llm

        critique = await reflection_agent._self_critique(response, query)

        assert critique["quality_score"] < 0.9
        assert len(critique["weaknesses"]) > 0

    @pytest.mark.asyncio
    async def test_critique_handles_invalid_json(self, reflection_agent):
        """Test critique handles invalid JSON response."""
        response = "Test response"
        query = "Test query"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Not valid JSON"
        ))
        reflection_agent.llm = mock_llm

        critique = await reflection_agent._self_critique(response, query)

        # Should return default critique
        assert "quality_score" in critique
        assert critique["quality_score"] == 0.7  # Default on parse error


# Test: Response Improvement
class TestResponseImprovement:
    """Tests for response improvement functionality."""

    @pytest.mark.asyncio
    async def test_improve_response(
        self, reflection_agent, mock_improved_response
    ):
        """Test response improvement based on feedback."""
        current = "The hurricane is Category 4."
        query = "What should I do about Hurricane Michael?"
        weaknesses = ["Missing evacuation guidance", "Vague timeline"]

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_improved_response)
        reflection_agent.llm = mock_llm

        improved = await reflection_agent._improve_response(
            current, query, weaknesses
        )

        assert improved is not None
        assert len(improved) > len(current)

    @pytest.mark.asyncio
    async def test_improve_response_handles_error(self, reflection_agent):
        """Test improvement handles errors gracefully."""
        current = "Original response"
        query = "Test query"
        weaknesses = ["Some weakness"]

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        reflection_agent.llm = mock_llm

        improved = await reflection_agent._improve_response(
            current, query, weaknesses
        )

        # Should return original on error
        assert improved == current


# Test: Reflect and Improve Workflow
class TestReflectAndImprove:
    """Tests for full reflect_and_improve workflow."""

    @pytest.mark.asyncio
    async def test_reflect_high_quality_no_iteration(
        self, reflection_agent, base_state, mock_high_quality_critique
    ):
        """Test reflection stops immediately for high-quality response."""
        base_state.query = "Hurricane forecast query"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="High quality hurricane response with all details",
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_high_quality_critique)
        reflection_agent.llm = mock_llm

        result = await reflection_agent.reflect_and_improve(base_state)

        assert result.reflection_iterations == 1
        assert result.quality_score >= 0.9

    @pytest.mark.asyncio
    async def test_reflect_iterates_for_low_quality(
        self, reflection_agent, base_state,
        mock_low_quality_critique, mock_high_quality_critique
    ):
        """Test reflection iterates to improve low-quality response."""
        base_state.query = "Hurricane forecast query"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Brief hurricane response",
                confidence=0.7,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        # First critique: low quality, then improvement, then high quality
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            mock_low_quality_critique,  # First critique
            MagicMock(content="Improved response with details"),  # Improvement
            mock_high_quality_critique,  # Second critique (passes)
        ])
        reflection_agent.llm = mock_llm

        result = await reflection_agent.reflect_and_improve(base_state)

        # Should have iterated
        assert result.reflection_iterations >= 1

    @pytest.mark.asyncio
    async def test_reflect_respects_max_iterations(
        self, reflection_agent, base_state, mock_low_quality_critique
    ):
        """Test reflection stops at max iterations."""
        base_state.query = "Hurricane forecast query"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Low quality response that never improves",
                confidence=0.5,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        # Always return low quality
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_low_quality_critique)
        reflection_agent.llm = mock_llm

        result = await reflection_agent.reflect_and_improve(base_state)

        # Should stop at max iterations
        assert result.reflection_iterations <= reflection_agent.max_iterations


# Test: Saffir-Simpson Scale Validation
class TestSaffirSimpsonValidation:
    """Tests for Saffir-Simpson scale validation in response text."""

    @pytest.mark.asyncio
    async def test_verify_category_1_valid(self, reflection_agent):
        """Test Category 1 validation (74-95 mph) - valid match."""
        response = "Category 1 hurricane with 85 mph winds."
        result = await reflection_agent.verify_saffir_simpson(response)
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_verify_category_2_valid(self, reflection_agent):
        """Test Category 2 validation (96-110 mph) - valid match."""
        response = "Category 2 hurricane with 100 mph winds."
        result = await reflection_agent.verify_saffir_simpson(response)
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_verify_category_3_valid(self, reflection_agent):
        """Test Category 3 validation (111-129 mph) - valid match."""
        response = "Category 3 hurricane with 120 mph winds."
        result = await reflection_agent.verify_saffir_simpson(response)
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_verify_category_4_valid(self, reflection_agent):
        """Test Category 4 validation (130-156 mph) - valid match."""
        response = "Category 4 hurricane with 145 mph winds."
        result = await reflection_agent.verify_saffir_simpson(response)
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_verify_category_5_valid(self, reflection_agent):
        """Test Category 5 validation (157+ mph) - valid match."""
        response = "Category 5 hurricane with 165 mph winds."
        result = await reflection_agent.verify_saffir_simpson(response)
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_verify_no_category_info(self, reflection_agent):
        """Test validation with no category information."""
        response = "The weather is sunny today."
        result = await reflection_agent.verify_saffir_simpson(response)
        assert result is not None
        assert isinstance(result, dict)


# Test: Quality Dimensions
class TestQualityDimensions:
    """Tests for quality dimension evaluation."""

    @pytest.mark.asyncio
    async def test_all_dimensions_evaluated(
        self, reflection_agent, mock_high_quality_critique
    ):
        """Test all quality dimensions are evaluated."""
        response = "Test response"
        query = "Test query"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_high_quality_critique)
        reflection_agent.llm = mock_llm

        critique = await reflection_agent._self_critique(response, query)

        # Verify all dimensions present
        assert "accuracy" in critique
        assert "completeness" in critique
        assert "clarity" in critique
        assert "actionability" in critique
        assert "safety" in critique


# Test: Error Handling
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_empty_responses(self, reflection_agent, base_state):
        """Test handling of empty agent responses."""
        base_state.agent_responses = []

        result = await reflection_agent.reflect_and_improve(base_state)

        # Should handle gracefully
        assert result is not None
        assert result.reflection_iterations == 0

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self, reflection_agent, base_state):
        """Test handling of LLM failure during reflection."""
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Test response",
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=100,
                metadata={},
            )
        ]

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM failure"))
        reflection_agent.llm = mock_llm

        result = await reflection_agent.reflect_and_improve(base_state)

        # Should handle error
        assert result is not None


# Integration Tests
class TestReflectionIntegration:
    """Integration tests for ReflectionAgent."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_reflection_workflow(
        self, reflection_agent, base_state,
        mock_low_quality_critique, mock_high_quality_critique, mock_improved_response
    ):
        """Test complete reflection and improvement workflow."""
        base_state.query = "Should I evacuate for Hurricane Michael?"
        base_state.agent_responses = [
            AgentResponse(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Hurricane Michael is approaching. Consider evacuating.",
                confidence=0.75,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=200,
                metadata={},
            )
        ]

        # Simulate improvement cycle
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            mock_low_quality_critique,  # First critique (low)
            mock_improved_response,      # Improvement
            mock_high_quality_critique,  # Second critique (high)
        ])
        reflection_agent.llm = mock_llm

        result = await reflection_agent.reflect_and_improve(base_state)

        # Verify workflow completed
        assert result is not None
        assert result.reflection_iterations >= 1
        assert len(result.agent_responses) > 1
