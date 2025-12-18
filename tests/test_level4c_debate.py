"""Tests for Level 4c Debate Agent.

This module tests the debate pattern implementation including:
- Multi-proposal generation
- Proposal scoring
- Debate rounds
- Winner selection
- Response synthesis
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.src.agents.debate_agent import (
    DebateAgent,
    DebateResult,
    Proposal,
    create_debate_agent,
)
from backend.src.models.multi_agent import AgentRole, MultiAgentState


class TestProposal:
    """Tests for Proposal dataclass."""

    def test_proposal_creation(self):
        """Test creating a proposal."""
        proposal = Proposal(
            agent_role=AgentRole.HURRICANE_SPECIALIST,
            content="Hurricane forecast analysis",
            confidence=0.85,
            reasoning="Based on NHC data",
        )

        assert proposal.agent_role == AgentRole.HURRICANE_SPECIALIST
        assert proposal.content == "Hurricane forecast analysis"
        assert proposal.confidence == 0.85
        assert proposal.reasoning == "Based on NHC data"
        assert proposal.score == 0.0  # Default
        assert proposal.evaluation is None  # Default

    def test_proposal_with_score(self):
        """Test proposal with score."""
        proposal = Proposal(
            agent_role=AgentRole.FORECASTER,
            content="Weather forecast",
            confidence=0.9,
            score=35.5,
            evaluation={"accuracy": 8, "completeness": 9},
        )

        assert proposal.score == 35.5
        assert proposal.evaluation["accuracy"] == 8


class TestDebateResult:
    """Tests for DebateResult dataclass."""

    def test_debate_result_creation(self):
        """Test creating a debate result."""
        winner = Proposal(
            agent_role=AgentRole.HURRICANE_SPECIALIST,
            content="Winning proposal",
            confidence=0.9,
            score=40,
        )
        runner_up = Proposal(
            agent_role=AgentRole.FORECASTER,
            content="Second place",
            confidence=0.8,
            score=35,
        )

        result = DebateResult(
            winner=winner,
            runner_up=runner_up,
            all_proposals=[winner, runner_up],
            final_response="Synthesized response",
            total_rounds=2,
            total_time_ms=1500.0,
        )

        assert result.winner.agent_role == AgentRole.HURRICANE_SPECIALIST
        assert result.runner_up.agent_role == AgentRole.FORECASTER
        assert len(result.all_proposals) == 2
        assert result.final_response == "Synthesized response"
        assert result.total_rounds == 2
        assert result.total_time_ms == 1500.0


class TestDebateAgentInit:
    """Tests for DebateAgent initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        agent = DebateAgent()

        assert agent.num_debaters == 3
        assert agent.debate_rounds == 2
        assert agent.llm is not None
        assert agent.judge_llm is not None

    def test_custom_initialization(self):
        """Test custom initialization."""
        agent = DebateAgent(
            model_name="gpt-4",
            num_debaters=5,
            debate_rounds=3,
        )

        assert agent.num_debaters == 5
        assert agent.debate_rounds == 3


class TestDebateProcess:
    """Tests for the debate process method."""

    @pytest.mark.asyncio
    async def test_process_adds_response(self):
        """Test that process adds a response to state."""
        agent = DebateAgent(num_debaters=2, debate_rounds=1)

        # Mock the LLM responses
        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(
                content="This is a proposal response. Confidence: HIGH"
            )

            with patch.object(
                agent.judge_llm, "ainvoke", new_callable=AsyncMock
            ) as mock_judge:
                mock_judge.return_value = MagicMock(
                    content='{"total_score": 35, "accuracy": 8, "completeness": 9}'
                )

                state = MultiAgentState(
                    query="Hurricane forecast for Tampa",
                    user_id="test_user",
                )

                result = await agent.process(state)

                assert len(result.agent_responses) >= 1
                assert result.current_agent == AgentRole.DEBATE


class TestRunDebate:
    """Tests for run_debate method."""

    @pytest.mark.asyncio
    async def test_run_debate_with_multiple_proposers(self):
        """Test running debate with multiple proposers."""
        agent = DebateAgent(num_debaters=3, debate_rounds=1)

        # Mock LLM responses
        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(
                content="Proposal content. Confidence: HIGH"
            )

            with patch.object(
                agent.judge_llm, "ainvoke", new_callable=AsyncMock
            ) as mock_judge:
                mock_judge.return_value = MagicMock(
                    content='{"total_score": 38, "accuracy": 9, "completeness": 10}'
                )

                state = MultiAgentState(
                    query="Hurricane forecast",
                    user_id="test_user",
                )

                proposers = [
                    AgentRole.HURRICANE_SPECIALIST,
                    AgentRole.FORECASTER,
                    AgentRole.HISTORICAL_ANALYST,
                ]

                result = await agent.run_debate(state, proposers)

                # Should have a debate response
                assert len(result.agent_responses) >= 1

    @pytest.mark.asyncio
    async def test_run_debate_handles_single_proposal(self):
        """Test debate handling when only one proposal is received."""
        agent = DebateAgent(num_debaters=2)

        # Mock LLM to only succeed once
        call_count = 0

        async def mock_ainvoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(content="Only proposal")
            raise RuntimeError("Simulated failure")

        with patch.object(agent.llm, "ainvoke", side_effect=mock_ainvoke):
            state = MultiAgentState(
                query="Test query",
                user_id="test_user",
            )

            proposers = [AgentRole.FORECASTER, AgentRole.RESEARCH]

            result = await agent.run_debate(state, proposers)

            # Should still produce a result (with single proposal warning)
            assert len(result.agent_responses) >= 1


class TestProposalGeneration:
    """Tests for proposal generation."""

    @pytest.mark.asyncio
    async def test_generate_proposals_parallel(self):
        """Test that proposals are generated in parallel."""
        agent = DebateAgent(num_debaters=3)

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(
                content="Test proposal content. Confidence: MEDIUM"
            )

            proposals = await agent._generate_proposals(
                query="Test query",
                proposers=[
                    AgentRole.HURRICANE_SPECIALIST,
                    AgentRole.FORECASTER,
                    AgentRole.RESEARCH,
                ],
            )

            # Should have generated proposals for all proposers
            assert len(proposals) == 3
            # LLM should have been called 3 times (parallel)
            assert mock_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_proposals_handles_failures(self):
        """Test that proposal generation handles individual failures."""
        agent = DebateAgent(num_debaters=3)

        call_count = 0

        async def mock_ainvoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated failure")
            return MagicMock(content="Proposal content")

        with patch.object(agent.llm, "ainvoke", side_effect=mock_ainvoke):
            proposals = await agent._generate_proposals(
                query="Test query",
                proposers=[
                    AgentRole.HURRICANE_SPECIALIST,
                    AgentRole.FORECASTER,
                    AgentRole.RESEARCH,
                ],
            )

            # Should have 2 proposals (1 failed)
            assert len(proposals) == 2


class TestProposalScoring:
    """Tests for proposal scoring."""

    @pytest.mark.asyncio
    async def test_score_proposals(self):
        """Test proposal scoring."""
        agent = DebateAgent()

        proposals = [
            Proposal(
                agent_role=AgentRole.FORECASTER,
                content="Forecast proposal",
                confidence=0.8,
            ),
            Proposal(
                agent_role=AgentRole.HURRICANE_SPECIALIST,
                content="Hurricane proposal",
                confidence=0.9,
            ),
        ]

        with patch.object(
            agent.judge_llm, "ainvoke", new_callable=AsyncMock
        ) as mock_judge:
            # Return different scores for different proposals
            mock_judge.side_effect = [
                MagicMock(
                    content='{"total_score": 30, "accuracy": 7, "completeness": 8}'
                ),
                MagicMock(
                    content='{"total_score": 40, "accuracy": 10, "completeness": 10}'
                ),
            ]

            scored = await agent._score_proposals(proposals, "Test query")

            assert len(scored) == 2
            assert scored[0].score == 30
            assert scored[1].score == 40

    @pytest.mark.asyncio
    async def test_score_proposals_handles_invalid_json(self):
        """Test that scoring handles invalid JSON responses."""
        agent = DebateAgent()

        proposals = [
            Proposal(
                agent_role=AgentRole.FORECASTER,
                content="Test proposal",
                confidence=0.8,
            )
        ]

        with patch.object(
            agent.judge_llm, "ainvoke", new_callable=AsyncMock
        ) as mock_judge:
            mock_judge.return_value = MagicMock(content="Invalid JSON response")

            scored = await agent._score_proposals(proposals, "Test query")

            # Should have fallback score
            assert len(scored) == 1
            assert scored[0].score == 25  # Default fallback score


class TestResponseSynthesis:
    """Tests for response synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_response_with_runner_up(self):
        """Test synthesis with winner and runner-up."""
        agent = DebateAgent()

        winner = Proposal(
            agent_role=AgentRole.HURRICANE_SPECIALIST,
            content="Winner content",
            confidence=0.9,
            score=40,
        )

        runner_up = Proposal(
            agent_role=AgentRole.FORECASTER,
            content="Runner-up content",
            confidence=0.8,
            score=35,
        )

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(
                content="Synthesized response combining best elements"
            )

            response = await agent._synthesize_response("Test query", winner, runner_up)

            assert "Synthesized" in response
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_response_without_runner_up(self):
        """Test synthesis with only winner (no runner-up)."""
        agent = DebateAgent()

        winner = Proposal(
            agent_role=AgentRole.HURRICANE_SPECIALIST,
            content="Winner content only",
            confidence=0.9,
            score=40,
        )

        response = await agent._synthesize_response("Test query", winner, None)

        # Should return winner content directly
        assert response == "Winner content only"


class TestDebateStats:
    """Tests for debate statistics."""

    def test_get_debate_stats_empty(self):
        """Test stats when no debates have been conducted."""
        agent = DebateAgent()

        stats = agent.get_debate_stats()

        assert stats["total_debates"] == 0
        assert stats["avg_duration_ms"] == 0
        assert stats["winner_distribution"] == {}

    @pytest.mark.asyncio
    async def test_get_debate_stats_with_data(self):
        """Test stats after conducting debates."""
        agent = DebateAgent(num_debaters=2, debate_rounds=1)

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content="Proposal. Confidence: HIGH")

            with patch.object(
                agent.judge_llm, "ainvoke", new_callable=AsyncMock
            ) as mock_judge:
                mock_judge.return_value = MagicMock(content='{"total_score": 35}')

                state = MultiAgentState(query="Test", user_id="test_user")

                # Run a debate
                await agent.run_debate(
                    state,
                    [AgentRole.FORECASTER, AgentRole.HURRICANE_SPECIALIST],
                )

                stats = agent.get_debate_stats()

                assert stats["total_debates"] >= 1
                assert stats["avg_duration_ms"] > 0


class TestFactoryFunction:
    """Tests for create_debate_agent factory function."""

    def test_create_with_defaults(self):
        """Test creating agent with default settings."""
        agent = create_debate_agent()

        assert isinstance(agent, DebateAgent)
        assert agent.num_debaters == 3

    def test_create_with_custom_settings(self):
        """Test creating agent with custom settings."""
        agent = create_debate_agent(
            model_name="gpt-4",
            num_debaters=5,
        )

        assert agent.num_debaters == 5


class TestInsufficientProposals:
    """Tests for handling insufficient proposals."""

    @pytest.mark.asyncio
    async def test_handle_zero_proposals(self):
        """Test handling when no proposals are generated."""
        agent = DebateAgent()

        with patch.object(agent.llm, "ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("All proposals failed")

            state = MultiAgentState(query="Test", user_id="test_user")

            result = await agent.run_debate(
                state,
                [AgentRole.FORECASTER, AgentRole.RESEARCH],
            )

            # Should have error response
            assert len(result.agent_responses) >= 1
            last_response = result.agent_responses[-1]
            assert last_response.confidence == 0.0 or "error" in str(
                last_response.metadata
            )

    @pytest.mark.asyncio
    async def test_handle_single_proposal(self):
        """Test handling when only one proposal is generated."""
        agent = DebateAgent()
        state = MultiAgentState(query="Test", user_id="test_user")

        # Only one proposal
        proposals = [
            Proposal(
                agent_role=AgentRole.FORECASTER,
                content="Single proposal",
                confidence=0.8,
            )
        ]

        result = agent._handle_insufficient_proposals(state, proposals)

        # Should use single proposal with reduced confidence
        assert len(result.agent_responses) >= 1
        response = result.agent_responses[-1]
        assert response.content == "Single proposal"
        assert response.confidence == 0.64  # 0.8 * 0.8
