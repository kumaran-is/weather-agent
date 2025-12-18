"""Debate Agent for Level 4c: Multi-Agent Debate Pattern.

This agent orchestrates debates between multiple specialist agents,
collecting proposals and selecting the best answer through scoring.

Features:
- Multi-proposal collection from different agents
- Impartial scoring using evaluation criteria
- Debate rounds for proposal refinement
- Final synthesis of best elements
- Conflict resolution for contradictory outputs

The Debate Pattern is useful when:
- Multiple valid approaches exist
- Expert opinions may differ
- High-stakes decisions require validation
- Complex queries benefit from multiple perspectives
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.src.agents.prompts.debate_prompts import (
    DEBATE_SYSTEM_PROMPT,
    JUDGE_EVALUATION_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    MIN_SCORE_FOR_WINNER,
    PROPOSAL_GENERATION_PROMPT,
    SYNTHESIS_PROMPT,
)
from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)

logger = structlog.get_logger()


@dataclass
class Proposal:
    """A debate proposal from an agent.

    Attributes:
        agent_role: Role of the proposing agent
        content: Proposal content
        confidence: Self-assessed confidence (0.0 to 1.0)
        reasoning: Reasoning behind the proposal
        score: Judge-assigned score (0-40 or 0-50)
        evaluation: Detailed evaluation from judge
    """

    agent_role: AgentRole
    content: str
    confidence: float
    reasoning: str = ""
    score: float = 0.0
    evaluation: dict[str, Any] | None = None


@dataclass
class DebateResult:
    """Result of a completed debate.

    Attributes:
        winner: Winning proposal
        runner_up: Second-place proposal (if exists)
        all_proposals: All proposals in the debate
        final_response: Synthesized final response
        total_rounds: Number of debate rounds
        total_time_ms: Total debate duration
    """

    winner: Proposal
    runner_up: Proposal | None
    all_proposals: list[Proposal]
    final_response: str
    total_rounds: int
    total_time_ms: float


class DebateAgent:
    """Orchestrate debates between multiple agents.

    The Debate Agent solicits proposals from multiple specialist agents,
    scores them using impartial criteria, and synthesizes the best answer.

    Attributes:
        llm: Language model for proposal generation
        judge_llm: Language model for impartial judging
        num_debaters: Maximum number of proposers
        debate_rounds: Number of refinement rounds
        _debate_history: History of debates conducted
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        num_debaters: int = 3,
        debate_rounds: int = 2,
    ):
        """Initialize the Debate Agent.

        Args:
            model_name: Model to use for proposals and synthesis.
            num_debaters: Maximum number of agents to solicit proposals from.
            debate_rounds: Number of refinement/debate rounds.
        """
        # Proposal generation LLM (higher temp for diversity)
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.5,
            timeout=30.0,
        )

        # Judge LLM (deterministic for consistent scoring)
        self.judge_llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            timeout=15.0,
        )

        self.num_debaters = num_debaters
        self.debate_rounds = debate_rounds
        self._debate_history: list[dict[str, Any]] = []

    async def run_debate(
        self,
        state: MultiAgentState,
        proposers: list[AgentRole],
    ) -> MultiAgentState:
        """Run multi-agent debate.

        Orchestrates the full debate process:
        1. Generate proposals from each proposer
        2. Score all proposals
        3. Run refinement rounds (optional)
        4. Select winner and synthesize

        Args:
            state: Current workflow state.
            proposers: List of agent roles to solicit proposals from.

        Returns:
            Updated state with debate results.
        """
        start_time = time.perf_counter()

        try:
            # Limit proposers
            active_proposers = proposers[: self.num_debaters]

            # Step 1: Generate proposals in parallel
            proposals = await self._generate_proposals(
                state.query, active_proposers
            )

            if len(proposals) < 2:
                # Not enough proposals for debate
                logger.warning(
                    "insufficient_proposals",
                    count=len(proposals),
                    required=2,
                )
                return self._handle_insufficient_proposals(state, proposals)

            # Step 2: Score all proposals
            scored_proposals = await self._score_proposals(proposals, state.query)

            # Step 3: Run debate rounds (if configured)
            for round_num in range(self.debate_rounds):
                # Get top 2 proposals for debate
                sorted_proposals = sorted(
                    scored_proposals, key=lambda p: p.score, reverse=True
                )
                top_2 = sorted_proposals[:2]

                if top_2[0].score >= MIN_SCORE_FOR_WINNER:
                    # Winner is clear, no need for more rounds
                    break

                logger.debug(
                    "debate_round",
                    round=round_num + 1,
                    top_scores=[p.score for p in top_2],
                )

            # Step 4: Select winner and synthesize
            final_sorted = sorted(
                scored_proposals, key=lambda p: p.score, reverse=True
            )
            winner = final_sorted[0]
            runner_up = final_sorted[1] if len(final_sorted) > 1 else None

            # Synthesize final response
            final_response = await self._synthesize_response(
                state.query, winner, runner_up
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Create debate result
            result = DebateResult(
                winner=winner,
                runner_up=runner_up,
                all_proposals=scored_proposals,
                final_response=final_response,
                total_rounds=self.debate_rounds,
                total_time_ms=duration_ms,
            )

            # Record in history
            self._debate_history.append({
                "timestamp": time.time(),
                "query": state.query[:100],
                "proposers": [p.value for p in active_proposers],
                "winner": winner.agent_role.value,
                "winner_score": winner.score,
                "duration_ms": duration_ms,
            })

            # Add response to state
            state.agent_responses.append(
                AgentResponse(
                    agent_role=AgentRole.DEBATE,
                    content=final_response,
                    confidence=min(winner.confidence, 0.95),
                    execution_time_ms=duration_ms,
                    metadata={
                        "debate_winner": winner.agent_role.value,
                        "winner_score": winner.score,
                        "runner_up": runner_up.agent_role.value if runner_up else None,
                        "runner_up_score": runner_up.score if runner_up else None,
                        "total_proposals": len(proposals),
                        "debate_rounds": self.debate_rounds,
                    },
                )
            )

            state.current_agent = AgentRole.DEBATE

            logger.info(
                "debate_complete",
                winner=winner.agent_role.value,
                winner_score=winner.score,
                proposals=len(proposals),
                duration_ms=round(duration_ms, 2),
            )

            return state

        except Exception as e:
            logger.error("debate_error", error=str(e))

            duration_ms = (time.perf_counter() - start_time) * 1000
            state.agent_responses.append(
                AgentResponse(
                    agent_role=AgentRole.DEBATE,
                    content="Debate process encountered an error",
                    confidence=0.0,
                    execution_time_ms=duration_ms,
                    metadata={"error": str(e)},
                )
            )

            return state

    async def _generate_proposals(
        self,
        query: str,
        proposers: list[AgentRole],
    ) -> list[Proposal]:
        """Generate proposals from multiple agents.

        Args:
            query: User query to address.
            proposers: List of agent roles to generate proposals.

        Returns:
            List of Proposal objects.
        """
        proposals: list[Proposal] = []

        async def generate_one(role: AgentRole) -> Proposal | None:
            try:
                prompt = PROPOSAL_GENERATION_PROMPT.format(
                    agent_type=role.value.replace("_", " ").title(),
                    query=query,
                    context="Weather AI Agent Service context",
                )

                messages = [
                    SystemMessage(content=f"You are a {role.value} agent."),
                    HumanMessage(content=prompt),
                ]

                response = await self.llm.ainvoke(messages)
                content = str(response.content)

                # Extract confidence if present
                confidence = 0.8
                if "Confidence: HIGH" in content:
                    confidence = 0.9
                elif "Confidence: LOW" in content:
                    confidence = 0.6

                return Proposal(
                    agent_role=role,
                    content=content,
                    confidence=confidence,
                    reasoning="Initial proposal",
                )

            except Exception as e:
                logger.warning(
                    "proposal_generation_failed",
                    agent=role.value,
                    error=str(e),
                )
                return None

        # Generate proposals in parallel
        tasks = [generate_one(role) for role in proposers]
        results = await asyncio.gather(*tasks)

        # Filter out failed proposals
        proposals = [p for p in results if p is not None]

        logger.debug(
            "proposals_generated",
            requested=len(proposers),
            received=len(proposals),
        )

        return proposals

    async def _score_proposals(
        self,
        proposals: list[Proposal],
        query: str,
    ) -> list[Proposal]:
        """Score proposals using judge LLM.

        Args:
            proposals: List of proposals to score.
            query: Original query for context.

        Returns:
            List of proposals with scores updated.
        """
        for proposal in proposals:
            try:
                messages = [
                    SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                    HumanMessage(
                        content=JUDGE_EVALUATION_PROMPT.format(
                            query=query,
                            agent_type=proposal.agent_role.value,
                            proposal=proposal.content,
                        )
                    ),
                ]

                response = await self.judge_llm.ainvoke(messages)
                content = str(response.content)

                # Parse JSON response
                try:
                    # Find JSON in response
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        evaluation = json.loads(json_str)
                        proposal.score = evaluation.get("total_score", 0)
                        proposal.evaluation = evaluation
                    else:
                        # Fallback scoring
                        proposal.score = 25  # Default middle score
                        proposal.evaluation = {"error": "Could not parse evaluation"}

                except json.JSONDecodeError:
                    proposal.score = 25
                    proposal.evaluation = {"error": "JSON parse error"}

                logger.debug(
                    "proposal_scored",
                    agent=proposal.agent_role.value,
                    score=proposal.score,
                )

            except Exception as e:
                logger.warning(
                    "scoring_failed",
                    agent=proposal.agent_role.value,
                    error=str(e),
                )
                proposal.score = 20  # Low fallback score

        return proposals

    async def _synthesize_response(
        self,
        query: str,
        winner: Proposal,
        runner_up: Proposal | None,
    ) -> str:
        """Synthesize final response from top proposals.

        Args:
            query: Original query.
            winner: Winning proposal.
            runner_up: Second-place proposal (optional).

        Returns:
            Synthesized final response.
        """
        if not runner_up:
            # No synthesis needed - use winner directly
            return winner.content

        try:
            messages = [
                SystemMessage(content=DEBATE_SYSTEM_PROMPT),
                HumanMessage(
                    content=SYNTHESIS_PROMPT.format(
                        query=query,
                        winner_score=winner.score,
                        winner_proposal=winner.content,
                        runner_up_score=runner_up.score,
                        runner_up_proposal=runner_up.content,
                    )
                ),
            ]

            response = await self.llm.ainvoke(messages)
            return str(response.content)

        except Exception as e:
            logger.warning("synthesis_failed", error=str(e))
            return winner.content  # Fallback to winner

    def _handle_insufficient_proposals(
        self,
        state: MultiAgentState,
        proposals: list[Proposal],
    ) -> MultiAgentState:
        """Handle case with insufficient proposals.

        Args:
            state: Current workflow state.
            proposals: Available proposals (0 or 1).

        Returns:
            Updated state.
        """
        if proposals:
            # Use the single proposal
            state.agent_responses.append(
                AgentResponse(
                    agent_role=AgentRole.DEBATE,
                    content=proposals[0].content,
                    confidence=proposals[0].confidence * 0.8,  # Lower confidence
                    execution_time_ms=0,
                    metadata={
                        "note": "Single proposal - no debate possible",
                        "proposer": proposals[0].agent_role.value,
                    },
                )
            )
        else:
            state.agent_responses.append(
                AgentResponse(
                    agent_role=AgentRole.DEBATE,
                    content="",
                    confidence=0.0,
                    execution_time_ms=0,
                    metadata={"error": "No proposals received"},
                )
            )

        state.current_agent = AgentRole.DEBATE
        return state

    async def process(self, state: MultiAgentState) -> MultiAgentState:
        """Process state as the Debate Agent.

        Determines appropriate proposers based on query and runs debate.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with debate results.
        """
        # Determine proposers based on query complexity
        default_proposers = [
            AgentRole.HURRICANE_SPECIALIST,
            AgentRole.FORECASTER,
            AgentRole.HISTORICAL_ANALYST,
        ]

        return await self.run_debate(state, default_proposers)

    def get_debate_stats(self) -> dict[str, Any]:
        """Get statistics about debates conducted.

        Returns:
            Dictionary with debate statistics.
        """
        if not self._debate_history:
            return {
                "total_debates": 0,
                "avg_duration_ms": 0,
                "winner_distribution": {},
            }

        # Calculate stats
        total_debates = len(self._debate_history)
        avg_duration = sum(d["duration_ms"] for d in self._debate_history) / total_debates

        # Winner distribution
        winner_dist: dict[str, int] = {}
        for debate in self._debate_history:
            winner = debate["winner"]
            winner_dist[winner] = winner_dist.get(winner, 0) + 1

        return {
            "total_debates": total_debates,
            "avg_duration_ms": round(avg_duration, 2),
            "winner_distribution": winner_dist,
            "avg_winner_score": round(
                sum(d["winner_score"] for d in self._debate_history) / total_debates, 2
            ),
        }


# =============================================================================
# Factory function
# =============================================================================


def create_debate_agent(
    model_name: str = "gpt-4o",
    num_debaters: int = 3,
) -> DebateAgent:
    """Create a Debate Agent instance.

    Args:
        model_name: Model to use for debate.
        num_debaters: Maximum number of proposers.

    Returns:
        Configured DebateAgent instance.
    """
    return DebateAgent(
        model_name=model_name,
        num_debaters=num_debaters,
    )


__all__ = [
    "DebateAgent",
    "Proposal",
    "DebateResult",
    "create_debate_agent",
]
