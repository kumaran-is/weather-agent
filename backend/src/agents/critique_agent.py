"""Critique Agent for Level 4b: Generator → Critic → Refiner Workflow.

CRITICAL RULES:
1. Implements Generator → Critic → Refiner pattern
2. Critic uses cheaper model (gpt-4o-mini) for cost efficiency
3. Refiner only runs if critique score < 0.85
4. Track improvement metrics for quality assurance

Architecture:
- Generator: Produces initial response (or uses existing)
- Critic: Evaluates and provides specific feedback
- Refiner: Improves response based on feedback

Design Principles:
- Two-model approach (expensive generator, cheap critic)
- Conditional refinement (only when needed)
- Weather-specific evaluation criteria
- Detailed feedback for targeted improvements
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.src.models.multi_agent import (
    AgentRole,
    AgentResponse,
    MultiAgentState,
)
from backend.src.agents.prompts.reflection_prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    CRITIQUE_EVALUATION_PROMPT,
    REFINEMENT_PROMPT,
)

logger = structlog.get_logger(__name__)


class CritiqueAgent:
    """Critique and refinement agent implementing Generator → Critic → Refiner.

    The Critique Agent provides quality assurance through a three-phase workflow:
    1. Generator: Create initial response (or use existing)
    2. Critic: Evaluate response and provide specific feedback
    3. Refiner: Improve response if critique score < 0.85

    Cost Optimization:
    - Generator uses gpt-4o for quality initial responses
    - Critic uses gpt-4o-mini for cheap evaluation
    - Refiner only runs when needed (conditional)

    Weather-Specific Evaluation:
    - Saffir-Simpson scale accuracy
    - Time specificity (EDT/UTC)
    - Evacuation guidance accuracy
    - Life-safety information prominence

    Attributes:
        generator_llm: ChatOpenAI for response generation
        critic_llm: ChatOpenAI for evaluation (cheaper model)
        agent_role: Fixed as AgentRole.VERIFICATION
        refinement_threshold: Score below which refinement runs (default: 0.85)
    """

    def __init__(
        self,
        generator_model: str = "gpt-4o",
        critic_model: str = "gpt-4o-mini",
        refinement_threshold: float = 0.85,
    ) -> None:
        """Initialize Critique Agent.

        Args:
            generator_model: LLM for generation/refinement (default: gpt-4o)
            critic_model: LLM for critique (default: gpt-4o-mini for cost)
            refinement_threshold: Score below which to refine (default: 0.85)
        """
        self.generator_llm = ChatOpenAI(
            model=generator_model,
            temperature=0.3,  # Some creativity for generation
            timeout=30.0,
        )
        self.critic_llm = ChatOpenAI(
            model=critic_model,
            temperature=0.0,  # Deterministic critique
            timeout=15.0,  # Fast evaluation
        )
        self.agent_role = AgentRole.VERIFICATION
        self.refinement_threshold = refinement_threshold

        logger.info(
            "critique_agent_initialized",
            generator_model=generator_model,
            critic_model=critic_model,
            refinement_threshold=refinement_threshold,
        )

    async def critique_and_refine(
        self,
        state: MultiAgentState,
    ) -> MultiAgentState:
        """Run Generator → Critic → Refiner workflow.

        Steps:
        1. Generator: Use existing response or generate new
        2. Critic: Evaluate and provide specific feedback
        3. Refiner: Improve if score < threshold

        Args:
            state: Current multi-agent state

        Returns:
            Updated state with critique/refined response
        """
        start_time = time.perf_counter()

        logger.info(
            "critique_workflow_started",
            query=state.query[:50],
            has_existing_response=bool(state.agent_responses),
        )

        try:
            # Step 1: Get or generate initial response
            if state.agent_responses:
                # Use existing response
                valid_responses = [
                    r for r in state.agent_responses
                    if r.content and not r.metadata.get("error")
                ]
                if valid_responses:
                    generated = valid_responses[-1].content
                else:
                    generated = await self._generate(state.query)
            else:
                generated = await self._generate(state.query)

            logger.debug(
                "critique_generator_complete",
                response_length=len(generated),
            )

            # Step 2: Critique the response
            critique = await self._critique(generated, state.query)
            critique_score = critique["score"]

            logger.info(
                "critique_evaluation_complete",
                score=critique_score,
                strengths_count=len(critique.get("strengths", [])),
                weaknesses_count=len(critique.get("weaknesses", [])),
            )

            # Step 3: Refine if below threshold
            was_refined = False
            if critique_score < self.refinement_threshold:
                refined = await self._refine(
                    original=generated,
                    feedback=critique["feedback"],
                    critical_errors=critique.get("critical_errors", []),
                    additions=critique.get("suggested_additions", []),
                    query=state.query,
                )
                final_response = refined
                final_confidence = min(critique_score + 0.1, 0.95)  # Boost for refinement
                was_refined = True

                logger.info(
                    "critique_refinement_complete",
                    original_score=critique_score,
                    refined_confidence=final_confidence,
                )
            else:
                final_response = generated
                final_confidence = critique_score

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Create agent response
            critique_response = AgentResponse(
                agent_role=self.agent_role,
                content=final_response,
                confidence=final_confidence,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=duration_ms,
                metadata={
                    "critique_score": critique_score,
                    "was_refined": was_refined,
                    "feedback": critique["feedback"],
                    "strengths": critique.get("strengths", []),
                    "weaknesses": critique.get("weaknesses", []),
                    "critical_errors": critique.get("critical_errors", []),
                    "refinement_threshold": self.refinement_threshold,
                },
            )

            state.agent_responses.append(critique_response)
            state.quality_score = final_confidence

            logger.info(
                "critique_workflow_complete",
                final_confidence=final_confidence,
                was_refined=was_refined,
                duration_ms=duration_ms,
            )

            return state

        except Exception as e:
            logger.error(
                "critique_workflow_error",
                error=str(e),
                error_type=type(e).__name__,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Add error response
            error_response = AgentResponse(
                agent_role=self.agent_role,
                content=f"Critique workflow failed: {type(e).__name__}",
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=duration_ms,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            state.agent_responses.append(error_response)
            return state

    async def _generate(self, query: str) -> str:
        """Generate initial response.

        Args:
            query: User's query

        Returns:
            Generated response text
        """
        messages = [
            SystemMessage(
                content="""You are a weather expert. Provide accurate, helpful weather information.

Key requirements:
1. Use Saffir-Simpson scale accurately for hurricanes
2. Provide specific times (EDT/UTC), never "soon" or "later"
3. Use letter-based evacuation zones (A, B, C)
4. Put life-safety information first
5. Be clear and actionable"""
            ),
            HumanMessage(content=query),
        ]

        result = await self.generator_llm.ainvoke(messages)
        return result.content

    async def _critique(
        self,
        response: str,
        query: str,
    ) -> dict[str, Any]:
        """Critique the response.

        Evaluates response on:
        - Factual accuracy (Saffir-Simpson compliance)
        - Completeness (query fully answered)
        - Clarity (easy to understand)
        - Actionability (useful guidance)

        Args:
            response: Response text to critique
            query: Original query for context

        Returns:
            Dictionary with score, feedback, strengths, weaknesses
        """
        messages = [
            SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
            HumanMessage(
                content=CRITIQUE_EVALUATION_PROMPT.format(
                    query=query,
                    response=response,
                )
            ),
        ]

        try:
            result = await self.critic_llm.ainvoke(messages)
            critique = json.loads(result.content)

            # Normalize score
            score = float(critique.get("score", 0.7))
            score = max(0.0, min(1.0, score))

            return {
                "score": score,
                "feedback": critique.get("feedback", "No specific feedback"),
                "strengths": critique.get("strengths", []),
                "weaknesses": critique.get("weaknesses", []),
                "critical_errors": critique.get("critical_errors", []),
                "suggested_additions": critique.get("suggested_additions", []),
                "suggested_removals": critique.get("suggested_removals", []),
            }

        except json.JSONDecodeError:
            logger.warning("critique_json_parse_error")
            return {
                "score": 0.7,
                "feedback": "Unable to parse critique",
                "strengths": [],
                "weaknesses": ["Parse error"],
                "critical_errors": [],
                "suggested_additions": [],
                "suggested_removals": [],
            }

        except Exception as e:
            logger.error("critique_error", error=str(e))
            return {
                "score": 0.7,
                "feedback": f"Critique failed: {type(e).__name__}",
                "strengths": [],
                "weaknesses": [],
                "critical_errors": [],
                "suggested_additions": [],
                "suggested_removals": [],
            }

    async def _refine(
        self,
        original: str,
        feedback: str,
        critical_errors: list[str],
        additions: list[str],
        query: str,
    ) -> str:
        """Refine response based on critique feedback.

        Args:
            original: Original response text
            feedback: Critique feedback
            critical_errors: List of critical errors to fix
            additions: Suggested content to add
            query: Original query for context

        Returns:
            Refined response text
        """
        # Format critical errors
        errors_str = "\n".join(f"- {e}" for e in critical_errors) if critical_errors else "None"
        additions_str = "\n".join(f"- {a}" for a in additions) if additions else "None"

        messages = [
            SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
            HumanMessage(
                content=REFINEMENT_PROMPT.format(
                    query=query,
                    original=original,
                    feedback=feedback,
                    critical_errors=errors_str,
                    additions=additions_str,
                )
            ),
        ]

        try:
            result = await self.generator_llm.ainvoke(messages)
            return result.content

        except Exception as e:
            logger.error("refinement_error", error=str(e))
            # Return original on error
            return original

    async def quick_critique(
        self,
        response: str,
        query: str,
    ) -> dict[str, Any]:
        """Perform quick critique without refinement.

        Useful for validation without the full workflow.

        Args:
            response: Response to critique
            query: Original query

        Returns:
            Critique results with score and feedback
        """
        return await self._critique(response, query)
