"""Reflection Agent for Level 4b: Self-Critique and Improvement.

CRITICAL RULES:
1. Reflection improves responses through self-critique
2. Maximum 3 iterations (prevent infinite loops)
3. Track improvement metrics (quality delta)
4. Stop early if quality threshold reached (≥0.9)
5. Always validate against Saffir-Simpson scale

Architecture:
- Self-critique: Identify weaknesses in current response
- Improvement: Generate better response addressing weaknesses
- Iteration: Repeat until quality threshold or max iterations

Design Principles:
- Objective quality scoring (accuracy, completeness, clarity, actionability, safety)
- Weather-specific validation (Saffir-Simpson, time specificity, evacuation zones)
- Incremental improvement with measurable quality delta
- Early termination when quality threshold reached
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
    REFLECTION_SYSTEM_PROMPT,
    SELF_CRITIQUE_PROMPT,
    IMPROVEMENT_PROMPT,
    QUALITY_THRESHOLDS,
    MAX_REFLECTION_ITERATIONS,
)

logger = structlog.get_logger(__name__)


class ReflectionAgent:
    """Self-reflection and improvement agent.

    The Reflection Agent implements iterative quality improvement:
    1. Analyze current response for quality dimensions
    2. Identify specific weaknesses
    3. Generate improved response addressing weaknesses
    4. Repeat until quality threshold (0.9) or max iterations (3)

    Quality Dimensions:
    - Accuracy: Factual correctness, Saffir-Simpson compliance
    - Completeness: Query fully answered
    - Clarity: Easy to understand
    - Actionability: Useful guidance provided
    - Safety: Life-safety concerns addressed

    Attributes:
        llm: ChatOpenAI model for critique and improvement
        agent_role: Fixed as AgentRole.VERIFICATION
        max_iterations: Maximum reflection cycles (default: 3)
        quality_threshold: Target quality score (default: 0.9)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        max_iterations: int = MAX_REFLECTION_ITERATIONS,
        quality_threshold: float = QUALITY_THRESHOLDS["excellent"],
    ) -> None:
        """Initialize Reflection Agent.

        Args:
            model_name: LLM model for critique/improvement (default: gpt-4o for accuracy)
            max_iterations: Maximum reflection iterations (default: 3)
            quality_threshold: Quality score to achieve before stopping (default: 0.9)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Slight creativity for improvement
            timeout=30.0,  # Allow time for quality analysis
        )
        self.agent_role = AgentRole.VERIFICATION
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

        logger.info(
            "reflection_agent_initialized",
            model=model_name,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
        )

    async def reflect_and_improve(
        self,
        state: MultiAgentState,
    ) -> MultiAgentState:
        """Reflect on response and improve iteratively.

        Steps:
        1. Get current response to reflect on
        2. Self-critique: Identify weaknesses
        3. Improve: Generate better response
        4. Repeat until quality threshold or max iterations

        Args:
            state: Current multi-agent state with response to improve

        Returns:
            Updated state with improved response and quality metrics
        """
        start_time = time.perf_counter()

        # Get response to reflect on
        if not state.agent_responses:
            logger.warning("no_responses_to_reflect")
            return state

        current_response = state.agent_responses[-1]
        if current_response.metadata.get("error"):
            logger.warning("skipping_error_response")
            return state

        best_response = current_response
        best_quality = current_response.confidence
        iteration = 0
        quality_history: list[float] = [best_quality]

        logger.info(
            "reflection_started",
            query=state.query[:50],
            initial_quality=best_quality,
            max_iterations=self.max_iterations,
        )

        while iteration < self.max_iterations:
            iteration += 1

            # Step 1: Self-critique
            critique = await self._self_critique(best_response.content, state.query)
            quality_score = critique["quality_score"]
            quality_history.append(quality_score)

            logger.debug(
                "reflection_critique",
                iteration=iteration,
                quality_score=quality_score,
                weaknesses_count=len(critique.get("weaknesses", [])),
            )

            # Step 2: Check if quality threshold reached
            if quality_score >= self.quality_threshold:
                logger.info(
                    "reflection_threshold_reached",
                    iteration=iteration,
                    quality=quality_score,
                    threshold=self.quality_threshold,
                )
                break

            # Step 3: Generate improved response
            weaknesses = critique.get("weaknesses", [])
            if not weaknesses:
                logger.info(
                    "no_weaknesses_found",
                    iteration=iteration,
                    quality=quality_score,
                )
                break

            improved_content = await self._improve_response(
                best_response.content,
                weaknesses,
                state.query,
            )

            # Step 4: Score improved response
            improved_critique = await self._self_critique(improved_content, state.query)
            improved_quality = improved_critique["quality_score"]

            logger.debug(
                "reflection_improvement",
                iteration=iteration,
                quality_before=quality_score,
                quality_after=improved_quality,
                delta=improved_quality - quality_score,
            )

            # Step 5: Keep if improved
            if improved_quality > best_quality:
                best_response = AgentResponse(
                    agent_role=self.agent_role,
                    content=improved_content,
                    confidence=improved_quality,
                    timestamp=datetime.now(timezone.utc),
                    execution_time_ms=0,  # Updated at end
                    metadata={
                        "iteration": iteration,
                        "improvements_made": weaknesses,
                        "quality_delta": improved_quality - best_quality,
                        "critique_summary": {
                            "accuracy": improved_critique.get("accuracy", 0),
                            "completeness": improved_critique.get("completeness", 0),
                            "clarity": improved_critique.get("clarity", 0),
                            "actionability": improved_critique.get("actionability", 0),
                            "safety": improved_critique.get("safety", 0),
                        },
                    },
                )
                best_quality = improved_quality
            else:
                # No improvement, stop iterating
                logger.info(
                    "no_quality_improvement",
                    iteration=iteration,
                    best_quality=best_quality,
                    attempted_quality=improved_quality,
                )
                break

        # Finalize
        duration_ms = (time.perf_counter() - start_time) * 1000
        best_response.execution_time_ms = duration_ms

        # Update metadata with reflection summary
        best_response.metadata["reflection_summary"] = {
            "iterations": iteration,
            "initial_quality": quality_history[0] if quality_history else 0,
            "final_quality": best_quality,
            "total_improvement": best_quality - (quality_history[0] if quality_history else 0),
            "quality_history": quality_history,
            "threshold_reached": best_quality >= self.quality_threshold,
        }

        # Add reflection response to state
        state.agent_responses.append(best_response)
        state.quality_score = best_quality
        state.reflection_iterations = iteration

        logger.info(
            "reflection_complete",
            iterations=iteration,
            initial_quality=quality_history[0] if quality_history else 0,
            final_quality=best_quality,
            duration_ms=duration_ms,
            threshold_reached=best_quality >= self.quality_threshold,
        )

        return state

    async def _self_critique(
        self,
        response: str,
        original_query: str,
    ) -> dict[str, Any]:
        """Perform self-critique on response.

        Evaluates response across quality dimensions:
        - Accuracy (Saffir-Simpson compliance)
        - Completeness (query answered)
        - Clarity (easy to understand)
        - Actionability (useful guidance)
        - Safety (life-safety addressed)

        Args:
            response: Response text to critique
            original_query: User's original query

        Returns:
            Dictionary with quality scores and identified weaknesses
        """
        critique_messages = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(
                content=SELF_CRITIQUE_PROMPT.format(
                    query=original_query,
                    response=response,
                )
            ),
        ]

        try:
            result = await self.llm.ainvoke(critique_messages)

            # Parse JSON response
            critique = json.loads(result.content)

            # Validate and normalize scores
            quality_score = float(critique.get("quality_score", 0.7))
            quality_score = max(0.0, min(1.0, quality_score))

            return {
                "quality_score": quality_score,
                "accuracy": float(critique.get("accuracy", 0.7)),
                "completeness": float(critique.get("completeness", 0.7)),
                "clarity": float(critique.get("clarity", 0.7)),
                "actionability": float(critique.get("actionability", 0.7)),
                "safety": float(critique.get("safety", 0.7)),
                "strengths": critique.get("strengths", []),
                "weaknesses": critique.get("weaknesses", []),
                "specific_issues": critique.get("specific_issues", []),
                "improvement_suggestions": critique.get("improvement_suggestions", []),
            }

        except json.JSONDecodeError as e:
            logger.warning(
                "critique_json_parse_error",
                error=str(e),
            )
            return {
                "quality_score": 0.7,
                "accuracy": 0.7,
                "completeness": 0.7,
                "clarity": 0.7,
                "actionability": 0.7,
                "safety": 0.7,
                "strengths": [],
                "weaknesses": ["Unable to parse critique response"],
                "specific_issues": [],
                "improvement_suggestions": [],
            }

        except Exception as e:
            logger.error(
                "critique_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "quality_score": 0.7,
                "weaknesses": [f"Critique failed: {type(e).__name__}"],
                "strengths": [],
                "specific_issues": [],
                "improvement_suggestions": [],
            }

    async def _improve_response(
        self,
        current_response: str,
        weaknesses: list[str],
        original_query: str,
    ) -> str:
        """Generate improved response addressing weaknesses.

        Args:
            current_response: Current response text
            weaknesses: List of identified weaknesses
            original_query: User's original query

        Returns:
            Improved response text
        """
        # Format weaknesses for prompt
        weaknesses_str = "\n".join(f"- {w}" for w in weaknesses)

        improvement_messages = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(
                content=IMPROVEMENT_PROMPT.format(
                    query=original_query,
                    current_response=current_response,
                    weaknesses=weaknesses_str,
                )
            ),
        ]

        try:
            result = await self.llm.ainvoke(improvement_messages)
            return result.content

        except Exception as e:
            logger.error(
                "improvement_generation_error",
                error=str(e),
            )
            # Return original on error
            return current_response

    async def verify_saffir_simpson(self, response: str) -> dict[str, Any]:
        """Verify Saffir-Simpson scale compliance in response.

        Checks that hurricane categories match wind speeds.

        Args:
            response: Response text to verify

        Returns:
            Dictionary with verification results
        """
        import re

        errors = []

        # Saffir-Simpson scale reference
        scale = {
            1: (74, 95),
            2: (96, 110),
            3: (111, 129),
            4: (130, 156),
            5: (157, 999),
        }

        # Find category mentions
        cat_pattern = r"category\s*(\d)|cat\s*(\d)"
        wind_pattern = r"(\d{2,3})\s*mph"

        categories = re.findall(cat_pattern, response.lower())
        wind_speeds = re.findall(wind_pattern, response.lower())

        for cat_match in categories:
            category = int(cat_match[0] or cat_match[1])

            if category in scale:
                min_wind, max_wind = scale[category]

                # Check if any mentioned wind speed is inconsistent
                for wind_str in wind_speeds:
                    wind = int(wind_str)

                    # Determine correct category for wind speed
                    correct_cat = 0
                    for cat, (min_w, max_w) in scale.items():
                        if min_w <= wind <= max_w:
                            correct_cat = cat
                            break

                    if correct_cat != category and wind < min_wind:
                        errors.append(
                            f"Category {category} stated but {wind} mph suggests Category {correct_cat}"
                        )

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "categories_found": [int(c[0] or c[1]) for c in categories],
            "wind_speeds_found": [int(w) for w in wind_speeds],
        }
