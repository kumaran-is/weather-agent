"""Pillar 1: Effectiveness Judge (LLM-as-Judge).

This module implements LLM-based answer correctness evaluation using
structured output and chain-of-thought reasoning.

Evaluation Criteria (each 0.0-1.0):
1. Correctness: Is the information accurate?
2. Completeness: Does it answer the full question?
3. Relevance: Is the response on-topic?
4. Clarity: Is the answer understandable?

Final Score = average of all 4 criteria

LangSmith Integration:
- All LLM calls are automatically traced via LangSmith
- Supports custom evaluator binding to LangSmith datasets
- Reasoning chains visible in LangSmith UI

Usage:
    >>> judge = EffectivenessJudge(llm=ChatOpenAI(model="gpt-4o-mini"))
    >>> result = await judge.evaluate(
    ...     query="What's the weather in Miami?",
    ...     actual_answer="It's sunny, 85°F in Miami",
    ...     expected_answer="Miami: 85°F, sunny"
    ... )
    >>> print(f"Score: {result.score}, Reasoning: {result.reasoning}")
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from backend.src.evaluation.models import EffectivenessResult

logger = logging.getLogger(__name__)


class JudgeOutput(BaseModel):
    """Structured output from LLM judge."""

    correctness: float = Field(ge=0.0, le=1.0, description="Information accuracy (0-1)")
    completeness: float = Field(ge=0.0, le=1.0, description="Answer completeness (0-1)")
    relevance: float = Field(ge=0.0, le=1.0, description="Response relevance (0-1)")
    clarity: float = Field(ge=0.0, le=1.0, description="Answer clarity (0-1)")
    reasoning: str = Field(description="Chain-of-thought reasoning for scores")


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a Weather AI Agent.
Your task is to judge the quality of AI responses to weather-related queries.

EVALUATION CRITERIA (score each 0.0 to 1.0):

1. CORRECTNESS (0.0-1.0): Is the information accurate?
   - 1.0: All facts correct, proper units, accurate data
   - 0.7: Minor inaccuracies that don't affect main point
   - 0.4: Some significant errors but partially correct
   - 0.0: Major factual errors or completely wrong

2. COMPLETENESS (0.0-1.0): Does it answer the full question?
   - 1.0: Fully answers all parts of the query
   - 0.7: Answers main question but misses some details
   - 0.4: Partially answers but missing key information
   - 0.0: Doesn't address the query at all

3. RELEVANCE (0.0-1.0): Is the response on-topic?
   - 1.0: Directly addresses the query, no irrelevant content
   - 0.7: Mostly relevant with minor tangents
   - 0.4: Contains relevant info but much irrelevant content
   - 0.0: Completely off-topic

4. CLARITY (0.0-1.0): Is the answer understandable?
   - 1.0: Clear, well-organized, easy to understand
   - 0.7: Generally clear with minor confusion
   - 0.4: Somewhat confusing or poorly organized
   - 0.0: Incomprehensible or extremely confusing

WEATHER-SPECIFIC CONSIDERATIONS:
- Hurricane categories MUST match Saffir-Simpson scale (Cat 1: 74-95 mph, Cat 5: 157+ mph)
- Temperature should be in sensible range for location/season
- Evacuation advice should be appropriate for the situation
- Safety-critical information must be accurate

OUTPUT FORMAT:
Return a JSON object with these exact fields:
{
  "correctness": <float 0-1>,
  "completeness": <float 0-1>,
  "relevance": <float 0-1>,
  "clarity": <float 0-1>,
  "reasoning": "<explain your scoring with specific examples>"
}

Be strict but fair. Weather information affects life-safety decisions."""


class EffectivenessJudge:
    """LLM-as-Judge for answer correctness evaluation.

    Uses structured output parsing to get consistent scores
    and chain-of-thought reasoning for explainability.

    Attributes:
        llm: Language model for judging
        parser: JSON output parser
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize judge with LLM.

        Args:
            llm: Language model (e.g., gpt-4o-mini, claude-3-haiku)
        """
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=JudgeOutput)
        self.last_raw_response: str = ""

    async def evaluate(
        self,
        query: str,
        actual_answer: str,
        expected_answer: str | None = None,
    ) -> EffectivenessResult:
        """Evaluate answer effectiveness using LLM-as-Judge.

        Args:
            query: Original user query
            actual_answer: Agent's response to evaluate
            expected_answer: Optional reference answer

        Returns:
            EffectivenessResult with scores and reasoning
        """
        # Build evaluation prompt
        user_prompt = self._build_prompt(query, actual_answer, expected_answer)

        try:
            # Call LLM with structured output
            messages = [
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response = await self.llm.ainvoke(messages)
            self.last_raw_response = response.content

            # Parse structured output
            parsed = self.parser.parse(response.content)

            # Calculate overall score (average of 4 criteria)
            score = (
                parsed["correctness"] +
                parsed["completeness"] +
                parsed["relevance"] +
                parsed["clarity"]
            ) / 4.0

            return EffectivenessResult(
                score=score,
                correctness=parsed["correctness"],
                completeness=parsed["completeness"],
                relevance=parsed["relevance"],
                clarity=parsed["clarity"],
                reasoning=parsed["reasoning"],
                expected_answer=expected_answer,
                actual_answer=actual_answer,
            )

        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            # Return conservative failure score
            return EffectivenessResult(
                score=0.0,
                correctness=0.0,
                completeness=0.0,
                relevance=0.0,
                clarity=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                expected_answer=expected_answer,
                actual_answer=actual_answer,
            )

    def _build_prompt(
        self,
        query: str,
        actual_answer: str,
        expected_answer: str | None,
    ) -> str:
        """Build evaluation prompt for LLM judge."""
        prompt_parts = [
            "Please evaluate the following AI response:\n",
            f"USER QUERY:\n{query}\n",
            f"AI RESPONSE:\n{actual_answer}\n",
        ]

        if expected_answer:
            prompt_parts.append(f"REFERENCE ANSWER (for comparison):\n{expected_answer}\n")
        else:
            prompt_parts.append(
                "Note: No reference answer provided. "
                "Evaluate based on general accuracy and quality.\n"
            )

        prompt_parts.append(
            "\nProvide your evaluation as JSON with scores and reasoning."
        )

        return "".join(prompt_parts)

    def evaluate_sync(
        self,
        query: str,
        actual_answer: str,
        expected_answer: str | None = None,
    ) -> EffectivenessResult:
        """Synchronous wrapper for evaluate (for non-async contexts).

        Note: Prefer async version when possible.
        """
        import asyncio
        return asyncio.run(self.evaluate(query, actual_answer, expected_answer))
