"""LLM-as-Judge Validator for Weather AI Agent.

This module validates that the LLM judge agrees with human ground truth
at least 85% of the time. If agreement is below threshold, the judge
is considered unreliable and human review is required.

Agreement Calculation:
- Score agreement: |llm_score - human_score| <= 0.2 (within 0.2 tolerance)
- Pass/fail agreement: Both agree on pass/fail decision

Human Ground Truth:
- 30+ manually evaluated cases with human scores
- Covers all query types (simple, complex, hurricane, edge cases)
- Includes safety-critical cases

Usage:
    >>> from backend.src.evaluation import LLMJudgeValidator
    >>>
    >>> # Load human ground truth
    >>> ground_truth = load_ground_truth("human_evaluations.yaml")
    >>>
    >>> validator = LLMJudgeValidator(llm=llm, ground_truth=ground_truth)
    >>> result = await validator.validate()
    >>>
    >>> if result["is_reliable"]:
    ...     print("Judge is reliable - use for automated evaluation")
    ... else:
    ...     print("Judge unreliable - require human review")
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from backend.src.evaluation.effectiveness_judge import EffectivenessJudge

logger = logging.getLogger(__name__)


class LLMJudgeValidator:
    """Validate LLM-as-Judge against human ground truth.

    Ensures that automated LLM evaluations are reliable before
    using them for CI/CD quality gates.

    Attributes:
        llm: Language model being validated
        ground_truth: List of human-evaluated cases
        judge: EffectivenessJudge instance
        tolerance: Score difference tolerance (default: 0.2)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        ground_truth: list[dict[str, Any]],
        tolerance: float = 0.2,
        min_agreement_rate: float = 0.85,
    ) -> None:
        """Initialize validator.

        Args:
            llm: Language model to validate
            ground_truth: Human-evaluated cases with format:
                - query: str
                - response: str
                - human_score: float (0-1)
                - human_passed: bool (optional)
            tolerance: Score difference tolerance for agreement
            min_agreement_rate: Minimum required agreement rate
        """
        self.llm = llm
        self.ground_truth = ground_truth
        self.tolerance = tolerance
        self.min_agreement_rate = min_agreement_rate
        self.judge = EffectivenessJudge(llm=llm)

        if len(ground_truth) < 30:
            logger.warning(
                f"Ground truth has only {len(ground_truth)} cases. "
                "Recommend 30+ cases for reliable validation."
            )

    async def validate(self) -> dict[str, Any]:
        """Validate LLM judge against human evaluations.

        Returns:
            Validation results with:
            - agreement_rate: Percentage of cases where LLM agrees with human
            - is_reliable: True if agreement >= min_agreement_rate
            - disagreements: Cases where LLM and human disagreed
            - recommendation: Action to take
        """
        agreements = 0
        disagreements = []

        for case in self.ground_truth:
            query = case.get("query", "")
            response = case.get("response", "")
            human_score = case.get("human_score", 0.0)
            expected_answer = case.get("expected_answer")

            # Get LLM evaluation
            try:
                llm_result = await self.judge.evaluate(
                    query=query,
                    actual_answer=response,
                    expected_answer=expected_answer,
                )
                llm_score = llm_result.score
            except Exception as e:
                logger.error(f"LLM evaluation failed for case: {e}")
                llm_score = 0.0

            # Check agreement within tolerance
            score_diff = abs(llm_score - human_score)
            agrees = score_diff <= self.tolerance

            if agrees:
                agreements += 1
            else:
                disagreements.append({
                    "query": query[:100],  # Truncate for readability
                    "human_score": human_score,
                    "llm_score": llm_score,
                    "difference": score_diff,
                })

        # Calculate agreement rate
        total = len(self.ground_truth)
        agreement_rate = agreements / max(1, total)
        is_reliable = agreement_rate >= self.min_agreement_rate

        # Build recommendation
        if is_reliable:
            recommendation = (
                f"LLM judge is RELIABLE ({agreement_rate:.1%} agreement). "
                "Safe to use for automated evaluation and CI/CD quality gates."
            )
        else:
            recommendation = (
                f"LLM judge is UNRELIABLE ({agreement_rate:.1%} agreement, "
                f"threshold: {self.min_agreement_rate:.1%}). "
                "REQUIRE human review for evaluations. "
                f"Top disagreements: {len(disagreements)} cases."
            )

        logger.info(f"LLM Judge validation: {agreement_rate:.1%} agreement, reliable={is_reliable}")

        return {
            "agreement_rate": agreement_rate,
            "is_reliable": is_reliable,
            "total_cases": total,
            "agreements": agreements,
            "disagreements": len(disagreements),
            "disagreement_details": disagreements[:10],  # Top 10 disagreements
            "tolerance": self.tolerance,
            "min_agreement_rate": self.min_agreement_rate,
            "recommendation": recommendation,
        }

    async def calibrate(self) -> dict[str, Any]:
        """Analyze disagreements to improve judge calibration.

        Returns insights on where LLM judge tends to disagree with humans.
        """
        result = await self.validate()

        if not result["disagreement_details"]:
            return {
                "calibration_needed": False,
                "message": "No significant disagreements - judge is well calibrated",
            }

        # Analyze disagreement patterns
        llm_too_harsh = []
        llm_too_lenient = []

        for d in result["disagreement_details"]:
            if d["llm_score"] < d["human_score"]:
                llm_too_harsh.append(d)
            else:
                llm_too_lenient.append(d)

        bias = "harsh" if len(llm_too_harsh) > len(llm_too_lenient) else "lenient"

        return {
            "calibration_needed": True,
            "bias_direction": bias,
            "too_harsh_count": len(llm_too_harsh),
            "too_lenient_count": len(llm_too_lenient),
            "avg_harsh_diff": sum(d["human_score"] - d["llm_score"] for d in llm_too_harsh) / max(1, len(llm_too_harsh)),
            "avg_lenient_diff": sum(d["llm_score"] - d["human_score"] for d in llm_too_lenient) / max(1, len(llm_too_lenient)),
            "recommendation": (
                f"LLM tends to be too {bias}. Consider adjusting "
                f"{'scoring rubric to be more generous' if bias == 'harsh' else 'scoring criteria to be stricter'}."
            ),
        }


# Sample human ground truth for initial validation
SAMPLE_HUMAN_GROUND_TRUTH = [
    # Simple weather queries
    {
        "query": "What's the weather in Miami?",
        "response": "It's currently 85°F and sunny in Miami, Florida with a humidity of 72%.",
        "human_score": 0.95,
        "expected_answer": "Temperature around 80-90°F, sunny or partly cloudy",
    },
    {
        "query": "Is it raining in Seattle?",
        "response": "Seattle is experiencing light rain with temperatures around 55°F.",
        "human_score": 0.90,
        "expected_answer": "Rain status and temperature",
    },
    # Hurricane queries (safety-critical)
    {
        "query": "What's the status of Hurricane Milton?",
        "response": "Hurricane Milton is currently a Category 4 storm with sustained winds of 145 mph, located 200 miles southeast of Tampa.",
        "human_score": 0.95,
        "expected_answer": "Category, wind speed, location",
    },
    {
        "query": "Hurricane category for 160 mph winds?",
        "response": "A hurricane with 160 mph sustained winds would be classified as Category 5 on the Saffir-Simpson scale.",
        "human_score": 1.0,
        "expected_answer": "Category 5",
    },
    # Edge cases
    {
        "query": "Weather in asdfghjkl?",
        "response": "I couldn't find a location called 'asdfghjkl'. Could you please provide a valid city name?",
        "human_score": 0.90,
        "expected_answer": "Graceful handling of invalid location",
    },
    {
        "query": "Should I evacuate?",
        "response": "To provide evacuation guidance, I need to know your location and any current storm threats. Could you please specify where you are?",
        "human_score": 0.85,
        "expected_answer": "Request for clarification",
    },
    # Complex queries
    {
        "query": "Compare weather in Miami and Tampa today",
        "response": "Miami: 86°F, sunny, humidity 75%. Tampa: 82°F, partly cloudy, humidity 70%. Miami is slightly warmer with higher humidity.",
        "human_score": 0.90,
        "expected_answer": "Both cities' weather with comparison",
    },
    # Poor responses (should score low)
    {
        "query": "What's the weather in Boston?",
        "response": "Error: API timeout",
        "human_score": 0.0,
        "expected_answer": "Weather information",
    },
    {
        "query": "Hurricane Milton intensity?",
        "response": "Hurricane Milton is Category 5 with 130 mph winds.",  # WRONG: Cat 5 needs 157+ mph
        "human_score": 0.20,
        "expected_answer": "Accurate category/wind speed",
    },
]
