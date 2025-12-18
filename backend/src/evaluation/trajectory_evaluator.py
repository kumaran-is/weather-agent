"""4-pillar Trajectory-based Evaluation for Weather AI Agent.

This module implements the core TrajectoryEvaluator that orchestrates all 4 pillars:
1. Effectiveness (40%): Is the answer correct? (LLM-as-Judge)
2. Efficiency (20%): Did agent take optimal path? (Deterministic)
3. Robustness (20%): Does it handle edge cases? (Heuristics)
4. Safety (20%): Any safety violations? (Zero tolerance)

Overall Score Calculation:
    overall = 0.4×E1 + 0.2×E2 + 0.2×E3 + 0.2×E4
    IF Safety < 1.0 → overall = 0.0 (instant fail)

Pass Threshold: overall >= 0.80 AND safety == 1.0

LangSmith Integration:
- TrajectoryEvaluator can be wrapped as a LangSmith RunEvaluator
- All evaluations are traced for observability
- Supports batch evaluation on LangSmith Datasets

Usage:
    >>> from langchain_openai import ChatOpenAI
    >>> from backend.src.evaluation import TrajectoryEvaluator
    >>>
    >>> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    >>> evaluator = TrajectoryEvaluator(llm=llm)
    >>>
    >>> result = await evaluator.evaluate(
    ...     query="What's the hurricane status near Miami?",
    ...     trajectory=[{"tool": "get_hurricane_data", "output": {...}}],
    ...     final_answer="Hurricane Milton is Category 4...",
    ...     expected_answer="Hurricane Milton, Cat 4, 130 mph",
    ...     expected_tools=["get_hurricane_data"]
    ... )
    >>> print(f"Passed: {result.passed}, Score: {result.overall_score}")
"""

import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from backend.src.evaluation.effectiveness_judge import EffectivenessJudge
from backend.src.evaluation.efficiency_scorer import EfficiencyScorer
from backend.src.evaluation.models import (
    EvaluationResult,
    PillarWeights,
)
from backend.src.evaluation.robustness_checker import RobustnessChecker
from backend.src.evaluation.safety_validator import SafetyValidator

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:
    """Evaluate agent decision trajectories across 4 dimensions.

    This is the main entry point for evaluating Weather AI Agent responses.
    It orchestrates all 4 pillar evaluators and calculates the overall score.

    Attributes:
        llm: Language model for LLM-as-Judge evaluations
        weights: Configurable pillar weights (default: 40/20/20/20)
        effectiveness_judge: Pillar 1 evaluator
        efficiency_scorer: Pillar 2 evaluator
        robustness_checker: Pillar 3 evaluator
        safety_validator: Pillar 4 evaluator
    """

    def __init__(
        self,
        llm: BaseChatModel,
        weights: PillarWeights | None = None,
        strict_safety: bool = True,
    ) -> None:
        """Initialize TrajectoryEvaluator with all pillar evaluators.

        Args:
            llm: Language model for LLM-as-Judge (e.g., gpt-4o-mini)
            weights: Custom pillar weights (default: 40/20/20/20)
            strict_safety: If True, any safety violation = instant fail
        """
        self.llm = llm
        self.weights = weights or PillarWeights()
        self.strict_safety = strict_safety

        # Initialize pillar evaluators
        self.effectiveness_judge = EffectivenessJudge(llm=llm)
        self.efficiency_scorer = EfficiencyScorer()
        self.robustness_checker = RobustnessChecker()
        self.safety_validator = SafetyValidator()

        # Track last evaluation details for debugging
        self.last_details: dict[str, Any] = {}

        logger.info(
            f"TrajectoryEvaluator initialized with weights: "
            f"E={self.weights.effectiveness}, "
            f"F={self.weights.efficiency}, "
            f"R={self.weights.robustness}, "
            f"S={self.weights.safety}"
        )

    async def evaluate(
        self,
        query: str,
        trajectory: list[dict[str, Any]],
        final_answer: str,
        expected_answer: str | None = None,
        expected_tools: list[str] | None = None,
        test_case_id: str = "",
        latency_budget_ms: float = 5000.0,
        token_budget: int = 4000,
        is_edge_case: bool = False,
        is_safety_critical: bool = False,
    ) -> EvaluationResult:
        """Run complete 4-pillar evaluation on agent trajectory.

        Args:
            query: Original user query
            trajectory: List of agent steps (tool calls, observations)
            final_answer: Agent's final response
            expected_answer: Reference answer for correctness check
            expected_tools: Expected tool calls for efficiency check
            test_case_id: Optional test case identifier
            latency_budget_ms: Maximum acceptable latency (default: 5000ms)
            token_budget: Maximum acceptable tokens (default: 4000)
            is_edge_case: If True, apply stricter robustness checks
            is_safety_critical: If True, apply stricter safety checks

        Returns:
            EvaluationResult with all pillar scores and overall result
        """
        start_time = time.time()

        logger.debug(f"Starting evaluation for query: {query[:50]}...")

        # Extract metrics from trajectory
        actual_tools = self._extract_tool_calls(trajectory)
        total_latency_ms = self._calculate_latency(trajectory)
        total_tokens = self._calculate_tokens(trajectory)

        # Run all 4 pillar evaluations
        # Pillar 1: Effectiveness (LLM-as-Judge)
        effectiveness_result = await self.effectiveness_judge.evaluate(
            query=query,
            actual_answer=final_answer,
            expected_answer=expected_answer,
        )

        # Pillar 2: Efficiency (Deterministic)
        efficiency_result = self.efficiency_scorer.evaluate(
            actual_tools=actual_tools,
            expected_tools=expected_tools or [],
            latency_ms=total_latency_ms,
            latency_budget_ms=latency_budget_ms,
            tokens_used=total_tokens,
            token_budget=token_budget,
        )

        # Pillar 3: Robustness (Heuristics)
        robustness_result = self.robustness_checker.evaluate(
            query=query,
            trajectory=trajectory,
            final_answer=final_answer,
            is_edge_case=is_edge_case,
        )

        # Pillar 4: Safety (Zero Tolerance)
        safety_result = self.safety_validator.evaluate(
            query=query,
            trajectory=trajectory,
            final_answer=final_answer,
            is_safety_critical=is_safety_critical,
        )

        # Calculate overall score with safety zero-tolerance
        overall_score, passed = EvaluationResult.calculate_overall(
            effectiveness=effectiveness_result.score,
            efficiency=efficiency_result.score,
            robustness=robustness_result.score,
            safety=safety_result.score,
            weights=self.weights,
        )

        # Apply strict safety override if enabled
        if self.strict_safety and not safety_result.is_safe:
            overall_score = 0.0
            passed = False
            logger.warning(
                f"Safety violation detected - instant fail: {safety_result.violations}"
            )

        execution_time_ms = (time.time() - start_time) * 1000

        # Build result
        result = EvaluationResult(
            effectiveness=effectiveness_result.score,
            efficiency=efficiency_result.score,
            robustness=robustness_result.score,
            safety=safety_result.score,
            overall_score=overall_score,
            passed=passed,
            effectiveness_details=effectiveness_result,
            efficiency_details=efficiency_result,
            robustness_details=robustness_result,
            safety_details=safety_result,
            query=query,
            test_case_id=test_case_id,
            execution_time_ms=execution_time_ms,
            weights_used=self.weights,
        )

        # Store for debugging
        self.last_details = {
            "effectiveness": effectiveness_result.model_dump(),
            "efficiency": efficiency_result.model_dump(),
            "robustness": robustness_result.model_dump(),
            "safety": safety_result.model_dump(),
        }

        logger.info(
            f"Evaluation complete: overall={overall_score:.3f}, "
            f"passed={passed}, time={execution_time_ms:.1f}ms"
        )

        return result

    def _extract_tool_calls(self, trajectory: list[dict[str, Any]]) -> list[str]:
        """Extract tool names from trajectory.

        Handles various trajectory formats:
        - {"tool": "tool_name", ...}
        - {"name": "tool_name", ...}
        - {"type": "tool_call", "tool": "tool_name", ...}
        """
        tools = []
        for step in trajectory:
            if isinstance(step, dict):
                tool_name = step.get("tool") or step.get("name") or step.get("tool_name")
                if tool_name:
                    tools.append(str(tool_name))
        return tools

    def _calculate_latency(self, trajectory: list[dict[str, Any]]) -> float:
        """Calculate total latency from trajectory.

        Looks for latency/duration fields in trajectory steps.
        Returns 0.0 if not available.
        """
        total_ms = 0.0
        for step in trajectory:
            if isinstance(step, dict):
                latency = step.get("latency_ms") or step.get("duration_ms") or step.get("time_ms", 0.0)
                total_ms += float(latency)
        return total_ms

    def _calculate_tokens(self, trajectory: list[dict[str, Any]]) -> int:
        """Calculate total tokens from trajectory.

        Looks for token usage fields in trajectory steps.
        Returns 0 if not available.
        """
        total_tokens = 0
        for step in trajectory:
            if isinstance(step, dict):
                tokens = step.get("tokens") or step.get("token_count") or step.get("usage", {}).get("total_tokens", 0)
                total_tokens += int(tokens)
        return total_tokens

    async def evaluate_batch(
        self,
        test_cases: list[dict[str, Any]],
    ) -> list[EvaluationResult]:
        """Evaluate multiple test cases.

        Args:
            test_cases: List of test case dictionaries with:
                - query: str
                - trajectory: list
                - final_answer: str
                - expected_answer: str (optional)
                - expected_tools: list (optional)
                - test_case_id: str (optional)

        Returns:
            List of EvaluationResult for each test case
        """
        results = []
        for case in test_cases:
            result = await self.evaluate(
                query=case.get("query", ""),
                trajectory=case.get("trajectory", []),
                final_answer=case.get("final_answer", ""),
                expected_answer=case.get("expected_answer"),
                expected_tools=case.get("expected_tools"),
                test_case_id=case.get("test_case_id", ""),
                is_edge_case=case.get("is_edge_case", False),
                is_safety_critical=case.get("is_safety_critical", False),
            )
            results.append(result)
        return results
