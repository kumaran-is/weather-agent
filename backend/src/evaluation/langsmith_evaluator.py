"""LangSmith Integration for Weather AI Agent Evaluation.

This module provides LangSmith-native evaluation capabilities:
1. Custom RunEvaluators for trajectory-based evaluation
2. Dataset integration for golden set testing
3. Experiment tracking and comparison
4. CI/CD quality gate integration

LangSmith is the PRIMARY evaluation framework for this project because:
- Native tracing for LangChain/LangGraph applications
- Datasets + golden sets + LLM-as-judge support
- Custom RunEvaluators that see full trajectory
- Built-in experiment comparison and tracking

Usage:
    >>> from backend.src.evaluation import LangSmithEvaluator
    >>>
    >>> evaluator = LangSmithEvaluator(
    ...     dataset_name="weather-golden-dataset",
    ...     project_name="weather-ai-evaluation"
    ... )
    >>>
    >>> # Run evaluation on golden dataset
    >>> results = await evaluator.run_evaluation()
    >>> print(f"Pass rate: {results.pass_rate}")
    >>>
    >>> # Check if quality gates pass (for CI/CD)
    >>> if evaluator.check_quality_gates(results):
    ...     print("Deployment approved")
    ... else:
    ...     print("Quality gates FAILED - deployment blocked")
"""

import os
import logging
import time
from datetime import datetime
from typing import Any, Callable

from langsmith import Client
from langsmith.evaluation import RunEvaluator, EvaluationResult as LSEvaluationResult
from langsmith.schemas import Run, Example
from langchain_core.language_models import BaseChatModel

from backend.src.evaluation.models import (
    EvaluationResult,
    EvaluationBatchResult,
    GoldenTestCase,
    PillarWeights,
)
from backend.src.evaluation.trajectory_evaluator import TrajectoryEvaluator

logger = logging.getLogger(__name__)


class WeatherAgentEvaluator(RunEvaluator):
    """LangSmith RunEvaluator for Weather AI Agent.

    This evaluator wraps our 4-pillar TrajectoryEvaluator as a
    LangSmith-compatible RunEvaluator for native integration.

    Can see full LangGraph trajectory including:
    - All tool calls and their outputs
    - Intermediate states
    - Final answer
    - Latency and token usage
    """

    def __init__(
        self,
        llm: BaseChatModel,
        weights: PillarWeights | None = None,
    ) -> None:
        """Initialize evaluator with LLM for effectiveness judging.

        Args:
            llm: Language model for LLM-as-Judge
            weights: Pillar weights (default: 40/20/20/20)
        """
        self.trajectory_evaluator = TrajectoryEvaluator(llm=llm, weights=weights)

    def evaluate_run(
        self,
        run: Run,
        example: Example | None = None,
    ) -> LSEvaluationResult:
        """Evaluate a single LangSmith run.

        This is called by LangSmith for each run in an experiment.

        Args:
            run: LangSmith Run object with full trajectory
            example: Optional Example from dataset with expected outputs

        Returns:
            LangSmith EvaluationResult with scores and feedback
        """
        import asyncio

        # Extract query, trajectory, and answer from run
        query = self._extract_query(run)
        trajectory = self._extract_trajectory(run)
        final_answer = self._extract_answer(run)

        # Extract expected values from example if available
        expected_answer = None
        expected_tools = None
        if example:
            expected_answer = example.outputs.get("expected_answer")
            expected_tools = example.outputs.get("expected_tools", [])

        # Run async evaluation in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self.trajectory_evaluator.evaluate(
                query=query,
                trajectory=trajectory,
                final_answer=final_answer,
                expected_answer=expected_answer,
                expected_tools=expected_tools,
                test_case_id=example.id if example else "",
            )
        )

        # Convert to LangSmith format
        return LSEvaluationResult(
            key="weather_agent_evaluation",
            score=result.overall_score,
            value={
                "passed": result.passed,
                "effectiveness": result.effectiveness,
                "efficiency": result.efficiency,
                "robustness": result.robustness,
                "safety": result.safety,
            },
            comment=self._build_comment(result),
        )

    def _extract_query(self, run: Run) -> str:
        """Extract original query from run inputs."""
        inputs = run.inputs or {}
        return (
            inputs.get("query") or
            inputs.get("input") or
            inputs.get("question") or
            str(inputs)
        )

    def _extract_trajectory(self, run: Run) -> list[dict[str, Any]]:
        """Extract tool call trajectory from run."""
        trajectory = []

        # Get child runs (tool calls)
        # Note: In actual LangSmith, you'd fetch child runs via client
        # This is a simplified version
        if hasattr(run, "child_runs") and run.child_runs:
            for child in run.child_runs:
                trajectory.append({
                    "tool": child.name,
                    "input": child.inputs,
                    "output": child.outputs,
                    "latency_ms": (
                        (child.end_time - child.start_time).total_seconds() * 1000
                        if child.end_time and child.start_time else 0
                    ),
                })

        return trajectory

    def _extract_answer(self, run: Run) -> str:
        """Extract final answer from run outputs."""
        outputs = run.outputs or {}
        return (
            outputs.get("answer") or
            outputs.get("output") or
            outputs.get("response") or
            str(outputs)
        )

    def _build_comment(self, result: EvaluationResult) -> str:
        """Build human-readable comment for evaluation result."""
        status = "PASSED" if result.passed else "FAILED"
        parts = [
            f"Status: {status}",
            f"Overall: {result.overall_score:.2f}",
            f"Effectiveness: {result.effectiveness:.2f}",
            f"Efficiency: {result.efficiency:.2f}",
            f"Robustness: {result.robustness:.2f}",
            f"Safety: {result.safety:.2f}",
        ]

        if not result.passed:
            if result.safety < 1.0 and result.safety_details:
                parts.append(f"Safety violations: {result.safety_details.violations}")
            if result.overall_score < 0.8:
                parts.append("Score below 0.80 threshold")

        return " | ".join(parts)


class LangSmithEvaluator:
    """High-level LangSmith evaluation orchestrator.

    Provides a simple interface for:
    - Running evaluations on golden datasets
    - Checking quality gates for CI/CD
    - Comparing experiments
    - Generating evaluation reports

    Attributes:
        client: LangSmith client
        dataset_name: Name of golden dataset in LangSmith
        project_name: LangSmith project name
        evaluator: WeatherAgentEvaluator instance
    """

    def __init__(
        self,
        llm: BaseChatModel,
        dataset_name: str = "weather-golden-dataset",
        project_name: str = "weather-ai-evaluation",
        weights: PillarWeights | None = None,
    ) -> None:
        """Initialize LangSmith evaluator.

        Args:
            llm: Language model for LLM-as-Judge
            dataset_name: Name of dataset in LangSmith
            project_name: LangSmith project name
            weights: Pillar weights (default: 40/20/20/20)
        """
        self.client = Client()
        self.dataset_name = dataset_name
        self.project_name = project_name
        self.llm = llm
        self.weights = weights or PillarWeights()
        self.evaluator = WeatherAgentEvaluator(llm=llm, weights=weights)

        logger.info(
            f"LangSmithEvaluator initialized: "
            f"dataset={dataset_name}, project={project_name}"
        )

    async def run_evaluation(
        self,
        target_func: Callable | None = None,
        experiment_prefix: str = "eval",
        max_concurrency: int = 4,
    ) -> EvaluationBatchResult:
        """Run evaluation on golden dataset.

        Args:
            target_func: Agent function to evaluate (optional)
            experiment_prefix: Prefix for experiment name
            max_concurrency: Max concurrent evaluations

        Returns:
            EvaluationBatchResult with aggregate scores
        """
        start_time = time.time()
        timestamp = datetime.utcnow().isoformat()

        logger.info(f"Starting evaluation on dataset: {self.dataset_name}")

        # Get dataset examples
        try:
            dataset = self.client.read_dataset(dataset_name=self.dataset_name)
            examples = list(self.client.list_examples(dataset_id=dataset.id))
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return EvaluationBatchResult(
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                pass_rate=0.0,
                avg_effectiveness=0.0,
                avg_efficiency=0.0,
                avg_robustness=0.0,
                safety_pass_rate=0.0,
                avg_overall=0.0,
                dataset_name=self.dataset_name,
                timestamp=timestamp,
            )

        # Run evaluations
        results: list[EvaluationResult] = []
        failed_ids: list[str] = []
        safety_violations = 0
        violation_types: list[str] = []

        for example in examples:
            # Extract test case from example
            query = example.inputs.get("query", "")
            expected_answer = example.outputs.get("expected_answer")
            expected_tools = example.outputs.get("expected_tools", [])

            # If target_func provided, run it; otherwise use existing run
            if target_func:
                try:
                    response = await target_func(query)
                    final_answer = response.get("answer", str(response))
                    trajectory = response.get("trajectory", [])
                except Exception as e:
                    logger.error(f"Target function failed for {example.id}: {e}")
                    final_answer = f"Error: {e}"
                    trajectory = []
            else:
                final_answer = example.outputs.get("actual_answer", "")
                trajectory = example.outputs.get("trajectory", [])

            # Evaluate
            result = await self.evaluator.trajectory_evaluator.evaluate(
                query=query,
                trajectory=trajectory,
                final_answer=final_answer,
                expected_answer=expected_answer,
                expected_tools=expected_tools,
                test_case_id=str(example.id),
            )
            results.append(result)

            if not result.passed:
                failed_ids.append(str(example.id))

            if result.safety < 1.0:
                safety_violations += 1
                if result.safety_details:
                    violation_types.extend([v.value for v in result.safety_details.violations])

        # Calculate aggregates
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        avg_effectiveness = sum(r.effectiveness for r in results) / max(1, total)
        avg_efficiency = sum(r.efficiency for r in results) / max(1, total)
        avg_robustness = sum(r.robustness for r in results) / max(1, total)
        avg_overall = sum(r.overall_score for r in results) / max(1, total)
        safety_pass_rate = sum(1 for r in results if r.safety == 1.0) / max(1, total)

        execution_time = (time.time() - start_time) * 1000

        return EvaluationBatchResult(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            pass_rate=passed / max(1, total),
            avg_effectiveness=avg_effectiveness,
            avg_efficiency=avg_efficiency,
            avg_robustness=avg_robustness,
            safety_pass_rate=safety_pass_rate,
            avg_overall=avg_overall,
            safety_violations=safety_violations,
            safety_violation_types=list(set(violation_types)),
            results=results,
            failed_test_ids=failed_ids,
            dataset_name=self.dataset_name,
            evaluation_time_ms=execution_time,
            timestamp=timestamp,
        )

    def check_quality_gates(
        self,
        results: EvaluationBatchResult,
        min_pass_rate: float = 0.85,
        min_effectiveness: float = 0.85,
        min_efficiency: float = 0.80,
        min_robustness: float = 0.80,
        max_safety_violations: int = 0,
    ) -> dict[str, Any]:
        """Check if evaluation results pass quality gates for CI/CD.

        Args:
            results: Evaluation batch results
            min_pass_rate: Minimum overall pass rate
            min_effectiveness: Minimum average effectiveness
            min_efficiency: Minimum average efficiency
            min_robustness: Minimum average robustness
            max_safety_violations: Maximum allowed safety violations (default: 0)

        Returns:
            Dict with gate status and details
        """
        gates = {
            "pass_rate": {
                "passed": results.pass_rate >= min_pass_rate,
                "actual": results.pass_rate,
                "threshold": min_pass_rate,
            },
            "effectiveness": {
                "passed": results.avg_effectiveness >= min_effectiveness,
                "actual": results.avg_effectiveness,
                "threshold": min_effectiveness,
            },
            "efficiency": {
                "passed": results.avg_efficiency >= min_efficiency,
                "actual": results.avg_efficiency,
                "threshold": min_efficiency,
            },
            "robustness": {
                "passed": results.avg_robustness >= min_robustness,
                "actual": results.avg_robustness,
                "threshold": min_robustness,
            },
            "safety": {
                "passed": results.safety_violations <= max_safety_violations,
                "actual": results.safety_violations,
                "threshold": max_safety_violations,
            },
        }

        all_passed = all(g["passed"] for g in gates.values())

        return {
            "all_passed": all_passed,
            "gates": gates,
            "recommendation": (
                "DEPLOY: All quality gates passed"
                if all_passed
                else "BLOCK: Quality gates failed - see details"
            ),
            "failed_gates": [k for k, v in gates.items() if not v["passed"]],
        }

    async def upload_golden_dataset(
        self,
        test_cases: list[GoldenTestCase],
        description: str = "Weather AI Agent Golden Dataset",
    ) -> str:
        """Upload golden dataset to LangSmith.

        Args:
            test_cases: List of GoldenTestCase objects
            description: Dataset description

        Returns:
            Dataset ID
        """
        # Create or update dataset
        try:
            dataset = self.client.create_dataset(
                dataset_name=self.dataset_name,
                description=description,
            )
        except Exception:
            # Dataset might already exist
            dataset = self.client.read_dataset(dataset_name=self.dataset_name)

        # Upload examples
        for case in test_cases:
            self.client.create_example(
                inputs={"query": case.query},
                outputs={
                    "expected_answer": case.expected_answer,
                    "expected_tools": case.expected_tools,
                    "expected_answer_contains": case.expected_answer_contains,
                },
                dataset_id=dataset.id,
                example_id=case.id,
            )

        logger.info(f"Uploaded {len(test_cases)} test cases to {self.dataset_name}")

        return str(dataset.id)
