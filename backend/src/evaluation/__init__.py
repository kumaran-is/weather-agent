"""Level 5b: Evaluation Framework with LangSmith Integration.

This module provides a comprehensive 4-pillar evaluation system for the Weather AI Agent:
1. Effectiveness (40%): Is the answer correct? (LLM-as-Judge)
2. Efficiency (20%): Did agent take optimal path? (Deterministic)
3. Robustness (20%): Does it handle edge cases? (Heuristics)
4. Safety (20%): Any safety violations? (Zero tolerance)

Architecture:
- LangSmith is the PRIMARY evaluation framework (native LangChain/LangGraph tracing)
- Custom RunEvaluators for trajectory-based evaluation
- Golden dataset stored as LangSmith Dataset
- Optional Ragas integration for RAG-specific metrics

Usage:
    >>> from backend.src.evaluation import TrajectoryEvaluator, EvaluationResult
    >>> evaluator = TrajectoryEvaluator(llm=llm)
    >>> result = await evaluator.evaluate(query, trajectory, final_answer)
    >>> print(f"Passed: {result.passed}, Score: {result.overall_score}")

LangSmith Integration:
    >>> from backend.src.evaluation import LangSmithEvaluator
    >>> ls_evaluator = LangSmithEvaluator(dataset_name="weather-golden-dataset")
    >>> results = await ls_evaluator.run_evaluation()
"""

from backend.src.evaluation.models import (
    EvaluationResult,
    EffectivenessResult,
    EfficiencyResult,
    RobustnessResult,
    SafetyResult,
    PillarWeights,
)
from backend.src.evaluation.trajectory_evaluator import TrajectoryEvaluator
from backend.src.evaluation.effectiveness_judge import EffectivenessJudge
from backend.src.evaluation.efficiency_scorer import EfficiencyScorer
from backend.src.evaluation.robustness_checker import RobustnessChecker
from backend.src.evaluation.safety_validator import SafetyValidator
from backend.src.evaluation.langsmith_evaluator import LangSmithEvaluator
from backend.src.evaluation.llm_judge_validator import LLMJudgeValidator

__all__ = [
    # Models
    "EvaluationResult",
    "EffectivenessResult",
    "EfficiencyResult",
    "RobustnessResult",
    "SafetyResult",
    "PillarWeights",
    # Evaluators
    "TrajectoryEvaluator",
    "EffectivenessJudge",
    "EfficiencyScorer",
    "RobustnessChecker",
    "SafetyValidator",
    # LangSmith Integration
    "LangSmithEvaluator",
    "LLMJudgeValidator",
]
