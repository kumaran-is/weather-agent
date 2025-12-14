"""Level 6a/6b: Evaluation Framework with LangSmith + Ragas + TruLens Integration.

This module provides a comprehensive 4-pillar evaluation system for the Weather AI Agent:
1. Effectiveness (40%): Is the answer correct? (LLM-as-Judge)
2. Efficiency (20%): Did agent take optimal path? (Deterministic)
3. Robustness (20%): Does it handle edge cases? (Heuristics)
4. Safety (20%): Any safety violations? (Zero tolerance)

Level 6a Additions (RAG-Specific Metrics):
- Faithfulness: Answer grounded in retrieved context (>0.90 target)
- Context Precision: Top-k chunks are relevant (>0.85 target)
- Context Recall: All relevant information retrieved (>0.85 target)
- Answer Relevancy: Answer addresses the query (>0.90 target)

Architecture:
- LangSmith is the PRIMARY evaluation framework (native LangChain/LangGraph tracing)
- Custom RunEvaluators for trajectory-based evaluation
- Golden dataset stored as LangSmith Dataset
- Ragas integration for RAG-specific metrics (Level 6a)

Usage:
    >>> from backend.src.evaluation import TrajectoryEvaluator, EvaluationResult
    >>> evaluator = TrajectoryEvaluator(llm=llm)
    >>> result = await evaluator.evaluate(query, trajectory, final_answer)
    >>> print(f"Passed: {result.passed}, Score: {result.overall_score}")

LangSmith Integration:
    >>> from backend.src.evaluation import LangSmithEvaluator
    >>> ls_evaluator = LangSmithEvaluator(dataset_name="weather-golden-dataset")
    >>> results = await ls_evaluator.run_evaluation()

Ragas Integration (Level 6a):
    >>> from backend.src.evaluation import RagasEvaluator, RagasResult
    >>> ragas_eval = RagasEvaluator(llm=llm, embeddings=embeddings)
    >>> result = await ragas_eval.evaluate_rag_query(query, contexts, answer)
    >>> print(f"Faithfulness: {result.faithfulness}, Passed: {result.passed}")

DeepEval Integration (Level 6a):
    >>> from backend.src.evaluation import DeepEvalIntegration, DeepEvalResult
    >>> deep_eval = DeepEvalIntegration(llm=llm)
    >>> result = await deep_eval.evaluate_query(query, answer, contexts)
    >>> print(f"Answer Relevancy: {result.answer_relevancy}, Passed: {result.passed}")
    >>> # Generate synthetic test cases
    >>> test_cases = deep_eval.generate_synthetic_test_cases(domain="weather", count=10)
    >>> suite_results = await deep_eval.run_test_suite(test_cases)

TruLens Integration (Level 6b):
    >>> from backend.src.evaluation import TruLensIntegration, TruLensFeedback
    >>> trulens = TruLensIntegration(llm=llm)
    >>> feedback = await trulens.evaluate_query(query, response, contexts)
    >>> print(f"Groundedness: {feedback.groundedness}, Passed: {feedback.passed}")
    >>> # Get feedback summary across all evaluations
    >>> summary = trulens.get_feedback_summary()
    >>> print(f"Overall Score: {summary['overall_score']}")
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
from backend.src.evaluation.ragas_evaluator import RagasEvaluator, RagasResult
from backend.src.evaluation.deepeval_integration import (
    DeepEvalIntegration,
    DeepEvalResult,
)
from backend.src.evaluation.trulens_integration import (
    TruLensIntegration,
    TruLensFeedback,
    TruLensRecord,
)
from backend.src.evaluation.langchain_benchmark import (
    LangChainBenchmark,
    BenchmarkResult,
    BenchmarkTask,
)
from backend.src.evaluation.retrieval_metrics import (
    MetricsCalculator,
    MRRCalculator,
    NDCGCalculator,
    BLEUCalculator,
    ROUGECalculator,
    PrecisionRecallCalculator,
    MAPCalculator,
    RetrievalMetrics,
    GenerationMetrics,
    CombinedMetrics,
    calculate_mrr,
    calculate_ndcg,
    calculate_bleu,
    calculate_rouge,
)
from backend.src.evaluation.promptfoo_integration import (
    PromptfooRunner,
    AssertionEvaluator,
    Assertion,
    AssertionType,
    TestCase,
    TestResult,
    EvalConfig,
    EvalSummary,
    WeatherPromptfooTests,
)
from backend.src.evaluation.openai_evals import (
    OpenAIEvalsRunner,
    GraderFactory,
    GraderType,
    Sample,
    EvalSpec,
    EvalResult,
    EvalRunResult,
    MatchGrader,
    IncludesGrader,
    FuzzyMatchGrader,
    WeatherEvals,
    create_match_eval,
    create_includes_eval,
)

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
    # Ragas Integration (Level 6a)
    "RagasEvaluator",
    "RagasResult",
    # DeepEval Integration (Level 6a)
    "DeepEvalIntegration",
    "DeepEvalResult",
    # TruLens Integration (Level 6b)
    "TruLensIntegration",
    "TruLensFeedback",
    "TruLensRecord",
    # LangChain Benchmark (Level 6b)
    "LangChainBenchmark",
    "BenchmarkResult",
    "BenchmarkTask",
    # Retrieval & Generation Metrics (Level 6c)
    "MetricsCalculator",
    "MRRCalculator",
    "NDCGCalculator",
    "BLEUCalculator",
    "ROUGECalculator",
    "PrecisionRecallCalculator",
    "MAPCalculator",
    "RetrievalMetrics",
    "GenerationMetrics",
    "CombinedMetrics",
    "calculate_mrr",
    "calculate_ndcg",
    "calculate_bleu",
    "calculate_rouge",
    # Promptfoo Integration (Level 6c)
    "PromptfooRunner",
    "AssertionEvaluator",
    "Assertion",
    "AssertionType",
    "TestCase",
    "TestResult",
    "EvalConfig",
    "EvalSummary",
    "WeatherPromptfooTests",
    # OpenAI Evals Integration (Level 6c)
    "OpenAIEvalsRunner",
    "GraderFactory",
    "GraderType",
    "Sample",
    "EvalSpec",
    "EvalResult",
    "EvalRunResult",
    "MatchGrader",
    "IncludesGrader",
    "FuzzyMatchGrader",
    "WeatherEvals",
    "create_match_eval",
    "create_includes_eval",
]
