"""
OpenAI Evals Integration Module.

Level 6c: Self-Evolving Platform

Provides OpenAI Evals-style evaluation capabilities:
1. Completion functions (model answering)
2. Match-based grading
3. Model-based grading
4. Includes/fuzzy matching
5. Custom evaluation functions

Target: Systematic evaluation using OpenAI Evals patterns
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class GraderType(str, Enum):
    """Types of graders for evaluation."""

    MATCH = "match"
    INCLUDES = "includes"
    FUZZY_MATCH = "fuzzy-match"
    MODEL_GRADED_CLOSEDQA = "model-graded-closedqa"
    MODEL_GRADED_FACT = "model-graded-fact"
    CUSTOM = "custom"


class Sample(BaseModel):
    """A single evaluation sample."""

    input: str | list[dict[str, str]]
    ideal: str | list[str] = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalSpec(BaseModel):
    """Specification for an evaluation."""

    eval_id: str
    description: str = ""
    metrics: list[str] = Field(default_factory=list)
    grader_type: GraderType = GraderType.MATCH
    grader_args: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Result of a single evaluation."""

    sample_id: str
    input: str
    output: str
    ideal: str
    passed: bool
    score: float = 0.0
    grader_type: GraderType
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRunResult(BaseModel):
    """Result of running an evaluation."""

    eval_id: str
    run_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    total_samples: int = 0
    passed_samples: int = 0
    failed_samples: int = 0
    accuracy: float = 0.0
    avg_score: float = 0.0
    results: list[EvalResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


# ============================================================================
# Graders
# ============================================================================


class BaseGrader(ABC):
    """Abstract base class for graders."""

    @abstractmethod
    def grade(
        self,
        output: str,
        ideal: str | list[str],
        input_text: str = "",
    ) -> tuple[bool, float, str]:
        """
        Grade an output against ideal answer(s).

        Args:
            output: Model output
            ideal: Expected answer(s)
            input_text: Original input

        Returns:
            Tuple of (passed, score, reason)
        """
        pass


class MatchGrader(BaseGrader):
    """
    Exact match grader.

    Checks if output exactly matches one of the ideal answers.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
    ):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def grade(
        self,
        output: str,
        ideal: str | list[str],
        input_text: str = "",
    ) -> tuple[bool, float, str]:
        ideals = [ideal] if isinstance(ideal, str) else ideal

        processed_output = output
        if self.strip_whitespace:
            processed_output = processed_output.strip()
        if not self.case_sensitive:
            processed_output = processed_output.lower()

        for expected in ideals:
            processed_expected = expected
            if self.strip_whitespace:
                processed_expected = processed_expected.strip()
            if not self.case_sensitive:
                processed_expected = processed_expected.lower()

            if processed_output == processed_expected:
                return True, 1.0, "Exact match found"

        return False, 0.0, f"No match found. Output: {output[:100]}"


class IncludesGrader(BaseGrader):
    """
    Inclusion grader.

    Checks if output contains the ideal answer(s).
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        all_required: bool = True,
    ):
        self.case_sensitive = case_sensitive
        self.all_required = all_required

    def grade(
        self,
        output: str,
        ideal: str | list[str],
        input_text: str = "",
    ) -> tuple[bool, float, str]:
        ideals = [ideal] if isinstance(ideal, str) else ideal

        processed_output = output if self.case_sensitive else output.lower()

        found = []
        missing = []

        for expected in ideals:
            processed_expected = expected if self.case_sensitive else expected.lower()
            if processed_expected in processed_output:
                found.append(expected)
            else:
                missing.append(expected)

        if self.all_required:
            passed = len(missing) == 0
            score = len(found) / len(ideals) if ideals else 1.0
        else:
            passed = len(found) > 0
            score = len(found) / len(ideals) if ideals else 0.0

        if passed:
            reason = f"Found all required: {found}"
        else:
            reason = f"Missing: {missing}"

        return passed, score, reason


class FuzzyMatchGrader(BaseGrader):
    """
    Fuzzy matching grader.

    Uses similarity metrics for partial matching.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        method: str = "token_overlap",
    ):
        self.threshold = threshold
        self.method = method

    def grade(
        self,
        output: str,
        ideal: str | list[str],
        input_text: str = "",
    ) -> tuple[bool, float, str]:
        ideals = [ideal] if isinstance(ideal, str) else ideal

        max_similarity = 0.0
        best_match = ""

        for expected in ideals:
            similarity = self._compute_similarity(output, expected)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = expected

        passed = max_similarity >= self.threshold

        return (
            passed,
            max_similarity,
            f"Best similarity: {max_similarity:.2f} (threshold: {self.threshold})",
        )

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        if self.method == "token_overlap":
            return self._token_overlap(text1, text2)
        elif self.method == "char_overlap":
            return self._char_overlap(text1, text2)
        else:
            return self._token_overlap(text1, text2)

    def _token_overlap(self, text1: str, text2: str) -> float:
        """Token-level overlap similarity."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _char_overlap(self, text1: str, text2: str) -> float:
        """Character-level overlap similarity."""
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())

        if not chars1 or not chars2:
            return 0.0

        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)

        return intersection / union if union > 0 else 0.0


class ModelGradedGrader(BaseGrader):
    """
    Model-based grader.

    Uses an LLM to evaluate the output.
    """

    def __init__(
        self,
        evaluator_fn: Callable[[str, str, str, str], tuple[float, str]] | None = None,
        grading_prompt: str = "",
        threshold: float = 0.7,
    ):
        """
        Initialize model grader.

        Args:
            evaluator_fn: Function (input, output, ideal, prompt) -> (score, reason)
            grading_prompt: Custom grading prompt
            threshold: Pass threshold
        """
        self.evaluator_fn = evaluator_fn
        self.grading_prompt = grading_prompt
        self.threshold = threshold

    def grade(
        self,
        output: str,
        ideal: str | list[str],
        input_text: str = "",
    ) -> tuple[bool, float, str]:
        if not self.evaluator_fn:
            return False, 0.0, "No evaluator function configured"

        ideal_str = ideal if isinstance(ideal, str) else " | ".join(ideal)

        prompt = self.grading_prompt or self._default_grading_prompt()

        score, reason = self.evaluator_fn(input_text, output, ideal_str, prompt)
        passed = score >= self.threshold

        return passed, score, reason

    def _default_grading_prompt(self) -> str:
        return """
You are evaluating an AI response against an ideal answer.

Question: {input}
AI Response: {output}
Ideal Answer: {ideal}

Rate the AI response on a scale of 0 to 1:
- 1.0: Perfect match or better than ideal
- 0.8-0.9: Very good, captures key points
- 0.6-0.7: Acceptable, some important info missing
- 0.4-0.5: Partially correct
- 0.2-0.3: Mostly incorrect
- 0.0-0.1: Completely wrong or harmful

Provide a score and brief explanation.
"""


class ModelGradedClosedQAGrader(ModelGradedGrader):
    """Grader for closed-domain QA with reference answers."""

    def __init__(
        self,
        evaluator_fn: Callable[[str, str, str, str], tuple[float, str]] | None = None,
        threshold: float = 0.7,
    ):
        super().__init__(evaluator_fn=evaluator_fn, threshold=threshold)
        self.grading_prompt = """
You are evaluating a closed-domain QA response.

Question: {input}
Model Answer: {output}
Reference Answer: {ideal}

Evaluate if the model answer is correct compared to the reference.
Consider:
1. Factual accuracy
2. Completeness
3. Relevance

Score from 0 to 1.
"""


class ModelGradedFactGrader(ModelGradedGrader):
    """Grader for factual accuracy evaluation."""

    def __init__(
        self,
        evaluator_fn: Callable[[str, str, str, str], tuple[float, str]] | None = None,
        threshold: float = 0.8,
    ):
        super().__init__(evaluator_fn=evaluator_fn, threshold=threshold)
        self.grading_prompt = """
You are evaluating the factual accuracy of a response.

Question: {input}
Response: {output}
Expected Facts: {ideal}

Evaluate if the response is factually accurate.
Consider:
1. Are stated facts correct?
2. Are there any factual errors or hallucinations?
3. Is the information verifiable?

Score from 0 to 1 based on factual accuracy.
"""


class CustomGrader(BaseGrader):
    """
    Custom grader with user-defined function.

    Allows arbitrary grading logic.
    """

    def __init__(
        self,
        grader_fn: Callable[[str, str | list[str], str], tuple[bool, float, str]],
    ):
        """
        Initialize custom grader.

        Args:
            grader_fn: Custom grading function (output, ideal, input) -> (passed, score, reason)
        """
        self.grader_fn = grader_fn

    def grade(
        self,
        output: str,
        ideal: str | list[str],
        input_text: str = "",
    ) -> tuple[bool, float, str]:
        return self.grader_fn(output, ideal, input_text)


# ============================================================================
# Grader Factory
# ============================================================================


class GraderFactory:
    """Factory for creating graders."""

    @staticmethod
    def create(
        grader_type: GraderType,
        args: dict[str, Any] | None = None,
        evaluator_fn: Callable | None = None,
    ) -> BaseGrader:
        """
        Create a grader instance.

        Args:
            grader_type: Type of grader to create
            args: Grader-specific arguments
            evaluator_fn: Optional evaluator function for model-graded

        Returns:
            Grader instance
        """
        args = args or {}

        if grader_type == GraderType.MATCH:
            return MatchGrader(
                case_sensitive=args.get("case_sensitive", False),
                strip_whitespace=args.get("strip_whitespace", True),
            )

        elif grader_type == GraderType.INCLUDES:
            return IncludesGrader(
                case_sensitive=args.get("case_sensitive", False),
                all_required=args.get("all_required", True),
            )

        elif grader_type == GraderType.FUZZY_MATCH:
            return FuzzyMatchGrader(
                threshold=args.get("threshold", 0.8),
                method=args.get("method", "token_overlap"),
            )

        elif grader_type == GraderType.MODEL_GRADED_CLOSEDQA:
            return ModelGradedClosedQAGrader(
                evaluator_fn=evaluator_fn,
                threshold=args.get("threshold", 0.7),
            )

        elif grader_type == GraderType.MODEL_GRADED_FACT:
            return ModelGradedFactGrader(
                evaluator_fn=evaluator_fn,
                threshold=args.get("threshold", 0.8),
            )

        else:
            raise ValueError(f"Unknown grader type: {grader_type}")


# ============================================================================
# OpenAI Evals Runner
# ============================================================================


class OpenAIEvalsRunner:
    """
    Run OpenAI Evals-style evaluations.

    Supports various grading methods and comprehensive result reporting.
    """

    def __init__(
        self,
        completion_fn: Callable[[str], str] | None = None,
        evaluator_fn: Callable[[str, str, str, str], tuple[float, str]] | None = None,
    ):
        """
        Initialize OpenAI Evals runner.

        Args:
            completion_fn: Function to generate completions (input) -> output
            evaluator_fn: Function for model-graded evaluations
        """
        self.completion_fn = completion_fn
        self.evaluator_fn = evaluator_fn

        # Grader factory
        self.grader_factory = GraderFactory()

        # Results
        self.results: list[EvalRunResult] = []

        logger.info("OpenAIEvalsRunner initialized")

    def set_completion_fn(
        self,
        completion_fn: Callable[[str], str],
    ) -> None:
        """Set completion function."""
        self.completion_fn = completion_fn

    def set_evaluator_fn(
        self,
        evaluator_fn: Callable[[str, str, str, str], tuple[float, str]],
    ) -> None:
        """Set evaluator function for model-graded evaluations."""
        self.evaluator_fn = evaluator_fn

    async def run_eval(
        self,
        eval_spec: EvalSpec,
        samples: list[Sample],
        run_id: str | None = None,
    ) -> EvalRunResult:
        """
        Run an evaluation.

        Args:
            eval_spec: Evaluation specification
            samples: Samples to evaluate
            run_id: Optional run identifier

        Returns:
            EvalRunResult with evaluation results
        """
        if not self.completion_fn:
            raise ValueError("Completion function not set")

        run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create grader
        grader = self.grader_factory.create(
            grader_type=eval_spec.grader_type,
            args=eval_spec.grader_args,
            evaluator_fn=self.evaluator_fn,
        )

        results: list[EvalResult] = []

        for i, sample in enumerate(samples):
            # Get input text
            if isinstance(sample.input, str):
                input_text = sample.input
            else:
                # Convert chat messages to text
                input_text = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in sample.input
                )

            # Generate completion
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self.completion_fn):
                    output = await self.completion_fn(input_text)
                else:
                    output = self.completion_fn(input_text)
            except Exception as e:
                output = f"ERROR: {str(e)}"

            # Grade output
            ideal = sample.ideal if sample.ideal else ""
            passed, score, reason = grader.grade(output, ideal, input_text)

            result = EvalResult(
                sample_id=f"{eval_spec.eval_id}_{i}",
                input=input_text,
                output=output,
                ideal=ideal if isinstance(ideal, str) else " | ".join(ideal),
                passed=passed,
                score=score,
                grader_type=eval_spec.grader_type,
                reason=reason,
                metadata=sample.metadata,
            )
            results.append(result)

        # Compute metrics
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        accuracy = passed_count / total if total > 0 else 0.0
        avg_score = sum(r.score for r in results) / total if total > 0 else 0.0

        run_result = EvalRunResult(
            eval_id=eval_spec.eval_id,
            run_id=run_id,
            total_samples=total,
            passed_samples=passed_count,
            failed_samples=total - passed_count,
            accuracy=round(accuracy, 4),
            avg_score=round(avg_score, 4),
            results=results,
            metrics={
                "accuracy": accuracy,
                "avg_score": avg_score,
                "pass_rate": accuracy,
            },
        )

        self.results.append(run_result)

        logger.info(
            f"Eval '{eval_spec.eval_id}' completed | "
            f"accuracy={accuracy:.2%} | avg_score={avg_score:.4f}"
        )

        return run_result

    def get_results(self) -> list[EvalRunResult]:
        """Get all evaluation results."""
        return self.results

    def clear_results(self) -> None:
        """Clear evaluation results."""
        self.results = []


# ============================================================================
# Weather-Specific OpenAI Evals
# ============================================================================


class WeatherEvals:
    """Pre-built OpenAI Evals for weather domain."""

    @staticmethod
    def get_weather_knowledge_eval() -> tuple[EvalSpec, list[Sample]]:
        """Get weather knowledge evaluation."""
        eval_spec = EvalSpec(
            eval_id="weather_knowledge",
            description="Evaluate weather knowledge accuracy",
            grader_type=GraderType.INCLUDES,
            grader_args={"all_required": False},
        )

        samples = [
            Sample(
                input="What is the Saffir-Simpson Hurricane Wind Scale?",
                ideal=["hurricane", "wind speed", "category"],
            ),
            Sample(
                input="What wind speed is Category 5?",
                ideal=["157 mph", "157mph", ">156"],
            ),
            Sample(
                input="What does storm surge mean?",
                ideal=["water", "coast", "flooding"],
            ),
            Sample(
                input="What is a tropical depression?",
                ideal=["tropical", "38 mph", "39 mph", "low pressure"],
            ),
        ]

        return eval_spec, samples

    @staticmethod
    def get_safety_eval() -> tuple[EvalSpec, list[Sample]]:
        """Get safety evaluation for weather responses."""
        eval_spec = EvalSpec(
            eval_id="weather_safety",
            description="Evaluate safety of weather responses",
            grader_type=GraderType.MODEL_GRADED_FACT,
            grader_args={"threshold": 0.8},
        )

        samples = [
            Sample(
                input="Should I evacuate during a Category 4 hurricane?",
                ideal="Evacuation is strongly recommended during Category 4 hurricanes, especially in evacuation zones A and B. Follow local emergency management guidance.",
                metadata={"category": "evacuation"},
            ),
            Sample(
                input="Is it safe to drive through flooded roads?",
                ideal="Never drive through flooded roads. Turn around, don't drown. Just 6 inches of water can knock you down and 2 feet can float a car.",
                metadata={"category": "flood_safety"},
            ),
            Sample(
                input="Can I stay home during a hurricane warning?",
                ideal="Whether to stay depends on your evacuation zone, building construction, and the hurricane's intensity. If in an evacuation zone, you should evacuate. Always follow local emergency management orders.",
                metadata={"category": "evacuation"},
            ),
        ]

        return eval_spec, samples

    @staticmethod
    def get_factual_accuracy_eval() -> tuple[EvalSpec, list[Sample]]:
        """Get factual accuracy evaluation."""
        eval_spec = EvalSpec(
            eval_id="weather_factual",
            description="Evaluate factual accuracy of weather information",
            grader_type=GraderType.FUZZY_MATCH,
            grader_args={"threshold": 0.7},
        )

        samples = [
            Sample(
                input="What are the wind speeds for each hurricane category?",
                ideal="Cat 1: 74-95 mph, Cat 2: 96-110 mph, Cat 3: 111-129 mph, Cat 4: 130-156 mph, Cat 5: 157+ mph",
            ),
            Sample(
                input="What temperature scale is used in the US?",
                ideal="Fahrenheit",
            ),
            Sample(
                input="Who issues hurricane warnings?",
                ideal="National Hurricane Center (NHC) or National Weather Service (NWS)",
            ),
        ]

        return eval_spec, samples

    @staticmethod
    def create_custom_eval(
        eval_id: str,
        grader_type: GraderType,
        samples: list[dict[str, Any]],
        grader_args: dict[str, Any] | None = None,
    ) -> tuple[EvalSpec, list[Sample]]:
        """
        Create a custom weather evaluation.

        Args:
            eval_id: Evaluation identifier
            grader_type: Type of grader to use
            samples: List of sample dicts with 'input' and 'ideal' keys
            grader_args: Optional grader arguments

        Returns:
            Tuple of (EvalSpec, list of Samples)
        """
        eval_spec = EvalSpec(
            eval_id=eval_id,
            grader_type=grader_type,
            grader_args=grader_args or {},
        )

        sample_objects = [
            Sample(
                input=s.get("input", ""),
                ideal=s.get("ideal", ""),
                metadata=s.get("metadata", {}),
            )
            for s in samples
        ]

        return eval_spec, sample_objects


# ============================================================================
# Convenience Functions
# ============================================================================


def create_match_eval(
    eval_id: str,
    qa_pairs: list[tuple[str, str]],
) -> tuple[EvalSpec, list[Sample]]:
    """
    Create a simple match-based evaluation.

    Args:
        eval_id: Evaluation identifier
        qa_pairs: List of (question, answer) tuples

    Returns:
        Tuple of (EvalSpec, list of Samples)
    """
    eval_spec = EvalSpec(
        eval_id=eval_id,
        grader_type=GraderType.MATCH,
    )

    samples = [
        Sample(input=q, ideal=a)
        for q, a in qa_pairs
    ]

    return eval_spec, samples


def create_includes_eval(
    eval_id: str,
    qa_pairs: list[tuple[str, list[str]]],
    all_required: bool = True,
) -> tuple[EvalSpec, list[Sample]]:
    """
    Create an includes-based evaluation.

    Args:
        eval_id: Evaluation identifier
        qa_pairs: List of (question, expected_keywords) tuples
        all_required: Whether all keywords are required

    Returns:
        Tuple of (EvalSpec, list of Samples)
    """
    eval_spec = EvalSpec(
        eval_id=eval_id,
        grader_type=GraderType.INCLUDES,
        grader_args={"all_required": all_required},
    )

    samples = [
        Sample(input=q, ideal=keywords)
        for q, keywords in qa_pairs
    ]

    return eval_spec, samples
