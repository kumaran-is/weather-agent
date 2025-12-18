"""
Tests for OpenAI Evals Integration Module.

Level 6c: Self-Evolving Platform

Tests cover:
1. Grader types and factory
2. Sample and EvalSpec models
3. Match, Includes, Fuzzy graders
4. Model-graded evaluation
5. OpenAI Evals runner
6. Weather-specific evaluations
"""


import pytest

from backend.src.evaluation.openai_evals import (
    CustomGrader,
    EvalResult,
    EvalRunResult,
    EvalSpec,
    FuzzyMatchGrader,
    GraderFactory,
    GraderType,
    IncludesGrader,
    MatchGrader,
    ModelGradedClosedQAGrader,
    ModelGradedFactGrader,
    ModelGradedGrader,
    OpenAIEvalsRunner,
    Sample,
    WeatherEvals,
    create_includes_eval,
    create_match_eval,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_eval_spec():
    """Create a sample evaluation spec."""
    return EvalSpec(
        eval_id="test_eval",
        description="Test evaluation",
        metrics=["accuracy"],
        grader_type=GraderType.MATCH,
        grader_args={"case_sensitive": False},
    )


@pytest.fixture
def sample_samples():
    """Create sample test data."""
    return [
        Sample(input="What is 2+2?", ideal="4"),
        Sample(input="Capital of France?", ideal="Paris"),
        Sample(input="Color of sky?", ideal=["blue", "Blue"]),
    ]


@pytest.fixture
def mock_completion_fn():
    """Create a mock completion function."""
    async def completion(input_text: str) -> str:
        if "2+2" in input_text:
            return "4"
        elif "France" in input_text:
            return "Paris"
        elif "sky" in input_text:
            return "blue"
        return "Unknown"
    return completion


@pytest.fixture
def evals_runner(mock_completion_fn):
    """Create OpenAI Evals runner with mock completion."""
    runner = OpenAIEvalsRunner(completion_fn=mock_completion_fn)
    return runner


# ============================================================================
# GraderType Tests
# ============================================================================


class TestGraderType:
    """Test GraderType enumeration."""

    def test_all_grader_types_exist(self):
        """Verify all expected grader types exist."""
        expected_types = [
            "match", "includes", "fuzzy-match",
            "model-graded-closedqa", "model-graded-fact", "custom"
        ]

        for type_name in expected_types:
            assert any(gt.value == type_name for gt in GraderType)

    def test_grader_type_values(self):
        """Verify grader type string values."""
        assert GraderType.MATCH.value == "match"
        assert GraderType.INCLUDES.value == "includes"
        assert GraderType.FUZZY_MATCH.value == "fuzzy-match"
        assert GraderType.MODEL_GRADED_CLOSEDQA.value == "model-graded-closedqa"


# ============================================================================
# Sample Model Tests
# ============================================================================


class TestSampleModel:
    """Test Sample Pydantic model."""

    def test_create_sample_with_string_input(self):
        """Test creating sample with string input."""
        sample = Sample(input="Test question", ideal="Test answer")

        assert sample.input == "Test question"
        assert sample.ideal == "Test answer"
        assert sample.metadata == {}

    def test_create_sample_with_chat_messages(self):
        """Test creating sample with chat message list."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        sample = Sample(input=messages, ideal="Hi there!")

        assert isinstance(sample.input, list)
        assert len(sample.input) == 2

    def test_create_sample_with_multiple_ideals(self):
        """Test creating sample with multiple ideal answers."""
        sample = Sample(
            input="What color is the sky?",
            ideal=["blue", "Blue", "sky blue"],
        )

        assert isinstance(sample.ideal, list)
        assert len(sample.ideal) == 3

    def test_create_sample_with_metadata(self):
        """Test creating sample with metadata."""
        sample = Sample(
            input="Test",
            ideal="Answer",
            metadata={"category": "test", "difficulty": "easy"},
        )

        assert sample.metadata["category"] == "test"
        assert sample.metadata["difficulty"] == "easy"


# ============================================================================
# EvalSpec Model Tests
# ============================================================================


class TestEvalSpecModel:
    """Test EvalSpec Pydantic model."""

    def test_create_eval_spec_basic(self):
        """Test creating basic eval spec."""
        spec = EvalSpec(eval_id="basic_eval")

        assert spec.eval_id == "basic_eval"
        assert spec.description == ""
        assert spec.grader_type == GraderType.MATCH

    def test_create_eval_spec_with_all_fields(self):
        """Test creating eval spec with all fields."""
        spec = EvalSpec(
            eval_id="full_eval",
            description="Full evaluation test",
            metrics=["accuracy", "f1"],
            grader_type=GraderType.FUZZY_MATCH,
            grader_args={"threshold": 0.9},
        )

        assert spec.eval_id == "full_eval"
        assert len(spec.metrics) == 2
        assert spec.grader_type == GraderType.FUZZY_MATCH
        assert spec.grader_args["threshold"] == 0.9


# ============================================================================
# EvalResult Model Tests
# ============================================================================


class TestEvalResultModel:
    """Test EvalResult Pydantic model."""

    def test_create_passing_result(self):
        """Test creating a passing eval result."""
        result = EvalResult(
            sample_id="test_001",
            input="What is 2+2?",
            output="4",
            ideal="4",
            passed=True,
            score=1.0,
            grader_type=GraderType.MATCH,
            reason="Exact match found",
        )

        assert result.passed is True
        assert result.score == 1.0
        assert result.grader_type == GraderType.MATCH

    def test_create_failing_result(self):
        """Test creating a failing eval result."""
        result = EvalResult(
            sample_id="test_002",
            input="What is 2+2?",
            output="5",
            ideal="4",
            passed=False,
            score=0.0,
            grader_type=GraderType.MATCH,
            reason="No match found",
        )

        assert result.passed is False
        assert result.score == 0.0


# ============================================================================
# MatchGrader Tests
# ============================================================================


class TestMatchGrader:
    """Test MatchGrader class."""

    def test_exact_match_pass(self):
        """Test exact match grading - pass."""
        grader = MatchGrader(case_sensitive=False)
        passed, score, reason = grader.grade("Paris", "Paris")

        assert passed is True
        assert score == 1.0
        assert "Exact match" in reason

    def test_exact_match_case_insensitive(self):
        """Test case-insensitive matching."""
        grader = MatchGrader(case_sensitive=False)
        passed, score, reason = grader.grade("PARIS", "paris")

        assert passed is True

    def test_exact_match_case_sensitive_fail(self):
        """Test case-sensitive matching failure."""
        grader = MatchGrader(case_sensitive=True)
        passed, score, reason = grader.grade("PARIS", "Paris")

        assert passed is False

    def test_exact_match_with_whitespace(self):
        """Test matching with whitespace handling."""
        grader = MatchGrader(strip_whitespace=True)
        passed, score, reason = grader.grade("  Paris  ", "Paris")

        assert passed is True

    def test_exact_match_multiple_ideals(self):
        """Test matching with multiple ideal answers."""
        grader = MatchGrader()
        passed, score, reason = grader.grade("blue", ["red", "blue", "green"])

        assert passed is True

    def test_exact_match_fail(self):
        """Test exact match grading - fail."""
        grader = MatchGrader()
        passed, score, reason = grader.grade("London", "Paris")

        assert passed is False
        assert score == 0.0


# ============================================================================
# IncludesGrader Tests
# ============================================================================


class TestIncludesGrader:
    """Test IncludesGrader class."""

    def test_includes_single_pass(self):
        """Test includes grading with single keyword - pass."""
        grader = IncludesGrader()
        output = "The capital of France is Paris."
        passed, score, reason = grader.grade(output, "Paris")

        assert passed is True
        assert score == 1.0

    def test_includes_multiple_all_required_pass(self):
        """Test includes grading with all keywords required - pass."""
        grader = IncludesGrader(all_required=True)
        output = "The weather shows sunny conditions with temperature of 85°F."
        passed, score, reason = grader.grade(output, ["sunny", "temperature"])

        assert passed is True

    def test_includes_multiple_all_required_fail(self):
        """Test includes grading with all keywords required - fail."""
        grader = IncludesGrader(all_required=True)
        output = "The weather shows sunny conditions."
        passed, score, reason = grader.grade(output, ["sunny", "rain"])

        assert passed is False
        assert 0 < score < 1.0  # Partial score

    def test_includes_any_match(self):
        """Test includes grading with any keyword match."""
        grader = IncludesGrader(all_required=False)
        output = "The weather shows sunny conditions."
        passed, score, reason = grader.grade(output, ["sunny", "rain"])

        assert passed is True  # Only one needs to match

    def test_includes_case_sensitive(self):
        """Test case-sensitive includes grading."""
        grader = IncludesGrader(case_sensitive=True)
        output = "The weather in MIAMI is sunny."
        passed, score, reason = grader.grade(output, "miami")

        assert passed is False

    def test_includes_case_insensitive(self):
        """Test case-insensitive includes grading."""
        grader = IncludesGrader(case_sensitive=False)
        output = "The weather in MIAMI is sunny."
        passed, score, reason = grader.grade(output, "miami")

        assert passed is True


# ============================================================================
# FuzzyMatchGrader Tests
# ============================================================================


class TestFuzzyMatchGrader:
    """Test FuzzyMatchGrader class."""

    def test_fuzzy_match_identical(self):
        """Test fuzzy match with identical strings."""
        grader = FuzzyMatchGrader(threshold=0.8)
        passed, score, reason = grader.grade(
            "The weather is sunny",
            "The weather is sunny"
        )

        assert passed is True
        assert score == 1.0

    def test_fuzzy_match_similar(self):
        """Test fuzzy match with similar strings."""
        grader = FuzzyMatchGrader(threshold=0.5, method="token_overlap")
        passed, score, reason = grader.grade(
            "sunny weather in Miami today",
            "Miami has sunny weather"
        )

        assert passed is True  # Should have good token overlap
        assert score > 0.5

    def test_fuzzy_match_fail(self):
        """Test fuzzy match with dissimilar strings."""
        grader = FuzzyMatchGrader(threshold=0.8)
        passed, score, reason = grader.grade(
            "The weather is sunny",
            "Quantum mechanics explained"
        )

        assert passed is False
        assert score < 0.8

    def test_fuzzy_match_char_overlap(self):
        """Test fuzzy match with character overlap method."""
        grader = FuzzyMatchGrader(threshold=0.5, method="char_overlap")
        passed, score, reason = grader.grade("hello", "hella")

        assert passed is True

    def test_fuzzy_match_multiple_ideals(self):
        """Test fuzzy match with multiple ideal answers."""
        grader = FuzzyMatchGrader(threshold=0.5)
        passed, score, reason = grader.grade(
            "sunny day",
            ["completely different", "rainy night", "sunny weather"]
        )

        # Should match best with "sunny weather"
        assert score > 0.3


# ============================================================================
# ModelGradedGrader Tests
# ============================================================================


class TestModelGradedGrader:
    """Test ModelGradedGrader class."""

    def test_model_graded_without_evaluator(self):
        """Test model graded without evaluator function."""
        grader = ModelGradedGrader()
        passed, score, reason = grader.grade("output", "ideal")

        assert passed is False
        assert "No evaluator function" in reason

    def test_model_graded_with_evaluator(self):
        """Test model graded with evaluator function."""
        def mock_evaluator(input_text, output, ideal, prompt):
            return 0.9, "Excellent response"

        grader = ModelGradedGrader(
            evaluator_fn=mock_evaluator,
            threshold=0.7
        )
        passed, score, reason = grader.grade("output", "ideal", "input")

        assert passed is True
        assert score == 0.9
        assert reason == "Excellent response"

    def test_model_graded_below_threshold(self):
        """Test model graded below threshold."""
        def mock_evaluator(input_text, output, ideal, prompt):
            return 0.5, "Partially correct"

        grader = ModelGradedGrader(
            evaluator_fn=mock_evaluator,
            threshold=0.7
        )
        passed, score, reason = grader.grade("output", "ideal", "input")

        assert passed is False
        assert score == 0.5


class TestModelGradedClosedQAGrader:
    """Test ModelGradedClosedQAGrader class."""

    def test_closed_qa_grader_has_prompt(self):
        """Test closed QA grader has appropriate prompt."""
        grader = ModelGradedClosedQAGrader()

        assert "closed-domain" in grader.grading_prompt.lower()
        assert "reference" in grader.grading_prompt.lower()


class TestModelGradedFactGrader:
    """Test ModelGradedFactGrader class."""

    def test_fact_grader_has_prompt(self):
        """Test fact grader has appropriate prompt."""
        grader = ModelGradedFactGrader()

        assert "factual" in grader.grading_prompt.lower()
        assert "accuracy" in grader.grading_prompt.lower()


# ============================================================================
# CustomGrader Tests
# ============================================================================


class TestCustomGrader:
    """Test CustomGrader class."""

    def test_custom_grader(self):
        """Test custom grader with user-defined function."""
        def custom_fn(output, ideal, input_text):
            if "miami" in output.lower():
                return True, 1.0, "Contains Miami"
            return False, 0.0, "Missing Miami"

        grader = CustomGrader(grader_fn=custom_fn)

        passed, score, reason = grader.grade("Weather in Miami is sunny", "ideal")
        assert passed is True

        passed, score, reason = grader.grade("Weather in New York", "ideal")
        assert passed is False


# ============================================================================
# GraderFactory Tests
# ============================================================================


class TestGraderFactory:
    """Test GraderFactory class."""

    def test_create_match_grader(self):
        """Test creating match grader."""
        grader = GraderFactory.create(
            GraderType.MATCH,
            args={"case_sensitive": True}
        )

        assert isinstance(grader, MatchGrader)
        assert grader.case_sensitive is True

    def test_create_includes_grader(self):
        """Test creating includes grader."""
        grader = GraderFactory.create(
            GraderType.INCLUDES,
            args={"all_required": False}
        )

        assert isinstance(grader, IncludesGrader)
        assert grader.all_required is False

    def test_create_fuzzy_grader(self):
        """Test creating fuzzy match grader."""
        grader = GraderFactory.create(
            GraderType.FUZZY_MATCH,
            args={"threshold": 0.9, "method": "char_overlap"}
        )

        assert isinstance(grader, FuzzyMatchGrader)
        assert grader.threshold == 0.9

    def test_create_model_graded_closedqa(self):
        """Test creating model graded closed QA grader."""
        grader = GraderFactory.create(
            GraderType.MODEL_GRADED_CLOSEDQA,
            args={"threshold": 0.8}
        )

        assert isinstance(grader, ModelGradedClosedQAGrader)

    def test_create_model_graded_fact(self):
        """Test creating model graded fact grader."""
        grader = GraderFactory.create(GraderType.MODEL_GRADED_FACT)

        assert isinstance(grader, ModelGradedFactGrader)


# ============================================================================
# OpenAIEvalsRunner Tests
# ============================================================================


class TestOpenAIEvalsRunner:
    """Test OpenAIEvalsRunner class."""

    def test_runner_initialization(self):
        """Test runner initializes correctly."""
        runner = OpenAIEvalsRunner()

        assert runner.completion_fn is None
        assert runner.evaluator_fn is None
        assert runner.results == []

    def test_set_completion_fn(self):
        """Test setting completion function."""
        runner = OpenAIEvalsRunner()

        def my_fn(x):
            return x

        runner.set_completion_fn(my_fn)
        assert runner.completion_fn is not None

    def test_set_evaluator_fn(self):
        """Test setting evaluator function."""
        runner = OpenAIEvalsRunner()

        def my_eval(i, o, e, p):
            return 1.0, "Good"

        runner.set_evaluator_fn(my_eval)
        assert runner.evaluator_fn is not None

    @pytest.mark.asyncio
    async def test_run_eval_without_completion_fn(self):
        """Test running eval without completion function raises error."""
        runner = OpenAIEvalsRunner()
        spec = EvalSpec(eval_id="test")
        samples = [Sample(input="test", ideal="answer")]

        with pytest.raises(ValueError, match="Completion function not set"):
            await runner.run_eval(spec, samples)

    @pytest.mark.asyncio
    async def test_run_eval_success(self, evals_runner):
        """Test running evaluation successfully."""
        spec = EvalSpec(
            eval_id="math_test",
            grader_type=GraderType.MATCH,
        )
        samples = [
            Sample(input="What is 2+2?", ideal="4"),
        ]

        result = await evals_runner.run_eval(spec, samples)

        assert isinstance(result, EvalRunResult)
        assert result.eval_id == "math_test"
        assert result.total_samples == 1
        assert result.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_run_eval_with_failures(self, evals_runner):
        """Test running evaluation with some failures."""
        spec = EvalSpec(
            eval_id="mixed_test",
            grader_type=GraderType.MATCH,
        )
        samples = [
            Sample(input="What is 2+2?", ideal="4"),  # Will pass
            Sample(input="What is 5+5?", ideal="10"),  # Will fail (returns "Unknown")
        ]

        result = await evals_runner.run_eval(spec, samples)

        assert result.total_samples == 2
        assert result.passed_samples == 1
        assert result.failed_samples == 1
        assert result.accuracy == 0.5

    @pytest.mark.asyncio
    async def test_run_eval_results_stored(self, evals_runner):
        """Test that results are stored in runner."""
        spec = EvalSpec(eval_id="stored_test", grader_type=GraderType.MATCH)
        samples = [Sample(input="What is 2+2?", ideal="4")]

        await evals_runner.run_eval(spec, samples)

        results = evals_runner.get_results()
        assert len(results) == 1
        assert results[0].eval_id == "stored_test"

    @pytest.mark.asyncio
    async def test_clear_results(self, evals_runner):
        """Test clearing results."""
        spec = EvalSpec(eval_id="clear_test", grader_type=GraderType.MATCH)
        samples = [Sample(input="What is 2+2?", ideal="4")]

        await evals_runner.run_eval(spec, samples)
        evals_runner.clear_results()

        assert len(evals_runner.get_results()) == 0


# ============================================================================
# WeatherEvals Tests
# ============================================================================


class TestWeatherEvals:
    """Test WeatherEvals pre-built evaluations."""

    def test_weather_knowledge_eval(self):
        """Test weather knowledge evaluation."""
        spec, samples = WeatherEvals.get_weather_knowledge_eval()

        assert spec.eval_id == "weather_knowledge"
        assert spec.grader_type == GraderType.INCLUDES
        assert len(samples) > 0

    def test_safety_eval(self):
        """Test safety evaluation."""
        spec, samples = WeatherEvals.get_safety_eval()

        assert spec.eval_id == "weather_safety"
        assert spec.grader_type == GraderType.MODEL_GRADED_FACT
        assert len(samples) > 0

        # Check samples have evacuation-related content
        assert any("evacuate" in s.input.lower() for s in samples)

    def test_factual_accuracy_eval(self):
        """Test factual accuracy evaluation."""
        spec, samples = WeatherEvals.get_factual_accuracy_eval()

        assert spec.eval_id == "weather_factual"
        assert spec.grader_type == GraderType.FUZZY_MATCH
        assert len(samples) > 0

    def test_create_custom_eval(self):
        """Test creating custom weather evaluation."""
        custom_samples = [
            {"input": "Is it raining?", "ideal": "Yes"},
            {"input": "Temperature?", "ideal": "75°F"},
        ]

        spec, samples = WeatherEvals.create_custom_eval(
            eval_id="custom_weather",
            grader_type=GraderType.INCLUDES,
            samples=custom_samples,
            grader_args={"all_required": False},
        )

        assert spec.eval_id == "custom_weather"
        assert len(samples) == 2


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_match_eval(self):
        """Test create_match_eval function."""
        qa_pairs = [
            ("What is 1+1?", "2"),
            ("Capital of UK?", "London"),
        ]

        spec, samples = create_match_eval("math_geo_test", qa_pairs)

        assert spec.eval_id == "math_geo_test"
        assert spec.grader_type == GraderType.MATCH
        assert len(samples) == 2
        assert samples[0].input == "What is 1+1?"
        assert samples[0].ideal == "2"

    def test_create_includes_eval(self):
        """Test create_includes_eval function."""
        qa_pairs = [
            ("Describe weather", ["temperature", "humidity"]),
            ("Hurricane info", ["wind", "category"]),
        ]

        spec, samples = create_includes_eval(
            "weather_includes",
            qa_pairs,
            all_required=True
        )

        assert spec.eval_id == "weather_includes"
        assert spec.grader_type == GraderType.INCLUDES
        assert spec.grader_args["all_required"] is True
        assert len(samples) == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestOpenAIEvalsIntegration:
    """Integration tests for OpenAI Evals."""

    @pytest.mark.asyncio
    async def test_weather_knowledge_integration(self):
        """Test weather knowledge eval with mock completion."""
        async def mock_completion(input_text: str) -> str:
            if "Saffir-Simpson" in input_text:
                return "The Saffir-Simpson Hurricane Wind Scale measures hurricane intensity by wind speed, categorized from 1 to 5."
            elif "Category 5" in input_text:
                return "Category 5 hurricanes have wind speeds of 157 mph or greater."
            return "Unknown"

        runner = OpenAIEvalsRunner(completion_fn=mock_completion)
        spec, samples = WeatherEvals.get_weather_knowledge_eval()

        # Run only first 2 samples for quick test
        result = await runner.run_eval(spec, samples[:2])

        assert result.total_samples == 2
        assert result.accuracy > 0  # At least partial success

    @pytest.mark.asyncio
    async def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Create custom eval
        qa_pairs = [
            ("Miami weather", ["Miami", "temperature"]),
            ("Hurricane category 4", ["category", "130"]),
        ]
        spec, samples = create_includes_eval("full_workflow", qa_pairs)

        # Mock completion
        async def completion(input_text: str) -> str:
            if "Miami" in input_text:
                return "Miami has a temperature of 85°F today."
            elif "category" in input_text.lower():
                return "Category 4 hurricanes have winds between 130-156 mph."
            return "Unknown"

        runner = OpenAIEvalsRunner(completion_fn=completion)
        result = await runner.run_eval(spec, samples)

        # Verify full workflow executed
        assert result.eval_id == "full_workflow"
        assert result.total_samples == 2
        assert len(result.results) == 2
        assert result.metrics["accuracy"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
