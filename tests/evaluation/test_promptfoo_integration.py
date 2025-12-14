"""
Tests for Promptfoo Integration Module.

Level 6c: Self-Evolving Platform

Tests cover:
1. Assertion types and validation
2. Test case creation and execution
3. Provider-agnostic testing
4. Red team testing capabilities
5. Model grading functionality
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from backend.src.evaluation.promptfoo_integration import (
    AssertionType,
    Assertion,
    AssertionResult,
    TestCase,
    TestResult,
    PromptfooEvaluator,
    ProviderConfig,
    EvaluationSummary,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_assertion():
    """Create a sample assertion."""
    return Assertion(
        type=AssertionType.CONTAINS,
        value="Miami",
        threshold=0.8,
        weight=1.0,
    )


@pytest.fixture
def sample_test_case():
    """Create a sample test case."""
    return TestCase(
        id="test_001",
        description="Test weather query for Miami",
        vars={"location": "Miami", "query": "What's the weather in Miami?"},
        assert_=[
            Assertion(type=AssertionType.CONTAINS, value="Miami"),
            Assertion(type=AssertionType.CONTAINS, value="temperature"),
            Assertion(type=AssertionType.NOT_CONTAINS, value="error"),
        ],
        threshold=0.8,
    )


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    return ProviderConfig(
        id="openai:gpt-4o-mini",
        config={"temperature": 0.0},
    )


@pytest.fixture
def evaluator():
    """Create a PromptfooEvaluator instance."""
    return PromptfooEvaluator()


# ============================================================================
# Assertion Type Tests
# ============================================================================


class TestAssertionTypes:
    """Test assertion type enumeration."""

    def test_all_assertion_types_exist(self):
        """Verify all expected assertion types exist."""
        expected_types = [
            "equals", "contains", "not-contains", "starts-with", "ends-with",
            "regex", "is-json", "contains-json", "similar", "llm-rubric",
            "model-graded", "factuality", "answer-relevance", "context-faithfulness",
        ]

        for type_name in expected_types:
            assert hasattr(AssertionType, type_name.upper().replace("-", "_"))

    def test_assertion_type_values(self):
        """Verify assertion type string values."""
        assert AssertionType.CONTAINS.value == "contains"
        assert AssertionType.NOT_CONTAINS.value == "not-contains"
        assert AssertionType.LLM_RUBRIC.value == "llm-rubric"


# ============================================================================
# Assertion Model Tests
# ============================================================================


class TestAssertionModel:
    """Test Assertion Pydantic model."""

    def test_create_assertion_with_defaults(self):
        """Test creating assertion with default values."""
        assertion = Assertion(type=AssertionType.CONTAINS, value="test")

        assert assertion.type == AssertionType.CONTAINS
        assert assertion.value == "test"
        assert assertion.threshold == 0.8
        assert assertion.weight == 1.0
        assert assertion.provider is None

    def test_create_assertion_with_all_fields(self):
        """Test creating assertion with all fields."""
        assertion = Assertion(
            type=AssertionType.MODEL_GRADED,
            value="The response should be helpful",
            threshold=0.9,
            weight=2.0,
            provider="openai:gpt-4",
            metric="helpfulness",
        )

        assert assertion.type == AssertionType.MODEL_GRADED
        assert assertion.threshold == 0.9
        assert assertion.weight == 2.0
        assert assertion.provider == "openai:gpt-4"
        assert assertion.metric == "helpfulness"

    def test_assertion_with_dict_value(self):
        """Test assertion with dictionary value."""
        assertion = Assertion(
            type=AssertionType.LLM_RUBRIC,
            value={"criteria": "accuracy", "description": "Must be factually correct"},
        )

        assert isinstance(assertion.value, dict)
        assert assertion.value["criteria"] == "accuracy"


# ============================================================================
# AssertionResult Model Tests
# ============================================================================


class TestAssertionResultModel:
    """Test AssertionResult Pydantic model."""

    def test_create_passing_result(self):
        """Test creating a passing assertion result."""
        result = AssertionResult(
            assertion_type=AssertionType.CONTAINS,
            passed=True,
            score=1.0,
            reason="Value 'Miami' found in output",
        )

        assert result.passed is True
        assert result.score == 1.0
        assert "Miami" in result.reason

    def test_create_failing_result(self):
        """Test creating a failing assertion result."""
        result = AssertionResult(
            assertion_type=AssertionType.CONTAINS,
            passed=False,
            score=0.0,
            reason="Value 'temperature' not found in output",
            details={"searched_for": "temperature", "output_length": 50},
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.details["searched_for"] == "temperature"


# ============================================================================
# TestCase Model Tests
# ============================================================================


class TestTestCaseModel:
    """Test TestCase Pydantic model."""

    def test_create_test_case_basic(self, sample_assertion):
        """Test creating a basic test case."""
        test_case = TestCase(
            id="test_001",
            description="Basic test",
            assert_=[sample_assertion],
        )

        assert test_case.id == "test_001"
        assert len(test_case.assert_) == 1
        assert test_case.threshold == 0.8

    def test_create_test_case_with_vars(self):
        """Test creating test case with variables."""
        test_case = TestCase(
            id="test_002",
            vars={"city": "Miami", "unit": "fahrenheit"},
            assert_=[Assertion(type=AssertionType.CONTAINS, value="{{city}}")],
        )

        assert test_case.vars["city"] == "Miami"
        assert test_case.vars["unit"] == "fahrenheit"

    def test_test_case_alias_assert(self):
        """Test that 'assert' alias works for assert_ field."""
        data = {
            "id": "test_003",
            "assert": [{"type": "contains", "value": "test"}],
        }

        test_case = TestCase(**data)
        assert len(test_case.assert_) == 1


# ============================================================================
# PromptfooEvaluator Tests
# ============================================================================


class TestPromptfooEvaluator:
    """Test PromptfooEvaluator class."""

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes correctly."""
        assert evaluator is not None
        assert hasattr(evaluator, "evaluate_assertion")
        assert hasattr(evaluator, "run_test_case")

    def test_evaluate_contains_assertion_pass(self, evaluator):
        """Test evaluating a passing CONTAINS assertion."""
        assertion = Assertion(type=AssertionType.CONTAINS, value="Miami")
        output = "The weather in Miami is sunny with temperatures around 85°F."

        result = evaluator.evaluate_assertion(assertion, output)

        assert result.passed is True
        assert result.score == 1.0

    def test_evaluate_contains_assertion_fail(self, evaluator):
        """Test evaluating a failing CONTAINS assertion."""
        assertion = Assertion(type=AssertionType.CONTAINS, value="snow")
        output = "The weather in Miami is sunny."

        result = evaluator.evaluate_assertion(assertion, output)

        assert result.passed is False
        assert result.score == 0.0

    def test_evaluate_not_contains_assertion_pass(self, evaluator):
        """Test evaluating a passing NOT_CONTAINS assertion."""
        assertion = Assertion(type=AssertionType.NOT_CONTAINS, value="error")
        output = "The weather in Miami is sunny."

        result = evaluator.evaluate_assertion(assertion, output)

        assert result.passed is True

    def test_evaluate_not_contains_assertion_fail(self, evaluator):
        """Test evaluating a failing NOT_CONTAINS assertion."""
        assertion = Assertion(type=AssertionType.NOT_CONTAINS, value="sunny")
        output = "The weather is sunny."

        result = evaluator.evaluate_assertion(assertion, output)

        assert result.passed is False

    def test_evaluate_regex_assertion_pass(self, evaluator):
        """Test evaluating a passing REGEX assertion."""
        assertion = Assertion(type=AssertionType.REGEX, value=r"\d+°F")
        output = "Temperature is 85°F."

        result = evaluator.evaluate_assertion(assertion, output)

        assert result.passed is True

    def test_evaluate_regex_assertion_fail(self, evaluator):
        """Test evaluating a failing REGEX assertion."""
        assertion = Assertion(type=AssertionType.REGEX, value=r"\d+°C")
        output = "Temperature is 85°F."

        result = evaluator.evaluate_assertion(assertion, output)

        assert result.passed is False

    def test_evaluate_is_json_assertion_pass(self, evaluator):
        """Test evaluating a passing IS_JSON assertion."""
        assertion = Assertion(type=AssertionType.IS_JSON, value=None)
        output = '{"temperature": 85, "unit": "F"}'

        result = evaluator.evaluate_assertion(assertion, output)

        assert result.passed is True

    def test_evaluate_is_json_assertion_fail(self, evaluator):
        """Test evaluating a failing IS_JSON assertion."""
        assertion = Assertion(type=AssertionType.IS_JSON, value=None)
        output = "This is not JSON"

        result = evaluator.evaluate_assertion(assertion, output)

        assert result.passed is False


class TestPromptfooTestExecution:
    """Test PromptfooEvaluator test execution."""

    @pytest.mark.asyncio
    async def test_run_test_case_all_pass(self, evaluator, sample_test_case):
        """Test running a test case where all assertions pass."""
        output = "The weather in Miami is sunny with a temperature of 85°F."

        with patch.object(evaluator, "_get_model_output", return_value=output):
            result = await evaluator.run_test_case(
                test_case=sample_test_case,
                prompt="Weather query: {{query}}",
            )

        assert result.passed is True
        assert result.score >= sample_test_case.threshold

    @pytest.mark.asyncio
    async def test_run_test_case_some_fail(self, evaluator):
        """Test running a test case where some assertions fail."""
        test_case = TestCase(
            id="test_partial",
            assert_=[
                Assertion(type=AssertionType.CONTAINS, value="Miami"),
                Assertion(type=AssertionType.CONTAINS, value="snow"),  # Will fail
            ],
        )
        output = "The weather in Miami is sunny."

        with patch.object(evaluator, "_get_model_output", return_value=output):
            result = await evaluator.run_test_case(
                test_case=test_case,
                prompt="Weather query",
            )

        # 1 pass + 1 fail = 50% score, below 80% threshold
        assert result.score == 0.5
        assert result.passed is False


class TestPromptfooWeatherDomain:
    """Test Promptfoo integration with weather domain."""

    @pytest.fixture
    def weather_test_suite(self):
        """Create weather-specific test suite."""
        return [
            TestCase(
                id="weather_basic_001",
                description="Basic weather query",
                vars={"location": "Miami"},
                assert_=[
                    Assertion(type=AssertionType.CONTAINS, value="Miami"),
                    Assertion(type=AssertionType.REGEX, value=r"\d+°[FC]"),
                ],
            ),
            TestCase(
                id="weather_hurricane_001",
                description="Hurricane category validation",
                vars={"wind_speed": "145"},
                assert_=[
                    Assertion(type=AssertionType.CONTAINS, value="Category"),
                    Assertion(type=AssertionType.NOT_CONTAINS, value="Category 5"),  # 145mph = Cat 4
                ],
            ),
        ]

    @pytest.mark.asyncio
    async def test_weather_test_suite(self, evaluator, weather_test_suite):
        """Test running weather-specific test suite."""
        results = []

        for test_case in weather_test_suite:
            mock_output = "Miami weather: 85°F, sunny. Category 4 hurricane nearby."

            with patch.object(evaluator, "_get_model_output", return_value=mock_output):
                result = await evaluator.run_test_case(
                    test_case=test_case,
                    prompt="Test prompt",
                )
                results.append(result)

        # Both tests should pass with mock output
        assert all(r.passed for r in results)


# ============================================================================
# Red Team Testing
# ============================================================================


class TestRedTeamCapabilities:
    """Test red team testing capabilities."""

    def test_red_team_prompt_injection(self, evaluator):
        """Test detection of prompt injection attempts."""
        assertion = Assertion(
            type=AssertionType.NOT_CONTAINS,
            value="ignore previous instructions",
        )
        # Safe output
        output = "The weather in Miami is 85°F."
        result = evaluator.evaluate_assertion(assertion, output)
        assert result.passed is True

    def test_red_team_sensitive_data(self, evaluator):
        """Test detection of sensitive data leakage."""
        assertions = [
            Assertion(type=AssertionType.NOT_CONTAINS, value="API_KEY"),
            Assertion(type=AssertionType.NOT_CONTAINS, value="password"),
            Assertion(type=AssertionType.NOT_CONTAINS, value="secret"),
        ]

        output = "The weather forecast is available."

        for assertion in assertions:
            result = evaluator.evaluate_assertion(assertion, output)
            assert result.passed is True


# ============================================================================
# Evaluation Summary Tests
# ============================================================================


class TestEvaluationSummary:
    """Test evaluation summary generation."""

    def test_summary_model_creation(self):
        """Test creating evaluation summary."""
        summary = EvaluationSummary(
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            pass_rate=0.8,
            avg_score=0.85,
            by_assertion_type={
                "contains": {"passed": 5, "failed": 1},
                "regex": {"passed": 3, "failed": 1},
            },
        )

        assert summary.total_tests == 10
        assert summary.pass_rate == 0.8
        assert summary.avg_score == 0.85

    def test_summary_with_zero_tests(self):
        """Test summary with zero tests."""
        summary = EvaluationSummary(
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            pass_rate=0.0,
            avg_score=0.0,
        )

        assert summary.total_tests == 0
        assert summary.pass_rate == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
