"""
Tests for DeepEval LLM Unit Testing Integration.

Level 6a: Advanced Evaluation Framework

Tests:
1. Single query evaluation
2. Test suite execution
3. Synthetic test case generation
4. Threshold validation
5. Fallback heuristic metrics
"""

import pytest
from backend.src.evaluation.deepeval_integration import (
    DeepEvalIntegration,
    DeepEvalResult,
)


class TestDeepEvalIntegration:
    """Test suite for DeepEvalIntegration."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator without LLM (uses fallback heuristics)."""
        return DeepEvalIntegration(llm=None)

    @pytest.mark.asyncio
    async def test_evaluate_query_returns_result(self, evaluator):
        """Test that evaluation returns a DeepEvalResult."""
        result = await evaluator.evaluate_query(
            query="What is the weather in Tampa?",
            answer="The weather in Tampa is sunny with a temperature of 85°F.",
            retrieved_contexts=[
                "Tampa weather today is sunny with 85°F.",
                "Tampa is located in Florida.",
            ],
        )

        assert isinstance(result, DeepEvalResult)
        assert 0 <= result.answer_relevancy <= 1
        assert 0 <= result.faithfulness <= 1
        assert 0 <= result.contextual_precision <= 1
        assert 0 <= result.contextual_recall <= 1
        assert 0 <= result.overall_score <= 1

    @pytest.mark.asyncio
    async def test_high_faithfulness_when_grounded(self, evaluator):
        """Test that faithfulness is high when answer is grounded in context."""
        result = await evaluator.evaluate_query(
            query="What is the hurricane category?",
            answer="Hurricane Michael is a Category 4 storm with 130 mph winds.",
            retrieved_contexts=[
                "Hurricane Michael is Category 4 with winds of 130 mph.",
                "The storm is approaching Florida.",
            ],
        )

        # Answer directly reflects context
        assert result.faithfulness >= 0.5

    @pytest.mark.asyncio
    async def test_low_faithfulness_when_not_grounded(self, evaluator):
        """Test that faithfulness is lower when answer contains info not in context."""
        result = await evaluator.evaluate_query(
            query="What is the temperature?",
            answer="The temperature is 95°F with high humidity and storm approaching.",
            retrieved_contexts=[
                "Today is Monday.",
                "The sky is blue.",
            ],
        )

        # Answer has info not in context
        assert result.faithfulness < 0.8

    @pytest.mark.asyncio
    async def test_answer_relevancy_addresses_query(self, evaluator):
        """Test that answer relevancy is high when answer addresses query."""
        result = await evaluator.evaluate_query(
            query="What is the weather forecast for tomorrow?",
            answer="Tomorrow's weather forecast shows sunny skies with a temperature of 88°F.",
            retrieved_contexts=[
                "Tomorrow's forecast shows sunny skies.",
                "Temperature will be 88°F.",
            ],
        )

        # Answer directly addresses the query
        assert result.answer_relevancy > 0

    @pytest.mark.asyncio
    async def test_contextual_precision_with_relevant_contexts(self, evaluator):
        """Test contextual precision when contexts are relevant to query."""
        result = await evaluator.evaluate_query(
            query="hurricane wind speed category",
            answer="Hurricane categories are based on wind speed.",
            retrieved_contexts=[
                "Hurricane winds are measured in mph to determine category.",
                "Category 4 hurricanes have winds of 130-156 mph.",
                "Wind speed affects storm surge height.",
            ],
        )

        # Contexts are relevant to query
        assert result.contextual_precision > 0

    @pytest.mark.asyncio
    async def test_overall_score_calculation(self, evaluator):
        """Test that overall score is weighted properly."""
        result = await evaluator.evaluate_query(
            query="test query",
            answer="test answer",
            retrieved_contexts=["test context"],
        )

        # Overall should be weighted average
        expected = (
            result.answer_relevancy * 0.25
            + result.faithfulness * 0.35
            + result.contextual_precision * 0.20
            + result.contextual_recall * 0.20
        )

        assert abs(result.overall_score - expected) < 0.01

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, evaluator):
        """Test handling of empty query."""
        result = await evaluator.evaluate_query(
            query="",
            answer="some answer",
            retrieved_contexts=["some context"],
        )

        assert isinstance(result, DeepEvalResult)

    @pytest.mark.asyncio
    async def test_empty_context_handling(self, evaluator):
        """Test handling of empty context."""
        result = await evaluator.evaluate_query(
            query="test query",
            answer="test answer",
            retrieved_contexts=[],
        )

        assert isinstance(result, DeepEvalResult)

    @pytest.mark.asyncio
    async def test_details_include_method(self, evaluator):
        """Test that result includes method information."""
        result = await evaluator.evaluate_query(
            query="test",
            answer="answer",
            retrieved_contexts=["context"],
        )

        assert "method" in result.details
        assert result.details["method"] == "fallback_heuristic"

    @pytest.mark.asyncio
    async def test_test_case_id_preserved(self, evaluator):
        """Test that test case ID is preserved in result."""
        result = await evaluator.evaluate_query(
            query="test",
            answer="answer",
            retrieved_contexts=["context"],
            test_case_id="test_123",
        )

        assert result.test_case_id == "test_123"


class TestDeepEvalTestSuite:
    """Test suite for test suite execution."""

    @pytest.fixture
    def evaluator(self):
        return DeepEvalIntegration(llm=None)

    @pytest.mark.asyncio
    async def test_run_test_suite(self, evaluator):
        """Test batch evaluation of multiple queries."""
        test_cases = [
            {
                "query": "What is the temperature?",
                "answer": "The temperature is 85°F.",
                "contexts": ["Temperature today is 85°F."],
            },
            {
                "query": "Is it raining?",
                "answer": "No, it is sunny today.",
                "contexts": ["Weather is sunny with no rain."],
            },
            {
                "query": "What is the wind speed?",
                "answer": "Wind speed is 15 mph.",
                "contexts": ["Wind is blowing at 15 mph from the east."],
            },
        ]

        result = await evaluator.run_test_suite(test_cases)

        assert "avg_answer_relevancy" in result
        assert "avg_faithfulness" in result
        assert "avg_contextual_precision" in result
        assert "avg_contextual_recall" in result
        assert "total_tests" in result
        assert result["total_tests"] == 3
        assert "pass_rate" in result
        assert "results" in result
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_test_suite_with_expected_output(self, evaluator):
        """Test suite evaluation with expected outputs."""
        test_cases = [
            {
                "id": "test_1",
                "query": "What is the hurricane category?",
                "answer": "Category 4",
                "contexts": ["Hurricane Michael is Category 4."],
                "expected": "The hurricane is Category 4 with 130 mph winds.",
            },
        ]

        result = await evaluator.run_test_suite(test_cases)

        assert "passed_count" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["test_case_id"] == "test_1"


class TestSyntheticTestGeneration:
    """Test suite for synthetic test case generation."""

    @pytest.fixture
    def evaluator(self):
        return DeepEvalIntegration(llm=None)

    def test_generate_synthetic_test_cases(self, evaluator):
        """Test synthetic test case generation."""
        test_cases = evaluator.generate_synthetic_test_cases(
            domain="weather", count=5
        )

        assert len(test_cases) == 5

        for tc in test_cases:
            assert "id" in tc
            assert "query" in tc
            assert "contexts" in tc
            assert "answer" in tc
            assert "expected" in tc
            assert isinstance(tc["contexts"], list)
            assert len(tc["contexts"]) > 0

    def test_synthetic_cases_have_unique_ids(self, evaluator):
        """Test that generated test cases have unique IDs."""
        test_cases = evaluator.generate_synthetic_test_cases(count=10)

        ids = [tc["id"] for tc in test_cases]
        assert len(ids) == len(set(ids))  # All unique

    def test_synthetic_cases_contain_weather_content(self, evaluator):
        """Test that generated cases contain weather-related content."""
        test_cases = evaluator.generate_synthetic_test_cases(
            domain="weather", count=10
        )

        weather_keywords = [
            "temperature", "hurricane", "wind", "weather", "storm",
            "evacuation", "forecast", "°F", "mph",
        ]

        for tc in test_cases:
            combined = f"{tc['query']} {tc['answer']} {' '.join(tc['contexts'])}"
            has_weather_keyword = any(
                keyword.lower() in combined.lower()
                for keyword in weather_keywords
            )
            assert has_weather_keyword, f"No weather keyword in: {combined}"


class TestDeepEvalThresholds:
    """Test threshold management."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        evaluator = DeepEvalIntegration()

        assert evaluator.thresholds["answer_relevancy"] == 0.85
        assert evaluator.thresholds["faithfulness"] == 0.85
        assert evaluator.thresholds["contextual_precision"] == 0.85
        assert evaluator.thresholds["contextual_recall"] == 0.85

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        custom = {"faithfulness": 0.95, "answer_relevancy": 0.90}
        evaluator = DeepEvalIntegration(thresholds=custom)

        assert evaluator.thresholds["faithfulness"] == 0.95
        assert evaluator.thresholds["answer_relevancy"] == 0.90

    @pytest.mark.asyncio
    async def test_passed_based_on_thresholds(self):
        """Test that passed flag respects thresholds."""
        # Use very low thresholds to ensure passing
        evaluator = DeepEvalIntegration(
            thresholds={
                "answer_relevancy": 0.1,
                "faithfulness": 0.1,
                "contextual_precision": 0.1,
                "contextual_recall": 0.1,
            }
        )

        result = await evaluator.evaluate_query(
            query="weather",
            answer="weather is nice",
            retrieved_contexts=["weather is nice today"],
        )

        # With low thresholds, should pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_failed_with_high_thresholds(self):
        """Test that passed flag is False with high thresholds."""
        # Use very high thresholds that can't be met
        evaluator = DeepEvalIntegration(
            thresholds={
                "answer_relevancy": 0.99,
                "faithfulness": 0.99,
                "contextual_precision": 0.99,
                "contextual_recall": 0.99,
            }
        )

        result = await evaluator.evaluate_query(
            query="x",
            answer="y",
            retrieved_contexts=["z"],
        )

        # With extremely high thresholds, should fail
        assert result.passed is False


class TestDeepEvalMetricWeights:
    """Test metric weight configuration."""

    def test_default_weights(self):
        """Test default weight values."""
        evaluator = DeepEvalIntegration()

        assert evaluator.WEIGHTS["answer_relevancy"] == 0.25
        assert evaluator.WEIGHTS["faithfulness"] == 0.35  # Highest
        assert evaluator.WEIGHTS["contextual_precision"] == 0.20
        assert evaluator.WEIGHTS["contextual_recall"] == 0.20

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0."""
        evaluator = DeepEvalIntegration()

        total = sum(evaluator.WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_overall_score_uses_weights(self, ):
        """Test that overall score calculation uses weights correctly."""
        evaluator = DeepEvalIntegration()

        result = await evaluator.evaluate_query(
            query="What is the temperature in Tampa?",
            answer="The temperature is 85 degrees Fahrenheit in Tampa.",
            retrieved_contexts=["Tampa temperature is 85°F today."],
        )

        # Manually calculate expected overall score
        expected = (
            result.answer_relevancy * evaluator.WEIGHTS["answer_relevancy"]
            + result.faithfulness * evaluator.WEIGHTS["faithfulness"]
            + result.contextual_precision * evaluator.WEIGHTS["contextual_precision"]
            + result.contextual_recall * evaluator.WEIGHTS["contextual_recall"]
        )

        assert abs(result.overall_score - expected) < 0.001
