"""
Tests for Ragas RAG Evaluation.

Level 6a: RAG-Specific Metrics Testing

Tests:
1. Faithfulness scoring (answer grounded in context)
2. Context precision (top-k chunks relevant)
3. Context recall (all relevant info retrieved)
4. Answer relevancy (addresses query)
5. Dataset evaluation (batch processing)
"""

import pytest

from backend.src.evaluation.ragas_evaluator import RagasEvaluator, RagasResult


class TestRagasEvaluator:
    """Test suite for RagasEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator without LLM (uses fallback heuristics)."""
        return RagasEvaluator(llm=None, embeddings=None)

    @pytest.mark.asyncio
    async def test_evaluate_rag_query_returns_result(self, evaluator):
        """Test that evaluation returns a RagasResult."""
        result = await evaluator.evaluate_rag_query(
            query="What is the weather in Tampa?",
            retrieved_chunks=[
                "Tampa weather today is sunny with 85°F.",
                "Tampa is located in Florida.",
            ],
            generated_answer="The weather in Tampa is sunny with a temperature of 85°F.",
        )

        assert isinstance(result, RagasResult)
        assert 0 <= result.faithfulness <= 1
        assert 0 <= result.context_precision <= 1
        assert 0 <= result.answer_relevancy <= 1
        assert 0 <= result.overall_score <= 1

    @pytest.mark.asyncio
    async def test_high_faithfulness_when_grounded(self, evaluator):
        """Test that faithfulness is high when answer is grounded in context."""
        result = await evaluator.evaluate_rag_query(
            query="What is the hurricane category?",
            retrieved_chunks=[
                "Hurricane Michael is Category 4 with winds of 130 mph.",
                "The storm is approaching Florida.",
            ],
            generated_answer="Hurricane Michael is a Category 4 storm with 130 mph winds.",
        )

        # Answer directly reflects context, should have high faithfulness
        assert result.faithfulness >= 0.5

    @pytest.mark.asyncio
    async def test_low_faithfulness_when_not_grounded(self, evaluator):
        """Test that faithfulness is lower when answer contains info not in context."""
        result = await evaluator.evaluate_rag_query(
            query="What is the temperature?",
            retrieved_chunks=[
                "Today is Monday.",
                "The sky is blue.",
            ],
            generated_answer="The temperature is 95°F with high humidity and storm approaching.",
        )

        # Answer has info not in context
        assert result.faithfulness < 0.8

    @pytest.mark.asyncio
    async def test_context_precision_with_relevant_chunks(self, evaluator):
        """Test context precision when chunks are relevant to query."""
        result = await evaluator.evaluate_rag_query(
            query="hurricane wind speed category",
            retrieved_chunks=[
                "Hurricane winds are measured in mph to determine category.",
                "Category 4 hurricanes have winds of 130-156 mph.",
                "Wind speed affects storm surge height.",
            ],
            generated_answer="Hurricane categories are based on wind speed.",
        )

        # Chunks are relevant to query
        assert result.context_precision > 0

    @pytest.mark.asyncio
    async def test_context_recall_with_ground_truth(self, evaluator):
        """Test context recall when ground truth is provided."""
        result = await evaluator.evaluate_rag_query(
            query="What is the temperature in Tampa?",
            retrieved_chunks=[
                "Tampa temperature is 85°F today.",
                "Humidity is at 78%.",
            ],
            generated_answer="The temperature in Tampa is 85°F.",
            ground_truth_answer="Tampa has a temperature of 85°F with 78% humidity.",
        )

        # Context contains ground truth information
        assert result.context_recall > 0

    @pytest.mark.asyncio
    async def test_answer_relevancy_addresses_query(self, evaluator):
        """Test that answer relevancy is high when answer addresses query."""
        result = await evaluator.evaluate_rag_query(
            query="What is the weather forecast for tomorrow?",
            retrieved_chunks=[
                "Tomorrow's forecast shows sunny skies.",
                "Temperature will be 88°F.",
            ],
            generated_answer="Tomorrow's weather forecast shows sunny skies with a temperature of 88°F.",
        )

        # Answer directly addresses the query
        assert result.answer_relevancy > 0

    @pytest.mark.asyncio
    async def test_overall_score_calculation(self, evaluator):
        """Test that overall score is weighted properly."""
        result = await evaluator.evaluate_rag_query(
            query="test query",
            retrieved_chunks=["test context"],
            generated_answer="test answer",
        )

        # Overall should be weighted average
        expected = (
            result.faithfulness * 0.4
            + result.context_precision * 0.2
            + result.context_recall * 0.2
            + result.answer_relevancy * 0.2
        )

        assert abs(result.overall_score - expected) < 0.01

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, evaluator):
        """Test handling of empty query."""
        result = await evaluator.evaluate_rag_query(
            query="",
            retrieved_chunks=["some context"],
            generated_answer="some answer",
        )

        assert isinstance(result, RagasResult)

    @pytest.mark.asyncio
    async def test_empty_context_handling(self, evaluator):
        """Test handling of empty context."""
        result = await evaluator.evaluate_rag_query(
            query="test query",
            retrieved_chunks=[],
            generated_answer="test answer",
        )

        assert isinstance(result, RagasResult)

    @pytest.mark.asyncio
    async def test_details_include_method(self, evaluator):
        """Test that result includes method information."""
        result = await evaluator.evaluate_rag_query(
            query="test",
            retrieved_chunks=["context"],
            generated_answer="answer",
        )

        assert "method" in result.details
        assert result.details["method"] == "fallback_heuristic"


class TestRagasDatasetEvaluation:
    """Test suite for dataset evaluation."""

    @pytest.fixture
    def evaluator(self):
        return RagasEvaluator(llm=None, embeddings=None)

    @pytest.mark.asyncio
    async def test_evaluate_dataset(self, evaluator):
        """Test batch evaluation of multiple queries."""
        queries = [
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

        result = await evaluator.evaluate_dataset(queries)

        assert "avg_faithfulness" in result
        assert "avg_context_precision" in result
        assert "avg_answer_relevancy" in result
        assert "total_queries" in result
        assert result["total_queries"] == 3
        assert "pass_rate" in result

    @pytest.mark.asyncio
    async def test_evaluate_dataset_with_ground_truth(self, evaluator):
        """Test dataset evaluation with ground truth answers."""
        queries = [
            {
                "query": "What is the hurricane category?",
                "answer": "Category 4",
                "contexts": ["Hurricane Michael is Category 4."],
                "ground_truth": "The hurricane is Category 4 with 130 mph winds.",
            },
        ]

        result = await evaluator.evaluate_dataset(queries)

        assert "avg_context_recall" in result


class TestRagasThresholds:
    """Test threshold management."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        evaluator = RagasEvaluator()
        thresholds = evaluator.get_thresholds()

        assert thresholds["faithfulness"] == 0.90
        assert thresholds["context_precision"] == 0.85
        assert thresholds["context_recall"] == 0.85
        assert thresholds["answer_relevancy"] == 0.90

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        custom = {"faithfulness": 0.95, "context_precision": 0.90}
        evaluator = RagasEvaluator(thresholds=custom)
        thresholds = evaluator.get_thresholds()

        assert thresholds["faithfulness"] == 0.95
        assert thresholds["context_precision"] == 0.90

    def test_set_thresholds(self):
        """Test updating thresholds."""
        evaluator = RagasEvaluator()
        evaluator.set_thresholds({"faithfulness": 0.80})

        thresholds = evaluator.get_thresholds()
        assert thresholds["faithfulness"] == 0.80

    @pytest.mark.asyncio
    async def test_passed_based_on_thresholds(self):
        """Test that passed flag respects thresholds."""
        # Use very low thresholds to ensure passing
        evaluator = RagasEvaluator(
            thresholds={
                "faithfulness": 0.1,
                "context_precision": 0.1,
                "context_recall": 0.1,
                "answer_relevancy": 0.1,
            }
        )

        result = await evaluator.evaluate_rag_query(
            query="weather",
            retrieved_chunks=["weather is nice"],
            generated_answer="weather is nice today",
        )

        # With low thresholds, should pass
        assert result.passed is True
