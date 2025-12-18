"""
Tests for TruLens Real-Time Evaluation Integration.

Level 6b: Self-Improvement Platform

Tests:
1. Single query feedback evaluation
2. Groundedness calculation
3. Answer relevance calculation
4. Context relevance calculation
5. Feedback summary aggregation
6. Record storage and retrieval
7. Threshold management
"""

import pytest

from backend.src.evaluation.trulens_integration import (
    TruLensFeedback,
    TruLensIntegration,
    TruLensRecord,
)


class TestTruLensIntegration:
    """Test suite for TruLensIntegration."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator without LLM (uses fallback heuristics)."""
        return TruLensIntegration(llm=None)

    @pytest.mark.asyncio
    async def test_evaluate_query_returns_feedback(self, evaluator):
        """Test that evaluation returns a TruLensFeedback."""
        feedback = await evaluator.evaluate_query(
            query="What is the weather in Tampa?",
            response="The weather in Tampa is sunny with a temperature of 85°F.",
            contexts=[
                "Tampa weather today is sunny with 85°F.",
                "Tampa is located in Florida.",
            ],
        )

        assert isinstance(feedback, TruLensFeedback)
        assert 0 <= feedback.groundedness <= 1
        assert 0 <= feedback.answer_relevance <= 1
        assert 0 <= feedback.context_relevance <= 1
        assert 0 <= feedback.overall_score <= 1

    @pytest.mark.asyncio
    async def test_high_groundedness_when_grounded(self, evaluator):
        """Test that groundedness is high when response is grounded in context."""
        feedback = await evaluator.evaluate_query(
            query="What is the hurricane category?",
            response="Hurricane Michael is a Category 4 storm with 130 mph winds.",
            contexts=[
                "Hurricane Michael is Category 4 with winds of 130 mph.",
                "The storm is approaching Florida.",
            ],
        )

        # Response directly reflects context
        assert feedback.groundedness >= 0.5

    @pytest.mark.asyncio
    async def test_low_groundedness_when_not_grounded(self, evaluator):
        """Test that groundedness is lower when response contains info not in context."""
        feedback = await evaluator.evaluate_query(
            query="What is the temperature?",
            response="The temperature is 95°F with high humidity and storm approaching.",
            contexts=[
                "Today is Monday.",
                "The sky is blue.",
            ],
        )

        # Response has info not in context
        assert feedback.groundedness < 0.8

    @pytest.mark.asyncio
    async def test_answer_relevance_addresses_query(self, evaluator):
        """Test that answer relevance is high when response addresses query."""
        feedback = await evaluator.evaluate_query(
            query="What is the weather forecast for tomorrow?",
            response="Tomorrow's weather forecast shows sunny skies with a temperature of 88°F.",
            contexts=["Tomorrow's forecast shows sunny skies."],
        )

        # Response directly addresses the query
        assert feedback.answer_relevance > 0

    @pytest.mark.asyncio
    async def test_context_relevance_with_relevant_contexts(self, evaluator):
        """Test context relevance when contexts are relevant to query."""
        feedback = await evaluator.evaluate_query(
            query="hurricane wind speed category",
            response="Hurricane categories are based on wind speed.",
            contexts=[
                "Hurricane winds are measured in mph to determine category.",
                "Category 4 hurricanes have winds of 130-156 mph.",
                "Wind speed affects storm surge height.",
            ],
        )

        # Contexts are relevant to query
        assert feedback.context_relevance > 0

    @pytest.mark.asyncio
    async def test_overall_score_is_average(self, evaluator):
        """Test that overall score is average of all feedback scores."""
        feedback = await evaluator.evaluate_query(
            query="test query",
            response="test response",
            contexts=["test context"],
        )

        # Overall should be average
        expected = (
            feedback.groundedness
            + feedback.answer_relevance
            + feedback.context_relevance
        ) / 3

        assert abs(feedback.overall_score - expected) < 0.01

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, evaluator):
        """Test handling of empty query."""
        feedback = await evaluator.evaluate_query(
            query="",
            response="some response",
            contexts=["some context"],
        )

        assert isinstance(feedback, TruLensFeedback)

    @pytest.mark.asyncio
    async def test_empty_context_handling(self, evaluator):
        """Test handling of empty context."""
        feedback = await evaluator.evaluate_query(
            query="test query",
            response="test response",
            contexts=[],
        )

        assert isinstance(feedback, TruLensFeedback)

    @pytest.mark.asyncio
    async def test_details_include_method(self, evaluator):
        """Test that feedback includes method information."""
        feedback = await evaluator.evaluate_query(
            query="test",
            response="response",
            contexts=["context"],
        )

        assert "method" in feedback.details
        assert feedback.details["method"] == "fallback_heuristic"

    @pytest.mark.asyncio
    async def test_record_id_preserved(self, evaluator):
        """Test that record ID is preserved in stored records."""
        await evaluator.evaluate_query(
            query="test",
            response="response",
            contexts=["context"],
            record_id="test_123",
        )

        records = evaluator.get_records()
        assert len(records) == 1
        assert records[0]["record_id"] == "test_123"


class TestTruLensFeedbackSummary:
    """Test suite for feedback summary aggregation."""

    @pytest.fixture
    def evaluator(self):
        return TruLensIntegration(llm=None)

    @pytest.mark.asyncio
    async def test_feedback_summary_empty(self, evaluator):
        """Test feedback summary with no records."""
        summary = evaluator.get_feedback_summary()

        assert summary["total_records"] == 0
        assert summary["avg_groundedness"] == 0.0
        assert summary["overall_score"] == 0.0

    @pytest.mark.asyncio
    async def test_feedback_summary_with_records(self, evaluator):
        """Test feedback summary with multiple records."""
        # Add multiple evaluations
        await evaluator.evaluate_query(
            query="What is the temperature?",
            response="The temperature is 85°F.",
            contexts=["Temperature today is 85°F."],
        )
        await evaluator.evaluate_query(
            query="Is it raining?",
            response="No, it is sunny today.",
            contexts=["Weather is sunny with no rain."],
        )
        await evaluator.evaluate_query(
            query="What is the wind speed?",
            response="Wind speed is 15 mph.",
            contexts=["Wind is blowing at 15 mph from the east."],
        )

        summary = evaluator.get_feedback_summary()

        assert summary["total_records"] == 3
        assert "avg_groundedness" in summary
        assert "avg_answer_relevance" in summary
        assert "avg_context_relevance" in summary
        assert "overall_score" in summary
        assert "pass_rate" in summary

    @pytest.mark.asyncio
    async def test_get_records_with_limit(self, evaluator):
        """Test getting records with limit."""
        # Add multiple evaluations
        for i in range(5):
            await evaluator.evaluate_query(
                query=f"Query {i}",
                response=f"Response {i}",
                contexts=[f"Context {i}"],
            )

        # Get only 3 records
        records = evaluator.get_records(limit=3)
        assert len(records) == 3

    @pytest.mark.asyncio
    async def test_clear_records(self, evaluator):
        """Test clearing all records."""
        # Add some evaluations
        await evaluator.evaluate_query(
            query="test",
            response="response",
            contexts=["context"],
        )

        assert len(evaluator.records) > 0

        # Clear records
        evaluator.clear_records()

        assert len(evaluator.records) == 0


class TestTruLensThresholds:
    """Test threshold management."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        evaluator = TruLensIntegration()
        thresholds = evaluator.get_thresholds()

        assert thresholds["groundedness"] == 0.85
        assert thresholds["answer_relevance"] == 0.85
        assert thresholds["context_relevance"] == 0.85

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        custom = {"groundedness": 0.90, "answer_relevance": 0.80}
        evaluator = TruLensIntegration(thresholds=custom)
        thresholds = evaluator.get_thresholds()

        assert thresholds["groundedness"] == 0.90
        assert thresholds["answer_relevance"] == 0.80

    def test_set_thresholds(self):
        """Test updating thresholds."""
        evaluator = TruLensIntegration()
        evaluator.set_thresholds({"groundedness": 0.70})

        thresholds = evaluator.get_thresholds()
        assert thresholds["groundedness"] == 0.70

    @pytest.mark.asyncio
    async def test_passed_based_on_thresholds(self):
        """Test that passed flag respects thresholds."""
        # Use very low thresholds to ensure passing
        evaluator = TruLensIntegration(
            thresholds={
                "groundedness": 0.1,
                "answer_relevance": 0.1,
                "context_relevance": 0.1,
            }
        )

        feedback = await evaluator.evaluate_query(
            query="weather",
            response="weather is nice",
            contexts=["weather is nice today"],
        )

        # With low thresholds, should pass
        assert feedback.passed is True

    @pytest.mark.asyncio
    async def test_failed_with_high_thresholds(self):
        """Test that passed flag is False with high thresholds."""
        # Use very high thresholds that can't be met
        evaluator = TruLensIntegration(
            thresholds={
                "groundedness": 0.99,
                "answer_relevance": 0.99,
                "context_relevance": 0.99,
            }
        )

        feedback = await evaluator.evaluate_query(
            query="x",
            response="y",
            contexts=["z"],
        )

        # With extremely high thresholds, should fail
        assert feedback.passed is False


class TestTruLensChainWrapping:
    """Test chain wrapping functionality."""

    def test_wrap_chain_without_trulens(self):
        """Test that wrap_chain returns original chain when TruLens unavailable."""
        evaluator = TruLensIntegration()

        # Mock chain
        mock_chain = object()

        # Without TruLens, should return original
        wrapped = evaluator.wrap_chain(mock_chain)
        assert wrapped is mock_chain

    def test_app_id_preserved(self):
        """Test that app_id is preserved."""
        evaluator = TruLensIntegration(app_id="custom-app-id")
        assert evaluator.app_id == "custom-app-id"


class TestTruLensRecordModel:
    """Test TruLensRecord model."""

    def test_record_creation(self):
        """Test creating a TruLensRecord."""
        feedback = TruLensFeedback(
            groundedness=0.8,
            answer_relevance=0.9,
            context_relevance=0.85,
            overall_score=0.85,
            passed=True,
        )

        record = TruLensRecord(
            record_id="test_001",
            query="What is the weather?",
            response="It is sunny.",
            contexts=["Sunny weather today."],
            feedback=feedback,
        )

        assert record.record_id == "test_001"
        assert record.query == "What is the weather?"
        assert record.feedback.groundedness == 0.8

    def test_record_serialization(self):
        """Test that record can be serialized to dict."""
        feedback = TruLensFeedback(
            groundedness=0.8,
            answer_relevance=0.9,
            context_relevance=0.85,
            overall_score=0.85,
            passed=True,
        )

        record = TruLensRecord(
            record_id="test_001",
            query="test",
            response="response",
            contexts=["context"],
            feedback=feedback,
        )

        data = record.model_dump()

        assert isinstance(data, dict)
        assert data["record_id"] == "test_001"
        assert data["feedback"]["groundedness"] == 0.8
