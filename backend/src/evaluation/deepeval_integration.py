"""
DeepEval Integration for LLM Unit Testing.

Level 6a: Advanced Evaluation Framework

Metrics:
- Answer Relevancy: Does the answer address the query?
- Faithfulness: Is the answer grounded in context?
- Contextual Precision: Are retrieved contexts relevant?
- Contextual Recall: Was all needed info retrieved?
- Synthetic test data generation

Target: >0.85 on all metrics
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DeepEvalResult(BaseModel):
    """Result from DeepEval evaluation."""

    answer_relevancy: float = Field(ge=0.0, le=1.0, description="Answer addresses query")
    faithfulness: float = Field(ge=0.0, le=1.0, description="Answer grounded in context")
    contextual_precision: float = Field(ge=0.0, le=1.0, description="Contexts are relevant")
    contextual_recall: float = Field(ge=0.0, le=1.0, description="All info retrieved")
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted overall score")
    passed: bool = Field(description="Whether evaluation passed thresholds")
    test_case_id: str | None = Field(default=None, description="Test case identifier")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")


class DeepEvalIntegration:
    """
    DeepEval LLM unit testing framework.

    Provides comprehensive LLM evaluation including:
    - Answer Relevancy
    - Faithfulness
    - Contextual Precision
    - Contextual Recall
    - Synthetic test data generation
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "answer_relevancy": 0.85,
        "faithfulness": 0.85,
        "contextual_precision": 0.85,
        "contextual_recall": 0.85,
    }

    # Weights for overall score
    WEIGHTS = {
        "answer_relevancy": 0.25,
        "faithfulness": 0.35,  # Highest weight - no hallucination
        "contextual_precision": 0.20,
        "contextual_recall": 0.20,
    }

    def __init__(
        self,
        llm: Any | None = None,
        thresholds: dict[str, float] | None = None,
    ):
        """
        Initialize DeepEval integration.

        Args:
            llm: Language model for evaluation
            thresholds: Custom thresholds (uses defaults if not provided)
        """
        self.llm = llm
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()

        # Try to import DeepEval
        self._deepeval_available = False
        try:
            from deepeval import evaluate
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
            )
            from deepeval.test_case import LLMTestCase

            self._AnswerRelevancyMetric = AnswerRelevancyMetric
            self._FaithfulnessMetric = FaithfulnessMetric
            self._ContextualPrecisionMetric = ContextualPrecisionMetric
            self._ContextualRecallMetric = ContextualRecallMetric
            self._LLMTestCase = LLMTestCase
            self._evaluate = evaluate
            self._deepeval_available = True
            logger.info("DeepEval library loaded successfully")
        except ImportError as e:
            logger.warning(f"DeepEval not available, using fallback: {e}")

    async def evaluate_query(
        self,
        query: str,
        answer: str,
        retrieved_contexts: list[str],
        expected_output: str | None = None,
        test_case_id: str | None = None,
    ) -> DeepEvalResult:
        """
        Evaluate single query with DeepEval metrics.

        Args:
            query: User's query
            answer: Generated answer
            retrieved_contexts: List of retrieved context strings
            expected_output: Optional expected answer
            test_case_id: Optional test case identifier

        Returns:
            DeepEvalResult with all metric scores
        """
        if self._deepeval_available and self.llm is not None:
            return await self._evaluate_with_deepeval(
                query, answer, retrieved_contexts, expected_output, test_case_id
            )
        else:
            return await self._evaluate_fallback(
                query, answer, retrieved_contexts, expected_output, test_case_id
            )

    async def _evaluate_with_deepeval(
        self,
        query: str,
        answer: str,
        retrieved_contexts: list[str],
        expected_output: str | None = None,
        test_case_id: str | None = None,
    ) -> DeepEvalResult:
        """Evaluate using DeepEval library."""
        try:
            # Create test case
            test_case = self._LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=retrieved_contexts,
                expected_output=expected_output,
            )

            # Create metrics
            metrics = [
                self._AnswerRelevancyMetric(model=self.llm),
                self._FaithfulnessMetric(model=self.llm),
                self._ContextualPrecisionMetric(model=self.llm),
                self._ContextualRecallMetric(model=self.llm),
            ]

            # Run evaluation
            result = self._evaluate([test_case], metrics)

            # Extract scores
            answer_relevancy = result.test_results[0].metrics_data[0].score
            faithfulness = result.test_results[0].metrics_data[1].score
            contextual_precision = result.test_results[0].metrics_data[2].score
            contextual_recall = result.test_results[0].metrics_data[3].score

        except Exception as e:
            logger.warning(f"DeepEval evaluation failed: {e}")
            return await self._evaluate_fallback(
                query, answer, retrieved_contexts, expected_output, test_case_id
            )

        # Calculate overall score
        overall = (
            answer_relevancy * self.WEIGHTS["answer_relevancy"]
            + faithfulness * self.WEIGHTS["faithfulness"]
            + contextual_precision * self.WEIGHTS["contextual_precision"]
            + contextual_recall * self.WEIGHTS["contextual_recall"]
        )

        # Determine if passed
        passed = (
            answer_relevancy >= self.thresholds["answer_relevancy"]
            and faithfulness >= self.thresholds["faithfulness"]
            and contextual_precision >= self.thresholds["contextual_precision"]
            and contextual_recall >= self.thresholds["contextual_recall"]
        )

        return DeepEvalResult(
            answer_relevancy=answer_relevancy,
            faithfulness=faithfulness,
            contextual_precision=contextual_precision,
            contextual_recall=contextual_recall,
            overall_score=overall,
            passed=passed,
            test_case_id=test_case_id,
            details={
                "method": "deepeval",
                "thresholds": self.thresholds,
            },
        )

    async def _evaluate_fallback(
        self,
        query: str,
        answer: str,
        retrieved_contexts: list[str],
        expected_output: str | None = None,
        test_case_id: str | None = None,
    ) -> DeepEvalResult:
        """Fallback evaluation when DeepEval is not available."""
        # Calculate metrics using heuristics
        answer_relevancy = self._calculate_answer_relevancy(query, answer)
        faithfulness = self._calculate_faithfulness(answer, retrieved_contexts)
        contextual_precision = self._calculate_contextual_precision(
            query, retrieved_contexts
        )
        contextual_recall = self._calculate_contextual_recall(
            expected_output, retrieved_contexts
        ) if expected_output else 0.85

        # Calculate overall score
        overall = (
            answer_relevancy * self.WEIGHTS["answer_relevancy"]
            + faithfulness * self.WEIGHTS["faithfulness"]
            + contextual_precision * self.WEIGHTS["contextual_precision"]
            + contextual_recall * self.WEIGHTS["contextual_recall"]
        )

        # Determine if passed
        passed = (
            answer_relevancy >= self.thresholds["answer_relevancy"]
            and faithfulness >= self.thresholds["faithfulness"]
            and contextual_precision >= self.thresholds["contextual_precision"]
        )

        return DeepEvalResult(
            answer_relevancy=answer_relevancy,
            faithfulness=faithfulness,
            contextual_precision=contextual_precision,
            contextual_recall=contextual_recall,
            overall_score=overall,
            passed=passed,
            test_case_id=test_case_id,
            details={
                "method": "fallback_heuristic",
                "thresholds": self.thresholds,
            },
        )

    def _calculate_answer_relevancy(self, query: str, answer: str) -> float:
        """Calculate answer relevancy score."""
        if not query or not answer:
            return 0.0

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Remove stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
            "when", "where", "who", "which", "this", "that", "these", "those",
        }
        query_words = query_words - stop_words

        if not query_words:
            return 0.5

        overlap = len(query_words & answer_words)
        base_score = overlap / len(query_words)

        # Bonus for longer answers (more detailed)
        length_bonus = min(0.2, len(answer.split()) / 100)

        return min(1.0, base_score + length_bonus)

    def _calculate_faithfulness(
        self, answer: str, contexts: list[str]
    ) -> float:
        """Calculate faithfulness score."""
        if not answer or not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()
        answer_lower = answer.lower()

        # Extract significant terms
        answer_terms = [w for w in answer_lower.split() if len(w) > 4]

        if not answer_terms:
            return 0.5

        found = sum(1 for term in answer_terms if term in combined_context)
        return found / len(answer_terms)

    def _calculate_contextual_precision(
        self, query: str, contexts: list[str]
    ) -> float:
        """Calculate contextual precision score."""
        if not query or not contexts:
            return 0.0

        query_terms = set(query.lower().split())

        relevant_count = 0
        for context in contexts:
            context_terms = set(context.lower().split())
            if len(query_terms & context_terms) >= 2:
                relevant_count += 1

        return relevant_count / len(contexts) if contexts else 0.0

    def _calculate_contextual_recall(
        self, expected: str, contexts: list[str]
    ) -> float:
        """Calculate contextual recall score."""
        if not expected or not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()
        expected_terms = [w for w in expected.lower().split() if len(w) > 4]

        if not expected_terms:
            return 0.5

        found = sum(1 for term in expected_terms if term in combined_context)
        return found / len(expected_terms)

    async def run_test_suite(
        self, test_cases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Run evaluation on a test suite.

        Args:
            test_cases: List of test case dicts with query, answer, contexts, expected

        Returns:
            Aggregate results and pass/fail summary
        """
        results = []

        for i, tc in enumerate(test_cases):
            result = await self.evaluate_query(
                query=tc["query"],
                answer=tc["answer"],
                retrieved_contexts=tc.get("contexts", []),
                expected_output=tc.get("expected"),
                test_case_id=tc.get("id", f"test_{i}"),
            )
            results.append(result)

        # Aggregate metrics
        avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_precision = sum(r.contextual_precision for r in results) / len(results)
        avg_recall = sum(r.contextual_recall for r in results) / len(results)
        avg_overall = sum(r.overall_score for r in results) / len(results)
        pass_count = sum(1 for r in results if r.passed)

        return {
            "avg_answer_relevancy": round(avg_relevancy, 4),
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_contextual_precision": round(avg_precision, 4),
            "avg_contextual_recall": round(avg_recall, 4),
            "overall_score": round(avg_overall, 4),
            "total_tests": len(test_cases),
            "passed_count": pass_count,
            "pass_rate": round(pass_count / len(test_cases), 4),
            "results": [r.model_dump() for r in results],
        }

    def generate_synthetic_test_cases(
        self, domain: str = "weather", count: int = 10
    ) -> list[dict[str, Any]]:
        """
        Generate synthetic test cases for a domain.

        Args:
            domain: Domain for test cases (e.g., "weather", "hurricane")
            count: Number of test cases to generate

        Returns:
            List of synthetic test case dicts
        """
        # Weather domain test templates
        weather_templates = [
            {
                "query": "What is the current temperature in {city}?",
                "context_template": "The current temperature in {city} is {temp}°F with {conditions}.",
                "cities": ["Tampa", "Miami", "Orlando", "Jacksonville", "Naples"],
                "temps": [75, 80, 85, 90, 95],
                "conditions": ["sunny skies", "partly cloudy", "cloudy", "light rain"],
            },
            {
                "query": "Is there a hurricane warning for {region}?",
                "context_template": "Hurricane warning for {region}: {status}. Wind speeds {wind} mph.",
                "regions": ["Florida Gulf Coast", "Atlantic Coast", "Tampa Bay"],
                "statuses": ["Active", "Watch", "Advisory"],
                "winds": [75, 100, 130, 157],
            },
            {
                "query": "Should I evacuate from {zone}?",
                "context_template": "Evacuation {order} for zone {zone}. Storm surge expected: {surge} feet.",
                "zones": ["A", "B", "C"],
                "orders": ["mandatory", "voluntary", "advised"],
                "surges": [3, 6, 9, 12],
            },
        ]

        import random

        test_cases = []
        for i in range(count):
            template = random.choice(weather_templates)

            # Generate random values
            if "cities" in template:
                city = random.choice(template["cities"])
                temp = random.choice(template["temps"])
                cond = random.choice(template["conditions"])

                query = template["query"].format(city=city)
                context = template["context_template"].format(
                    city=city, temp=temp, conditions=cond
                )
                expected = f"The temperature in {city} is {temp}°F."

            elif "regions" in template:
                region = random.choice(template["regions"])
                status = random.choice(template["statuses"])
                wind = random.choice(template["winds"])

                query = template["query"].format(region=region)
                context = template["context_template"].format(
                    region=region, status=status, wind=wind
                )
                expected = f"Hurricane warning is {status.lower()} for {region}."

            else:
                zone = random.choice(template["zones"])
                order = random.choice(template["orders"])
                surge = random.choice(template["surges"])

                query = template["query"].format(zone=zone)
                context = template["context_template"].format(
                    zone=zone, order=order, surge=surge
                )
                expected = f"Evacuation is {order} for zone {zone}."

            test_cases.append({
                "id": f"synthetic_{i+1}",
                "query": query,
                "contexts": [context],
                "answer": expected,  # Use expected as answer for testing
                "expected": expected,
            })

        return test_cases
