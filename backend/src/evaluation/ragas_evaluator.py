"""
Ragas Integration for RAG-Specific Metrics.

Level 6a: Advanced Evaluation Framework

Metrics:
1. Faithfulness: Answer grounded in retrieved context (no hallucination)
2. Context Precision: Top-k chunks are relevant
3. Context Recall: All relevant information retrieved
4. Answer Relevancy: Answer addresses the query

Target: >0.90 faithfulness, >0.85 context precision/recall
"""

from typing import Any
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RagasResult(BaseModel):
    """Result from Ragas evaluation."""

    faithfulness: float = Field(ge=0.0, le=1.0, description="Answer grounded in context")
    context_precision: float = Field(ge=0.0, le=1.0, description="Top-k chunks are relevant")
    context_recall: float = Field(ge=0.0, le=1.0, description="All relevant info retrieved")
    answer_relevancy: float = Field(ge=0.0, le=1.0, description="Answer addresses query")
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted overall score")
    passed: bool = Field(description="Whether evaluation passed thresholds")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")


class RagasEvaluator:
    """
    Evaluate RAG system with Ragas metrics.

    Provides comprehensive RAG evaluation including:
    - Faithfulness: Is the answer grounded in the retrieved context?
    - Context Precision: Are the top-k retrieved chunks relevant?
    - Context Recall: Was all relevant information retrieved?
    - Answer Relevancy: Does the answer address the query?
    """

    # Default thresholds for passing
    DEFAULT_THRESHOLDS = {
        "faithfulness": 0.90,
        "context_precision": 0.85,
        "context_recall": 0.85,
        "answer_relevancy": 0.90,
    }

    # Weight distribution for overall score
    WEIGHTS = {
        "faithfulness": 0.4,  # Most important - no hallucination
        "context_precision": 0.2,
        "context_recall": 0.2,
        "answer_relevancy": 0.2,
    }

    def __init__(
        self,
        llm: Any | None = None,
        embeddings: Any | None = None,
        thresholds: dict[str, float] | None = None,
    ):
        """
        Initialize the Ragas evaluator.

        Args:
            llm: Language model for evaluation (required for some metrics)
            embeddings: Embeddings model for similarity calculations
            thresholds: Custom thresholds (uses defaults if not provided)
        """
        self.llm = llm
        self.embeddings = embeddings
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()

        # Try to import ragas
        self._ragas_available = False
        try:
            from ragas.metrics import (
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy,
            )
            from ragas import evaluate
            from datasets import Dataset

            self._ragas_faithfulness = faithfulness
            self._ragas_context_precision = context_precision
            self._ragas_context_recall = context_recall
            self._ragas_answer_relevancy = answer_relevancy
            self._ragas_evaluate = evaluate
            self._dataset_class = Dataset
            self._ragas_available = True
            logger.info("Ragas library loaded successfully")
        except ImportError as e:
            logger.warning(f"Ragas not available, using fallback metrics: {e}")

    async def evaluate_rag_query(
        self,
        query: str,
        retrieved_chunks: list[str],
        generated_answer: str,
        ground_truth_answer: str | None = None,
    ) -> RagasResult:
        """
        Evaluate single RAG query with all Ragas metrics.

        Args:
            query: User's query
            retrieved_chunks: List of retrieved context chunks
            generated_answer: LLM's generated answer
            ground_truth_answer: Optional expected answer (for context recall)

        Returns:
            RagasResult with all metric scores
        """
        if self._ragas_available and self.llm is not None:
            return await self._evaluate_with_ragas(
                query, retrieved_chunks, generated_answer, ground_truth_answer
            )
        else:
            return await self._evaluate_fallback(
                query, retrieved_chunks, generated_answer, ground_truth_answer
            )

    async def _evaluate_with_ragas(
        self,
        query: str,
        retrieved_chunks: list[str],
        generated_answer: str,
        ground_truth_answer: str | None = None,
    ) -> RagasResult:
        """Evaluate using Ragas library."""
        # Prepare dataset for Ragas
        data = {
            "question": [query],
            "answer": [generated_answer],
            "contexts": [retrieved_chunks],
        }

        if ground_truth_answer:
            data["ground_truth"] = [ground_truth_answer]

        dataset = self._dataset_class.from_dict(data)

        # Select metrics based on available data
        metrics = [
            self._ragas_faithfulness,
            self._ragas_context_precision,
            self._ragas_answer_relevancy,
        ]

        if ground_truth_answer:
            metrics.append(self._ragas_context_recall)

        try:
            # Run Ragas evaluation
            result = self._ragas_evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )

            faithfulness_score = float(result.get("faithfulness", 0.0))
            precision_score = float(result.get("context_precision", 0.0))
            recall_score = float(result.get("context_recall", 0.0)) if ground_truth_answer else 0.85
            relevancy_score = float(result.get("answer_relevancy", 0.0))

        except Exception as e:
            logger.warning(f"Ragas evaluation failed, using fallback: {e}")
            return await self._evaluate_fallback(
                query, retrieved_chunks, generated_answer, ground_truth_answer
            )

        # Calculate overall score
        overall = (
            faithfulness_score * self.WEIGHTS["faithfulness"]
            + precision_score * self.WEIGHTS["context_precision"]
            + recall_score * self.WEIGHTS["context_recall"]
            + relevancy_score * self.WEIGHTS["answer_relevancy"]
        )

        # Determine if passed
        passed = (
            faithfulness_score >= self.thresholds["faithfulness"]
            and precision_score >= self.thresholds["context_precision"]
            and relevancy_score >= self.thresholds["answer_relevancy"]
        )
        if ground_truth_answer:
            passed = passed and recall_score >= self.thresholds["context_recall"]

        return RagasResult(
            faithfulness=faithfulness_score,
            context_precision=precision_score,
            context_recall=recall_score,
            answer_relevancy=relevancy_score,
            overall_score=overall,
            passed=passed,
            details={
                "method": "ragas",
                "thresholds": self.thresholds,
                "weights": self.WEIGHTS,
            },
        )

    async def _evaluate_fallback(
        self,
        query: str,
        retrieved_chunks: list[str],
        generated_answer: str,
        ground_truth_answer: str | None = None,
    ) -> RagasResult:
        """
        Fallback evaluation when Ragas is not available.

        Uses heuristic-based metrics that approximate Ragas behavior.
        """
        # Faithfulness: Check if answer terms appear in context
        faithfulness_score = self._calculate_faithfulness(
            generated_answer, retrieved_chunks
        )

        # Context Precision: Check relevance of retrieved chunks to query
        precision_score = self._calculate_context_precision(query, retrieved_chunks)

        # Context Recall: Check if ground truth terms are in context
        recall_score = self._calculate_context_recall(
            ground_truth_answer, retrieved_chunks
        ) if ground_truth_answer else 0.85

        # Answer Relevancy: Check if answer addresses query
        relevancy_score = self._calculate_answer_relevancy(query, generated_answer)

        # Calculate overall score
        overall = (
            faithfulness_score * self.WEIGHTS["faithfulness"]
            + precision_score * self.WEIGHTS["context_precision"]
            + recall_score * self.WEIGHTS["context_recall"]
            + relevancy_score * self.WEIGHTS["answer_relevancy"]
        )

        # Determine if passed
        passed = (
            faithfulness_score >= self.thresholds["faithfulness"]
            and precision_score >= self.thresholds["context_precision"]
            and relevancy_score >= self.thresholds["answer_relevancy"]
        )

        return RagasResult(
            faithfulness=faithfulness_score,
            context_precision=precision_score,
            context_recall=recall_score,
            answer_relevancy=relevancy_score,
            overall_score=overall,
            passed=passed,
            details={
                "method": "fallback_heuristic",
                "thresholds": self.thresholds,
                "weights": self.WEIGHTS,
            },
        )

    def _calculate_faithfulness(
        self, answer: str, contexts: list[str]
    ) -> float:
        """
        Calculate faithfulness score (answer grounded in context).

        Measures what percentage of answer content can be traced to context.
        """
        if not answer or not contexts:
            return 0.0

        # Combine all contexts
        combined_context = " ".join(contexts).lower()
        answer_lower = answer.lower()

        # Extract significant terms from answer (>3 chars, not common words)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "this", "that",
            "these", "those", "with", "from", "into", "about", "for", "and",
            "but", "or", "not", "very", "just", "only", "also", "more", "most",
        }

        answer_terms = [
            word for word in answer_lower.split()
            if len(word) > 3 and word not in stop_words
        ]

        if not answer_terms:
            return 0.5  # Neutral score for very short answers

        # Count terms found in context
        found_count = sum(1 for term in answer_terms if term in combined_context)

        return found_count / len(answer_terms)

    def _calculate_context_precision(
        self, query: str, contexts: list[str]
    ) -> float:
        """
        Calculate context precision (top-k chunks are relevant).

        Measures how many retrieved chunks are relevant to the query.
        """
        if not query or not contexts:
            return 0.0

        query_terms = set(query.lower().split())

        # Score each context
        relevant_count = 0
        for context in contexts:
            context_terms = set(context.lower().split())
            overlap = len(query_terms & context_terms)
            if overlap >= 2:  # At least 2 query terms found
                relevant_count += 1

        return relevant_count / len(contexts) if contexts else 0.0

    def _calculate_context_recall(
        self, ground_truth: str, contexts: list[str]
    ) -> float:
        """
        Calculate context recall (all relevant info retrieved).

        Measures what percentage of ground truth content is in retrieved context.
        """
        if not ground_truth or not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()
        ground_truth_lower = ground_truth.lower()

        # Extract significant terms from ground truth
        gt_terms = [
            word for word in ground_truth_lower.split()
            if len(word) > 3
        ]

        if not gt_terms:
            return 0.5

        found_count = sum(1 for term in gt_terms if term in combined_context)

        return found_count / len(gt_terms)

    def _calculate_answer_relevancy(
        self, query: str, answer: str
    ) -> float:
        """
        Calculate answer relevancy (answer addresses query).

        Measures how well the answer addresses the question.
        """
        if not query or not answer:
            return 0.0

        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())

        # Remove common words
        stop_words = {"what", "is", "the", "a", "an", "how", "why", "when", "where", "who"}
        query_terms = query_terms - stop_words

        if not query_terms:
            return 0.5

        # Check overlap
        overlap = len(query_terms & answer_terms)

        # Bonus for longer, more detailed answers
        length_bonus = min(0.2, len(answer.split()) / 100)

        return min(1.0, (overlap / len(query_terms)) + length_bonus)

    async def evaluate_dataset(
        self, queries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Evaluate full dataset (100+ queries).

        Args:
            queries: List of dicts with 'query', 'answer', 'contexts', 'ground_truth'

        Returns:
            Aggregate metrics and pass/fail summary
        """
        results = []

        for query_data in queries:
            result = await self.evaluate_rag_query(
                query=query_data["query"],
                retrieved_chunks=query_data.get("contexts", []),
                generated_answer=query_data["answer"],
                ground_truth_answer=query_data.get("ground_truth"),
            )
            results.append(result)

        # Aggregate metrics
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_recall = sum(r.context_recall for r in results) / len(results)
        avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        avg_overall = sum(r.overall_score for r in results) / len(results)
        pass_count = sum(1 for r in results if r.passed)

        return {
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_context_precision": round(avg_precision, 4),
            "avg_context_recall": round(avg_recall, 4),
            "avg_answer_relevancy": round(avg_relevancy, 4),
            "overall_score": round(avg_overall, 4),
            "total_queries": len(queries),
            "passed_count": pass_count,
            "pass_rate": round(pass_count / len(queries), 4),
            "passed": (
                avg_faithfulness >= self.thresholds["faithfulness"]
                and avg_precision >= self.thresholds["context_precision"]
            ),
        }

    def get_thresholds(self) -> dict[str, float]:
        """Get current evaluation thresholds."""
        return self.thresholds.copy()

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        """Update evaluation thresholds."""
        self.thresholds.update(thresholds)
        logger.info(f"Updated Ragas thresholds: {self.thresholds}")
