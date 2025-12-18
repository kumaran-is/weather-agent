"""
TruLens Integration for Real-Time RAG Evaluation.

Level 6b: Self-Improvement Platform

Feedback Functions:
1. Groundedness: Answer grounded in retrieved context
2. Answer Relevance: Answer addresses query
3. Context Relevance: Retrieved context relevant to query

Target: >0.85 on all feedback functions, real-time monitoring
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TruLensFeedback(BaseModel):
    """Feedback result from TruLens evaluation."""

    groundedness: float = Field(ge=0.0, le=1.0, description="Answer grounded in context")
    answer_relevance: float = Field(ge=0.0, le=1.0, description="Answer addresses query")
    context_relevance: float = Field(ge=0.0, le=1.0, description="Context relevant to query")
    overall_score: float = Field(ge=0.0, le=1.0, description="Average of all scores")
    passed: bool = Field(description="Whether all thresholds are met")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")


class TruLensRecord(BaseModel):
    """Record from TruLens evaluation."""

    record_id: str
    query: str
    response: str
    contexts: list[str] = Field(default_factory=list)
    feedback: TruLensFeedback
    timestamp: str | None = None


class TruLensIntegration:
    """
    Real-time evaluation with TruLens.

    Provides feedback functions for:
    - Groundedness: Is the answer grounded in the retrieved context?
    - Answer Relevance: Does the answer address the query?
    - Context Relevance: Is the retrieved context relevant to the query?

    Falls back to heuristic-based feedback when TruLens is not available.
    """

    # Default thresholds for passing
    DEFAULT_THRESHOLDS = {
        "groundedness": 0.85,
        "answer_relevance": 0.85,
        "context_relevance": 0.85,
    }

    def __init__(
        self,
        llm: Any | None = None,
        app_id: str = "weather-ai-agent",
        thresholds: dict[str, float] | None = None,
    ):
        """
        Initialize TruLens integration.

        Args:
            llm: Language model for feedback functions
            app_id: Application ID for TruLens tracking
            thresholds: Custom thresholds (uses defaults if not provided)
        """
        self.llm = llm
        self.app_id = app_id
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()

        # Records storage (in-memory for fallback mode)
        self.records: list[TruLensRecord] = []

        # Try to import TruLens
        self._trulens_available = False
        self._tru = None
        self._tru_chain = None

        try:
            from trulens.apps.langchain import TruChain
            from trulens.core import Feedback, TruSession
            from trulens.core.guardrails.base import context_filter
            from trulens.providers.openai import OpenAI as TruOpenAI

            self._TruSession = TruSession
            self._TruChain = TruChain
            self._TruOpenAI = TruOpenAI
            self._Feedback = Feedback
            self._context_filter = context_filter
            self._trulens_available = True
            logger.info("TruLens library loaded successfully")

            # Initialize TruLens session
            self._tru = self._TruSession()

        except ImportError as e:
            logger.warning(f"TruLens not available, using fallback heuristics: {e}")

    async def evaluate_query(
        self,
        query: str,
        response: str,
        contexts: list[str],
        record_id: str | None = None,
    ) -> TruLensFeedback:
        """
        Evaluate single query with TruLens feedback functions.

        Args:
            query: User's query
            response: Generated response
            contexts: Retrieved context chunks
            record_id: Optional record identifier

        Returns:
            TruLensFeedback with all feedback scores
        """
        if self._trulens_available and self.llm is not None:
            return await self._evaluate_with_trulens(query, response, contexts, record_id)
        else:
            return await self._evaluate_fallback(query, response, contexts, record_id)

    async def _evaluate_with_trulens(
        self,
        query: str,
        response: str,
        contexts: list[str],
        record_id: str | None = None,
    ) -> TruLensFeedback:
        """Evaluate using TruLens library."""
        try:
            # Create provider for feedback functions
            provider = self._TruOpenAI(model_engine=self.llm if isinstance(self.llm, str) else "gpt-4")

            # Define feedback functions
            f_groundedness = self._Feedback(provider.groundedness_measure_with_cot_reasons)
            f_answer_relevance = self._Feedback(provider.relevance)
            f_context_relevance = self._Feedback(provider.context_relevance)

            # Compute feedback scores
            combined_context = "\n".join(contexts)

            groundedness_score = f_groundedness(
                query=query,
                context=combined_context,
                response=response,
            )
            answer_relevance_score = f_answer_relevance(
                prompt=query,
                response=response,
            )
            context_relevance_score = f_context_relevance(
                prompt=query,
                context=combined_context,
            )

        except Exception as e:
            logger.warning(f"TruLens evaluation failed, using fallback: {e}")
            return await self._evaluate_fallback(query, response, contexts, record_id)

        # Calculate overall score
        overall = (
            groundedness_score + answer_relevance_score + context_relevance_score
        ) / 3

        # Determine if passed
        passed = (
            groundedness_score >= self.thresholds["groundedness"]
            and answer_relevance_score >= self.thresholds["answer_relevance"]
            and context_relevance_score >= self.thresholds["context_relevance"]
        )

        feedback = TruLensFeedback(
            groundedness=groundedness_score,
            answer_relevance=answer_relevance_score,
            context_relevance=context_relevance_score,
            overall_score=overall,
            passed=passed,
            details={
                "method": "trulens",
                "thresholds": self.thresholds,
                "app_id": self.app_id,
            },
        )

        # Store record
        self._store_record(query, response, contexts, feedback, record_id)

        return feedback

    async def _evaluate_fallback(
        self,
        query: str,
        response: str,
        contexts: list[str],
        record_id: str | None = None,
    ) -> TruLensFeedback:
        """Fallback evaluation when TruLens is not available."""
        # Calculate heuristic feedback scores
        groundedness_score = self._calculate_groundedness(response, contexts)
        answer_relevance_score = self._calculate_answer_relevance(query, response)
        context_relevance_score = self._calculate_context_relevance(query, contexts)

        # Calculate overall score
        overall = (
            groundedness_score + answer_relevance_score + context_relevance_score
        ) / 3

        # Determine if passed
        passed = (
            groundedness_score >= self.thresholds["groundedness"]
            and answer_relevance_score >= self.thresholds["answer_relevance"]
            and context_relevance_score >= self.thresholds["context_relevance"]
        )

        feedback = TruLensFeedback(
            groundedness=groundedness_score,
            answer_relevance=answer_relevance_score,
            context_relevance=context_relevance_score,
            overall_score=overall,
            passed=passed,
            details={
                "method": "fallback_heuristic",
                "thresholds": self.thresholds,
                "app_id": self.app_id,
            },
        )

        # Store record
        self._store_record(query, response, contexts, feedback, record_id)

        return feedback

    def _calculate_groundedness(
        self, response: str, contexts: list[str]
    ) -> float:
        """
        Calculate groundedness score (response grounded in context).

        Measures what percentage of response content can be traced to context.
        """
        if not response or not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()
        response_lower = response.lower()

        # Extract significant terms from response
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "this", "that",
            "these", "those", "with", "from", "into", "about", "for", "and",
            "but", "or", "not", "very", "just", "only", "also", "more", "most",
        }

        response_terms = [
            word for word in response_lower.split()
            if len(word) > 3 and word not in stop_words
        ]

        if not response_terms:
            return 0.5

        # Count terms found in context
        found_count = sum(1 for term in response_terms if term in combined_context)

        return found_count / len(response_terms)

    def _calculate_answer_relevance(
        self, query: str, response: str
    ) -> float:
        """
        Calculate answer relevance score (response addresses query).

        Measures how well the response addresses the question.
        """
        if not query or not response:
            return 0.0

        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())

        # Remove common words
        stop_words = {"what", "is", "the", "a", "an", "how", "why", "when", "where", "who"}
        query_terms = query_terms - stop_words

        if not query_terms:
            return 0.5

        # Check overlap
        overlap = len(query_terms & response_terms)
        base_score = overlap / len(query_terms)

        # Bonus for longer, more detailed responses
        length_bonus = min(0.2, len(response.split()) / 100)

        return min(1.0, base_score + length_bonus)

    def _calculate_context_relevance(
        self, query: str, contexts: list[str]
    ) -> float:
        """
        Calculate context relevance score (context relevant to query).

        Measures how relevant the retrieved contexts are to the query.
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

    def _store_record(
        self,
        query: str,
        response: str,
        contexts: list[str],
        feedback: TruLensFeedback,
        record_id: str | None = None,
    ) -> None:
        """Store evaluation record."""
        import uuid
        from datetime import datetime

        record = TruLensRecord(
            record_id=record_id or str(uuid.uuid4()),
            query=query,
            response=response,
            contexts=contexts,
            feedback=feedback,
            timestamp=datetime.now().isoformat(),
        )
        self.records.append(record)

        # Keep only last 1000 records
        if len(self.records) > 1000:
            self.records = self.records[-1000:]

    def get_feedback_summary(self) -> dict[str, Any]:
        """Get summary of all feedback scores."""
        if not self.records:
            return {
                "total_records": 0,
                "avg_groundedness": 0.0,
                "avg_answer_relevance": 0.0,
                "avg_context_relevance": 0.0,
                "overall_score": 0.0,
                "pass_rate": 0.0,
            }

        avg_groundedness = sum(
            r.feedback.groundedness for r in self.records
        ) / len(self.records)
        avg_answer_relevance = sum(
            r.feedback.answer_relevance for r in self.records
        ) / len(self.records)
        avg_context_relevance = sum(
            r.feedback.context_relevance for r in self.records
        ) / len(self.records)
        overall = (avg_groundedness + avg_answer_relevance + avg_context_relevance) / 3
        pass_count = sum(1 for r in self.records if r.feedback.passed)

        return {
            "total_records": len(self.records),
            "avg_groundedness": round(avg_groundedness, 4),
            "avg_answer_relevance": round(avg_answer_relevance, 4),
            "avg_context_relevance": round(avg_context_relevance, 4),
            "overall_score": round(overall, 4),
            "pass_rate": round(pass_count / len(self.records), 4),
            "passed": (
                avg_groundedness >= self.thresholds["groundedness"]
                and avg_answer_relevance >= self.thresholds["answer_relevance"]
            ),
        }

    def get_records(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get evaluation records."""
        records = self.records[offset : offset + limit]
        return [r.model_dump() for r in records]

    def clear_records(self) -> None:
        """Clear all stored records."""
        self.records = []
        logger.info("TruLens records cleared")

    def wrap_chain(self, chain: Any) -> Any:
        """
        Wrap LangChain chain with TruLens monitoring.

        Returns wrapped chain or original chain if TruLens unavailable.
        """
        if not self._trulens_available:
            logger.warning("TruLens not available, returning unwrapped chain")
            return chain

        try:
            # Create TruChain wrapper
            provider = self._TruOpenAI(model_engine="gpt-4")

            f_groundedness = self._Feedback(provider.groundedness_measure_with_cot_reasons)
            f_relevance = self._Feedback(provider.relevance)

            tru_chain = self._TruChain(
                chain,
                app_name=self.app_id,
                feedbacks=[f_groundedness, f_relevance],
            )

            self._tru_chain = tru_chain
            logger.info(f"Chain wrapped with TruLens monitoring: {self.app_id}")
            return tru_chain

        except Exception as e:
            logger.warning(f"Failed to wrap chain with TruLens: {e}")
            return chain

    def get_thresholds(self) -> dict[str, float]:
        """Get current evaluation thresholds."""
        return self.thresholds.copy()

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        """Update evaluation thresholds."""
        self.thresholds.update(thresholds)
        logger.info(f"Updated TruLens thresholds: {self.thresholds}")

    def launch_dashboard(self, port: int = 8501) -> None:
        """
        Launch TruLens dashboard for real-time monitoring.

        Only works when TruLens is available.
        """
        if not self._trulens_available or self._tru is None:
            logger.warning("TruLens not available, cannot launch dashboard")
            return

        try:
            self._tru.run_dashboard(port=port)
            logger.info(f"TruLens dashboard launched on port {port}")
        except Exception as e:
            logger.error(f"Failed to launch TruLens dashboard: {e}")
