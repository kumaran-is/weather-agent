"""
Auto-Prompt Engineering System.

Level 6b: Self-Improvement Platform

Process:
1. Generate prompt variations (5-10 variations)
2. A/B test variations on sample queries
3. Select best-performing prompt
4. Deploy to production
5. Continuous monitoring and re-optimization

Target: 20-40% improvement in prompt effectiveness
"""

from typing import Any
import logging
import time
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class VariationResult(BaseModel):
    """Result from evaluating a single prompt variation."""

    variation_id: str
    prompt_text: str
    scores: list[float] = Field(default_factory=list)
    avg_score: float = 0.0
    latency_avg_ms: float = 0.0
    queries_tested: int = 0
    technique: str = "original"


class OptimizationResult(BaseModel):
    """Result from prompt optimization cycle."""

    original_prompt: str
    best_prompt: str
    best_variation_id: str
    improvement_pct: float
    original_score: float
    best_score: float
    variations_tested: int
    queries_per_variation: int
    optimization_time_ms: float
    all_scores: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class OptimizationStatus(str, Enum):
    """Status of optimization process."""

    PENDING = "pending"
    GENERATING = "generating_variations"
    TESTING = "testing_variations"
    COMPLETED = "completed"
    FAILED = "failed"


class AutoPromptOptimizer:
    """
    Automatic prompt optimization through A/B testing.

    Provides:
    - Generate prompt variations using multiple techniques
    - A/B test variations on sample queries
    - Select and deploy best-performing prompt
    - Track optimization history
    """

    def __init__(
        self,
        llm: Any | None = None,
        evaluator: Any | None = None,
    ):
        """
        Initialize the auto-prompt optimizer.

        Args:
            llm: Language model for generating variations and responses
            evaluator: Evaluator for scoring responses (e.g., DeepEvalIntegration)
        """
        self.llm = llm
        self.evaluator = evaluator

        # Optimization history
        self.optimization_history: list[OptimizationResult] = []

        # Current status
        self.status = OptimizationStatus.PENDING

        logger.info("AutoPromptOptimizer initialized")

    async def generate_variations(
        self,
        base_prompt: str,
        num_variations: int = 5,
        techniques: list[str] | None = None,
    ) -> list[VariationResult]:
        """
        Generate prompt variations using LLM or heuristic techniques.

        Args:
            base_prompt: Original prompt to generate variations from
            num_variations: Number of variations to generate
            techniques: Optional list of techniques to use

        Returns:
            List of VariationResult with generated prompts
        """
        self.status = OptimizationStatus.GENERATING

        techniques = techniques or [
            "formal_tone",
            "concise",
            "structured",
            "examples_added",
            "question_focused",
        ]

        variations = [
            VariationResult(
                variation_id="original",
                prompt_text=base_prompt,
                technique="original",
            )
        ]

        for i, technique in enumerate(techniques[:num_variations]):
            variation_text = await self._generate_single_variation(
                base_prompt, technique
            )
            variations.append(
                VariationResult(
                    variation_id=f"variation_{i + 1}",
                    prompt_text=variation_text,
                    technique=technique,
                )
            )

        logger.info(f"Generated {len(variations) - 1} prompt variations")
        return variations

    async def _generate_single_variation(
        self, base_prompt: str, technique: str
    ) -> str:
        """Generate a single prompt variation using specified technique."""
        if self.llm is not None:
            return await self._generate_with_llm(base_prompt, technique)
        else:
            return self._generate_heuristic(base_prompt, technique)

    async def _generate_with_llm(
        self, base_prompt: str, technique: str
    ) -> str:
        """Generate variation using LLM."""
        try:
            prompt = f"""You are a prompt engineering expert. Generate a variation of the following prompt using the '{technique}' technique.

Original prompt:
{base_prompt}

Generate a variation that:
- Achieves the same goal
- Uses the '{technique}' approach
- Is clear and effective

Variation:"""

            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(prompt)
                return response.content.strip() if hasattr(response, "content") else str(response).strip()
            elif callable(self.llm):
                response = self.llm(prompt)
                return str(response).strip()
            else:
                return self._generate_heuristic(base_prompt, technique)

        except Exception as e:
            logger.warning(f"LLM variation generation failed: {e}")
            return self._generate_heuristic(base_prompt, technique)

    def _generate_heuristic(self, base_prompt: str, technique: str) -> str:
        """Generate variation using heuristic transformations."""
        transformations = {
            "formal_tone": lambda p: f"Please {p.lower().replace('you are', 'kindly act as').replace('!', '.')}",
            "concise": lambda p: " ".join(p.split()[:len(p.split()) // 2 + 10]) + "...",
            "structured": lambda p: f"# Instructions\n{p}\n\n# Format\nProvide a clear, structured response.",
            "examples_added": lambda p: f"{p}\n\nExample: If asked about weather, provide temperature, conditions, and forecast.",
            "question_focused": lambda p: f"When receiving a query, consider:\n1. What is being asked?\n2. What information is needed?\n\n{p}",
            "step_by_step": lambda p: f"{p}\n\nApproach step by step:\n1. Understand the query\n2. Gather relevant information\n3. Formulate response",
            "context_aware": lambda p: f"Consider the context and user intent.\n\n{p}",
            "verbose": lambda p: f"Detailed instructions:\n{p}\n\nPlease ensure your response is comprehensive and addresses all aspects of the query.",
        }

        transform = transformations.get(technique, lambda p: p)
        return transform(base_prompt)

    async def ab_test_variations(
        self,
        variations: list[VariationResult],
        test_queries: list[str],
    ) -> dict[str, float]:
        """
        A/B test prompt variations on sample queries.

        Args:
            variations: List of prompt variations to test
            test_queries: Sample queries to test with

        Returns:
            Dictionary of variation_id -> average score
        """
        self.status = OptimizationStatus.TESTING
        scores: dict[str, list[float]] = {v.variation_id: [] for v in variations}
        latencies: dict[str, list[float]] = {v.variation_id: [] for v in variations}

        for query in test_queries:
            for variation in variations:
                start_time = time.time()

                try:
                    # Generate response
                    response = await self._get_response(variation.prompt_text, query)

                    # Evaluate response
                    score = await self._evaluate_response(query, response)

                    latency_ms = (time.time() - start_time) * 1000
                    scores[variation.variation_id].append(score)
                    latencies[variation.variation_id].append(latency_ms)
                    variation.scores.append(score)

                except Exception as e:
                    logger.warning(f"Test failed for {variation.variation_id}: {e}")
                    scores[variation.variation_id].append(0.0)

        # Calculate averages
        avg_scores = {}
        for var_id, var_scores in scores.items():
            avg_score = sum(var_scores) / len(var_scores) if var_scores else 0.0
            avg_scores[var_id] = avg_score

            # Update variation result
            for v in variations:
                if v.variation_id == var_id:
                    v.avg_score = avg_score
                    v.queries_tested = len(var_scores)
                    if latencies[var_id]:
                        v.latency_avg_ms = sum(latencies[var_id]) / len(latencies[var_id])

        logger.info(f"A/B test complete: {len(test_queries)} queries × {len(variations)} variations")
        return avg_scores

    async def _get_response(self, prompt: str, query: str) -> str:
        """Get response using the prompt and query."""
        full_prompt = f"{prompt}\n\nQuery: {query}"

        if self.llm is not None:
            try:
                if hasattr(self.llm, "ainvoke"):
                    response = await self.llm.ainvoke(full_prompt)
                    return response.content if hasattr(response, "content") else str(response)
                elif callable(self.llm):
                    response = self.llm(full_prompt)
                    return str(response)
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")

        # Fallback: simulate response
        return self._simulate_response(query)

    def _simulate_response(self, query: str) -> str:
        """Simulate a response for testing without LLM."""
        # Generate plausible response based on query
        if "weather" in query.lower():
            return "The current weather shows sunny conditions with a temperature of 85°F. Humidity is at 65%."
        elif "hurricane" in query.lower():
            return "The hurricane is currently Category 4 with winds of 130 mph. Expected landfall is Wednesday evening."
        elif "evacuation" in query.lower():
            return "Evacuation is recommended for Zone A residents. Shelter locations are available at local schools."
        else:
            return f"Based on your query about '{query[:30]}...', here is the relevant information."

    async def _evaluate_response(self, query: str, response: str) -> float:
        """Evaluate response quality."""
        if self.evaluator is not None:
            try:
                if hasattr(self.evaluator, "evaluate_query"):
                    result = await self.evaluator.evaluate_query(
                        query=query,
                        answer=response,
                        retrieved_contexts=[],
                    )
                    return result.overall_score if hasattr(result, "overall_score") else 0.5
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        # Fallback: heuristic evaluation
        return self._heuristic_evaluation(query, response)

    def _heuristic_evaluation(self, query: str, response: str) -> float:
        """Heuristic-based response evaluation."""
        if not response:
            return 0.0

        score = 0.0

        # Length check (reasonable length gets higher score)
        word_count = len(response.split())
        if 10 <= word_count <= 200:
            score += 0.3
        elif word_count > 200:
            score += 0.2
        elif word_count > 5:
            score += 0.1

        # Query term overlap
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        overlap = len(query_terms & response_terms) / max(len(query_terms), 1)
        score += overlap * 0.4

        # Contains specific data (numbers, units)
        if any(char.isdigit() for char in response):
            score += 0.15

        # Contains weather-related terms
        weather_terms = ["temperature", "weather", "forecast", "hurricane", "wind", "rain"]
        if any(term in response.lower() for term in weather_terms):
            score += 0.15

        return min(1.0, score)

    async def optimize_prompt(
        self,
        base_prompt: str,
        test_queries: list[str],
        num_variations: int = 5,
    ) -> OptimizationResult:
        """
        Full optimization cycle.

        Args:
            base_prompt: Original prompt to optimize
            test_queries: Sample queries for testing
            num_variations: Number of variations to generate

        Returns:
            OptimizationResult with best prompt and metrics
        """
        start_time = time.time()

        try:
            # Step 1: Generate variations
            variations = await self.generate_variations(
                base_prompt, num_variations=num_variations
            )

            # Step 2: A/B test variations
            scores = await self.ab_test_variations(variations, test_queries)

            # Step 3: Find best variation
            best_var_id = max(scores, key=scores.get)
            best_variation = next(
                (v for v in variations if v.variation_id == best_var_id), variations[0]
            )

            original_score = scores.get("original", 0.0)
            best_score = scores.get(best_var_id, 0.0)
            improvement = (
                ((best_score - original_score) / max(original_score, 0.01)) * 100
                if original_score > 0
                else 0.0
            )

            self.status = OptimizationStatus.COMPLETED

            result = OptimizationResult(
                original_prompt=base_prompt,
                best_prompt=best_variation.prompt_text,
                best_variation_id=best_var_id,
                improvement_pct=round(improvement, 2),
                original_score=round(original_score, 4),
                best_score=round(best_score, 4),
                variations_tested=len(variations),
                queries_per_variation=len(test_queries),
                optimization_time_ms=round((time.time() - start_time) * 1000, 2),
                all_scores={k: round(v, 4) for k, v in scores.items()},
                details={
                    "best_technique": best_variation.technique,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Store in history
            self.optimization_history.append(result)

            logger.info(
                f"Optimization complete: {improvement:.1f}% improvement | "
                f"Best: {best_var_id} ({best_variation.technique})"
            )

            return result

        except Exception as e:
            self.status = OptimizationStatus.FAILED
            logger.error(f"Optimization failed: {e}")
            raise

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get history of all optimizations."""
        return [r.model_dump() for r in self.optimization_history]

    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history = []
        logger.info("Optimization history cleared")

    def get_status(self) -> OptimizationStatus:
        """Get current optimization status."""
        return self.status
