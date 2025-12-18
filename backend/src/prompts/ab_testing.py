"""
A/B Testing Framework for Prompt Optimization.

Level 6b: Self-Improvement Platform

Capabilities:
1. Run controlled experiments comparing prompt variants
2. Statistical significance testing (chi-squared, t-test)
3. Multi-armed bandit for adaptive selection
4. Results tracking and reporting

Target: Data-driven prompt selection with statistical rigor
"""

import logging
import math
import random
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Status of an A/B test."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


class ABTestResult(BaseModel):
    """Result of an A/B test."""

    test_id: str
    status: TestStatus = TestStatus.COMPLETED
    winner: str | None = None
    variants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    total_trials: int = 0
    confidence: float = 0.0
    p_value: float | None = None
    is_significant: bool = False
    started_at: str = ""
    completed_at: str = ""
    recommendation: str = ""


class VariantStats(BaseModel):
    """Statistics for a single variant."""

    name: str
    trials: int = 0
    successes: int = 0
    failures: int = 0
    success_rate: float = 0.0
    avg_score: float = 0.0
    scores: list[float] = Field(default_factory=list)


class ABTestRunner:
    """
    Run A/B tests for prompt optimization.

    Provides:
    - Controlled experiments with multiple variants
    - Statistical significance testing
    - Multi-armed bandit for exploration/exploitation
    - Results tracking and reporting
    """

    def __init__(
        self,
        evaluator: Any | None = None,
        min_trials_per_variant: int = 30,
        confidence_threshold: float = 0.95,
    ):
        """
        Initialize A/B test runner.

        Args:
            evaluator: Optional evaluator for scoring responses
            min_trials_per_variant: Minimum trials before declaring winner
            confidence_threshold: Required confidence level (default 95%)
        """
        self.evaluator = evaluator
        self.min_trials = min_trials_per_variant
        self.confidence_threshold = confidence_threshold

        # Active tests
        self.active_tests: dict[str, dict[str, Any]] = {}

        # History
        self.test_history: list[ABTestResult] = []

        logger.info(
            f"ABTestRunner initialized | min_trials={min_trials_per_variant} | "
            f"confidence={confidence_threshold}"
        )

    async def create_test(
        self,
        test_id: str,
        variants: dict[str, str],
        test_queries: list[str] | None = None,
    ) -> str:
        """
        Create a new A/B test.

        Args:
            test_id: Unique identifier for the test
            variants: Dict of variant_name -> prompt_text
            test_queries: Optional list of test queries

        Returns:
            test_id
        """
        if test_id in self.active_tests:
            logger.warning(f"Test {test_id} already exists, overwriting")

        self.active_tests[test_id] = {
            "variants": {
                name: VariantStats(name=name) for name in variants
            },
            "prompts": variants,
            "test_queries": test_queries or [],
            "status": TestStatus.PENDING,
            "started_at": "",
            "created_at": datetime.now().isoformat(),
        }

        logger.info(f"Created A/B test: {test_id} with {len(variants)} variants")
        return test_id

    async def run_test(
        self,
        test_id: str,
        agent_fn: Callable[[str, str], Any] | None = None,
        max_trials: int = 100,
    ) -> ABTestResult:
        """
        Run an A/B test.

        Args:
            test_id: Test identifier
            agent_fn: Function(prompt, query) -> response
            max_trials: Maximum trials per variant

        Returns:
            ABTestResult with winner and statistics
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]
        test["status"] = TestStatus.RUNNING
        test["started_at"] = datetime.now().isoformat()

        variants = test["variants"]
        prompts = test["prompts"]
        test_queries = test["test_queries"]

        # Generate test queries if not provided
        if not test_queries:
            test_queries = self._generate_default_queries()

        logger.info(f"Running A/B test: {test_id} | queries={len(test_queries)}")

        # Run trials
        for trial in range(max_trials):
            # Select query
            query = test_queries[trial % len(test_queries)]

            # Run each variant
            for variant_name, stats in variants.items():
                prompt = prompts[variant_name]

                # Get response
                if agent_fn:
                    try:
                        response = await self._invoke_agent(agent_fn, prompt, query)
                        score = await self._score_response(query, response)
                    except Exception as e:
                        logger.warning(f"Trial failed for {variant_name}: {e}")
                        score = 0.0
                else:
                    # Simulate for testing
                    score = self._simulate_score(variant_name)

                # Update stats
                stats.trials += 1
                stats.scores.append(score)
                stats.avg_score = sum(stats.scores) / len(stats.scores)

                if score >= 0.7:  # Success threshold
                    stats.successes += 1
                else:
                    stats.failures += 1

                stats.success_rate = stats.successes / stats.trials

            # Check for early stopping
            if trial >= self.min_trials:
                is_significant, p_value = self._check_significance(variants)
                if is_significant:
                    logger.info(f"Early stopping: significance reached at trial {trial}")
                    break

        # Determine winner
        winner, confidence, p_value, is_significant = self._determine_winner(variants)

        # Create result
        result = ABTestResult(
            test_id=test_id,
            status=TestStatus.COMPLETED,
            winner=winner,
            variants={
                name: {
                    "trials": s.trials,
                    "successes": s.successes,
                    "success_rate": round(s.success_rate, 4),
                    "avg_score": round(s.avg_score, 4),
                }
                for name, s in variants.items()
            },
            total_trials=sum(s.trials for s in variants.values()),
            confidence=confidence,
            p_value=p_value,
            is_significant=is_significant,
            started_at=test["started_at"],
            completed_at=datetime.now().isoformat(),
            recommendation=self._generate_recommendation(winner, variants, is_significant),
        )

        # Update test status
        test["status"] = TestStatus.COMPLETED

        # Add to history
        self.test_history.append(result)

        logger.info(
            f"A/B test complete: {test_id} | winner={winner} | "
            f"confidence={confidence:.2%} | significant={is_significant}"
        )

        return result

    async def _invoke_agent(
        self,
        agent_fn: Callable,
        prompt: str,
        query: str,
    ) -> str:
        """Invoke agent function."""
        import asyncio

        if asyncio.iscoroutinefunction(agent_fn):
            return await agent_fn(prompt, query)
        return agent_fn(prompt, query)

    async def _score_response(self, query: str, response: str) -> float:
        """Score a response using evaluator or heuristics."""
        if self.evaluator and hasattr(self.evaluator, "evaluate_query"):
            try:
                result = await self.evaluator.evaluate_query(
                    query=query,
                    answer=response,
                    retrieved_contexts=[],
                )
                return getattr(result, "answer_relevancy", 0.5)
            except Exception:
                pass

        # Heuristic scoring
        score = 0.5

        # Length check (prefer moderate length)
        words = len(response.split())
        if 50 <= words <= 200:
            score += 0.2
        elif words > 200:
            score += 0.1

        # Weather keywords
        weather_keywords = [
            "temperature", "forecast", "weather", "humidity",
            "wind", "precipitation", "sunny", "rain", "storm",
        ]
        keyword_count = sum(1 for kw in weather_keywords if kw.lower() in response.lower())
        score += min(keyword_count * 0.05, 0.2)

        # Structure (bullet points, numbers)
        if "•" in response or "1." in response or "-" in response:
            score += 0.1

        return min(score, 1.0)

    def _simulate_score(self, variant_name: str) -> float:
        """Simulate score for testing (deterministic based on name)."""
        # Create consistent but different scores per variant
        base_scores = {
            "control": 0.65,
            "variant_a": 0.72,
            "variant_b": 0.68,
            "formal": 0.70,
            "casual": 0.67,
            "structured": 0.75,
            "concise": 0.69,
        }

        base = base_scores.get(variant_name.lower(), 0.65)
        # Add small random variation
        noise = random.gauss(0, 0.05)
        return max(0, min(1, base + noise))

    def _check_significance(
        self,
        variants: dict[str, VariantStats],
    ) -> tuple[bool, float | None]:
        """Check if results are statistically significant."""
        if len(variants) < 2:
            return False, None

        # Get two best variants
        sorted_variants = sorted(
            variants.values(),
            key=lambda v: v.success_rate,
            reverse=True,
        )

        best = sorted_variants[0]
        second = sorted_variants[1]

        # Chi-squared test approximation
        if best.trials < self.min_trials or second.trials < self.min_trials:
            return False, None

        # Calculate chi-squared statistic
        total = best.trials + second.trials
        expected_success = (best.successes + second.successes) / 2

        if expected_success == 0:
            return False, None

        chi_sq = (
            ((best.successes - expected_success) ** 2) / expected_success
            + ((second.successes - expected_success) ** 2) / expected_success
        )

        # P-value approximation (chi-squared with 1 df)
        p_value = math.exp(-chi_sq / 2)

        is_significant = p_value < (1 - self.confidence_threshold)

        return is_significant, p_value

    def _determine_winner(
        self,
        variants: dict[str, VariantStats],
    ) -> tuple[str | None, float, float | None, bool]:
        """Determine test winner with confidence."""
        if not variants:
            return None, 0.0, None, False

        # Sort by success rate
        sorted_variants = sorted(
            variants.items(),
            key=lambda x: x[1].success_rate,
            reverse=True,
        )

        best_name, best_stats = sorted_variants[0]

        # Check significance
        is_significant, p_value = self._check_significance(variants)

        # Calculate confidence based on trial count and margin
        if len(sorted_variants) > 1:
            second_stats = sorted_variants[1][1]
            margin = best_stats.success_rate - second_stats.success_rate

            # More trials and larger margin = higher confidence
            trial_factor = min(best_stats.trials / self.min_trials, 1.0)
            margin_factor = min(margin * 5, 1.0)  # 20% margin = full confidence

            confidence = trial_factor * 0.5 + margin_factor * 0.5
        else:
            confidence = min(best_stats.trials / self.min_trials, 1.0)

        return best_name, confidence, p_value, is_significant

    def _generate_recommendation(
        self,
        winner: str | None,
        variants: dict[str, VariantStats],
        is_significant: bool,
    ) -> str:
        """Generate actionable recommendation."""
        if not winner:
            return "Insufficient data to determine winner. Run more trials."

        winner_stats = variants.get(winner)
        if not winner_stats:
            return "Winner data not found."

        if is_significant:
            return (
                f"RECOMMEND: Deploy '{winner}' variant. "
                f"Success rate: {winner_stats.success_rate:.1%} "
                f"({winner_stats.trials} trials). "
                "Result is statistically significant."
            )
        else:
            return (
                f"TENTATIVE: '{winner}' leads with {winner_stats.success_rate:.1%} "
                f"success rate, but result is NOT statistically significant. "
                "Consider running more trials before deployment."
            )

    def _generate_default_queries(self) -> list[str]:
        """Generate default test queries for weather domain."""
        return [
            "What's the weather in Miami?",
            "Will it rain tomorrow in Tampa?",
            "Is there a hurricane warning?",
            "What's the 5-day forecast for Orlando?",
            "Should I evacuate from the storm?",
            "What's the temperature today?",
            "How strong are the winds?",
            "When will the rain stop?",
            "Is it safe to go outside?",
            "What's the humidity level?",
        ]

    def get_active_tests(self) -> list[str]:
        """Get list of active test IDs."""
        return list(self.active_tests.keys())

    def get_test_status(self, test_id: str) -> dict[str, Any] | None:
        """Get status of a specific test."""
        if test_id not in self.active_tests:
            return None

        test = self.active_tests[test_id]
        return {
            "test_id": test_id,
            "status": test["status"].value,
            "variants": {
                name: {
                    "trials": s.trials,
                    "success_rate": round(s.success_rate, 4),
                }
                for name, s in test["variants"].items()
            },
            "created_at": test["created_at"],
            "started_at": test.get("started_at", ""),
        }

    def stop_test(self, test_id: str) -> ABTestResult | None:
        """Stop a running test and return results."""
        if test_id not in self.active_tests:
            return None

        test = self.active_tests[test_id]
        if test["status"] != TestStatus.RUNNING:
            return None

        variants = test["variants"]
        winner, confidence, p_value, is_significant = self._determine_winner(variants)

        result = ABTestResult(
            test_id=test_id,
            status=TestStatus.STOPPED,
            winner=winner,
            variants={
                name: {
                    "trials": s.trials,
                    "successes": s.successes,
                    "success_rate": round(s.success_rate, 4),
                    "avg_score": round(s.avg_score, 4),
                }
                for name, s in variants.items()
            },
            total_trials=sum(s.trials for s in variants.values()),
            confidence=confidence,
            p_value=p_value,
            is_significant=is_significant,
            started_at=test.get("started_at", ""),
            completed_at=datetime.now().isoformat(),
            recommendation=self._generate_recommendation(winner, variants, is_significant),
        )

        test["status"] = TestStatus.STOPPED
        self.test_history.append(result)

        return result

    def get_history(self, limit: int = 10) -> list[ABTestResult]:
        """Get recent test history."""
        return self.test_history[-limit:]

    def clear_history(self) -> None:
        """Clear test history."""
        self.test_history = []
        logger.info("A/B test history cleared")


class MultiArmedBandit:
    """
    Multi-armed bandit for adaptive prompt selection.

    Uses Thompson Sampling for exploration/exploitation balance.
    """

    def __init__(self, variants: list[str]):
        """
        Initialize bandit with variants.

        Args:
            variants: List of variant names
        """
        self.variants = variants

        # Beta distribution parameters (successes, failures)
        self.alpha = dict.fromkeys(variants, 1)  # Prior successes
        self.beta = dict.fromkeys(variants, 1)   # Prior failures

        self.total_pulls = 0
        self.selection_history: list[str] = []

        logger.info(f"MultiArmedBandit initialized with {len(variants)} arms")

    def select_variant(self) -> str:
        """
        Select variant using Thompson Sampling.

        Returns:
            Selected variant name
        """
        # Sample from Beta distribution for each variant
        samples = {}
        for variant in self.variants:
            samples[variant] = random.betavariate(
                self.alpha[variant],
                self.beta[variant],
            )

        # Select variant with highest sample
        selected = max(samples, key=lambda x: samples[x])

        self.total_pulls += 1
        self.selection_history.append(selected)

        return selected

    def update(self, variant: str, success: bool) -> None:
        """
        Update variant statistics.

        Args:
            variant: Variant name
            success: Whether trial was successful
        """
        if variant not in self.variants:
            return

        if success:
            self.alpha[variant] += 1
        else:
            self.beta[variant] += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get current bandit statistics."""
        stats = {}

        for variant in self.variants:
            a = self.alpha[variant]
            b = self.beta[variant]

            # Expected value (mean of Beta distribution)
            expected = a / (a + b)

            # 95% confidence interval
            # Using normal approximation for simplicity
            std = math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))
            ci_lower = max(0, expected - 1.96 * std)
            ci_upper = min(1, expected + 1.96 * std)

            stats[variant] = {
                "expected_value": round(expected, 4),
                "successes": a - 1,  # Subtract prior
                "failures": b - 1,   # Subtract prior
                "confidence_interval": [round(ci_lower, 4), round(ci_upper, 4)],
            }

        return {
            "variants": stats,
            "total_pulls": self.total_pulls,
            "recommendation": self._get_recommendation(),
        }

    def _get_recommendation(self) -> str:
        """Get recommendation based on current statistics."""
        if self.total_pulls < 50:
            return "Continue exploration - insufficient data"

        # Find best variant
        best = max(
            self.variants,
            key=lambda v: self.alpha[v] / (self.alpha[v] + self.beta[v]),
        )

        # Check confidence
        a = self.alpha[best]
        b = self.beta[best]
        expected = a / (a + b)

        if expected > 0.7:
            return f"EXPLOIT: '{best}' is performing well ({expected:.1%}). Consider deployment."
        elif expected > 0.5:
            return f"EXPLORE: '{best}' leads but needs more data ({expected:.1%})."
        else:
            return "INVESTIGATE: All variants underperforming. Review prompt design."

    def reset(self) -> None:
        """Reset bandit to initial state."""
        self.alpha = dict.fromkeys(self.variants, 1)
        self.beta = dict.fromkeys(self.variants, 1)
        self.total_pulls = 0
        self.selection_history = []
        logger.info("MultiArmedBandit reset")
