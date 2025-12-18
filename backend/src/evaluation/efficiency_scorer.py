"""Pillar 2: Efficiency Scorer (Deterministic Metrics).

This module implements deterministic efficiency evaluation based on:
1. Tool accuracy: Did agent use expected tools?
2. Call efficiency: Minimal necessary tool calls?
3. Latency: Within time budget?
4. Token usage: Within token budget?

All scoring is deterministic (no LLM calls) for fast evaluation.

Scoring Formula:
- Tool accuracy: Jaccard similarity between actual vs expected tools
- Call efficiency: 1.0 - (extra_calls / max_allowed_extra)
- Latency score: 1.0 - (actual_latency / budget), capped at 0.0
- Token score: 1.0 - (actual_tokens / budget), capped at 0.0

Final Score = weighted average with configurable weights

Usage:
    >>> scorer = EfficiencyScorer()
    >>> result = scorer.evaluate(
    ...     actual_tools=["get_weather", "get_forecast"],
    ...     expected_tools=["get_weather"],
    ...     latency_ms=1500.0,
    ...     latency_budget_ms=5000.0,
    ...     tokens_used=1200,
    ...     token_budget=4000
    ... )
    >>> print(f"Score: {result.score}")
"""

import logging

from backend.src.evaluation.models import EfficiencyResult

logger = logging.getLogger(__name__)


class EfficiencyScorer:
    """Deterministic efficiency scoring for agent trajectories.

    Evaluates whether the agent took an efficient path by checking:
    - Did it use the right tools?
    - Did it avoid unnecessary tool calls?
    - Was it fast enough?
    - Did it stay within token budget?

    Attributes:
        tool_weight: Weight for tool accuracy (default: 0.4)
        call_weight: Weight for call efficiency (default: 0.2)
        latency_weight: Weight for latency score (default: 0.2)
        token_weight: Weight for token score (default: 0.2)
    """

    def __init__(
        self,
        tool_weight: float = 0.4,
        call_weight: float = 0.2,
        latency_weight: float = 0.2,
        token_weight: float = 0.2,
    ) -> None:
        """Initialize scorer with configurable weights.

        Args:
            tool_weight: Weight for tool accuracy (0-1)
            call_weight: Weight for call efficiency (0-1)
            latency_weight: Weight for latency score (0-1)
            token_weight: Weight for token score (0-1)

        Note: Weights should sum to 1.0
        """
        self.tool_weight = tool_weight
        self.call_weight = call_weight
        self.latency_weight = latency_weight
        self.token_weight = token_weight

        total = tool_weight + call_weight + latency_weight + token_weight
        if abs(total - 1.0) > 0.001:
            logger.warning(f"Efficiency weights sum to {total}, not 1.0")

    def evaluate(
        self,
        actual_tools: list[str],
        expected_tools: list[str],
        latency_ms: float = 0.0,
        latency_budget_ms: float = 5000.0,
        tokens_used: int = 0,
        token_budget: int = 4000,
        max_extra_calls: int = 5,
    ) -> EfficiencyResult:
        """Evaluate trajectory efficiency.

        Args:
            actual_tools: Tools actually called by agent
            expected_tools: Expected tools for this query
            latency_ms: Total execution time in milliseconds
            latency_budget_ms: Maximum acceptable latency
            tokens_used: Total tokens consumed
            token_budget: Maximum acceptable tokens
            max_extra_calls: Maximum extra calls before score = 0

        Returns:
            EfficiencyResult with component scores
        """
        # Calculate tool accuracy (Jaccard similarity)
        tool_accuracy = self._calculate_tool_accuracy(actual_tools, expected_tools)

        # Calculate call efficiency
        call_efficiency = self._calculate_call_efficiency(
            actual_tools, expected_tools, max_extra_calls
        )

        # Calculate latency score
        latency_score = self._calculate_latency_score(latency_ms, latency_budget_ms)

        # Calculate token score
        token_score = self._calculate_token_score(tokens_used, token_budget)

        # Weighted final score
        final_score = (
            self.tool_weight * tool_accuracy +
            self.call_weight * call_efficiency +
            self.latency_weight * latency_score +
            self.token_weight * token_score
        )

        return EfficiencyResult(
            score=final_score,
            tool_accuracy=tool_accuracy,
            call_efficiency=call_efficiency,
            latency_score=latency_score,
            token_score=token_score,
            actual_tool_calls=actual_tools,
            expected_tool_calls=expected_tools,
            total_latency_ms=latency_ms,
            total_tokens=tokens_used,
        )

    def _calculate_tool_accuracy(
        self,
        actual: list[str],
        expected: list[str],
    ) -> float:
        """Calculate Jaccard similarity between actual and expected tools.

        Jaccard = |intersection| / |union|

        Special cases:
        - Both empty: 1.0 (perfect match)
        - One empty: 0.0 if other has items
        """
        if not expected and not actual:
            return 1.0  # Both empty = perfect match

        actual_set = set(actual)
        expected_set = set(expected)

        if not expected_set:
            # No expected tools - score based on whether agent called anything
            return 1.0 if not actual_set else 0.8  # Slight penalty for unnecessary calls

        intersection = actual_set & expected_set
        union = actual_set | expected_set

        if not union:
            return 1.0

        return len(intersection) / len(union)

    def _calculate_call_efficiency(
        self,
        actual: list[str],
        expected: list[str],
        max_extra: int,
    ) -> float:
        """Calculate efficiency based on extra/missing calls.

        Penalizes both:
        - Extra calls (unnecessary work)
        - Missing calls (incomplete execution)
        """
        expected_set = set(expected)
        actual_set = set(actual)

        # Count extra calls (not expected)
        extra_calls = len(actual_set - expected_set)

        # Count missing calls (expected but not made)
        missing_calls = len(expected_set - actual_set)

        # Total deviation from optimal
        total_deviation = extra_calls + missing_calls

        if total_deviation == 0:
            return 1.0

        # Score decreases with deviation
        # At max_extra deviation, score = 0.0
        efficiency = max(0.0, 1.0 - (total_deviation / max_extra))

        return efficiency

    def _calculate_latency_score(
        self,
        latency_ms: float,
        budget_ms: float,
    ) -> float:
        """Calculate latency score (1.0 = instant, 0.0 = at/over budget).

        Linear decay from 1.0 to 0.0 as latency approaches budget.
        """
        if budget_ms <= 0:
            return 1.0 if latency_ms == 0 else 0.0

        if latency_ms <= 0:
            return 1.0

        # Linear score: 1.0 at 0ms, 0.0 at budget
        score = 1.0 - (latency_ms / budget_ms)

        return max(0.0, min(1.0, score))

    def _calculate_token_score(
        self,
        tokens: int,
        budget: int,
    ) -> float:
        """Calculate token efficiency score.

        Linear decay from 1.0 to 0.0 as tokens approach budget.
        """
        if budget <= 0:
            return 1.0 if tokens == 0 else 0.0

        if tokens <= 0:
            return 1.0

        # Linear score: 1.0 at 0 tokens, 0.0 at budget
        score = 1.0 - (tokens / budget)

        return max(0.0, min(1.0, score))
