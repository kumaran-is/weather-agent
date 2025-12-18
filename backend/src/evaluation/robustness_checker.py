"""Pillar 3: Robustness Checker (Edge Case Handling).

This module implements heuristic-based robustness evaluation:
1. Error handling: Does agent handle errors gracefully?
2. Missing data: Does it respond appropriately to gaps?
3. Ambiguity: Does it ask for clarification when needed?
4. Recovery: Can it self-correct after mistakes?

Scoring is based on pattern matching and heuristic rules.

Edge Case Categories:
- Invalid location names
- Missing weather data
- Ambiguous queries
- Out-of-range dates
- Network/API failures
- Malformed requests

Usage:
    >>> checker = RobustnessChecker()
    >>> result = checker.evaluate(
    ...     query="Weather in asdfghjkl?",
    ...     trajectory=[{"error": "Location not found"}],
    ...     final_answer="I couldn't find that location. Could you please provide a valid city name?"
    ... )
    >>> print(f"Score: {result.score}")
"""

import logging
import re
from typing import Any

from backend.src.evaluation.models import RobustnessResult

logger = logging.getLogger(__name__)


class RobustnessChecker:
    """Heuristic-based robustness evaluation.

    Checks how well the agent handles edge cases and errors.
    No LLM calls - uses pattern matching for speed.

    Attributes:
        error_weight: Weight for error handling (default: 0.3)
        missing_weight: Weight for missing data handling (default: 0.3)
        ambiguity_weight: Weight for ambiguity handling (default: 0.2)
        recovery_weight: Weight for recovery capability (default: 0.2)
    """

    # Patterns indicating graceful error handling
    GRACEFUL_ERROR_PATTERNS = [
        r"couldn't find",
        r"unable to locate",
        r"not available",
        r"please provide",
        r"could you clarify",
        r"try again",
        r"sorry",
        r"apologize",
        r"valid (city|location|date)",
        r"no (data|information|results)",
        r"currently unavailable",
    ]

    # Patterns indicating poor error handling
    BAD_ERROR_PATTERNS = [
        r"error:",
        r"exception",
        r"traceback",
        r"undefined",
        r"null",
        r"NaN",
        r"stacktrace",
        r"\[\s*\]",  # Empty array
        r"\{\s*\}",  # Empty object
    ]

    # Patterns indicating clarification request
    CLARIFICATION_PATTERNS = [
        r"which (city|location|date)",
        r"could you (specify|clarify)",
        r"did you mean",
        r"be more specific",
        r"multiple (cities|locations)",
        r"please choose",
        r"which one",
    ]

    # Patterns indicating recovery attempt
    RECOVERY_PATTERNS = [
        r"let me try",
        r"alternative",
        r"instead",
        r"another approach",
        r"fallback",
        r"similar location",
        r"nearby",
    ]

    def __init__(
        self,
        error_weight: float = 0.3,
        missing_weight: float = 0.3,
        ambiguity_weight: float = 0.2,
        recovery_weight: float = 0.2,
    ) -> None:
        """Initialize checker with configurable weights."""
        self.error_weight = error_weight
        self.missing_weight = missing_weight
        self.ambiguity_weight = ambiguity_weight
        self.recovery_weight = recovery_weight

    def evaluate(
        self,
        query: str,
        trajectory: list[dict[str, Any]],
        final_answer: str,
        is_edge_case: bool = False,
    ) -> RobustnessResult:
        """Evaluate robustness of agent response.

        Args:
            query: Original user query
            trajectory: Agent execution steps
            final_answer: Agent's final response
            is_edge_case: If True, apply stricter scoring

        Returns:
            RobustnessResult with component scores
        """
        # Detect if this query is an edge case
        detected_edge_cases = self._detect_edge_cases(query, trajectory)

        # If not an edge case and no errors in trajectory, full score
        if not detected_edge_cases and not self._has_errors(trajectory):
            return RobustnessResult(
                score=1.0,
                error_handling=1.0,
                missing_data_handling=1.0,
                ambiguity_handling=1.0,
                recovery_capability=1.0,
                edge_cases_tested=["normal_query"],
                edge_cases_passed=["normal_query"],
            )

        # Calculate component scores
        error_handling = self._score_error_handling(final_answer, trajectory)
        missing_data = self._score_missing_data_handling(final_answer, trajectory)
        ambiguity = self._score_ambiguity_handling(query, final_answer)
        recovery = self._score_recovery(trajectory, final_answer)

        # Calculate weighted score
        score = (
            self.error_weight * error_handling +
            self.missing_weight * missing_data +
            self.ambiguity_weight * ambiguity +
            self.recovery_weight * recovery
        )

        # Apply stricter scoring for explicit edge cases
        if is_edge_case and score < 0.8:
            score *= 0.9  # 10% penalty for failing explicit edge case

        # Determine which edge cases passed
        passed_cases = []
        if error_handling >= 0.7:
            passed_cases.extend([c for c in detected_edge_cases if "error" in c.lower()])
        if missing_data >= 0.7:
            passed_cases.extend([c for c in detected_edge_cases if "missing" in c.lower()])
        if ambiguity >= 0.7:
            passed_cases.extend([c for c in detected_edge_cases if "ambig" in c.lower()])

        return RobustnessResult(
            score=score,
            error_handling=error_handling,
            missing_data_handling=missing_data,
            ambiguity_handling=ambiguity,
            recovery_capability=recovery,
            edge_cases_tested=detected_edge_cases,
            edge_cases_passed=passed_cases,
        )

    def _detect_edge_cases(
        self,
        query: str,
        trajectory: list[dict[str, Any]],
    ) -> list[str]:
        """Detect which edge cases are present in query/trajectory."""
        edge_cases = []

        # Invalid location patterns
        if re.search(r"\b[a-z]{8,}\b", query.lower()) and not any(
            city in query.lower() for city in ["francisco", "angeles", "washington"]
        ):
            edge_cases.append("potentially_invalid_location")

        # Missing data in trajectory
        for step in trajectory:
            if isinstance(step, dict):
                if step.get("error") or step.get("exception"):
                    edge_cases.append("error_in_trajectory")
                if "not found" in str(step.get("output", "")).lower():
                    edge_cases.append("missing_data")

        # Ambiguous query patterns
        ambiguous_patterns = [
            r"\bweather\b(?!\s+(in|at|for|near))",  # "weather" without location
            r"here\b",  # Relative location
            r"(this|next)\s+week",  # Relative time without anchor
        ]
        for pattern in ambiguous_patterns:
            if re.search(pattern, query.lower()):
                edge_cases.append("ambiguous_query")
                break

        return list(set(edge_cases))  # Remove duplicates

    def _has_errors(self, trajectory: list[dict[str, Any]]) -> bool:
        """Check if trajectory contains any errors."""
        for step in trajectory:
            if isinstance(step, dict):
                if step.get("error") or step.get("exception") or step.get("failed"):
                    return True
        return False

    def _score_error_handling(
        self,
        final_answer: str,
        trajectory: list[dict[str, Any]],
    ) -> float:
        """Score error handling quality."""
        answer_lower = final_answer.lower()

        # Check for bad error exposure
        for pattern in self.BAD_ERROR_PATTERNS:
            if re.search(pattern, answer_lower, re.IGNORECASE):
                return 0.2  # Exposed raw errors = bad

        # Check for graceful error handling
        graceful_matches = sum(
            1 for p in self.GRACEFUL_ERROR_PATTERNS
            if re.search(p, answer_lower, re.IGNORECASE)
        )

        # If trajectory has errors but answer handles gracefully
        if self._has_errors(trajectory):
            if graceful_matches >= 2:
                return 1.0
            elif graceful_matches >= 1:
                return 0.8
            else:
                return 0.4  # Had errors but didn't communicate clearly

        return 1.0  # No errors = full score

    def _score_missing_data_handling(
        self,
        final_answer: str,
        trajectory: list[dict[str, Any]],
    ) -> float:
        """Score handling of missing data."""
        # Check if trajectory indicates missing data
        has_missing_data = False
        for step in trajectory:
            if isinstance(step, dict):
                output = str(step.get("output", ""))
                if any(p in output.lower() for p in ["not found", "no data", "unavailable"]):
                    has_missing_data = True
                    break

        if not has_missing_data:
            return 1.0  # No missing data = full score

        # Check if answer acknowledges missing data appropriately
        answer_lower = final_answer.lower()
        acknowledgment_patterns = [
            r"no (data|information)",
            r"(data|information) (not available|unavailable)",
            r"couldn't (find|retrieve|get)",
            r"currently unavailable",
        ]

        for pattern in acknowledgment_patterns:
            if re.search(pattern, answer_lower):
                return 0.9  # Acknowledged missing data

        # Check if hallucinated data
        if len(final_answer) > 100 and "unfortunately" not in answer_lower:
            return 0.4  # Might have hallucinated instead of admitting

        return 0.7

    def _score_ambiguity_handling(self, query: str, final_answer: str) -> float:
        """Score handling of ambiguous queries."""
        # Detect if query is ambiguous
        is_ambiguous = False
        ambiguous_indicators = [
            len(query.split()) < 4,  # Very short query
            "weather" in query.lower() and not re.search(r"\bin\b|\bat\b|\bfor\b", query.lower()),
        ]

        if any(ambiguous_indicators):
            is_ambiguous = True

        if not is_ambiguous:
            return 1.0  # Clear query = full score

        # Check if answer asks for clarification
        answer_lower = final_answer.lower()
        for pattern in self.CLARIFICATION_PATTERNS:
            if re.search(pattern, answer_lower, re.IGNORECASE):
                return 1.0  # Asked for clarification = great

        # Check if answer made reasonable assumption
        if re.search(r"assuming|based on|i'll use|defaulting to", answer_lower):
            return 0.8  # Made assumption but stated it

        return 0.6  # Didn't clarify or state assumption

    def _score_recovery(
        self,
        trajectory: list[dict[str, Any]],
        final_answer: str,
    ) -> float:
        """Score recovery/self-correction capability."""
        # Check if there was a failure followed by recovery
        had_failure = False
        had_retry = False

        for i, step in enumerate(trajectory):
            if isinstance(step, dict):
                if step.get("error") or "failed" in str(step.get("status", "")).lower():
                    had_failure = True
                if had_failure and i > 0:
                    if step.get("retry") or "retry" in str(step).lower():
                        had_retry = True

        if not had_failure:
            return 1.0  # No failure = full score

        # Check for recovery patterns in answer
        answer_lower = final_answer.lower()
        for pattern in self.RECOVERY_PATTERNS:
            if re.search(pattern, answer_lower, re.IGNORECASE):
                return 0.9 if had_retry else 0.8

        if had_retry:
            return 0.7  # Retried but didn't communicate

        return 0.5  # Failed without recovery attempt
