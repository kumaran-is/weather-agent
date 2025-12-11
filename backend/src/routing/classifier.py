"""Query Classifier for Auto-Routing.

This module provides the main QueryClassifier class that combines signal
extraction and rule evaluation to produce routing decisions.

Usage:
    classifier = QueryClassifier()
    decision = classifier.classify(
        query="Is Hurricane Milton going to hit Tampa?",
        memory_context=memory_context,  # Optional
    )
    print(decision.tier)  # QueryTier.STANDARD
    print(decision.agent_level)  # "l4a"
"""

import logging
from typing import Any

from backend.src.routing.models import (
    ContextSignal,
    QueryAnalysisSignal,
    QueryTier,
    RoutingDecision,
)
from backend.src.routing.rules import evaluate_rules, get_agent_level
from backend.src.routing.signals import extract_context_signal, extract_query_signal

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Intent-based query classifier for auto-routing.

    This classifier analyzes query text and conversation context to determine
    the appropriate agent tier. It's designed to be fast (< 1ms) with no
    external API calls.

    The classifier follows these principles:
    1. Route based on USER INTENT (what they asked for)
    2. Don't pre-fetch external data (agents fetch what they need)
    3. Use simple, fast rules (no LLM classification)
    4. Escalate based on explicit signals, not speculation
    """

    def __init__(self):
        """Initialize the classifier."""
        self._classification_count = 0

    def classify(
        self,
        query: str,
        memory_context: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Classify a query and determine routing.

        Args:
            query: User's query text
            memory_context: Optional memory context from MemoryManager

        Returns:
            RoutingDecision with tier, agent_level, confidence, and reasoning
        """
        self._classification_count += 1

        # Extract signals (< 1ms each)
        query_signal = extract_query_signal(query)
        context_signal = extract_context_signal(memory_context)

        # Evaluate rules to get tier
        tier, confidence, rule_name, reasoning = evaluate_rules(
            query_signal, context_signal
        )

        # Map tier to agent level
        agent_level = get_agent_level(tier)

        # Build decision
        decision = RoutingDecision(
            tier=tier,
            agent_level=agent_level,
            confidence=confidence,
            primary_signal=rule_name,
            query_signal=query_signal,
            context_signal=context_signal,
            reasoning=reasoning,
        )

        # Log the decision
        logger.info(
            f"🎯 Query classified | "
            f"tier: {tier.value} | "
            f"agent_level: {agent_level} | "
            f"confidence: {confidence:.2f} | "
            f"rule: {rule_name} | "
            f"patterns: {query_signal.matched_patterns}"
        )

        return decision

    def get_stats(self) -> dict[str, Any]:
        """Get classifier statistics.

        Returns:
            dict with classification count and other stats
        """
        return {
            "classification_count": self._classification_count,
        }


# Module-level singleton for convenience
_classifier: QueryClassifier | None = None


def get_classifier() -> QueryClassifier:
    """Get or create the singleton classifier instance.

    Returns:
        QueryClassifier singleton instance
    """
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier


def classify_query(
    query: str,
    memory_context: dict[str, Any] | None = None,
) -> RoutingDecision:
    """Convenience function to classify a query.

    Args:
        query: User's query text
        memory_context: Optional memory context

    Returns:
        RoutingDecision with routing information
    """
    return get_classifier().classify(query, memory_context)
