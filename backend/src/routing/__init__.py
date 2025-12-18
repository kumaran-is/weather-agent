"""Auto-Routing Module for Weather AI Agent.

This module provides intelligent query classification and routing based on
query intent and conversation context. It replaces the explicit `use_multi_agent`
and `agent_level` flags with automatic classification.

Usage:
    from backend.src.routing import classify_query, QueryTier

    decision = classify_query(
        query="Is Hurricane Milton going to hit Tampa?",
        memory_context=memory_context,
    )

    print(decision.tier)        # QueryTier.STANDARD
    print(decision.agent_level) # "l4a"
    print(decision.confidence)  # 0.90
    print(decision.reasoning)   # "Query mentions hurricane/storm → STANDARD tier"

Components:
- QueryClassifier: Main classifier class
- classify_query: Convenience function for classification
- QueryTier: Enum for query complexity tiers
- RoutingDecision: Classification result with all details
"""

from backend.src.routing.classifier import (
    QueryClassifier,
    classify_query,
    get_classifier,
)
from backend.src.routing.models import (
    TIER_TO_AGENT_LEVEL,
    AgentLevelMapping,
    ContextSignal,
    QueryAnalysisSignal,
    QueryTier,
    RoutingDecision,
)
from backend.src.routing.rules import evaluate_rules, get_agent_level
from backend.src.routing.signals import extract_context_signal, extract_query_signal

__all__ = [
    # Main interface
    "QueryClassifier",
    "classify_query",
    "get_classifier",
    # Models
    "QueryTier",
    "RoutingDecision",
    "QueryAnalysisSignal",
    "ContextSignal",
    "AgentLevelMapping",
    "TIER_TO_AGENT_LEVEL",
    # Utilities
    "evaluate_rules",
    "get_agent_level",
    "extract_query_signal",
    "extract_context_signal",
]
