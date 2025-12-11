"""Routing Rules for Query Classification.

This module defines the rules that map signals to routing decisions.
Rules are evaluated in priority order - first matching rule wins.

Rule Priority (highest to lowest):
1. Emergency keywords → EMERGENCY (L4C)
2. Complex analysis patterns → COMPLEX (L4B)
3. Storm/hurricane mentions → STANDARD (L4A)
4. Context escalation (follow-up) → Inherit previous tier
5. Default → SIMPLE (Basic)
"""

from dataclasses import dataclass
from typing import Callable

from backend.src.routing.models import (
    AgentLevelMapping,
    ContextSignal,
    QueryAnalysisSignal,
    QueryTier,
    TIER_TO_AGENT_LEVEL,
)


@dataclass
class RoutingRule:
    """A single routing rule with condition and result.

    Attributes:
        name: Rule identifier for logging
        priority: Lower number = higher priority
        condition: Function that evaluates signals
        tier: Resulting tier if condition is True
        confidence: Confidence score for this rule
        reasoning_template: Template for reasoning string
    """

    name: str
    priority: int
    condition: Callable[[QueryAnalysisSignal, ContextSignal], bool]
    tier: QueryTier
    confidence: float
    reasoning_template: str


# =============================================================================
# ROUTING RULES (Ordered by priority)
# =============================================================================

ROUTING_RULES: list[RoutingRule] = [
    # Rule 1: Emergency keywords (highest priority)
    RoutingRule(
        name="emergency_keywords",
        priority=1,
        condition=lambda q, c: q.has_emergency_keywords,
        tier=QueryTier.EMERGENCY,
        confidence=0.95,
        reasoning_template="Query contains emergency/safety keywords → EMERGENCY tier (L4C with HITL)",
    ),

    # Rule 2: Complex analysis patterns
    RoutingRule(
        name="complex_analysis",
        priority=2,
        condition=lambda q, c: q.has_complex_keywords and q.has_storm_keywords,
        tier=QueryTier.COMPLEX,
        confidence=0.90,
        reasoning_template="Query requests storm analysis/comparison → COMPLEX tier (L4B 8-agent)",
    ),

    # Rule 3: Complex patterns without storm (still complex)
    RoutingRule(
        name="complex_general",
        priority=3,
        condition=lambda q, c: q.has_complex_keywords or q.is_multi_question,
        tier=QueryTier.COMPLEX,
        confidence=0.85,
        reasoning_template="Query requests analysis or contains multiple questions → COMPLEX tier (L4B)",
    ),

    # Rule 4: Storm/hurricane mentions
    RoutingRule(
        name="storm_mention",
        priority=4,
        condition=lambda q, c: q.has_storm_keywords,
        tier=QueryTier.STANDARD,
        confidence=0.90,
        reasoning_template="Query mentions hurricane/storm → STANDARD tier (L4A 3-agent)",
    ),

    # Rule 5: Context escalation - don't downgrade from previous EMERGENCY
    RoutingRule(
        name="context_escalation_emergency",
        priority=5,
        condition=lambda q, c: c.is_follow_up and c.previous_tier == QueryTier.EMERGENCY,
        tier=QueryTier.EMERGENCY,
        confidence=0.85,
        reasoning_template="Follow-up in EMERGENCY conversation → Maintain EMERGENCY tier",
    ),

    # Rule 6: Context escalation - hurricane history
    RoutingRule(
        name="context_hurricane_history",
        priority=6,
        condition=lambda q, c: c.is_follow_up and c.has_hurricane_history,
        tier=QueryTier.STANDARD,
        confidence=0.80,
        reasoning_template="Follow-up to hurricane conversation → STANDARD tier (L4A)",
    ),

    # Rule 7: Conditional language (suggests need for reasoning)
    RoutingRule(
        name="conditional_language",
        priority=7,
        condition=lambda q, c: q.has_conditional_language and q.word_count > 10,
        tier=QueryTier.STANDARD,
        confidence=0.75,
        reasoning_template="Query has conditional language requiring reasoning → STANDARD tier",
    ),

    # Rule 8: Long queries (might be complex)
    RoutingRule(
        name="long_query",
        priority=8,
        condition=lambda q, c: q.word_count > 30,
        tier=QueryTier.STANDARD,
        confidence=0.70,
        reasoning_template="Long query (>30 words) may need multi-agent → STANDARD tier",
    ),

    # Rule 9: Default - simple weather query
    RoutingRule(
        name="default_simple",
        priority=99,
        condition=lambda q, c: True,  # Always matches
        tier=QueryTier.SIMPLE,
        confidence=0.80,
        reasoning_template="Simple weather query → SIMPLE tier (basic agent)",
    ),
]


def evaluate_rules(
    query_signal: QueryAnalysisSignal,
    context_signal: ContextSignal,
) -> tuple[QueryTier, float, str, str]:
    """Evaluate routing rules against signals.

    Rules are evaluated in priority order. First matching rule wins.

    Args:
        query_signal: Signal extracted from query text
        context_signal: Signal extracted from context

    Returns:
        Tuple of (tier, confidence, rule_name, reasoning)
    """
    # Sort rules by priority (should already be sorted, but ensure)
    sorted_rules = sorted(ROUTING_RULES, key=lambda r: r.priority)

    for rule in sorted_rules:
        if rule.condition(query_signal, context_signal):
            return (
                rule.tier,
                rule.confidence,
                rule.name,
                rule.reasoning_template,
            )

    # Fallback (should never reach due to default rule)
    return (
        QueryTier.SIMPLE,
        0.75,
        "fallback",
        "No rules matched → Default to SIMPLE",
    )


def get_agent_level(tier: QueryTier) -> str:
    """Map tier to agent level string.

    Args:
        tier: Query tier

    Returns:
        Agent level string (basic, l4a, l4b, l4c)
    """
    return TIER_TO_AGENT_LEVEL.get(tier, AgentLevelMapping.BASIC).value
