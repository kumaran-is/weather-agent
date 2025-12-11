"""Routing Models for Auto-Routing Architecture.

This module defines the data models for the intent-based query classification system.
The classifier analyzes query text and conversation context to determine the appropriate
agent tier without requiring explicit flags.

Tiers:
- SIMPLE: Basic weather queries → Basic single agent
- STANDARD: Hurricane/storm queries → L4A (3-agent)
- COMPLEX: Analysis/comparison queries → L4B (8-agent)
- EMERGENCY: Safety/evacuation queries → L4C (15-agent) with HITL
"""

from dataclasses import dataclass, field
from enum import Enum


class QueryTier(str, Enum):
    """Query complexity tier determining agent routing.

    Each tier maps to a specific agent configuration:
    - SIMPLE → Basic single agent (fast, low cost)
    - STANDARD → L4A 3-agent (Triage, Hurricane Specialist, Alert Manager)
    - COMPLEX → L4B 8-agent (+ Supervisor, Forecaster, Historical, Research)
    - EMERGENCY → L4C 15-agent (+ HITL triggers, Emergency Response)
    """

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"
    EMERGENCY = "emergency"


class AgentLevelMapping(str, Enum):
    """Maps QueryTier to actual agent level configuration.

    This replaces the old AgentLevel enum that was exposed in the API.
    Now routing is internal and automatic.
    """

    BASIC = "basic"  # Single agent
    L4A = "l4a"      # 3-agent
    L4B = "l4b"      # 8-agent
    L4C = "l4c"      # 15-agent


# Tier to Agent Level mapping
TIER_TO_AGENT_LEVEL: dict[QueryTier, AgentLevelMapping] = {
    QueryTier.SIMPLE: AgentLevelMapping.BASIC,
    QueryTier.STANDARD: AgentLevelMapping.L4A,
    QueryTier.COMPLEX: AgentLevelMapping.L4B,
    QueryTier.EMERGENCY: AgentLevelMapping.L4C,
}


@dataclass
class QueryAnalysisSignal:
    """Signal extracted from query text analysis.

    Attributes:
        has_emergency_keywords: Query contains safety/evacuation terms
        has_complex_keywords: Query asks for analysis/comparison
        has_storm_keywords: Query mentions hurricanes/storms
        is_multi_question: Query contains multiple questions
        word_count: Total words in query
        has_conditional_language: Query contains if/when/should
        matched_patterns: List of patterns that matched
    """

    has_emergency_keywords: bool = False
    has_complex_keywords: bool = False
    has_storm_keywords: bool = False
    is_multi_question: bool = False
    word_count: int = 0
    has_conditional_language: bool = False
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class ContextSignal:
    """Signal extracted from conversation memory/context.

    Attributes:
        is_follow_up: Query is a follow-up to previous conversation
        previous_tier: Tier used in previous query (if any)
        conversation_depth: Number of exchanges on same topic
        has_hurricane_history: User previously asked about hurricanes
    """

    is_follow_up: bool = False
    previous_tier: QueryTier | None = None
    conversation_depth: int = 0
    has_hurricane_history: bool = False


@dataclass
class RoutingDecision:
    """Final routing decision with confidence and reasoning.

    Attributes:
        tier: Determined query tier (SIMPLE/STANDARD/COMPLEX/EMERGENCY)
        agent_level: Mapped agent level (basic/l4a/l4b/l4c)
        confidence: Confidence score (0.0-1.0)
        primary_signal: What drove the decision (query_keywords, context, etc.)
        query_signal: Query analysis signal details
        context_signal: Context analysis signal details
        reasoning: Human-readable explanation of routing decision
        hitl_required: Whether HITL approval may be needed (EMERGENCY tier)
    """

    tier: QueryTier
    agent_level: str
    confidence: float
    primary_signal: str
    query_signal: QueryAnalysisSignal
    context_signal: ContextSignal
    reasoning: str
    hitl_required: bool = False

    def __post_init__(self):
        """Set HITL requirement based on tier."""
        self.hitl_required = self.tier == QueryTier.EMERGENCY

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/response."""
        return {
            "tier": self.tier.value,
            "agent_level": self.agent_level,
            "confidence": self.confidence,
            "primary_signal": self.primary_signal,
            "reasoning": self.reasoning,
            "hitl_required": self.hitl_required,
            "matched_patterns": self.query_signal.matched_patterns,
        }
