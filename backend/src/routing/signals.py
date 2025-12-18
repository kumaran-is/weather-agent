"""Signal Extraction for Query Classification.

This module extracts signals from query text and conversation context
to determine routing decisions. All signal extraction is designed to be
fast (< 1ms) with no external API calls.

Signal Types:
1. Query Analysis Signal - Extracted from query text via regex/keywords
2. Context Signal - Extracted from memory context (if available)
"""

import re
from typing import Any

from backend.src.routing.models import ContextSignal, QueryAnalysisSignal, QueryTier

# =============================================================================
# KEYWORD PATTERNS (Compiled for performance)
# =============================================================================

# EMERGENCY: User explicitly asks about safety/evacuation
EMERGENCY_PATTERNS = [
    re.compile(r"\bshould\s+i\s+evacuate\b", re.IGNORECASE),
    re.compile(r"\bam\s+i\s+safe\b", re.IGNORECASE),
    re.compile(r"\bis\s+it\s+dangerous\b", re.IGNORECASE),
    re.compile(r"\blife[- ]?threatening\b", re.IGNORECASE),
    re.compile(r"\bwill\s+i\s+die\b", re.IGNORECASE),
    re.compile(r"\bemergency\b", re.IGNORECASE),
    re.compile(r"\bmandatory\s+evacuation\b", re.IGNORECASE),
    re.compile(r"\bshelter\s+in\s+place\b", re.IGNORECASE),
    re.compile(r"\bevacuate\s+now\b", re.IGNORECASE),
    re.compile(r"\bimmediate\s+danger\b", re.IGNORECASE),
]

EMERGENCY_KEYWORDS = frozenset([
    "evacuate", "evacuation", "shelter", "danger", "dangerous",
    "life-threatening", "lifethreatening", "emergency", "urgent",
    "critical", "deadly", "fatal", "survive", "survival",
])

# COMPLEX: User asks for analysis, comparison, or multi-day forecasts
COMPLEX_PATTERNS = [
    re.compile(r"\bcompare\s+.+\s+to\b", re.IGNORECASE),
    re.compile(r"\bhistorical\s+(data|pattern|trend|comparison)\b", re.IGNORECASE),
    re.compile(r"\bforecast\s+(next\s+)?\d+\s+days?\b", re.IGNORECASE),
    re.compile(r"\banalyze\b", re.IGNORECASE),
    re.compile(r"\bpattern\s+analysis\b", re.IGNORECASE),
    re.compile(r"\btrend\s+(analysis|over)\b", re.IGNORECASE),
    re.compile(r"\bpredict(ion)?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+if\b", re.IGNORECASE),
]

COMPLEX_KEYWORDS = frozenset([
    "compare", "comparison", "analyze", "analysis", "historical",
    "pattern", "trend", "predict", "prediction", "forecast",
    "comprehensive", "detailed", "in-depth", "multiple",
    "various", "several", "different",
])

# STANDARD: User asks about hurricanes/storms
STORM_PATTERNS = [
    re.compile(r"\bhurricane\s+\w+\b", re.IGNORECASE),  # Hurricane [Name]
    re.compile(r"\btropical\s+storm\b", re.IGNORECASE),
    re.compile(r"\bcat(egory)?\s*[1-5]\b", re.IGNORECASE),
    re.compile(r"\bstorm\s+surge\b", re.IGNORECASE),
    re.compile(r"\blandfall\b", re.IGNORECASE),
    re.compile(r"\bcone\s+of\s+(uncertainty|concern)\b", re.IGNORECASE),
    re.compile(r"\bnhc\b", re.IGNORECASE),
    re.compile(r"\btrack(ing)?\s+(the\s+)?storm\b", re.IGNORECASE),
]

STORM_KEYWORDS = frozenset([
    "hurricane", "tropical", "cyclone", "typhoon", "storm",
    "nhc", "advisory", "surge", "landfall", "cone", "tracking",
    "eye", "eyewall", "category", "cat1", "cat2", "cat3", "cat4", "cat5",
])

# Conditional language indicators (suggest complexity)
CONDITIONAL_PATTERNS = [
    re.compile(r"\bif\s+.+\s+(then|will|would|should)\b", re.IGNORECASE),
    re.compile(r"\bwhen\s+should\s+i\b", re.IGNORECASE),
    re.compile(r"\bshould\s+i\s+(prepare|plan|leave|stay)\b", re.IGNORECASE),
]


# =============================================================================
# SIGNAL EXTRACTION FUNCTIONS
# =============================================================================

def extract_query_signal(query: str) -> QueryAnalysisSignal:
    """Extract signal from query text.

    Performs fast regex and keyword matching to determine query characteristics.
    Designed to complete in < 1ms.

    Args:
        query: User's query text

    Returns:
        QueryAnalysisSignal with extracted features
    """
    query_lower = query.lower()
    words = query_lower.split()
    matched_patterns: list[str] = []

    # Check emergency patterns and keywords
    has_emergency = False
    for pattern in EMERGENCY_PATTERNS:
        if pattern.search(query):
            has_emergency = True
            matched_patterns.append(f"emergency_pattern:{pattern.pattern[:30]}")
            break

    if not has_emergency:
        for word in words:
            if word in EMERGENCY_KEYWORDS:
                has_emergency = True
                matched_patterns.append(f"emergency_keyword:{word}")
                break

    # Check complex patterns and keywords
    has_complex = False
    for pattern in COMPLEX_PATTERNS:
        if pattern.search(query):
            has_complex = True
            matched_patterns.append(f"complex_pattern:{pattern.pattern[:30]}")
            break

    if not has_complex:
        for word in words:
            if word in COMPLEX_KEYWORDS:
                has_complex = True
                matched_patterns.append(f"complex_keyword:{word}")
                break

    # Check storm patterns and keywords
    has_storm = False
    for pattern in STORM_PATTERNS:
        if pattern.search(query):
            has_storm = True
            matched_patterns.append(f"storm_pattern:{pattern.pattern[:30]}")
            break

    if not has_storm:
        for word in words:
            if word in STORM_KEYWORDS:
                has_storm = True
                matched_patterns.append(f"storm_keyword:{word}")
                break

    # Check for multiple questions
    question_marks = query.count("?")
    is_multi_question = question_marks >= 2

    # Check for conditional language
    has_conditional = any(p.search(query) for p in CONDITIONAL_PATTERNS)

    return QueryAnalysisSignal(
        has_emergency_keywords=has_emergency,
        has_complex_keywords=has_complex,
        has_storm_keywords=has_storm,
        is_multi_question=is_multi_question,
        word_count=len(words),
        has_conditional_language=has_conditional,
        matched_patterns=matched_patterns,
    )


def extract_context_signal(memory_context: dict[str, Any] | None) -> ContextSignal:
    """Extract signal from conversation memory context.

    Analyzes memory context to determine if this is a follow-up query
    and what tier was used previously.

    Args:
        memory_context: Memory context dict from MemoryManager (or None)

    Returns:
        ContextSignal with extracted context features
    """
    if not memory_context:
        return ContextSignal()

    # Extract previous conversation info
    conversation_history = memory_context.get("conversation_history", [])
    previous_queries = memory_context.get("previous_queries", [])

    # Determine if follow-up
    is_follow_up = len(conversation_history) > 0 or len(previous_queries) > 0

    # Check for previous tier (if tracked in memory)
    previous_tier = None
    last_routing = memory_context.get("last_routing_decision")
    if last_routing and isinstance(last_routing, dict):
        tier_value = last_routing.get("tier")
        if tier_value:
            try:
                previous_tier = QueryTier(tier_value)
            except ValueError:
                pass

    # Conversation depth
    conversation_depth = len(conversation_history)

    # Check for hurricane history in conversation
    has_hurricane_history = False
    for entry in conversation_history:
        if isinstance(entry, dict):
            query_text = entry.get("query", "").lower()
            if any(kw in query_text for kw in ["hurricane", "storm", "tropical", "evacuat"]):
                has_hurricane_history = True
                break

    # Also check previous queries
    for prev_query in previous_queries:
        if isinstance(prev_query, str):
            if any(kw in prev_query.lower() for kw in ["hurricane", "storm", "tropical", "evacuat"]):
                has_hurricane_history = True
                break

    return ContextSignal(
        is_follow_up=is_follow_up,
        previous_tier=previous_tier,
        conversation_depth=conversation_depth,
        has_hurricane_history=has_hurricane_history,
    )
