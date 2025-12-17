"""Query Type Detection for Context Optimization.

Level 8a: Auto-detect query type based on content analysis for adaptive
context window optimization.

Query Types:
- EMERGENCY: Life-safety queries (evacuation, hurricane, warning)
  → Preserves more context (30-40% reduction, safety info prioritized)
- COMPLEX: Multi-part queries (compare, plan, trip, analyze)
  → Balanced optimization (50-60% reduction)
- SIMPLE: Single-point queries (temperature, wind)
  → Aggressive optimization (60-70% reduction)
- STANDARD: Default for normal weather queries
  → Standard optimization (50-60% reduction)

Usage:
    >>> from backend.src.context import detect_query_type
    >>> query_type = detect_query_type("Should I evacuate for the hurricane?")
    >>> print(query_type)  # "EMERGENCY"

    >>> query_type = detect_query_type("What's the temperature in Miami?")
    >>> print(query_type)  # "SIMPLE"
"""

from __future__ import annotations

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

# Type alias for query types
QueryType = Literal["EMERGENCY", "COMPLEX", "SIMPLE", "STANDARD"]


# Emergency keywords - life-safety critical
EMERGENCY_KEYWORDS: list[str] = [
    # Evacuation
    "evacuate",
    "evacuation",
    "mandatory evacuation",
    "should i leave",
    "is it safe",
    "safe to stay",
    # Hurricane severity
    "hurricane",
    "tornado",
    "category 4",
    "category 5",
    "cat 4",
    "cat 5",
    "major hurricane",
    # Warnings/Danger
    "warning",
    "emergency",
    "danger",
    "dangerous",
    "urgent",
    "critical",
    "life-threatening",
    "life threatening",
    # Shelter/Safety
    "shelter",
    "safe zone",
    "storm surge",
    "flooding",
    "flash flood",
    # Extreme conditions
    "extreme weather",
    "severe weather",
    "emergency alert",
]

# Complex query patterns (regex)
COMPLEX_PATTERNS: list[str] = [
    # Comparisons
    r"\bcompare\b",
    r"\bvs\b",
    r"\bversus\b",
    r"\bdifference\b",
    r"\bbetter\b.*\bor\b",
    # Planning
    r"\bplan\b",
    r"\bplanning\b",
    r"\btrip\b",
    r"\btravel\b",
    r"\bjourney\b",
    r"\bitinerary\b",
    r"\bvacation\b",
    # Analysis
    r"\banalyze\b",
    r"\banalysis\b",
    r"\btrend\b",
    r"\btrends\b",
    r"\bpattern\b",
    r"\bhistorical\b",
    # Multi-day
    r"\bnext \d+ days\b",
    r"\b\d+ day forecast\b",
    r"\bweekend\b",
    r"\bweek\b",
    r"\bmonth\b",
    # Multiple items
    r"\bmultiple\b",
    r"\bseveral\b",
    r"\bboth\b",
    r"\band\b.*\band\b",  # Multiple conditions (X and Y and Z)
    # Multi-location
    r"\bcities\b",
    r"\blocations\b",
    r"\bareas\b",
]

# Simple query patterns (regex) - single data point requests
SIMPLE_PATTERNS: list[str] = [
    # Temperature queries
    r"^what('s| is) the (temperature|temp)\b",
    r"^(current|today's?) (temperature|temp)\b",
    r"^how (hot|cold|warm|cool)\b",
    # Simple weather state
    r"^(is it|will it) (rain|raining|snow|snowing|sunny|cloudy)\b",
    r"^(is it|will it) (hot|cold|warm|cool)\b",
    # Basic queries
    r"^what('s| is) the weather\b",
    r"^(current|today's?) weather\b",
    # Single metrics
    r"^(what is|what's|how much) (humidity|wind|pressure|uv)\b",
]


def detect_query_type(query: str) -> QueryType:
    """Auto-detect query type based on content analysis.

    Args:
        query: User's weather query string

    Returns:
        Query type: "EMERGENCY" | "COMPLEX" | "SIMPLE" | "STANDARD"

    Priority Order:
        1. EMERGENCY (highest): Life-safety keywords override all
        2. COMPLEX: Multi-part analysis, comparisons, planning
        3. SIMPLE: Single data point requests
        4. STANDARD: Default for everything else

    Examples:
        >>> detect_query_type("Should I evacuate for Category 5?")
        'EMERGENCY'

        >>> detect_query_type("Compare Miami vs Tampa weather")
        'COMPLEX'

        >>> detect_query_type("What's the temperature?")
        'SIMPLE'

        >>> detect_query_type("What's the forecast for tomorrow?")
        'STANDARD'
    """
    if not query or not query.strip():
        logger.warning("Empty query provided, defaulting to STANDARD")
        return "STANDARD"

    query_lower = query.lower().strip()

    # Priority 1: EMERGENCY - Life-safety queries (highest priority)
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in query_lower:
            logger.info(
                f"🚨 Query type: EMERGENCY | "
                f"keyword='{keyword}' | "
                f"query='{query[:50]}...'"
            )
            return "EMERGENCY"

    # Priority 2: COMPLEX - Multi-part analysis
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, query_lower):
            logger.info(
                f"📊 Query type: COMPLEX | "
                f"pattern='{pattern}' | "
                f"query='{query[:50]}...'"
            )
            return "COMPLEX"

    # Priority 3: SIMPLE - Single data point
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, query_lower):
            logger.info(
                f"📍 Query type: SIMPLE | "
                f"pattern='{pattern}' | "
                f"query='{query[:50]}...'"
            )
            return "SIMPLE"

    # Default: STANDARD
    logger.info(f"📋 Query type: STANDARD | query='{query[:50]}...'")
    return "STANDARD"


def get_optimization_config(query_type: QueryType) -> dict:
    """Get optimization configuration for a query type.

    Returns recommended optimization parameters based on query type:
    - target_tokens: Target token count after optimization
    - min_relevance: Minimum relevance score for chunks
    - preserve_safety: Whether to prioritize safety information

    Args:
        query_type: The detected query type

    Returns:
        Dict with optimization configuration

    Example:
        >>> config = get_optimization_config("EMERGENCY")
        >>> print(config)
        {
            "target_tokens": 6000,
            "min_relevance": 0.3,
            "preserve_safety": True,
            "expected_reduction_pct": "30-40%"
        }
    """
    configs = {
        "EMERGENCY": {
            "target_tokens": 6000,  # Higher limit - preserve more
            "min_relevance": 0.3,  # Lower threshold - keep more context
            "preserve_safety": True,
            "expected_reduction_pct": "30-40%",
        },
        "COMPLEX": {
            "target_tokens": 4500,  # Moderate limit
            "min_relevance": 0.5,  # Standard threshold
            "preserve_safety": False,
            "expected_reduction_pct": "50-55%",
        },
        "STANDARD": {
            "target_tokens": 4000,  # Standard limit
            "min_relevance": 0.5,  # Standard threshold
            "preserve_safety": False,
            "expected_reduction_pct": "50-60%",
        },
        "SIMPLE": {
            "target_tokens": 2500,  # Lower limit - aggressive optimization
            "min_relevance": 0.6,  # Higher threshold - keep only relevant
            "preserve_safety": False,
            "expected_reduction_pct": "60-70%",
        },
    }

    return configs.get(query_type, configs["STANDARD"])
