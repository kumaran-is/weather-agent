"""
Hierarchical Loading for Context Optimization.

Phase 5 of the 5-phase context optimization pipeline.

Provides staged context loading strategy that prioritizes critical
information for immediate loading while deferring secondary content
for lazy loading when needed.

Key Features:
- Critical vs Secondary content classification
- Load level management (CRITICAL, SECONDARY, OPTIONAL)
- Lazy loading support for complex queries
- Token budget distribution across levels
"""

from typing import Any
import logging

logger = logging.getLogger(__name__)


class LoadLevel:
    """Load level constants for hierarchical loading."""

    CRITICAL = "CRITICAL"  # Must be loaded immediately
    SECONDARY = "SECONDARY"  # Load if budget allows
    OPTIONAL = "OPTIONAL"  # Lazy load only if needed


class HierarchicalLoader:
    """
    Phase 5: Hierarchical loading strategy.

    Prepares a load order for context based on importance levels,
    enabling efficient staged loading of context.
    """

    # Critical patterns - always load first
    CRITICAL_PATTERNS = [
        "You are",
        "SYSTEM:",
        "System:",
        "Instructions:",
        "IMPORTANT:",
        "WARNING:",
        "CRITICAL:",
        "EMERGENCY:",
        "User:",
        "Human:",
        "Query:",
        "Question:",
        "evacuation",
        "warning",
        "alert",
        "hurricane",
        "category",
    ]

    # Secondary patterns - load if budget allows
    SECONDARY_PATTERNS = [
        "Assistant:",
        "AI:",
        "Answer:",
        "Response:",
        "forecast",
        "temperature",
        "wind",
        "humidity",
        "pressure",
        "conditions",
    ]

    # Token budget distribution by query type
    BUDGET_DISTRIBUTION = {
        "SIMPLE": {"CRITICAL": 0.7, "SECONDARY": 0.3, "OPTIONAL": 0.0},
        "STANDARD": {"CRITICAL": 0.5, "SECONDARY": 0.4, "OPTIONAL": 0.1},
        "COMPLEX": {"CRITICAL": 0.4, "SECONDARY": 0.4, "OPTIONAL": 0.2},
        "EMERGENCY": {"CRITICAL": 0.6, "SECONDARY": 0.3, "OPTIONAL": 0.1},
    }

    def __init__(self, max_critical_lines: int = 30, max_secondary_lines: int = 50):
        """
        Initialize the hierarchical loader.

        Args:
            max_critical_lines: Maximum lines for critical content
            max_secondary_lines: Maximum lines for secondary content
        """
        self.max_critical_lines = max_critical_lines
        self.max_secondary_lines = max_secondary_lines

        logger.debug(
            f"HierarchicalLoader initialized | max_critical={max_critical_lines} | "
            f"max_secondary={max_secondary_lines}"
        )

    def prepare_load_order(
        self, context: str, query_type: str = "STANDARD"
    ) -> list[dict[str, Any]]:
        """
        Prepare hierarchical load order for context.

        Args:
            context: Full context string
            query_type: Type of query (SIMPLE, STANDARD, COMPLEX, EMERGENCY)

        Returns:
            List of load level dicts with level, content, and token estimate
        """
        if not context:
            return []

        query_type = query_type.upper() if query_type else "STANDARD"
        if query_type not in self.BUDGET_DISTRIBUTION:
            query_type = "STANDARD"

        lines = context.split("\n")
        load_order = []

        # Classify lines into load levels
        critical_lines = []
        secondary_lines = []
        optional_lines = []

        for line in lines:
            if not line.strip():
                continue

            level = self._classify_line(line)

            if level == LoadLevel.CRITICAL:
                critical_lines.append(line)
            elif level == LoadLevel.SECONDARY:
                secondary_lines.append(line)
            else:
                optional_lines.append(line)

        # Build load order with limits
        if critical_lines:
            critical_content = self._build_level_content(
                critical_lines, self.max_critical_lines
            )
            load_order.append(
                {
                    "level": LoadLevel.CRITICAL,
                    "content": critical_content,
                    "tokens": len(critical_content) // 4,
                    "line_count": len(critical_lines[: self.max_critical_lines]),
                    "priority": 1,
                }
            )

        if secondary_lines:
            # For SIMPLE queries, limit secondary content more aggressively
            max_lines = self.max_secondary_lines
            if query_type == "SIMPLE":
                max_lines = max_lines // 2

            secondary_content = self._build_level_content(secondary_lines, max_lines)
            load_order.append(
                {
                    "level": LoadLevel.SECONDARY,
                    "content": secondary_content,
                    "tokens": len(secondary_content) // 4,
                    "line_count": len(secondary_lines[:max_lines]),
                    "priority": 2,
                }
            )

        # Only include optional for COMPLEX queries
        if optional_lines and query_type in ["COMPLEX", "EMERGENCY"]:
            optional_content = self._build_level_content(optional_lines, 20)
            load_order.append(
                {
                    "level": LoadLevel.OPTIONAL,
                    "content": optional_content,
                    "tokens": len(optional_content) // 4,
                    "line_count": len(optional_lines[:20]),
                    "priority": 3,
                }
            )

        logger.debug(
            f"Load order prepared | query_type={query_type} | "
            f"levels={len(load_order)} | "
            f"total_tokens={sum(l['tokens'] for l in load_order)}"
        )

        return load_order

    def _classify_line(self, line: str) -> str:
        """Classify a line into a load level."""
        # Check for critical patterns
        for pattern in self.CRITICAL_PATTERNS:
            if pattern.lower() in line.lower():
                return LoadLevel.CRITICAL

        # Check for secondary patterns
        for pattern in self.SECONDARY_PATTERNS:
            if pattern.lower() in line.lower():
                return LoadLevel.SECONDARY

        # Check for structural markers
        if line.strip().startswith("#"):
            return LoadLevel.SECONDARY
        if line.strip().startswith("{") or line.strip().startswith("["):
            return LoadLevel.SECONDARY
        if "|" in line and line.count("|") >= 2:
            return LoadLevel.SECONDARY

        return LoadLevel.OPTIONAL

    def _build_level_content(
        self, lines: list[str], max_lines: int
    ) -> str:
        """Build content string from lines with limit."""
        limited_lines = lines[:max_lines]
        return "\n".join(limited_lines)

    def get_critical_only(self, context: str) -> str:
        """
        Get only critical content from context.

        Useful for simple queries or when token budget is tight.
        """
        load_order = self.prepare_load_order(context, "SIMPLE")

        for level_info in load_order:
            if level_info["level"] == LoadLevel.CRITICAL:
                return level_info["content"]

        return ""

    def get_up_to_level(
        self, context: str, max_level: str, query_type: str = "STANDARD"
    ) -> str:
        """
        Get content up to a specified level.

        Args:
            context: Full context string
            max_level: Maximum level to include (CRITICAL, SECONDARY, OPTIONAL)
            query_type: Query type for budget distribution

        Returns:
            Combined content up to and including the specified level
        """
        load_order = self.prepare_load_order(context, query_type)

        level_priority = {
            LoadLevel.CRITICAL: 1,
            LoadLevel.SECONDARY: 2,
            LoadLevel.OPTIONAL: 3,
        }

        max_priority = level_priority.get(max_level, 2)

        content_parts = []
        for level_info in load_order:
            if level_priority.get(level_info["level"], 3) <= max_priority:
                content_parts.append(level_info["content"])

        return "\n\n".join(content_parts)

    def lazy_load_check(
        self,
        current_tokens: int,
        token_budget: int,
        query_type: str,
    ) -> dict[str, Any]:
        """
        Check if lazy loading is needed and what level to load.

        Args:
            current_tokens: Current token count in context
            token_budget: Total token budget
            query_type: Query type

        Returns:
            Dict with should_load, next_level, and available_tokens
        """
        budget_dist = self.BUDGET_DISTRIBUTION.get(
            query_type.upper(), self.BUDGET_DISTRIBUTION["STANDARD"]
        )

        used_ratio = current_tokens / max(token_budget, 1)

        # Determine what level we're at based on usage
        if used_ratio < budget_dist["CRITICAL"]:
            next_level = LoadLevel.SECONDARY
            available = int(
                (budget_dist["CRITICAL"] + budget_dist["SECONDARY"]) * token_budget
                - current_tokens
            )
        elif used_ratio < budget_dist["CRITICAL"] + budget_dist["SECONDARY"]:
            next_level = LoadLevel.OPTIONAL
            available = int(token_budget - current_tokens)
        else:
            next_level = None
            available = 0

        should_load = next_level is not None and available > 100

        return {
            "should_load": should_load,
            "next_level": next_level,
            "available_tokens": available,
            "current_usage_ratio": round(used_ratio, 2),
        }

    def get_load_statistics(
        self, context: str, query_type: str = "STANDARD"
    ) -> dict[str, Any]:
        """
        Get statistics about the load order.

        Useful for debugging and optimization.
        """
        load_order = self.prepare_load_order(context, query_type)

        total_tokens = sum(l["tokens"] for l in load_order)
        level_breakdown = {
            l["level"]: {"tokens": l["tokens"], "lines": l["line_count"]}
            for l in load_order
        }

        return {
            "query_type": query_type,
            "total_tokens": total_tokens,
            "levels": len(load_order),
            "breakdown": level_breakdown,
            "budget_distribution": self.BUDGET_DISTRIBUTION.get(
                query_type.upper(), self.BUDGET_DISTRIBUTION["STANDARD"]
            ),
        }
