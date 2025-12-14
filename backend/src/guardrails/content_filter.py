"""
Content Filtering Module.

Level 6c: Self-Evolving Platform

Provides content filtering capabilities:
1. Harmful content detection
2. Category-based filtering
3. Configurable sensitivity
4. Safe content flagging

Target: Zero harmful content in production
"""

from typing import Any
import logging
import re
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FilterCategory(str, Enum):
    """Categories of content to filter."""

    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    MISINFORMATION = "misinformation"
    SPAM = "spam"
    PII = "pii"
    PROFANITY = "profanity"
    DANGEROUS_ADVICE = "dangerous_advice"


class ContentFilterResult(BaseModel):
    """Result of content filtering."""

    content: str
    is_safe: bool
    categories_flagged: list[FilterCategory] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
    filtered_content: str | None = None


class ContentFilter:
    """
    Content filtering for safety and compliance.

    Detects and filters harmful content across multiple categories.
    """

    # Patterns for different categories
    PATTERNS: dict[FilterCategory, list[str]] = {
        FilterCategory.HATE_SPEECH: [
            r"\b(hate|racist|sexist|homophobic)\b",
        ],
        FilterCategory.VIOLENCE: [
            r"\b(kill|murder|attack|assault|weapon)\b",
        ],
        FilterCategory.SELF_HARM: [
            r"\b(suicide|self.harm|hurt yourself)\b",
        ],
        FilterCategory.PII: [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
        ],
        FilterCategory.PROFANITY: [
            # Placeholder - actual implementation would use comprehensive list
            r"\b(damn|crap)\b",
        ],
        FilterCategory.DANGEROUS_ADVICE: [
            r"\b(don't evacuate|stay during hurricane|ignore warning)\b",
            r"\b(drive through flood|safe to stay)\b",
        ],
    }

    def __init__(
        self,
        enabled_categories: list[FilterCategory] | None = None,
        sensitivity: float = 0.5,
    ):
        """
        Initialize content filter.

        Args:
            enabled_categories: Categories to filter (defaults to all)
            sensitivity: Filtering sensitivity (0-1, higher = stricter)
        """
        self.enabled_categories = enabled_categories or list(FilterCategory)
        self.sensitivity = sensitivity

        # Compile patterns
        self.compiled_patterns: dict[FilterCategory, list[re.Pattern]] = {}
        for category in self.enabled_categories:
            if category in self.PATTERNS:
                self.compiled_patterns[category] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in self.PATTERNS[category]
                ]

        logger.info(
            f"ContentFilter initialized | categories={len(self.enabled_categories)} | "
            f"sensitivity={sensitivity}"
        )

    def filter(self, content: str) -> ContentFilterResult:
        """
        Filter content for harmful categories.

        Args:
            content: Content to filter

        Returns:
            ContentFilterResult with filtering status
        """
        flagged_categories: list[FilterCategory] = []
        details: dict[str, Any] = {}

        for category in self.enabled_categories:
            patterns = self.compiled_patterns.get(category, [])

            matches = []
            for pattern in patterns:
                found = pattern.findall(content)
                if found:
                    matches.extend(found)

            if matches:
                flagged_categories.append(category)
                details[category.value] = {
                    "matches": matches[:5],  # Limit to 5 matches
                    "count": len(matches),
                }

        # Calculate confidence based on matches and sensitivity
        if flagged_categories:
            match_count = sum(details.get(c.value, {}).get("count", 0) for c in flagged_categories)
            confidence = min(1.0, 0.5 + match_count * 0.1 + self.sensitivity * 0.3)
        else:
            confidence = 1.0 - self.sensitivity * 0.1

        is_safe = len(flagged_categories) == 0

        return ContentFilterResult(
            content=content,
            is_safe=is_safe,
            categories_flagged=flagged_categories,
            confidence=round(confidence, 4),
            details=details,
            filtered_content=self._redact_content(content, flagged_categories) if not is_safe else None,
        )

    def _redact_content(
        self,
        content: str,
        categories: list[FilterCategory],
    ) -> str:
        """Redact flagged content."""
        redacted = content

        for category in categories:
            patterns = self.compiled_patterns.get(category, [])
            for pattern in patterns:
                redacted = pattern.sub("[REDACTED]", redacted)

        return redacted

    def filter_pii(self, content: str) -> ContentFilterResult:
        """Filter specifically for PII."""
        original_categories = self.enabled_categories
        self.enabled_categories = [FilterCategory.PII]
        self.compiled_patterns = {
            FilterCategory.PII: [
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.PATTERNS[FilterCategory.PII]
            ]
        }

        result = self.filter(content)

        # Restore original categories
        self.enabled_categories = original_categories

        return result

    def filter_weather_safety(self, content: str) -> ContentFilterResult:
        """Filter for dangerous weather advice."""
        # Check for dangerous advice patterns
        dangerous_patterns = [
            r"don't evacuate",
            r"ignore.*warning",
            r"safe to stay.*hurricane",
            r"drive through.*flood",
            r"wait.*last minute",
            r"no need to.*evacuate",
        ]

        flagged = []
        details = {}

        for pattern in dangerous_patterns:
            compiled = re.compile(pattern, re.IGNORECASE)
            matches = compiled.findall(content)
            if matches:
                flagged.append(FilterCategory.DANGEROUS_ADVICE)
                details["dangerous_advice"] = {
                    "matches": matches,
                    "count": len(matches),
                }
                break

        return ContentFilterResult(
            content=content,
            is_safe=len(flagged) == 0,
            categories_flagged=flagged,
            confidence=0.95 if flagged else 1.0,
            details=details,
        )

    def add_pattern(self, category: FilterCategory, pattern: str) -> None:
        """Add a pattern to a category."""
        if category not in self.compiled_patterns:
            self.compiled_patterns[category] = []

        self.compiled_patterns[category].append(
            re.compile(pattern, re.IGNORECASE)
        )

        if category not in self.PATTERNS:
            self.PATTERNS[category] = []
        self.PATTERNS[category].append(pattern)

    def get_enabled_categories(self) -> list[str]:
        """Get enabled filter categories."""
        return [c.value for c in self.enabled_categories]

    def set_sensitivity(self, sensitivity: float) -> None:
        """Set filtering sensitivity."""
        self.sensitivity = max(0.0, min(1.0, sensitivity))
