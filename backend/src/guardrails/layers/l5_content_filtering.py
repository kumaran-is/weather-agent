"""Layer 5: Content Filtering.

Filters prohibited, harmful, or inappropriate content from
both input queries and output responses.

Categories Filtered:
- Hate speech and discrimination
- Violence and threats
- Adult/explicit content
- Illegal activities
- Misinformation topics
- Off-topic requests (non-weather)

Weather AI Agent Focus:
- Blocks non-weather queries
- Filters weather-related misinformation
- Prevents misuse for harmful purposes
"""

import re
from typing import Any

from backend.src.guardrails.layers.base import BaseGuardrailLayer
from backend.src.guardrails.models import (
    GuardrailLayer,
    GuardrailViolation,
    ViolationSeverity,
)


class ContentFilteringLayer(BaseGuardrailLayer):
    """Layer 5: Content Filtering.

    Blocks prohibited content and off-topic requests.
    """

    layer = GuardrailLayer.L5_CONTENT_FILTERING

    # Weather-related keywords (topic validation)
    WEATHER_KEYWORDS = [
        "weather", "temperature", "forecast", "rain", "snow", "wind",
        "humidity", "storm", "hurricane", "tornado", "typhoon", "cyclone",
        "climate", "sunny", "cloudy", "fog", "precipitation", "pressure",
        "heat", "cold", "freeze", "flood", "drought", "lightning",
        "thunder", "hail", "blizzard", "celsius", "fahrenheit",
        "high", "low", "alert", "warning", "watch", "advisory",
        "evacuate", "evacuation", "shelter", "emergency",
    ]

    # Prohibited content patterns (high severity)
    PROHIBITED_PATTERNS = [
        # Violence/threats
        (r"how\s+to\s+(?:kill|murder|harm|attack)", "violence"),
        (r"(?:bomb|explosive|weapon)\s+(?:instructions?|how\s+to)", "weapons"),
        # Hate speech indicators
        (r"(?:all|those)\s+\w+\s+(?:should\s+die|deserve\s+to)", "hate_speech"),
        # Illegal activities
        (r"how\s+to\s+(?:hack|steal|fraud)", "illegal"),
        # Explicit content
        (r"(?:porn|xxx|nsfw|explicit)", "adult_content"),
    ]

    # Off-topic patterns (medium severity)
    OFF_TOPIC_PATTERNS = [
        (r"write\s+(?:me\s+)?(?:a\s+)?(?:poem|story|essay|code)", "creative_writing"),
        (r"(?:help\s+me\s+)?(?:with\s+)?(?:homework|assignment|exam)", "academic"),
        (r"(?:stock|crypto|bitcoin|investment)\s+(?:advice|price)", "financial"),
        (r"(?:medical|health|symptom|diagnosis)\s+(?:advice|help)", "medical"),
        (r"(?:legal|lawyer|lawsuit|sue)\s+(?:advice|help)", "legal"),
        (r"(?:recipe|cook|bake|food)\s+(?:for|how)", "cooking"),
    ]

    # Misinformation indicators for weather
    MISINFORMATION_PATTERNS = [
        (r"climate\s+change\s+(?:is\s+)?(?:fake|hoax|scam)", "climate_denial"),
        (r"(?:chemtrail|weather\s+control|haarp)", "conspiracy"),
        (r"(?:government|they)\s+(?:control|manipulate)\s+weather", "conspiracy"),
    ]

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Filter prohibited and off-topic content.

        Args:
            content: Text to filter
            context: May contain 'check_type' (input/output)

        Returns:
            List of violations found
        """
        violations: list[GuardrailViolation] = []
        context = context or {}
        check_type = context.get("check_type", "input")
        content_lower = content.lower()

        # Check prohibited content (always block)
        for pattern, category in self.PROHIBITED_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.CRITICAL,
                        message=f"Prohibited content detected: {category}",
                        details={
                            "category": category,
                            "check_type": check_type,
                        },
                        remediation="Remove prohibited content",
                    )
                )

        # Check topic relevance (input only)
        if check_type == "input":
            if not self._is_weather_related(content_lower):
                # Check if it's a known off-topic category
                off_topic_category = None
                for pattern, category in self.OFF_TOPIC_PATTERNS:
                    if re.search(pattern, content_lower, re.IGNORECASE):
                        off_topic_category = category
                        break

                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.MEDIUM,
                        message="Query appears to be off-topic (not weather-related)",
                        details={
                            "detected_topic": off_topic_category or "unknown",
                            "expected_topic": "weather",
                        },
                        remediation="Please ask a weather-related question",
                    )
                )

        # Check misinformation patterns
        for pattern, category in self.MISINFORMATION_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.HIGH,
                        message=f"Potential misinformation topic: {category}",
                        details={
                            "category": category,
                            "check_type": check_type,
                        },
                        remediation="Use verified weather information sources",
                    )
                )

        # Check for excessive capitalization (shouting/spam)
        if self._is_excessive_caps(content):
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.LOW,
                    message="Excessive capitalization detected",
                    details={"caps_ratio": self._caps_ratio(content)},
                    remediation="Use normal capitalization",
                )
            )

        return violations

    def _is_weather_related(self, content: str) -> bool:
        """Check if content is weather-related.

        Args:
            content: Lowercase text to check

        Returns:
            True if weather-related
        """
        # Check for weather keywords
        for keyword in self.WEATHER_KEYWORDS:
            if keyword in content:
                return True

        # Check for location + weather pattern
        location_weather = re.search(
            r"(?:in|at|near|around)\s+\w+.*(?:weather|temperature|forecast)",
            content,
            re.IGNORECASE,
        )
        if location_weather:
            return True

        # Check for "what's the" + weather terms
        whats_pattern = re.search(
            r"what(?:'s|is)\s+(?:the\s+)?(?:weather|temperature|forecast|humidity)",
            content,
            re.IGNORECASE,
        )
        if whats_pattern:
            return True

        return False

    def _is_excessive_caps(self, content: str, threshold: float = 0.7) -> bool:
        """Check for excessive capitalization."""
        letters = [c for c in content if c.isalpha()]
        if len(letters) < 10:
            return False
        caps_count = sum(1 for c in letters if c.isupper())
        return caps_count / len(letters) > threshold

    def _caps_ratio(self, content: str) -> float:
        """Calculate capitalization ratio."""
        letters = [c for c in content if c.isalpha()]
        if not letters:
            return 0.0
        caps_count = sum(1 for c in letters if c.isupper())
        return caps_count / len(letters)
