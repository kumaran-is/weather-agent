"""Layer 8: Output Validation.

Validates agent output quality, format, and completeness
before returning to users.

Validation Checks:
1. Response length (not too short or too long)
2. Completeness (answers the question)
3. Format quality (readable structure)
4. Citation presence (for factual claims)
5. Actionability (for safety queries)
6. Time specificity (exact times, not vague)

Weather AI Focus:
- Must include specific times (EDT/UTC)
- Must include actionable guidance for alerts
- Must not use vague terms ("soon", "later")
"""

import re
from typing import Any

from backend.src.guardrails.layers.base import BaseGuardrailLayer
from backend.src.guardrails.models import (
    GuardrailConfig,
    GuardrailLayer,
    GuardrailViolation,
    ViolationSeverity,
)


class OutputValidationLayer(BaseGuardrailLayer):
    """Layer 8: Output Validation.

    Ensures output meets quality and completeness standards.
    """

    layer = GuardrailLayer.L8_OUTPUT_VALIDATION

    # Minimum response length (characters)
    MIN_RESPONSE_LENGTH = 20

    # Vague time terms to flag
    VAGUE_TIME_TERMS = [
        r"\b(?:soon|shortly|later|eventually)\b",
        r"\b(?:in\s+a\s+(?:bit|while|moment))\b",
        r"\b(?:sometime|around)\s+(?:today|tomorrow|this\s+week)\b",
    ]

    # Required elements for hurricane responses
    HURRICANE_REQUIRED = [
        ("category", r"category\s*\d"),
        ("wind_speed", r"\d+\s*(?:mph|km/h|knots)"),
        ("action", r"(?:evacuate|shelter|prepare|monitor|stay)"),
    ]

    # Required elements for forecast responses
    FORECAST_REQUIRED = [
        ("temperature", r"\d+°?\s*[fFcC]"),
        ("condition", r"(?:sunny|cloudy|rain|snow|clear|overcast|partly|mostly)"),
    ]

    # Patterns indicating incomplete response
    INCOMPLETE_PATTERNS = [
        r"i\s+(?:don't|do\s+not)\s+have\s+(?:enough|that)\s+information",
        r"(?:cannot|can't)\s+(?:answer|respond|help)",
        r"(?:i'm\s+)?(?:sorry|unable)\s+to\s+(?:help|assist|provide)",
        r"please\s+(?:try|ask)\s+again",
        r"error\s+(?:occurred|processing)",
    ]

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize with configuration."""
        super().__init__(config)
        self.max_length = self.config.max_output_length

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Validate output quality.

        Args:
            content: Agent output to validate
            context: Should contain 'query' and optionally 'query_type'

        Returns:
            List of validation violations
        """
        violations: list[GuardrailViolation] = []
        context = context or {}
        query = context.get("query", "")
        query_type = context.get("query_type", self._detect_query_type(query))

        # Check length constraints
        length_violations = self._check_length(content)
        violations.extend(length_violations)

        # Check completeness
        completeness_violations = self._check_completeness(content, query_type)
        violations.extend(completeness_violations)

        # Check for incomplete/error responses
        error_violations = self._check_error_responses(content)
        violations.extend(error_violations)

        # Check time specificity (critical for weather)
        time_violations = self._check_time_specificity(content, query_type)
        violations.extend(time_violations)

        # Check actionability for safety queries
        if query_type in ["hurricane", "alert", "emergency"]:
            action_violations = self._check_actionability(content)
            violations.extend(action_violations)

        # Check hurricane category validation (CRITICAL - life-safety)
        if query_type == "hurricane":
            category_violations = self._check_hurricane_category(content)
            violations.extend(category_violations)

        # Check format quality
        format_violations = self._check_format_quality(content)
        violations.extend(format_violations)

        return violations

    def _check_length(self, content: str) -> list[GuardrailViolation]:
        """Check response length constraints."""
        violations = []

        if len(content) < self.MIN_RESPONSE_LENGTH:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.MEDIUM,
                    message=f"Response too short ({len(content)} chars, min {self.MIN_RESPONSE_LENGTH})",
                    details={
                        "length": len(content),
                        "min_required": self.MIN_RESPONSE_LENGTH,
                    },
                    remediation="Provide more detailed response",
                )
            )

        if len(content) > self.max_length:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.LOW,
                    message=f"Response exceeds max length ({len(content)} > {self.max_length})",
                    details={
                        "length": len(content),
                        "max_allowed": self.max_length,
                    },
                    remediation="Summarize response",
                )
            )

        return violations

    def _check_completeness(
        self,
        content: str,
        query_type: str,
    ) -> list[GuardrailViolation]:
        """Check if response includes required elements."""
        violations = []
        content_lower = content.lower()

        # Select required elements based on query type
        required_elements = []
        if query_type == "hurricane":
            required_elements = self.HURRICANE_REQUIRED
        elif query_type == "forecast":
            required_elements = self.FORECAST_REQUIRED

        for element_name, pattern in required_elements:
            if not re.search(pattern, content_lower, re.IGNORECASE):
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.MEDIUM,
                        message=f"Missing required element: {element_name}",
                        details={
                            "missing_element": element_name,
                            "query_type": query_type,
                        },
                        remediation=f"Include {element_name} in response",
                    )
                )

        return violations

    def _check_error_responses(self, content: str) -> list[GuardrailViolation]:
        """Check for incomplete or error responses."""
        violations = []
        content_lower = content.lower()

        for pattern in self.INCOMPLETE_PATTERNS:
            if re.search(pattern, content_lower):
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.HIGH,
                        message="Response indicates inability to answer",
                        details={"pattern": pattern[:40]},
                        remediation="Provide available information or clear guidance",
                    )
                )

        return violations

    def _check_time_specificity(
        self,
        content: str,
        query_type: str,
    ) -> list[GuardrailViolation]:
        """Check for vague time references."""
        violations = []

        # Only critical for time-sensitive query types
        time_sensitive_types = ["hurricane", "alert", "emergency", "forecast"]
        if query_type not in time_sensitive_types:
            return []

        for pattern in self.VAGUE_TIME_TERMS:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.HIGH,
                        message="Vague time reference in time-sensitive response",
                        details={
                            "pattern": pattern[:30],
                            "query_type": query_type,
                        },
                        remediation="Use specific times (e.g., '2:00 PM EDT', 'Tuesday morning')",
                    )
                )

        # Check for timezone specification in time-sensitive content
        if query_type in ["hurricane", "alert"]:
            has_times = re.search(r"\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)", content, re.IGNORECASE)
            has_timezone = re.search(r"(?:EDT|EST|CDT|CST|PDT|PST|UTC|GMT)", content)

            if has_times and not has_timezone:
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.MEDIUM,
                        message="Times mentioned without timezone specification",
                        details={"has_times": True, "has_timezone": False},
                        remediation="Include timezone (e.g., EDT, UTC) with all times",
                    )
                )

        return violations

    def _check_actionability(self, content: str) -> list[GuardrailViolation]:
        """Check for actionable guidance in safety responses."""
        violations = []

        # Action words that should be present
        action_patterns = [
            r"(?:should|need\s+to|must|recommend|advised\s+to)",
            r"(?:evacuate|shelter|prepare|monitor|contact|call)",
            r"(?:if\s+you\s+are\s+in|residents\s+(?:of|in))",
        ]

        has_actionable = any(
            re.search(p, content, re.IGNORECASE) for p in action_patterns
        )

        if not has_actionable:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.HIGH,
                    message="Safety response lacks actionable guidance",
                    details={"patterns_checked": len(action_patterns)},
                    remediation="Include specific actions users should take",
                )
            )

        return violations

    def _check_hurricane_category(self, content: str) -> list[GuardrailViolation]:
        """Validate hurricane category matches wind speed (Saffir-Simpson scale).

        CRITICAL SAFETY VALIDATION - ZERO TOLERANCE FOR ERRORS

        Saffir-Simpson Scale:
        - Category 1: 74-95 mph
        - Category 2: 96-110 mph
        - Category 3: 111-129 mph
        - Category 4: 130-156 mph
        - Category 5: 157+ mph
        """
        violations = []

        # Step 1: Remove educational/definitional patterns to avoid false positives
        # These patterns are used when explaining the scale, not asserting a storm's category
        educational_patterns = [
            # Standard formats with optional bold (**): Category X (Y-Z mph), (Y+ mph), (Y mph and higher)
            r"\*{0,2}category\s*\d\s*\([^)]*\d+\s*(?:mph\s+and\s+higher|[-+]\d*\s*mph|mph)[^)]*\)\*{0,2}",
            # Colon format: **Category X**: (Y mph) or **Category X: (Y mph)**
            r"\*{0,2}category\s*\d\s*:\*{0,2}\s*\([^)]*\d+\s*mph[^)]*\)\*{0,2}",
            # Requirement format: Category X requires Y-Z mph
            r"category\s*\d\s+requires\s+[\d\-]+\s*mph",
            # List item format: "1. **Category X (Y-Z mph)**"
            r"\d+\.\s+\*{0,2}category\s*\d\s*\([^)]*\d+\s*mph[^)]*\)\*{0,2}",
        ]

        cleaned_content = content
        for edu_pattern in educational_patterns:
            cleaned_content = re.sub(edu_pattern, "", cleaned_content, flags=re.IGNORECASE)

        # Step 2: Find ALL category-wind speed pairs in proximity (same sentence/clause)
        # Pattern: "Category X ... Y mph" within ~200 characters
        # This ensures we match contextually related pairs, not random first matches
        pattern = r"category\s*(\d)[^.!?]{0,200}?(\d+)\s*mph"
        matches = list(re.finditer(pattern, cleaned_content, re.IGNORECASE))

        # If no proximity matches found, fall back to checking if there's a single
        # category and single wind speed in the entire content
        if not matches:
            category_match = re.search(r"category\s*(\d)", content, re.IGNORECASE)
            wind_match = re.search(r"(\d+)\s*mph", content, re.IGNORECASE)

            if category_match and wind_match:
                matches = [(category_match.group(1), wind_match.group(1))]
            else:
                return violations

        # Validate each category-wind pair found
        for match in matches:
            if isinstance(match, tuple):
                category, wind_speed = int(match[0]), int(match[1])
            else:
                category, wind_speed = int(match.group(1)), int(match.group(2))

            # Validate category matches wind speed (Saffir-Simpson scale)
            correct_category = self._get_saffir_simpson_category(wind_speed)

            if category != correct_category:
                # Get expected range for stated category
                category_ranges = {
                    1: "74-95 mph",
                    2: "96-110 mph",
                    3: "111-129 mph",
                    4: "130-156 mph",
                    5: "157+ mph"
                }

                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.CRITICAL,
                        message=f"Hurricane category error: Category {category} requires {category_ranges.get(category, 'invalid')}, but {wind_speed} mph indicates Category {correct_category}",
                        details={
                            "stated_category": category,
                            "wind_speed_mph": wind_speed,
                            "correct_category": correct_category,
                            "violation_type": "hurricane_category_error",
                        },
                        remediation=f"Change to Category {correct_category} or verify wind speed accuracy",
                    )
                )

        return violations

    def _get_saffir_simpson_category(self, wind_speed_mph: int) -> int:
        """Determine correct hurricane category from wind speed.

        Args:
            wind_speed_mph: Wind speed in miles per hour

        Returns:
            Correct category (1-5)
        """
        if wind_speed_mph >= 157:
            return 5
        elif wind_speed_mph >= 130:
            return 4
        elif wind_speed_mph >= 111:
            return 3
        elif wind_speed_mph >= 96:
            return 2
        elif wind_speed_mph >= 74:
            return 1
        else:
            # Below hurricane threshold (tropical storm)
            return 0

    def _check_format_quality(self, content: str) -> list[GuardrailViolation]:
        """Check basic formatting quality."""
        violations = []

        # Check for reasonable sentence structure
        sentences = re.split(r"[.!?]+", content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(valid_sentences) == 0 and len(content) > 50:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.LOW,
                    message="Response lacks proper sentence structure",
                    details={"content_length": len(content)},
                    remediation="Format response with complete sentences",
                )
            )

        # Check for excessive punctuation
        if re.search(r"[!?]{3,}", content):
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.LOW,
                    message="Excessive punctuation in response",
                    details={"pattern": "multiple_punctuation"},
                    remediation="Use single punctuation marks",
                )
            )

        return violations

    def _detect_query_type(self, query: str) -> str:
        """Detect query type from content."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["hurricane", "storm", "cyclone", "evacuate"]):
            return "hurricane"
        elif any(kw in query_lower for kw in ["alert", "warning", "watch", "emergency"]):
            return "alert"
        elif any(kw in query_lower for kw in ["forecast", "tomorrow", "next", "week"]):
            return "forecast"
        else:
            return "general"
