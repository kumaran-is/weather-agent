"""Layer 7: Bias Mitigation.

Detects and mitigates bias in weather-related responses,
ensuring fair and equitable information delivery.

Bias Types Detected:
1. Geographic bias (favoring certain regions)
2. Socioeconomic bias (assumptions about wealth)
3. Demographic bias (assumptions about people)
4. Accessibility bias (assuming capabilities)
5. Language bias (discriminatory language)

Weather AI Focus:
- Equal evacuation urgency regardless of area wealth
- No assumptions about resources or mobility
- Consistent safety advice across all demographics
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


class BiasMitigationLayer(BaseGuardrailLayer):
    """Layer 7: Bias Mitigation.

    Detects potentially biased language or recommendations.
    """

    layer = GuardrailLayer.L7_BIAS_MITIGATION

    # Socioeconomic bias patterns
    SOCIOECONOMIC_PATTERNS = [
        (r"(?:poor|low-income|disadvantaged)\s+(?:neighborhood|area|community)", "socioeconomic_labeling"),
        (r"(?:wealthy|affluent|rich)\s+(?:neighborhood|area|community)", "socioeconomic_labeling"),
        (r"(?:those|people)\s+(?:can|should)\s+afford", "wealth_assumption"),
        (r"if\s+you\s+(?:can|cannot)\s+afford", "wealth_assumption"),
        (r"(?:people\s+)?in\s+(?:nice|bad|good|rough)\s+(?:areas?|neighborhoods?)", "area_judgment"),
    ]

    # Demographic bias patterns
    DEMOGRAPHIC_PATTERNS = [
        (r"(?:elderly|old)\s+people\s+(?:should|need\s+to|must)", "age_generalization"),
        (r"(?:young|younger)\s+people\s+(?:don't|won't|can)", "age_generalization"),
        (r"(?:men|women)\s+(?:typically|usually|often)\s+(?:should|need)", "gender_generalization"),
        (r"(?:as\s+a|being\s+a)\s+(?:man|woman|male|female)", "gender_assumption"),
    ]

    # Geographic preference patterns
    GEOGRAPHIC_PATTERNS = [
        (r"(?:better|more\s+important)\s+(?:to\s+)?(?:protect|evacuate|warn)\s+\w+\s+(?:than|over)", "geographic_preference"),
        (r"(?:priority|prioritize)\s+(?:should\s+be\s+)?(?:given\s+to\s+)?\w+\s+(?:area|region|city)", "geographic_preference"),
    ]

    # Victim-blaming patterns
    VICTIM_BLAMING_PATTERNS = [
        (r"(?:should\s+have|could\s+have)\s+(?:known|prepared|left)\s+(?:better|earlier)", "victim_blaming"),
        (r"(?:their|your)\s+(?:own\s+)?fault", "victim_blaming"),
        (r"(?:people|they|residents)\s+(?:should|need\s+to)\s+know\s+better", "victim_blaming"),
        (r"(?:typical|expected)\s+for\s+(?:that|this|those)\s+(?:area|people|region)", "stereotyping"),
    ]

    # Accessibility bias patterns
    ACCESSIBILITY_PATTERNS = [
        (r"(?:just|simply)\s+(?:drive|walk|run)\s+(?:away|out|to)", "mobility_assumption"),
        (r"(?:everyone|all)\s+(?:can|should)\s+(?:easily|quickly)\s+(?:evacuate|leave)", "capability_assumption"),
        (r"(?:call|use|access)\s+(?:the\s+)?internet", "technology_assumption"),
    ]

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Detect bias in content.

        Args:
            content: Text to check for bias
            context: Additional context (unused)

        Returns:
            List of bias violations found
        """
        violations: list[GuardrailViolation] = []
        content_lower = content.lower()

        # Check all bias categories
        pattern_categories = [
            (self.SOCIOECONOMIC_PATTERNS, "Socioeconomic bias", ViolationSeverity.HIGH),
            (self.DEMOGRAPHIC_PATTERNS, "Demographic bias", ViolationSeverity.HIGH),
            (self.GEOGRAPHIC_PATTERNS, "Geographic bias", ViolationSeverity.MEDIUM),
            (self.VICTIM_BLAMING_PATTERNS, "Victim-blaming language", ViolationSeverity.HIGH),
            (self.ACCESSIBILITY_PATTERNS, "Accessibility bias", ViolationSeverity.MEDIUM),
        ]

        for patterns, category_name, severity in pattern_categories:
            for pattern, bias_type in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    violations.append(
                        self.create_violation(
                            severity=severity,
                            message=f"{category_name} detected: {bias_type}",
                            details={
                                "category": category_name,
                                "bias_type": bias_type,
                                "pattern": pattern[:50],
                            },
                            remediation=self._get_remediation(bias_type),
                        )
                    )

        # Check for unequal urgency in evacuation advice
        urgency_violations = self._check_evacuation_equity(content)
        violations.extend(urgency_violations)

        # Check for inclusive language
        inclusivity_violations = self._check_inclusivity(content)
        violations.extend(inclusivity_violations)

        return violations

    def _check_evacuation_equity(self, content: str) -> list[GuardrailViolation]:
        """Ensure evacuation advice is equitable."""
        violations = []

        # Check for differentiated urgency based on area
        differentiated_patterns = [
            r"(?:some|certain)\s+areas?\s+(?:should|need\s+to)\s+(?:evacuate|leave)\s+(?:first|immediately)",
            r"(?:higher|lower)\s+priority\s+(?:for|to)\s+(?:evacuate|evacuating)",
            r"(?:if\s+you\s+live\s+in)\s+(?:a\s+)?(?:nice|good|expensive|cheap)\s+(?:area|neighborhood)",
        ]

        for pattern in differentiated_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.HIGH,
                        message="Evacuation advice appears to differentiate based on area type",
                        details={"pattern": pattern[:40]},
                        remediation="Base evacuation urgency on risk factors (storm surge, elevation) not area demographics",
                    )
                )

        return violations

    def _check_inclusivity(self, content: str) -> list[GuardrailViolation]:
        """Check for inclusive language."""
        violations = []

        # Non-inclusive terms to avoid
        non_inclusive_terms = [
            ("guys", "everyone/folks"),
            ("manpower", "workforce/personnel"),
            ("mankind", "humanity/people"),
            ("crazy weather", "extreme/severe weather"),
            ("lame excuse", "weak/poor excuse"),
        ]

        content_lower = content.lower()
        for term, alternative in non_inclusive_terms:
            if re.search(rf"\b{term}\b", content_lower):
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.LOW,
                        message=f"Consider more inclusive language: '{term}' → '{alternative}'",
                        details={
                            "term": term,
                            "suggested": alternative,
                        },
                        remediation=f"Use '{alternative}' instead of '{term}'",
                    )
                )

        return violations

    def _get_remediation(self, bias_type: str) -> str:
        """Get remediation suggestion for bias type."""
        remediations = {
            "socioeconomic_labeling": "Describe areas by geographic features, not wealth",
            "wealth_assumption": "Provide advice without assumptions about financial resources",
            "area_judgment": "Use neutral descriptors for locations",
            "age_generalization": "Avoid generalizing about age groups",
            "gender_generalization": "Use gender-neutral language",
            "gender_assumption": "Don't assume gender affects weather response",
            "geographic_preference": "Treat all affected areas with equal urgency",
            "victim_blaming": "Focus on actionable advice, not judgment",
            "stereotyping": "Avoid generalizations about regions or people",
            "mobility_assumption": "Include options for those with limited mobility",
            "capability_assumption": "Consider varying capabilities and resources",
            "technology_assumption": "Provide multiple contact methods",
        }
        return remediations.get(bias_type, "Use neutral, inclusive language")
