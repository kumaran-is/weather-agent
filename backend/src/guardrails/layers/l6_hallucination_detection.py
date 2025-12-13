"""Layer 6: Hallucination Detection.

Detects hallucinated or fabricated information in agent outputs,
especially critical for weather and hurricane data accuracy.

Detection Methods:
1. Fact verification against tool outputs
2. Consistency checking (internal contradictions)
3. Plausibility checking (impossible values)
4. Source attribution verification
5. Saffir-Simpson scale validation (CRITICAL)

CRITICAL: Hurricane information MUST match retrieved data.
Any mismatch = instant fail.
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


class HallucinationDetectionLayer(BaseGuardrailLayer):
    """Layer 6: Hallucination Detection.

    Identifies fabricated or inaccurate information in outputs.
    Critical for life-safety weather information.
    """

    layer = GuardrailLayer.L6_HALLUCINATION_DETECTION

    # Saffir-Simpson Scale (wind speeds in mph)
    SAFFIR_SIMPSON = {
        1: (74, 95),
        2: (96, 110),
        3: (111, 129),
        4: (130, 156),
        5: (157, float("inf")),
    }

    # Physical plausibility bounds for weather data
    PLAUSIBILITY_BOUNDS = {
        "temperature_f": (-130, 140),  # Fahrenheit: coldest to hottest recorded
        "temperature_c": (-90, 60),  # Celsius
        "wind_speed_mph": (0, 250),  # Max theoretical wind
        "wind_speed_kph": (0, 400),
        "humidity_percent": (0, 100),
        "pressure_mb": (870, 1084),  # Barometric pressure
        "precipitation_inches": (0, 100),  # 24h max
    }

    # Patterns indicating confident but unverified claims
    OVERCONFIDENCE_PATTERNS = [
        r"(?:will\s+)?definitely\s+(?:be|hit|occur)",
        r"100%\s+(?:chance|certain|sure)",
        r"guaranteed\s+to",
        r"there\s+is\s+no\s+(?:doubt|question)",
        r"absolutely\s+certain",
    ]

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Detect hallucinations in output.

        Args:
            content: Agent output to verify
            context: Should contain 'trajectory' with tool outputs

        Returns:
            List of hallucination violations
        """
        violations: list[GuardrailViolation] = []
        context = context or {}
        trajectory = context.get("trajectory", [])

        # Check hurricane category/wind consistency (CRITICAL)
        hurricane_violations = self._check_hurricane_consistency(content)
        violations.extend(hurricane_violations)

        # Check against trajectory data
        if trajectory:
            grounding_violations = self._check_trajectory_grounding(content, trajectory)
            violations.extend(grounding_violations)

        # Check physical plausibility
        plausibility_violations = self._check_plausibility(content)
        violations.extend(plausibility_violations)

        # Check for overconfident claims
        overconfidence_violations = self._check_overconfidence(content)
        violations.extend(overconfidence_violations)

        # Check internal consistency
        consistency_violations = self._check_internal_consistency(content)
        violations.extend(consistency_violations)

        return violations

    def _check_hurricane_consistency(self, content: str) -> list[GuardrailViolation]:
        """Verify hurricane category matches Saffir-Simpson scale.

        CRITICAL: Cat 5 requires 157+ mph. Any mismatch = fail.
        """
        violations = []

        # Extract category mentions
        category_pattern = r"category\s*(\d)"
        wind_pattern = r"(\d{2,3})\s*(?:mph|miles\s*per\s*hour)"

        categories = re.findall(category_pattern, content, re.IGNORECASE)
        winds = re.findall(wind_pattern, content, re.IGNORECASE)

        if not categories or not winds:
            return []  # No hurricane data to validate

        for cat_str in categories:
            cat = int(cat_str)
            if cat not in self.SAFFIR_SIMPSON:
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.CRITICAL,
                        message=f"Invalid hurricane category: {cat}",
                        details={"invalid_category": cat},
                        remediation="Use valid category (1-5)",
                    )
                )
                continue

            min_wind, max_wind = self.SAFFIR_SIMPSON[cat]

            for wind_str in winds:
                wind = int(wind_str)

                if wind < min_wind:
                    violations.append(
                        self.create_violation(
                            severity=ViolationSeverity.CRITICAL,
                            message=(
                                f"Hurricane Category {cat} stated with {wind} mph - "
                                f"INCORRECT! Category {cat} requires {min_wind}-{max_wind} mph"
                            ),
                            details={
                                "stated_category": cat,
                                "stated_wind": wind,
                                "required_min": min_wind,
                                "required_max": max_wind,
                            },
                            remediation="Correct category to match wind speed",
                        )
                    )
                elif wind > max_wind and cat < 5:
                    correct_cat = self._get_category_for_wind(wind)
                    violations.append(
                        self.create_violation(
                            severity=ViolationSeverity.CRITICAL,
                            message=(
                                f"Wind speed {wind} mph indicates Category {correct_cat}, "
                                f"not Category {cat} as stated"
                            ),
                            details={
                                "stated_category": cat,
                                "correct_category": correct_cat,
                                "wind_speed": wind,
                            },
                            remediation=f"Update category to {correct_cat}",
                        )
                    )

        return violations

    def _get_category_for_wind(self, wind_mph: int) -> int:
        """Get correct category for wind speed."""
        for cat, (min_w, max_w) in self.SAFFIR_SIMPSON.items():
            if min_w <= wind_mph <= max_w:
                return cat
        return 5 if wind_mph >= 157 else 0

    def _check_trajectory_grounding(
        self,
        content: str,
        trajectory: list[dict[str, Any]],
    ) -> list[GuardrailViolation]:
        """Check if output is grounded in trajectory data."""
        violations = []

        # Extract facts from trajectory
        retrieved_facts = self._extract_trajectory_facts(trajectory)

        # Extract claimed facts from output
        claimed_facts = self._extract_claimed_facts(content)

        # Compare critical facts
        for fact_type, claimed_value in claimed_facts.items():
            retrieved_value = retrieved_facts.get(fact_type)

            if retrieved_value is None:
                # Can't verify - might be hallucination
                if fact_type in ["category", "wind_speed", "storm_name"]:
                    violations.append(
                        self.create_violation(
                            severity=ViolationSeverity.HIGH,
                            message=f"Claimed {fact_type} not found in retrieved data",
                            details={
                                "fact_type": fact_type,
                                "claimed": claimed_value,
                                "retrieved": None,
                            },
                            remediation="Only state facts from retrieved data",
                        )
                    )
            elif str(claimed_value) != str(retrieved_value):
                # Mismatch - definite hallucination
                severity = (
                    ViolationSeverity.CRITICAL
                    if fact_type in ["category", "wind_speed"]
                    else ViolationSeverity.HIGH
                )
                violations.append(
                    self.create_violation(
                        severity=severity,
                        message=f"{fact_type} mismatch: claimed {claimed_value}, retrieved {retrieved_value}",
                        details={
                            "fact_type": fact_type,
                            "claimed": claimed_value,
                            "retrieved": retrieved_value,
                        },
                        remediation="Use retrieved data, not fabricated values",
                    )
                )

        return violations

    def _extract_trajectory_facts(self, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract verifiable facts from trajectory."""
        facts = {}

        for step in trajectory:
            if not isinstance(step, dict):
                continue

            output = step.get("output", {})
            if isinstance(output, dict):
                for key in ["category", "wind_speed", "max_wind", "storm_name", "temperature"]:
                    if key in output:
                        facts[key] = output[key]

        return facts

    def _extract_claimed_facts(self, content: str) -> dict[str, Any]:
        """Extract facts claimed in output."""
        facts = {}

        # Category
        cat_match = re.search(r"category\s*(\d)", content, re.IGNORECASE)
        if cat_match:
            facts["category"] = int(cat_match.group(1))

        # Wind speed
        wind_match = re.search(r"(\d{2,3})\s*mph", content, re.IGNORECASE)
        if wind_match:
            facts["wind_speed"] = int(wind_match.group(1))

        # Temperature
        temp_match = re.search(r"(\d{1,3})°?\s*[fF]", content)
        if temp_match:
            facts["temperature"] = int(temp_match.group(1))

        return facts

    def _check_plausibility(self, content: str) -> list[GuardrailViolation]:
        """Check for physically impossible values."""
        violations = []

        # Temperature check (Fahrenheit)
        temp_matches = re.findall(r"(-?\d{1,3})°?\s*[fF]", content)
        for temp_str in temp_matches:
            temp = int(temp_str)
            min_t, max_t = self.PLAUSIBILITY_BOUNDS["temperature_f"]
            if temp < min_t or temp > max_t:
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.HIGH,
                        message=f"Implausible temperature: {temp}°F",
                        details={
                            "value": temp,
                            "unit": "fahrenheit",
                            "valid_range": (min_t, max_t),
                        },
                        remediation="Verify temperature value",
                    )
                )

        # Wind speed check
        wind_matches = re.findall(r"(\d{2,3})\s*mph", content, re.IGNORECASE)
        for wind_str in wind_matches:
            wind = int(wind_str)
            min_w, max_w = self.PLAUSIBILITY_BOUNDS["wind_speed_mph"]
            if wind > max_w:
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.HIGH,
                        message=f"Implausible wind speed: {wind} mph",
                        details={
                            "value": wind,
                            "unit": "mph",
                            "max_plausible": max_w,
                        },
                        remediation="Verify wind speed value",
                    )
                )

        # Humidity check
        humidity_matches = re.findall(r"(\d{1,3})%\s*(?:humidity|relative)", content, re.IGNORECASE)
        for hum_str in humidity_matches:
            hum = int(hum_str)
            if hum > 100:
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.HIGH,
                        message=f"Implausible humidity: {hum}%",
                        details={"value": hum, "max_valid": 100},
                        remediation="Humidity cannot exceed 100%",
                    )
                )

        return violations

    def _check_overconfidence(self, content: str) -> list[GuardrailViolation]:
        """Check for overconfident predictions."""
        violations = []

        for pattern in self.OVERCONFIDENCE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.MEDIUM,
                        message="Overconfident weather prediction detected",
                        details={"pattern": pattern[:30]},
                        remediation="Use probabilistic language for forecasts",
                    )
                )

        return violations

    def _check_internal_consistency(self, content: str) -> list[GuardrailViolation]:
        """Check for internal contradictions."""
        violations = []

        # Check for contradictory temperature claims
        temp_matches = re.findall(r"(\d{1,3})°?\s*[fF]", content)
        if len(temp_matches) > 1:
            temps = [int(t) for t in temp_matches]
            # Large temperature variance in same response is suspicious
            if max(temps) - min(temps) > 50:
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.MEDIUM,
                        message="Large temperature variance in response",
                        details={
                            "temperatures": temps,
                            "variance": max(temps) - min(temps),
                        },
                        remediation="Clarify which temperature applies to which context",
                    )
                )

        return violations
