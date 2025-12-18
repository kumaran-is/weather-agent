"""
Output Validation Module.

Level 6c: Self-Evolving Platform

Provides response validation:
1. Format validation (JSON, markdown, structured)
2. Length validation
3. Required field validation
4. Domain-specific validation (weather)
5. Quality scoring

Target: 99%+ response quality compliance
"""

import json
import logging
import re
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RuleType(str, Enum):
    """Types of validation rules."""

    LENGTH = "length"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    JSON_VALID = "json_valid"
    CUSTOM = "custom"
    REQUIRED_FIELDS = "required_fields"


class ValidationRule(BaseModel):
    """A single validation rule."""

    id: str
    name: str
    rule_type: RuleType
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    severity: str = "warning"  # info, warning, error, critical
    weight: float = Field(ge=0.0, le=1.0, default=1.0)


class RuleResult(BaseModel):
    """Result of applying a single rule."""

    rule_id: str
    rule_name: str
    passes: bool
    message: str = ""
    severity: str = "warning"
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Complete validation result."""

    content: str
    is_valid: bool
    score: float = Field(ge=0.0, le=1.0)
    rule_results: list[RuleResult] = Field(default_factory=list)
    total_rules: int = 0
    passed_rules: int = 0
    failed_rules: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class OutputValidator:
    """
    Output validation for response quality assurance.

    Validates AI responses against configurable rules.
    """

    def __init__(
        self,
        rules: list[ValidationRule] | None = None,
        pass_threshold: float = 0.8,
    ):
        """
        Initialize output validator.

        Args:
            rules: List of validation rules
            pass_threshold: Score threshold to pass validation
        """
        self.rules = rules or []
        self.pass_threshold = pass_threshold

        # Custom validators
        self.custom_validators: dict[str, Callable[[str, dict], bool]] = {}

        logger.info(
            f"OutputValidator initialized | rules={len(self.rules)} | "
            f"threshold={pass_threshold}"
        )

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)

    def register_custom_validator(
        self,
        name: str,
        validator: Callable[[str, dict], bool],
    ) -> None:
        """Register a custom validator function."""
        self.custom_validators[name] = validator

    def validate(self, content: str) -> ValidationResult:
        """
        Validate content against all rules.

        Args:
            content: Content to validate

        Returns:
            ValidationResult with validation status
        """
        rule_results: list[RuleResult] = []
        warnings: list[str] = []
        errors: list[str] = []

        for rule in self.rules:
            result = self._apply_rule(content, rule)
            rule_results.append(result)

            if not result.passes:
                if result.severity in ["warning", "info"]:
                    warnings.append(f"{result.rule_name}: {result.message}")
                else:
                    errors.append(f"{result.rule_name}: {result.message}")

        # Calculate score
        total_weight = sum(r.weight for r in self.rules)
        passed_weight = sum(
            r.weight for r, result in zip(self.rules, rule_results) if result.passes
        )
        score = passed_weight / total_weight if total_weight > 0 else 1.0

        passed_rules = sum(1 for r in rule_results if r.passes)
        failed_rules = len(rule_results) - passed_rules

        return ValidationResult(
            content=content,
            is_valid=score >= self.pass_threshold and len(errors) == 0,
            score=round(score, 4),
            rule_results=rule_results,
            total_rules=len(self.rules),
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            warnings=warnings,
            errors=errors,
        )

    def _apply_rule(self, content: str, rule: ValidationRule) -> RuleResult:
        """Apply a single validation rule."""
        passes = False
        message = ""
        details: dict[str, Any] = {}

        try:
            if rule.rule_type == RuleType.LENGTH:
                passes, message, details = self._validate_length(content, rule.parameters)

            elif rule.rule_type == RuleType.CONTAINS:
                passes, message, details = self._validate_contains(content, rule.parameters)

            elif rule.rule_type == RuleType.NOT_CONTAINS:
                passes, message, details = self._validate_not_contains(content, rule.parameters)

            elif rule.rule_type == RuleType.REGEX:
                passes, message, details = self._validate_regex(content, rule.parameters)

            elif rule.rule_type == RuleType.JSON_VALID:
                passes, message, details = self._validate_json(content, rule.parameters)

            elif rule.rule_type == RuleType.REQUIRED_FIELDS:
                passes, message, details = self._validate_required_fields(content, rule.parameters)

            elif rule.rule_type == RuleType.CUSTOM:
                passes, message, details = self._validate_custom(content, rule.parameters)

        except Exception as e:
            passes = False
            message = f"Validation error: {str(e)}"

        return RuleResult(
            rule_id=rule.id,
            rule_name=rule.name,
            passes=passes,
            message=message,
            severity=rule.severity if not passes else "info",
            details=details,
        )

    def _validate_length(
        self,
        content: str,
        params: dict[str, Any],
    ) -> tuple[bool, str, dict]:
        """Validate content length."""
        min_len = params.get("min", 0)
        max_len = params.get("max", float("inf"))
        unit = params.get("unit", "chars")  # chars or words

        if unit == "words":
            length = len(content.split())
        else:
            length = len(content)

        passes = min_len <= length <= max_len

        if not passes:
            if length < min_len:
                message = f"Content too short: {length} {unit} (min: {min_len})"
            else:
                message = f"Content too long: {length} {unit} (max: {max_len})"
        else:
            message = f"Length OK: {length} {unit}"

        return passes, message, {"length": length, "unit": unit}

    def _validate_contains(
        self,
        content: str,
        params: dict[str, Any],
    ) -> tuple[bool, str, dict]:
        """Validate content contains required elements."""
        required = params.get("required", [])
        case_sensitive = params.get("case_sensitive", False)

        check_content = content if case_sensitive else content.lower()
        missing = []

        for item in required:
            check_item = item if case_sensitive else item.lower()
            if check_item not in check_content:
                missing.append(item)

        passes = len(missing) == 0
        message = "Contains all required elements" if passes else f"Missing: {', '.join(missing)}"

        return passes, message, {"missing": missing}

    def _validate_not_contains(
        self,
        content: str,
        params: dict[str, Any],
    ) -> tuple[bool, str, dict]:
        """Validate content does not contain prohibited elements."""
        prohibited = params.get("prohibited", [])
        case_sensitive = params.get("case_sensitive", False)

        check_content = content if case_sensitive else content.lower()
        found = []

        for item in prohibited:
            check_item = item if case_sensitive else item.lower()
            if check_item in check_content:
                found.append(item)

        passes = len(found) == 0
        message = "No prohibited content found" if passes else f"Found prohibited: {', '.join(found)}"

        return passes, message, {"found": found}

    def _validate_regex(
        self,
        content: str,
        params: dict[str, Any],
    ) -> tuple[bool, str, dict]:
        """Validate content matches regex pattern."""
        pattern = params.get("pattern", "")
        should_match = params.get("should_match", True)

        if not pattern:
            return True, "No pattern specified", {}

        try:
            compiled = re.compile(pattern, re.IGNORECASE if params.get("ignore_case") else 0)
            matches = compiled.findall(content)
            has_match = len(matches) > 0

            if should_match:
                passes = has_match
                message = f"Pattern matched {len(matches)} time(s)" if passes else "Pattern not found"
            else:
                passes = not has_match
                message = "Pattern not found (as expected)" if passes else f"Pattern matched {len(matches)} time(s)"

            return passes, message, {"matches": matches[:5]}

        except re.error as e:
            return False, f"Invalid regex: {e}", {}

    def _validate_json(
        self,
        content: str,
        params: dict[str, Any],
    ) -> tuple[bool, str, dict]:
        """Validate content is valid JSON."""
        # Try to find JSON in content
        json_pattern = r'\{[^{}]*\}|\[[^\[\]]*\]'
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                return True, "Valid JSON found", {"parsed": parsed}
            except json.JSONDecodeError:
                continue

        # Try parsing entire content
        try:
            parsed = json.loads(content)
            return True, "Content is valid JSON", {"parsed": parsed}
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", {}

    def _validate_required_fields(
        self,
        content: str,
        params: dict[str, Any],
    ) -> tuple[bool, str, dict]:
        """Validate content contains required fields/sections."""
        fields = params.get("fields", [])
        content_lower = content.lower()

        missing = []
        for field in fields:
            # Check for field as header or key
            patterns = [
                f"{field.lower()}:",
                f"**{field.lower()}**",
                f"# {field.lower()}",
                f"## {field.lower()}",
            ]
            if not any(p in content_lower for p in patterns):
                missing.append(field)

        passes = len(missing) == 0
        message = "All required fields present" if passes else f"Missing fields: {', '.join(missing)}"

        return passes, message, {"missing": missing}

    def _validate_custom(
        self,
        content: str,
        params: dict[str, Any],
    ) -> tuple[bool, str, dict]:
        """Apply custom validator."""
        validator_name = params.get("validator_name", "")
        validator = self.custom_validators.get(validator_name)

        if validator is None:
            return False, f"Custom validator '{validator_name}' not found", {}

        try:
            result = validator(content, params)
            passes = result if isinstance(result, bool) else result[0]
            message = "Custom validation passed" if passes else "Custom validation failed"
            return passes, message, {}

        except Exception as e:
            return False, f"Custom validator error: {e}", {}

    @classmethod
    def weather_validator(cls) -> "OutputValidator":
        """Create a validator for weather responses."""
        rules = [
            ValidationRule(
                id="weather_min_length",
                name="Minimum Length",
                rule_type=RuleType.LENGTH,
                description="Response must be at least 50 words",
                parameters={"min": 50, "unit": "words"},
                severity="warning",
                weight=0.6,
            ),
            ValidationRule(
                id="weather_max_length",
                name="Maximum Length",
                rule_type=RuleType.LENGTH,
                description="Response should not exceed 500 words",
                parameters={"max": 500, "unit": "words"},
                severity="warning",
                weight=0.4,
            ),
            ValidationRule(
                id="weather_no_placeholder",
                name="No Placeholders",
                rule_type=RuleType.NOT_CONTAINS,
                description="Response should not contain placeholder text",
                parameters={
                    "prohibited": ["[placeholder]", "[insert", "[TODO", "XXX"],
                },
                severity="error",
                weight=1.0,
            ),
            ValidationRule(
                id="weather_has_data",
                name="Contains Weather Data",
                rule_type=RuleType.REGEX,
                description="Response should contain weather-related data",
                parameters={
                    "pattern": r"\d+\s*°[FC]|\d+\s*(mph|km/h)|forecast|weather|temperature",
                    "should_match": True,
                    "ignore_case": True,
                },
                severity="warning",
                weight=0.8,
            ),
            ValidationRule(
                id="weather_time_specificity",
                name="Time Specificity",
                rule_type=RuleType.REGEX,
                description="Time references should be specific",
                parameters={
                    "pattern": r"\b(soon|later|sometime)\b",
                    "should_match": False,
                    "ignore_case": True,
                },
                severity="warning",
                weight=0.6,
            ),
        ]

        return cls(rules=rules, pass_threshold=0.75)

    @classmethod
    def hurricane_validator(cls) -> "OutputValidator":
        """Create a validator for hurricane responses."""
        rules = [
            ValidationRule(
                id="hurricane_category",
                name="Hurricane Category Check",
                rule_type=RuleType.REGEX,
                description="Hurricane category should be mentioned if relevant",
                parameters={
                    "pattern": r"category\s*[1-5]|cat\s*[1-5]",
                    "should_match": True,
                    "ignore_case": True,
                },
                severity="warning",
                weight=0.7,
            ),
            ValidationRule(
                id="hurricane_safety",
                name="Safety Information",
                rule_type=RuleType.CONTAINS,
                description="Hurricane responses should include safety information",
                parameters={
                    "required": ["safety"],
                },
                severity="warning",
                weight=0.8,
            ),
            ValidationRule(
                id="hurricane_no_downplay",
                name="No Downplaying",
                rule_type=RuleType.NOT_CONTAINS,
                description="Should not downplay hurricane risks",
                parameters={
                    "prohibited": ["nothing to worry", "no big deal", "minor storm"],
                },
                severity="critical",
                weight=1.0,
            ),
            ValidationRule(
                id="hurricane_source",
                name="Source Attribution",
                rule_type=RuleType.REGEX,
                description="Should cite authoritative sources",
                parameters={
                    "pattern": r"NHC|National Hurricane Center|NWS|NOAA",
                    "should_match": True,
                    "ignore_case": False,
                },
                severity="warning",
                weight=0.6,
            ),
        ]

        return cls(rules=rules, pass_threshold=0.80)

    def get_rules(self) -> list[dict[str, Any]]:
        """Get all validation rules."""
        return [r.model_dump() for r in self.rules]
