"""Layer 1: Input Validation.

Validates input format, length, and schema before processing.
First line of defense against malformed requests.

Checks:
- Maximum input length
- Minimum input length (not empty)
- Character encoding (UTF-8)
- Control character detection
- Schema validation (if applicable)
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


class InputValidationLayer(BaseGuardrailLayer):
    """Layer 1: Input Validation.

    Validates basic input properties before further processing.
    """

    layer = GuardrailLayer.L1_INPUT_VALIDATION

    # Minimum reasonable query length
    MIN_INPUT_LENGTH = 2

    # Control characters to reject (except newlines, tabs)
    CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    # Excessive repetition pattern (spam detection)
    REPETITION_PATTERN = re.compile(r"(.)\1{20,}")

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize with configuration."""
        super().__init__(config)
        self.max_length = self.config.max_input_length

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Validate input format and length.

        Args:
            content: User input to validate
            context: Additional context (unused)

        Returns:
            List of violations (empty if valid)
        """
        violations: list[GuardrailViolation] = []

        # Check: Empty or too short
        if not content or len(content.strip()) < self.MIN_INPUT_LENGTH:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.MEDIUM,
                    message=f"Input too short (min {self.MIN_INPUT_LENGTH} chars)",
                    details={"length": len(content) if content else 0},
                    remediation="Provide a more detailed query",
                )
            )

        # Check: Too long
        if content and len(content) > self.max_length:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.HIGH,
                    message=f"Input exceeds maximum length ({self.max_length} chars)",
                    details={
                        "length": len(content),
                        "max_allowed": self.max_length,
                    },
                    remediation="Shorten your query or break into multiple requests",
                )
            )

        # Check: Control characters
        if content and self.CONTROL_CHAR_PATTERN.search(content):
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.HIGH,
                    message="Input contains invalid control characters",
                    details={"pattern": "control_chars"},
                    remediation="Remove special characters from input",
                )
            )

        # Check: Encoding issues
        if content:
            try:
                content.encode("utf-8").decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                violations.append(
                    self.create_violation(
                        severity=ViolationSeverity.HIGH,
                        message="Input has encoding issues (must be valid UTF-8)",
                        details={"encoding": "utf-8_error"},
                        remediation="Ensure input is valid UTF-8 text",
                    )
                )

        # Check: Excessive repetition (possible spam)
        if content and self.REPETITION_PATTERN.search(content):
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.MEDIUM,
                    message="Input contains excessive character repetition",
                    details={"pattern": "repetition"},
                    remediation="Remove repeated characters",
                )
            )

        # Check: Only whitespace
        if content and content.strip() == "":
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.MEDIUM,
                    message="Input contains only whitespace",
                    details={"whitespace_only": True},
                    remediation="Provide actual query content",
                )
            )

        return violations
