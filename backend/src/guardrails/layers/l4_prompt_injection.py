"""Layer 4: Prompt Injection Detection.

Detects and blocks prompt injection attacks that attempt to
manipulate the AI agent into unauthorized behavior.

Attack Patterns Detected:
- Instruction override attempts ("ignore previous instructions")
- Role manipulation ("you are now")
- System prompt extraction ("reveal your system prompt")
- Jailbreak attempts ("DAN mode", "developer mode")
- Delimiter injection (attempting to close prompts)
- Encoding attacks (base64, unicode tricks)

CRITICAL: Any detected injection = INSTANT BLOCK
"""

import base64
import re
from typing import Any

from backend.src.guardrails.layers.base import BaseGuardrailLayer
from backend.src.guardrails.models import (
    GuardrailLayer,
    GuardrailViolation,
    ViolationSeverity,
)


class PromptInjectionLayer(BaseGuardrailLayer):
    """Layer 4: Prompt Injection Detection.

    Identifies and blocks prompt injection attacks.
    Zero tolerance - any detection results in block.
    """

    layer = GuardrailLayer.L4_PROMPT_INJECTION

    # Instruction override patterns
    OVERRIDE_PATTERNS = [
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)",
        r"disregard\s+(?:all\s+)?(?:previous|prior|above)",
        r"forget\s+(?:everything|all|your)\s+(?:instructions|training|rules)",
        r"new\s+instructions?\s*[:=]",
        r"override\s+(?:system|safety|previous)",
        r"bypass\s+(?:safety|security|filters?)",
    ]

    # Role manipulation patterns
    ROLE_PATTERNS = [
        r"you\s+are\s+now\s+(?:a|an|the)",
        r"pretend\s+(?:to\s+be|you\s+are)",
        r"act\s+as\s+(?:if\s+you\s+are|a|an)",
        r"roleplay\s+as",
        r"assume\s+(?:the\s+)?(?:role|identity)\s+of",
        r"from\s+now\s+on\s+you\s+are",
    ]

    # Jailbreak patterns
    JAILBREAK_PATTERNS = [
        r"\bDAN\b",  # Do Anything Now
        r"developer\s+mode",
        r"jailbreak",
        r"unlocked\s+mode",
        r"evil\s+mode",
        r"unrestricted\s+mode",
        r"no\s+(?:limits?|restrictions?|rules?)",
        r"SUDO\s+mode",
    ]

    # System prompt extraction
    EXTRACTION_PATTERNS = [
        r"(?:reveal|show|print|display|output)\s+(?:your\s+)?system\s+prompt",
        r"what\s+(?:are|is)\s+your\s+(?:instructions?|system\s+prompt|rules?)",
        r"(?:repeat|echo)\s+(?:your\s+)?(?:initial\s+)?(?:instructions?|prompt)",
        r"system\s*[:=]",
        r"<\s*system\s*>",
    ]

    # Delimiter injection patterns
    DELIMITER_PATTERNS = [
        r"```\s*(?:system|end|ignore)",
        r"</?\s*(?:system|prompt|instructions?)\s*>",
        r"\[\s*(?:SYSTEM|INST|END)\s*\]",
        r"###\s*(?:SYSTEM|END|NEW)",
        r"Human\s*:\s*$",  # Trying to fake conversation turns
        r"Assistant\s*:\s*$",
    ]

    # Suspicious encodings (might hide injection)
    ENCODING_PATTERNS = [
        r"base64\s*[:=]",
        r"\\x[0-9a-fA-F]{2}",  # Hex encoding
        r"\\u[0-9a-fA-F]{4}",  # Unicode escapes
    ]

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Detect prompt injection attempts.

        Args:
            content: User input to check
            context: Additional context (unused)

        Returns:
            List of violations (any = block request)
        """
        violations: list[GuardrailViolation] = []

        # Normalize content for checking
        content_lower = content.lower()
        content_normalized = self._normalize_text(content)

        # Check all pattern categories
        pattern_checks = [
            (self.OVERRIDE_PATTERNS, "instruction_override", "Instruction override attempt"),
            (self.ROLE_PATTERNS, "role_manipulation", "Role manipulation attempt"),
            (self.JAILBREAK_PATTERNS, "jailbreak", "Jailbreak attempt"),
            (self.EXTRACTION_PATTERNS, "prompt_extraction", "System prompt extraction attempt"),
            (self.DELIMITER_PATTERNS, "delimiter_injection", "Delimiter injection attempt"),
        ]

        for patterns, attack_type, message in pattern_checks:
            for pattern in patterns:
                if re.search(pattern, content_normalized, re.IGNORECASE):
                    violations.append(
                        self.create_violation(
                            severity=ViolationSeverity.CRITICAL,
                            message=message,
                            details={
                                "attack_type": attack_type,
                                "pattern": pattern[:50],  # Truncate for logs
                            },
                            remediation="Remove manipulation attempts from query",
                        )
                    )
                    break  # One violation per category is enough

        # Check for encoded content that might hide injections
        encoded_check = self._check_encoded_content(content)
        if encoded_check:
            violations.append(encoded_check)

        # Check for suspicious character patterns
        char_check = self._check_suspicious_chars(content)
        if char_check:
            violations.append(char_check)

        return violations

    def _normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching.

        Handles common obfuscation attempts.
        """
        # Remove extra whitespace
        normalized = " ".join(text.split())

        # Handle common leetspeak substitutions
        substitutions = {
            "0": "o",
            "1": "i",
            "3": "e",
            "4": "a",
            "5": "s",
            "7": "t",
            "@": "a",
            "$": "s",
        }

        for char, replacement in substitutions.items():
            normalized = normalized.replace(char, replacement)

        return normalized

    def _check_encoded_content(self, content: str) -> GuardrailViolation | None:
        """Check for base64 or other encoded content."""
        # Look for base64-like patterns
        base64_pattern = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
        matches = base64_pattern.findall(content)

        for match in matches:
            try:
                # Try to decode
                decoded = base64.b64decode(match + "==").decode("utf-8", errors="ignore")
                # Check if decoded content contains injection patterns
                if any(
                    re.search(p, decoded, re.IGNORECASE)
                    for patterns in [self.OVERRIDE_PATTERNS, self.JAILBREAK_PATTERNS]
                    for p in patterns
                ):
                    return self.create_violation(
                        severity=ViolationSeverity.CRITICAL,
                        message="Encoded prompt injection detected",
                        details={"encoding": "base64"},
                        remediation="Remove encoded content from query",
                    )
            except Exception:
                pass  # Not valid base64

        return None

    def _check_suspicious_chars(self, content: str) -> GuardrailViolation | None:
        """Check for suspicious Unicode or special characters."""
        # Check for invisible/zero-width characters
        invisible_chars = [
            "\u200b",  # Zero-width space
            "\u200c",  # Zero-width non-joiner
            "\u200d",  # Zero-width joiner
            "\ufeff",  # Zero-width no-break space
            "\u2060",  # Word joiner
        ]

        for char in invisible_chars:
            if char in content:
                return self.create_violation(
                    severity=ViolationSeverity.HIGH,
                    message="Suspicious invisible characters detected",
                    details={"char_type": "zero_width"},
                    remediation="Remove hidden characters from query",
                )

        # Check for homoglyph attacks (letters that look like other letters)
        # This is a simplified check - production would use a comprehensive homoglyph database
        suspicious_ranges = [
            (0x0400, 0x04FF),  # Cyrillic (can look like Latin)
            (0x0370, 0x03FF),  # Greek
        ]

        for char in content:
            code = ord(char)
            for start, end in suspicious_ranges:
                if start <= code <= end:
                    return self.create_violation(
                        severity=ViolationSeverity.MEDIUM,
                        message="Potential homoglyph attack detected",
                        details={"char_range": f"U+{start:04X}-U+{end:04X}"},
                        remediation="Use standard ASCII characters",
                    )

        return None
