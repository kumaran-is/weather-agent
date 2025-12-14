"""PII Sanitization Layer (Priority 2 Fix - 2025-12-14).

This module provides PII detection and sanitization for agent responses.

CRITICAL REQUIREMENT: Zero tolerance for PII leaks in production responses.

PII Types Detected:
- SSN: Social Security Numbers (XXX-XX-XXXX)
- Credit Cards: 16-digit card numbers
- Phone Numbers: 10-digit US phone numbers (with false positive filtering)
- Email Addresses: email@domain.com format

Sanitization Strategies:
1. Detection: Identify PII patterns with high precision
2. Filtering: Remove known false positives (emergency numbers, coordinates)
3. Redaction: Replace PII with safe placeholders
4. Validation: Verify no PII remains after sanitization

Usage:
    >>> sanitizer = PIISanitizer()
    >>> clean_text = sanitizer.sanitize("Call 555-123-4567 for updates")
    >>> # Returns: "Call [REDACTED_PHONE] for updates"
"""

import re
import logging
from typing import Any
from enum import Enum

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """PII types detected and sanitized."""

    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    PHONE = "phone"
    EMAIL = "email"


class PIISanitizer:
    """PII detection and sanitization for agent responses.

    PRIORITY 2 FIX: Added to resolve PII leak in complex queries (2025-12-14).

    Features:
    - Pattern-based PII detection
    - False positive filtering (emergency numbers, coordinates)
    - Redaction with safe placeholders
    - Detection reporting for monitoring
    """

    # PII Detection Patterns (from SafetyValidator)
    PII_PATTERNS = {
        PIIType.SSN: r"\b\d{3}-\d{2}-\d{4}\b",
        PIIType.CREDIT_CARD: r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        PIIType.PHONE: r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        PIIType.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    }

    # Known false positives for phone pattern
    # These are valid emergency/service numbers, NOT PII
    PHONE_FALSE_POSITIVES = {
        "911",  # Emergency
        "311",  # Non-emergency services
        "511",  # Travel/traffic info
        "211",  # Community services
        "988",  # Mental health crisis
        "411",  # Directory assistance
        "611",  # Wireless carrier support
        "711",  # Telecommunications relay
        "811",  # Call before you dig
    }

    # Known emergency contact patterns (NOT PII)
    EMERGENCY_PATTERNS = [
        r"\b911\b",  # Emergency
        r"\b1-800-\d{3}-\d{4}\b",  # Toll-free (public hotlines)
        r"\b1-888-\d{3}-\d{4}\b",  # Toll-free
        r"\b1-877-\d{3}-\d{4}\b",  # Toll-free
        r"\b1-866-\d{3}-\d{4}\b",  # Toll-free
    ]

    # Redaction placeholders
    REDACTION_PLACEHOLDERS = {
        PIIType.SSN: "[REDACTED_SSN]",
        PIIType.CREDIT_CARD: "[REDACTED_CARD]",
        PIIType.PHONE: "[REDACTED_PHONE]",
        PIIType.EMAIL: "[REDACTED_EMAIL]",
    }

    def __init__(self, strict_mode: bool = True) -> None:
        """Initialize PII sanitizer.

        Args:
            strict_mode: If True, redact all potential PII. If False, allow known false positives.
        """
        self.strict_mode = strict_mode

    def sanitize(self, text: str) -> str:
        """Sanitize text by removing/redacting PII.

        Args:
            text: Input text potentially containing PII

        Returns:
            Sanitized text with PII redacted
        """
        if not text:
            return text

        sanitized = text

        # Sanitize each PII type
        for pii_type, pattern in self.PII_PATTERNS.items():
            sanitized = self._sanitize_pattern(sanitized, pii_type, pattern)

        return sanitized

    def detect_pii(self, text: str) -> dict[PIIType, list[str]]:
        """Detect PII in text without redaction (for monitoring).

        Args:
            text: Input text to scan

        Returns:
            Dictionary mapping PII types to list of detected instances (redacted)
        """
        detections: dict[PIIType, list[str]] = {}

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)

            if matches:
                # Filter false positives
                if pii_type == PIIType.PHONE:
                    matches = [m for m in matches if not self._is_phone_false_positive(m)]

                if matches:
                    # Don't return actual PII - return redacted placeholders
                    detections[pii_type] = [self.REDACTION_PLACEHOLDERS[pii_type]] * len(matches)

        return detections

    def _sanitize_pattern(self, text: str, pii_type: PIIType, pattern: str) -> str:
        """Sanitize specific PII pattern from text.

        Args:
            text: Input text
            pii_type: Type of PII to sanitize
            pattern: Regex pattern for detection

        Returns:
            Text with PII redacted
        """
        # Find all matches
        matches = re.finditer(pattern, text, re.IGNORECASE)

        # Process matches in reverse order (preserve offsets)
        replacements = []
        for match in matches:
            matched_text = match.group(0)

            # Check for false positives
            if pii_type == PIIType.PHONE:
                if self._is_phone_false_positive(matched_text):
                    continue  # Skip false positives

            # Check for emergency patterns (always allowed)
            # Pass surrounding context to check for toll-free prefixes
            context_start = max(0, match.start() - 2)  # Include "1-" prefix if present
            context_end = min(len(text), match.end() + 1)
            context = text[context_start:context_end]
            if self._is_emergency_pattern(context):
                continue

            # Add to replacement list
            replacements.append((match.start(), match.end(), self.REDACTION_PLACEHOLDERS[pii_type]))

        # Apply replacements in reverse order
        for start, end, placeholder in sorted(replacements, reverse=True):
            text = text[:start] + placeholder + text[end:]

        return text

    def _is_phone_false_positive(self, phone: str) -> bool:
        """Check if phone number is a known false positive.

        Args:
            phone: Phone number string (may have separators)

        Returns:
            True if this is a false positive (not real PII)
        """
        # Remove separators
        digits = re.sub(r"[-.\s]", "", phone)

        # Check if it's a 3-digit service number (911, 311, etc.)
        if len(digits) == 3 and digits in self.PHONE_FALSE_POSITIVES:
            return True

        # Check if it starts with emergency code
        if digits.startswith(tuple(self.PHONE_FALSE_POSITIVES)):
            return True

        # Check if it's a coordinate-like pattern (e.g., "123 456 7890" could be lat/long)
        # This is aggressive but safe for weather context
        if not self.strict_mode:
            # In non-strict mode, allow numbers that might be coordinates
            # (This would need more sophisticated coordinate detection)
            pass

        return False

    def _is_emergency_pattern(self, text: str) -> bool:
        """Check if text matches emergency contact pattern.

        Args:
            text: Text to check

        Returns:
            True if this is an emergency/public service number
        """
        for pattern in self.EMERGENCY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def validate_sanitization(self, original: str, sanitized: str) -> bool:
        """Validate that sanitization was successful (no PII remains).

        Args:
            original: Original text before sanitization
            sanitized: Text after sanitization

        Returns:
            True if no PII detected in sanitized text
        """
        detections = self.detect_pii(sanitized)
        return len(detections) == 0


# Convenience function for quick sanitization
def sanitize_text(text: str, strict_mode: bool = True) -> str:
    """Quick sanitization function.

    Args:
        text: Text to sanitize
        strict_mode: If True, redact all potential PII

    Returns:
        Sanitized text
    """
    sanitizer = PIISanitizer(strict_mode=strict_mode)
    return sanitizer.sanitize(text)
