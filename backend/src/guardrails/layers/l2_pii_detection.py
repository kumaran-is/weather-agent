"""Layer 2: PII Detection.

Detects and optionally redacts Personally Identifiable Information (PII)
in both input and output to ensure privacy compliance.

Detects:
- Social Security Numbers (SSN)
- Credit Card Numbers
- Phone Numbers
- Email Addresses
- Physical Addresses
- Dates of Birth
- Driver's License Numbers
- Passport Numbers
- Bank Account Numbers
- IP Addresses

Compliance: HIPAA, GDPR, CCPA, PCI-DSS
"""

import re
from typing import Any

from backend.src.guardrails.layers.base import BaseGuardrailLayer
from backend.src.guardrails.models import (
    GuardrailConfig,
    GuardrailLayer,
    GuardrailViolation,
    PIIType,
    ViolationSeverity,
)


class PIIDetectionLayer(BaseGuardrailLayer):
    """Layer 2: PII Detection.

    Scans content for personally identifiable information
    and either blocks or redacts based on configuration.
    """

    layer = GuardrailLayer.L2_PII_DETECTION

    # PII Detection Patterns
    PII_PATTERNS: dict[PIIType, re.Pattern[str]] = {
        PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        PIIType.CREDIT_CARD: re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b"
        ),
        PIIType.PHONE: re.compile(
            # Enhanced pattern to catch ALL phone number formats:
            # (555) 123-4567, 555-123-4567, 555.123.4567, 5551234567, +1-555-123-4567, etc.
            r"(?<!\w)(?:\+1[-.\s]?)?"  # Not preceded by word char; optional country code
            r"(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|"  # Standard formats with separators
            r"\d{10})(?!\w)"  # 10 digits with NO separators, not followed by word char
        ),
        PIIType.EMAIL: re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        PIIType.DATE_OF_BIRTH: re.compile(
            r"\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b"
        ),
        PIIType.IP_ADDRESS: re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
        PIIType.BANK_ACCOUNT: re.compile(
            r"\b\d{8,17}\b"  # Bank accounts are typically 8-17 digits
        ),
        PIIType.DRIVERS_LICENSE: re.compile(
            r"\b[A-Z]{1,2}\d{6,8}\b"  # Simplified pattern
        ),
    }

    # More specific patterns for addresses
    ADDRESS_PATTERNS = [
        re.compile(r"\b\d{1,5}\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)\b", re.IGNORECASE),
        re.compile(r"\b(?:P\.?O\.?\s*Box|PO\s*Box)\s*\d+\b", re.IGNORECASE),
    ]

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize with configuration."""
        super().__init__(config)
        self.pii_types = self.config.pii_types_to_detect
        self.redact = self.config.pii_redact

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Scan content for PII.

        Args:
            content: Text to scan for PII
            context: Additional context (unused)

        Returns:
            List of violations for each PII type found
        """
        violations: list[GuardrailViolation] = []
        content_lower = content.lower()

        # Check each configured PII type
        for pii_type in self.pii_types:
            if pii_type in self.PII_PATTERNS:
                pattern = self.PII_PATTERNS[pii_type]
                matches = pattern.findall(content)

                if matches:
                    # Don't include actual PII in violation details!
                    violations.append(
                        self.create_violation(
                            severity=self._get_severity(pii_type),
                            message=f"Detected {pii_type.value} ({len(matches)} occurrence(s))",
                            details={
                                "pii_type": pii_type.value,
                                "count": len(matches),
                                "will_redact": self.redact,
                            },
                            remediation=f"Remove {pii_type.value} from content",
                        )
                    )

        # Check for addresses (special handling)
        if PIIType.ADDRESS in self.pii_types:
            for pattern in self.ADDRESS_PATTERNS:
                if pattern.search(content):
                    violations.append(
                        self.create_violation(
                            severity=ViolationSeverity.HIGH,
                            message="Detected physical address",
                            details={
                                "pii_type": PIIType.ADDRESS.value,
                                "will_redact": self.redact,
                            },
                            remediation="Remove address information from content",
                        )
                    )
                    break  # Only report once

        return violations

    def _get_severity(self, pii_type: PIIType) -> ViolationSeverity:
        """Get severity level for PII type.

        Critical: SSN, Credit Card, Bank Account (financial/identity)
        High: Phone, Email, Address (contact info)
        Medium: DOB, IP (less sensitive)
        """
        critical_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT}
        high_types = {PIIType.PHONE, PIIType.EMAIL, PIIType.ADDRESS, PIIType.DRIVERS_LICENSE, PIIType.PASSPORT}

        if pii_type in critical_types:
            return ViolationSeverity.CRITICAL
        elif pii_type in high_types:
            return ViolationSeverity.HIGH
        else:
            return ViolationSeverity.MEDIUM

    def redact_pii(self, content: str) -> str:
        """Redact all detected PII from content.

        Args:
            content: Text with PII

        Returns:
            Text with PII replaced by [REDACTED]
        """
        redacted = content

        for pii_type in self.pii_types:
            if pii_type in self.PII_PATTERNS:
                pattern = self.PII_PATTERNS[pii_type]
                redacted = pattern.sub(f"[{pii_type.value.upper()}_REDACTED]", redacted)

        # Redact addresses
        if PIIType.ADDRESS in self.pii_types:
            for pattern in self.ADDRESS_PATTERNS:
                redacted = pattern.sub("[ADDRESS_REDACTED]", redacted)

        return redacted
