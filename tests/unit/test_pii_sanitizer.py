"""Unit Tests for PII Sanitizer (Priority 2 Fix - 2025-12-14).

PRIORITY 2 FIX: Tests for PII detection and sanitization to resolve PII leak in complex queries.

This module tests:
1. PIISanitizer.sanitize() - Main sanitization function
2. PIISanitizer.detect_pii() - Detection without redaction (monitoring)
3. PIISanitizer._is_phone_false_positive() - False positive filtering
4. PIISanitizer._is_emergency_pattern() - Emergency number allowlist
5. PIISanitizer.validate_sanitization() - Verify no PII remains

Test Coverage:
- Valid PII detection (SSN, credit cards, emails, phone numbers)
- False positive filtering (911, 311, 511, toll-free numbers)
- Redaction placeholders
- Emergency pattern allowlist
- Multiple PII types in single text
- Edge cases (empty text, no PII, only false positives)

Critical Requirement: ZERO TOLERANCE for PII leaks in production responses.
"""


from backend.src.safety.pii_sanitizer import PIISanitizer, PIIType, sanitize_text


class TestPIISanitizerBasicDetection:
    """Test basic PII detection and sanitization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sanitizer = PIISanitizer(strict_mode=True)

    # ========== SSN DETECTION ==========

    def test_sanitize_ssn(self):
        """Should detect and redact SSN."""
        text = "My SSN is 123-45-6789 for verification."
        sanitized = self.sanitizer.sanitize(text)

        assert "123-45-6789" not in sanitized
        assert "[REDACTED_SSN]" in sanitized
        assert sanitized == "My SSN is [REDACTED_SSN] for verification."

    def test_sanitize_multiple_ssn(self):
        """Should detect and redact multiple SSNs."""
        text = "SSN 123-45-6789 and SSN 987-65-4321 found."
        sanitized = self.sanitizer.sanitize(text)

        assert "123-45-6789" not in sanitized
        assert "987-65-4321" not in sanitized
        assert sanitized.count("[REDACTED_SSN]") == 2

    def test_detect_ssn_without_sanitization(self):
        """Should detect SSN without redacting (for monitoring)."""
        text = "SSN is 123-45-6789."
        detections = self.sanitizer.detect_pii(text)

        assert PIIType.SSN in detections
        assert len(detections[PIIType.SSN]) == 1
        # Should return placeholder, not actual SSN
        assert detections[PIIType.SSN][0] == "[REDACTED_SSN]"

    # ========== CREDIT CARD DETECTION ==========

    def test_sanitize_credit_card_spaces(self):
        """Should detect and redact credit card with spaces."""
        text = "Card number is 1234 5678 9012 3456."
        sanitized = self.sanitizer.sanitize(text)

        assert "1234 5678 9012 3456" not in sanitized
        assert "[REDACTED_CARD]" in sanitized

    def test_sanitize_credit_card_dashes(self):
        """Should detect and redact credit card with dashes."""
        text = "Card: 1234-5678-9012-3456"
        sanitized = self.sanitizer.sanitize(text)

        assert "1234-5678-9012-3456" not in sanitized
        assert "[REDACTED_CARD]" in sanitized

    def test_sanitize_credit_card_no_separators(self):
        """Should detect and redact credit card without separators."""
        text = "Card number: 1234567890123456"
        sanitized = self.sanitizer.sanitize(text)

        assert "1234567890123456" not in sanitized
        assert "[REDACTED_CARD]" in sanitized

    # ========== EMAIL DETECTION ==========

    def test_sanitize_email(self):
        """Should detect and redact email address."""
        text = "Contact me at john.doe@example.com for updates."
        sanitized = self.sanitizer.sanitize(text)

        assert "john.doe@example.com" not in sanitized
        assert "[REDACTED_EMAIL]" in sanitized

    def test_sanitize_multiple_emails(self):
        """Should detect and redact multiple email addresses."""
        text = "Email alice@test.com or bob@example.org."
        sanitized = self.sanitizer.sanitize(text)

        assert "alice@test.com" not in sanitized
        assert "bob@example.org" not in sanitized
        assert sanitized.count("[REDACTED_EMAIL]") == 2

    # ========== PHONE NUMBER DETECTION ==========

    def test_sanitize_phone_dashes(self):
        """Should detect and redact phone number with dashes."""
        text = "Call 555-123-4567 for assistance."
        sanitized = self.sanitizer.sanitize(text)

        assert "555-123-4567" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized

    def test_sanitize_phone_dots(self):
        """Should detect and redact phone number with dots."""
        text = "Phone: 555.123.4567"
        sanitized = self.sanitizer.sanitize(text)

        assert "555.123.4567" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized

    def test_sanitize_phone_spaces(self):
        """Should detect and redact phone number with spaces."""
        text = "Contact: 555 123 4567"
        sanitized = self.sanitizer.sanitize(text)

        assert "555 123 4567" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized

    def test_sanitize_phone_no_separators(self):
        """Should detect and redact phone number without separators."""
        text = "Mobile: 5551234567"
        sanitized = self.sanitizer.sanitize(text)

        assert "5551234567" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized


class TestPIISanitizerFalsePositives:
    """Test false positive filtering (emergency numbers, public hotlines)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sanitizer = PIISanitizer(strict_mode=True)

    # ========== EMERGENCY NUMBERS (SHOULD NOT REDACT) ==========

    def test_emergency_911_not_redacted(self):
        """Emergency number 911 should NOT be redacted."""
        text = "Call 911 for emergency assistance."
        sanitized = self.sanitizer.sanitize(text)

        assert "911" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized

    def test_service_311_not_redacted(self):
        """Service number 311 should NOT be redacted."""
        text = "Call 311 for non-emergency services."
        sanitized = self.sanitizer.sanitize(text)

        assert "311" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized

    def test_traffic_511_not_redacted(self):
        """Traffic info 511 should NOT be redacted."""
        text = "Dial 511 for traffic updates."
        sanitized = self.sanitizer.sanitize(text)

        assert "511" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized

    def test_community_211_not_redacted(self):
        """Community services 211 should NOT be redacted."""
        text = "Call 211 for community resources."
        sanitized = self.sanitizer.sanitize(text)

        assert "211" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized

    def test_mental_health_988_not_redacted(self):
        """Mental health crisis 988 should NOT be redacted."""
        text = "Dial 988 for mental health support."
        sanitized = self.sanitizer.sanitize(text)

        assert "988" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized

    # ========== TOLL-FREE NUMBERS (SHOULD NOT REDACT) ==========

    def test_tollfree_800_not_redacted(self):
        """Toll-free 1-800 number should NOT be redacted."""
        text = "Call 1-800-123-4567 for customer service."
        sanitized = self.sanitizer.sanitize(text)

        assert "1-800-123-4567" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized

    def test_tollfree_888_not_redacted(self):
        """Toll-free 1-888 number should NOT be redacted."""
        text = "Hotline: 1-888-555-1234"
        sanitized = self.sanitizer.sanitize(text)

        assert "1-888-555-1234" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized

    def test_tollfree_877_not_redacted(self):
        """Toll-free 1-877 number should NOT be redacted."""
        text = "Support: 1-877-999-0000"
        sanitized = self.sanitizer.sanitize(text)

        assert "1-877-999-0000" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized

    def test_tollfree_866_not_redacted(self):
        """Toll-free 1-866 number should NOT be redacted."""
        text = "Contact 1-866-123-4567."
        sanitized = self.sanitizer.sanitize(text)

        assert "1-866-123-4567" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized


class TestPIISanitizerMixedContent:
    """Test sanitization with multiple PII types in same text."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sanitizer = PIISanitizer(strict_mode=True)

    def test_sanitize_mixed_pii_types(self):
        """Should sanitize multiple PII types in single text."""
        text = (
            "Contact John at john@example.com or call 555-123-4567. "
            "SSN: 123-45-6789, Card: 1234-5678-9012-3456."
        )
        sanitized = self.sanitizer.sanitize(text)

        # All PII should be redacted
        assert "john@example.com" not in sanitized
        assert "555-123-4567" not in sanitized
        assert "123-45-6789" not in sanitized
        assert "1234-5678-9012-3456" not in sanitized

        # All placeholders should be present
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_PHONE]" in sanitized
        assert "[REDACTED_SSN]" in sanitized
        assert "[REDACTED_CARD]" in sanitized

    def test_sanitize_pii_with_emergency_numbers(self):
        """Should sanitize PII but preserve emergency numbers."""
        text = (
            "For emergencies call 911. "
            "For non-emergencies, contact 555-123-4567 or admin@weather.gov."
        )
        sanitized = self.sanitizer.sanitize(text)

        # Emergency number preserved
        assert "911" in sanitized

        # PII redacted
        assert "555-123-4567" not in sanitized
        assert "admin@weather.gov" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized
        assert "[REDACTED_EMAIL]" in sanitized

    def test_detect_multiple_pii_types(self):
        """Should detect multiple PII types without redaction."""
        text = "Email: test@example.com, Phone: 555-123-4567"
        detections = self.sanitizer.detect_pii(text)

        assert PIIType.EMAIL in detections
        assert PIIType.PHONE in detections
        assert len(detections[PIIType.EMAIL]) == 1
        assert len(detections[PIIType.PHONE]) == 1


class TestPIISanitizerValidation:
    """Test sanitization validation and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sanitizer = PIISanitizer(strict_mode=True)

    def test_validate_sanitization_success(self):
        """Should validate that sanitization removed all PII."""
        original = "Contact john@example.com or 555-123-4567."
        sanitized = self.sanitizer.sanitize(original)

        is_valid = self.sanitizer.validate_sanitization(original, sanitized)
        assert is_valid is True

    def test_validate_sanitization_failure(self):
        """Should detect if PII remains after sanitization."""
        original = "Email: test@example.com"
        # Simulate incomplete sanitization
        incomplete_sanitized = "Email: test@example.com"

        is_valid = self.sanitizer.validate_sanitization(original, incomplete_sanitized)
        assert is_valid is False

    def test_sanitize_empty_text(self):
        """Should handle empty text without errors."""
        text = ""
        sanitized = self.sanitizer.sanitize(text)

        assert sanitized == ""

    def test_sanitize_none_text(self):
        """Should handle None text without errors."""
        text = None
        sanitized = self.sanitizer.sanitize(text)

        assert sanitized is None

    def test_sanitize_no_pii(self):
        """Should return original text if no PII detected."""
        text = "Hurricane Category 5 approaching with 165 mph winds."
        sanitized = self.sanitizer.sanitize(text)

        assert sanitized == text

    def test_sanitize_only_false_positives(self):
        """Should not redact if only false positives detected."""
        text = "Call 911, 311, or 511 for assistance. Hotline: 1-800-123-4567."
        sanitized = self.sanitizer.sanitize(text)

        # All numbers should be preserved (emergency/toll-free)
        assert "911" in sanitized
        assert "311" in sanitized
        assert "511" in sanitized
        assert "1-800-123-4567" in sanitized
        assert "[REDACTED_PHONE]" not in sanitized


class TestPIISanitizerStrictMode:
    """Test strict mode vs non-strict mode behavior."""

    def test_strict_mode_enabled(self):
        """Strict mode should redact all potential PII."""
        sanitizer = PIISanitizer(strict_mode=True)
        text = "Contact 555-123-4567 for updates."
        sanitized = sanitizer.sanitize(text)

        assert "555-123-4567" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized

    def test_non_strict_mode(self):
        """Non-strict mode behavior (currently same as strict)."""
        sanitizer = PIISanitizer(strict_mode=False)
        text = "Contact 555-123-4567 for updates."
        sanitized = sanitizer.sanitize(text)

        # Currently, non-strict mode still redacts (future: could be more permissive)
        assert "555-123-4567" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized


class TestPIISanitizerConvenienceFunction:
    """Test convenience function for quick sanitization."""

    def test_sanitize_text_function_default(self):
        """Should sanitize using convenience function with defaults."""
        text = "Email: test@example.com, Phone: 555-123-4567"
        sanitized = sanitize_text(text)

        assert "test@example.com" not in sanitized
        assert "555-123-4567" not in sanitized
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_PHONE]" in sanitized

    def test_sanitize_text_function_strict(self):
        """Should sanitize using convenience function with strict mode."""
        text = "SSN: 123-45-6789"
        sanitized = sanitize_text(text, strict_mode=True)

        assert "123-45-6789" not in sanitized
        assert "[REDACTED_SSN]" in sanitized

    def test_sanitize_text_function_non_strict(self):
        """Should sanitize using convenience function with non-strict mode."""
        text = "Card: 1234-5678-9012-3456"
        sanitized = sanitize_text(text, strict_mode=False)

        assert "1234-5678-9012-3456" not in sanitized
        assert "[REDACTED_CARD]" in sanitized


class TestPIISanitizerRealWorldScenarios:
    """Test real-world weather agent scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sanitizer = PIISanitizer(strict_mode=True)

    def test_weather_alert_with_contact_info(self):
        """Should sanitize weather alert with contact information."""
        text = (
            "Hurricane warning issued for Miami. Evacuate immediately. "
            "For assistance, call emergency services at 911 or "
            "contact your local office at 305-555-1234. "
            "Updates: weather.alerts@miami.gov"
        )
        sanitized = self.sanitizer.sanitize(text)

        # Emergency number preserved
        assert "911" in sanitized

        # PII redacted
        assert "305-555-1234" not in sanitized
        assert "weather.alerts@miami.gov" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized
        assert "[REDACTED_EMAIL]" in sanitized

        # Core message preserved
        assert "Hurricane warning" in sanitized
        assert "Evacuate immediately" in sanitized

    def test_evacuation_guidance_clean(self):
        """Should preserve evacuation guidance without PII."""
        text = (
            "Category 5 hurricane approaching with 165 mph winds. "
            "Immediate evacuation required for zones A, B, and C. "
            "Storm surge 15-20 feet expected. "
            "Leave within 2 hours. Call 911 if unable to evacuate."
        )
        sanitized = self.sanitizer.sanitize(text)

        # Should be unchanged (no PII, 911 is emergency number)
        assert sanitized == text

    def test_hurricane_status_with_pii(self):
        """Should sanitize hurricane status update with embedded PII."""
        text = (
            "Hurricane Milton (Category 4, 145 mph) tracking toward Tampa Bay. "
            "Landfall expected 6 PM EDT. For shelter info, contact "
            "813-555-9999 or shelters@tampa.gov. SSN verification required: 123-45-6789."
        )
        sanitized = self.sanitizer.sanitize(text)

        # Hurricane details preserved
        assert "Hurricane Milton" in sanitized
        assert "Category 4" in sanitized
        assert "145 mph" in sanitized
        assert "Tampa Bay" in sanitized

        # PII redacted
        assert "813-555-9999" not in sanitized
        assert "shelters@tampa.gov" not in sanitized
        assert "123-45-6789" not in sanitized
        assert "[REDACTED_PHONE]" in sanitized
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_SSN]" in sanitized
