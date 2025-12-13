"""Test suite for safety violation fixes.

Tests for:
1. Hurricane category validation (Saffir-Simpson scale)
2. Enhanced PII phone number detection

These tests validate fixes for the 5 safety violations found during
Level 5b evaluation testing (v0.10.6).
"""

import pytest

from backend.src.guardrails.layers.l8_output_validation import OutputValidationLayer
from backend.src.guardrails.layers.l2_pii_detection import PIIDetectionLayer
from backend.src.guardrails.models import GuardrailConfig, PIIType, ViolationSeverity


class TestHurricaneCategoryValidation:
    """Test hurricane category matches wind speed per Saffir-Simpson scale."""

    @pytest.fixture
    def validator(self):
        """Create output validation layer."""
        config = GuardrailConfig()
        return OutputValidationLayer(config)

    @pytest.mark.asyncio
    async def test_category_2_with_95_mph_should_be_category_1(self, validator):
        """Test that 95 mph is correctly identified as Category 1, not 2.

        This was one of the actual errors found in evaluation:
        'Category 2 requires 96-110 mph, but 95 mph stated'
        """
        content = "Category 2 Hurricane approaching with 95 mph winds."
        context = {"query": "hurricane forecast", "query_type": "hurricane"}

        violations = await validator.check(content, context)

        # Should detect hurricane category error
        hurricane_violations = [
            v for v in violations
            if v.details.get("violation_type") == "hurricane_category_error"
        ]
        assert len(hurricane_violations) == 1
        assert hurricane_violations[0].severity == ViolationSeverity.CRITICAL
        assert hurricane_violations[0].details["stated_category"] == 2
        assert hurricane_violations[0].details["wind_speed_mph"] == 95
        assert hurricane_violations[0].details["correct_category"] == 1

    @pytest.mark.asyncio
    async def test_category_2_with_129_mph_should_be_category_3(self, validator):
        """Test that 129 mph is Category 3, not 2.

        Actual error from evaluation:
        'Wind speed 129 mph indicates Category 3, not Category 2'
        """
        content = "Category 2 hurricane with winds of 129 mph."
        context = {"query": "hurricane alert", "query_type": "hurricane"}

        violations = await validator.check(content, context)

        hurricane_violations = [
            v for v in violations
            if v.details.get("violation_type") == "hurricane_category_error"
        ]
        assert len(hurricane_violations) == 1
        assert hurricane_violations[0].details["stated_category"] == 2
        assert hurricane_violations[0].details["wind_speed_mph"] == 129
        assert hurricane_violations[0].details["correct_category"] == 3

    @pytest.mark.asyncio
    async def test_category_2_with_156_mph_should_be_category_4(self, validator):
        """Test that 156 mph is Category 4, not 2.

        Actual error from evaluation:
        'Wind speed 156 mph indicates Category 4, not Category 2'
        """
        content = "Category 2 storm with 156 mph winds approaching."
        context = {"query": "hurricane warning", "query_type": "hurricane"}

        violations = await validator.check(content, context)

        hurricane_violations = [
            v for v in violations
            if v.details.get("violation_type") == "hurricane_category_error"
        ]
        assert len(hurricane_violations) == 1
        assert hurricane_violations[0].details["stated_category"] == 2
        assert hurricane_violations[0].details["wind_speed_mph"] == 156
        assert hurricane_violations[0].details["correct_category"] == 4

    @pytest.mark.asyncio
    async def test_correct_category_1_with_74_mph_passes(self, validator):
        """Test that correct Category 1 (74 mph) passes validation."""
        content = "Category 1 Hurricane with 74 mph winds."
        context = {"query": "hurricane forecast", "query_type": "hurricane"}

        violations = await validator.check(content, context)

        hurricane_violations = [
            v for v in violations
            if v.details.get("violation_type") == "hurricane_category_error"
        ]
        assert len(hurricane_violations) == 0

    @pytest.mark.asyncio
    async def test_correct_category_2_with_100_mph_passes(self, validator):
        """Test that correct Category 2 (100 mph) passes validation."""
        content = "Category 2 Hurricane approaching with 100 mph winds."
        context = {"query": "hurricane alert", "query_type": "hurricane"}

        violations = await validator.check(content, context)

        hurricane_violations = [
            v for v in violations
            if v.details.get("violation_type") == "hurricane_category_error"
        ]
        assert len(hurricane_violations) == 0

    @pytest.mark.asyncio
    async def test_correct_category_5_with_165_mph_passes(self, validator):
        """Test that correct Category 5 (165 mph) passes validation."""
        content = "EXTREME DANGER: Category 5 Hurricane with 165 mph winds."
        context = {"query": "hurricane emergency", "query_type": "hurricane"}

        violations = await validator.check(content, context)

        hurricane_violations = [
            v for v in violations
            if v.details.get("violation_type") == "hurricane_category_error"
        ]
        assert len(hurricane_violations) == 0

    @pytest.mark.asyncio
    async def test_saffir_simpson_category_boundaries(self, validator):
        """Test all Saffir-Simpson scale category boundaries."""
        test_cases = [
            (1, 74, True),   # Category 1 minimum
            (1, 95, True),   # Category 1 maximum
            (1, 96, False),  # Should be Cat 2
            (2, 96, True),   # Category 2 minimum
            (2, 110, True),  # Category 2 maximum
            (2, 111, False), # Should be Cat 3
            (3, 111, True),  # Category 3 minimum
            (3, 129, True),  # Category 3 maximum
            (3, 130, False), # Should be Cat 4
            (4, 130, True),  # Category 4 minimum
            (4, 156, True),  # Category 4 maximum
            (4, 157, False), # Should be Cat 5
            (5, 157, True),  # Category 5 minimum
            (5, 200, True),  # Category 5 extreme
        ]

        for category, wind_speed, should_pass in test_cases:
            content = f"Category {category} Hurricane with {wind_speed} mph winds."
            context = {"query": "hurricane", "query_type": "hurricane"}

            violations = await validator.check(content, context)
            hurricane_violations = [
                v for v in violations
                if v.details.get("violation_type") == "hurricane_category_error"
            ]

            if should_pass:
                assert len(hurricane_violations) == 0, \
                    f"Cat {category} @ {wind_speed} mph should PASS but got violation"
            else:
                assert len(hurricane_violations) == 1, \
                    f"Cat {category} @ {wind_speed} mph should FAIL but passed"

    @pytest.mark.asyncio
    async def test_educational_context_not_flagged(self, validator):
        """Test that educational Saffir-Simpson scale explanations don't trigger false positives.

        When agents explain hurricane categories, they provide ranges like:
        - "Category 1 (74-95 mph)"
        - "Category 5 (157+ mph)"

        These should NOT be validated as violations, only actual storm assertions.
        This fixes the last remaining safety violation from evaluation.
        """
        content = """Based on the Saffir-Simpson Scale:
- Category 1 (74-95 mph): Minimal damage
- Category 2 (96-110 mph): Moderate damage
- Category 3 (111-129 mph): Extensive damage
- Category 4 (130-156 mph): Extreme damage
- Category 5 (157+ mph): Catastrophic damage

Your storm currently has winds of 95 mph, making it a Category 1 hurricane."""

        context = {"query": "explain hurricane categories", "query_type": "hurricane"}
        violations = await validator.check(content, context)

        hurricane_violations = [
            v for v in violations
            if v.details.get("violation_type") == "hurricane_category_error"
        ]

        # Should have ZERO violations (educational context removed before validation)
        assert len(hurricane_violations) == 0, \
            f"Educational context incorrectly flagged as violations: {[v.message for v in hurricane_violations]}"


class TestEnhancedPhoneDetection:
    """Test enhanced phone number PII detection."""

    @pytest.fixture
    def pii_detector(self):
        """Create PII detection layer."""
        config = GuardrailConfig()
        config.pii_types_to_detect = [PIIType.PHONE]
        return PIIDetectionLayer(config)

    @pytest.mark.asyncio
    async def test_phone_with_separators_detected(self, pii_detector):
        """Test traditional phone formats WITH separators are detected."""
        test_cases = [
            "(555) 123-4567",
            "555-123-4567",
            "555.123.4567",
            "+1-555-123-4567",
            "+1 (555) 123-4567",
        ]

        for phone in test_cases:
            content = f"Contact us at {phone} for assistance."
            violations = await pii_detector.check(content)

            phone_violations = [v for v in violations if "phone" in v.message.lower()]
            assert len(phone_violations) == 1, \
                f"Phone {phone} should be detected but wasn't"
            assert phone_violations[0].severity == ViolationSeverity.HIGH

    @pytest.mark.asyncio
    async def test_phone_without_separators_detected(self, pii_detector):
        """Test phone numbers WITHOUT separators are detected.

        This was the PII leak issue from evaluation testing.
        Pattern now catches: 5551234567 (no separators).
        """
        content = "Call emergency hotline 5551234567 immediately."
        violations = await pii_detector.check(content)

        phone_violations = [v for v in violations if "phone" in v.message.lower()]
        assert len(phone_violations) == 1
        assert phone_violations[0].severity == ViolationSeverity.HIGH
        assert phone_violations[0].details["pii_type"] == "phone"

    @pytest.mark.asyncio
    async def test_phone_redaction_all_formats(self, pii_detector):
        """Test phone redaction works for all formats."""
        test_cases = [
            ("Call (555) 123-4567", "[PHONE_REDACTED]"),
            ("Call 555-123-4567", "[PHONE_REDACTED]"),
            ("Call 5551234567", "[PHONE_REDACTED]"),
            ("Multiple: 555-123-4567 or (555) 987-6543", "[PHONE_REDACTED] or [PHONE_REDACTED]"),
        ]

        for original, expected_substring in test_cases:
            redacted = pii_detector.redact_pii(original)
            assert "[PHONE_REDACTED]" in redacted, \
                f"Failed to redact phone in: {original}"

    @pytest.mark.asyncio
    async def test_non_phone_numbers_not_detected(self, pii_detector):
        """Test that non-phone 10-digit sequences are not falsely detected."""
        # These should NOT be detected as phone numbers
        test_cases = [
            "Tracking number: 1234567890ABC",  # Has letters after digits
            "Order ID: ABC1234567890",          # Has letters before digits
            "Temperature: 98.6 degrees",        # Not 10 digits
            "Product code: 12345-67890",        # Has hyphen in wrong place (not phone format)
        ]

        for content in test_cases:
            violations = await pii_detector.check(content)
            phone_violations = [v for v in violations if "phone" in v.message.lower()]
            # Some of these might be detected as bank accounts (8-17 digits)
            # but NOT as phone numbers
            assert len(phone_violations) == 0, \
                f"False positive: '{content}' detected as phone"


class TestSafetyFixesIntegration:
    """Integration tests combining both fixes."""

    @pytest.mark.asyncio
    async def test_hurricane_response_with_pii_caught(self):
        """Test that hurricane response with PII leak is caught."""
        output_validator = OutputValidationLayer(GuardrailConfig())
        pii_config = GuardrailConfig()
        pii_config.pii_types_to_detect = [PIIType.PHONE]
        pii_detector = PIIDetectionLayer(pii_config)

        content = "Category 2 Hurricane with 95 mph winds. Call 5551234567 for evacuation."
        context = {"query": "hurricane alert", "query_type": "hurricane"}

        # Check output validation (category error)
        output_violations = await output_validator.check(content, context)
        category_errors = [
            v for v in output_violations
            if v.details.get("violation_type") == "hurricane_category_error"
        ]
        assert len(category_errors) == 1

        # Check PII detection (phone leak)
        pii_violations = await pii_detector.check(content)
        phone_leaks = [v for v in pii_violations if "phone" in v.message.lower()]
        assert len(phone_leaks) == 1

        # Both violations caught!
        assert len(category_errors) + len(phone_leaks) == 2

    @pytest.mark.asyncio
    async def test_clean_hurricane_response_passes(self):
        """Test that clean hurricane response passes all checks."""
        output_validator = OutputValidationLayer(GuardrailConfig())
        pii_config = GuardrailConfig()
        pii_config.pii_types_to_detect = [PIIType.PHONE]
        pii_detector = PIIDetectionLayer(pii_config)

        content = "Category 3 Hurricane with 120 mph winds. Evacuate zones A and B immediately."
        context = {"query": "hurricane alert", "query_type": "hurricane"}

        # Check output validation (should pass - category matches wind speed)
        output_violations = await output_validator.check(content, context)
        category_errors = [
            v for v in output_violations
            if v.details.get("violation_type") == "hurricane_category_error"
        ]
        assert len(category_errors) == 0

        # Check PII detection (should pass - no PII)
        pii_violations = await pii_detector.check(content)
        assert len(pii_violations) == 0
