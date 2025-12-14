"""
Tests for Content Filter Module.

Level 6c: Self-Evolving Platform

Tests:
1. Safe content detection
2. Category-based filtering
3. PII detection
4. Weather safety filtering
5. Configurable sensitivity
"""

import pytest
from backend.src.guardrails.content_filter import (
    ContentFilter,
    ContentFilterResult,
    FilterCategory,
)


class TestContentFilter:
    """Test ContentFilter class."""

    @pytest.fixture
    def content_filter(self):
        """Create default content filter."""
        return ContentFilter()

    def test_filter_safe_content(self, content_filter):
        """Test filtering safe content."""
        safe_content = "The weather in Miami is sunny with a temperature of 85°F."

        result = content_filter.filter(safe_content)

        assert isinstance(result, ContentFilterResult)
        assert result.is_safe is True
        assert len(result.categories_flagged) == 0

    def test_filter_pii_ssn(self, content_filter):
        """Test filtering SSN."""
        content_with_ssn = "My social security number is 123-45-6789."

        result = content_filter.filter(content_with_ssn)

        assert result.is_safe is False
        assert FilterCategory.PII in result.categories_flagged

    def test_filter_pii_credit_card(self, content_filter):
        """Test filtering credit card numbers."""
        content_with_cc = "My card number is 1234567890123456."

        result = content_filter.filter(content_with_cc)

        assert result.is_safe is False
        assert FilterCategory.PII in result.categories_flagged

    def test_filter_pii_email(self, content_filter):
        """Test filtering email addresses."""
        content_with_email = "Contact me at user@example.com for more info."

        result = content_filter.filter(content_with_email)

        assert result.is_safe is False
        assert FilterCategory.PII in result.categories_flagged

    def test_filter_pii_phone(self, content_filter):
        """Test filtering phone numbers."""
        content_with_phone = "Call me at 555-123-4567."

        result = content_filter.filter(content_with_phone)

        assert result.is_safe is False
        assert FilterCategory.PII in result.categories_flagged

    def test_filter_dangerous_advice(self, content_filter):
        """Test filtering dangerous weather advice."""
        dangerous_content = "Don't evacuate, the hurricane will pass."

        result = content_filter.filter(dangerous_content)

        assert result.is_safe is False
        assert FilterCategory.DANGEROUS_ADVICE in result.categories_flagged


class TestContentFilterRedaction:
    """Test content redaction."""

    @pytest.fixture
    def content_filter(self):
        return ContentFilter()

    def test_redact_pii(self, content_filter):
        """Test redacting PII."""
        content = "My SSN is 123-45-6789 and email is user@example.com."

        result = content_filter.filter(content)

        assert result.filtered_content is not None
        assert "123-45-6789" not in result.filtered_content
        assert "user@example.com" not in result.filtered_content
        assert "[REDACTED]" in result.filtered_content


class TestWeatherSafetyFilter:
    """Test weather-specific safety filtering."""

    @pytest.fixture
    def content_filter(self):
        return ContentFilter()

    def test_detect_dont_evacuate(self, content_filter):
        """Test detecting 'don't evacuate' advice."""
        content = "You don't need to evacuate, it's just a small storm."

        result = content_filter.filter_weather_safety(content)

        assert result.is_safe is False

    def test_detect_ignore_warning(self, content_filter):
        """Test detecting 'ignore warning' advice."""
        content = "You can ignore the warning, it's overblown."

        result = content_filter.filter_weather_safety(content)

        assert result.is_safe is False

    def test_detect_drive_through_flood(self, content_filter):
        """Test detecting 'drive through flood' advice."""
        content = "You can drive through the flooded road, it's not deep."

        result = content_filter.filter_weather_safety(content)

        assert result.is_safe is False

    def test_safe_weather_advice(self, content_filter):
        """Test safe weather advice passes."""
        content = """
        Based on NWS guidance, you should evacuate if you're in Zone A.
        Do not drive through flooded roads. Seek shelter immediately.
        """

        result = content_filter.filter_weather_safety(content)

        assert result.is_safe is True


class TestFilterConfiguration:
    """Test filter configuration options."""

    def test_custom_categories(self):
        """Test filter with custom categories."""
        filter = ContentFilter(enabled_categories=[FilterCategory.PII])

        # Should only filter PII
        content = "Email: user@example.com. Some violent text."

        result = filter.filter(content)

        # Should flag PII but not violence
        if not result.is_safe:
            assert FilterCategory.PII in result.categories_flagged
            assert FilterCategory.VIOLENCE not in result.categories_flagged

    def test_sensitivity_setting(self):
        """Test sensitivity adjustment."""
        # High sensitivity
        high_filter = ContentFilter(sensitivity=0.9)
        # Low sensitivity
        low_filter = ContentFilter(sensitivity=0.1)

        content = "Some borderline content."

        high_result = high_filter.filter(content)
        low_result = low_filter.filter(content)

        # Both should pass for safe content
        # But confidence might differ

    def test_add_custom_pattern(self):
        """Test adding custom pattern."""
        filter = ContentFilter()
        filter.add_pattern(FilterCategory.SPAM, r"\bfree money\b")

        content = "Get free money now!"

        result = filter.filter(content)

        # Should detect spam
        assert FilterCategory.SPAM in result.categories_flagged

    def test_get_enabled_categories(self):
        """Test getting enabled categories."""
        filter = ContentFilter()
        categories = filter.get_enabled_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0

    def test_set_sensitivity(self):
        """Test setting sensitivity."""
        filter = ContentFilter(sensitivity=0.5)
        filter.set_sensitivity(0.8)

        assert filter.sensitivity == 0.8

        # Should clamp to [0, 1]
        filter.set_sensitivity(1.5)
        assert filter.sensitivity == 1.0

        filter.set_sensitivity(-0.5)
        assert filter.sensitivity == 0.0


class TestPIIFiltering:
    """Test PII-specific filtering."""

    def test_filter_pii_method(self):
        """Test filter_pii method."""
        filter = ContentFilter()

        content = "My SSN is 123-45-6789. The weather is nice."

        result = filter.filter_pii(content)

        assert FilterCategory.PII in result.categories_flagged


class TestFilterResult:
    """Test ContentFilterResult model."""

    def test_result_properties(self):
        """Test result has all expected properties."""
        result = ContentFilterResult(
            content="Test content",
            is_safe=True,
            categories_flagged=[],
            confidence=0.95,
            details={},
        )

        assert result.content == "Test content"
        assert result.is_safe is True
        assert result.confidence == 0.95
        assert result.filtered_content is None

    def test_result_with_filtered_content(self):
        """Test result with filtered content."""
        result = ContentFilterResult(
            content="Original content",
            is_safe=False,
            categories_flagged=[FilterCategory.PII],
            confidence=0.9,
            details={"pii": {"matches": ["123-45-6789"]}},
            filtered_content="[REDACTED] content",
        )

        assert result.filtered_content == "[REDACTED] content"
        assert len(result.categories_flagged) == 1
