"""
Tests for Output Validator Module.

Level 6c: Self-Evolving Platform

Tests:
1. Length validation
2. Contains/not contains validation
3. Regex validation
4. JSON validation
5. Weather-specific validation
6. Custom validators
"""

import pytest

from backend.src.guardrails.output_validator import (
    OutputValidator,
    RuleType,
    ValidationResult,
    ValidationRule,
)


class TestOutputValidator:
    """Test OutputValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator with basic rules."""
        rules = [
            ValidationRule(
                id="min_length",
                name="Minimum Length",
                rule_type=RuleType.LENGTH,
                parameters={"min": 10, "unit": "words"},
                severity="warning",
                weight=0.5,
            ),
            ValidationRule(
                id="no_placeholder",
                name="No Placeholders",
                rule_type=RuleType.NOT_CONTAINS,
                parameters={"prohibited": ["[TODO]", "[PLACEHOLDER]"]},
                severity="error",
                weight=1.0,
            ),
        ]
        return OutputValidator(rules=rules, pass_threshold=0.7)

    def test_validate_good_content(self, validator):
        """Test validating good content."""
        content = """
        The weather in Miami is sunny with a temperature of 85°F.
        Humidity is at 65% with winds from the southeast at 10 mph.
        """

        result = validator.validate(content)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.score >= 0.7

    def test_validate_short_content(self, validator):
        """Test validating short content."""
        content = "Sunny."

        result = validator.validate(content)

        # Should fail minimum length
        assert result.passed_rules < result.total_rules

    def test_validate_with_placeholder(self, validator):
        """Test validating content with placeholder."""
        content = "The weather is [TODO] with temperature of 85°F."

        result = validator.validate(content)

        assert result.is_valid is False
        assert len(result.errors) > 0


class TestLengthValidation:
    """Test length validation rules."""

    def test_min_length_words(self):
        """Test minimum length in words."""
        rule = ValidationRule(
            id="min_words",
            name="Min Words",
            rule_type=RuleType.LENGTH,
            parameters={"min": 5, "unit": "words"},
        )
        validator = OutputValidator(rules=[rule])

        short_result = validator.validate("One two three.")
        long_result = validator.validate("One two three four five six seven.")

        assert short_result.passed_rules == 0
        assert long_result.passed_rules == 1

    def test_max_length_words(self):
        """Test maximum length in words."""
        rule = ValidationRule(
            id="max_words",
            name="Max Words",
            rule_type=RuleType.LENGTH,
            parameters={"max": 5, "unit": "words"},
        )
        validator = OutputValidator(rules=[rule])

        short_result = validator.validate("One two three.")
        long_result = validator.validate("One two three four five six seven eight nine ten.")

        assert short_result.passed_rules == 1
        assert long_result.passed_rules == 0

    def test_length_chars(self):
        """Test length in characters."""
        rule = ValidationRule(
            id="min_chars",
            name="Min Chars",
            rule_type=RuleType.LENGTH,
            parameters={"min": 20, "unit": "chars"},
        )
        validator = OutputValidator(rules=[rule])

        short_result = validator.validate("Short.")
        long_result = validator.validate("This is a much longer piece of text.")

        assert short_result.passed_rules == 0
        assert long_result.passed_rules == 1


class TestContainsValidation:
    """Test contains validation rules."""

    def test_contains_required(self):
        """Test required content validation."""
        rule = ValidationRule(
            id="has_temp",
            name="Has Temperature",
            rule_type=RuleType.CONTAINS,
            parameters={"required": ["temperature", "forecast"]},
        )
        validator = OutputValidator(rules=[rule])

        good_result = validator.validate("The temperature forecast is 85°F.")
        bad_result = validator.validate("It's sunny today.")

        assert good_result.passed_rules == 1
        assert bad_result.passed_rules == 0

    def test_not_contains_prohibited(self):
        """Test prohibited content validation."""
        rule = ValidationRule(
            id="no_prohibited",
            name="No Prohibited",
            rule_type=RuleType.NOT_CONTAINS,
            parameters={"prohibited": ["ignore", "don't worry"]},
        )
        validator = OutputValidator(rules=[rule])

        good_result = validator.validate("Please evacuate immediately.")
        bad_result = validator.validate("Don't worry about the storm.")

        assert good_result.passed_rules == 1
        assert bad_result.passed_rules == 0

    def test_case_sensitivity(self):
        """Test case sensitivity option."""
        rule = ValidationRule(
            id="case_test",
            name="Case Test",
            rule_type=RuleType.CONTAINS,
            parameters={"required": ["WEATHER"], "case_sensitive": True},
        )
        validator = OutputValidator(rules=[rule])

        upper_result = validator.validate("WEATHER is sunny.")
        lower_result = validator.validate("weather is sunny.")

        assert upper_result.passed_rules == 1
        assert lower_result.passed_rules == 0


class TestRegexValidation:
    """Test regex validation rules."""

    def test_regex_should_match(self):
        """Test regex that should match."""
        rule = ValidationRule(
            id="temp_pattern",
            name="Temperature Pattern",
            rule_type=RuleType.REGEX,
            parameters={"pattern": r"\d+\s*°F", "should_match": True},
        )
        validator = OutputValidator(rules=[rule])

        good_result = validator.validate("Temperature is 85°F.")
        bad_result = validator.validate("Temperature is warm.")

        assert good_result.passed_rules == 1
        assert bad_result.passed_rules == 0

    def test_regex_should_not_match(self):
        """Test regex that should not match."""
        rule = ValidationRule(
            id="no_vague",
            name="No Vague Time",
            rule_type=RuleType.REGEX,
            parameters={"pattern": r"\b(soon|later)\b", "should_match": False},
        )
        validator = OutputValidator(rules=[rule])

        good_result = validator.validate("Rain expected at 3pm.")
        bad_result = validator.validate("Rain expected soon.")

        assert good_result.passed_rules == 1
        assert bad_result.passed_rules == 0

    def test_regex_ignore_case(self):
        """Test regex with ignore case option."""
        rule = ValidationRule(
            id="weather_pattern",
            name="Weather Pattern",
            rule_type=RuleType.REGEX,
            parameters={"pattern": r"WEATHER", "should_match": True, "ignore_case": True},
        )
        validator = OutputValidator(rules=[rule])

        result = validator.validate("The weather is nice.")

        assert result.passed_rules == 1


class TestJSONValidation:
    """Test JSON validation rules."""

    def test_valid_json_content(self):
        """Test valid JSON content."""
        rule = ValidationRule(
            id="json_valid",
            name="Valid JSON",
            rule_type=RuleType.JSON_VALID,
        )
        validator = OutputValidator(rules=[rule])

        json_content = '{"temperature": 85, "conditions": "sunny"}'
        result = validator.validate(json_content)

        assert result.passed_rules == 1

    def test_invalid_json_content(self):
        """Test invalid JSON content."""
        rule = ValidationRule(
            id="json_valid",
            name="Valid JSON",
            rule_type=RuleType.JSON_VALID,
        )
        validator = OutputValidator(rules=[rule])

        invalid_content = "This is not JSON: {broken"
        result = validator.validate(invalid_content)

        assert result.passed_rules == 0

    def test_json_embedded_in_text(self):
        """Test JSON embedded in text."""
        rule = ValidationRule(
            id="json_valid",
            name="Valid JSON",
            rule_type=RuleType.JSON_VALID,
        )
        validator = OutputValidator(rules=[rule])

        content = 'Here is the data: {"temp": 85}'
        result = validator.validate(content)

        assert result.passed_rules == 1


class TestWeatherValidator:
    """Test weather-specific validator."""

    @pytest.fixture
    def weather_validator(self):
        return OutputValidator.weather_validator()

    def test_good_weather_response(self, weather_validator):
        """Test validating good weather response."""
        response = """
        The current weather in Tampa shows sunny conditions with a temperature of 85°F.
        Humidity is at 65% with winds from the southeast at 10 mph. The forecast for
        tomorrow indicates partly cloudy skies with a high of 87°F. No severe weather
        is expected in the area.
        """

        result = weather_validator.validate(response)

        assert result.is_valid is True

    def test_weather_response_with_placeholder(self, weather_validator):
        """Test weather response with placeholder fails."""
        response = "The weather is [placeholder] with temperature of 85°F."

        result = weather_validator.validate(response)

        assert result.is_valid is False

    def test_weather_response_too_short(self, weather_validator):
        """Test short weather response gets warning."""
        response = "Sunny."

        result = weather_validator.validate(response)

        # Should have warnings about length
        assert result.passed_rules < result.total_rules


class TestHurricaneValidator:
    """Test hurricane-specific validator."""

    @pytest.fixture
    def hurricane_validator(self):
        return OutputValidator.hurricane_validator()

    def test_good_hurricane_response(self, hurricane_validator):
        """Test validating good hurricane response."""
        response = """
        According to the National Hurricane Center (NHC), Hurricane Milton is currently
        a Category 4 storm with winds of 130 mph. For your safety, residents in Zone A
        and Zone B should evacuate immediately. Please follow local emergency management
        guidance.
        """

        result = hurricane_validator.validate(response)

        assert result.is_valid is True

    def test_hurricane_downplaying_fails(self, hurricane_validator):
        """Test hurricane response that downplays risks."""
        response = """
        The hurricane is nothing to worry about. It's just a minor storm.
        You can stay home and wait it out.
        """

        result = hurricane_validator.validate(response)

        # Should fail the no-downplaying rule
        assert result.is_valid is False


class TestCustomValidators:
    """Test custom validator functions."""

    def test_register_custom_validator(self):
        """Test registering a custom validator."""

        def word_count_validator(content: str, params: dict) -> bool:
            min_words = params.get("min_words", 10)
            return len(content.split()) >= min_words

        validator = OutputValidator()
        validator.register_custom_validator("word_count", word_count_validator)

        rule = ValidationRule(
            id="custom_wc",
            name="Custom Word Count",
            rule_type=RuleType.CUSTOM,
            parameters={"validator_name": "word_count", "min_words": 5},
        )
        validator.add_rule(rule)

        short_result = validator.validate("One two.")
        long_result = validator.validate("One two three four five six.")

        assert short_result.passed_rules == 0
        assert long_result.passed_rules == 1

    def test_missing_custom_validator(self):
        """Test missing custom validator fails gracefully."""
        validator = OutputValidator()

        rule = ValidationRule(
            id="missing",
            name="Missing Validator",
            rule_type=RuleType.CUSTOM,
            parameters={"validator_name": "nonexistent"},
        )
        validator.add_rule(rule)

        result = validator.validate("Test content.")

        assert result.passed_rules == 0


class TestValidationResult:
    """Test ValidationResult model."""

    def test_result_properties(self):
        """Test result has all expected properties."""
        result = ValidationResult(
            content="Test",
            is_valid=True,
            score=0.9,
            rule_results=[],
            total_rules=5,
            passed_rules=4,
            failed_rules=1,
        )

        assert result.content == "Test"
        assert result.is_valid is True
        assert result.score == 0.9
        assert result.passed_rules == 4
        assert result.failed_rules == 1

    def test_get_rules(self):
        """Test getting all rules."""
        rules = [
            ValidationRule(
                id="r1",
                name="Rule 1",
                rule_type=RuleType.LENGTH,
                parameters={},
            ),
        ]
        validator = OutputValidator(rules=rules)

        retrieved_rules = validator.get_rules()

        assert len(retrieved_rules) == 1
        assert retrieved_rules[0]["id"] == "r1"
