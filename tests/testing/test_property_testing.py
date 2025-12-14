"""
Tests for Property-Based Testing Module.

Level 6c: Self-Evolving Platform

Tests:
1. Property test runner functionality
2. Weather generators
3. Weather property tests
4. Edge case discovery
"""

import pytest
from backend.src.testing.property_testing import (
    PropertyTestRunner,
    PropertyTestResult,
    WeatherGenerators,
    WeatherPropertyTests,
    run_weather_properties,
)


class TestPropertyTestRunner:
    """Test PropertyTestRunner class."""

    @pytest.fixture
    def runner(self):
        """Create property test runner."""
        return PropertyTestRunner(max_examples=50, seed=42)

    def test_initialization(self, runner):
        """Test runner initialization."""
        assert runner.max_examples == 50
        assert runner.results == []

    def test_run_simple_property(self, runner):
        """Test running a simple property."""
        def always_true(x: int) -> bool:
            return x >= 0

        result = runner.run_property(
            "positive_numbers",
            always_true,
            {"x": lambda: abs(hash(str(__import__("random").random()))) % 100},
        )

        assert isinstance(result, PropertyTestResult)
        assert result.property_name == "positive_numbers"
        assert result.passed is True
        assert result.failures == 0

    def test_run_failing_property(self, runner):
        """Test running a failing property."""
        def always_fails(x: int) -> bool:
            return False

        result = runner.run_property(
            "always_fails",
            always_fails,
            {"x": lambda: 1},
        )

        assert result.passed is False
        assert result.failures > 0
        assert len(result.counterexamples) > 0

    def test_run_property_with_exception(self, runner):
        """Test property that raises exception."""
        def raises_exception(x: int) -> bool:
            raise ValueError("Test error")

        result = runner.run_property(
            "exception_property",
            raises_exception,
            {"x": lambda: 1},
        )

        assert result.passed is False
        assert "Test error" in result.error_message

    def test_run_all_properties(self, runner):
        """Test running multiple properties."""
        properties = [
            ("prop1", lambda x: x >= 0, {"x": lambda: 1}),
            ("prop2", lambda x: x < 100, {"x": lambda: 50}),
        ]

        results = runner.run_all(properties)

        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_get_summary(self, runner):
        """Test getting summary."""
        # Run some properties
        runner.run_property("pass", lambda x: True, {"x": lambda: 1})
        runner.run_property("fail", lambda x: False, {"x": lambda: 1})

        summary = runner.get_summary()

        assert summary["total_properties"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 50.0

    def test_clear_results(self, runner):
        """Test clearing results."""
        runner.run_property("test", lambda x: True, {"x": lambda: 1})
        assert len(runner.results) == 1

        runner.clear_results()
        assert len(runner.results) == 0

    def test_counterexamples_limited(self, runner):
        """Test that counterexamples are limited to 5."""
        runner.max_examples = 100

        result = runner.run_property(
            "many_failures",
            lambda x: False,
            {"x": lambda: 1},
        )

        assert len(result.counterexamples) <= 5

    def test_execution_time_tracked(self, runner):
        """Test that execution time is tracked."""
        result = runner.run_property(
            "timed",
            lambda x: True,
            {"x": lambda: 1},
        )

        assert result.execution_time_ms > 0


class TestWeatherGenerators:
    """Test WeatherGenerators class."""

    def test_temperature_f_range(self):
        """Test Fahrenheit temperature is in valid range."""
        for _ in range(100):
            temp = WeatherGenerators.temperature_f()
            assert -50 <= temp <= 130

    def test_temperature_c_range(self):
        """Test Celsius temperature is in valid range."""
        for _ in range(100):
            temp = WeatherGenerators.temperature_c()
            assert -45 <= temp <= 55

    def test_humidity_pct_range(self):
        """Test humidity is in valid range."""
        for _ in range(100):
            humidity = WeatherGenerators.humidity_pct()
            assert 0 <= humidity <= 100

    def test_wind_speed_mph_range(self):
        """Test wind speed is in valid range."""
        for _ in range(100):
            wind = WeatherGenerators.wind_speed_mph()
            assert 0 <= wind <= 200

    def test_hurricane_category_range(self):
        """Test hurricane category is valid."""
        for _ in range(100):
            cat = WeatherGenerators.hurricane_category()
            assert 1 <= cat <= 5

    def test_hurricane_wind_speed_range(self):
        """Test hurricane wind speed is valid."""
        for _ in range(100):
            wind = WeatherGenerators.hurricane_wind_speed()
            assert 74 <= wind <= 200

    def test_location_valid(self):
        """Test location is from valid list."""
        valid_cities = {
            "Miami", "Tampa", "Orlando", "Jacksonville", "Fort Lauderdale",
            "New Orleans", "Houston", "Galveston", "Mobile", "Pensacola",
        }
        for _ in range(100):
            loc = WeatherGenerators.location()
            assert loc in valid_cities

    def test_evacuation_zone_valid(self):
        """Test evacuation zone is valid."""
        valid_zones = {"A", "B", "C", "D", "E", "None"}
        for _ in range(100):
            zone = WeatherGenerators.evacuation_zone()
            assert zone in valid_zones

    def test_time_zone_valid(self):
        """Test time zone is valid."""
        valid_zones = {"ET", "CT", "MT", "PT", "UTC"}
        for _ in range(100):
            tz = WeatherGenerators.time_zone()
            assert tz in valid_zones

    def test_query_contains_location(self):
        """Test query contains a location."""
        valid_cities = {
            "miami", "tampa", "orlando", "jacksonville", "fort lauderdale",
            "new orleans", "houston", "galveston", "mobile", "pensacola",
        }
        for _ in range(50):
            query = WeatherGenerators.query().lower()
            assert any(city in query for city in valid_cities)

    def test_response_contains_weather_data(self):
        """Test response contains weather data."""
        for _ in range(50):
            response = WeatherGenerators.response()
            # Should contain temperature
            assert "°F" in response
            # Should contain humidity
            assert "%" in response

    def test_invalid_string_returns_edge_cases(self):
        """Test invalid string generator returns edge cases."""
        edge_cases_found = set()
        for _ in range(100):
            s = WeatherGenerators.invalid_string()
            edge_cases_found.add(s[:10] if len(s) > 10 else s)

        # Should have found various edge cases
        assert len(edge_cases_found) > 1


class TestWeatherPropertyTests:
    """Test WeatherPropertyTests class."""

    def test_hurricane_category_wind_speed_cat1(self):
        """Test Cat 1 wind speed matching."""
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(1, 74) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(1, 85) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(1, 95) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(1, 96) is False

    def test_hurricane_category_wind_speed_cat2(self):
        """Test Cat 2 wind speed matching."""
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(2, 96) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(2, 110) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(2, 111) is False

    def test_hurricane_category_wind_speed_cat3(self):
        """Test Cat 3 wind speed matching."""
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(3, 111) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(3, 129) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(3, 130) is False

    def test_hurricane_category_wind_speed_cat4(self):
        """Test Cat 4 wind speed matching."""
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(4, 130) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(4, 156) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(4, 157) is False

    def test_hurricane_category_wind_speed_cat5(self):
        """Test Cat 5 wind speed matching."""
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(5, 157) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(5, 185) is True
        assert WeatherPropertyTests.hurricane_category_matches_wind_speed(5, 200) is True

    def test_temperature_valid_range(self):
        """Test temperature validation."""
        assert WeatherPropertyTests.temperature_valid_range(-128.6) is True
        assert WeatherPropertyTests.temperature_valid_range(134.1) is True
        assert WeatherPropertyTests.temperature_valid_range(75.0) is True
        assert WeatherPropertyTests.temperature_valid_range(-130) is False
        assert WeatherPropertyTests.temperature_valid_range(140) is False

    def test_humidity_percentage_valid(self):
        """Test humidity validation."""
        assert WeatherPropertyTests.humidity_percentage_valid(0) is True
        assert WeatherPropertyTests.humidity_percentage_valid(50) is True
        assert WeatherPropertyTests.humidity_percentage_valid(100) is True
        assert WeatherPropertyTests.humidity_percentage_valid(-1) is False
        assert WeatherPropertyTests.humidity_percentage_valid(101) is False

    def test_wind_speed_non_negative(self):
        """Test wind speed validation."""
        assert WeatherPropertyTests.wind_speed_non_negative(0) is True
        assert WeatherPropertyTests.wind_speed_non_negative(100) is True
        assert WeatherPropertyTests.wind_speed_non_negative(-1) is False

    def test_evacuation_zone_format(self):
        """Test evacuation zone format."""
        assert WeatherPropertyTests.evacuation_zone_format("A") is True
        assert WeatherPropertyTests.evacuation_zone_format("B") is True
        assert WeatherPropertyTests.evacuation_zone_format("None") is True
        assert WeatherPropertyTests.evacuation_zone_format("X") is False
        assert WeatherPropertyTests.evacuation_zone_format("") is False

    def test_response_contains_temperature(self):
        """Test response contains temperature."""
        assert WeatherPropertyTests.response_contains_temperature("Temperature is 85°F") is True
        assert WeatherPropertyTests.response_contains_temperature("It's 30°C today") is True
        assert WeatherPropertyTests.response_contains_temperature("It's warm today") is False

    def test_response_no_injection(self):
        """Test response has no injection."""
        assert WeatherPropertyTests.response_no_injection("Normal weather text") is True
        assert WeatherPropertyTests.response_no_injection("<script>alert('xss')</script>") is False
        assert WeatherPropertyTests.response_no_injection("'; DROP TABLE users; --") is False

    def test_fahrenheit_celsius_conversion(self):
        """Test F/C conversion is reversible."""
        assert WeatherPropertyTests.fahrenheit_celsius_conversion(32) is True
        assert WeatherPropertyTests.fahrenheit_celsius_conversion(100) is True
        assert WeatherPropertyTests.fahrenheit_celsius_conversion(-40) is True

    def test_time_zone_abbreviation_valid(self):
        """Test time zone abbreviation validation."""
        assert WeatherPropertyTests.time_zone_abbreviation_valid("ET") is True
        assert WeatherPropertyTests.time_zone_abbreviation_valid("UTC") is True
        assert WeatherPropertyTests.time_zone_abbreviation_valid("EST") is True
        assert WeatherPropertyTests.time_zone_abbreviation_valid("XYZ") is False

    def test_get_all_properties_returns_list(self):
        """Test get_all_properties returns valid list."""
        properties = WeatherPropertyTests.get_all_properties()

        assert isinstance(properties, list)
        assert len(properties) > 0

        for name, fn, generators in properties:
            assert isinstance(name, str)
            assert callable(fn)
            assert isinstance(generators, dict)


class TestRunWeatherProperties:
    """Test run_weather_properties convenience function."""

    def test_run_weather_properties(self):
        """Test convenience function returns summary."""
        summary = run_weather_properties(max_examples=20)

        assert "total_properties" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "pass_rate" in summary
        assert summary["total_properties"] > 0

    def test_run_weather_properties_with_custom_examples(self):
        """Test with custom max_examples."""
        summary = run_weather_properties(max_examples=10)

        # Should have run with 10 examples per property
        assert summary["total_iterations"] == summary["total_properties"] * 10


class TestPropertyTestResult:
    """Test PropertyTestResult model."""

    def test_default_values(self):
        """Test default values."""
        result = PropertyTestResult(
            property_name="test",
            passed=True,
        )

        assert result.property_name == "test"
        assert result.passed is True
        assert result.iterations == 100
        assert result.failures == 0
        assert result.counterexamples == []
        assert result.error_message == ""
        assert result.execution_time_ms == 0.0

    def test_with_failures(self):
        """Test with failure data."""
        result = PropertyTestResult(
            property_name="failing",
            passed=False,
            iterations=50,
            failures=5,
            counterexamples=[{"x": 1}, {"x": 2}],
            error_message="Test failed",
            execution_time_ms=123.45,
        )

        assert result.passed is False
        assert result.failures == 5
        assert len(result.counterexamples) == 2
        assert result.execution_time_ms == 123.45
