"""
Property-Based Testing Module.

Level 6c: Self-Evolving Platform

Provides Hypothesis-style property testing for:
1. Weather data validation
2. Response format verification
3. Safety constraint checking
4. Edge case discovery

Target: Discover edge cases automatically through property testing
"""

from typing import Any, Callable
import logging
import random
import string
import re
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PropertyTestResult(BaseModel):
    """Result of a property test run."""

    property_name: str
    passed: bool
    iterations: int = 100
    failures: int = 0
    counterexamples: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str = ""
    execution_time_ms: float = 0.0


class PropertyTestRunner:
    """
    Run property-based tests using generated inputs.

    Provides Hypothesis-style testing without requiring the Hypothesis library.
    """

    def __init__(
        self,
        max_examples: int = 100,
        seed: int | None = None,
    ):
        """
        Initialize property test runner.

        Args:
            max_examples: Maximum examples to generate per property
            seed: Optional random seed for reproducibility
        """
        self.max_examples = max_examples
        if seed is not None:
            random.seed(seed)

        self.results: list[PropertyTestResult] = []

        logger.info(f"PropertyTestRunner initialized | max_examples={max_examples}")

    def run_property(
        self,
        property_name: str,
        property_fn: Callable[..., bool],
        generators: dict[str, Callable[[], Any]],
    ) -> PropertyTestResult:
        """
        Run a single property test.

        Args:
            property_name: Name of the property being tested
            property_fn: Function that returns True if property holds
            generators: Dict of parameter_name -> generator function

        Returns:
            PropertyTestResult with test outcome
        """
        import time
        start_time = time.time()

        failures = 0
        counterexamples: list[dict[str, Any]] = []
        error_message = ""

        for _ in range(self.max_examples):
            # Generate inputs
            inputs = {name: gen() for name, gen in generators.items()}

            try:
                result = property_fn(**inputs)
                if not result:
                    failures += 1
                    counterexamples.append(inputs)
                    if len(counterexamples) >= 5:  # Limit counterexamples
                        break
            except Exception as e:
                failures += 1
                counterexamples.append({"inputs": inputs, "error": str(e)})
                error_message = str(e)
                break

        execution_time_ms = (time.time() - start_time) * 1000

        result = PropertyTestResult(
            property_name=property_name,
            passed=failures == 0,
            iterations=self.max_examples,
            failures=failures,
            counterexamples=counterexamples[:5],
            error_message=error_message,
            execution_time_ms=round(execution_time_ms, 2),
        )

        self.results.append(result)

        logger.info(
            f"Property test '{property_name}' | passed={result.passed} | "
            f"failures={failures}/{self.max_examples}"
        )

        return result

    def run_all(
        self,
        properties: list[tuple[str, Callable, dict[str, Callable]]],
    ) -> list[PropertyTestResult]:
        """
        Run multiple property tests.

        Args:
            properties: List of (name, property_fn, generators) tuples

        Returns:
            List of PropertyTestResult
        """
        results = []
        for name, prop_fn, generators in properties:
            result = self.run_property(name, prop_fn, generators)
            results.append(result)
        return results

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all test results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        return {
            "total_properties": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total * 100, 2) if total > 0 else 0,
            "total_iterations": sum(r.iterations for r in self.results),
            "total_time_ms": sum(r.execution_time_ms for r in self.results),
        }

    def clear_results(self) -> None:
        """Clear test results."""
        self.results = []


# Generators for weather-domain testing
class WeatherGenerators:
    """Generator functions for weather-related test data."""

    @staticmethod
    def temperature_f() -> float:
        """Generate temperature in Fahrenheit (-50 to 130)."""
        return round(random.uniform(-50, 130), 1)

    @staticmethod
    def temperature_c() -> float:
        """Generate temperature in Celsius (-45 to 55)."""
        return round(random.uniform(-45, 55), 1)

    @staticmethod
    def humidity_pct() -> float:
        """Generate humidity percentage (0-100)."""
        return round(random.uniform(0, 100), 1)

    @staticmethod
    def wind_speed_mph() -> float:
        """Generate wind speed in mph (0-200)."""
        return round(random.uniform(0, 200), 1)

    @staticmethod
    def hurricane_category() -> int:
        """Generate hurricane category (1-5)."""
        return random.randint(1, 5)

    @staticmethod
    def hurricane_wind_speed() -> float:
        """Generate hurricane wind speed (74-200 mph)."""
        return round(random.uniform(74, 200), 1)

    @staticmethod
    def location() -> str:
        """Generate a random location."""
        cities = [
            "Miami", "Tampa", "Orlando", "Jacksonville", "Fort Lauderdale",
            "New Orleans", "Houston", "Galveston", "Mobile", "Pensacola",
        ]
        return random.choice(cities)

    @staticmethod
    def evacuation_zone() -> str:
        """Generate evacuation zone."""
        return random.choice(["A", "B", "C", "D", "E", "None"])

    @staticmethod
    def time_zone() -> str:
        """Generate time zone."""
        return random.choice(["ET", "CT", "MT", "PT", "UTC"])

    @staticmethod
    def query() -> str:
        """Generate weather query."""
        templates = [
            "What's the weather in {location}?",
            "Will it rain in {location} tomorrow?",
            "Is there a hurricane warning for {location}?",
            "What's the forecast for {location}?",
            "Should I evacuate from {location}?",
        ]
        template = random.choice(templates)
        return template.format(location=WeatherGenerators.location())

    @staticmethod
    def response() -> str:
        """Generate weather response."""
        temp = WeatherGenerators.temperature_f()
        humidity = WeatherGenerators.humidity_pct()
        location = WeatherGenerators.location()

        templates = [
            f"The weather in {location} is {temp}°F with {humidity}% humidity.",
            f"Current conditions in {location}: {temp}°F, humidity {humidity}%.",
            f"{location} forecast: Temperature {temp}°F, humidity {humidity}%.",
        ]
        return random.choice(templates)

    @staticmethod
    def invalid_string() -> str:
        """Generate invalid/edge case string."""
        edge_cases = [
            "",
            " ",
            "\n\t",
            "a" * 10000,  # Very long
            "🌪️🌊⛈️",  # Unicode/emoji
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE weather; --",  # SQL injection
        ]
        return random.choice(edge_cases)


class WeatherPropertyTests:
    """Collection of weather-domain property tests."""

    @staticmethod
    def hurricane_category_matches_wind_speed(category: int, wind_speed: float) -> bool:
        """
        Property: Hurricane category must match wind speed per Saffir-Simpson scale.

        Cat 1: 74-95 mph
        Cat 2: 96-110 mph
        Cat 3: 111-129 mph
        Cat 4: 130-156 mph
        Cat 5: 157+ mph
        """
        category_ranges = {
            1: (74, 95),
            2: (96, 110),
            3: (111, 129),
            4: (130, 156),
            5: (157, float("inf")),
        }

        min_speed, max_speed = category_ranges.get(category, (0, 0))
        return min_speed <= wind_speed <= max_speed

    @staticmethod
    def temperature_valid_range(temp_f: float) -> bool:
        """
        Property: Temperature must be in valid Earth range.

        Valid range: -128.6°F to 134.1°F (recorded extremes on Earth)
        """
        return -128.6 <= temp_f <= 134.1

    @staticmethod
    def humidity_percentage_valid(humidity: float) -> bool:
        """
        Property: Humidity must be 0-100%.
        """
        return 0 <= humidity <= 100

    @staticmethod
    def wind_speed_non_negative(wind_speed: float) -> bool:
        """
        Property: Wind speed must be non-negative.
        """
        return wind_speed >= 0

    @staticmethod
    def evacuation_zone_format(zone: str) -> bool:
        """
        Property: Evacuation zone must be single letter A-E or 'None'.
        """
        return zone in ["A", "B", "C", "D", "E", "None"]

    @staticmethod
    def response_contains_temperature(response: str) -> bool:
        """
        Property: Weather response should contain temperature.
        """
        # Check for temperature pattern
        temp_pattern = r"\d+\s*°[FC]"
        return bool(re.search(temp_pattern, response))

    @staticmethod
    def response_no_injection(response: str) -> bool:
        """
        Property: Response should not contain injection attempts.
        """
        injection_patterns = [
            r"<script>",
            r"DROP\s+TABLE",
            r"--\s*$",
            r";\s*DELETE",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False
        return True

    @staticmethod
    def fahrenheit_celsius_conversion(temp_f: float) -> bool:
        """
        Property: F to C conversion should be reversible.

        C = (F - 32) * 5/9
        F = C * 9/5 + 32
        """
        temp_c = (temp_f - 32) * 5 / 9
        recovered_f = temp_c * 9 / 5 + 32
        return abs(temp_f - recovered_f) < 0.001

    @staticmethod
    def time_zone_abbreviation_valid(tz: str) -> bool:
        """
        Property: Time zone should be valid abbreviation.
        """
        valid_zones = ["ET", "CT", "MT", "PT", "UTC", "EST", "CST", "MST", "PST", "EDT", "CDT", "MDT", "PDT"]
        return tz in valid_zones

    @classmethod
    def get_all_properties(cls) -> list[tuple[str, Callable, dict[str, Callable]]]:
        """Get all property tests with their generators."""
        return [
            (
                "hurricane_category_matches_wind_speed",
                cls.hurricane_category_matches_wind_speed,
                {
                    "category": WeatherGenerators.hurricane_category,
                    "wind_speed": WeatherGenerators.hurricane_wind_speed,
                },
            ),
            (
                "temperature_valid_range",
                cls.temperature_valid_range,
                {"temp_f": WeatherGenerators.temperature_f},
            ),
            (
                "humidity_percentage_valid",
                cls.humidity_percentage_valid,
                {"humidity": WeatherGenerators.humidity_pct},
            ),
            (
                "wind_speed_non_negative",
                cls.wind_speed_non_negative,
                {"wind_speed": WeatherGenerators.wind_speed_mph},
            ),
            (
                "evacuation_zone_format",
                cls.evacuation_zone_format,
                {"zone": WeatherGenerators.evacuation_zone},
            ),
            (
                "fahrenheit_celsius_conversion",
                cls.fahrenheit_celsius_conversion,
                {"temp_f": WeatherGenerators.temperature_f},
            ),
            (
                "time_zone_abbreviation_valid",
                cls.time_zone_abbreviation_valid,
                {"tz": WeatherGenerators.time_zone},
            ),
        ]


def run_weather_properties(max_examples: int = 100) -> dict[str, Any]:
    """
    Convenience function to run all weather property tests.

    Args:
        max_examples: Maximum examples per property

    Returns:
        Test summary dict
    """
    runner = PropertyTestRunner(max_examples=max_examples)
    properties = WeatherPropertyTests.get_all_properties()
    runner.run_all(properties)
    return runner.get_summary()
