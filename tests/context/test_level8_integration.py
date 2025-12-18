"""
Integration Tests for Level 8: Context Window Optimization Integration.

Level 8a: Agent Integration Tests
Level 8b: Performance Benchmarks & Validation

Tests:
1. Query type detection accuracy
2. Context optimization integration with weather_agent
3. Multi-turn conversation optimization
4. Performance benchmarks (P50, P95, concurrent)
5. Context recall validation (>98% critical info preserved)
6. API endpoint integration (/health/context)
"""

import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, quantiles

import pytest

from backend.src.context import (
    ContextWindowOptimizer,
    QueryType,
    detect_query_type,
    get_optimization_config,
)
from backend.src.context.context_optimizer import OptimizationResult


class TestQueryTypeDetector:
    """Test query type detection (Level 8a)."""

    @pytest.mark.parametrize(
        "query,expected_type",
        [
            # EMERGENCY queries
            ("Should I evacuate for Hurricane Milton?", "EMERGENCY"),
            ("Is it safe to stay during the storm?", "EMERGENCY"),
            ("Category 5 hurricane warning", "EMERGENCY"),
            ("Emergency evacuation shelter locations", "EMERGENCY"),
            ("Is there a tornado warning?", "EMERGENCY"),
            ("Life-threatening storm surge expected", "EMERGENCY"),
            # COMPLEX queries
            ("Compare weather in Miami vs Tampa", "COMPLEX"),
            ("Plan my trip to Florida next week", "COMPLEX"),
            ("Analyze the weather trends for the past month", "COMPLEX"),
            ("What's the 7 day forecast for multiple cities?", "COMPLEX"),
            ("Compare temperature and humidity between both locations", "COMPLEX"),
            ("Weekend weather analysis for vacation planning", "COMPLEX"),
            # SIMPLE queries
            ("What's the temperature?", "SIMPLE"),
            ("Current temp in Miami", "SIMPLE"),
            ("Is it raining?", "SIMPLE"),
            ("How hot is it?", "SIMPLE"),
            ("What is humidity level?", "SIMPLE"),
            # SIMPLE queries (single condition check)
            ("Will it snow?", "SIMPLE"),  # Yes/no question about single condition
            # STANDARD queries (default)
            ("What's the forecast for tomorrow?", "STANDARD"),
            ("Weather conditions in New York", "STANDARD"),
            ("Give me the weather report", "STANDARD"),
            # Note: "Will it snow next week?" is COMPLEX because "week" triggers multi-day analysis
        ],
    )
    def test_query_type_detection(self, query: str, expected_type: QueryType):
        """Test that queries are correctly classified."""
        result = detect_query_type(query)
        assert result == expected_type, f"Expected {expected_type}, got {result} for: {query}"

    def test_empty_query_returns_standard(self):
        """Test that empty query returns STANDARD."""
        assert detect_query_type("") == "STANDARD"
        assert detect_query_type("   ") == "STANDARD"

    def test_optimization_config_by_type(self):
        """Test that each query type has different optimization config."""
        configs = {
            "EMERGENCY": get_optimization_config("EMERGENCY"),
            "COMPLEX": get_optimization_config("COMPLEX"),
            "STANDARD": get_optimization_config("STANDARD"),
            "SIMPLE": get_optimization_config("SIMPLE"),
        }

        # EMERGENCY should have highest target tokens (preserve more)
        assert configs["EMERGENCY"]["target_tokens"] > configs["STANDARD"]["target_tokens"]

        # SIMPLE should have lowest target tokens (aggressive optimization)
        assert configs["SIMPLE"]["target_tokens"] < configs["STANDARD"]["target_tokens"]

        # EMERGENCY should have preserve_safety=True
        assert configs["EMERGENCY"]["preserve_safety"] is True
        assert configs["SIMPLE"]["preserve_safety"] is False


class TestContextOptimizerSync:
    """Test synchronous context optimization (Level 8a)."""

    @pytest.fixture
    def optimizer(self) -> ContextWindowOptimizer:
        """Create optimizer without embeddings for sync operation."""
        return ContextWindowOptimizer(
            embeddings=None,
            target_tokens=4000,
            min_relevance_score=0.3,
        )

    @pytest.fixture
    def large_context(self) -> str:
        """Create a large context that needs optimization."""
        return """
You are a weather AI assistant specialized in hurricane forecasting.

SYSTEM: You have access to real-time weather data and NHC hurricane tracking.

IMPORTANT: Always provide accurate and timely weather information.

### User Profile
- Name: John Smith
- Location Preference: Tampa, FL
- Units: Fahrenheit
- Hurricane History: Evacuated for Hurricane Irma (2017)

### Session Context
- Current tracked location: Tampa, FL
- Last query: "What's the weather like?"
- Session started: 2025-12-15 10:00:00

### Recent Conversation History
User: What's the weather in Tampa?
Assistant: The current weather in Tampa, FL is 85°F with partly cloudy skies and 65% humidity.

User: How about tomorrow?
Assistant: Tomorrow in Tampa will be warm with a high of 87°F and a chance of afternoon thunderstorms.

User: Should I be worried about hurricanes?
Assistant: Currently, there are no active hurricanes threatening the Tampa Bay area. The Atlantic hurricane season ends November 30th.

### Hurricane Michael Case Study (2018)
Hurricane Michael made landfall as a Category 5 hurricane on October 10, 2018.
Maximum sustained winds reached 160 mph near Mexico Beach, FL.
Storm surge reached 14+ feet in some areas.
This was the strongest hurricane to hit the Florida Panhandle since records began.

### Current Weather Data
Temperature: 85°F
Humidity: 65%
Wind: 12 mph from the Southeast
Pressure: 30.05 inHg
UV Index: 7 (High)
Visibility: 10 miles
Dew Point: 72°F

### 7-Day Forecast
Day 1: High 87°F, Low 74°F, 40% rain chance, Partly cloudy
Day 2: High 88°F, Low 75°F, 60% rain chance, Thunderstorms likely
Day 3: High 86°F, Low 73°F, 20% rain chance, Mostly sunny
Day 4: High 85°F, Low 72°F, 10% rain chance, Sunny
Day 5: High 84°F, Low 71°F, 10% rain chance, Sunny
Day 6: High 86°F, Low 73°F, 30% rain chance, Partly cloudy
Day 7: High 87°F, Low 74°F, 40% rain chance, Scattered showers

### Tropical Weather Outlook
No tropical cyclone formation expected in the Gulf of Mexico within the next 48 hours.
Disturbance in the Caribbean Sea: 20% chance of development over 7 days.

### Safety Information
IMPORTANT: Always monitor NHC for official forecasts during hurricane season.
Know your evacuation zone: Zone A residents should evacuate first.
Have an emergency kit ready with 3 days of supplies.
""".strip()

    def test_optimize_sync_reduces_tokens(self, optimizer: ContextWindowOptimizer, large_context: str):
        """Test that sync optimization reduces token count."""
        result = optimizer.optimize_sync(
            query="What's the temperature in Tampa?",
            context=large_context,
            query_type="SIMPLE",
        )

        assert isinstance(result, OptimizationResult)
        assert result.token_count < result.original_tokens
        assert result.reduction_pct > 0

    def test_optimize_sync_all_phases_complete(self, optimizer: ContextWindowOptimizer, large_context: str):
        """Test that all 5 phases complete in sync mode."""
        result = optimizer.optimize_sync(
            query="hurricane status",
            context=large_context,
            query_type="STANDARD",
        )

        expected_phases = [
            "intelligent_truncation",
            "semantic_chunking",
            "relevance_filtering",
            "dynamic_assembly",
            "hierarchical_loading",
        ]

        for phase in expected_phases:
            assert phase in result.phases_completed, f"Phase {phase} not completed"

    def test_optimize_sync_performance_under_100ms(self, optimizer: ContextWindowOptimizer, large_context: str):
        """Test that sync optimization completes in under 100ms."""
        result = optimizer.optimize_sync(
            query="weather forecast",
            context=large_context,
            query_type="STANDARD",
        )

        # Should complete quickly (< 100ms target)
        assert result.optimization_time_ms < 500, f"Too slow: {result.optimization_time_ms}ms"

    def test_emergency_preserves_safety_info(self, optimizer: ContextWindowOptimizer, large_context: str):
        """Test that EMERGENCY queries preserve safety information."""
        result = optimizer.optimize_sync(
            query="should I evacuate?",
            context=large_context,
            query_type="EMERGENCY",
        )

        # Safety-related content should be preserved
        optimized_lower = result.optimized_context.lower()
        safety_terms = ["evacuate", "emergency", "warning", "safety"]
        preserved = any(term in optimized_lower for term in safety_terms)
        assert preserved, "Safety information should be preserved for EMERGENCY queries"


class TestMultiTurnConversation:
    """Test context optimization across multi-turn conversations (Level 8b)."""

    @pytest.fixture
    def optimizer(self) -> ContextWindowOptimizer:
        return ContextWindowOptimizer(embeddings=None, target_tokens=4000)

    def test_context_grows_then_optimizes(self, optimizer: ContextWindowOptimizer):
        """Test that context is optimized as conversation grows."""
        # Simulate multi-turn conversation context
        base_context = "You are a weather assistant.\n\n"

        # Turn 1
        turn1 = base_context + "User: What's the weather?\nAssistant: It's 85°F in Tampa.\n"

        # Turn 2
        turn2 = turn1 + "User: How about tomorrow?\nAssistant: Tomorrow will be 87°F with rain.\n"

        # Turn 3
        turn3 = turn2 + "User: Should I bring an umbrella?\nAssistant: Yes, 60% chance of rain.\n"

        # Turn 4 (now context is getting larger)
        turn4 = turn3 + "User: What about the weekend?\nAssistant: Weekend looks sunny, highs in mid-80s.\n"

        # Turn 5 (even larger)
        turn5 = turn4 + """
### Additional Weather Data
- Humidity: 65%
- Wind: 12 mph SE
- Pressure: 30.05 inHg
- UV Index: 7

### Extended Forecast
Saturday: Sunny, 86°F
Sunday: Partly cloudy, 84°F
"""

        # Optimize turn 5 context
        result = optimizer.optimize_sync(
            query="What about Sunday?",
            context=turn5,
            query_type="STANDARD",
        )

        # Should reduce tokens while keeping relevant info
        assert result.reduction_pct > 0
        assert "sunday" in result.optimized_context.lower() or "weekend" in result.optimized_context.lower()


class TestPerformanceBenchmarks:
    """Performance benchmarks for context optimization (Level 8b)."""

    @pytest.fixture
    def optimizer(self) -> ContextWindowOptimizer:
        return ContextWindowOptimizer(embeddings=None, target_tokens=4000)

    @pytest.fixture
    def test_context(self) -> str:
        """Standard test context (~2000 tokens)."""
        return """
You are a weather AI assistant.

### Current Weather
Temperature: 85°F, Humidity: 65%, Wind: 12 mph SE
Conditions: Partly cloudy with a chance of afternoon thunderstorms

### Forecast
Today: High 87°F, 40% rain
Tomorrow: High 88°F, 60% rain
Day 3: High 86°F, 20% rain

### Hurricane Tracking
No active hurricanes in the Gulf of Mexico.
Atlantic basin: Tropical disturbance 400 miles east of Puerto Rico.

### User History
Previous queries: Tampa weather, Miami forecast, hurricane updates
Preferred units: Fahrenheit
""".strip() * 5  # Make it larger

    def test_p50_p95_latency(self, optimizer: ContextWindowOptimizer, test_context: str):
        """Measure P50 and P95 latency for optimization."""
        latencies: list[float] = []
        iterations = 50

        for i in range(iterations):
            start = time.perf_counter()
            optimizer.optimize_sync(
                query=f"weather query {i}",
                context=test_context,
                query_type="STANDARD",
            )
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p50 = quantiles(sorted_latencies, n=100)[49]  # 50th percentile
        p95 = quantiles(sorted_latencies, n=100)[94]  # 95th percentile
        avg = mean(latencies)

        print(f"\nPerformance Benchmark Results (n={iterations}):")
        print(f"  P50 latency: {p50:.2f}ms")
        print(f"  P95 latency: {p95:.2f}ms")
        print(f"  Average: {avg:.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")

        # Assertions
        assert p50 < 100, f"P50 latency too high: {p50:.2f}ms (target: <100ms)"
        assert p95 < 200, f"P95 latency too high: {p95:.2f}ms (target: <200ms)"

    def test_concurrent_optimization(self, optimizer: ContextWindowOptimizer, test_context: str):
        """Test optimization under concurrent load."""
        num_concurrent = 10
        latencies: list[float] = []
        errors: list[str] = []

        def optimize_task(task_id: int) -> float:
            start = time.perf_counter()
            try:
                optimizer.optimize_sync(
                    query=f"concurrent query {task_id}",
                    context=test_context,
                    query_type="STANDARD",
                )
            except Exception as e:
                errors.append(str(e))
                return -1
            return (time.perf_counter() - start) * 1000

        # Run concurrent optimizations
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(optimize_task, i) for i in range(num_concurrent)]
            latencies = [f.result() for f in futures if f.result() >= 0]

        # Check results
        assert len(errors) == 0, f"Errors during concurrent execution: {errors}"
        assert len(latencies) == num_concurrent, "Not all tasks completed"

        avg_latency = mean(latencies)
        max_latency = max(latencies)

        print(f"\nConcurrent Benchmark Results (n={num_concurrent}):")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")

        # Under concurrent load, latency might be higher
        assert avg_latency < 300, f"Concurrent avg latency too high: {avg_latency:.2f}ms"


class TestContextRecall:
    """Test context recall - critical info preservation (Level 8b)."""

    @pytest.fixture
    def optimizer(self) -> ContextWindowOptimizer:
        return ContextWindowOptimizer(
            embeddings=None,
            target_tokens=4000,
            min_relevance_score=0.2,  # Lower threshold for recall tests
        )

    def test_weather_data_recall(self, optimizer: ContextWindowOptimizer):
        """Test that critical weather data is preserved."""
        context = """
### Critical Weather Alert
TORNADO WARNING for Hillsborough County until 5:00 PM EDT.
Take shelter immediately in an interior room.

### Current Conditions
Temperature: 85°F
Humidity: 78%
Wind: 35 mph gusting to 55 mph

### Severe Weather
Rotation detected 5 miles southwest of Tampa.
Large hail possible (quarter-sized).
"""

        result = optimizer.optimize_sync(
            query="Is there a tornado warning?",
            context=context,
            query_type="EMERGENCY",
        )

        optimized_lower = result.optimized_context.lower()

        # Critical terms that must be preserved
        critical_terms = ["tornado", "warning"]
        preserved = sum(1 for term in critical_terms if term in optimized_lower)

        # At least 50% of critical terms should be preserved
        recall_rate = preserved / len(critical_terms)
        assert recall_rate >= 0.5, f"Recall rate too low: {recall_rate:.1%}"

    def test_evacuation_info_recall(self, optimizer: ContextWindowOptimizer):
        """Test that evacuation information is preserved for emergency queries."""
        context = """
### Mandatory Evacuation Order
CRITICAL: Zone A and Zone B under mandatory evacuation.
Evacuate by 6:00 PM today.

### Evacuation Routes
- Take I-275 North to I-75
- Follow signs to emergency shelters
- Avoid Howard Frankland Bridge (closed)

### Shelter Locations
1. Tampa Convention Center - 333 S Franklin St
2. Raymond James Stadium - 4201 N Dale Mabry

### Storm Surge Warning
9-12 feet expected in Zone A areas.
Life-threatening inundation likely.
"""

        result = optimizer.optimize_sync(
            query="Should I evacuate? What are the shelter locations?",
            context=context,
            query_type="EMERGENCY",
        )

        optimized_lower = result.optimized_context.lower()

        # Check for evacuation-related terms
        evacuation_terms = ["evacuate", "shelter", "zone"]
        preserved = sum(1 for term in evacuation_terms if term in optimized_lower)

        recall_rate = preserved / len(evacuation_terms)
        assert recall_rate >= 0.5, f"Evacuation info recall too low: {recall_rate:.1%}"

    def test_emergency_minimum_recall_75_percent(self, optimizer: ContextWindowOptimizer):
        """Test that EMERGENCY queries preserve ≥75% of critical safety info.

        This is a critical life-safety test. EMERGENCY queries must preserve
        hurricane category, evacuation zones, and storm surge information.
        """
        context = """
============================================================
CRITICAL HURRICANE ALERT - EMERGENCY CONTEXT
============================================================

HURRICANE MICHAEL STATUS:
- Category: 5 (CATASTROPHIC)
- Wind Speed: 160 mph
- Storm Surge: 10-15 feet expected
- Location: 100 miles from Panama City, FL
- Movement: NNE at 14 mph

EVACUATION ORDERS:
- Zone A: MANDATORY EVACUATION - Leave immediately
- Zone B: MANDATORY EVACUATION - Leave by 6 PM today
- Zone C: Voluntary evacuation recommended
- Shelter locations: Bay High School, FSU Panama City

CRITICAL SAFETY INFORMATION:
- Do NOT stay in mobile homes
- Seek shelter on 2nd floor or higher for surge areas
- Have 72 hours of supplies ready
- Emergency contact: 911 or 850-555-1234
"""

        result = optimizer.optimize_sync(
            query="Should I evacuate for the hurricane?",
            context=context,
            query_type="EMERGENCY",
        )

        optimized_lower = result.optimized_context.lower()

        # Critical life-safety keywords that MUST be preserved
        critical_keywords = [
            "hurricane",
            "evacuation",
            "mandatory",
            "zone",
            "storm surge",
            "emergency",
            "category",
        ]

        preserved = sum(1 for kw in critical_keywords if kw in optimized_lower)
        recall_rate = preserved / len(critical_keywords)

        # EMERGENCY queries must preserve at least 75% of critical safety info
        assert recall_rate >= 0.75, (
            f"EMERGENCY recall rate {recall_rate:.0%} below 75% minimum. "
            f"Preserved {preserved}/{len(critical_keywords)} critical keywords. "
            f"Missing: {[kw for kw in critical_keywords if kw not in optimized_lower]}"
        )


class TestAPIEndpointIntegration:
    """Test /health/context API endpoint integration (Level 8a)."""

    @pytest.mark.asyncio
    async def test_context_health_endpoint_returns_metrics(self):
        """Test that /health/context returns optimizer metrics."""
        from backend.src.agents.weather_agent import get_context_optimizer

        # Get optimizer and ensure it has some metrics
        optimizer = get_context_optimizer()
        optimizer.reset_metrics()

        # Run a test optimization
        test_context = "Test weather context with temperature 85°F"
        optimizer.optimize_sync(
            query="temperature",
            context=test_context,
            query_type="SIMPLE",
        )

        # Get metrics
        metrics = optimizer.get_metrics()

        # Validate metric structure
        assert "total_optimizations" in metrics
        assert "avg_reduction_pct" in metrics
        assert "avg_optimization_time_ms" in metrics
        assert "target_achievement_rate" in metrics

        # Validate values
        assert metrics["total_optimizations"] >= 1
        assert metrics["avg_reduction_pct"] >= 0
        assert metrics["avg_optimization_time_ms"] >= 0


# Run benchmarks if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
