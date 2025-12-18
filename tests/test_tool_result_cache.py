"""Unit tests for ToolResultCache (Level 9b).

Tests validate the tool result caching functionality:
- Life-safety bypass (get_hurricane_alerts NEVER cached)
- Per-tool TTL and threshold configuration
- Tool argument normalization
- Force refresh parameter
- Cache statistics per tool

CRITICAL SAFETY TESTS:
- Hurricane alerts must NEVER be cached (life-safety critical)
- Emergency alerts must NEVER be cached
- Evacuation orders must NEVER be cached
"""

from unittest.mock import AsyncMock

import pytest

from backend.src.cache.tool_cache.tool_cache_config import ToolCacheConfig, ToolCacheSettings
from backend.src.cache.tool_cache.tool_result_cache import ToolResultCache


class TestToolResultCacheLifeSafety:
    """Test life-safety bypass functionality - CRITICAL."""

    @pytest.mark.asyncio
    async def test_hurricane_alerts_never_cached(self):
        """CRITICAL: get_hurricane_alerts must NEVER be cached."""
        cache = ToolResultCache()

        result = await cache.get_tool_result(
            tool_name="get_hurricane_alerts",
            tool_args={"location": "Miami"},
        )

        assert result.hit is False
        assert cache.bypasses == 1

    @pytest.mark.asyncio
    async def test_emergency_alerts_never_cached(self):
        """CRITICAL: get_emergency_alerts must NEVER be cached."""
        cache = ToolResultCache()

        result = await cache.get_tool_result(
            tool_name="get_emergency_alerts",
            tool_args={"location": "Tampa"},
        )

        assert result.hit is False
        assert cache.bypasses == 1

    @pytest.mark.asyncio
    async def test_evacuation_orders_never_cached(self):
        """CRITICAL: get_evacuation_orders must NEVER be cached."""
        cache = ToolResultCache()

        result = await cache.get_tool_result(
            tool_name="get_evacuation_orders",
            tool_args={"county": "Hillsborough"},
        )

        assert result.hit is False
        assert cache.bypasses == 1

    @pytest.mark.asyncio
    async def test_shelter_locations_never_cached(self):
        """CRITICAL: get_shelter_locations must NEVER be cached."""
        cache = ToolResultCache()

        result = await cache.get_tool_result(
            tool_name="get_shelter_locations",
            tool_args={"location": "Clearwater"},
        )

        assert result.hit is False
        assert cache.bypasses == 1

    @pytest.mark.asyncio
    async def test_bypass_tools_not_stored(self):
        """CRITICAL: Bypass tools must not be stored even when set is called."""
        cache = ToolResultCache()

        # Attempt to cache hurricane alerts
        await cache.set_tool_result(
            tool_name="get_hurricane_alerts",
            tool_args={"location": "Miami"},
            result={"alerts": []},
        )

        # Verify it wasn't cached
        result = await cache.get_tool_result(
            tool_name="get_hurricane_alerts",
            tool_args={"location": "Miami"},
        )

        assert result.hit is False


class TestToolResultCacheBasicOperations:
    """Test basic cache operations."""

    @pytest.mark.asyncio
    async def test_cache_hit_tier1(self):
        """Test Tier 1 cache hit for cacheable tool."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = '{"forecast": "sunny"}'

        cache = ToolResultCache(redis_client=redis_mock)

        result = await cache.get_tool_result(
            tool_name="get_forecast",
            tool_args={"location": "Miami"},
        )

        assert result.hit is True
        assert result.tier == "tier1"
        assert result.value == {"forecast": "sunny"}

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss for cacheable tool."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None

        semantic_mock = AsyncMock()
        semantic_mock.search.return_value = []

        cache = ToolResultCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
        )

        result = await cache.get_tool_result(
            tool_name="get_forecast",
            tool_args={"location": "Miami"},
        )

        assert result.hit is False
        assert "get_forecast" in cache.tool_misses

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test set followed by get returns cached value."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = '{"forecast": "rainy"}'

        cache = ToolResultCache(redis_client=redis_mock)

        # Set
        await cache.set_tool_result(
            tool_name="get_forecast",
            tool_args={"location": "Seattle"},
            result={"forecast": "rainy"},
        )

        # Get
        result = await cache.get_tool_result(
            tool_name="get_forecast",
            tool_args={"location": "Seattle"},
        )

        assert result.hit is True
        assert result.value["forecast"] == "rainy"


class TestToolResultCacheForceRefresh:
    """Test force refresh functionality."""

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self):
        """Test force_refresh=True bypasses cache."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = '{"forecast": "cached_data"}'

        cache = ToolResultCache(redis_client=redis_mock)

        result = await cache.get_tool_result(
            tool_name="get_forecast",
            tool_args={"location": "Miami"},
            force_refresh=True,
        )

        assert result.hit is False
        # Redis should not have been called
        redis_mock.get.assert_not_called()


class TestToolResultCachePerToolConfig:
    """Test per-tool configuration."""

    def test_different_tools_different_ttl(self):
        """Test different tools have different TTL."""
        config = ToolCacheConfig()

        forecast_ttl = config.get_ttl("get_forecast")
        geocode_ttl = config.get_ttl("geocode_location")
        current_ttl = config.get_ttl("get_current_weather")

        # Geocoding should have longest TTL (cities don't move)
        assert geocode_ttl > forecast_ttl
        # Current weather should have shortest TTL
        assert current_ttl < forecast_ttl

    def test_different_tools_different_thresholds(self):
        """Test different tools have different similarity thresholds."""
        config = ToolCacheConfig()

        geocode_threshold = config.get_threshold("geocode_location")
        forecast_threshold = config.get_threshold("get_forecast")

        # Geocoding should have higher threshold (exact city match)
        assert geocode_threshold > forecast_threshold

    def test_hurricane_alerts_disabled(self):
        """Test hurricane alerts are disabled in config."""
        config = ToolCacheConfig()

        assert config.should_bypass("get_hurricane_alerts") is True
        settings = config.get_config("get_hurricane_alerts")
        assert settings.enabled is False

    @pytest.mark.asyncio
    async def test_tool_config_applied_during_lookup(self):
        """Test tool-specific config is applied during cache lookup."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None

        semantic_mock = AsyncMock()
        semantic_mock.search.return_value = []

        cache = ToolResultCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
        )

        # Get forecast - should use forecast config
        await cache.get_tool_result(
            tool_name="get_forecast",
            tool_args={"location": "Miami"},
        )

        # Verify threshold was set from tool config
        forecast_config = cache.tool_config.get_config("get_forecast")
        assert cache.threshold == forecast_config.threshold


class TestToolResultCacheLocationNormalization:
    """Test location argument normalization."""

    @pytest.mark.asyncio
    async def test_location_normalized_in_cache_key(self):
        """Test location arguments are normalized in cache key."""
        cache = ToolResultCache()

        # Build key input for both variations
        key1 = cache._build_key_input(
            tool_name="get_forecast",
            tool_args={"location": "SF"},
        )
        key2 = cache._build_key_input(
            tool_name="get_forecast",
            tool_args={"location": "San Francisco"},
        )

        # After normalization, both should produce same key input
        # (SF normalizes to san francisco)
        assert "san francisco" in key1.lower()
        assert "san francisco" in key2.lower()

    @pytest.mark.asyncio
    async def test_city_argument_normalized(self):
        """Test 'city' argument is also normalized."""
        cache = ToolResultCache()

        key = cache._build_key_input(
            tool_name="get_forecast",
            tool_args={"city": "NYC"},
        )

        assert "new york city" in key.lower()


class TestToolResultCacheEmbeddingInput:
    """Test semantic embedding input generation."""

    def test_embedding_input_natural_language(self):
        """Test embedding input is natural language."""
        cache = ToolResultCache()

        embedding = cache._generate_embedding_input(
            "get_forecast:{\"location\":\"miami\"}",
            tool_name="get_forecast",
            tool_args={"location": "Miami"},
        )

        # Should be "get_forecast for miami" not JSON
        assert "get_forecast for" in embedding.lower()
        assert "miami" in embedding.lower()

    def test_embedding_input_with_storm_name(self):
        """Test embedding input handles storm_name argument."""
        cache = ToolResultCache()

        embedding = cache._generate_embedding_input(
            "get_storm_forecast:{\"storm_name\":\"Milton\"}",
            tool_name="get_storm_forecast",
            tool_args={"storm_name": "Milton"},
        )

        assert "hurricane milton" in embedding.lower()


class TestToolResultCacheStats:
    """Test cache statistics."""

    @pytest.mark.asyncio
    async def test_per_tool_stats(self):
        """Test per-tool statistics tracking."""
        redis_mock = AsyncMock()
        redis_mock.get.side_effect = [
            '{"data": 1}',  # Hit
            None,  # Miss
            '{"data": 2}',  # Hit
        ]

        semantic_mock = AsyncMock()
        semantic_mock.search.return_value = []

        cache = ToolResultCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
        )

        await cache.get_tool_result("get_forecast", {"location": "Miami"})
        await cache.get_tool_result("get_forecast", {"location": "Boston"})
        await cache.get_tool_result("get_current_weather", {"location": "NYC"})

        stats = cache.get_stats()

        assert "per_tool" in stats
        assert stats["per_tool"]["get_forecast"]["hits"] == 1
        assert stats["per_tool"]["get_forecast"]["misses"] == 1
        assert stats["per_tool"]["get_current_weather"]["hits"] == 1

    @pytest.mark.asyncio
    async def test_bypass_counter(self):
        """Test bypass counter tracks life-safety bypasses."""
        cache = ToolResultCache()

        await cache.get_tool_result("get_hurricane_alerts", {"location": "Miami"})
        await cache.get_tool_result("get_emergency_alerts", {"location": "Tampa"})
        await cache.get_tool_result("get_evacuation_orders", {"location": "Key West"})

        stats = cache.get_stats()
        assert stats["bypasses"] == 3

    def test_reset_stats(self):
        """Test stats reset clears all counters."""
        cache = ToolResultCache()
        cache.tool_hits["get_forecast"] = 10
        cache.tool_misses["get_forecast"] = 5
        cache.bypasses = 3

        cache.reset_stats()

        assert len(cache.tool_hits) == 0
        assert len(cache.tool_misses) == 0
        assert cache.bypasses == 0


class TestToolResultCacheSerialization:
    """Test serialization/deserialization."""

    def test_serialize_dict(self):
        """Test dict serialization to JSON."""
        cache = ToolResultCache()

        result = cache._serialize({"forecast": "sunny", "temp": 85})
        assert '"forecast": "sunny"' in result or '"forecast":"sunny"' in result

    def test_deserialize_json(self):
        """Test JSON deserialization to dict."""
        cache = ToolResultCache()

        result = cache._deserialize('{"forecast": "sunny", "temp": 85}')
        assert result["forecast"] == "sunny"
        assert result["temp"] == 85


class TestToolCacheConfig:
    """Test ToolCacheConfig class."""

    def test_default_configs_exist(self):
        """Test default configurations are present."""
        config = ToolCacheConfig()

        assert "get_forecast" in config.configs
        assert "get_current_weather" in config.configs
        assert "geocode_location" in config.configs
        assert "get_hurricane_alerts" in config.configs

    def test_custom_configs_merge(self):
        """Test custom configs are merged with defaults."""
        custom = {"custom_tool": ToolCacheSettings(ttl=120, threshold=0.80)}
        config = ToolCacheConfig(custom_configs=custom)

        assert config.get_config("custom_tool").ttl == 120
        # Default still exists
        assert config.get_config("get_forecast").ttl > 0

    def test_custom_bypass_tools_merge(self):
        """Test custom bypass tools are merged."""
        config = ToolCacheConfig(custom_bypass_tools={"custom_critical_tool"})

        assert config.should_bypass("get_hurricane_alerts")  # Default
        assert config.should_bypass("custom_critical_tool")  # Custom

    def test_list_cacheable_tools(self):
        """Test listing cacheable tools."""
        config = ToolCacheConfig()

        cacheable = config.list_cacheable_tools()

        assert "get_forecast" in cacheable
        assert "get_hurricane_alerts" not in cacheable  # Disabled

    def test_list_bypass_tools(self):
        """Test listing bypass tools."""
        config = ToolCacheConfig()

        bypass = config.list_bypass_tools()

        assert "get_hurricane_alerts" in bypass
        assert "get_emergency_alerts" in bypass
        assert "get_evacuation_orders" in bypass
        assert "get_shelter_locations" in bypass

    def test_should_use_semantic(self):
        """Test should_use_semantic check."""
        config = ToolCacheConfig()

        # Geocoding bypasses semantic (exact match only)
        assert config.should_use_semantic("geocode_location") is False

        # Forecast uses semantic
        assert config.should_use_semantic("get_forecast") is True

        # Hurricane alerts disabled entirely
        assert config.should_use_semantic("get_hurricane_alerts") is False
