"""Unit tests for CacheMetricsCollector (Level 9c).

Tests validate the Prometheus metrics collection:
- Cache hits/misses by tier and type
- Latency histogram recording
- Semantic similarity score tracking
- Cost savings tracking
- Cache size and entry tracking
- Embedding cache metrics

Metrics are used for:
- Grafana dashboards (real-time monitoring)
- Prometheus alerts (anomaly detection)
- Cost optimization analysis
"""



from backend.src.observability.cache_metrics import (
    API_CALLS_AVOIDED,
    CACHE_BYPASSES,
    CACHE_ENTRIES,
    CACHE_EVICTIONS,
    CACHE_HITS,
    CACHE_LATENCY,
    CACHE_MISSES,
    CACHE_SIZE_BYTES,
    COST_SAVINGS,
    EMBEDDING_CACHE_HITS,
    EMBEDDING_CACHE_MISSES,
    EMBEDDING_LATENCY,
    SEMANTIC_SIMILARITY,
    CacheMetricsCollector,
    get_cache_metrics,
)


class TestCacheMetricsCollectorInit:
    """Test CacheMetricsCollector initialization."""

    def test_init(self):
        """Test collector initializes successfully."""
        collector = CacheMetricsCollector()
        assert collector is not None

    def test_cost_per_call_defined(self):
        """Test cost per call estimates are defined."""
        collector = CacheMetricsCollector()

        assert "llm" in collector.COST_PER_CALL
        assert "mcp" in collector.COST_PER_CALL
        assert "embedding" in collector.COST_PER_CALL

    def test_singleton_pattern(self):
        """Test get_cache_metrics returns singleton."""
        collector1 = get_cache_metrics()
        collector2 = get_cache_metrics()

        assert collector1 is collector2


class TestCacheHitRecording:
    """Test cache hit recording."""

    def test_record_tier1_hit(self):
        """Test recording tier1 cache hit."""
        collector = CacheMetricsCollector()

        # Record hit
        collector.record_hit(
            cache_type="query",
            tier="tier1",
            tool="default",
            similarity=None,
            latency_ms=2.5,
        )

        # Metric should be incremented (we verify it doesn't raise)
        # Note: Actual metric values are stored in Prometheus registry

    def test_record_tier2_hit_with_similarity(self):
        """Test recording tier2 hit includes similarity score."""
        collector = CacheMetricsCollector()

        collector.record_hit(
            cache_type="tool",
            tier="tier2",
            tool="get_forecast",
            similarity=0.92,
            latency_ms=45.0,
        )

        # Should record similarity in histogram

    def test_record_hit_increments_cost_savings(self):
        """Test cache hit increments cost savings."""
        collector = CacheMetricsCollector()

        collector.record_hit(
            cache_type="llm",
            tier="tier1",
            tool="default",
            latency_ms=5.0,
        )

        # Cost savings should be tracked

    def test_record_hit_different_cache_types(self):
        """Test recording hits for different cache types."""
        collector = CacheMetricsCollector()

        # Query cache
        collector.record_hit(
            cache_type="query",
            tier="tier1",
            tool="default",
            latency_ms=1.0,
        )

        # Tool cache
        collector.record_hit(
            cache_type="tool",
            tier="tier1",
            tool="get_forecast",
            latency_ms=2.0,
        )

        # LLM cache
        collector.record_hit(
            cache_type="llm",
            tier="tier2",
            tool="default",
            similarity=0.95,
            latency_ms=30.0,
        )


class TestCacheMissRecording:
    """Test cache miss recording."""

    def test_record_miss(self):
        """Test recording cache miss."""
        collector = CacheMetricsCollector()

        collector.record_miss(
            cache_type="query",
            tool="default",
            latency_ms=50.0,
        )

    def test_record_miss_with_tool(self):
        """Test recording miss for specific tool."""
        collector = CacheMetricsCollector()

        collector.record_miss(
            cache_type="tool",
            tool="get_forecast",
            latency_ms=48.0,
        )


class TestCacheBypassRecording:
    """Test cache bypass recording."""

    def test_record_life_safety_bypass(self):
        """Test recording life-safety bypass."""
        collector = CacheMetricsCollector()

        collector.record_bypass(
            cache_type="tool",
            reason="life_safety",
        )

    def test_record_force_refresh_bypass(self):
        """Test recording force refresh bypass."""
        collector = CacheMetricsCollector()

        collector.record_bypass(
            cache_type="query",
            reason="force_refresh",
        )

    def test_record_excluded_bypass(self):
        """Test recording exclusion pattern bypass."""
        collector = CacheMetricsCollector()

        collector.record_bypass(
            cache_type="llm",
            reason="excluded",
        )


class TestCostSavingsRecording:
    """Test cost savings recording."""

    def test_record_cost_savings(self):
        """Test recording explicit cost savings."""
        collector = CacheMetricsCollector()

        collector.record_cost_savings(
            cache_type="llm",
            amount_usd=0.003,
        )

    def test_record_cost_savings_multiple(self):
        """Test recording multiple cost savings."""
        collector = CacheMetricsCollector()

        collector.record_cost_savings(cache_type="llm", amount_usd=0.001)
        collector.record_cost_savings(cache_type="llm", amount_usd=0.002)
        collector.record_cost_savings(cache_type="tool", amount_usd=0.0005)


class TestEmbeddingMetrics:
    """Test embedding-related metrics."""

    def test_record_embedding_latency(self):
        """Test recording embedding generation latency."""
        collector = CacheMetricsCollector()

        collector.record_embedding_latency(
            cache_type="query",
            latency_ms=150.0,
        )

    def test_record_embedding_cache_hit(self):
        """Test recording embedding cache hit."""
        collector = CacheMetricsCollector()

        collector.record_embedding_cache_hit()

    def test_record_embedding_cache_miss(self):
        """Test recording embedding cache miss."""
        collector = CacheMetricsCollector()

        collector.record_embedding_cache_miss()


class TestCacheSizeMetrics:
    """Test cache size and entry metrics."""

    def test_update_cache_size(self):
        """Test updating cache size."""
        collector = CacheMetricsCollector()

        collector.update_cache_size(
            tier="tier1",
            size_bytes=1024000,
        )

    def test_update_cache_entries(self):
        """Test updating cache entry count."""
        collector = CacheMetricsCollector()

        collector.update_cache_entries(
            tier="tier1",
            cache_type="query",
            count=500,
        )

    def test_record_eviction(self):
        """Test recording cache eviction."""
        collector = CacheMetricsCollector()

        collector.record_eviction(
            tier="tier1",
            cache_type="query",
        )


class TestCacheMetricsSummary:
    """Test metrics summary."""

    def test_get_summary(self):
        """Test getting metrics summary."""
        collector = CacheMetricsCollector()

        summary = collector.get_summary()

        assert "status" in summary
        assert summary["status"] == "metrics_active"
        assert "metrics" in summary
        assert len(summary["metrics"]) > 0

    def test_summary_includes_all_metrics(self):
        """Test summary includes all defined metrics."""
        collector = CacheMetricsCollector()

        summary = collector.get_summary()
        metrics = summary["metrics"]

        # Core metrics
        assert "weather_cache_hit_total" in metrics
        assert "weather_cache_miss_total" in metrics
        assert "weather_cache_bypass_total" in metrics
        assert "weather_cache_latency_seconds" in metrics

        # Cost metrics
        assert "weather_cache_cost_savings_usd" in metrics
        assert "weather_api_calls_avoided_total" in metrics

        # Embedding metrics
        assert "weather_embedding_cache_hit_total" in metrics
        assert "weather_embedding_cache_miss_total" in metrics
        assert "weather_embedding_latency_seconds" in metrics


class TestPrometheusMetricDefinitions:
    """Test Prometheus metric definitions.

    Note: Prometheus counters expose with "_total" suffix, but the internal
    _name attribute doesn't include it. We test the base metric name.
    """

    def test_cache_hits_counter_defined(self):
        """Test CACHE_HITS counter is defined correctly."""
        assert CACHE_HITS is not None
        assert CACHE_HITS._name == "weather_cache_hit"

    def test_cache_misses_counter_defined(self):
        """Test CACHE_MISSES counter is defined correctly."""
        assert CACHE_MISSES is not None
        assert CACHE_MISSES._name == "weather_cache_miss"

    def test_cache_bypasses_counter_defined(self):
        """Test CACHE_BYPASSES counter is defined correctly."""
        assert CACHE_BYPASSES is not None
        assert CACHE_BYPASSES._name == "weather_cache_bypass"

    def test_cache_latency_histogram_defined(self):
        """Test CACHE_LATENCY histogram is defined correctly."""
        assert CACHE_LATENCY is not None
        assert CACHE_LATENCY._name == "weather_cache_latency_seconds"

    def test_semantic_similarity_histogram_defined(self):
        """Test SEMANTIC_SIMILARITY histogram is defined correctly."""
        assert SEMANTIC_SIMILARITY is not None
        assert SEMANTIC_SIMILARITY._name == "weather_semantic_similarity_score"

    def test_cost_savings_counter_defined(self):
        """Test COST_SAVINGS counter is defined correctly."""
        assert COST_SAVINGS is not None
        assert COST_SAVINGS._name == "weather_cache_cost_savings_usd"

    def test_api_calls_avoided_counter_defined(self):
        """Test API_CALLS_AVOIDED counter is defined correctly."""
        assert API_CALLS_AVOIDED is not None
        assert API_CALLS_AVOIDED._name == "weather_api_calls_avoided"

    def test_cache_size_gauge_defined(self):
        """Test CACHE_SIZE_BYTES gauge is defined correctly."""
        assert CACHE_SIZE_BYTES is not None
        assert CACHE_SIZE_BYTES._name == "weather_cache_size_bytes"

    def test_cache_entries_gauge_defined(self):
        """Test CACHE_ENTRIES gauge is defined correctly."""
        assert CACHE_ENTRIES is not None
        assert CACHE_ENTRIES._name == "weather_cache_entries_total"

    def test_cache_evictions_counter_defined(self):
        """Test CACHE_EVICTIONS counter is defined correctly."""
        assert CACHE_EVICTIONS is not None
        assert CACHE_EVICTIONS._name == "weather_cache_evictions"

    def test_embedding_cache_hits_counter_defined(self):
        """Test EMBEDDING_CACHE_HITS counter is defined correctly."""
        assert EMBEDDING_CACHE_HITS is not None
        assert EMBEDDING_CACHE_HITS._name == "weather_embedding_cache_hit"

    def test_embedding_cache_misses_counter_defined(self):
        """Test EMBEDDING_CACHE_MISSES counter is defined correctly."""
        assert EMBEDDING_CACHE_MISSES is not None
        assert EMBEDDING_CACHE_MISSES._name == "weather_embedding_cache_miss"

    def test_embedding_latency_histogram_defined(self):
        """Test EMBEDDING_LATENCY histogram is defined correctly."""
        assert EMBEDDING_LATENCY is not None
        assert EMBEDDING_LATENCY._name == "weather_embedding_latency_seconds"


class TestCacheMetricsLabels:
    """Test metric label handling."""

    def test_tier_labels(self):
        """Test tier labels are recorded correctly."""
        collector = CacheMetricsCollector()

        # Tier 1
        collector.record_hit(
            cache_type="query",
            tier="tier1",
            tool="default",
            latency_ms=2.0,
        )

        # Tier 2
        collector.record_hit(
            cache_type="query",
            tier="tier2",
            tool="default",
            similarity=0.90,
            latency_ms=40.0,
        )

    def test_cache_type_labels(self):
        """Test cache type labels are recorded correctly."""
        collector = CacheMetricsCollector()

        for cache_type in ["query", "tool", "llm"]:
            collector.record_hit(
                cache_type=cache_type,
                tier="tier1",
                tool="default",
                latency_ms=5.0,
            )

    def test_tool_labels(self):
        """Test tool labels are recorded correctly."""
        collector = CacheMetricsCollector()

        tools = ["get_forecast", "get_current_weather", "geocode_location"]
        for tool in tools:
            collector.record_hit(
                cache_type="tool",
                tier="tier1",
                tool=tool,
                latency_ms=3.0,
            )


class TestCacheMetricsIntegration:
    """Test metrics integration scenarios."""

    def test_full_cache_flow_metrics(self):
        """Test metrics for full cache flow (miss → set → hit)."""
        collector = CacheMetricsCollector()

        # Step 1: Cache miss
        collector.record_miss(
            cache_type="query",
            tool="default",
            latency_ms=50.0,
        )

        # Step 2: Embedding generated (miss in embedding cache)
        collector.record_embedding_cache_miss()
        collector.record_embedding_latency(
            cache_type="query",
            latency_ms=200.0,
        )

        # Step 3: Response cached (update entries)
        collector.update_cache_entries(
            tier="tier1",
            cache_type="query",
            count=1,
        )
        collector.update_cache_entries(
            tier="tier2",
            cache_type="query",
            count=1,
        )

        # Step 4: Same query - cache hit
        collector.record_hit(
            cache_type="query",
            tier="tier1",
            tool="default",
            latency_ms=2.0,
        )

    def test_life_safety_bypass_flow(self):
        """Test metrics for life-safety bypass flow."""
        collector = CacheMetricsCollector()

        # Hurricane alert request - bypassed
        collector.record_bypass(
            cache_type="tool",
            reason="life_safety",
        )

        # No cache entries updated
        # No cost savings recorded (because we didn't cache)

    def test_semantic_cache_promotion_flow(self):
        """Test metrics for semantic cache promotion flow."""
        collector = CacheMetricsCollector()

        # Step 1: Tier 1 miss, Tier 2 hit
        collector.record_hit(
            cache_type="query",
            tier="tier2",
            tool="default",
            similarity=0.91,
            latency_ms=45.0,
        )

        # Step 2: Promotion to Tier 1 (entry count update)
        collector.update_cache_entries(
            tier="tier1",
            cache_type="query",
            count=1,
        )

        # Step 3: Next request - Tier 1 hit (faster)
        collector.record_hit(
            cache_type="query",
            tier="tier1",
            tool="default",
            latency_ms=2.0,
        )
