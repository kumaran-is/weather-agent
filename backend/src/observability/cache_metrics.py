"""Prometheus Metrics for Semantic Cache.

Level 9c: Comprehensive observability for all cache layers.

Metrics Categories:
    1. Cache Hits/Misses by tier and type
    2. Latency histograms by tier
    3. Semantic similarity score distribution
    4. Cost savings tracking
    5. Cache size and entry counts
    6. Embedding cache metrics

Signal Correlation:
    All histogram metrics support exemplars for trace correlation.
    Use observe_with_exemplar() to include trace_id in observations.

Grafana Dashboard:
    monitoring/grafana/dashboards/semantic_cache_dashboard.json

Prometheus Alerts:
    monitoring/prometheus/cache_alerts.yml

Usage:
    from backend.src.observability.cache_metrics import CacheMetricsCollector, get_cache_metrics

    metrics = get_cache_metrics()

    # Record cache hit
    metrics.record_hit(
        cache_type="query",
        tier="tier1",
        tool="default",
        similarity=None,
        latency_ms=2.5,
    )

    # Record cache miss
    metrics.record_miss(cache_type="tool", tool="get_forecast", latency_ms=45.0)

    # Record cost savings
    metrics.record_cost_savings(cache_type="llm", amount_usd=0.002)
"""

import logging
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from backend.src.observability.metrics import get_exemplar_labels, observe_with_exemplar

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Hit/Miss Counters
# ============================================================================

CACHE_HITS = Counter(
    "weather_cache_hit_total",
    "Total cache hits by tier and type",
    ["tier", "cache_type", "tool"],
)

CACHE_MISSES = Counter(
    "weather_cache_miss_total",
    "Total cache misses by type",
    ["cache_type", "tool"],
)

CACHE_BYPASSES = Counter(
    "weather_cache_bypass_total",
    "Total cache bypasses (life-safety or force refresh)",
    ["cache_type", "reason"],
)


# ============================================================================
# Latency Histograms
# ============================================================================

CACHE_LATENCY = Histogram(
    "weather_cache_latency_seconds",
    "Cache lookup latency in seconds",
    ["tier", "cache_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

EMBEDDING_LATENCY = Histogram(
    "weather_embedding_latency_seconds",
    "Embedding generation latency in seconds",
    ["cache_type"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)


# ============================================================================
# Semantic Similarity Scores
# ============================================================================

SEMANTIC_SIMILARITY = Histogram(
    "weather_semantic_similarity_score",
    "Semantic similarity scores for tier2 hits",
    ["cache_type"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)


# ============================================================================
# Cost Tracking
# ============================================================================

COST_SAVINGS = Counter(
    "weather_cache_cost_savings_usd",
    "Estimated cost savings from cache hits in USD",
    ["cache_type"],
)

API_CALLS_AVOIDED = Counter(
    "weather_api_calls_avoided_total",
    "Number of API calls avoided due to caching",
    ["api_type"],  # mcp | llm | embedding
)


# ============================================================================
# Cache Size and Entries
# ============================================================================

CACHE_SIZE_BYTES = Gauge(
    "weather_cache_size_bytes",
    "Current cache size in bytes",
    ["tier"],
)

CACHE_ENTRIES = Gauge(
    "weather_cache_entries_total",
    "Current number of cache entries",
    ["tier", "cache_type"],
)

CACHE_EVICTIONS = Counter(
    "weather_cache_evictions_total",
    "Total cache evictions",
    ["tier", "cache_type"],
)


# ============================================================================
# Embedding Cache Metrics
# ============================================================================

EMBEDDING_CACHE_HITS = Counter(
    "weather_embedding_cache_hit_total",
    "Embedding cache hits",
)

EMBEDDING_CACHE_MISSES = Counter(
    "weather_embedding_cache_miss_total",
    "Embedding cache misses",
)


# ============================================================================
# Cache Metrics Collector
# ============================================================================


class CacheMetricsCollector:
    """Collect and record cache metrics.

    Provides a unified interface for recording cache metrics across all
    cache types (query, tool, llm) and tiers (tier1, tier2).

    Attributes:
        COST_PER_CALL: Estimated cost per API call for savings calculation
    """

    # Cost estimates per API call (used for savings tracking)
    COST_PER_CALL = {
        "llm": 0.002,  # ~$0.002 per LLM call
        "mcp": 0.001,  # ~$0.001 per MCP tool call
        "embedding": 0.0001,  # ~$0.0001 per embedding
    }

    def __init__(self):
        """Initialize metrics collector."""
        logger.info("✅ CacheMetricsCollector initialized")

    def record_hit(
        self,
        cache_type: str,
        tier: str,
        tool: str = "default",
        similarity: float | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a cache hit.

        Args:
            cache_type: "query" | "tool" | "llm"
            tier: "tier1" | "tier2"
            tool: Tool name (for tool cache) or "default"
            similarity: Semantic similarity score (for tier2 hits)
            latency_ms: Lookup latency in milliseconds
        """
        # Record hit counter
        CACHE_HITS.labels(tier=tier, cache_type=cache_type, tool=tool).inc()

        # Record latency with exemplar
        latency_seconds = latency_ms / 1000
        exemplar = get_exemplar_labels()
        observe_with_exemplar(
            CACHE_LATENCY.labels(tier=tier, cache_type=cache_type),
            latency_seconds,
            exemplar,
        )

        # Record similarity for tier2 hits
        if similarity is not None and tier == "tier2":
            SEMANTIC_SIMILARITY.labels(cache_type=cache_type).observe(similarity)

        # Record cost savings
        api_type = "llm" if cache_type == "llm" else "mcp"
        cost = self.COST_PER_CALL.get(api_type, 0.001)
        COST_SAVINGS.labels(cache_type=cache_type).inc(cost)
        API_CALLS_AVOIDED.labels(api_type=api_type).inc()

        logger.debug(
            f"📊 Cache HIT recorded | type={cache_type} | tier={tier} | "
            f"tool={tool} | latency={latency_ms:.2f}ms"
        )

    def record_miss(
        self,
        cache_type: str,
        tool: str = "default",
        latency_ms: float = 0.0,
    ) -> None:
        """Record a cache miss.

        Args:
            cache_type: "query" | "tool" | "llm"
            tool: Tool name (for tool cache) or "default"
            latency_ms: Lookup latency in milliseconds
        """
        # Record miss counter
        CACHE_MISSES.labels(cache_type=cache_type, tool=tool).inc()

        # Record latency with exemplar (tier="miss")
        latency_seconds = latency_ms / 1000
        exemplar = get_exemplar_labels()
        observe_with_exemplar(
            CACHE_LATENCY.labels(tier="miss", cache_type=cache_type),
            latency_seconds,
            exemplar,
        )

        logger.debug(
            f"📊 Cache MISS recorded | type={cache_type} | "
            f"tool={tool} | latency={latency_ms:.2f}ms"
        )

    def record_bypass(
        self,
        cache_type: str,
        reason: str,
    ) -> None:
        """Record a cache bypass.

        Args:
            cache_type: "query" | "tool" | "llm"
            reason: "life_safety" | "force_refresh" | "excluded"
        """
        CACHE_BYPASSES.labels(cache_type=cache_type, reason=reason).inc()

        logger.debug(f"📊 Cache BYPASS recorded | type={cache_type} | reason={reason}")

    def record_cost_savings(
        self,
        cache_type: str,
        amount_usd: float,
    ) -> None:
        """Record cost savings from cache hit.

        Args:
            cache_type: "query" | "tool" | "llm"
            amount_usd: Estimated savings in USD
        """
        COST_SAVINGS.labels(cache_type=cache_type).inc(amount_usd)

    def record_embedding_latency(
        self,
        cache_type: str,
        latency_ms: float,
    ) -> None:
        """Record embedding generation latency.

        Args:
            cache_type: Cache type requesting embedding
            latency_ms: Embedding latency in milliseconds
        """
        latency_seconds = latency_ms / 1000
        exemplar = get_exemplar_labels()
        observe_with_exemplar(
            EMBEDDING_LATENCY.labels(cache_type=cache_type),
            latency_seconds,
            exemplar,
        )

    def record_embedding_cache_hit(self) -> None:
        """Record embedding cache hit."""
        EMBEDDING_CACHE_HITS.inc()

    def record_embedding_cache_miss(self) -> None:
        """Record embedding cache miss."""
        EMBEDDING_CACHE_MISSES.inc()

    def update_cache_size(
        self,
        tier: str,
        size_bytes: int,
    ) -> None:
        """Update current cache size.

        Args:
            tier: "tier1" | "tier2" | "embedding"
            size_bytes: Current size in bytes
        """
        CACHE_SIZE_BYTES.labels(tier=tier).set(size_bytes)

    def update_cache_entries(
        self,
        tier: str,
        cache_type: str,
        count: int,
    ) -> None:
        """Update current cache entry count.

        Args:
            tier: "tier1" | "tier2" | "embedding"
            cache_type: "query" | "tool" | "llm"
            count: Current entry count
        """
        CACHE_ENTRIES.labels(tier=tier, cache_type=cache_type).set(count)

    def record_eviction(
        self,
        tier: str,
        cache_type: str,
    ) -> None:
        """Record cache eviction.

        Args:
            tier: "tier1" | "tier2" | "embedding"
            cache_type: "query" | "tool" | "llm"
        """
        CACHE_EVICTIONS.labels(tier=tier, cache_type=cache_type).inc()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of cache metrics.

        Returns:
            Dict with current metric values
        """
        # Note: This provides a snapshot, not real values from Prometheus
        # Real values should be queried from Prometheus directly
        return {
            "status": "metrics_active",
            "metrics": [
                "weather_cache_hit_total",
                "weather_cache_miss_total",
                "weather_cache_bypass_total",
                "weather_cache_latency_seconds",
                "weather_semantic_similarity_score",
                "weather_cache_cost_savings_usd",
                "weather_api_calls_avoided_total",
                "weather_cache_size_bytes",
                "weather_cache_entries_total",
                "weather_cache_evictions_total",
                "weather_embedding_cache_hit_total",
                "weather_embedding_cache_miss_total",
                "weather_embedding_latency_seconds",
            ],
        }


# Singleton instance
_cache_metrics: CacheMetricsCollector | None = None


def get_cache_metrics() -> CacheMetricsCollector:
    """Get singleton CacheMetricsCollector instance.

    Returns:
        CacheMetricsCollector instance
    """
    global _cache_metrics
    if _cache_metrics is None:
        _cache_metrics = CacheMetricsCollector()
    return _cache_metrics
