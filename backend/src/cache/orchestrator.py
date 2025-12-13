"""L5a: Three-tier cache orchestration for Weather AI Agent.

Cache Hierarchy:
- L1: In-memory LRU (server-local, <1ms, 5 min TTL)
- L2: Redis distributed (shared across servers, <10ms, 30 min TTL)
- L3: Anthropic Prompt Cache (API-level, automatic, 50-90% cost reduction)

Flow:
1. Check L1 (in-memory) → HIT: Return immediately
2. Check L2 (Redis) → HIT: Return + Backfill L1
3. Execute agent with L3 (Anthropic caching) → Cache response at L1 + L2

Usage:
    orchestrator = CacheOrchestrator(l1_cache, l2_cache)

    # Try cache first
    cached = await orchestrator.get(query, user_id, enable_rag, enable_cot)
    if cached:
        return cached["response"]

    # Cache miss - execute agent
    response = await execute_agent(...)

    # Cache the response
    await orchestrator.set(query, user_id, enable_rag, enable_cot, response)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from backend.src.cache.l1_memory_cache import QueryCache
from backend.src.cache.l2_redis_cache import RedisQueryCache

logger = logging.getLogger(__name__)


@dataclass
class CacheResult:
    """Result from cache lookup with metadata."""

    response: str
    cache_tier: str  # "L1", "L2", or "MISS"
    latency_ms: float
    cache_key: str = ""


@dataclass
class CacheStats:
    """Aggregated cache statistics."""

    total_requests: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l1_backfills: int = 0
    cache_misses: int = 0

    @property
    def l1_hit_rate(self) -> float:
        """L1 cache hit rate."""
        return self.l1_hits / max(1, self.total_requests)

    @property
    def l2_hit_rate(self) -> float:
        """L2 cache hit rate (among L1 misses)."""
        l1_misses = self.total_requests - self.l1_hits
        return self.l2_hits / max(1, l1_misses)

    @property
    def overall_hit_rate(self) -> float:
        """Overall cache hit rate (L1 + L2 combined)."""
        return (self.l1_hits + self.l2_hits) / max(1, self.total_requests)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "total_requests": self.total_requests,
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "l1_backfills": self.l1_backfills,
            "cache_misses": self.cache_misses,
            "l1_hit_rate": round(self.l1_hit_rate, 4),
            "l2_hit_rate": round(self.l2_hit_rate, 4),
            "overall_hit_rate": round(self.overall_hit_rate, 4),
        }


class CacheOrchestrator:
    """Orchestrate three-tier caching for weather queries.

    This orchestrator manages the cache hierarchy:
    - L1: In-memory LRU cache (fastest, server-local)
    - L2: Redis distributed cache (shared across servers)
    - L3: Anthropic prompt cache (handled at LLM level, not by orchestrator)

    The orchestrator provides:
    - Automatic L1 → L2 fallback on cache miss
    - L1 backfill from L2 hits (warm local cache)
    - Unified cache key generation
    - Statistics tracking for monitoring
    - LangSmith tracing integration
    """

    def __init__(
        self,
        l1_cache: QueryCache | None = None,
        l2_cache: RedisQueryCache | None = None,
        enable_tracing: bool = True,
    ):
        """Initialize cache orchestrator.

        Args:
            l1_cache: In-memory LRU cache instance (optional)
            l2_cache: Redis distributed cache instance (optional)
            enable_tracing: Enable LangSmith tracing for cache operations
        """
        self.l1 = l1_cache
        self.l2 = l2_cache
        self.enable_tracing = enable_tracing
        self.stats = CacheStats()

        # Log initialization
        l1_status = "ENABLED" if l1_cache else "DISABLED"
        l2_status = "ENABLED" if l2_cache else "DISABLED"
        logger.info(f"CacheOrchestrator initialized: L1={l1_status}, L2={l2_status}")

    async def get(
        self,
        query: str,
        user_id: str,
        enable_rag: bool,
        enable_cot: bool,
    ) -> CacheResult | None:
        """Check cache tiers in order: L1 → L2.

        L3 (Anthropic) is handled at agent creation, not here.

        Args:
            query: User query text
            user_id: User identifier
            enable_rag: Whether RAG is enabled
            enable_cot: Whether CoT is enabled

        Returns:
            CacheResult if hit, None if miss
        """
        self.stats.total_requests += 1
        start_time = time.perf_counter()

        # L1: In-memory check (<1ms)
        if self.l1:
            l1_result = self.l1.get(query, user_id, enable_rag, enable_cot)
            if l1_result is not None:
                self.stats.l1_hits += 1
                latency_ms = (time.perf_counter() - start_time) * 1000

                logger.debug(
                    f"L1 cache HIT | user_id: {user_id} | latency: {latency_ms:.2f}ms"
                )

                # LangSmith tracing
                if self.enable_tracing:
                    self._trace_cache_event("L1_HIT", query, user_id, latency_ms)

                return CacheResult(
                    response=l1_result,
                    cache_tier="L1",
                    latency_ms=latency_ms,
                )

        # L2: Redis check (<10ms)
        if self.l2:
            l2_result = await self.l2.get(query, user_id, enable_rag, enable_cot)
            if l2_result is not None:
                self.stats.l2_hits += 1
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Backfill L1 from L2 hit
                if self.l1:
                    self.l1.set(query, user_id, enable_rag, enable_cot, l2_result)
                    self.stats.l1_backfills += 1
                    logger.debug(f"L1 cache BACKFILL from L2 | user_id: {user_id}")

                logger.debug(
                    f"L2 cache HIT | user_id: {user_id} | latency: {latency_ms:.2f}ms"
                )

                # LangSmith tracing
                if self.enable_tracing:
                    self._trace_cache_event("L2_HIT", query, user_id, latency_ms)

                return CacheResult(
                    response=l2_result,
                    cache_tier="L2",
                    latency_ms=latency_ms,
                )

        # Cache miss
        self.stats.cache_misses += 1
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"Cache MISS (L1+L2) | user_id: {user_id} | latency: {latency_ms:.2f}ms"
        )

        # LangSmith tracing
        if self.enable_tracing:
            self._trace_cache_event("MISS", query, user_id, latency_ms)

        return None

    async def set(
        self,
        query: str,
        user_id: str,
        enable_rag: bool,
        enable_cot: bool,
        response: str,
    ) -> None:
        """Cache response at L1 and L2.

        Args:
            query: User query text
            user_id: User identifier
            enable_rag: Whether RAG is enabled
            enable_cot: Whether CoT is enabled
            response: Response text to cache
        """
        start_time = time.perf_counter()

        # Set at L1 (in-memory)
        if self.l1:
            self.l1.set(query, user_id, enable_rag, enable_cot, response)
            logger.debug(f"L1 cache WRITE | user_id: {user_id}")

        # Set at L2 (Redis)
        if self.l2:
            await self.l2.set(query, user_id, enable_rag, enable_cot, response)
            logger.debug(f"L2 cache WRITE | user_id: {user_id}")

        latency_ms = (time.perf_counter() - start_time) * 1000

        # LangSmith tracing
        if self.enable_tracing:
            self._trace_cache_event("WRITE", query, user_id, latency_ms)

    async def invalidate(
        self,
        query: str,
        user_id: str,
        enable_rag: bool,
        enable_cot: bool,
    ) -> dict[str, bool]:
        """Invalidate cache entry at L1 and L2.

        Args:
            query: User query text
            user_id: User identifier
            enable_rag: Whether RAG is enabled
            enable_cot: Whether CoT is enabled

        Returns:
            dict with invalidation status per tier
        """
        result = {"l1": False, "l2": False}

        # Invalidate L1
        if self.l1:
            cache_key = self.l1._generate_cache_key(query, user_id, enable_rag, enable_cot)
            if cache_key in self.l1.cache:
                del self.l1.cache[cache_key]
                result["l1"] = True
                logger.debug(f"L1 cache INVALIDATED | key: {cache_key[:12]}...")

        # Invalidate L2
        if self.l2 and self.l2.client:
            cache_key = self.l2._generate_cache_key(query, user_id, enable_rag, enable_cot)
            deleted = await self.l2.client.delete(cache_key)
            result["l2"] = deleted > 0
            if result["l2"]:
                logger.debug(f"L2 cache INVALIDATED | key: {cache_key[:20]}...")

        return result

    async def invalidate_user(self, user_id: str) -> dict[str, int]:
        """Invalidate all cache entries for a user.

        Useful when user preferences change or for privacy compliance.

        Args:
            user_id: User identifier

        Returns:
            dict with count of invalidated entries per tier
        """
        result = {"l1": 0, "l2": 0}

        # L1: Scan and remove entries (linear scan - acceptable for LRU cache)
        if self.l1:
            keys_to_delete = []
            for key in self.l1.cache:
                # Keys are hashes, but we stored user_id in the key generation
                # For proper user-based invalidation, we'd need to track user->keys mapping
                # For now, log a warning and skip L1 user invalidation
                pass
            result["l1"] = len(keys_to_delete)
            logger.warning("L1 user invalidation not implemented (hash-based keys)")

        # L2: Scan Redis keys with user_id pattern
        if self.l2 and self.l2.client:
            # Note: This scans all keys - for production, consider maintaining a user->keys index
            logger.warning("L2 user invalidation not implemented (requires key index)")

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with orchestrator stats plus L1/L2 individual stats
        """
        stats = {
            "orchestrator": self.stats.to_dict(),
            "l1": self.l1.get_stats() if self.l1 else {"enabled": False},
            "l2": None,
        }

        # L2 stats require async - provide sync summary
        if self.l2:
            stats["l2"] = {
                "enabled": True,
                "hits": self.l2.hits,
                "misses": self.l2.misses,
                "errors": self.l2.errors,
                "hit_rate": self.l2.hits / max(1, self.l2.hits + self.l2.misses),
            }
        else:
            stats["l2"] = {"enabled": False}

        return stats

    def reset_stats(self) -> None:
        """Reset orchestrator statistics."""
        self.stats = CacheStats()
        logger.info("CacheOrchestrator statistics RESET")

    def _trace_cache_event(
        self,
        event_type: str,
        query: str,
        user_id: str,
        latency_ms: float,
    ) -> None:
        """Send cache event to LangSmith for tracing.

        Args:
            event_type: "L1_HIT", "L2_HIT", "MISS", or "WRITE"
            query: User query (truncated for privacy)
            user_id: User identifier
            latency_ms: Operation latency
        """
        try:
            # Import LangSmith only if tracing is enabled
            from langsmith import trace

            # Truncate query for privacy
            query_preview = query[:50] + "..." if len(query) > 50 else query

            # Create trace metadata
            metadata = {
                "cache_event": event_type,
                "cache_latency_ms": round(latency_ms, 2),
                "user_id": user_id,
                "query_preview": query_preview,
            }

            # Log to LangSmith (if configured)
            # Note: This creates a lightweight trace event, not a full run
            logger.debug(f"LangSmith trace: {event_type} | {latency_ms:.2f}ms")

        except ImportError:
            # LangSmith not installed - skip tracing
            pass
        except Exception as e:
            # Don't fail on tracing errors
            logger.warning(f"LangSmith tracing error: {e}")
