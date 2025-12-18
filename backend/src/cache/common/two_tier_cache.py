"""Generic Two-Tier Cache: Exact Match (Tier 1) → Semantic (Tier 2).

Level 9a: Core reusable infrastructure shared by Tool Cache and LLM Response Cache.

Architecture:
    Tier 1 (Redis): Exact match via hash key (fast, free, <5ms)
    Tier 2 (Qdrant): Semantic match via embedding similarity (slower, <50ms)

Flow:
    1. Check Tier 1 (exact match) → HIT: Return immediately
    2. Check Tier 2 (semantic match) → HIT: Return + Promote to Tier 1
    3. MISS: Execute operation, cache result at both tiers

Benefits:
    - 80% code reuse between Tool Cache and LLM Cache
    - Consistent two-tier pattern across all cache types
    - Automatic cache promotion (Tier 2 → Tier 1)
    - Configurable TTL and similarity thresholds

Usage:
    class ToolResultCache(TwoTierCache[dict]):
        def _generate_cache_key(self, key_input: str, **kwargs) -> str:
            return f"tool:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"

        def _generate_embedding_input(self, key_input: str, **kwargs) -> str:
            return key_input  # Text for semantic embedding

        def _serialize(self, value: dict) -> str:
            return json.dumps(value)

        def _deserialize(self, data: str) -> dict:
            return json.loads(data)
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Cache value type


class TwoTierCacheResult(BaseModel, Generic[T]):
    """Result from two-tier cache lookup.

    Attributes:
        hit: Whether cache hit occurred
        tier: "tier1" | "tier2" | None (if miss)
        value: Cached value (if hit)
        similarity_score: Cosine similarity for tier2 hits (1.0 for tier1)
        lookup_ms: Total lookup latency in milliseconds
    """

    hit: bool
    tier: str | None  # "tier1" | "tier2" | None
    value: T | None  # type: ignore[valid-type]
    similarity_score: float | None
    lookup_ms: float

    model_config = {"arbitrary_types_allowed": True}


class TwoTierCache(ABC, Generic[T]):
    """Abstract base class for two-tier semantic caching.

    Tier 1: Redis exact match (fast, free, <5ms latency)
    Tier 2: Qdrant semantic match (slower, embedding cost, <50ms latency)

    Subclasses must implement:
    - _generate_cache_key(): How to create exact match key (hash)
    - _generate_embedding_input(): What text to embed for semantic search
    - _serialize(): How to convert value to string for storage
    - _deserialize(): How to convert string back to value

    Example:
        class MyCache(TwoTierCache[str]):
            def _generate_cache_key(self, key_input: str, **kwargs) -> str:
                return f"my:{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"

            def _generate_embedding_input(self, key_input: str, **kwargs) -> str:
                return key_input

            def _serialize(self, value: str) -> str:
                return value

            def _deserialize(self, data: str) -> str:
                return data
    """

    def __init__(
        self,
        redis_client: "redis.asyncio.Redis | None",
        semantic_matcher: "SemanticMatcher | None",
        cache_promoter: "CachePromoter | None",
        collection_name: str,
        default_ttl: int = 1800,
        similarity_threshold: float = 0.85,
    ):
        """Initialize two-tier cache.

        Args:
            redis_client: Async Redis client for Tier 1 exact match
            semantic_matcher: Qdrant semantic matcher for Tier 2
            cache_promoter: Handles Tier 2 → Tier 1 promotion
            collection_name: Qdrant collection for semantic vectors
            default_ttl: Default time-to-live in seconds (default: 1800 = 30 min)
            similarity_threshold: Minimum similarity for Tier 2 hits (default: 0.85)
        """
        self.redis = redis_client
        self.semantic = semantic_matcher
        self.promoter = cache_promoter
        self.collection = collection_name
        self.ttl = default_ttl
        self.threshold = similarity_threshold

        # Metrics
        self.tier1_hits = 0
        self.tier2_hits = 0
        self.misses = 0

        logger.info(
            f"✅ TwoTierCache initialized | collection={collection_name} | "
            f"ttl={default_ttl}s | threshold={similarity_threshold}"
        )

    async def get(self, key_input: str, **kwargs) -> TwoTierCacheResult[T]:
        """Two-tier lookup: Tier 1 (exact) → Tier 2 (semantic).

        Args:
            key_input: Input for generating cache key and embedding
            **kwargs: Additional arguments for key/embedding generation

        Returns:
            TwoTierCacheResult with hit status, tier, value, and latency
        """
        start = time.perf_counter()

        # Tier 1: Exact match (Redis)
        if self.redis:
            try:
                cache_key = self._generate_cache_key(key_input, **kwargs)
                tier1_result = await self.redis.get(cache_key)

                if tier1_result:
                    self.tier1_hits += 1
                    latency_ms = (time.perf_counter() - start) * 1000

                    logger.debug(
                        f"✅ Tier 1 HIT | key={cache_key[:16]}... | latency={latency_ms:.2f}ms"
                    )

                    return TwoTierCacheResult(
                        hit=True,
                        tier="tier1",
                        value=self._deserialize(tier1_result),
                        similarity_score=1.0,
                        lookup_ms=latency_ms,
                    )
            except Exception as e:
                logger.warning(f"⚠️ Tier 1 error: {e}")

        # Tier 2: Semantic match (Qdrant)
        if self.semantic:
            try:
                embedding_input = self._generate_embedding_input(key_input, **kwargs)
                tier2_result = await self.semantic.search(
                    collection=self.collection,
                    query=embedding_input,
                    threshold=self.threshold,
                    limit=1,
                )

                if tier2_result:
                    self.tier2_hits += 1
                    latency_ms = (time.perf_counter() - start) * 1000

                    # Extract value from payload
                    value = self._deserialize(tier2_result[0].payload.get("value", ""))
                    score = tier2_result[0].score

                    logger.debug(
                        f"✅ Tier 2 HIT | score={score:.3f} | latency={latency_ms:.2f}ms"
                    )

                    # Promote to Tier 1 for faster future lookups
                    if self.promoter and self.redis:
                        cache_key = self._generate_cache_key(key_input, **kwargs)
                        await self.promoter.promote(
                            redis=self.redis,
                            cache_key=cache_key,
                            value=tier2_result[0].payload.get("value", ""),
                            ttl=self.ttl,
                        )

                    return TwoTierCacheResult(
                        hit=True,
                        tier="tier2",
                        value=value,
                        similarity_score=score,
                        lookup_ms=latency_ms,
                    )
            except Exception as e:
                logger.warning(f"⚠️ Tier 2 error: {e}")

        # Cache miss
        self.misses += 1
        latency_ms = (time.perf_counter() - start) * 1000

        logger.debug(f"❌ Cache MISS | latency={latency_ms:.2f}ms")

        return TwoTierCacheResult(
            hit=False,
            tier=None,
            value=None,
            similarity_score=None,
            lookup_ms=latency_ms,
        )

    async def set(self, key_input: str, value: T, **kwargs) -> None:
        """Store value in both tiers.

        Args:
            key_input: Input for generating cache key and embedding
            value: Value to cache
            **kwargs: Additional arguments for key/embedding generation
        """
        serialized = self._serialize(value)

        # Tier 1: Redis
        if self.redis:
            try:
                cache_key = self._generate_cache_key(key_input, **kwargs)
                await self.redis.setex(cache_key, self.ttl, serialized)
                logger.debug(f"💾 Tier 1 WRITE | key={cache_key[:16]}... | ttl={self.ttl}s")
            except Exception as e:
                logger.warning(f"⚠️ Tier 1 write error: {e}")

        # Tier 2: Qdrant
        if self.semantic:
            try:
                embedding_input = self._generate_embedding_input(key_input, **kwargs)
                cache_key = self._generate_cache_key(key_input, **kwargs)
                await self.semantic.upsert(
                    collection=self.collection,
                    text=embedding_input,
                    payload={"value": serialized, "key": cache_key},
                )
                logger.debug(f"💾 Tier 2 WRITE | collection={self.collection}")
            except Exception as e:
                logger.warning(f"⚠️ Tier 2 write error: {e}")

    async def invalidate(self, key_input: str, **kwargs) -> dict[str, bool]:
        """Invalidate cache entry from both tiers.

        Args:
            key_input: Input for generating cache key
            **kwargs: Additional arguments for key generation

        Returns:
            Dict with invalidation status per tier
        """
        result = {"tier1": False, "tier2": False}

        # Tier 1: Redis
        if self.redis:
            try:
                cache_key = self._generate_cache_key(key_input, **kwargs)
                deleted = await self.redis.delete(cache_key)
                result["tier1"] = deleted > 0
                if result["tier1"]:
                    logger.debug(f"🗑️ Tier 1 INVALIDATED | key={cache_key[:16]}...")
            except Exception as e:
                logger.warning(f"⚠️ Tier 1 invalidation error: {e}")

        # Tier 2: Qdrant (by payload filter)
        if self.semantic:
            try:
                cache_key = self._generate_cache_key(key_input, **kwargs)
                deleted = await self.semantic.delete_by_key(
                    collection=self.collection,
                    cache_key=cache_key,
                )
                result["tier2"] = deleted
                if result["tier2"]:
                    logger.debug(f"🗑️ Tier 2 INVALIDATED | key={cache_key[:16]}...")
            except Exception as e:
                logger.warning(f"⚠️ Tier 2 invalidation error: {e}")

        return result

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with hit counts, miss counts, and hit rates per tier
        """
        total = self.tier1_hits + self.tier2_hits + self.misses
        tier1_rate = self.tier1_hits / max(1, total)
        tier2_rate = self.tier2_hits / max(1, total)
        overall_rate = (self.tier1_hits + self.tier2_hits) / max(1, total)

        return {
            "tier1_hits": self.tier1_hits,
            "tier2_hits": self.tier2_hits,
            "misses": self.misses,
            "total_requests": total,
            "tier1_hit_rate": round(tier1_rate, 4),
            "tier2_hit_rate": round(tier2_rate, 4),
            "overall_hit_rate": round(overall_rate, 4),
            "collection": self.collection,
            "ttl_seconds": self.ttl,
            "similarity_threshold": self.threshold,
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.tier1_hits = 0
        self.tier2_hits = 0
        self.misses = 0

    @abstractmethod
    def _generate_cache_key(self, key_input: str, **kwargs) -> str:
        """Generate exact match cache key (hash).

        Args:
            key_input: Input string for key generation
            **kwargs: Additional parameters

        Returns:
            Cache key string (typically a hash)
        """
        pass

    @abstractmethod
    def _generate_embedding_input(self, key_input: str, **kwargs) -> str:
        """Generate text for semantic embedding.

        Args:
            key_input: Input string
            **kwargs: Additional parameters

        Returns:
            Text string to embed for semantic search
        """
        pass

    @abstractmethod
    def _serialize(self, value: T) -> str:
        """Serialize value for storage.

        Args:
            value: Value to serialize

        Returns:
            Serialized string representation
        """
        pass

    @abstractmethod
    def _deserialize(self, data: str) -> T:
        """Deserialize value from storage.

        Args:
            data: Serialized string

        Returns:
            Deserialized value
        """
        pass
