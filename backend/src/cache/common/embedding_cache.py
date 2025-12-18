"""Embedding Cache to avoid redundant embedding API calls.

Level 9a: Cache embeddings to reduce OpenAI embedding costs by 50%+.

Architecture:
    - In-memory LRU cache for recent embeddings
    - Optional Redis backend for cross-server sharing
    - Configurable max size and TTL

Cost Impact:
    - text-embedding-3-small: $0.02 per 1M tokens
    - 100 queries/day × 50 tokens avg = 5000 tokens = $0.0001/day
    - With 60% cache hit rate = $0.00004/day (60% savings)

Usage:
    from backend.src.cache.common import EmbeddingCache

    cache = EmbeddingCache(max_size=10000, ttl_seconds=3600)

    # Get or create embedding
    embedding = await cache.get_or_create("weather in Miami", embeddings)

    # Direct get/set
    await cache.set("weather in Miami", [0.1, 0.2, ...])
    embedding = await cache.get("weather in Miami")
"""

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """In-memory LRU cache for embeddings.

    Caches embedding vectors to avoid redundant API calls.
    Uses hash-based keys for efficient lookup.

    Attributes:
        cache: Dict mapping text hash → (embedding, timestamp)
        max_size: Maximum cache entries
        ttl_seconds: Time-to-live in seconds
        hits/misses: Hit/miss counters
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600,
        redis_client: Any = None,  # Optional redis.asyncio.Redis
    ):
        """Initialize embedding cache.

        Args:
            max_size: Maximum entries in memory (default: 10000)
            ttl_seconds: TTL in seconds (default: 3600 = 1 hour)
            redis_client: Optional Redis client for distributed caching
        """
        self.cache: dict[str, tuple[list[float], float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.redis = redis_client
        self.redis_prefix = "emb:"

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        logger.info(
            f"✅ EmbeddingCache initialized | max_size={max_size} | "
            f"ttl={ttl_seconds}s | redis={'ENABLED' if redis_client else 'DISABLED'}"
        )

    def _hash_text(self, text: str) -> str:
        """Generate hash key from text.

        Args:
            text: Text to hash

        Returns:
            SHA-256 hash (first 32 chars)
        """
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    async def get(self, text: str) -> list[float] | None:
        """Get cached embedding for text.

        Args:
            text: Text to look up

        Returns:
            Embedding vector if cached and not expired, None otherwise
        """
        key = self._hash_text(text)

        # Check in-memory cache first
        if key in self.cache:
            embedding, cached_at = self.cache[key]
            age = time.time() - cached_at

            if age < self.ttl_seconds:
                self.hits += 1
                logger.debug(f"✅ Embedding cache HIT (memory) | key={key[:12]}...")
                return embedding
            else:
                # Expired
                del self.cache[key]

        # Check Redis if available
        if self.redis:
            try:
                redis_key = f"{self.redis_prefix}{key}"
                data = await self.redis.get(redis_key)

                if data:
                    embedding = json.loads(data)
                    # Backfill to memory
                    await self._set_memory(key, embedding)
                    self.hits += 1
                    logger.debug(f"✅ Embedding cache HIT (redis) | key={key[:12]}...")
                    return embedding
            except Exception as e:
                logger.warning(f"⚠️ Redis embedding cache error: {e}")

        self.misses += 1
        logger.debug(f"❌ Embedding cache MISS | key={key[:12]}...")
        return None

    async def set(self, text: str, embedding: list[float]) -> None:
        """Cache embedding for text.

        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        key = self._hash_text(text)

        # Set in memory
        await self._set_memory(key, embedding)

        # Set in Redis if available
        if self.redis:
            try:
                redis_key = f"{self.redis_prefix}{key}"
                await self.redis.setex(
                    redis_key,
                    self.ttl_seconds,
                    json.dumps(embedding),
                )
                logger.debug(f"💾 Embedding cache WRITE (redis) | key={key[:12]}...")
            except Exception as e:
                logger.warning(f"⚠️ Redis embedding cache write error: {e}")

    async def _set_memory(self, key: str, embedding: list[float]) -> None:
        """Set embedding in memory cache with LRU eviction.

        Args:
            key: Hash key
            embedding: Embedding vector
        """
        # Evict oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
            self.evictions += 1
            logger.debug(f"🗑️ Embedding cache EVICTION | key={oldest_key[:12]}...")

        self.cache[key] = (embedding, time.time())
        logger.debug(f"💾 Embedding cache WRITE (memory) | key={key[:12]}...")

    async def get_or_create(
        self,
        text: str,
        embeddings: Any,  # LangChain embeddings instance
    ) -> list[float]:
        """Get embedding from cache or create and cache it.

        Args:
            text: Text to embed
            embeddings: LangChain embeddings instance

        Returns:
            Embedding vector
        """
        # Try cache first
        cached = await self.get(text)
        if cached is not None:
            return cached

        # Generate embedding
        embedding = await embeddings.aembed_query(text)

        # Cache for future
        await self.set(text, embedding)

        return embedding

    def clear(self) -> None:
        """Clear all cached embeddings."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"🗑️ Embedding cache CLEARED | {count} entries removed")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hit rate, size, and counts
        """
        total = self.hits + self.misses
        hit_rate = self.hits / max(1, total)

        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(hit_rate, 4),
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "redis_enabled": self.redis is not None,
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
