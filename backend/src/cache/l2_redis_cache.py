"""L2: Redis distributed cache for query responses.

This module provides shared caching across multiple server instances using Redis.

Performance:
- Hit latency: <10ms (Redis network roundtrip)
- Hit rate: 30-40% (common queries across all servers)
- Cost savings: 100% (no LLM calls)
- Storage: Unlimited (Redis-managed)

Usage:
    cache = RedisQueryCache()
    await cache.connect()

    # Try to get cached response
    response = await cache.get(query, user_id, enable_rag, enable_cot)

    if response is None:
        # Cache miss - fetch from API
        response = await fetch_response(...)
        await cache.set(query, user_id, enable_rag, enable_cot, response)

    await cache.close()
"""

import hashlib
import json
import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisQueryCache:
    """L2: Redis distributed cache for common queries across servers.

    This cache stores query responses in Redis, enabling sharing across
    multiple server instances. It provides higher hit rates than L1 by
    serving common queries from a shared pool.

    DESIGN DISTINCTION (L1 vs L2):
    - L1 (in-memory): User-specific cache - includes user_id for personalized responses
    - L2 (Redis): Query-level cache - EXCLUDES user_id for cross-user sharing

    Cache Key Generation:
    - Query text (normalized: lowercase, whitespace trimmed)
    - Feature flags (RAG/CoT affect output)
    - NOT included: user_id (enables cross-user sharing), session_id, timestamp

    Cache Entry Structure:
    Redis Key: "weather:cache:{cache_key_hash}"
    Redis Value: JSON string of response text
    Redis TTL: 30 minutes (1800 seconds)

    Life-Safety Bypass:
    Hurricane and emergency queries are NEVER cached to ensure fresh, accurate data.
    """

    # 🔴 CRITICAL: Life-safety keywords that trigger cache bypass
    LIFE_SAFETY_KEYWORDS = {
        "hurricane", "hurricanes",
        "evacuation", "evacuate", "evacuating",
        "emergency", "emergencies",
        "alert", "alerts", "warning", "warnings",
        "storm surge", "landfall",
        "category 3", "category 4", "category 5", "cat 3", "cat 4", "cat 5",
        "evacuation zone", "evacuation order",
        "shelter", "shelters",
        "life-threatening", "life threatening",
        "dangerous", "danger",
    }

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 1800,
        key_prefix: str = "weather:cache:",
    ):
        """Initialize Redis cache with configurable URL and TTL.

        Args:
            redis_url: Redis connection URL (default: redis://localhost:6379/0)
            ttl_seconds: Time-to-live in seconds (default: 1800 = 30 minutes)
            key_prefix: Redis key prefix for namespacing (default: weather:cache:)
        """
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.client: redis.Redis | None = None

        # Metrics
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.bypasses = 0  # Life-safety bypass counter

        logger.info(
            f"✅ L2 cache initialized (url={redis_url}, ttl={ttl_seconds}s)"
        )

    def _is_life_safety_query(self, query: str) -> bool:
        """Check if query contains life-safety keywords that should bypass cache.

        Hurricane and emergency queries must ALWAYS fetch fresh data to avoid
        serving stale, potentially dangerous information.

        Args:
            query: User query text

        Returns:
            True if query contains life-safety keywords, False otherwise
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.LIFE_SAFETY_KEYWORDS)

    async def connect(self) -> None:
        """Establish connection to Redis server.

        Raises:
            redis.ConnectionError: If connection fails
        """
        try:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            await self.client.ping()
            logger.info(f"✅ L2 cache connected to Redis: {self.redis_url}")
        except Exception as e:
            logger.error(f"❌ L2 cache connection failed: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection gracefully."""
        if self.client:
            await self.client.aclose()
            logger.info("✅ L2 cache connection closed")

    def _generate_cache_key(
        self,
        query: str,
        user_id: str,  # kept for API compatibility, but NOT used in key
        enable_rag: bool,
        enable_cot: bool,
    ) -> str:
        """Generate cache key from query parameters.

        IMPORTANT: L2 cache is query-level (shared across users).
        Unlike L1, user_id is NOT included in the cache key.

        Cache key includes:
        - Query text (normalized: lowercase, whitespace trimmed)
        - Feature flags (RAG/CoT affect output)

        NOT included:
        - user_id (enables cross-user sharing for same queries)
        - session_id (same user can have multiple sessions)
        - timestamp (changes every request)

        Args:
            query: User query text
            user_id: User identifier (ignored - kept for API compatibility)
            enable_rag: Whether RAG is enabled
            enable_cot: Whether CoT is enabled

        Returns:
            Redis key with prefix (e.g., "weather:cache:abc123...")
        """
        # Normalize query (case-insensitive, whitespace trimmed)
        normalized_query = query.strip().lower()

        # Build cache key data - NOTE: user_id intentionally excluded
        # L2 is query-level (shared), L1 is user-level (personalized)
        key_data = {
            "query": normalized_query,
            # user_id excluded - enables cross-user cache sharing
            "enable_rag": enable_rag,
            "enable_cot": enable_cot,
        }

        # Generate deterministic hash
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()

        return f"{self.key_prefix}{key_hash}"

    async def get(
        self,
        query: str,
        user_id: str,
        enable_rag: bool,
        enable_cot: bool,
    ) -> str | None:
        """Get cached response if available and not expired.

        🔴 CRITICAL: Life-safety queries (hurricane, emergency) are NEVER cached.

        Args:
            query: User query text
            user_id: User identifier
            enable_rag: Whether RAG is enabled
            enable_cot: Whether CoT is enabled

        Returns:
            Cached response text if hit, None if miss or error
        """
        # 🔴 LIFE-SAFETY BYPASS: Never return cached data for hurricane/emergency queries
        if self._is_life_safety_query(query):
            self.bypasses += 1
            logger.warning(
                f"🔴 L2 cache BYPASS (life-safety): Query contains hurricane/emergency keywords"
            )
            return None

        if not self.client:
            logger.warning("⚠️  L2 cache not connected, skipping")
            return None

        cache_key = self._generate_cache_key(query, user_id, enable_rag, enable_cot)

        try:
            response = await self.client.get(cache_key)

            if response:
                # Cache hit
                self.hits += 1
                # Get TTL for logging
                ttl = await self.client.ttl(cache_key)
                logger.debug(
                    f"✅ L2 cache HIT: {cache_key[len(self.key_prefix):len(self.key_prefix)+12]}... (ttl: {ttl}s)"
                )
                return response
            else:
                # Cache miss
                self.misses += 1
                logger.debug(
                    f"❌ L2 cache MISS: {cache_key[len(self.key_prefix):len(self.key_prefix)+12]}..."
                )
                return None

        except Exception as e:
            # Redis error - graceful degradation
            self.errors += 1
            logger.error(f"⚠️  L2 cache GET error: {e}")
            return None

    async def set(
        self,
        query: str,
        user_id: str,
        enable_rag: bool,
        enable_cot: bool,
        response: str,
    ) -> None:
        """Cache response with TTL.

        🔴 CRITICAL: Life-safety queries (hurricane, emergency) are NEVER cached.

        Args:
            query: User query text
            user_id: User identifier
            enable_rag: Whether RAG is enabled
            enable_cot: Whether CoT is enabled
            response: Response text to cache
        """
        # 🔴 LIFE-SAFETY BYPASS: Never cache hurricane/emergency responses
        if self._is_life_safety_query(query):
            self.bypasses += 1
            logger.warning(
                f"🔴 L2 cache BYPASS (life-safety): Refusing to cache hurricane/emergency response"
            )
            return  # Do not cache

        if not self.client:
            logger.warning("⚠️  L2 cache not connected, skipping write")
            return

        cache_key = self._generate_cache_key(query, user_id, enable_rag, enable_cot)

        try:
            # Store with TTL
            await self.client.set(
                cache_key,
                response,
                ex=self.ttl_seconds,
            )
            logger.debug(
                f"💾 L2 cache WRITE: {cache_key[len(self.key_prefix):len(self.key_prefix)+12]}... (ttl: {self.ttl_seconds}s)"
            )

        except Exception as e:
            # Redis error - graceful degradation
            self.errors += 1
            logger.error(f"⚠️  L2 cache SET error: {e}")

    async def clear(self) -> None:
        """Clear all cache entries with this prefix."""
        if not self.client:
            logger.warning("⚠️  L2 cache not connected, cannot clear")
            return

        try:
            # Find all keys with our prefix
            cursor = 0
            count = 0

            while True:
                cursor, keys = await self.client.scan(
                    cursor=cursor,
                    match=f"{self.key_prefix}*",
                    count=100,
                )

                if keys:
                    await self.client.delete(*keys)
                    count += len(keys)

                if cursor == 0:
                    break

            logger.info(f"🗑️  L2 cache CLEARED: {count} entries removed")

        except Exception as e:
            self.errors += 1
            logger.error(f"⚠️  L2 cache CLEAR error: {e}")

    async def get_stats(self) -> dict[str, any]:
        """Get cache statistics.

        Returns:
            dict with hits, misses, errors, hit_rate, redis_info
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        stats = {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }

        # Add Redis info if connected
        if self.client:
            try:
                info = await self.client.info("memory")
                stats["redis_memory_mb"] = info.get("used_memory", 0) / (1024 * 1024)
                stats["redis_peak_memory_mb"] = info.get("used_memory_peak", 0) / (1024 * 1024)
            except Exception as e:
                logger.error(f"⚠️  L2 cache STATS error: {e}")

        return stats
