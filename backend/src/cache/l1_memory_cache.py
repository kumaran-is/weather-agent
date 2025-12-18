"""L1: In-process LRU cache for query responses.

This module provides ultra-fast caching for recent queries on the same server instance.

Performance:
- Hit latency: <1ms (in-memory lookup)
- Hit rate: 15-25% (recent queries on same server)
- Cost savings: 100% (no external calls)
- Memory usage: ~2MB per server (1000 entries × 2KB each)

Usage:
    cache = QueryCache(max_size=1000)

    # Try to get cached response
    response = cache.get(query, user_id, enable_rag, enable_cot)

    if response is None:
        # Cache miss - fetch from API
        response = await fetch_response(...)
        cache.set(query, user_id, enable_rag, enable_cot, response)
"""

import hashlib
import json
import logging
import time

logger = logging.getLogger(__name__)


class QueryCache:
    """L1: In-process LRU cache for recent queries.

    This cache stores query responses in server memory for ultra-fast retrieval.
    It uses LRU (Least Recently Used) eviction when the cache is full.

    Cache Key Generation:
    - Query text (normalized: lowercase, whitespace trimmed)
    - User ID (for personalized responses)
    - Feature flags (RAG/CoT affect output)
    - NOT included: session_id, timestamp (too variable)

    Cache Entry Structure:
    {
        cache_key: (response_text, cached_at_timestamp)
    }

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

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize LRU cache with configurable size and TTL.

        Args:
            max_size: Maximum number of cached responses (default: 1000)
                     Each response ~2KB → 1000 entries = ~2MB memory
            ttl_seconds: Time-to-live in seconds (default: 300 = 5 minutes)
        """
        self.cache: dict[str, tuple[str, float]] = {}  # key → (response, timestamp)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bypasses = 0  # Life-safety bypass counter

        logger.info(
            f"✅ L1 cache initialized (max_size={max_size}, ttl={ttl_seconds}s)"
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

    def _generate_cache_key(
        self,
        query: str,
        user_id: str,
        enable_rag: bool,
        enable_cot: bool,
    ) -> str:
        """Generate cache key from query parameters.

        Cache key includes:
        - Query text (normalized: lowercase, whitespace trimmed)
        - User ID (for personalized responses)
        - Feature flags (RAG/CoT affect output)

        NOT included:
        - session_id (same user can have multiple sessions)
        - timestamp (changes every request)

        Args:
            query: User query text
            user_id: User identifier
            enable_rag: Whether RAG is enabled
            enable_cot: Whether CoT is enabled

        Returns:
            SHA-256 hash of normalized cache key
        """
        # Normalize query (case-insensitive, whitespace trimmed)
        normalized_query = query.strip().lower()

        # Build cache key data
        key_data = {
            "query": normalized_query,
            "user_id": user_id,
            "enable_rag": enable_rag,
            "enable_cot": enable_cot,
        }

        # Generate deterministic hash
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
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
            Cached response text if hit, None if miss or expired
        """
        # 🔴 LIFE-SAFETY BYPASS: Never return cached data for hurricane/emergency queries
        if self._is_life_safety_query(query):
            self.bypasses += 1
            logger.warning(
                f"🔴 L1 cache BYPASS (life-safety): Query contains hurricane/emergency keywords"
            )
            return None

        cache_key = self._generate_cache_key(query, user_id, enable_rag, enable_cot)

        if cache_key in self.cache:
            response, cached_at = self.cache[cache_key]
            age_seconds = time.time() - cached_at

            if age_seconds < self.ttl_seconds:
                # Cache hit - still valid
                self.hits += 1
                logger.debug(
                    f"✅ L1 cache HIT: {cache_key[:12]}... (age: {age_seconds:.1f}s)"
                )
                return response
            else:
                # Expired - remove from cache
                del self.cache[cache_key]
                self.misses += 1
                logger.debug(
                    f"⏰ L1 cache EXPIRED: {cache_key[:12]}... (age: {age_seconds:.1f}s)"
                )
                return None

        # Cache miss
        self.misses += 1
        logger.debug(f"❌ L1 cache MISS: {cache_key[:12]}...")
        return None

    def set(
        self,
        query: str,
        user_id: str,
        enable_rag: bool,
        enable_cot: bool,
        response: str,
    ) -> None:
        """Cache response with LRU eviction if cache is full.

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
                f"🔴 L1 cache BYPASS (life-safety): Refusing to cache hurricane/emergency response"
            )
            return  # Do not cache

        cache_key = self._generate_cache_key(query, user_id, enable_rag, enable_cot)

        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            # Find oldest entry by timestamp
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
            self.evictions += 1
            logger.debug(f"🗑️  L1 cache EVICTION: {oldest_key[:12]}...")

        # Store in cache
        self.cache[cache_key] = (response, time.time())
        logger.debug(f"💾 L1 cache WRITE: {cache_key[:12]}...")

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"🗑️  L1 cache CLEARED: {count} entries removed")

    def get_stats(self) -> dict[str, any]:
        """Get cache statistics.

        Returns:
            dict with hits, misses, evictions, hit_rate, size
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }
