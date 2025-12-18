"""Unit tests for L1: In-process LRU cache.

Tests validate the L1 cache implementation for:
- Cache key generation (deterministic hashing)
- Get/Set operations with TTL expiration
- LRU eviction when cache is full
- Cache statistics (hits, misses, evictions, hit_rate)
- Cache clearing

Expected Performance:
- Hit latency: <1ms (in-memory dictionary lookup)
- Hit rate: 15-25% (recent queries on same server)
- Cost savings: 100% (no external calls)
- Memory usage: ~2MB per server (1000 entries × 2KB each)
"""

import time

from backend.src.cache.l1_memory_cache import QueryCache


class TestL1MemoryCache:
    """Test L1: In-process LRU cache functionality."""

    def test_cache_key_generation_deterministic(self):
        """Test that cache key generation is deterministic and consistent."""
        cache = QueryCache()

        # Same input should generate same key
        key1 = cache._generate_cache_key(
            query="What's the weather in London?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        key2 = cache._generate_cache_key(
            query="What's the weather in London?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        assert key1 == key2, "Same inputs should generate same cache key"

        # Verify key format (SHA-256 hash)
        assert len(key1) == 64, "Cache key should be 64 characters (SHA-256 hex)"

    def test_cache_key_query_normalization(self):
        """Test that cache key normalizes query text (lowercase, trimmed)."""
        cache = QueryCache()

        # Different capitalization and whitespace should generate same key
        key1 = cache._generate_cache_key(
            query="  What's the WEATHER in London?  ",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        key2 = cache._generate_cache_key(
            query="what's the weather in london?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        assert key1 == key2, "Normalized queries should generate same key"

    def test_cache_key_different_user_ids(self):
        """Test that different user IDs generate different cache keys."""
        cache = QueryCache()

        key1 = cache._generate_cache_key(
            query="What's the weather?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        key2 = cache._generate_cache_key(
            query="What's the weather?",
            user_id="user456",
            enable_rag=True,
            enable_cot=False,
        )
        assert key1 != key2, "Different user IDs should generate different keys"

    def test_cache_key_different_feature_flags(self):
        """Test that different feature flags generate different cache keys."""
        cache = QueryCache()

        # RAG enabled vs disabled
        key1 = cache._generate_cache_key(
            query="What's the weather?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        key2 = cache._generate_cache_key(
            query="What's the weather?",
            user_id="user123",
            enable_rag=False,
            enable_cot=False,
        )
        assert key1 != key2, "Different RAG settings should generate different keys"

        # CoT enabled vs disabled
        key3 = cache._generate_cache_key(
            query="What's the weather?",
            user_id="user123",
            enable_rag=True,
            enable_cot=True,
        )
        assert key1 != key3, "Different CoT settings should generate different keys"

    def test_cache_set_and_get_hit(self):
        """Test cache set and successful get (cache hit)."""
        cache = QueryCache(ttl_seconds=300)

        # Set cache entry
        cache.set(
            query="What's the weather in London?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
            response="The weather in London is 15°C and rainy.",
        )

        # Get cache entry (should hit)
        response = cache.get(
            query="What's the weather in London?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert response == "The weather in London is 15°C and rainy."
        assert cache.hits == 1
        assert cache.misses == 0

    def test_cache_get_miss(self):
        """Test cache get with no entry (cache miss)."""
        cache = QueryCache()

        # Get non-existent entry
        response = cache.get(
            query="What's the weather in London?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert response is None
        assert cache.hits == 0
        assert cache.misses == 1

    def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        cache = QueryCache(ttl_seconds=1)  # 1 second TTL

        # Set cache entry
        cache.set(
            query="What's the weather?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
            response="15°C and sunny",
        )

        # Immediate get (should hit)
        response = cache.get(
            query="What's the weather?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        assert response == "15°C and sunny"
        assert cache.hits == 1

        # Wait for TTL to expire
        time.sleep(1.5)

        # Get after TTL (should miss)
        response = cache.get(
            query="What's the weather?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        assert response is None
        assert cache.hits == 1  # No change
        assert cache.misses == 1  # Incremented

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = QueryCache(max_size=3)  # Small cache for testing

        # Fill cache with 3 entries
        for i in range(3):
            cache.set(
                query=f"Query {i}",
                user_id="user123",
                enable_rag=True,
                enable_cot=False,
                response=f"Response {i}",
            )

        assert len(cache.cache) == 3
        assert cache.evictions == 0

        # Add 4th entry (should evict oldest)
        time.sleep(0.1)  # Ensure different timestamps
        cache.set(
            query="Query 3",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
            response="Response 3",
        )

        assert len(cache.cache) == 3  # Still at max
        assert cache.evictions == 1  # One eviction

        # Oldest entry (Query 0) should be evicted
        response = cache.get(
            query="Query 0",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        assert response is None  # Evicted

        # Newest entry (Query 3) should still be present
        response = cache.get(
            query="Query 3",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        assert response == "Response 3"

    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = QueryCache()

        # Add entries
        for i in range(5):
            cache.set(
                query=f"Query {i}",
                user_id="user123",
                enable_rag=True,
                enable_cot=False,
                response=f"Response {i}",
            )

        assert len(cache.cache) == 5

        # Clear cache
        cache.clear()

        assert len(cache.cache) == 0

        # All entries should be gone
        for i in range(5):
            response = cache.get(
                query=f"Query {i}",
                user_id="user123",
                enable_rag=True,
                enable_cot=False,
            )
            assert response is None

    def test_cache_stats(self):
        """Test cache statistics calculation."""
        cache = QueryCache(max_size=10, ttl_seconds=300)

        # Initial stats (no requests)
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["size"] == 0
        assert stats["max_size"] == 10
        assert stats["ttl_seconds"] == 300

        # Add some cache hits and misses
        cache.set("Q1", "user123", True, False, "R1")
        cache.set("Q2", "user123", True, False, "R2")

        cache.get("Q1", "user123", True, False)  # Hit
        cache.get("Q2", "user123", True, False)  # Hit
        cache.get("Q3", "user123", True, False)  # Miss

        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3  # 66.7%
        assert stats["size"] == 2

    def test_cache_hit_rate_calculation(self):
        """Test hit rate calculation with various scenarios."""
        cache = QueryCache()

        # Scenario: 70% hit rate
        # Add 10 entries
        for i in range(10):
            cache.set(f"Q{i}", "user123", True, False, f"R{i}")

        # 7 hits
        for i in range(7):
            cache.get(f"Q{i}", "user123", True, False)

        # 3 misses
        for i in range(10, 13):
            cache.get(f"Q{i}", "user123", True, False)

        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.7  # 7/10 = 70%
