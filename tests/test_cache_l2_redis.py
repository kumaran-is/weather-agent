"""Unit tests for L2: Redis distributed cache.

Tests validate the L2 cache implementation for:
- Redis connection and disconnection
- Async get/set operations with TTL
- Cache key generation (deterministic hashing)
- Graceful degradation on Redis errors
- Cache statistics (hits, misses, errors, hit_rate)
- Cache clearing with prefix matching

Expected Performance:
- Hit latency: <10ms (Redis network roundtrip)
- Hit rate: 30-40% (common queries across all servers)
- Cost savings: 100% (no LLM calls)
- Storage: Unlimited (Redis-managed)

Note: These tests use mock Redis to avoid external dependencies.
For integration tests with real Redis, see test_cache_integration.py
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.cache.l2_redis_cache import RedisQueryCache


@pytest.fixture
def mock_redis():
    """Create a mocked Redis client."""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.scan = AsyncMock(return_value=(0, []))
    mock.ttl = AsyncMock(return_value=1800)
    mock.aclose = AsyncMock()
    mock.info = AsyncMock(return_value={
        "used_memory": 1024 * 1024,  # 1MB
        "used_memory_peak": 2 * 1024 * 1024,  # 2MB
    })
    return mock


class TestL2RedisCache:
    """Test L2: Redis distributed cache functionality."""

    @pytest.mark.asyncio
    async def test_redis_connection_success(self, mock_redis):
        """Test successful Redis connection."""
        with patch("backend.src.cache.l2_redis_cache.redis.from_url", return_value=mock_redis):
            cache = RedisQueryCache(redis_url="redis://localhost:6379/0")
            await cache.connect()

            # Verify connection
            assert cache.client is not None
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self):
        """Test graceful handling of Redis connection failure."""
        with patch("backend.src.cache.l2_redis_cache.redis.from_url", side_effect=Exception("Connection failed")):
            cache = RedisQueryCache(redis_url="redis://localhost:6379/0")

            with pytest.raises(Exception):
                await cache.connect()

    @pytest.mark.asyncio
    async def test_redis_disconnect(self, mock_redis):
        """Test graceful Redis disconnection."""
        with patch("backend.src.cache.l2_redis_cache.redis.from_url", return_value=mock_redis):
            cache = RedisQueryCache()
            await cache.connect()
            await cache.close()

            mock_redis.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_key_generation_deterministic(self):
        """Test that cache key generation is deterministic and includes prefix."""
        cache = RedisQueryCache(key_prefix="test:")

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

        assert key1 == key2
        assert key1.startswith("test:")
        assert len(key1) == len("test:") + 64  # prefix + SHA-256 hash

    @pytest.mark.asyncio
    async def test_cache_set_and_get_hit(self, mock_redis):
        """Test cache set and successful get (cache hit)."""
        # Mock Redis to return cached value
        mock_redis.get.return_value = "The weather in London is 15°C and rainy."

        with patch("backend.src.cache.l2_redis_cache.redis.from_url", return_value=mock_redis):
            cache = RedisQueryCache(ttl_seconds=1800)
            await cache.connect()

            # Set cache entry
            await cache.set(
                query="What's the weather in London?",
                user_id="user123",
                enable_rag=True,
                enable_cot=False,
                response="The weather in London is 15°C and rainy.",
            )

            # Verify set was called with correct TTL
            mock_redis.set.assert_called_once()
            call_args = mock_redis.set.call_args
            assert call_args.kwargs["ex"] == 1800  # TTL

            # Get cache entry (should hit)
            response = await cache.get(
                query="What's the weather in London?",
                user_id="user123",
                enable_rag=True,
                enable_cot=False,
            )

            assert response == "The weather in London is 15°C and rainy."
            assert cache.hits == 1
            assert cache.misses == 0

    @pytest.mark.asyncio
    async def test_cache_get_miss(self, mock_redis):
        """Test cache get with no entry (cache miss)."""
        # Mock Redis to return None (cache miss)
        mock_redis.get.return_value = None

        with patch("backend.src.cache.l2_redis_cache.redis.from_url", return_value=mock_redis):
            cache = RedisQueryCache()
            await cache.connect()

            # Get non-existent entry
            response = await cache.get(
                query="What's the weather in London?",
                user_id="user123",
                enable_rag=True,
                enable_cot=False,
            )

            assert response is None
            assert cache.hits == 0
            assert cache.misses == 1

    @pytest.mark.asyncio
    async def test_cache_get_without_connection(self):
        """Test cache get gracefully handles no connection."""
        cache = RedisQueryCache()
        # Don't connect

        # Get should return None without error
        response = await cache.get(
            query="What's the weather?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert response is None
        assert cache.hits == 0
        assert cache.misses == 0  # Not counted as miss (no connection)

    @pytest.mark.asyncio
    async def test_cache_set_without_connection(self):
        """Test cache set gracefully handles no connection."""
        cache = RedisQueryCache()
        # Don't connect

        # Set should not raise error
        await cache.set(
            query="What's the weather?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
            response="15°C",
        )

        # No error raised (graceful degradation)

    @pytest.mark.asyncio
    async def test_cache_get_error_handling(self, mock_redis):
        """Test graceful error handling during get operation."""
        # Mock Redis to raise error
        mock_redis.get.side_effect = Exception("Redis error")

        with patch("backend.src.cache.l2_redis_cache.redis.from_url", return_value=mock_redis):
            cache = RedisQueryCache()
            await cache.connect()

            # Get should return None on error
            response = await cache.get(
                query="What's the weather?",
                user_id="user123",
                enable_rag=True,
                enable_cot=False,
            )

            assert response is None
            assert cache.errors == 1

    @pytest.mark.asyncio
    async def test_cache_set_error_handling(self, mock_redis):
        """Test graceful error handling during set operation."""
        # Mock Redis to raise error
        mock_redis.set.side_effect = Exception("Redis error")

        with patch("backend.src.cache.l2_redis_cache.redis.from_url", return_value=mock_redis):
            cache = RedisQueryCache()
            await cache.connect()

            # Set should not raise error
            await cache.set(
                query="What's the weather?",
                user_id="user123",
                enable_rag=True,
                enable_cot=False,
                response="15°C",
            )

            assert cache.errors == 1

    @pytest.mark.asyncio
    async def test_cache_clear(self, mock_redis):
        """Test cache clear operation with prefix matching."""
        # Mock scan to return matching keys
        mock_redis.scan.side_effect = [
            (100, ["weather:cache:key1", "weather:cache:key2"]),  # First scan
            (0, ["weather:cache:key3"]),  # Second scan (cursor 0 = done)
        ]

        with patch("backend.src.cache.l2_redis_cache.redis.from_url", return_value=mock_redis):
            cache = RedisQueryCache(key_prefix="weather:cache:")
            await cache.connect()

            await cache.clear()

            # Verify delete was called for all keys
            assert mock_redis.delete.call_count == 2
            # First call with key1, key2
            mock_redis.delete.assert_any_call("weather:cache:key1", "weather:cache:key2")
            # Second call with key3
            mock_redis.delete.assert_any_call("weather:cache:key3")

    @pytest.mark.asyncio
    async def test_cache_stats(self, mock_redis):
        """Test cache statistics calculation."""
        with patch("backend.src.cache.l2_redis_cache.redis.from_url", return_value=mock_redis):
            cache = RedisQueryCache(ttl_seconds=1800)
            await cache.connect()

            # Initial stats
            stats = await cache.get_stats()
            assert stats["hits"] == 0
            assert stats["misses"] == 0
            assert stats["errors"] == 0
            assert stats["hit_rate"] == 0.0
            assert stats["ttl_seconds"] == 1800

            # Add some hits and misses
            mock_redis.get.side_effect = [
                "Response 1",  # Hit
                "Response 2",  # Hit
                None,  # Miss
            ]

            await cache.get("Q1", "user123", True, False)  # Hit
            await cache.get("Q2", "user123", True, False)  # Hit
            await cache.get("Q3", "user123", True, False)  # Miss

            # Check stats
            stats = await cache.get_stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 1
            assert stats["hit_rate"] == 2 / 3  # 66.7%

    @pytest.mark.asyncio
    async def test_cache_stats_with_redis_memory_info(self, mock_redis):
        """Test cache statistics include Redis memory info."""
        with patch("backend.src.cache.l2_redis_cache.redis.from_url", return_value=mock_redis):
            cache = RedisQueryCache()
            await cache.connect()

            stats = await cache.get_stats()

            # Verify Redis memory stats included
            assert "redis_memory_mb" in stats
            assert "redis_peak_memory_mb" in stats
            assert stats["redis_memory_mb"] == 1.0  # 1MB
            assert stats["redis_peak_memory_mb"] == 2.0  # 2MB
