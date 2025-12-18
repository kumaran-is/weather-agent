"""Unit tests for L5a: Cache Orchestrator.

Tests validate the cache orchestrator for:
- L1 → L2 cache lookup flow
- L1 backfill from L2 hits
- Cache set operations (L1 + L2)
- Cache invalidation
- Statistics tracking
- Graceful degradation when cache layers unavailable
"""

from unittest.mock import AsyncMock, Mock

import pytest

from backend.src.cache.l1_memory_cache import QueryCache
from backend.src.cache.orchestrator import (
    CacheOrchestrator,
    CacheResult,
    CacheStats,
)


class TestCacheOrchestrator:
    """Test L5a: Cache orchestrator functionality."""

    @pytest.fixture
    def l1_cache(self):
        """Create L1 cache for testing."""
        return QueryCache(max_size=100, ttl_seconds=300)

    @pytest.fixture
    def mock_l2_cache(self):
        """Create mock L2 Redis cache."""
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock()
        mock.client = AsyncMock()
        mock._generate_cache_key = Mock(return_value="weather:cache:test_key")
        # Mock L2 stats attributes
        mock.hits = 0
        mock.misses = 0
        mock.errors = 0
        return mock

    @pytest.fixture
    def orchestrator(self, l1_cache, mock_l2_cache):
        """Create cache orchestrator with both layers."""
        return CacheOrchestrator(
            l1_cache=l1_cache,
            l2_cache=mock_l2_cache,
            enable_tracing=False,  # Disable tracing for tests
        )

    @pytest.mark.asyncio
    async def test_get_l1_hit(self, orchestrator, l1_cache):
        """Test cache get with L1 hit (fastest path)."""
        # Pre-populate L1 cache
        l1_cache.set(
            query="What's the weather in Miami?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
            response="The weather in Miami is 85°F and sunny.",
        )

        # Get from orchestrator (should hit L1)
        result = await orchestrator.get(
            query="What's the weather in Miami?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert result is not None
        assert result.cache_tier == "L1"
        assert result.response == "The weather in Miami is 85°F and sunny."
        assert result.latency_ms < 5  # L1 should be < 5ms
        assert orchestrator.stats.l1_hits == 1

    @pytest.mark.asyncio
    async def test_get_l2_hit_with_backfill(self, orchestrator, l1_cache, mock_l2_cache):
        """Test cache get with L2 hit and L1 backfill."""
        # Configure L2 to return a hit
        mock_l2_cache.get.return_value = "The weather in Tampa is 82°F."

        # Get from orchestrator (L1 miss → L2 hit)
        result = await orchestrator.get(
            query="What's the weather in Tampa?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert result is not None
        assert result.cache_tier == "L2"
        assert result.response == "The weather in Tampa is 82°F."
        assert orchestrator.stats.l2_hits == 1
        assert orchestrator.stats.l1_backfills == 1

        # Verify L1 was backfilled (next get should hit L1)
        result2 = await orchestrator.get(
            query="What's the weather in Tampa?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert result2 is not None
        assert result2.cache_tier == "L1"  # Now L1 hit after backfill
        assert orchestrator.stats.l1_hits == 1

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, orchestrator, mock_l2_cache):
        """Test cache get with complete miss (L1 + L2)."""
        # Configure L2 to return miss
        mock_l2_cache.get.return_value = None

        # Get from orchestrator (should miss)
        result = await orchestrator.get(
            query="What's the weather in Orlando?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert result is None
        assert orchestrator.stats.cache_misses == 1

    @pytest.mark.asyncio
    async def test_set_writes_to_both_layers(self, orchestrator, l1_cache, mock_l2_cache):
        """Test that set writes to both L1 and L2."""
        await orchestrator.set(
            query="What's the weather in Key West?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
            response="Key West: 88°F and humid.",
        )

        # Verify L1 was written
        l1_response = l1_cache.get(
            query="What's the weather in Key West?",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        assert l1_response == "Key West: 88°F and humid."

        # Verify L2 set was called
        mock_l2_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_both_layers(self, orchestrator, l1_cache, mock_l2_cache):
        """Test that invalidate removes from both L1 and L2."""
        # Pre-populate L1
        l1_cache.set(
            query="Old data",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
            response="Outdated response",
        )

        # Configure L2 delete to return success
        mock_l2_cache.client.delete.return_value = 1

        # Invalidate
        result = await orchestrator.invalidate(
            query="Old data",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert result["l1"] is True
        assert result["l2"] is True

        # Verify L1 entry is gone
        l1_response = l1_cache.get(
            query="Old data",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )
        assert l1_response is None

    @pytest.mark.asyncio
    async def test_graceful_degradation_l2_unavailable(self, l1_cache):
        """Test orchestrator works with only L1 (L2 unavailable)."""
        orchestrator = CacheOrchestrator(
            l1_cache=l1_cache,
            l2_cache=None,  # L2 not available
            enable_tracing=False,
        )

        # Set should work (L1 only)
        await orchestrator.set(
            query="Test query",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
            response="Test response",
        )

        # Get should work (L1 only)
        result = await orchestrator.get(
            query="Test query",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert result is not None
        assert result.cache_tier == "L1"
        assert result.response == "Test response"

    @pytest.mark.asyncio
    async def test_graceful_degradation_l1_unavailable(self, mock_l2_cache):
        """Test orchestrator works with only L2 (L1 unavailable)."""
        orchestrator = CacheOrchestrator(
            l1_cache=None,  # L1 not available
            l2_cache=mock_l2_cache,
            enable_tracing=False,
        )

        # Configure L2 to return hit
        mock_l2_cache.get.return_value = "L2 response"

        # Get should work (L2 only)
        result = await orchestrator.get(
            query="Test query",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert result is not None
        assert result.cache_tier == "L2"
        assert result.response == "L2 response"
        # No backfill attempted (L1 is None)

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_cache(self):
        """Test orchestrator returns None when no cache available."""
        orchestrator = CacheOrchestrator(
            l1_cache=None,
            l2_cache=None,
            enable_tracing=False,
        )

        result = await orchestrator.get(
            query="Test query",
            user_id="user123",
            enable_rag=True,
            enable_cot=False,
        )

        assert result is None

    def test_stats_calculation(self, l1_cache, mock_l2_cache):
        """Test statistics calculation and reporting."""
        orchestrator = CacheOrchestrator(
            l1_cache=l1_cache,
            l2_cache=mock_l2_cache,
            enable_tracing=False,
        )

        # Manually set stats for testing
        orchestrator.stats.total_requests = 100
        orchestrator.stats.l1_hits = 25
        orchestrator.stats.l2_hits = 35
        orchestrator.stats.l1_backfills = 35
        orchestrator.stats.cache_misses = 40

        # Get stats
        stats = orchestrator.get_stats()

        # Verify orchestrator stats
        assert stats["orchestrator"]["l1_hit_rate"] == 0.25  # 25/100
        assert stats["orchestrator"]["overall_hit_rate"] == 0.60  # 60/100

    def test_cache_stats_dataclass(self):
        """Test CacheStats dataclass calculations."""
        stats = CacheStats()

        # Initial state
        assert stats.total_requests == 0
        assert stats.l1_hit_rate == 0.0
        assert stats.overall_hit_rate == 0.0

        # Add some data
        stats.total_requests = 100
        stats.l1_hits = 20
        stats.l2_hits = 30
        stats.cache_misses = 50

        # Verify calculations
        assert stats.l1_hit_rate == 0.20
        assert stats.l2_hit_rate == 0.375  # 30 / (100 - 20) = 30/80
        assert stats.overall_hit_rate == 0.50  # 50/100

    def test_cache_result_dataclass(self):
        """Test CacheResult dataclass."""
        result = CacheResult(
            response="Test response",
            cache_tier="L1",
            latency_ms=0.5,
            cache_key="test_key",
        )

        assert result.response == "Test response"
        assert result.cache_tier == "L1"
        assert result.latency_ms == 0.5
        assert result.cache_key == "test_key"

    def test_stats_reset(self, orchestrator):
        """Test statistics reset."""
        # Add some stats
        orchestrator.stats.total_requests = 100
        orchestrator.stats.l1_hits = 25

        # Reset
        orchestrator.reset_stats()

        # Verify reset
        assert orchestrator.stats.total_requests == 0
        assert orchestrator.stats.l1_hits == 0
