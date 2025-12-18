"""Unit tests for TwoTierCache base class (Level 9a).

Tests validate the two-tier cache pattern:
- Tier 1 (Redis): Exact match via hash key (fast, <5ms)
- Tier 2 (Qdrant): Semantic match via embeddings (<50ms)
- Cache promotion: Tier 2 hit → Tier 1 backfill
- Cache statistics tracking

Expected Performance:
- Tier 1 hit: <5ms
- Tier 2 hit: <50ms (with promotion to T1)
- Cache promotion reduces future latency
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.src.cache.common.two_tier_cache import TwoTierCache, TwoTierCacheResult


class ConcreteTwoTierCache(TwoTierCache[str]):
    """Concrete implementation for testing abstract base class."""

    def _generate_cache_key(self, key_input: str, **kwargs) -> str:
        """Simple key generation for testing."""
        return f"test:{hash(key_input) % 10000:04d}"

    def _generate_embedding_input(self, key_input: str, **kwargs) -> str:
        """Pass through for testing."""
        return key_input

    def _serialize(self, value: str) -> str:
        """Identity serialization for strings."""
        return value

    def _deserialize(self, data: str) -> str:
        """Identity deserialization for strings."""
        return data


class TestTwoTierCacheResult:
    """Test TwoTierCacheResult model."""

    def test_hit_result(self):
        """Test cache hit result construction."""
        result = TwoTierCacheResult(
            hit=True,
            tier="tier1",
            value="test_value",
            similarity_score=1.0,
            lookup_ms=2.5,
        )
        assert result.hit is True
        assert result.tier == "tier1"
        assert result.value == "test_value"
        assert result.similarity_score == 1.0
        assert result.lookup_ms == 2.5

    def test_miss_result(self):
        """Test cache miss result construction."""
        result = TwoTierCacheResult(
            hit=False,
            tier=None,
            value=None,
            similarity_score=None,
            lookup_ms=5.0,
        )
        assert result.hit is False
        assert result.tier is None
        assert result.value is None
        assert result.similarity_score is None
        assert result.lookup_ms == 5.0

    def test_tier2_result_with_similarity(self):
        """Test tier2 result includes similarity score."""
        result = TwoTierCacheResult(
            hit=True,
            tier="tier2",
            value="semantic_match",
            similarity_score=0.92,
            lookup_ms=45.0,
        )
        assert result.tier == "tier2"
        assert result.similarity_score == 0.92


class TestTwoTierCacheInitialization:
    """Test TwoTierCache initialization."""

    def test_init_with_all_components(self):
        """Test initialization with all components."""
        redis_mock = AsyncMock()
        semantic_mock = AsyncMock()
        promoter_mock = AsyncMock()

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=promoter_mock,
            collection_name="test_collection",
            default_ttl=600,
            similarity_threshold=0.90,
        )

        assert cache.redis is redis_mock
        assert cache.semantic is semantic_mock
        assert cache.promoter is promoter_mock
        assert cache.collection == "test_collection"
        assert cache.ttl == 600
        assert cache.threshold == 0.90

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        cache = ConcreteTwoTierCache(
            redis_client=None,
            semantic_matcher=None,
            cache_promoter=None,
            collection_name="default_collection",
        )

        assert cache.ttl == 1800  # Default 30 minutes
        assert cache.threshold == 0.85  # Default threshold

    def test_init_stats_zeroed(self):
        """Test that stats are zeroed on initialization."""
        cache = ConcreteTwoTierCache(
            redis_client=None,
            semantic_matcher=None,
            cache_promoter=None,
            collection_name="test",
        )

        assert cache.tier1_hits == 0
        assert cache.tier2_hits == 0
        assert cache.misses == 0


class TestTwoTierCacheTier1:
    """Test Tier 1 (Redis exact match) operations."""

    @pytest.mark.asyncio
    async def test_tier1_hit(self):
        """Test Tier 1 cache hit."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = "cached_value"

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=None,
            cache_promoter=None,
            collection_name="test",
        )

        result = await cache.get("test_query")

        assert result.hit is True
        assert result.tier == "tier1"
        assert result.value == "cached_value"
        assert result.similarity_score == 1.0
        assert cache.tier1_hits == 1

    @pytest.mark.asyncio
    async def test_tier1_miss_to_tier2(self):
        """Test Tier 1 miss falls through to Tier 2."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None  # Tier 1 miss

        # Create mock for semantic matcher with scored point
        semantic_mock = AsyncMock()
        scored_point = MagicMock()
        scored_point.payload = {"value": "semantic_value"}
        scored_point.score = 0.92
        semantic_mock.search.return_value = [scored_point]

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
        )

        result = await cache.get("test_query")

        assert result.hit is True
        assert result.tier == "tier2"
        assert result.value == "semantic_value"
        assert result.similarity_score == 0.92
        assert cache.tier2_hits == 1

    @pytest.mark.asyncio
    async def test_tier1_error_fallback_to_tier2(self):
        """Test Tier 1 error gracefully falls through to Tier 2."""
        redis_mock = AsyncMock()
        redis_mock.get.side_effect = Exception("Redis connection error")

        semantic_mock = AsyncMock()
        scored_point = MagicMock()
        scored_point.payload = {"value": "fallback_value"}
        scored_point.score = 0.88
        semantic_mock.search.return_value = [scored_point]

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
        )

        result = await cache.get("test_query")

        # Should still get Tier 2 result despite Tier 1 error
        assert result.hit is True
        assert result.tier == "tier2"


class TestTwoTierCacheTier2:
    """Test Tier 2 (Qdrant semantic match) operations."""

    @pytest.mark.asyncio
    async def test_tier2_semantic_hit(self):
        """Test Tier 2 semantic match hit."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None  # Tier 1 miss

        semantic_mock = AsyncMock()
        scored_point = MagicMock()
        scored_point.payload = {"value": "semantic_result"}
        scored_point.score = 0.95
        semantic_mock.search.return_value = [scored_point]

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
        )

        result = await cache.get("test_query")

        assert result.hit is True
        assert result.tier == "tier2"
        assert result.value == "semantic_result"
        assert result.similarity_score == 0.95
        semantic_mock.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_tier2_miss(self):
        """Test complete cache miss (both tiers)."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None

        semantic_mock = AsyncMock()
        semantic_mock.search.return_value = []  # No results

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
        )

        result = await cache.get("test_query")

        assert result.hit is False
        assert result.tier is None
        assert result.value is None
        assert cache.misses == 1

    @pytest.mark.asyncio
    async def test_tier2_error_results_in_miss(self):
        """Test Tier 2 error results in cache miss."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None

        semantic_mock = AsyncMock()
        semantic_mock.search.side_effect = Exception("Qdrant error")

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
        )

        result = await cache.get("test_query")

        assert result.hit is False
        assert cache.misses == 1


class TestTwoTierCachePromotion:
    """Test cache promotion (Tier 2 → Tier 1 backfill)."""

    @pytest.mark.asyncio
    async def test_tier2_hit_triggers_promotion(self):
        """Test that Tier 2 hit triggers promotion to Tier 1."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None

        semantic_mock = AsyncMock()
        scored_point = MagicMock()
        scored_point.payload = {"value": "promoted_value"}
        scored_point.score = 0.91
        semantic_mock.search.return_value = [scored_point]

        promoter_mock = AsyncMock()

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=promoter_mock,
            collection_name="test",
            default_ttl=600,
        )

        await cache.get("test_query")

        # Verify promotion was called
        promoter_mock.promote.assert_called_once()
        call_args = promoter_mock.promote.call_args
        assert call_args.kwargs["redis"] is redis_mock
        assert call_args.kwargs["value"] == "promoted_value"
        assert call_args.kwargs["ttl"] == 600

    @pytest.mark.asyncio
    async def test_no_promotion_without_promoter(self):
        """Test no promotion attempt when promoter is None."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None

        semantic_mock = AsyncMock()
        scored_point = MagicMock()
        scored_point.payload = {"value": "value"}
        scored_point.score = 0.90
        semantic_mock.search.return_value = [scored_point]

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,  # No promoter
            collection_name="test",
        )

        # Should not raise exception
        result = await cache.get("test_query")
        assert result.hit is True


class TestTwoTierCacheSet:
    """Test cache set operations."""

    @pytest.mark.asyncio
    async def test_set_writes_to_both_tiers(self):
        """Test set writes to both Tier 1 and Tier 2."""
        redis_mock = AsyncMock()
        semantic_mock = AsyncMock()

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
            default_ttl=300,
        )

        await cache.set("test_key", "test_value")

        # Verify Redis write
        redis_mock.setex.assert_called_once()

        # Verify Qdrant write
        semantic_mock.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_tier1_only(self):
        """Test set with only Tier 1 available."""
        redis_mock = AsyncMock()

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=None,  # No Tier 2
            cache_promoter=None,
            collection_name="test",
        )

        await cache.set("test_key", "test_value")

        redis_mock.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_tier2_only(self):
        """Test set with only Tier 2 available."""
        semantic_mock = AsyncMock()

        cache = ConcreteTwoTierCache(
            redis_client=None,  # No Tier 1
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
        )

        await cache.set("test_key", "test_value")

        semantic_mock.upsert.assert_called_once()


class TestTwoTierCacheInvalidate:
    """Test cache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_both_tiers(self):
        """Test invalidation from both tiers."""
        redis_mock = AsyncMock()
        redis_mock.delete.return_value = 1  # 1 key deleted

        semantic_mock = AsyncMock()
        semantic_mock.delete_by_key.return_value = True

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
        )

        result = await cache.invalidate("test_key")

        assert result["tier1"] is True
        assert result["tier2"] is True
        redis_mock.delete.assert_called_once()
        semantic_mock.delete_by_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_not_found(self):
        """Test invalidation when key not found."""
        redis_mock = AsyncMock()
        redis_mock.delete.return_value = 0  # No keys deleted

        semantic_mock = AsyncMock()
        semantic_mock.delete_by_key.return_value = False

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
        )

        result = await cache.invalidate("nonexistent_key")

        assert result["tier1"] is False
        assert result["tier2"] is False


class TestTwoTierCacheStats:
    """Test cache statistics."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics are tracked correctly."""
        redis_mock = AsyncMock()
        redis_mock.get.side_effect = [
            "cached",  # Hit 1
            "cached",  # Hit 2
            None,  # Miss
        ]

        semantic_mock = AsyncMock()
        semantic_mock.search.return_value = []  # No semantic results

        cache = ConcreteTwoTierCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
            cache_promoter=None,
            collection_name="test",
        )

        await cache.get("query1")  # Tier 1 hit
        await cache.get("query2")  # Tier 1 hit
        await cache.get("query3")  # Miss

        stats = cache.get_stats()

        assert stats["tier1_hits"] == 2
        assert stats["misses"] == 1
        assert stats["total_requests"] == 3
        assert stats["tier1_hit_rate"] == pytest.approx(0.6667, rel=0.01)
        assert stats["overall_hit_rate"] == pytest.approx(0.6667, rel=0.01)

    def test_reset_stats(self):
        """Test stats reset."""
        cache = ConcreteTwoTierCache(
            redis_client=None,
            semantic_matcher=None,
            cache_promoter=None,
            collection_name="test",
        )

        cache.tier1_hits = 10
        cache.tier2_hits = 5
        cache.misses = 3

        cache.reset_stats()

        assert cache.tier1_hits == 0
        assert cache.tier2_hits == 0
        assert cache.misses == 0


class TestTwoTierCacheNoBackends:
    """Test behavior with no backends configured."""

    @pytest.mark.asyncio
    async def test_get_with_no_backends(self):
        """Test get returns miss when no backends configured."""
        cache = ConcreteTwoTierCache(
            redis_client=None,
            semantic_matcher=None,
            cache_promoter=None,
            collection_name="test",
        )

        result = await cache.get("any_query")

        assert result.hit is False
        assert result.tier is None
        assert cache.misses == 1

    @pytest.mark.asyncio
    async def test_set_with_no_backends(self):
        """Test set is no-op when no backends configured."""
        cache = ConcreteTwoTierCache(
            redis_client=None,
            semantic_matcher=None,
            cache_promoter=None,
            collection_name="test",
        )

        # Should not raise exception
        await cache.set("key", "value")
