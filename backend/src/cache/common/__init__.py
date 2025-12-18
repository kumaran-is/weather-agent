"""Common cache infrastructure for two-tier semantic caching.

Level 9a: Shared components used by Tool Cache and LLM Response Cache.

Architecture:
- TwoTierCache: Generic base class for Tier 1 (exact) + Tier 2 (semantic)
- SemanticMatcher: Qdrant vector similarity search
- CacheKeyGenerator: SHA-256 hash generation
- CachePromoter: Tier 2 → Tier 1 promotion logic
- EmbeddingCache: Cache embeddings to avoid re-embedding

Usage:
    from backend.src.cache.common import (
        TwoTierCache,
        TwoTierCacheResult,
        SemanticMatcher,
        CacheKeyGenerator,
        CachePromoter,
        EmbeddingCache,
    )

    class MyCache(TwoTierCache[str]):
        def _generate_cache_key(self, key_input: str, **kwargs) -> str:
            return CacheKeyGenerator.generate(key_input)

        def _generate_embedding_input(self, key_input: str, **kwargs) -> str:
            return key_input

        def _serialize(self, value: str) -> str:
            return value

        def _deserialize(self, data: str) -> str:
            return data
"""

from backend.src.cache.common.cache_key_generator import CacheKeyGenerator
from backend.src.cache.common.cache_promoter import CachePromoter
from backend.src.cache.common.embedding_cache import EmbeddingCache
from backend.src.cache.common.semantic_matcher import SemanticMatcher
from backend.src.cache.common.two_tier_cache import (
    TwoTierCache,
    TwoTierCacheResult,
)

__all__ = [
    "TwoTierCache",
    "TwoTierCacheResult",
    "SemanticMatcher",
    "CacheKeyGenerator",
    "CachePromoter",
    "EmbeddingCache",
]
