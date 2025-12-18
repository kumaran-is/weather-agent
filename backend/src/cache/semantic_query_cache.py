"""Q3 Semantic Query Cache for weather queries.

Level 9a: Third layer of query caching using semantic similarity.

Cache Hierarchy:
    Q1 (L1): In-memory LRU - Exact hash match (user-specific)
    Q2 (L2): Redis distributed - Exact hash match (cross-user sharing)
    Q3: Qdrant semantic - Similarity match (semantic understanding)

Flow:
    1. Check Q1 (exact, user-specific) → HIT: Return immediately
    2. Check Q2 (exact, shared) → HIT: Return + Backfill Q1
    3. Check Q3 (semantic) → HIT: Return + Backfill Q1/Q2
    4. MISS: Execute agent, cache at all tiers

Benefits:
    - "SF weather" matches "San Francisco weather" (semantic)
    - "What's the temp?" matches "current temperature" (semantic)
    - 25-40% additional hit rate on similar queries

Performance:
    - Q3 latency: <50ms (Qdrant search + embedding lookup)
    - Similarity threshold: 0.85 (high precision)
    - TTL: 30 minutes (weather data freshness)

Usage:
    from backend.src.cache.semantic_query_cache import SemanticQueryCache

    cache = SemanticQueryCache(
        redis_client=redis,
        semantic_matcher=matcher,
        cache_promoter=promoter,
    )

    # Lookup
    result = await cache.get_response("weather in San Francisco")
    if result.hit:
        return result.value  # Cached response

    # Store
    await cache.set_response("weather in San Francisco", response)
"""

import json
import logging
from typing import Any

import redis.asyncio as redis

from backend.src.cache.common.cache_key_generator import CacheKeyGenerator
from backend.src.cache.common.cache_promoter import CachePromoter
from backend.src.cache.common.semantic_matcher import SemanticMatcher
from backend.src.cache.common.two_tier_cache import TwoTierCache, TwoTierCacheResult
from backend.src.cache.query_normalizer import QueryNormalizer, get_query_normalizer

logger = logging.getLogger(__name__)


class SemanticQueryCache(TwoTierCache[dict[str, Any]]):
    """Q3: Semantic query cache using Qdrant.

    Extends TwoTierCache for weather query responses.
    Integrates with QueryNormalizer for improved matching.

    Key Features:
        - Query normalization ("SF" → "san francisco")
        - Semantic similarity matching (threshold: 0.85)
        - Automatic Q1/Q2 backfill on Q3 hits
        - Configurable TTL per query type

    Attributes:
        normalizer: QueryNormalizer instance
        collection: Qdrant collection name
        threshold: Similarity threshold for hits
    """

    COLLECTION_NAME = "semantic_query_cache"

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        semantic_matcher: SemanticMatcher | None = None,
        cache_promoter: CachePromoter | None = None,
        normalizer: QueryNormalizer | None = None,
        default_ttl: int = 1800,
        similarity_threshold: float = 0.85,
    ):
        """Initialize Q3 semantic query cache.

        Args:
            redis_client: Async Redis client for Tier 1 exact match
            semantic_matcher: Qdrant semantic matcher
            cache_promoter: Handles promotion to exact match cache
            normalizer: Query normalizer (uses default if None)
            default_ttl: TTL in seconds (default: 1800 = 30 min)
            similarity_threshold: Minimum similarity score (default: 0.85)
        """
        super().__init__(
            redis_client=redis_client,
            semantic_matcher=semantic_matcher,
            cache_promoter=cache_promoter,
            collection_name=self.COLLECTION_NAME,
            default_ttl=default_ttl,
            similarity_threshold=similarity_threshold,
        )

        self.normalizer = normalizer or get_query_normalizer()

        logger.info(
            f"✅ SemanticQueryCache (Q3) initialized | "
            f"collection={self.COLLECTION_NAME} | "
            f"ttl={default_ttl}s | threshold={similarity_threshold}"
        )

    async def get_response(
        self,
        query: str,
        enable_rag: bool = True,
        enable_cot: bool = True,
    ) -> TwoTierCacheResult[dict[str, Any]]:
        """Get cached response for query.

        Args:
            query: User query text
            enable_rag: RAG enabled flag (affects cache key)
            enable_cot: Chain-of-thought enabled flag

        Returns:
            TwoTierCacheResult with hit status, tier, and cached response
        """
        # Build key input with feature flags
        key_input = self._build_key_input(query, enable_rag, enable_cot)

        return await self.get(key_input, query=query, enable_rag=enable_rag, enable_cot=enable_cot)

    async def set_response(
        self,
        query: str,
        response: dict[str, Any],
        enable_rag: bool = True,
        enable_cot: bool = True,
    ) -> None:
        """Cache query response.

        Args:
            query: User query text
            response: Response to cache (dict with 'text', 'metadata', etc.)
            enable_rag: RAG enabled flag
            enable_cot: Chain-of-thought enabled flag
        """
        key_input = self._build_key_input(query, enable_rag, enable_cot)

        await self.set(key_input, response, query=query, enable_rag=enable_rag, enable_cot=enable_cot)

    def _build_key_input(
        self,
        query: str,
        enable_rag: bool,
        enable_cot: bool,
    ) -> str:
        """Build key input string from query and flags.

        Args:
            query: User query
            enable_rag: RAG flag
            enable_cot: CoT flag

        Returns:
            Key input string for cache operations
        """
        # Normalize query
        normalized = self.normalizer.normalize(query)

        # Combine with flags
        return f"{normalized}:rag={enable_rag}:cot={enable_cot}"

    def _generate_cache_key(self, key_input: str, **kwargs) -> str:
        """Generate exact match cache key.

        Args:
            key_input: Combined query and flags string

        Returns:
            Hash-based cache key with q3: prefix
        """
        return CacheKeyGenerator.generate(key_input, prefix="q3:")

    def _generate_embedding_input(self, key_input: str, **kwargs) -> str:
        """Generate text for semantic embedding.

        Uses just the normalized query (without flags) for embedding,
        as semantic similarity should match query meaning, not flags.

        Args:
            key_input: Combined query and flags string

        Returns:
            Normalized query text for embedding
        """
        # Extract original query from kwargs if available
        query = kwargs.get("query", "")
        if query:
            return self.normalizer.normalize(query)

        # Fallback: Parse key_input (format: "normalized_query:rag=True:cot=True")
        return key_input.split(":rag=")[0]

    def _serialize(self, value: dict[str, Any]) -> str:
        """Serialize response dict to JSON string.

        Args:
            value: Response dictionary

        Returns:
            JSON string
        """
        return json.dumps(value, default=str)

    def _deserialize(self, data: str) -> dict[str, Any]:
        """Deserialize JSON string to response dict.

        Args:
            data: JSON string

        Returns:
            Response dictionary
        """
        return json.loads(data)

    async def check_similarity(
        self,
        query1: str,
        query2: str,
    ) -> float:
        """Check semantic similarity between two queries.

        Useful for debugging and tuning threshold.

        Args:
            query1: First query
            query2: Second query

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not self.semantic:
            return 0.0

        try:
            # Normalize both
            norm1 = self.normalizer.normalize(query1)
            norm2 = self.normalizer.normalize(query2)

            # Get embeddings
            emb1 = await self.semantic._get_embedding(norm1)
            emb2 = await self.semantic._get_embedding(norm2)

            # Calculate cosine similarity
            import numpy as np

            vec1 = np.array(emb1)
            vec2 = np.array(emb2)

            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

            return float(similarity)

        except Exception as e:
            logger.warning(f"⚠️ Similarity check error: {e}")
            return 0.0


# Factory function for creating Q3 cache
async def create_semantic_query_cache(
    redis_url: str = "redis://localhost:6379/0",
    qdrant_url: str = "http://localhost:6333",
    ttl: int = 1800,
    threshold: float = 0.85,
) -> SemanticQueryCache:
    """Create and initialize SemanticQueryCache.

    Factory function that sets up all dependencies.

    Args:
        redis_url: Redis connection URL
        qdrant_url: Qdrant connection URL
        ttl: Cache TTL in seconds
        threshold: Similarity threshold

    Returns:
        Initialized SemanticQueryCache instance
    """
    from qdrant_client import AsyncQdrantClient

    from backend.src.cache.common import CachePromoter, EmbeddingCache, SemanticMatcher
    from backend.src.rag.embeddings import create_embeddings

    # Create Redis client
    redis_client = await redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
    )

    # Create Qdrant client
    qdrant_client = AsyncQdrantClient(url=qdrant_url)

    # Create embeddings
    embeddings = create_embeddings()
    embedding_cache = EmbeddingCache()

    # Create semantic matcher
    semantic_matcher = SemanticMatcher(
        client=qdrant_client,
        embeddings=embeddings,
        embedding_cache=embedding_cache,
    )

    # Ensure collection exists
    await semantic_matcher.ensure_collection(SemanticQueryCache.COLLECTION_NAME)

    # Create promoter
    cache_promoter = CachePromoter()

    # Create and return cache
    return SemanticQueryCache(
        redis_client=redis_client,
        semantic_matcher=semantic_matcher,
        cache_promoter=cache_promoter,
        default_ttl=ttl,
        similarity_threshold=threshold,
    )
