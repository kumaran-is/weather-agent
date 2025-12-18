"""Caching module for Weather AI Agent.

Level 9: Multi-level caching architecture with semantic matching.

Cache Layers:
    Query Response Cache:
    - Q1 (L1): In-process LRU cache (exact match, user-specific, 5min TTL)
    - Q2 (L2): Redis distributed cache (exact match, shared, 30min TTL)
    - Q3: Qdrant semantic cache (similarity match, shared, 30min TTL)

    Tool Result Cache (Level 9b):
    - T1: Redis exact match (fast, per-tool TTL)
    - T2: Qdrant semantic match (similar tool calls)

    LLM Response Cache (Level 9c):
    - R1: Redis exact match (identical prompts)
    - R2: Qdrant semantic match (similar prompts)

    Provider Cache:
    - L3: Anthropic prompt cache (API-level, automatic, 90% cost reduction)

Usage:
    from backend.src.cache import (
        # L1/L2 exact match caches
        QueryCache, RedisQueryCache,
        # L3 Anthropic prompt cache
        AnthropicCacheMetrics, prepare_cached_system_prompt,
        # Cache orchestrator
        CacheOrchestrator, CacheResult, CacheStats,
        # Level 9a: Query normalization and semantic cache
        QueryNormalizer, SemanticQueryCache,
        # Level 9b: Tool result cache
        ToolResultCache, ToolCacheConfig, cached_tool,
        # Level 9c: LLM response cache
        LLMResponseCache, LLMCacheConfig,
    )

    # L5a/L9a: Use CacheOrchestrator for unified Q1+Q2+Q3 cache management
    orchestrator = CacheOrchestrator(l1_cache, l2_cache, q3_cache)
    cached = await orchestrator.get(query, user_id, enable_rag, enable_cot)
    if cached:
        return cached.response  # Q1, Q2, or Q3 hit

    # Level 9b: Tool caching with decorator
    @cached_tool("get_forecast")
    async def get_forecast(location: str) -> dict:
        return await mcp.call("get_forecast", {"location": location})
"""

from backend.src.cache.l1_memory_cache import QueryCache

# Try to import L2 Redis cache (optional - graceful degradation if redis not installed)
try:
    from backend.src.cache.l2_redis_cache import RedisQueryCache
except ImportError:
    RedisQueryCache = None  # type: ignore

# Level 9a: Common cache infrastructure
from backend.src.cache.common import (
    CacheKeyGenerator,
    CachePromoter,
    EmbeddingCache,
    SemanticMatcher,
    TwoTierCache,
    TwoTierCacheResult,
)
from backend.src.cache.l3_anthropic_cache import (
    AnthropicCacheMetrics,
    add_cache_control,
    extract_cache_stats,
    prepare_cached_messages,
    prepare_cached_system_prompt,
    prepare_cached_tools,
)

# Level 9c: LLM response cache
from backend.src.cache.llm_cache import (
    LLMCacheConfig,
    LLMResponseCache,
)

# L5a: Cache orchestrator for unified cache management
from backend.src.cache.orchestrator import (
    CacheOrchestrator,
    CacheResult,
    CacheStats,
)

# Level 9a: Query normalization and semantic cache
from backend.src.cache.query_normalizer import QueryNormalizer, get_query_normalizer
from backend.src.cache.semantic_query_cache import SemanticQueryCache

# Level 9b: Tool result cache
from backend.src.cache.tool_cache import (
    ToolCacheConfig,
    ToolCacheSettings,
    ToolResultCache,
    cached_tool,
    get_tool_cache,
)

__all__ = [
    # L1/L2 exact match caches
    "QueryCache",
    "RedisQueryCache",
    # L3 Anthropic cache utilities
    "AnthropicCacheMetrics",
    "add_cache_control",
    "prepare_cached_system_prompt",
    "prepare_cached_tools",
    "prepare_cached_messages",
    "extract_cache_stats",
    # L5a: Cache orchestrator
    "CacheOrchestrator",
    "CacheResult",
    "CacheStats",
    # Level 9a: Common infrastructure
    "TwoTierCache",
    "TwoTierCacheResult",
    "SemanticMatcher",
    "CacheKeyGenerator",
    "CachePromoter",
    "EmbeddingCache",
    # Level 9a: Query normalization and semantic cache
    "QueryNormalizer",
    "get_query_normalizer",
    "SemanticQueryCache",
    # Level 9b: Tool result cache
    "ToolResultCache",
    "ToolCacheConfig",
    "ToolCacheSettings",
    "cached_tool",
    "get_tool_cache",
    # Level 9c: LLM response cache
    "LLMResponseCache",
    "LLMCacheConfig",
]
