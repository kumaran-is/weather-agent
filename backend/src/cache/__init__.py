"""Caching module for Weather AI Agent.

L5a: Multi-level caching architecture for 50-90% cost reduction.

Cache Layers:
- L1: In-process LRU cache (fast, server-local, 5min TTL)
- L2: Redis distributed cache (shared, 30min TTL)
- L3: Anthropic prompt cache (API-level, automatic)

Usage:
    from backend.src.cache import (
        QueryCache, RedisQueryCache, AnthropicCacheMetrics,
        CacheOrchestrator, CacheResult, CacheStats,
        prepare_cached_system_prompt, prepare_cached_tools,
    )

    # L5a: Use CacheOrchestrator for unified L1+L2 cache management
    orchestrator = CacheOrchestrator(l1_cache, l2_cache)
    cached = await orchestrator.get(query, user_id, enable_rag, enable_cot)
    if cached:
        return cached.response  # L1 or L2 hit

    # Cache miss - execute agent with L3 Anthropic caching
    response = await execute_agent(...)

    # Cache the response at L1+L2
    await orchestrator.set(query, user_id, enable_rag, enable_cot, response)

    # L3 cache (Anthropic prompt caching) - automatic with ChatAnthropic
    # Just enable via create_tuned_llm(enable_cache=True)
"""

from backend.src.cache.l1_memory_cache import QueryCache

# Try to import L2 Redis cache (optional - graceful degradation if redis not installed)
try:
    from backend.src.cache.l2_redis_cache import RedisQueryCache
except ImportError:
    RedisQueryCache = None  # type: ignore

from backend.src.cache.l3_anthropic_cache import (
    AnthropicCacheMetrics,
    add_cache_control,
    extract_cache_stats,
    prepare_cached_messages,
    prepare_cached_system_prompt,
    prepare_cached_tools,
)

# L5a: Cache orchestrator for unified cache management
from backend.src.cache.orchestrator import (
    CacheOrchestrator,
    CacheResult,
    CacheStats,
)

__all__ = [
    # L1/L2 caches
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
]
