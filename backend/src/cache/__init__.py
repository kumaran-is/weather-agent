"""Caching module for Weather AI Agent.

L5a: Multi-level caching architecture for 50-90% cost reduction.

Cache Layers:
- L1: In-process LRU cache (fast, server-local, 5min TTL)
- L2: Redis distributed cache (shared, 30min TTL)
- L3: Anthropic prompt cache (API-level, automatic)

Usage:
    from backend.src.cache import QueryCache, RedisQueryCache, AnthropicCacheMetrics
    from backend.src.cache import prepare_cached_system_prompt, prepare_cached_tools

    # L1 cache
    l1_cache = QueryCache(max_size=1000)
    response = l1_cache.get(query, user_id, enable_rag, enable_cot)

    # L2 cache
    l2_cache = RedisQueryCache()
    await l2_cache.connect()
    response = await l2_cache.get(query, user_id, enable_rag, enable_cot)

    # L3 cache (Anthropic prompt caching)
    system = prepare_cached_system_prompt(base_prompt, memory_context)
    tools = prepare_cached_tools(tool_definitions)
    response = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=system,
        tools=tools,
        messages=messages,
    )
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

__all__ = [
    "QueryCache",
    "RedisQueryCache",
    "AnthropicCacheMetrics",
    "add_cache_control",
    "prepare_cached_system_prompt",
    "prepare_cached_tools",
    "prepare_cached_messages",
    "extract_cache_stats",
]
