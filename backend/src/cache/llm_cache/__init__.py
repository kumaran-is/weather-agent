"""LLM Response Cache module for caching LLM API responses.

Level 9c: Two-tier caching for LLM responses (R1 Redis exact → R2 Qdrant semantic).

Benefits:
    - Reduce LLM API costs by 30%+
    - Semantic matching for similar prompts
    - Exclusion patterns for agent tool-calling
    - Configurable per-model settings

Architecture:
    R1 (Redis): Exact match on prompt hash (fast, <5ms)
    R2 (Qdrant): Semantic match for similar prompts (<50ms)

Limitations:
    - Agent tool-calling prompts have varying context (low hit rate)
    - Best for: Simple Q&A, summarization, formatting tasks
    - Less effective for: Complex multi-turn conversations

Usage:
    from backend.src.cache.llm_cache import (
        LLMResponseCache,
        LLMCacheConfig,
    )

    config = LLMCacheConfig(ttl=3600, threshold=0.90)
    cache = LLMResponseCache(redis_client, semantic_matcher, config=config)

    # Lookup
    result = await cache.get_response("What is the capital of France?")
    if result.hit:
        return result.value  # Cached response

    # Store
    await cache.set_response("What is the capital of France?", "Paris is the capital...")
"""

from backend.src.cache.llm_cache.llm_cache_config import LLMCacheConfig
from backend.src.cache.llm_cache.llm_response_cache import LLMResponseCache

__all__ = [
    "LLMResponseCache",
    "LLMCacheConfig",
]
