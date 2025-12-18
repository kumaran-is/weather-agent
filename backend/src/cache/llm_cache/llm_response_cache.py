"""LLM Response Cache: R1 (Redis Exact) → R2 (Qdrant Semantic).

Level 9c: Caches LLM responses to reduce API costs.

Architecture:
    R1 (Redis): Exact match on prompt hash (fast, <5ms)
    R2 (Qdrant): Semantic match for similar prompts (<50ms)

Benefits:
    - 30%+ reduction in LLM API costs
    - Semantic matching for similar prompts
    - Per-model threshold configuration
    - Automatic exclusion of tool-calling prompts

Limitations:
    - Agent tool-calling prompts have varying context (low hit rate)
    - Best for: Simple Q&A, summarization, formatting tasks
    - Less effective for: Complex multi-turn conversations

Note:
    This cache is COMPLEMENTARY to Anthropic's prompt caching (L3).
    L3 caches at the API level (prefix matching), while R1/R2 cache
    at the response level (semantic matching).

Usage:
    from backend.src.cache.llm_cache import LLMResponseCache, LLMCacheConfig

    config = LLMCacheConfig(ttl=3600, threshold=0.90)
    cache = LLMResponseCache(redis_client, semantic_matcher, config=config)

    # Lookup
    result = await cache.get_response("What is the capital of France?")
    if result.hit:
        return result.value  # Cached response

    # Store
    await cache.set_response("What is the capital of France?", "Paris is the capital...")
"""

import logging
from typing import Any

import redis.asyncio as redis

from backend.src.cache.common.cache_key_generator import CacheKeyGenerator
from backend.src.cache.common.cache_promoter import CachePromoter
from backend.src.cache.common.semantic_matcher import SemanticMatcher
from backend.src.cache.common.two_tier_cache import TwoTierCache, TwoTierCacheResult
from backend.src.cache.llm_cache.llm_cache_config import LLMCacheConfig, get_llm_cache_config

logger = logging.getLogger(__name__)


class LLMResponseCache(TwoTierCache[str]):
    """Two-tier cache for LLM responses.

    Extends TwoTierCache with LLM-specific logic:
    - Exclusion patterns for tool-calling prompts
    - Per-model similarity thresholds
    - Cost tracking for savings estimation

    Key Features:
        - R1 (Redis): Exact match on prompt hash
        - R2 (Qdrant): Semantic match for similar prompts
        - Automatic exclusion of agent tool-calling
        - Model-aware threshold configuration

    Attributes:
        config: LLMCacheConfig instance
        excluded_count: Number of prompts excluded from caching
        cost_savings: Estimated cost savings from cache hits
    """

    COLLECTION_NAME = "llm_response_cache"

    # Estimated cost per LLM call (used for cost savings tracking)
    COST_PER_CALL = {
        "claude-3-opus": 0.015,
        "claude-3-sonnet": 0.003,
        "claude-3-haiku": 0.00025,
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.0015,
        "default": 0.002,
    }

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        semantic_matcher: SemanticMatcher | None = None,
        cache_promoter: CachePromoter | None = None,
        config: LLMCacheConfig | None = None,
    ):
        """Initialize LLM response cache.

        Args:
            redis_client: Async Redis client for R1 exact match
            semantic_matcher: Qdrant semantic matcher for R2
            cache_promoter: Handles R2 → R1 promotion
            config: LLM cache configuration
        """
        self.config = config or get_llm_cache_config()

        super().__init__(
            redis_client=redis_client,
            semantic_matcher=semantic_matcher,
            cache_promoter=cache_promoter,
            collection_name=self.COLLECTION_NAME,
            default_ttl=self.config.ttl,
            similarity_threshold=self.config.threshold,
        )

        # Metrics
        self.excluded_count = 0
        self.cost_savings_usd = 0.0
        self.model_hits: dict[str, int] = {}

        logger.info(
            f"✅ LLMResponseCache initialized | collection={self.COLLECTION_NAME} | "
            f"ttl={self.config.ttl}s | threshold={self.config.threshold}"
        )

    async def get_response(
        self,
        prompt: str,
        model: str = "default",
    ) -> TwoTierCacheResult[str]:
        """Get cached LLM response or indicate cache miss.

        Args:
            prompt: The LLM prompt
            model: Model identifier for cache partitioning

        Returns:
            TwoTierCacheResult with hit/miss and cached response
        """
        # Check exclusion patterns
        if not self.config.should_cache(prompt):
            self.excluded_count += 1
            logger.debug(f"⚠️ LLM cache EXCLUDED | model={model}")
            return TwoTierCacheResult(
                hit=False,
                tier=None,
                value=None,
                similarity_score=None,
                lookup_ms=0.0,
            )

        # Get model-specific threshold
        self.threshold = self.config.get_threshold(model)

        # Build key input
        key_input = f"{model}:{prompt}"

        # Use parent's get method
        result = await self.get(key_input, prompt=prompt, model=model)

        # Track metrics on hit
        if result.hit:
            self.model_hits[model] = self.model_hits.get(model, 0) + 1
            cost = self.COST_PER_CALL.get(model, self.COST_PER_CALL["default"])
            self.cost_savings_usd += cost

        return result

    async def set_response(
        self,
        prompt: str,
        response: str,
        model: str = "default",
    ) -> None:
        """Cache LLM response.

        Args:
            prompt: The LLM prompt
            response: LLM response to cache
            model: Model identifier
        """
        # Don't cache excluded prompts
        if not self.config.should_cache(prompt):
            logger.debug(f"⚠️ NOT CACHING (excluded) | model={model}")
            return

        # Get model-specific threshold
        self.threshold = self.config.get_threshold(model)

        # Build key input
        key_input = f"{model}:{prompt}"

        # Use parent's set method
        await self.set(key_input, response, prompt=prompt, model=model)

    def _generate_cache_key(self, key_input: str, **kwargs) -> str:
        """Generate exact match cache key from model:prompt.

        Args:
            key_input: Combined model and prompt string

        Returns:
            Hash-based cache key with llm: prefix
        """
        return CacheKeyGenerator.generate(key_input, prefix="llm:")

    def _generate_embedding_input(self, key_input: str, **kwargs) -> str:
        """Generate semantic embedding input.

        Uses just the prompt (without model prefix) for embedding,
        as semantic similarity should match prompt meaning.

        Args:
            key_input: Combined model:prompt string

        Returns:
            Prompt text for embedding
        """
        # Extract prompt from kwargs if available
        prompt = kwargs.get("prompt", "")
        if prompt:
            return prompt

        # Fallback: Parse key_input (format: "model:prompt")
        if ":" in key_input:
            return key_input.split(":", 1)[1]
        return key_input

    def _serialize(self, value: str) -> str:
        """Serialize response string (passthrough).

        Args:
            value: LLM response string

        Returns:
            Response string (unchanged)
        """
        return value

    def _deserialize(self, data: str) -> str:
        """Deserialize response string (passthrough).

        Args:
            data: Cached response string

        Returns:
            Response string (unchanged)
        """
        return data

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dict with overall stats, cost savings, and per-model breakdown
        """
        base_stats = super().get_stats()

        # Add LLM-specific stats
        base_stats["excluded_count"] = self.excluded_count
        base_stats["cost_savings_usd"] = round(self.cost_savings_usd, 4)
        base_stats["per_model"] = {
            model: {
                "hits": hits,
                "threshold": self.config.get_threshold(model),
                "estimated_savings_usd": round(
                    hits * self.COST_PER_CALL.get(model, self.COST_PER_CALL["default"]), 4
                ),
            }
            for model, hits in self.model_hits.items()
        }

        return base_stats

    def reset_stats(self) -> None:
        """Reset all cache statistics."""
        super().reset_stats()
        self.excluded_count = 0
        self.cost_savings_usd = 0.0
        self.model_hits.clear()
