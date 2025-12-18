"""Unit tests for LLMResponseCache (Level 9c).

Tests validate the LLM response caching functionality:
- Exclusion patterns for agent tool-calling prompts
- Per-model similarity thresholds
- Cost savings tracking
- Prompt caching logic

Key Design Decision:
- Agent tool-calling prompts are EXCLUDED because they have varying
  context and caching them would produce incorrect results.
- Simple Q&A and deterministic tasks are cacheable.
"""

from unittest.mock import AsyncMock

import pytest

from backend.src.cache.llm_cache.llm_cache_config import LLMCacheConfig
from backend.src.cache.llm_cache.llm_response_cache import LLMResponseCache


class TestLLMCacheConfigExclusionPatterns:
    """Test exclusion pattern matching."""

    def test_tool_call_patterns_excluded(self):
        """Test tool call patterns are excluded from caching."""
        config = LLMCacheConfig()

        # ReAct patterns
        assert config.should_cache("Action: get_weather") is False
        assert config.should_cache("Thought: I need to check the weather") is False
        assert config.should_cache("Observation: The weather is sunny") is False

    def test_function_call_patterns_excluded(self):
        """Test function call patterns are excluded."""
        config = LLMCacheConfig()

        assert config.should_cache("function_call: get_forecast") is False
        assert config.should_cache('{"function_call": "get_weather"}') is False
        assert config.should_cache('{"tool_calls": []}') is False

    def test_xml_tool_patterns_excluded(self):
        """Test XML tool patterns are excluded."""
        config = LLMCacheConfig()

        assert config.should_cache("<tool>get_weather</tool>") is False
        assert config.should_cache("<function_call>forecast</function_call>") is False
        assert config.should_cache("<tool_use>check weather</tool_use>") is False

    def test_langchain_patterns_excluded(self):
        """Test LangChain agent patterns are excluded."""
        config = LLMCacheConfig()

        assert config.should_cache("AgentAction: check weather") is False
        assert config.should_cache("AgentFinish: task complete") is False

    def test_system_tool_prompts_excluded(self):
        """Test system prompts with tools are excluded."""
        config = LLMCacheConfig()

        prompt = "You have access to the following tools: get_weather, get_forecast"
        assert config.should_cache(prompt) is False

        prompt = "Available tools: weather_check"
        assert config.should_cache(prompt) is False

    def test_simple_queries_cached(self):
        """Test simple Q&A queries are cacheable."""
        config = LLMCacheConfig()

        assert config.should_cache("What is the capital of France?") is True
        assert config.should_cache("Explain photosynthesis") is True
        assert config.should_cache("Summarize this text: ...") is True

    def test_weather_questions_cached(self):
        """Test weather questions without tools are cacheable."""
        config = LLMCacheConfig()

        assert config.should_cache("What causes hurricanes?") is True
        assert config.should_cache("How do meteorologists predict weather?") is True

    def test_case_insensitive_matching(self):
        """Test exclusion patterns are case insensitive."""
        config = LLMCacheConfig()

        assert config.should_cache("ACTION: check weather") is False
        assert config.should_cache("THOUGHT: analyzing") is False
        assert config.should_cache("<TOOL>get_weather</TOOL>") is False


class TestLLMCacheConfigThresholds:
    """Test per-model threshold configuration."""

    def test_model_specific_thresholds(self):
        """Test different models have different thresholds."""
        config = LLMCacheConfig()

        # Claude models
        assert config.get_threshold("claude-3-opus") == 0.92
        assert config.get_threshold("claude-3-sonnet") == 0.90
        assert config.get_threshold("claude-3-haiku") == 0.88

        # GPT models
        assert config.get_threshold("gpt-4") == 0.90
        assert config.get_threshold("gpt-3.5-turbo") == 0.85

    def test_unknown_model_uses_default(self):
        """Test unknown model uses default threshold."""
        config = LLMCacheConfig(threshold=0.85)

        assert config.get_threshold("unknown-model") == 0.85
        assert config.get_threshold("my-custom-model") == 0.85


class TestLLMCacheConfigManagement:
    """Test config management methods."""

    def test_add_exclusion_pattern(self):
        """Test adding new exclusion pattern."""
        config = LLMCacheConfig()
        initial_count = len(config.exclusion_patterns)

        config.add_exclusion_pattern(r"custom_pattern")

        assert len(config.exclusion_patterns) == initial_count + 1
        assert config.should_cache("custom_pattern: test") is False

    def test_add_duplicate_pattern(self):
        """Test adding duplicate pattern doesn't create duplicate."""
        config = LLMCacheConfig()
        config.add_exclusion_pattern(r"new_pattern")
        count = len(config.exclusion_patterns)

        config.add_exclusion_pattern(r"new_pattern")

        assert len(config.exclusion_patterns) == count

    def test_remove_exclusion_pattern(self):
        """Test removing exclusion pattern."""
        config = LLMCacheConfig()

        # Add and then remove
        config.add_exclusion_pattern(r"removable_pattern")
        assert config.should_cache("removable_pattern test") is False

        result = config.remove_exclusion_pattern(r"removable_pattern")

        assert result is True
        assert config.should_cache("removable_pattern test") is True

    def test_remove_nonexistent_pattern(self):
        """Test removing nonexistent pattern returns False."""
        config = LLMCacheConfig()

        result = config.remove_exclusion_pattern(r"nonexistent")

        assert result is False

    def test_disabled_config(self):
        """Test disabled config never caches."""
        config = LLMCacheConfig(enabled=False)

        assert config.should_cache("Simple question?") is False
        assert config.should_cache("Any prompt") is False


class TestLLMResponseCacheOperations:
    """Test LLMResponseCache operations."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit for simple prompt."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = "Paris is the capital of France."

        cache = LLMResponseCache(redis_client=redis_mock)

        result = await cache.get_response("What is the capital of France?")

        assert result.hit is True
        assert result.value == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None

        semantic_mock = AsyncMock()
        semantic_mock.search.return_value = []

        cache = LLMResponseCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
        )

        result = await cache.get_response("Novel question?")

        assert result.hit is False

    @pytest.mark.asyncio
    async def test_excluded_prompt_not_cached(self):
        """Test excluded prompts return miss immediately."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = "should not be returned"

        cache = LLMResponseCache(redis_client=redis_mock)

        result = await cache.get_response("Action: get_weather Miami")

        assert result.hit is False
        assert cache.excluded_count == 1
        # Redis should not have been called
        redis_mock.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_excluded_prompt_not_stored(self):
        """Test excluded prompts are not stored."""
        redis_mock = AsyncMock()
        semantic_mock = AsyncMock()

        cache = LLMResponseCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
        )

        await cache.set_response(
            prompt="Action: get_forecast",
            response="Tool response",
        )

        # Neither backend should have been called
        redis_mock.setex.assert_not_called()
        semantic_mock.upsert.assert_not_called()


class TestLLMResponseCacheModelHandling:
    """Test per-model cache handling."""

    @pytest.mark.asyncio
    async def test_model_partitioning(self):
        """Test cache is partitioned by model."""
        cache = LLMResponseCache()

        # Different models should generate different cache keys
        key1 = cache._generate_cache_key("gpt-4:test prompt")
        key2 = cache._generate_cache_key("claude-3-sonnet:test prompt")

        assert key1 != key2

    @pytest.mark.asyncio
    async def test_model_threshold_applied(self):
        """Test model-specific threshold is applied."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None

        semantic_mock = AsyncMock()
        semantic_mock.search.return_value = []

        cache = LLMResponseCache(
            redis_client=redis_mock,
            semantic_matcher=semantic_mock,
        )

        # Query with specific model
        await cache.get_response(
            prompt="Test prompt",
            model="claude-3-opus",
        )

        # Threshold should be set to claude-3-opus threshold
        assert cache.threshold == 0.92


class TestLLMResponseCacheCostTracking:
    """Test cost savings tracking."""

    @pytest.mark.asyncio
    async def test_cost_savings_tracked(self):
        """Test cost savings are tracked on cache hits."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = "cached response"

        cache = LLMResponseCache(redis_client=redis_mock)

        await cache.get_response("Test prompt", model="gpt-4")

        assert cache.cost_savings_usd > 0

    @pytest.mark.asyncio
    async def test_cost_savings_by_model(self):
        """Test cost savings vary by model."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = "cached response"

        cache = LLMResponseCache(redis_client=redis_mock)

        # GPT-4 is expensive
        await cache.get_response("Test 1", model="gpt-4")
        gpt4_savings = cache.cost_savings_usd

        cache.cost_savings_usd = 0  # Reset

        # GPT-3.5 is cheaper
        await cache.get_response("Test 2", model="gpt-3.5-turbo")
        gpt35_savings = cache.cost_savings_usd

        # GPT-4 should have saved more
        assert gpt4_savings > gpt35_savings

    @pytest.mark.asyncio
    async def test_per_model_hit_tracking(self):
        """Test per-model hit tracking."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = "cached"

        cache = LLMResponseCache(redis_client=redis_mock)

        await cache.get_response("Q1", model="gpt-4")
        await cache.get_response("Q2", model="gpt-4")
        await cache.get_response("Q3", model="claude-3-sonnet")

        stats = cache.get_stats()

        assert stats["per_model"]["gpt-4"]["hits"] == 2
        assert stats["per_model"]["claude-3-sonnet"]["hits"] == 1


class TestLLMResponseCacheStats:
    """Test cache statistics."""

    @pytest.mark.asyncio
    async def test_excluded_count_tracking(self):
        """Test excluded count tracks excluded prompts."""
        cache = LLMResponseCache()

        await cache.get_response("Action: test1")
        await cache.get_response("Action: test2")
        await cache.get_response("Action: test3")

        assert cache.excluded_count == 3

    @pytest.mark.asyncio
    async def test_stats_include_cost_savings(self):
        """Test stats include cost savings."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = "cached"

        cache = LLMResponseCache(redis_client=redis_mock)
        await cache.get_response("Test", model="gpt-4")

        stats = cache.get_stats()

        assert "cost_savings_usd" in stats
        assert stats["cost_savings_usd"] > 0

    def test_reset_stats(self):
        """Test stats reset clears all counters."""
        cache = LLMResponseCache()
        cache.excluded_count = 10
        cache.cost_savings_usd = 5.0
        cache.model_hits = {"gpt-4": 5}

        cache.reset_stats()

        assert cache.excluded_count == 0
        assert cache.cost_savings_usd == 0.0
        assert len(cache.model_hits) == 0


class TestLLMResponseCacheSerialization:
    """Test serialization/deserialization."""

    def test_serialize_string(self):
        """Test string serialization (passthrough)."""
        cache = LLMResponseCache()

        result = cache._serialize("Test response")

        assert result == "Test response"

    def test_deserialize_string(self):
        """Test string deserialization (passthrough)."""
        cache = LLMResponseCache()

        result = cache._deserialize("Cached response")

        assert result == "Cached response"


class TestLLMResponseCacheEmbeddingInput:
    """Test embedding input generation."""

    def test_embedding_extracts_prompt(self):
        """Test embedding input extracts prompt from key_input."""
        cache = LLMResponseCache()

        # Key input format: "model:prompt"
        embedding = cache._generate_embedding_input(
            "gpt-4:What is the weather?",
            prompt="What is the weather?",
        )

        assert embedding == "What is the weather?"

    def test_embedding_fallback_parsing(self):
        """Test embedding input parses key_input if prompt not provided."""
        cache = LLMResponseCache()

        embedding = cache._generate_embedding_input("gpt-4:Test prompt here")

        assert embedding == "Test prompt here"


class TestLLMCacheConfigSingleton:
    """Test singleton pattern."""

    def test_get_llm_cache_config_singleton(self):
        """Test get_llm_cache_config returns same instance."""
        from backend.src.cache.llm_cache.llm_cache_config import get_llm_cache_config

        config1 = get_llm_cache_config()
        config2 = get_llm_cache_config()

        assert config1 is config2
