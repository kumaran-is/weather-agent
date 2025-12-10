"""Unit tests for L3: Anthropic prompt caching utilities.

Tests validate the L3 cache utilities for:
- Cache control marker addition
- System prompt preparation with cache control
- Tools preparation with cache control
- Messages preparation with cache control
- Cache statistics extraction from API responses
- Cache metrics tracking

Expected Performance:
- Hit latency: Same as LLM call (transparent to client)
- Hit rate: 60-70% (static prompts, common memory contexts)
- Cost savings: 90% (cache reads = 10% of input token cost)
- TTL: 5 minutes (automatic, managed by Anthropic)
"""

import pytest
from unittest.mock import Mock

from backend.src.cache.l3_anthropic_cache import (
    add_cache_control,
    extract_cache_stats,
    prepare_cached_messages,
    prepare_cached_system_prompt,
    prepare_cached_tools,
    AnthropicCacheMetrics,
)


class TestL3AnthropicCache:
    """Test L3: Anthropic prompt caching utilities."""

    def test_add_cache_control_to_string(self):
        """Test adding cache control to string content."""
        result = add_cache_control("System prompt text")

        assert result == {
            "type": "text",
            "text": "System prompt text",
            "cache_control": {"type": "ephemeral"},
        }

    def test_add_cache_control_to_dict(self):
        """Test adding cache control to dict content block."""
        content = {
            "type": "text",
            "text": "System prompt text",
        }
        result = add_cache_control(content)

        assert result["cache_control"] == {"type": "ephemeral"}
        assert result["text"] == "System prompt text"

    def test_add_cache_control_to_list(self):
        """Test adding cache control to list of content blocks (last block only)."""
        content = [
            {"type": "text", "text": "Block 1"},
            {"type": "text", "text": "Block 2"},
            {"type": "text", "text": "Block 3"},
        ]
        result = add_cache_control(content)

        # Only last block should have cache_control
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]
        assert result[2]["cache_control"] == {"type": "ephemeral"}

    def test_prepare_cached_system_prompt_base_only(self):
        """Test system prompt preparation with base prompt only."""
        base_prompt = "You are a helpful weather assistant."
        result = prepare_cached_system_prompt(base_prompt)

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == base_prompt
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_prepare_cached_system_prompt_with_memory(self):
        """Test system prompt preparation with base + memory context."""
        base_prompt = "You are a helpful weather assistant."
        memory_context = "User prefers Celsius. Home location: London."

        result = prepare_cached_system_prompt(base_prompt, memory_context)

        assert len(result) == 2

        # Base prompt (first block, cached)
        assert result[0]["text"] == base_prompt
        assert result[0]["cache_control"] == {"type": "ephemeral"}

        # Memory context (second block, cached)
        assert "User & Session Context" in result[1]["text"]
        assert memory_context in result[1]["text"]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_prepare_cached_tools(self):
        """Test tools preparation with cache control on last tool."""
        tools = [
            {"name": "get_weather", "description": "Get current weather"},
            {"name": "get_forecast", "description": "Get weather forecast"},
            {"name": "get_alerts", "description": "Get weather alerts"},
        ]

        result = prepare_cached_tools(tools)

        # Only last tool should have cache_control
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]
        assert result[2]["cache_control"] == {"type": "ephemeral"}

        # Tools should be unchanged otherwise
        assert result[0]["name"] == "get_weather"
        assert result[2]["name"] == "get_alerts"

    def test_prepare_cached_tools_empty_list(self):
        """Test tools preparation with empty list."""
        tools = []
        result = prepare_cached_tools(tools)

        assert result == []

    def test_prepare_cached_messages_no_caching(self):
        """Test messages preparation with caching disabled (cache_recent_turns=0)."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "15°C and sunny"},
            {"role": "user", "content": "Will it rain tomorrow?"},
        ]

        result = prepare_cached_messages(messages, cache_recent_turns=0)

        # Messages should be unchanged
        assert result == messages
        for msg in result:
            if isinstance(msg.get("content"), dict):
                assert "cache_control" not in msg["content"]

    def test_prepare_cached_messages_with_caching(self):
        """Test messages preparation with recent turns cached."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "15°C and sunny"},
            {"role": "user", "content": "Will it rain tomorrow?"},
            {"role": "assistant", "content": "Rain likely"},
        ]

        result = prepare_cached_messages(messages, cache_recent_turns=2)

        # Last 2 messages should be cached
        # Message at index 2 (3rd message) should have cache_control
        assert result[2]["content"]["cache_control"] == {"type": "ephemeral"}

    def test_extract_cache_stats_cache_hit(self):
        """Test cache statistics extraction from API response with cache hit."""
        # Mock Anthropic API response with cache hit
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 10000  # 10K tokens from cache
        mock_response.usage.input_tokens = 5000  # 5K uncached tokens

        result = extract_cache_stats(mock_response)

        assert result["cache_creation_input_tokens"] == 0
        assert result["cache_read_input_tokens"] == 10000
        assert result["input_tokens"] == 5000
        assert result["total_input_tokens"] == 15000

        # Cache hit rate: 10K / 15K = 66.7%
        assert result["cache_hit_rate"] == 66.7

        # Cost savings calculation:
        # Normal cost: 5K × 1.0 = 5K
        # Cache write: 0 × 1.25 = 0
        # Cache read: 10K × 0.1 = 1K
        # Actual cost: 6K
        # No cache cost: 15K × 1.0 = 15K
        # Savings: (15K - 6K) / 15K = 60%
        assert result["cost_savings_percent"] == 60.0

    def test_extract_cache_stats_cache_write(self):
        """Test cache statistics extraction with cache write (first request)."""
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.cache_creation_input_tokens = 10000  # Writing to cache
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.input_tokens = 5000

        result = extract_cache_stats(mock_response)

        assert result["cache_creation_input_tokens"] == 10000
        assert result["cache_read_input_tokens"] == 0
        assert result["cache_hit_rate"] == 0.0

        # Cost with cache write is higher initially (1.25x)
        # Normal cost: 5K × 1.0 = 5K
        # Cache write: 10K × 1.25 = 12.5K
        # Actual cost: 17.5K
        # No cache cost: 15K × 1.0 = 15K
        # Savings: (15K - 17.5K) / 15K = -16.7% (negative = more expensive)
        assert result["cost_savings_percent"] < 0  # More expensive on first write

    def test_extract_cache_stats_no_caching(self):
        """Test cache statistics extraction with no caching."""
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.input_tokens = 15000

        result = extract_cache_stats(mock_response)

        assert result["cache_hit_rate"] == 0.0
        assert result["cost_savings_percent"] == 0.0

    def test_anthropic_cache_metrics_initialization(self):
        """Test AnthropicCacheMetrics initialization."""
        metrics = AnthropicCacheMetrics()

        assert metrics.total_cache_creation == 0
        assert metrics.total_cache_read == 0
        assert metrics.total_input == 0
        assert metrics.request_count == 0

    def test_anthropic_cache_metrics_update(self):
        """Test metrics update from API response."""
        metrics = AnthropicCacheMetrics()

        # Mock response
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.cache_creation_input_tokens = 1000
        mock_response.usage.cache_read_input_tokens = 5000
        mock_response.usage.input_tokens = 2000

        metrics.update(mock_response)

        assert metrics.total_cache_creation == 1000
        assert metrics.total_cache_read == 5000
        assert metrics.total_input == 2000
        assert metrics.request_count == 1

    def test_anthropic_cache_metrics_aggregate_hit_rate(self):
        """Test aggregate hit rate calculation across multiple requests."""
        metrics = AnthropicCacheMetrics()

        # Request 1: Cache write
        mock_response1 = Mock()
        mock_response1.usage = Mock()
        mock_response1.usage.cache_creation_input_tokens = 10000
        mock_response1.usage.cache_read_input_tokens = 0
        mock_response1.usage.input_tokens = 5000

        metrics.update(mock_response1)

        # Request 2: Cache hit
        mock_response2 = Mock()
        mock_response2.usage = Mock()
        mock_response2.usage.cache_creation_input_tokens = 0
        mock_response2.usage.cache_read_input_tokens = 10000
        mock_response2.usage.input_tokens = 5000

        metrics.update(mock_response2)

        # Aggregate hit rate: 10K reads / 30K total = 33.3%
        hit_rate = metrics.get_aggregate_hit_rate()
        assert abs(hit_rate - 33.3) < 0.1  # Allow small rounding error

    def test_anthropic_cache_metrics_aggregate_savings(self):
        """Test aggregate cost savings calculation."""
        metrics = AnthropicCacheMetrics()

        # Multiple requests with cache hits
        for _ in range(5):
            mock_response = Mock()
            mock_response.usage = Mock()
            mock_response.usage.cache_creation_input_tokens = 0
            mock_response.usage.cache_read_input_tokens = 10000
            mock_response.usage.input_tokens = 5000

            metrics.update(mock_response)

        # All requests have 66.7% cache hit rate
        # Should result in ~60% cost savings
        savings = metrics.get_aggregate_savings()
        assert savings == 60.0

    def test_anthropic_cache_metrics_get_stats(self):
        """Test get_stats method returns complete metrics."""
        metrics = AnthropicCacheMetrics()

        # Add some data
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.cache_creation_input_tokens = 1000
        mock_response.usage.cache_read_input_tokens = 5000
        mock_response.usage.input_tokens = 2000

        metrics.update(mock_response)

        stats = metrics.get_stats()

        assert stats["request_count"] == 1
        assert stats["total_cache_creation"] == 1000
        assert stats["total_cache_read"] == 5000
        assert stats["total_input"] == 2000
        assert "aggregate_hit_rate" in stats
        assert "aggregate_savings_percent" in stats

    def test_anthropic_cache_metrics_reset(self):
        """Test metrics reset."""
        metrics = AnthropicCacheMetrics()

        # Add data
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.cache_creation_input_tokens = 1000
        mock_response.usage.cache_read_input_tokens = 5000
        mock_response.usage.input_tokens = 2000

        metrics.update(mock_response)

        # Reset
        metrics.reset()

        assert metrics.total_cache_creation == 0
        assert metrics.total_cache_read == 0
        assert metrics.total_input == 0
        assert metrics.request_count == 0
