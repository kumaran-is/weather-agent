"""L3: Anthropic API-level prompt caching.

This module provides utilities for enabling Anthropic's prompt caching feature,
which caches static/semi-static content (system prompts, tool definitions, memory context)
at the API level for 90% cost reduction on cache reads.

Performance:
- Hit latency: Same as LLM call (transparent to client)
- Hit rate: 60-70% (static prompts, common memory contexts)
- Cost savings: 90% (cache reads = 10% of input token cost)
- TTL: 5 minutes (automatic, managed by Anthropic)

Anthropic Prompt Caching Details:
- Cache breakpoints: Up to 4 per request
- Hierarchy: tools → system → messages
- Pricing: Cache writes 1.25x base, cache reads 0.1x base
- TTL: 5 minutes default (1 hour available at 2x cost)
- Minimum cacheable: 1024 tokens (Claude 3 models)

Usage:
    # Prepare messages with cache control markers
    messages = prepare_cached_messages(
        system_prompt=system_prompt,
        tools=tools,
        memory_context=memory_context,
        user_query=user_query,
    )

    # Invoke Anthropic with caching enabled
    response = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        system=system,  # With cache_control markers
        tools=tools,    # With cache_control markers
    )

    # Check cache usage
    cache_stats = extract_cache_stats(response)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def add_cache_control(
    content: list[dict[str, Any]] | dict[str, Any] | str,
    cache_type: str = "ephemeral",
) -> list[dict[str, Any]] | dict[str, Any]:
    """Add cache_control marker to content for Anthropic prompt caching.

    Cache Control Hierarchy (applied in order):
    1. Tools (most static, cached first)
    2. System prompt (static, cached second)
    3. Memory context (semi-static, cached third)
    4. Recent messages (optional, cached last)

    Args:
        content: Content to add cache control to (text, dict, or list)
        cache_type: Type of cache control (default: "ephemeral" = 5min TTL)

    Returns:
        Content with cache_control marker added
    """
    # If string, convert to content block format
    if isinstance(content, str):
        return {
            "type": "text",
            "text": content,
            "cache_control": {"type": cache_type},
        }

    # If dict (single content block), add cache_control
    if isinstance(content, dict):
        content["cache_control"] = {"type": cache_type}
        return content

    # If list of content blocks, add cache_control to last block
    if isinstance(content, list) and len(content) > 0:
        # Only add cache_control to last block (Anthropic requirement)
        content[-1]["cache_control"] = {"type": cache_type}
        return content

    return content


def prepare_cached_system_prompt(
    base_system_prompt: str,
    memory_context: str | None = None,
) -> list[dict[str, Any]]:
    """Prepare system prompt with cache control for optimal caching.

    Cache Strategy:
    - Base system prompt: Always cached (static across all queries)
    - Memory context: Cached if provided (semi-static per user)

    Args:
        base_system_prompt: Static system prompt text
        memory_context: Optional user/session memory context

    Returns:
        System prompt as list of content blocks with cache_control markers
    """
    system_blocks = []

    # Base system prompt (static, always cached)
    system_blocks.append({
        "type": "text",
        "text": base_system_prompt,
        "cache_control": {"type": "ephemeral"},
    })

    # Memory context (semi-static, cache if present)
    if memory_context:
        system_blocks.append({
            "type": "text",
            "text": f"\n\n## User & Session Context\n\n{memory_context}",
            "cache_control": {"type": "ephemeral"},
        })

    logger.debug(
        f"📝 Prepared cached system prompt: {len(system_blocks)} blocks"
    )

    return system_blocks


def prepare_cached_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prepare tools with cache control for optimal caching.

    Cache Strategy:
    - All tools: Always cached (static across all queries)
    - Cache control: Applied to last tool only (Anthropic requirement)

    Args:
        tools: List of tool definitions

    Returns:
        Tools list with cache_control marker on last tool
    """
    if not tools:
        return tools

    # Add cache control to last tool (Anthropic requirement)
    tools_copy = tools.copy()
    tools_copy[-1]["cache_control"] = {"type": "ephemeral"}

    logger.debug(
        f"🔧 Prepared cached tools: {len(tools)} tools, cache on last"
    )

    return tools_copy


def prepare_cached_messages(
    messages: list[dict[str, Any]],
    cache_recent_turns: int = 0,
) -> list[dict[str, Any]]:
    """Prepare message history with optional cache control for recent turns.

    Cache Strategy:
    - Recent turns: Optionally cache last N conversation turns
    - Use case: Long conversations where recent context is frequently reused

    Args:
        messages: Message history
        cache_recent_turns: Number of recent turns to cache (0 = no caching)

    Returns:
        Messages with cache_control markers if cache_recent_turns > 0
    """
    if cache_recent_turns <= 0 or not messages:
        return messages

    messages_copy = messages.copy()

    # Find the message to cache (last N turns)
    # Anthropic requires cache_control on last content block only
    cache_index = max(0, len(messages) - cache_recent_turns)

    if cache_index < len(messages):
        # Get the message to cache
        message = messages_copy[cache_index]

        # Add cache_control to message content
        if isinstance(message.get("content"), str):
            messages_copy[cache_index]["content"] = {
                "type": "text",
                "text": message["content"],
                "cache_control": {"type": "ephemeral"},
            }
        elif isinstance(message.get("content"), list):
            # Add cache_control to last content block
            messages_copy[cache_index]["content"][-1]["cache_control"] = {
                "type": "ephemeral"
            }

        logger.debug(
            f"💬 Prepared cached messages: {cache_recent_turns} turns cached"
        )

    return messages_copy


def extract_cache_stats(response: Any) -> dict[str, Any]:
    """Extract cache usage statistics from Anthropic API response.

    Args:
        response: Anthropic API response object

    Returns:
        Dictionary with cache statistics:
        - cache_creation_input_tokens: Tokens written to cache
        - cache_read_input_tokens: Tokens read from cache
        - input_tokens: Total input tokens (uncached)
        - cache_hit_rate: Percentage of tokens served from cache
        - cost_savings_percent: Estimated cost savings from caching
    """
    usage = getattr(response, "usage", {})

    cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    input_tokens = getattr(usage, "input_tokens", 0)

    total_input = cache_creation + cache_read + input_tokens
    cache_hit_rate = (cache_read / total_input * 100) if total_input > 0 else 0.0

    # Cost calculation (approximate):
    # - Normal input: 1.0x base cost
    # - Cache write: 1.25x base cost
    # - Cache read: 0.1x base cost
    normal_cost = input_tokens * 1.0
    write_cost = cache_creation * 1.25
    read_cost = cache_read * 0.1
    actual_cost = normal_cost + write_cost + read_cost

    # Cost if no caching (all tokens at 1.0x)
    no_cache_cost = total_input * 1.0

    cost_savings = ((no_cache_cost - actual_cost) / no_cache_cost * 100) if no_cache_cost > 0 else 0.0

    stats = {
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "input_tokens": input_tokens,
        "total_input_tokens": total_input,
        "cache_hit_rate": round(cache_hit_rate, 1),
        "cost_savings_percent": round(cost_savings, 1),
    }

    logger.debug(
        f"📊 L3 cache stats: {cache_read}/{total_input} tokens cached "
        f"({cache_hit_rate:.1f}% hit rate, {cost_savings:.1f}% savings)"
    )

    return stats


class AnthropicCacheMetrics:
    """Track Anthropic prompt cache metrics across requests.

    This class maintains running statistics for cache performance:
    - Total cache creations (writes)
    - Total cache reads (hits)
    - Total uncached tokens
    - Aggregate hit rate
    - Aggregate cost savings
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.total_cache_creation = 0
        self.total_cache_read = 0
        self.total_input = 0
        self.request_count = 0

        logger.info("✅ L3 cache metrics initialized")

    def update(self, response: Any) -> None:
        """Update metrics from API response.

        Args:
            response: Anthropic API response object
        """
        stats = extract_cache_stats(response)

        self.total_cache_creation += stats["cache_creation_input_tokens"]
        self.total_cache_read += stats["cache_read_input_tokens"]
        self.total_input += stats["input_tokens"]
        self.request_count += 1

        logger.debug(
            f"📊 L3 metrics updated: {self.request_count} requests, "
            f"{self.get_aggregate_hit_rate():.1f}% hit rate"
        )

    def get_aggregate_hit_rate(self) -> float:
        """Calculate aggregate cache hit rate across all requests.

        Returns:
            Cache hit rate as percentage (0-100)
        """
        total = self.total_cache_creation + self.total_cache_read + self.total_input
        return (self.total_cache_read / total * 100) if total > 0 else 0.0

    def get_aggregate_savings(self) -> float:
        """Calculate aggregate cost savings across all requests.

        Returns:
            Cost savings as percentage (0-100)
        """
        # Normal cost
        normal_cost = self.total_input * 1.0
        write_cost = self.total_cache_creation * 1.25
        read_cost = self.total_cache_read * 0.1
        actual_cost = normal_cost + write_cost + read_cost

        # Cost without caching
        total = self.total_cache_creation + self.total_cache_read + self.total_input
        no_cache_cost = total * 1.0

        return ((no_cache_cost - actual_cost) / no_cache_cost * 100) if no_cache_cost > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate cache statistics.

        Returns:
            Dictionary with aggregate metrics
        """
        return {
            "request_count": self.request_count,
            "total_cache_creation": self.total_cache_creation,
            "total_cache_read": self.total_cache_read,
            "total_input": self.total_input,
            "aggregate_hit_rate": round(self.get_aggregate_hit_rate(), 1),
            "aggregate_savings_percent": round(self.get_aggregate_savings(), 1),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_cache_creation = 0
        self.total_cache_read = 0
        self.total_input = 0
        self.request_count = 0

        logger.info("🗑️  L3 cache metrics RESET")
