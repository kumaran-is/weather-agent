"""Cache Key Generator for exact match caching.

Level 9a: Shared utility for generating deterministic cache keys.

Key Design:
    - SHA-256 hash for collision resistance
    - Deterministic: Same input → same key
    - Configurable prefix for namespacing
    - Truncated hash for readability (16 chars by default)

Usage:
    from backend.src.cache.common import CacheKeyGenerator

    # Basic usage
    key = CacheKeyGenerator.generate("weather in Miami")
    # Returns: "a1b2c3d4e5f6g7h8"

    # With prefix
    key = CacheKeyGenerator.generate("weather in Miami", prefix="tool:")
    # Returns: "tool:a1b2c3d4e5f6g7h8"

    # From dict (for tool args)
    key = CacheKeyGenerator.from_dict({"location": "Miami", "days": 7})
    # Returns: "c4d5e6f7g8h9i0j1"
"""

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CacheKeyGenerator:
    """Generate deterministic cache keys using SHA-256.

    Provides consistent hashing for exact-match cache lookups.
    Supports string inputs, dictionaries, and custom prefixes.
    """

    @staticmethod
    def generate(
        input_string: str,
        prefix: str = "",
        hash_length: int = 16,
    ) -> str:
        """Generate cache key from string input.

        Args:
            input_string: Input text to hash
            prefix: Optional prefix (e.g., "tool:", "llm:")
            hash_length: Truncated hash length (default: 16 chars)

        Returns:
            Cache key string: "{prefix}{hash}"

        Example:
            >>> CacheKeyGenerator.generate("weather in Miami", prefix="query:")
            'query:a1b2c3d4e5f6g7h8'
        """
        # Normalize input (lowercase, strip whitespace)
        normalized = input_string.strip().lower()

        # Generate SHA-256 hash
        hash_digest = hashlib.sha256(normalized.encode()).hexdigest()

        # Truncate to specified length
        truncated = hash_digest[:hash_length]

        return f"{prefix}{truncated}"

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        prefix: str = "",
        hash_length: int = 16,
        exclude_keys: set[str] | None = None,
    ) -> str:
        """Generate cache key from dictionary (e.g., tool arguments).

        Args:
            data: Dictionary to hash
            prefix: Optional prefix
            hash_length: Truncated hash length
            exclude_keys: Keys to exclude from hashing (e.g., {"timestamp"})

        Returns:
            Cache key string

        Example:
            >>> CacheKeyGenerator.from_dict({"location": "Miami", "days": 7})
            'c4d5e6f7g8h9i0j1'
        """
        # Filter excluded keys
        if exclude_keys:
            data = {k: v for k, v in data.items() if k not in exclude_keys}

        # Sort keys for deterministic serialization
        sorted_json = json.dumps(data, sort_keys=True, default=str)

        return CacheKeyGenerator.generate(sorted_json, prefix=prefix, hash_length=hash_length)

    @staticmethod
    def from_tool_call(
        tool_name: str,
        tool_args: dict[str, Any],
        prefix: str = "tool:",
        hash_length: int = 16,
    ) -> str:
        """Generate cache key for tool call.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments dictionary
            prefix: Key prefix (default: "tool:")
            hash_length: Truncated hash length

        Returns:
            Cache key for the tool call

        Example:
            >>> CacheKeyGenerator.from_tool_call("get_forecast", {"location": "Miami"})
            'tool:e7f8g9h0i1j2k3l4'
        """
        # Combine tool name and args
        combined = {
            "tool": tool_name.lower(),
            "args": tool_args,
        }

        return CacheKeyGenerator.from_dict(combined, prefix=prefix, hash_length=hash_length)

    @staticmethod
    def from_llm_prompt(
        prompt: str,
        model: str = "default",
        prefix: str = "llm:",
        hash_length: int = 16,
    ) -> str:
        """Generate cache key for LLM prompt.

        Args:
            prompt: LLM prompt text
            model: Model identifier for cache partitioning
            prefix: Key prefix (default: "llm:")
            hash_length: Truncated hash length

        Returns:
            Cache key for the LLM call

        Example:
            >>> CacheKeyGenerator.from_llm_prompt("What is the weather?", model="claude-3")
            'llm:b2c3d4e5f6g7h8i9'
        """
        # Combine model and prompt
        combined = f"{model}:{prompt}"

        return CacheKeyGenerator.generate(combined, prefix=prefix, hash_length=hash_length)

    @staticmethod
    def from_query(
        query: str,
        enable_rag: bool = True,
        enable_cot: bool = True,
        prefix: str = "query:",
        hash_length: int = 16,
    ) -> str:
        """Generate cache key for weather query.

        Args:
            query: User query text
            enable_rag: RAG enabled flag (affects output)
            enable_cot: Chain-of-thought enabled flag
            prefix: Key prefix (default: "query:")
            hash_length: Truncated hash length

        Returns:
            Cache key for the query

        Example:
            >>> CacheKeyGenerator.from_query("weather in Miami", enable_rag=True)
            'query:f6g7h8i9j0k1l2m3'
        """
        # Build key data (similar to L1/L2 cache key generation)
        key_data = {
            "query": query.strip().lower(),
            "enable_rag": enable_rag,
            "enable_cot": enable_cot,
        }

        return CacheKeyGenerator.from_dict(key_data, prefix=prefix, hash_length=hash_length)
