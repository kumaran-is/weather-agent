"""LLM Cache Configuration.

Level 9c: Exclusion patterns for agent tool-calling prompts.

Configuration Philosophy:
    - Agent tool-calling prompts: EXCLUDED (context varies too much)
    - Simple Q&A: CACHED (high hit rate potential)
    - Summarization: CACHED (deterministic output)
    - Formatting tasks: CACHED (deterministic output)

Exclusion Patterns:
    Tool-calling prompts contain markers like "Action:", "Thought:",
    "<tool>", etc. These have highly variable context and caching
    them would produce incorrect results.

Per-Model Settings:
    Different models may have different caching strategies:
    - claude-3-sonnet: Higher threshold (0.92) for precision
    - gpt-4: Standard threshold (0.90)
    - Fast models: Lower threshold (0.85) for cost savings

Usage:
    from backend.src.cache.llm_cache import LLMCacheConfig

    config = LLMCacheConfig(
        ttl=3600,
        threshold=0.90,
        exclusion_patterns=["tool.*call", "Action:"],
    )

    # Check if prompt should be cached
    if config.should_cache("What is the weather?"):
        # Safe to cache
        pass

    if not config.should_cache("Action: get_forecast"):
        # Tool-calling prompt - don't cache
        pass
"""

import logging
import re
from typing import ClassVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMCacheConfig(BaseModel):
    """Configuration for LLM response caching.

    Attributes:
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        threshold: Semantic similarity threshold (default: 0.90)
        enabled: Whether LLM caching is enabled
        exclusion_patterns: Regex patterns to exclude from caching
    """

    ttl: int = Field(default=3600, description="Cache TTL in seconds")
    threshold: float = Field(default=0.90, description="Semantic similarity threshold")
    enabled: bool = Field(default=True, description="LLM cache enabled")

    # Patterns to exclude from caching (agent tool-calling, etc.)
    exclusion_patterns: list[str] = Field(
        default=[
            # ReAct agent patterns
            r"tool.*call",
            r"function.*call",
            r"Action:",
            r"Thought:",
            r"Observation:",
            # XML tool patterns
            r"<tool>",
            r"<function_call>",
            r"<tool_use>",
            # OpenAI function calling
            r'"function_call"',
            r'"tool_calls"',
            # LangChain patterns
            r"AgentAction",
            r"AgentFinish",
            # System prompts with tools
            r"You have access to the following tools:",
            r"Available tools:",
        ],
        description="Regex patterns to exclude from caching",
    )

    # Per-model threshold overrides
    model_thresholds: dict[str, float] = Field(
        default={
            "claude-3-opus": 0.92,
            "claude-3-sonnet": 0.90,
            "claude-3-haiku": 0.88,
            "gpt-4": 0.90,
            "gpt-4-turbo": 0.90,
            "gpt-3.5-turbo": 0.85,
        },
        description="Per-model similarity thresholds",
    )

    # Default exclusion patterns (always applied)
    DEFAULT_EXCLUSION_PATTERNS: ClassVar[list[str]] = [
        r"tool.*call",
        r"Action:",
        r"<tool>",
    ]

    def should_cache(self, prompt: str) -> bool:
        """Check if prompt should be cached.

        Returns False if prompt matches any exclusion pattern.

        Args:
            prompt: LLM prompt text

        Returns:
            True if prompt should be cached, False otherwise
        """
        if not self.enabled:
            return False

        # Check exclusion patterns
        for pattern in self.exclusion_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.debug(f"❌ LLM cache EXCLUDED | pattern={pattern}")
                return False

        return True

    def get_threshold(self, model: str = "default") -> float:
        """Get similarity threshold for a specific model.

        Args:
            model: Model identifier (e.g., "claude-3-sonnet")

        Returns:
            Similarity threshold for the model
        """
        return self.model_thresholds.get(model, self.threshold)

    def add_exclusion_pattern(self, pattern: str) -> None:
        """Add a new exclusion pattern.

        Args:
            pattern: Regex pattern to add
        """
        if pattern not in self.exclusion_patterns:
            self.exclusion_patterns.append(pattern)
            logger.info(f"✅ Added exclusion pattern: {pattern}")

    def remove_exclusion_pattern(self, pattern: str) -> bool:
        """Remove an exclusion pattern.

        Args:
            pattern: Regex pattern to remove

        Returns:
            True if pattern was removed, False if not found
        """
        if pattern in self.exclusion_patterns:
            self.exclusion_patterns.remove(pattern)
            logger.info(f"✅ Removed exclusion pattern: {pattern}")
            return True
        return False


# Singleton instance
_llm_cache_config: LLMCacheConfig | None = None


def get_llm_cache_config() -> LLMCacheConfig:
    """Get singleton LLMCacheConfig instance.

    Returns:
        Default LLMCacheConfig instance
    """
    global _llm_cache_config
    if _llm_cache_config is None:
        _llm_cache_config = LLMCacheConfig()
    return _llm_cache_config
