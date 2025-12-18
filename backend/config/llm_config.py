"""
LLM Configuration for Weather AI Agent Service
Level 2: Tuned parameters for weather domain use cases
Level 5a: Anthropic prompt caching for 50-90% cost reduction
"""

import logging
from dataclasses import dataclass
from typing import Any

from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM parameters by use case"""

    temperature: float
    top_p: float
    reason: str
    # L5a: Prompt caching support
    enable_cache: bool = True  # Enable Anthropic prompt caching by default


# Use-case specific LLM configurations
LLM_CONFIGS = {
    "emergency": LLMConfig(
        temperature=0.3,
        top_p=0.7,
        reason="Maximum accuracy for life-safety decisions (hurricanes, severe weather)",
    ),
    "forecast": LLMConfig(
        temperature=0.5,
        top_p=0.8,
        reason="Balanced accuracy and natural language for weather forecasts",
    ),
    "conversational": LLMConfig(
        temperature=0.7,
        top_p=0.9,
        reason="Natural, engaging responses for general weather queries",
    ),
}


def create_tuned_llm(
    use_case: str = "forecast",
    model: str = "claude-sonnet-4-20250514",
    enable_cache: bool = True,  # L5a: Enable prompt caching
) -> ChatAnthropic:
    """
    Create LLM with tuned parameters for weather domain.

    L5a Enhancement: Anthropic prompt caching for 50-90% cost reduction.
    - Cache reads: 0.1x token cost (90% savings)
    - Cache writes: 1.25x token cost (first request)
    - TTL: 5 minutes (automatic, managed by Anthropic)

    Args:
        use_case: "emergency", "forecast", or "conversational"
        model: Claude model to use (default: claude-sonnet-4-20250514)
        enable_cache: Enable Anthropic prompt caching (default: True)

    Returns:
        ChatAnthropic: Configured LLM instance with tuned parameters and caching

    Examples:
        >>> # For hurricane alerts (maximum accuracy, with caching)
        >>> llm = create_tuned_llm(use_case="emergency")

        >>> # For daily forecasts (balanced, with caching)
        >>> llm = create_tuned_llm(use_case="forecast")

        >>> # Disable caching for testing
        >>> llm = create_tuned_llm(use_case="forecast", enable_cache=False)
    """
    config = LLM_CONFIGS.get(use_case, LLM_CONFIGS["forecast"])

    # L5a: Build model_kwargs with optional prompt caching
    model_kwargs: dict[str, Any] = {"top_p": config.top_p}

    # Enable Anthropic prompt caching (beta feature)
    # This enables cache_control markers in system prompts and tools
    if enable_cache:
        # Add beta header for prompt caching
        model_kwargs["extra_headers"] = {
            "anthropic-beta": "prompt-caching-2024-07-31"
        }
        logger.debug(f"L3 cache: Prompt caching ENABLED for {use_case} use case")
    else:
        logger.debug(f"L3 cache: Prompt caching DISABLED for {use_case} use case")

    return ChatAnthropic(
        model=model,
        temperature=config.temperature,
        model_kwargs=model_kwargs,
    )


def get_llm_config_info(use_case: str = "forecast") -> dict[str, str | float]:
    """
    Get LLM configuration information for a use case

    Args:
        use_case: "emergency", "forecast", or "conversational"

    Returns:
        Dictionary with temperature, top_p, and reason
    """
    config = LLM_CONFIGS.get(use_case, LLM_CONFIGS["forecast"])

    return {
        "use_case": use_case,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "reason": config.reason,
    }
