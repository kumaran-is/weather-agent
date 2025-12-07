"""
LLM Configuration for Weather AI Agent Service
Level 2: Tuned parameters for weather domain use cases
"""

from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic


@dataclass
class LLMConfig:
    """Configuration for LLM parameters by use case"""

    temperature: float
    top_p: float
    reason: str


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
    use_case: str = "forecast", model: str = "claude-sonnet-4-20250514"
) -> ChatAnthropic:
    """
    Create LLM with tuned parameters for weather domain

    Args:
        use_case: "emergency", "forecast", or "conversational"
        model: Claude model to use (default: claude-sonnet-4-20250514)

    Returns:
        ChatAnthropic: Configured LLM instance with tuned parameters

    Examples:
        >>> # For hurricane alerts (maximum accuracy)
        >>> llm = create_tuned_llm(use_case="emergency")

        >>> # For daily forecasts (balanced)
        >>> llm = create_tuned_llm(use_case="forecast")

        >>> # For casual queries (natural language)
        >>> llm = create_tuned_llm(use_case="conversational")
    """
    config = LLM_CONFIGS.get(use_case, LLM_CONFIGS["forecast"])

    return ChatAnthropic(
        model=model,
        temperature=config.temperature,
        model_kwargs={"top_p": config.top_p},
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
