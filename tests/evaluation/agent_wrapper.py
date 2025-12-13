"""Agent wrapper for golden dataset evaluation.

This module provides a wrapper function to call the Weather AI Agent
during evaluation. It integrates with the real agent instead of using mocks.
"""

import logging
from typing import Any

from backend.src.agents.weather_agent import query_weather

logger = logging.getLogger(__name__)


async def call_weather_agent(query: str, user_id: str = "eval_user") -> dict[str, Any]:
    """Call the Weather AI Agent for evaluation.

    This function wraps the weather agent to provide a consistent interface
    for the golden dataset runner.

    Args:
        query: User's weather question
        user_id: User ID for session tracking (default: "eval_user")

    Returns:
        dict with keys:
            - answer: Agent's response string
            - trajectory: List of tool calls (empty for now, can be enhanced)
            - error: Error message if agent failed (optional)

    Example:
        >>> result = await call_weather_agent("What's the weather in Miami?")
        >>> print(result["answer"])
    """
    try:
        # Call the real Weather AI Agent
        # Enable RAG for better accuracy, disable CoT for faster evaluation
        answer = await query_weather(
            user_query=query,
            use_case="default",
            enable_rag=True,
            enable_cot=False,  # Disable CoT for faster evaluation
        )

        logger.info(f"Agent response for '{query[:50]}...': {len(answer)} chars")

        return {
            "answer": answer,
            "trajectory": [],  # Can be populated with tool calls if needed
        }

    except Exception as e:
        logger.error(f"Agent failed for query '{query}': {e}", exc_info=True)
        return {
            "answer": f"Error: Agent invocation failed - {str(e)}",
            "trajectory": [{"error": str(e)}],
            "error": str(e),
        }
