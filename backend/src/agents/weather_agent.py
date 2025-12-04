"""Weather ReAct agent implementation.

This module provides the core weather agent using the ReAct (Reasoning + Acting) pattern.

Level 1 Implementation:
- Basic ReAct agent with zero-shot prompting
- LangChain v1.0+ create_agent() API (modern StateGraph-based agent)
- OpenAI GPT-4o-mini model (cost-effective and fast)
- 3 tools: get_current_weather, get_forecast, retrieve_weather_context
- Temperature 0.7 (will tune in L2)
- NO structured output (deferred to L2)
- NO LLM parameter tuning (deferred to L2)
- NO few-shot examples (deferred to L2)
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from backend.src.tools.weather_tools import (
    get_current_weather,
    get_forecast,
    retrieve_weather_context
)
import os


def create_weather_agent():
    """Create Level 1 ReAct weather agent.

    Creates a basic tool-calling agent that can answer weather queries using
    three tools: get_current_weather, get_forecast, and retrieve_weather_context.

    Returns:
        CompiledStateGraph: LangChain agent graph configured with weather tools

    Example:
        >>> agent = create_weather_agent()
        >>> result = await agent.ainvoke({
        ...     "messages": [{"role": "user", "content": "What's the weather in London?"}]
        ... })
        >>> print(result)

    Note:
        This uses LangChain v1.0+ create_agent() API.
        Returns a StateGraph which is the modern approach for agents.
    """
    # Create LLM with default parameters for Level 1
    # Will tune temperature, max_tokens, etc. in Level 2
    # Using OpenAI GPT-4o-mini as primary model (cost-effective and fast)
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,  # Balanced creativity and consistency
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Define tools (all 3 MCP weather tools)
    tools = [get_current_weather, get_forecast, retrieve_weather_context]

    # Create agent using LangChain v1.0+ create_agent API
    # This returns a CompiledStateGraph with tool calling built-in
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are a professional weather assistant. Answer weather questions accurately and concisely using the available tools.",
        debug=False  # Set to True for verbose logging
    )

    return agent_graph


async def query_weather(user_query: str) -> str:
    """Query weather agent with user input.

    Convenience function for querying the weather agent with a simple string input.

    Args:
        user_query: User's weather question (e.g., "What's the weather in London?")

    Returns:
        str: Agent's response to the query

    Raises:
        Exception: If agent invocation fails

    Example:
        >>> response = await query_weather("Will it rain in Seattle tomorrow?")
        >>> print(response)

    Note:
        For production use with state management, use the agent directly
        and pass WeatherAgentState through a LangGraph workflow.
    """
    agent_graph = create_weather_agent()

    # Invoke agent with user query
    # create_agent() expects "messages" in the state
    # The agent will automatically:
    # 1. Parse the user messages
    # 2. Choose and call appropriate tool(s)
    # 3. Process tool results
    # 4. Generate final answer
    result = await agent_graph.ainvoke({
        "messages": [{"role": "user", "content": user_query}]
    })

    # Extract the final response from the agent's output
    # StateGraph returns a dict with "messages" key
    if isinstance(result, dict) and "messages" in result:
        # Get the last AI message
        messages = result["messages"]
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                return last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                return last_message["content"]

    # Fallback: return string representation
    return str(result)
