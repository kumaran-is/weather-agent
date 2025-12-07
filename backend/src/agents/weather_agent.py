"""Unified Weather Agent with Progressive Enhancements (ReAct + RAG + CoT).

This module provides a single, unified weather agent that supports:
- Level 1: Basic ReAct agent
- Level 2 : RAG-enhanced agent (8 tools)
- Level 2 : Chain-of-Thought reasoning (5-step framework + few-shot examples)
- Level 2 : Hybrid search (70% semantic + 30% keyword BM25)

All enhancements are opt-in via parameters (enable_rag, enable_cot) for backward compatibility.

Level 1 Implementation:
- Basic ReAct agent with zero-shot prompting
- LangChain v1.0+ create_agent() API (modern StateGraph-based agent)
- OpenAI GPT-4o-mini model (cost-effective and fast)
- 3 tools: get_current_weather, get_forecast, retrieve_weather_context
- Temperature 0.7 (default)

Level 2 Enhancements:
- ✅ LLM parameter tuning with 3 use cases
- ✅ RAG integration with 5 enhanced tools
  * analyze_trends: Historical weather pattern analysis
  * identify_patterns: Anomaly and pattern detection
  * compare_conditions: Cross-location weather comparison
  * retrieve_weather_knowledge_tool: Semantic knowledge retrieval
  * hybrid_search_weather_knowledge: Hybrid search (70% semantic + 30% BM25)
- ✅ Chain-of-Thought reasoning
  * 5-step framework: Decompose → Gather → Analyze → Synthesize → Recommend
  * 4 few-shot examples (planning, safety, travel, agricultural)
  * Explicit reasoning transparency for complex queries
- ✅ Use-case specific configurations:
  * emergency: temp=0.3, top_p=0.7 (maximum accuracy for hurricanes)
  * forecast: temp=0.5, top_p=0.8 (balanced, default)
  * conversational: temp=0.7, top_p=0.9 (natural language)
- ✅ Supports both OpenAI and Claude models via llm_config
- ✅ Backward compatible (default behavior unchanged)

Configuration Matrix:
| enable_rag | enable_cot | Tools | Prompt | Use Case |
|------------|------------|-------|--------|----------|
| False      | False      | 3 MCP | Simple | Level 1: Basic weather queries |
| True       | False      | 8     | Simple | Level 2: Historical analysis + hybrid search |
| True       | True       | 8     | CoT    | Level 2: Complex planning/safety + hybrid search |

Still Deferred:
- Structured output (with Pydantic models)
- Hybrid search 
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from backend.config.llm_config import create_tuned_llm  # Level 2: Tuned LLM (centralized config)
from backend.src.tools.weather_tools import (
    get_current_weather,
    get_forecast,
    retrieve_weather_context,
)
from backend.src.tools.rag_tools import get_rag_tools  # Level 2: RAG-enhanced tools
from backend.src.agents.prompts import (  # Level 2: CoT prompts
    WEATHER_ASSISTANT_SYSTEM_PROMPT,
    COT_WEATHER_SYSTEM_PROMPT,
)
from backend.config.settings import settings
import logging

logger = logging.getLogger(__name__)


def create_weather_agent(
    use_case: str = "default",
    model_name: str | None = None,
    enable_rag: bool = True,
    enable_cot: bool = False,
):
    """Create ReAct weather agent with progressive enhancements.

    Creates a unified tool-calling agent with optional RAG and CoT capabilities:
    - Level 1: Basic agent (3 MCP tools, simple prompt)
    - Level 2 : RAG-enhanced (7 tools, simple prompt)
    - Level 2 : CoT reasoning (7 tools, 5-step framework + few-shot examples)

    Args:
        use_case: LLM configuration use case (Level 2 enhancement)
            - "default": GPT-4o-mini with temp=0.7 (Level 1 behavior)
            - "emergency": Claude with temp=0.3, top_p=0.7 (maximum accuracy)
            - "forecast": Claude with temp=0.5, top_p=0.8 (balanced)
            - "conversational": Claude with temp=0.7, top_p=0.9 (natural)
        model_name: Optional model override (e.g., "claude-sonnet-4-20250514")
        enable_rag: Enable RAG-enhanced tools (Level 2, default: True)
            - If True: Agent has 7 tools (3 MCP + 4 RAG)
            - If False: Agent has 3 tools (3 MCP only, Level 1 behavior)
        enable_cot: Enable Chain-of-Thought reasoning (Level 2, default: False)
            - If True: Uses 5-step framework (Decompose → Gather → Analyze → Synthesize → Recommend)
            - If False: Uses simple system prompt
            - Includes 4 few-shot examples when enabled

    Returns:
        CompiledStateGraph: LangChain agent graph configured with weather tools

    Example (Level 1 - basic agent):
        >>> agent = create_weather_agent(enable_rag=False, enable_cot=False)
        >>> result = await agent.ainvoke({
        ...     "messages": [{"role": "user", "content": "What's the weather in London?"}]
        ... })

    Example (Level 2 - RAG without CoT):
        >>> agent = create_weather_agent(enable_rag=True, enable_cot=False)
        >>> result = await agent.ainvoke({
        ...     "messages": [{"role": "user", "content": "Analyze temperature trends in Tokyo"}]
        ... })

    Example (Level 2 - RAG + CoT):
        >>> agent = create_weather_agent(enable_rag=True, enable_cot=True)
        >>> result = await agent.ainvoke({
        ...     "messages": [{"role": "user", "content": "Should I plan outdoor activities this weekend in Seattle?"}]
        ... })
        >>> # Agent will show 5-step reasoning:
        >>> # 1. Decompose: Identify weekend = Sat/Sun, factors = temp/rain/wind
        >>> # 2. Gather: Call get_forecast for 7 days
        >>> # 3. Analyze: Compare Saturday vs Sunday conditions
        >>> # 4. Synthesize: Saturday good, Sunday risky
        >>> # 5. Recommend: Plan Saturday, have backup for Sunday

    Note:
        - Uses LangChain v1.0+ create_agent() API
        - Returns StateGraph (modern agent pattern)
        - Backward compatible (enable_rag=False, enable_cot=False preserves Level 1)
        - Level 2: Up to 7 tools (3 MCP + 4 RAG)
        - CoT recommended for complex planning, safety-critical queries
    """
    # Level 2: Use tuned LLM if use_case is specified
    # Level 1: Use default OpenAI GPT-4o-mini
    if use_case != "default":
        # Level 2: Create tuned LLM with use-case specific parameters
        model = create_tuned_llm(use_case=use_case, model=model_name) if model_name else create_tuned_llm(use_case=use_case)
    else:
        # Level 1: Default OpenAI GPT-4o-mini (backward compatible)
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,  # Balanced creativity and consistency
            api_key=settings.OPENAI_API_KEY
        )

    # Define tools: MCP weather tools (Level 1) + optional RAG tools (Level 2)
    tools = [get_current_weather, get_forecast, retrieve_weather_context]

    # Level 2: Add RAG-enhanced tools if enabled
    if enable_rag:
        rag_tools = get_rag_tools()
        tools.extend(rag_tools)
        logger.info(f"Agent created with {len(tools)} tools (3 MCP + {len(rag_tools)} RAG)")
    else:
        logger.info(f"Agent created with {len(tools)} tools (3 MCP only)")

    # Select system prompt based on CoT enablement (Level 2 Batch 4)
    if enable_cot:
        # Use Chain-of-Thought prompt with 5-step framework + 4 few-shot examples
        system_prompt = COT_WEATHER_SYSTEM_PROMPT
        logger.info("CoT reasoning enabled (5-step framework + few-shot examples)")
    else:
        # Use simple system prompt (Level 1 / Level 2 Batch 3)
        system_prompt = WEATHER_ASSISTANT_SYSTEM_PROMPT
        if enable_rag:
            # Append RAG capabilities note to simple prompt
            system_prompt += "\n\nYou have access to historical weather data and can analyze trends, identify patterns, and compare conditions across locations."

    # Create agent using LangChain v1.0+ create_agent API
    # This returns a CompiledStateGraph with tool calling built-in
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        debug=False  # Set to True for verbose logging
    )

    return agent_graph


async def query_weather(
    user_query: str,
    use_case: str = "default",
    enable_rag: bool = True,
    enable_cot: bool = False,
) -> str:
    """Query weather agent with user input.

    Convenience function for querying the unified weather agent with a simple string input.

    Args:
        user_query: User's weather question (e.g., "What's the weather in London?")
        use_case: LLM configuration use case (Level 2 enhancement)
            - "default": Level 1 behavior (GPT-4o-mini, temp=0.7)
            - "emergency": Maximum accuracy for hurricanes (temp=0.3)
            - "forecast": Balanced for forecasts (temp=0.5, default for L2)
            - "conversational": Natural language (temp=0.7)
        enable_rag: Enable RAG-enhanced tools (Level 2, default: True)
        enable_cot: Enable Chain-of-Thought reasoning (Level 2, default: False)

    Returns:
        str: Agent's response to the query

    Raises:
        Exception: If agent invocation fails

    Example (Level 1 - basic):
        >>> response = await query_weather(
        ...     "Will it rain in Seattle tomorrow?",
        ...     enable_rag=False,
        ...     enable_cot=False
        ... )
        >>> print(response)

    Example (Level 2 Batch 3 - RAG without CoT):
        >>> response = await query_weather(
        ...     "Analyze temperature trends in Tokyo over the past 40 years",
        ...     use_case="default",
        ...     enable_rag=True,
        ...     enable_cot=False
        ... )
        >>> print(response)

    Example (Level 2 Batch 4 - RAG + CoT):
        >>> response = await query_weather(
        ...     "Should I plan outdoor activities this weekend in Seattle?",
        ...     use_case="forecast",
        ...     enable_rag=True,
        ...     enable_cot=True
        ... )
        >>> print(response)
        >>> # Shows 5-step reasoning: Decompose → Gather → Analyze → Synthesize → Recommend

    Example (Emergency with CoT):
        >>> response = await query_weather(
        ...     "Should I evacuate for Category 4 hurricane?",
        ...     use_case="emergency",
        ...     enable_rag=True,
        ...     enable_cot=True
        ... )
        >>> print(response)
        >>> # Shows explicit safety reasoning with high confidence

    Note:
        For production use with state management, use the agent directly
        and pass WeatherAgentState through a LangGraph workflow.
    """
    # Create unified agent with specified configuration (Level 2 enhancements)
    agent_graph = create_weather_agent(
        use_case=use_case, enable_rag=enable_rag, enable_cot=enable_cot
    )

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


# ============================================================================
# LangSmith Studio Compatible Graph Factory
# ============================================================================

def create_weather_agent_graph(config: RunnableConfig = None):  # noqa: ARG001
    """Create production weather agent graph with RUNTIME-DYNAMIC configuration.

    **CRITICAL DIFFERENCE**: This graph dynamically selects tools and prompts
    at RUNTIME based on each invocation's configurable parameters, not at
    graph creation time. This enables true per-request configuration.

    **Dual Input Support** (4-Tier Configuration Priority):
    1. **State parameters** (LangSmith Studio) - Highest priority
       - Pass enable_rag/enable_cot at root level of input JSON
    2. **Config parameters** (FastAPI API)
       - Pass via config["configurable"]["enable_rag"]
    3. **Environment variables** (.env)
       - ENABLE_RAG, ENABLE_COT from .env file
    4. **Code defaults** (True/True) - Lowest priority

    This provides maximum flexibility:
    - Set defaults via .env (ENABLE_RAG=true, ENABLE_COT=true)
    - Override at runtime via API or Studio
    - No code changes needed to toggle features
    - Each invocation can have different configuration

    Args:
        config: RunnableConfig (not used in factory, but passed to nodes at runtime)

    Returns:
        CompiledStateGraph: Weather agent that reads config at RUNTIME

    Example (LangSmith Studio):
        {
          "messages": [...],
          "enable_rag": false,    # At root level
          "enable_cot": true
        }

    Example (FastAPI API):
        >>> agent = create_weather_agent_graph()
        >>> result = agent.invoke(
        ...     {"messages": [...]},
        ...     config={"configurable": {"enable_rag": False, "enable_cot": True}}
        ... )

    Example (.env defaults - no overrides):
        {
          "messages": [...]
          # Uses ENABLE_RAG=true, ENABLE_COT=true from .env
        }
    """
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage, AIMessage
    from backend.config.settings import settings

    # Define state (extends MessagesState with config overrides)
    class AgentState(MessagesState):
        """State for weather agent with runtime config tracking."""
        enable_rag: bool = None  # Optional override from input
        enable_cot: bool = None  # Optional override from input

    # Create graph
    workflow = StateGraph(AgentState)

    def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        """Call model with dynamically selected tools and prompt based on runtime config."""
        # Dual input support:
        # 1. LangSmith Studio: Passes enable_rag/enable_cot in state (root level)
        # 2. FastAPI: Passes enable_rag/enable_cot in config["configurable"]

        # Extract from both sources
        configurable = config.get("configurable", {}) if config else {}

        # Priority: State (Studio) > Config (API) > Environment (.env) > Code default
        enable_rag = state.get("enable_rag")  # Studio
        if enable_rag is None:
            enable_rag = configurable.get("enable_rag")  # API
        if enable_rag is None:
            enable_rag = settings.ENABLE_RAG  # .env

        enable_cot = state.get("enable_cot")  # Studio
        if enable_cot is None:
            enable_cot = configurable.get("enable_cot")  # API
        if enable_cot is None:
            enable_cot = settings.ENABLE_COT  # .env

        # Determine source for logging
        rag_source = 'state (Studio)' if state.get('enable_rag') is not None else \
                    'config (API)' if configurable.get('enable_rag') is not None else \
                    '.env'
        cot_source = 'state (Studio)' if state.get('enable_cot') is not None else \
                    'config (API)' if configurable.get('enable_cot') is not None else \
                    '.env'

        # Log configuration (helpful for debugging)
        logger.debug(f"Config: enable_rag={enable_rag} (from {rag_source}), enable_cot={enable_cot} (from {cot_source})")

        # Get other parameters from config
        use_case = configurable.get("use_case", "default")
        model_name = configurable.get("model_name", None)

        # Select model
        if use_case != "default":
            model = create_tuned_llm(use_case=use_case, model=model_name) if model_name else create_tuned_llm(use_case=use_case)
        else:
            model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                api_key=settings.OPENAI_API_KEY
            )

        # Dynamically select tools based on runtime config
        tools = [get_current_weather, get_forecast, retrieve_weather_context]
        if enable_rag:
            rag_tools = get_rag_tools()
            tools.extend(rag_tools)
            logger.info(f"Using {len(tools)} tools (3 MCP + {len(rag_tools)} RAG)")
        else:
            logger.info(f"Using {len(tools)} tools (3 MCP only)")

        # Dynamically select prompt based on runtime config
        if enable_cot:
            system_prompt = COT_WEATHER_SYSTEM_PROMPT
            logger.info("Using CoT reasoning (5-step framework)")
        else:
            system_prompt = WEATHER_ASSISTANT_SYSTEM_PROMPT
            if enable_rag:
                system_prompt += "\n\nYou have access to historical weather data and can analyze trends, identify patterns, and compare conditions across locations."
            logger.info("Using simple prompting")

        # Bind tools to model
        model_with_tools = model.bind_tools(tools)

        # Prepend system message if not already present
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        # Invoke model
        response = model_with_tools.invoke(messages)

        # Return state with response
        return {"messages": [response]}

    async def execute_tools(state: AgentState, config: RunnableConfig) -> AgentState:
        """Execute tools called by the model (async-aware for hybrid_search)."""
        from langchain_core.messages import ToolMessage

        # Dual input support (same priority as call_model)
        configurable = config.get("configurable", {}) if config else {}

        enable_rag = state.get("enable_rag")  # Studio
        if enable_rag is None:
            enable_rag = configurable.get("enable_rag")  # API
        if enable_rag is None:
            enable_rag = settings.ENABLE_RAG  # .env

        # Build tool list (same as in call_model)
        tools = [get_current_weather, get_forecast, retrieve_weather_context]
        if enable_rag:
            rag_tools = get_rag_tools()
            tools.extend(rag_tools)

        # Get the last AI message with tool calls
        last_message = state["messages"][-1]

        # Execute each tool call
        tool_messages = []
        for tool_call in last_message.tool_calls:
            # Find the matching tool by name
            tool = next((t for t in tools if t.name == tool_call["name"]), None)
            if tool:
                try:
                    # Execute the tool (async-aware)
                    # Check if the tool's function is async
                    if hasattr(tool, 'coroutine') and tool.coroutine:
                        # Async tool - use ainvoke
                        result = await tool.ainvoke(tool_call["args"])
                    else:
                        # Sync tool - use invoke
                        result = tool.invoke(tool_call["args"])

                    tool_messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    ))
                except Exception as e:
                    # Handle tool execution errors
                    tool_messages.append(ToolMessage(
                        content=f"Error executing {tool_call['name']}: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    ))
            else:
                # Tool not found (shouldn't happen if enable_rag is respected)
                tool_messages.append(ToolMessage(
                    content=f"Tool '{tool_call['name']}' not available (enable_rag={enable_rag})",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                ))

        # Return state with tool results
        return {"messages": tool_messages}

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)

    # Add edges
    workflow.add_edge(START, "agent")

    # Conditional edge: if model called tools, execute them; otherwise end
    def should_continue(state: AgentState) -> str:
        """Check if we should continue to tool execution or end."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

    # After tool execution, always go back to agent
    workflow.add_edge("tools", "agent")

    # Compile and return
    # Note: recursion_limit (default=25) can be overridden via config at invoke time
    # Updated prompts now include clear termination criteria to prevent loops
    return workflow.compile()
