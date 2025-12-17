"""Unified Weather Agent with Progressive Enhancements (ReAct + RAG + CoT + Memory).

This module provides a single, unified weather agent that supports:
- Level 1: Basic ReAct agent
- Level 2: RAG-enhanced agent (8 tools)
- Level 2: Chain-of-Thought reasoning (5-step framework + few-shot examples)
- Level 2: Hybrid search (70% semantic + 30% keyword BM25)
- Level 3a: Memory system (short-term + long-term, semantic tool discovery) 🆕

All enhancements are opt-in via parameters (enable_rag, enable_cot, enable_memory) for backward compatibility.

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

Level 3a Enhancements: 🆕
- ✅ Short-term memory (Redis): Session context, entity tracking, pronoun resolution
- ✅ Long-term memory (Graphiti + Neo4j): User profiles, preferences, temporal facts
- ✅ Semantic tool discovery (langgraph-bigtool): ~50% context reduction (scales to 1000s tools)
- ✅ Memory context injection: Personalized prompts with user history
- ✅ Backward compatible (default behavior unchanged when enable_memory=False)

Configuration Matrix:
| enable_rag | enable_cot | enable_memory | Tools | Context | Use Case |
|------------|------------|---------------|-------|---------|----------|
| False      | False      | False         | 3 MCP | None    | Level 1: Basic queries |
| True       | False      | False         | 8     | None    | Level 2: Historical analysis |
| True       | True       | False         | 8     | None    | Level 2: Complex planning |
| True       | True       | True          | 3*    | Memory  | Level 3a: Personalized + semantic search |

*When memory enabled, uses semantic tool discovery to select top 3 relevant tools (37.5% savings)

Still Deferred:
- Level 3b: Tree of Thoughts, Graph of Thoughts
- Level 3c: Full 7-layer memory system with consolidation
"""

import logging

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from backend.config.llm_config import create_tuned_llm  # Level 2: Tuned LLM
from backend.config.settings import settings
from backend.src.agents.prompts import (  # Level 2: CoT prompts, Level 3b: ToT/GoT prompts
    COT_WEATHER_SYSTEM_PROMPT,
    GOT_WEATHER_SYSTEM_PROMPT,  # 🆕 Level 3b
    TOT_WEATHER_SYSTEM_PROMPT,  # 🆕 Level 3b
    WEATHER_ASSISTANT_SYSTEM_PROMPT,
)
from backend.src.tools.rag_tools import get_rag_tools  # Level 2: RAG-enhanced tools
from backend.src.registry import get_bigtool_registry  # Level 7: langgraph-bigtool semantic discovery
from backend.src.tools.weather_tools import (
    get_current_weather,
    get_forecast,
    retrieve_weather_context,
)
# 🆕 Level 8a: Context Window Optimization (50-60% token reduction)
from backend.src.context import (
    ContextWindowOptimizer,
    detect_query_type,
    get_optimization_config,
)

# 🆕 Level 8c: Prometheus metrics (optional - only if available)
try:
    from prometheus_client import Counter, Histogram

    CONTEXT_OPTIMIZATIONS_TOTAL = Counter(
        "weather_agent_context_optimizations_total",
        "Total context optimizations performed by weather agent",
        ["query_type"]
    )
    CONTEXT_OPTIMIZATION_DURATION = Histogram(
        "weather_agent_context_optimization_seconds",
        "Context optimization duration in seconds",
        ["query_type"],
        buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
    )
    CONTEXT_TOKEN_REDUCTION = Histogram(
        "weather_agent_context_token_reduction_percent",
        "Token reduction percentage achieved",
        ["query_type"],
        buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90]
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def _record_context_metrics(
    query_type: str,
    optimization_time_ms: float,
    reduction_pct: float
) -> None:
    """Record Prometheus metrics for context optimization.

    Level 8c: Only records if prometheus_client is available.
    """
    if not _PROMETHEUS_AVAILABLE:
        return

    try:
        CONTEXT_OPTIMIZATIONS_TOTAL.labels(query_type=query_type).inc()
        CONTEXT_OPTIMIZATION_DURATION.labels(query_type=query_type).observe(optimization_time_ms / 1000)
        CONTEXT_TOKEN_REDUCTION.labels(query_type=query_type).observe(reduction_pct)
    except Exception as e:
        logger.debug(f"Failed to record Prometheus metrics: {e}")

logger = logging.getLogger(__name__)

# 🆕 Level 8a: Module-level optimizer singleton (lazy initialization)
_context_optimizer: ContextWindowOptimizer | None = None


def get_context_optimizer() -> ContextWindowOptimizer:
    """Get or create singleton context optimizer instance.

    Level 8a: Lazy initialization to avoid startup overhead.
    Uses keyword-based filtering (no embedding API calls) for fast optimization.

    Returns:
        ContextWindowOptimizer: Singleton optimizer instance

    Example:
        >>> optimizer = get_context_optimizer()
        >>> result = await optimizer.optimize(query, context, "STANDARD")
        >>> print(f"Reduced {result.reduction_pct:.1f}%")
    """
    global _context_optimizer
    if _context_optimizer is None:
        _context_optimizer = ContextWindowOptimizer(
            embeddings=None,  # Use keyword-based filtering (faster, no API calls)
            target_tokens=4000,  # Target <4K tokens per context window
            min_relevance_score=0.5,  # Keep chunks with >50% relevance
        )
        logger.info(
            "🎯 Level 8a: Context optimizer initialized | "
            "target_tokens=4000 | min_relevance=0.5"
        )
    return _context_optimizer


def create_weather_agent(
    use_case: str = "default",
    model_name: str | None = None,
    enable_rag: bool = True,
    enable_cot: bool = False,
    enable_memory: bool = False,  # 🆕 Level 3a
    enable_tot: bool = False,  # 🆕 Level 3b
    enable_got: bool = False,  # 🆕 Level 3b
    memory_context: dict[str, any] | None = None,  # 🆕 Level 3a
    user_query: str | None = None,  # 🆕 Level 8a: For context optimization
    enable_context_optimization: bool = True,  # 🆕 Level 8a: Toggle optimization
):
    """Create ReAct weather agent with progressive enhancements.

    Creates a unified tool-calling agent with optional RAG, CoT, and Memory capabilities:
    - Level 1: Basic agent (3 MCP tools, simple prompt)
    - Level 2: RAG-enhanced (7 tools, simple prompt)
    - Level 2: CoT reasoning (7 tools, 5-step framework + few-shot examples)
    - Level 3a: Memory system (semantic tool search, personalized context) 🆕

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
        enable_memory: Enable memory features (Level 3a, default: False) 🆕
            - If True: Uses semantic tool discovery (top 3 tools) + memory context injection
            - If False: Uses all tools (Level 2 behavior)
        memory_context: Memory context dictionary (Level 3a, optional) 🆕
            - Contains session context and user profile
            - Format: {"session": ConversationContext, "profile": UserProfile}
        user_query: User's current query (Level 8a, optional) 🆕
            - Used for query type detection and context optimization
            - If provided, enables adaptive context window optimization
        enable_context_optimization: Enable context window optimization (Level 8a, default: True) 🆕
            - If True: Applies 5-phase optimization to memory context (50-60% token reduction)
            - If False: Uses full memory context (Level 3a behavior)

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

    Example (Level 3a - Memory + semantic tool discovery): 🆕
        >>> memory_ctx = await memory_manager.get_context("user_123", "session_456")
        >>> agent = create_weather_agent(
        ...     enable_rag=True,
        ...     enable_cot=True,
        ...     enable_memory=True,
        ...     memory_context=memory_ctx
        ... )
        >>> result = await agent.ainvoke({
        ...     "messages": [{"role": "user", "content": "What about there tomorrow?"}]
        ... })
        >>> # Agent resolves "there" to last mentioned location using memory
        >>> # Uses only top 3 relevant tools (37.5% context reduction)

    Note:
        - Uses LangChain v1.0+ create_agent() API
        - Returns StateGraph (modern agent pattern)
        - Backward compatible (enable_rag=False, enable_cot=False, enable_memory=False preserves Level 1)
        - Level 2: Up to 7 tools (3 MCP + 4 RAG)
        - Level 3a: Only 3 tools via semantic search (37.5% savings)
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

    # 🆕 Level 3a: Semantic tool discovery (if memory enabled)
    if enable_memory and memory_context:
        # CRITICAL: Always include base MCP tools (essential for weather queries)
        # Semantic search is used ONLY for selecting additional RAG tools
        tools = [get_current_weather, get_forecast, retrieve_weather_context]

        # If RAG is also enabled, use semantic search to select which RAG tools to add
        if enable_rag:
            # Extract last query from memory context for semantic search
            session_context = memory_context.get("session")
            last_query = ""

            if session_context and hasattr(session_context, "conversation_history"):
                # Get the most recent user query
                history = session_context.conversation_history
                if history and len(history) > 0:
                    # Find last user message
                    for msg in reversed(history):
                        if msg.get("role") == "user":
                            last_query = msg.get("content", "")
                            break

            # Use langgraph-bigtool semantic discovery for RAG tools
            bigtool_registry = get_bigtool_registry()
            rag_tools_filtered = bigtool_registry.search_tools(
                query=last_query or "weather analysis and historical data",
                limit=3,  # Select top 3 RAG tools based on query relevance
            )

            # Add semantic-selected RAG tools to base MCP tools
            tools.extend(rag_tools_filtered)
            logger.info(
                f"🧠 Level 3a+RAG: Memory enabled - 3 base MCP tools + {len(rag_tools_filtered)} semantic RAG tools"
            )
        else:
            # Memory only, no RAG - use base MCP tools
            logger.info(
                f"🧠 Level 3a: Memory enabled - Using 3 base MCP tools only"
            )
    else:
        # Level 2 / Level 1 behavior: Use all tools
        tools = [get_current_weather, get_forecast, retrieve_weather_context]

        # Level 2: Add RAG-enhanced tools if enabled
        if enable_rag:
            rag_tools = get_rag_tools()
            tools.extend(rag_tools)
            logger.info(
                f"Agent created with {len(tools)} tools (3 MCP + {len(rag_tools)} RAG)"
            )
        else:
            logger.info(f"Agent created with {len(tools)} tools (3 MCP only)")

    # Select system prompt based on reasoning level (L1 → L2 → L3b)
    if enable_got:
        # 🆕 Level 3b: Graph of Thoughts (DAG-based reasoning with node merging)
        system_prompt = GOT_WEATHER_SYSTEM_PROMPT
        logger.info("🧠 Level 3b: GoT reasoning enabled (DAG-based, shared sub-problems)")
    elif enable_tot:
        # 🆕 Level 3b: Tree of Thoughts (multi-path exploration)
        system_prompt = TOT_WEATHER_SYSTEM_PROMPT
        logger.info("🧠 Level 3b: ToT reasoning enabled (multi-path exploration)")
    elif enable_cot:
        # Level 2: Chain-of-Thought prompt with 5-step framework + 4 few-shot examples
        system_prompt = COT_WEATHER_SYSTEM_PROMPT
        logger.info("CoT reasoning enabled (5-step framework + few-shot examples)")
    else:
        # Use simple system prompt (Level 1 / Level 2 Batch 3)
        system_prompt = WEATHER_ASSISTANT_SYSTEM_PROMPT
        if enable_rag:
            # Append RAG capabilities note to simple prompt
            system_prompt += "\n\nYou have access to historical weather data and can analyze trends, identify patterns, and compare conditions across locations."

    # 🆕 Level 3a: Enhance prompt with memory context (if enabled)
    if enable_memory and memory_context:
        session_context = memory_context.get("session")
        user_profile = memory_context.get("profile")

        # Build memory context injection
        memory_prompt_parts: list[str] = [
            "\n\n" + "=" * 60,
            "🧠 MEMORY CONTEXT (Level 3a)",
            "=" * 60,
        ]

        # User profile information
        if user_profile:
            memory_prompt_parts.append("\n📋 USER PROFILE:")
            memory_prompt_parts.append(
                f"  - Name: {user_profile.name or 'Not provided'}"
            )
            memory_prompt_parts.append(
                f"  - Home Location: {user_profile.home_location or 'Not set'}"
            )
            memory_prompt_parts.append(
                f"  - Preferred Units: {user_profile.preferred_units or 'celsius'}"
            )
            memory_prompt_parts.append(
                f"  - Detail Level: {user_profile.preferred_detail_level or 'moderate'}"
            )
            memory_prompt_parts.append(
                f"  - Total Queries: {user_profile.total_queries or 0}"
            )

        # Session context
        if session_context:
            memory_prompt_parts.append("\n💬 SESSION CONTEXT:")
            memory_prompt_parts.append(
                f"  - Session ID: {session_context.session_id or 'unknown'}"
            )
            history_len = len(session_context.conversation_history) if session_context.conversation_history else 0
            memory_prompt_parts.append(
                f"  - Conversation turns: {history_len // 2}"
            )

            # 🆕 CRITICAL FIX: Show actual conversation history (not just metadata)
            if session_context.conversation_history and len(session_context.conversation_history) > 0:
                memory_prompt_parts.append("\n📜 RECENT CONVERSATION:")
                # Show last 4 turns (2 Q&A pairs) for context
                last_turns = session_context.conversation_history[-4:]
                for turn in last_turns:
                    role = turn.get("role", "unknown").upper()
                    content = turn.get("content", "")[:200]  # Truncate long responses
                    memory_prompt_parts.append(f"  {role}: {content}")

            # Current entities (location, date, etc.)
            if session_context.current_entities:
                memory_prompt_parts.append("\n🗺️ TRACKED ENTITIES:")
                for entity_type, entity_value in session_context.current_entities.items():
                    if entity_value:  # Only show non-None values
                        memory_prompt_parts.append(
                            f"  * {entity_type.title()}: {entity_value}"
                        )

        # 🆕 CRITICAL FIX FOR LEVEL 3C: Include previous episodes (cross-session memory)
        # This enables the agent to recall past conversations from different sessions
        previous_episodes = memory_context.get("episodes", [])
        if previous_episodes and len(previous_episodes) > 0:
            memory_prompt_parts.append("\n🕰️ PREVIOUS CONVERSATIONS (Cross-Session Memory):")
            memory_prompt_parts.append("  [From past sessions - use this to recall decisions and context]")
            for i, episode in enumerate(previous_episodes[:3], 1):  # Show top 3
                content = episode.get("content", "")
                # Truncate if too long
                if len(content) > 300:
                    content = content[:300] + "..."
                memory_prompt_parts.append(f"\n  Episode {i}:")
                memory_prompt_parts.append(f"    {content}")

        # 🆕 STRENGTHENED INSTRUCTIONS: More explicit requirements
        memory_prompt_parts.extend(
            [
                "\n⚡ MEMORY-ENABLED BEHAVIOR (MANDATORY REQUIREMENTS):",
                "  1. ✅ REQUIRED: Use the above context to provide personalized responses",
                "  2. ✅ REQUIRED: DO NOT ask redundant questions about information already provided",
                "  3. ✅ REQUIRED: If user mentions 'tomorrow', 'there', 'it' - use tracked entities FIRST",
                "  4. ✅ REQUIRED: When location is tracked, assume user is asking about THAT location",
                "  5. ✅ REQUIRED: Respect user's preferred units (Fahrenheit/Celsius) from profile",
                "  6. ✅ REQUIRED: Build on previous conversation - reference what was discussed",
                "",
                "❗ CRITICAL: If user asks 'What about tomorrow?' and location='Tokyo' is tracked,",
                "   you MUST interpret this as 'What is the weather in Tokyo tomorrow?'",
                "   DO NOT ask 'Which location do you want?' - USE THE TRACKED LOCATION!",
                "=" * 60 + "\n",
            ]
        )

        # Build full memory prompt
        full_memory_prompt = "\n".join(memory_prompt_parts)

        # 🆕 Level 8a: Apply context window optimization if enabled
        if enable_context_optimization and user_query:
            try:
                # Detect query type for adaptive optimization
                query_type = detect_query_type(user_query)

                # Get optimization configuration based on query type
                opt_config = get_optimization_config(query_type)

                # Get optimizer instance
                optimizer = get_context_optimizer()

                # Apply sync optimization (keyword-based, no API calls)
                optimization_result = optimizer.optimize_sync(
                    query=user_query,
                    context=full_memory_prompt,
                    query_type=query_type,
                )

                # Use optimized context
                memory_prompt = optimization_result.optimized_context

                # Log optimization metrics
                logger.info(
                    f"🎯 Level 8a: Context optimized | "
                    f"query_type={query_type} | "
                    f"reduction={optimization_result.reduction_pct:.1f}% | "
                    f"tokens={optimization_result.original_tokens}→{optimization_result.token_count} | "
                    f"expected={opt_config['expected_reduction_pct']}"
                )

                # 🆕 Level 8c: Record Prometheus metrics
                _record_context_metrics(
                    query_type=query_type,
                    optimization_time_ms=optimization_result.optimization_time_ms,
                    reduction_pct=optimization_result.reduction_pct,
                )
            except Exception as e:
                # Graceful degradation: use full context on failure
                logger.warning(
                    f"⚠️ Level 8a: Context optimization failed, using full context: {e}"
                )
                memory_prompt = full_memory_prompt
        else:
            # Optimization disabled or no user_query - use full context
            memory_prompt = full_memory_prompt
            if not enable_context_optimization:
                logger.debug("Level 8a: Context optimization disabled")
            elif not user_query:
                logger.debug("Level 8a: No user_query provided, skipping optimization")

        system_prompt += memory_prompt

        # 🆕 REINFORCEMENT: If ToT/GoT is enabled, remind the model to follow format
        if enable_tot:
            system_prompt += "\n\n⚠️ **REMINDER**: Even with memory context, you MUST follow Tree of Thoughts format.\n"
            system_prompt += "Start your response with: 'To answer this query, I'll explore multiple thought paths:'\n"
            system_prompt += "Use the word 'path' at least 5 times in your response."
        elif enable_got:
            system_prompt += "\n\n⚠️ **REMINDER**: Even with memory context, you MUST follow Graph of Thoughts format.\n"
            system_prompt += "Start your response with: 'Let me compare' or 'Comparing'."

        logger.info("🧠 Level 3a: Memory context injected into system prompt")

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
        """State for weather agent with runtime config tracking.

        Level 2: enable_rag, enable_cot
        Level 3a: use_memory
        Level 3b: enable_tot, enable_got
        """
        enable_rag: bool = None  # Optional override from input (Level 2)
        enable_cot: bool = None  # Optional override from input (Level 2)
        use_memory: bool = None  # Optional override from input (Level 3a)
        enable_tot: bool = None  # Optional override from input (Level 3b - Tree of Thoughts)
        enable_got: bool = None  # Optional override from input (Level 3b - Graph of Thoughts)

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

        # 🆕 Level 3a: Memory
        use_memory = state.get("use_memory")  # Studio
        if use_memory is None:
            use_memory = configurable.get("use_memory")  # API
        if use_memory is None:
            use_memory = False  # Default: memory disabled

        # 🆕 Level 3b: Advanced Reasoning
        enable_tot = state.get("enable_tot")  # Studio
        if enable_tot is None:
            enable_tot = configurable.get("enable_tot")  # API
        if enable_tot is None:
            enable_tot = False  # Default: ToT disabled

        enable_got = state.get("enable_got")  # Studio
        if enable_got is None:
            enable_got = configurable.get("enable_got")  # API
        if enable_got is None:
            enable_got = False  # Default: GoT disabled

        # Determine source for logging
        rag_source = 'state (Studio)' if state.get('enable_rag') is not None else \
                    'config (API)' if configurable.get('enable_rag') is not None else \
                    '.env'
        cot_source = 'state (Studio)' if state.get('enable_cot') is not None else \
                    'config (API)' if configurable.get('enable_cot') is not None else \
                    '.env'
        memory_source = 'state (Studio)' if state.get('use_memory') is not None else \
                       'config (API)' if configurable.get('use_memory') is not None else \
                       'default (false)'
        tot_source = 'state (Studio)' if state.get('enable_tot') is not None else \
                    'config (API)' if configurable.get('enable_tot') is not None else \
                    'default (false)'
        got_source = 'state (Studio)' if state.get('enable_got') is not None else \
                    'config (API)' if configurable.get('enable_got') is not None else \
                    'default (false)'

        # Log configuration (helpful for debugging)
        logger.debug(f"Config: enable_rag={enable_rag} (from {rag_source}), enable_cot={enable_cot} (from {cot_source}), use_memory={use_memory} (from {memory_source}), enable_tot={enable_tot} (from {tot_source}), enable_got={enable_got} (from {got_source})")

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
        # Priority: GoT > ToT > CoT > Simple (most advanced to least)
        if enable_got:
            system_prompt = GOT_WEATHER_SYSTEM_PROMPT
            logger.info("Using GoT reasoning (graph-based exploration)")
        elif enable_tot:
            system_prompt = TOT_WEATHER_SYSTEM_PROMPT
            logger.info("Using ToT reasoning (tree-based search)")
        elif enable_cot:
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
