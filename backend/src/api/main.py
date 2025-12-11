"""FastAPI application for Weather AI Agent - Level 4 Multi-Agent + L5a Caching.

This module provides the main FastAPI application with REST endpoints for
weather queries and hurricane alert management with HITL approval.

Level 1 Implementation:
- 3 main endpoints: /weather/query, /weather/hurricane/alert, /weather/hurricane/approve
- 1 health check endpoint: /health
- Basic error handling
- Pydantic request/response validation

Level 2 Enhancements:
- Unified weather agent with flexible RAG and CoT configuration
- 3-tier configuration: Runtime overrides > Env vars (.env) > Code defaults
- Feature flags: ENABLE_RAG (8 tools), ENABLE_COT (5-step reasoning)
- LLM parameter tuning (temperature 0.5, top_p 0.8 for CoT)
- RAG integration with 8 tools (3 MCP + 5 RAG including hybrid search)
- Hybrid Search (70% semantic + 30% BM25 keyword)
- Chain-of-Thought with few-shot examples
- Pydantic v2 structured outputs with life-safety validation

Level 3a Enhancements:
- Memory system (short-term + long-term)
- Semantic tool discovery (37.5% context reduction)
- Personalized prompts with user history
- Pronoun resolution ("there" → tracked location)
- User profile and session management

Level 4 Multi-Agent Enhancements: 🆕
- L4a: 3-agent system (Triage, Hurricane Specialist, Alert Manager)
- L4b: 8-agent orchestration with Supervisor and parallel execution
- L4c: 15-agent production system with debate, reflection, and advanced features
- AUTO mode: Automatic agent level selection based on query complexity
- Confidence-based routing and risk assessment
- Multi-agent response synthesis and quality improvement

Level 5a Caching Enhancements:
- L1: In-process LRU cache (<1ms, 15-25% hit rate, 100% savings)
- L2: Redis distributed cache (<10ms, 30-40% hit rate, 100% savings)
- L3: Anthropic prompt cache (transparent, 60-70% hit rate, 90% savings)
- Target: 60-75% overall cost reduction ($0.051 → $0.01-0.05 per query)
- Expected savings: $3,805/month at 100K queries

Still Deferred:
- NO authentication (deferred to L5c)
- NO rate limiting (deferred to L5c)
- NO metrics tracking (deferred to L5c)

API Endpoints:
- POST /weather/query - Flexible weather queries (supports multi-agent, RAG, CoT, memory)
- POST /weather/hurricane/alert - Create hurricane alert (may require approval)
- POST /weather/hurricane/approve/{thread_id} - Approve or reject pending alert
- GET /health - Health check
- GET /cache/stats - Cache statistics (L1 + L2 + L3)
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from backend.config.cache_config import cache_config  # 🆕 L5a: Cache configuration
from backend.src.agents.weather_agent import create_weather_agent
from backend.src.cache import (  # 🆕 L5a: Cache imports
    AnthropicCacheMetrics,
    QueryCache,
    RedisQueryCache,
)
from backend.src.memory.manager import MemoryManager  # Level 3a: Memory support
from backend.src.models import (
    HealthCheckResponse,
    HurricaneAlertRequest,
    HurricaneAlertResponse,
    HurricaneApprovalRequest,
    HurricaneApprovalResponse,
    WeatherQuery,
    WeatherResponse,
)
# 🆕 v0.6.0: Auto-routing classifier (replaces explicit agent_level)
from backend.src.routing import classify_query, QueryTier
# 🆕 Level 4: Multi-agent workflow imports
from backend.src.orchestration.multi_agent_workflow import (
    compile_workflow,
    compile_level4b_workflow,
    invoke_workflow,
    invoke_workflow_v2,
)
from backend.src.workflows.weather_graph import get_weather_hitl_workflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize workflow (module-level, safe to initialize once)
workflow = get_weather_hitl_workflow()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown.

    This replaces the deprecated @app.on_event decorators with the modern
    lifespan context manager pattern (FastAPI 0.100+).

    The code before yield runs at startup, and the code after yield runs at shutdown.
    """
    # ========== STARTUP ==========
    logger.info("🚀 Weather AI Agent API starting up...")
    logger.info("Level 4 Multi-Agent + L5a Caching 🆕")
    logger.info("Level 4 Multi-Agent System:")
    logger.info("  - L4a: 3-agent (Triage, Hurricane Specialist, Alert Manager)")
    logger.info("  - L4b: 8-agent with Supervisor orchestration + parallel execution")
    logger.info("  - L4c: 15-agent production (debate, reflection, emergency)")
    logger.info("  - AUTO mode: Automatic agent selection based on complexity")
    logger.info("Level 3 Memory System:")
    logger.info("  - Short-term memory: Redis (30min TTL, entity tracking)")
    logger.info("  - Long-term memory: Graphiti + Neo4j (user profiles, temporal facts)")
    logger.info("  - Semantic tool discovery: 37.5% context reduction (8→3 tools)")
    logger.info("  - Pronoun resolution: 'there' → tracked location")
    logger.info("Level 5a Caching:")
    logger.info("  - L1 cache: In-process LRU (<1ms, 15-25% hit rate)")
    logger.info("  - L2 cache: Redis distributed (<10ms, 30-40% hit rate)")
    logger.info("  - L3 cache: Anthropic prompt (transparent, 60-70% hit rate)")
    logger.info("  - Target: 60-75% cost reduction")
    logger.info("API documentation available at /docs")

    # 🆕 L5a: Initialize cache layers
    if cache_config.CACHE_ENABLED:
        logger.info("💾 Initializing cache layers...")

        # L1: In-process LRU cache
        if cache_config.L1_CACHE_ENABLED:
            app.state.l1_cache = QueryCache(
                max_size=cache_config.L1_CACHE_MAX_SIZE,
                ttl_seconds=cache_config.L1_CACHE_TTL_SECONDS,
            )
            logger.info("✅ L1 cache initialized (in-process LRU)")
        else:
            app.state.l1_cache = None
            logger.info("⏭️  L1 cache disabled (config)")

        # L2: Redis distributed cache
        if cache_config.L2_CACHE_ENABLED:
            app.state.l2_cache = RedisQueryCache(
                redis_url=cache_config.L2_CACHE_REDIS_URL,
                ttl_seconds=cache_config.L2_CACHE_TTL_SECONDS,
                key_prefix=cache_config.L2_CACHE_KEY_PREFIX,
            )
            try:
                await app.state.l2_cache.connect()
                logger.info("✅ L2 cache initialized (Redis distributed)")
            except Exception as e:
                logger.error(f"❌ L2 cache connection failed: {e}")
                logger.error("   L2 cache will be disabled (graceful degradation)")
                app.state.l2_cache = None
        else:
            app.state.l2_cache = None
            logger.info("⏭️  L2 cache disabled (config)")

        # L3: Anthropic prompt cache metrics tracker
        if cache_config.L3_CACHE_ENABLED:
            app.state.l3_cache_metrics = AnthropicCacheMetrics()
            logger.info("✅ L3 cache metrics initialized (Anthropic prompt caching)")
        else:
            app.state.l3_cache_metrics = None
            logger.info("⏭️  L3 cache disabled (config)")
    else:
        app.state.l1_cache = None
        app.state.l2_cache = None
        app.state.l3_cache_metrics = None
        logger.info("⏭️  ALL caching disabled (config)")

    # Initialize memory manager in app.state (NOT global variable)
    # CRITICAL: This calls Graphiti.build_indices_and_constraints() which is expensive
    # By doing it at startup, we avoid 2-3s overhead on EVERY request
    logger.info("🧠 Initializing memory manager (Graphiti + Neo4j)...")
    app.state.memory_manager = MemoryManager()

    # Pre-warm Graphiti connection with retry logic (Neo4j may not be ready immediately)
    max_retries = 5
    retry_delay = 2.0  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            # Create a test context to ensure Graphiti is fully initialized
            await app.state.memory_manager.get_context(
                user_id="__startup_test__",
                session_id="__startup_session__"
            )
            logger.info(f"✅ Memory manager initialized successfully (attempt {attempt}/{max_retries})")
            break  # Success - exit retry loop
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"⚠️  Memory manager init attempt {attempt}/{max_retries} failed: {e}")
                logger.warning(f"   Retrying in {retry_delay}s... (Neo4j may still be starting)")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(f"❌ Failed to initialize memory manager after {max_retries} attempts: {e}")
                logger.error("   Memory features will be disabled until service restart")
                app.state.memory_manager = None

    # 🆕 Level 4: Pre-compile multi-agent workflows for faster request handling
    logger.info("🤖 Initializing Level 4 multi-agent workflows...")
    try:
        # Compile Level 4a workflow (3-agent)
        app.state.workflow_l4a = compile_workflow()
        logger.info("✅ Level 4a workflow compiled (3-agent: Triage, Specialist, Alert)")

        # Compile Level 4b workflow (8-agent with supervisor)
        app.state.workflow_l4b = compile_level4b_workflow()
        logger.info("✅ Level 4b workflow compiled (8-agent with Supervisor)")

        # Level 4c uses invoke_workflow_v2 with supervisor orchestration
        # No separate compilation needed - it dynamically orchestrates
        logger.info("✅ Level 4c ready (15-agent production via Supervisor)")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Level 4 workflows: {e}")
        logger.error("   Multi-agent features will fall back to basic agent")
        app.state.workflow_l4a = None
        app.state.workflow_l4b = None

    yield  # Application runs here

    # ========== SHUTDOWN ==========
    logger.info("Weather AI Agent API shutting down...")

    # 🆕 L5a: Close cache layers gracefully
    if hasattr(app.state, "l2_cache") and app.state.l2_cache:
        try:
            await app.state.l2_cache.close()
            logger.info("✅ L2 cache closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing L2 cache: {e}")

    # Close memory manager gracefully
    if hasattr(app.state, "memory_manager") and app.state.memory_manager:
        try:
            await app.state.memory_manager.close()
            logger.info("✅ Memory manager closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing memory manager: {e}")


# Create FastAPI app with lifespan context manager
app = FastAPI(
    title="Weather AI Agent API",
    description=(
        "Level 4 Multi-Agent System with AUTO-ROUTING (v0.6.0):\n\n"
        "**🆕 AUTO-ROUTING (v0.6.0):**\n"
        "- No explicit `use_multi_agent` or `agent_level` flags needed\n"
        "- Intelligent query classification based on intent\n"
        "- Simple queries → Basic agent (fast, low cost)\n"
        "- Hurricane queries → L4A 3-agent\n"
        "- Complex analysis → L4B 8-agent\n"
        "- Emergency/safety → L4C 15-agent with HITL\n\n"
        "**Level 4 Multi-Agent Orchestration:**\n"
        "- L4a: 3-agent (Triage, Hurricane Specialist, Alert Manager)\n"
        "- L4b: 8-agent with Supervisor orchestration + parallel execution\n"
        "- L4c: 15-agent production (debate, reflection, emergency)\n\n"
        "**Level 3 Memory System:**\n"
        "- Short-term: Redis (entity tracking, pronoun resolution)\n"
        "- Long-term: Graphiti + Neo4j (user profiles, temporal facts)\n\n"
        "**Level 2 RAG + CoT:**\n"
        "- Hybrid Search (70% semantic + 30% BM25)\n"
        "- Chain-of-Thought reasoning\n\n"
        "**Level 5a Caching:**\n"
        "- L1: In-process LRU, L2: Redis, L3: Anthropic prompt\n\n"
        "**HITL Approval:**\n"
        "- Emergency tier queries may trigger human approval"
    ),
    version="0.6.0",  # 🆕 Auto-routing version
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # ✅ Modern pattern (FastAPI 0.100+)
)

# Add CORS middleware (for Level 1, allow all origins)
# In production (L5c), configure specific allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will restrict in L5c
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Dependency Injection ==========

async def get_memory_manager(request: Request) -> MemoryManager | None:
    """Dependency injection for memory manager.

    Returns the app-level memory manager instance initialized at startup.
    Returns None if memory manager failed to initialize.

    Example:
        @app.post("/endpoint")
        async def endpoint(memory: MemoryManager = Depends(get_memory_manager)):
            if memory:
                context = await memory.get_context(...)
    """
    return getattr(request.app.state, "memory_manager", None)


# 🆕 L5a: Cache dependency injection
async def get_l1_cache(request: Request) -> QueryCache | None:
    """Dependency injection for L1 cache."""
    return getattr(request.app.state, "l1_cache", None)


async def get_l2_cache(request: Request) -> RedisQueryCache | None:
    """Dependency injection for L2 cache."""
    return getattr(request.app.state, "l2_cache", None)


async def get_l3_metrics(request: Request) -> AnthropicCacheMetrics | None:
    """Dependency injection for L3 cache metrics."""
    return getattr(request.app.state, "l3_cache_metrics", None)


# 🆕 Level 4: Workflow dependency injection
async def get_workflow_l4a(request: Request):
    """Dependency injection for Level 4a workflow (3-agent)."""
    return getattr(request.app.state, "workflow_l4a", None)


async def get_workflow_l4b(request: Request):
    """Dependency injection for Level 4b workflow (8-agent)."""
    return getattr(request.app.state, "workflow_l4b", None)


# 🆕 v0.6.0: Helper function removed - replaced by backend.src.routing.classify_query
# The old _resolve_agent_level function has been replaced by the new routing module
# which provides more sophisticated intent-based classification.


async def _invoke_basic_agent(
    query: str,
    effective_rag: bool,
    effective_cot: bool,
    effective_memory: bool,
    effective_tot: bool,
    effective_got: bool,
    memory_context: dict | None,
) -> str:
    """Invoke the basic single-agent weather agent (Level 1-3 behavior).

    This is the fallback when multi-agent is disabled or unavailable.

    Args:
        query: User's query
        effective_rag: Whether RAG is enabled
        effective_cot: Whether CoT is enabled
        effective_memory: Whether memory is enabled
        effective_tot: Whether ToT is enabled
        effective_got: Whether GoT is enabled
        memory_context: Memory context dict (if any)

    Returns:
        Response text from the agent
    """
    agent = create_weather_agent(
        use_case="default",
        enable_rag=effective_rag,
        enable_cot=effective_cot,
        enable_memory=effective_memory,
        enable_tot=effective_tot,
        enable_got=effective_got,
        memory_context=memory_context,
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": query}]
    })

    return result["messages"][-1].content


@app.post(
    "/weather/query",
    response_model=WeatherResponse,
    status_code=status.HTTP_200_OK,
    summary="Query weather conditions",
    description=(
        "Ask the weather agent about current conditions or forecasts. "
        "Uses AUTO-ROUTING (v0.6.0) to automatically select the optimal agent tier. "
        "Supports RAG, CoT, Memory, and 3-layer caching."
    ),
    tags=["Weather"]
)
async def weather_query_endpoint(
    query: WeatherQuery,
    enable_rag: bool | None = None,
    enable_cot: bool | None = None,
    enable_tot: bool | None = None,  # Level 3b
    enable_got: bool | None = None,  # Level 3b
    use_memory: bool | None = None,  # Level 3a
    # 🆕 v0.6.0: REMOVED use_multi_agent and agent_level
    # Routing is now automatic based on query intent
    memory_manager: MemoryManager | None = Depends(get_memory_manager),
    l1_cache: QueryCache | None = Depends(get_l1_cache),
    l2_cache: RedisQueryCache | None = Depends(get_l2_cache),
    l3_metrics: AnthropicCacheMetrics | None = Depends(get_l3_metrics),
    workflow_l4a=Depends(get_workflow_l4a),
    workflow_l4b=Depends(get_workflow_l4b),
):
    """Query weather agent with AUTO-ROUTING (v0.6.0).

    This endpoint accepts natural language weather questions and automatically
    routes to the optimal agent tier based on query intent classification.

    **🆕 AUTO-ROUTING (v0.6.0)**:
    No need to specify `use_multi_agent` or `agent_level` - routing is automatic!

    - SIMPLE queries (weather, temperature) → Basic agent
    - STANDARD queries (hurricane, storm) → L4A 3-agent
    - COMPLEX queries (compare, analyze) → L4B 8-agent
    - EMERGENCY queries (evacuate, safety) → L4C 15-agent with HITL

    **Configuration Priority** (for feature flags):
    1. Query parameters - Highest priority
    2. WeatherQuery model fields - Model defaults
    3. Environment variables - System defaults
    4. Code defaults - Fallback

    Example Request:
        POST /weather/query
        {
            "query": "Is Hurricane Milton going to hit Tampa?",
            "user_id": "user123",
            "session_id": "session456"
        }

    Example Response:
        {
            "response": "Based on current NHC data...",
            "user_id": "user123",
            "timestamp": "2025-12-03T16:20:00Z",
            "agents_invoked": ["triage", "hurricane_specialist", "alert_manager"],
            "agent_level": "l4a",
            "query_complexity": "standard",
            "execution_time_ms": 892.3,
            "cache_hit": false,
            "cache_layer": null
        }

    Args:
        query: WeatherQuery with user question, user_id, session_id
        enable_rag: Optional override for RAG (None = use default)
        enable_cot: Optional override for CoT (None = use default)
        enable_tot: Optional override for ToT (None = use default)
        enable_got: Optional override for GoT (None = use default)
        use_memory: Optional override for memory (None = use default)

    Returns:
        WeatherResponse with agent's answer and metadata

    Raises:
        HTTPException: 500 if agent query fails
    """
    from backend.config.settings import settings

    # Determine effective configuration (query param > model field > env var > default)
    effective_rag = (
        enable_rag
        if enable_rag is not None
        else (query.enable_rag if hasattr(query, "enable_rag") else settings.ENABLE_RAG)
    )
    effective_cot = (
        enable_cot
        if enable_cot is not None
        else (query.enable_cot if hasattr(query, "enable_cot") else settings.ENABLE_COT)
    )
    # 🆕 Level 3b: Advanced reasoning configuration
    effective_tot = (
        enable_tot
        if enable_tot is not None
        else (query.enable_tot if hasattr(query, "enable_tot") else False)
    )
    effective_got = (
        enable_got
        if enable_got is not None
        else (query.enable_got if hasattr(query, "enable_got") else False)
    )
    effective_memory = (
        use_memory
        if use_memory is not None
        else (query.use_memory if hasattr(query, "use_memory") else False)
    )

    # 🆕 v0.6.0: REMOVED use_multi_agent and agent_level configuration
    # Routing is now automatic via classify_query()

    # Generate session_id if not provided
    session_id = query.session_id or f"session_{uuid.uuid4().hex[:8]}"

    try:
        logger.info(
            f"Weather query received | "
            f"user_id: {query.user_id} | "
            f"session_id: {session_id} | "
            f"query: {query.query} | "
            f"enable_rag: {effective_rag} | "
            f"enable_cot: {effective_cot} | "
            f"enable_tot: {effective_tot} | "
            f"enable_got: {effective_got} | "
            f"use_memory: {effective_memory}"
        )

        # 🆕 L5a: Try L1 cache first (in-process, <1ms)
        cache_hit = False
        cache_layer = None
        response_text = None

        if l1_cache:
            response_text = l1_cache.get(
                query=query.query,
                user_id=query.user_id,
                enable_rag=effective_rag,
                enable_cot=effective_cot,
            )
            if response_text:
                cache_hit = True
                cache_layer = "L1"
                logger.info(f"💾 L1 cache HIT | user_id: {query.user_id}")

        # 🆕 L5a: Try L2 cache if L1 miss (Redis, <10ms)
        if not cache_hit and l2_cache:
            response_text = await l2_cache.get(
                query=query.query,
                user_id=query.user_id,
                enable_rag=effective_rag,
                enable_cot=effective_cot,
            )
            if response_text:
                cache_hit = True
                cache_layer = "L2"
                logger.info(f"💾 L2 cache HIT | user_id: {query.user_id}")

                # 🆕 L5a: Backfill L1 cache on L2 hit
                if l1_cache:
                    l1_cache.set(
                        query=query.query,
                        user_id=query.user_id,
                        enable_rag=effective_rag,
                        enable_cot=effective_cot,
                        response=response_text,
                    )
                    logger.debug(f"💾 L1 cache BACKFILL from L2")

        # 🆕 Level 4: Initialize multi-agent metadata
        agents_invoked: list[str] = []
        final_agent_level: str | None = None
        detected_complexity: str | None = None
        execution_time_ms: float | None = None

        # 🆕 L5a: Cache miss - invoke agent (L3 Anthropic caching automatic)
        if not cache_hit:
            logger.info(f"❌ Cache MISS (L1+L2) | user_id: {query.user_id} | Invoking agent...")

            import time
            start_time = time.time()

            # 🆕 Level 3a: Load memory context if enabled
            memory_context = None
            if effective_memory and query.user_id and memory_manager:
                # Use injected memory manager (initialized at startup via DI)
                memory_context = await memory_manager.get_context(
                    user_id=query.user_id,
                    session_id=session_id,
                )
                logger.info(f"🧠 Memory context loaded for user {query.user_id}")
            elif effective_memory and not memory_manager:
                logger.warning("⚠️  Memory requested but manager not initialized")

            # 🆕 v0.6.0: AUTO-ROUTING based on query intent classification
            routing_decision = classify_query(
                query=query.query,
                memory_context=memory_context,
            )

            logger.info(
                f"🎯 Auto-routing decision | "
                f"tier: {routing_decision.tier.value} | "
                f"agent_level: {routing_decision.agent_level} | "
                f"confidence: {routing_decision.confidence:.2f} | "
                f"rule: {routing_decision.primary_signal}"
            )

            # Route to appropriate workflow based on classification
            if routing_decision.tier == QueryTier.EMERGENCY:
                # EMERGENCY: L4C 15-agent with HITL potential
                final_agent_level = "l4c"
                detected_complexity = "emergency"
                workflow_result = await invoke_workflow_v2(
                    query=query.query,
                    user_id=query.user_id,
                    session_id=session_id,
                    memory_context=memory_context,
                    use_supervisor=True,
                )
                response_text = workflow_result.get("final_response", "")
                agents_invoked = workflow_result.get("agents_invoked", [])

            elif routing_decision.tier == QueryTier.COMPLEX:
                # COMPLEX: L4B 8-agent with supervisor
                if workflow_l4b:
                    final_agent_level = "l4b"
                    detected_complexity = "complex"
                    workflow_result = await workflow_l4b.ainvoke(
                        {
                            "query": query.query,
                            "user_id": query.user_id,
                            "session_id": session_id,
                            "memory_context": memory_context,
                            "agent_responses": [],
                            "agents_invoked": [],
                        },
                        config={"configurable": {"thread_id": session_id}},
                    )
                    response_text = workflow_result.get("final_response", "")
                    agents_invoked = workflow_result.get("agents_invoked", [])
                elif workflow_l4a:
                    # Fallback to L4A if L4B not available
                    logger.warning("⚠️  L4B workflow not available, using L4A")
                    final_agent_level = "l4a"
                    detected_complexity = "complex"
                    workflow_result = await workflow_l4a.ainvoke(
                        {
                            "query": query.query,
                            "user_id": query.user_id,
                            "session_id": session_id,
                            "memory_context": memory_context,
                            "agent_responses": [],
                            "agents_invoked": [],
                        },
                        config={"configurable": {"thread_id": session_id}},
                    )
                    response_text = workflow_result.get("final_response", "")
                    agents_invoked = workflow_result.get("agents_invoked", [])
                else:
                    # Fallback to basic agent
                    logger.warning("⚠️  No multi-agent workflows available, using basic agent")
                    final_agent_level = "basic"
                    detected_complexity = "complex"
                    response_text = await _invoke_basic_agent(
                        query=query.query,
                        effective_rag=effective_rag,
                        effective_cot=effective_cot,
                        effective_memory=effective_memory,
                        effective_tot=effective_tot,
                        effective_got=effective_got,
                        memory_context=memory_context,
                    )
                    agents_invoked = ["weather_agent"]

            elif routing_decision.tier == QueryTier.STANDARD:
                # STANDARD: L4A 3-agent system
                if workflow_l4a:
                    final_agent_level = "l4a"
                    detected_complexity = "standard"
                    workflow_result = await workflow_l4a.ainvoke(
                        {
                            "query": query.query,
                            "user_id": query.user_id,
                            "session_id": session_id,
                            "memory_context": memory_context,
                            "agent_responses": [],
                            "agents_invoked": [],
                        },
                        config={"configurable": {"thread_id": session_id}},
                    )
                    response_text = workflow_result.get("final_response", "")
                    agents_invoked = workflow_result.get("agents_invoked", [])
                else:
                    # Fallback to basic agent
                    logger.warning("⚠️  L4A workflow not available, using basic agent")
                    final_agent_level = "basic"
                    detected_complexity = "standard"
                    response_text = await _invoke_basic_agent(
                        query=query.query,
                        effective_rag=effective_rag,
                        effective_cot=effective_cot,
                        effective_memory=effective_memory,
                        effective_tot=effective_tot,
                        effective_got=effective_got,
                        memory_context=memory_context,
                    )
                    agents_invoked = ["weather_agent"]

            else:
                # SIMPLE: Basic single agent (fast, low cost)
                logger.info("🤖 Using BASIC single-agent mode (SIMPLE tier)")
                final_agent_level = "basic"
                detected_complexity = "simple"
                response_text = await _invoke_basic_agent(
                    query=query.query,
                    effective_rag=effective_rag,
                    effective_cot=effective_cot,
                    effective_memory=effective_memory,
                    effective_tot=effective_tot,
                    effective_got=effective_got,
                    memory_context=memory_context,
                )
                agents_invoked = ["weather_agent"]

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"✅ Agent invocation complete | "
                f"agent_level: {final_agent_level} | "
                f"agents: {agents_invoked} | "
                f"execution_ms: {execution_time_ms:.2f}"
            )

            # 🆕 L5a: Write to L1 and L2 caches
            if l1_cache:
                l1_cache.set(
                    query=query.query,
                    user_id=query.user_id,
                    enable_rag=effective_rag,
                    enable_cot=effective_cot,
                    response=response_text,
                )
                logger.debug(f"💾 L1 cache WRITE")

            if l2_cache:
                await l2_cache.set(
                    query=query.query,
                    user_id=query.user_id,
                    enable_rag=effective_rag,
                    enable_cot=effective_cot,
                    response=response_text,
                )
                logger.debug(f"💾 L2 cache WRITE")

            # 🆕 Level 3a: Save interaction to memory if enabled
            if effective_memory and query.user_id and memory_manager:
                await memory_manager.save_interaction(
                    user_id=query.user_id,
                    session_id=session_id,
                    query=query.query,
                    response=response_text,
                )
                logger.info(f"🧠 Interaction saved to memory for user {query.user_id}")

        logger.info(
            f"Weather query successful | "
            f"user_id: {query.user_id} | "
            f"cache_hit: {cache_hit} | "
            f"cache_layer: {cache_layer or 'MISS'} | "
            f"agent_level: {final_agent_level} | "  # 🆕 Level 4
            f"agents: {agents_invoked} | "  # 🆕 Level 4
            f"response_length: {len(response_text)}"
        )

        # Set cache_layer to "MISS" if still None (cache miss case)
        if cache_layer is None:
            cache_layer = "MISS"

        return WeatherResponse(
            response=response_text,
            user_id=query.user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            # 🆕 Level 4: Multi-agent metadata
            agents_invoked=agents_invoked,
            agent_level=final_agent_level,
            query_complexity=detected_complexity,
            execution_time_ms=execution_time_ms,
            # L5a: Cache metadata
            cache_hit=cache_hit,
            cache_layer=cache_layer,
        )

    except Exception as e:
        logger.error(
            f"Weather query failed | "
            f"user_id: {query.user_id} | "
            f"error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process weather query: {str(e)}"
        )


@app.post(
    "/weather/hurricane/alert",
    response_model=HurricaneAlertResponse,
    status_code=status.HTTP_200_OK,
    summary="Create hurricane alert",
    description="Create a hurricane alert that may require human approval (Cat 3+) before being sent",
    tags=["Hurricane Alerts"]
)
async def create_hurricane_alert(alert: HurricaneAlertRequest):
    """Create hurricane alert (may require human approval).

    This endpoint creates a hurricane alert that follows the HITL approval workflow:
    - Category 1-2: Auto-approved and sent immediately
    - Category 3-5: Requires human approval before sending

    Example Request (Cat 2 - auto-approved):
        POST /weather/hurricane/alert
        {
            "category": 2,
            "message": "Category 2 Hurricane Julia approaching with 95 mph winds",
            "thread_id": "alert-uuid"
        }

    Example Response (auto-approved):
        {
            "status": "sent",
            "thread_id": "alert-uuid"
        }

    Example Request (Cat 4 - requires approval):
        POST /weather/hurricane/alert
        {
            "category": 4,
            "message": "Category 4 Hurricane Ida approaching with 140 mph winds",
            "thread_id": "alert-uuid"
        }

    Example Response (pending approval):
        {
            "status": "pending_approval",
            "thread_id": "alert-uuid"
        }

    Args:
        alert: HurricaneAlertRequest with category, message, and thread_id

    Returns:
        HurricaneAlertResponse with status and thread_id

    Raises:
        HTTPException: 500 if workflow execution fails
    """
    config = {"configurable": {"thread_id": alert.thread_id}}

    try:
        logger.info(
            f"Hurricane alert received | "
            f"category: {alert.category} | "
            f"thread_id: {alert.thread_id}"
        )

        # Invoke workflow with hurricane alert data
        result = workflow.invoke({
            "user_id": "system",  # System-generated alert
            "session_id": alert.thread_id,
            "current_query": f"Category {alert.category} hurricane",
            "current_step": "input",
            "approved": False,
            "hurricane_category": alert.category,
            "alert_message": alert.message
        }, config)

        # Check if workflow is interrupted (pending human approval)
        if "__interrupt__" in result:
            logger.warning(
                f"Hurricane alert pending approval | "
                f"category: {alert.category} | "
                f"thread_id: {alert.thread_id}"
            )
            return HurricaneAlertResponse(
                status="pending_approval",
                thread_id=alert.thread_id,
                category=alert.category,
                message=alert.message
            )

        # Alert was auto-approved or completed
        final_status = "sent" if result.get("approved") else "cancelled"

        logger.info(
            f"Hurricane alert completed | "
            f"status: {final_status} | "
            f"category: {alert.category} | "
            f"thread_id: {alert.thread_id}"
        )

        return HurricaneAlertResponse(
            status=final_status,
            thread_id=alert.thread_id,
            category=alert.category,
            message=alert.message
        )

    except Exception as e:
        logger.error(
            f"Hurricane alert failed | "
            f"category: {alert.category} | "
            f"thread_id: {alert.thread_id} | "
            f"error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create hurricane alert: {str(e)}"
        )


@app.post(
    "/weather/hurricane/approve/{thread_id}",
    response_model=HurricaneApprovalResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve or reject hurricane alert",
    description="Approve or reject a pending hurricane alert (Cat 3+) after human review",
    tags=["Hurricane Alerts"]
)
async def approve_hurricane_alert(
    thread_id: str,
    approval: HurricaneApprovalRequest
):
    """Approve or reject pending hurricane alert.

    This endpoint is called by human reviewers to approve or reject
    Category 3+ hurricane alerts that are pending in the workflow.

    Example Request (approve):
        POST /weather/hurricane/approve/alert-uuid
        {
            "approved": true
        }

    Example Response:
        {
            "status": "sent",
            "thread_id": "alert-uuid"
        }

    Example Request (reject):
        POST /weather/hurricane/approve/alert-uuid
        {
            "approved": false
        }

    Example Response:
        {
            "status": "cancelled",
            "thread_id": "alert-uuid"
        }

    Args:
        thread_id: Thread ID of the pending alert workflow
        approval: HurricaneApprovalRequest with approval decision

    Returns:
        HurricaneApprovalResponse with final status

    Raises:
        HTTPException: 404 if thread_id not found
        HTTPException: 500 if workflow resume fails
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info(
            f"Hurricane approval received | "
            f"thread_id: {thread_id} | "
            f"approved: {approval.approved}"
        )

        # Resume workflow with approval decision
        result = workflow.invoke(
            Command(resume={"approved": approval.approved}),
            config
        )

        # Determine final status
        final_status = "sent" if result.get("approved") else "cancelled"

        logger.info(
            f"Hurricane approval completed | "
            f"status: {final_status} | "
            f"thread_id: {thread_id}"
        )

        return HurricaneApprovalResponse(
            status=final_status,
            thread_id=thread_id
        )

    except GraphInterrupt as e:
        # Thread expired, completed, or not found
        logger.warning(
            f"Hurricane approval - thread not found | "
            f"thread_id: {thread_id} | "
            f"error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found, expired, or already completed"
        )
    except Exception as e:
        logger.error(
            f"Hurricane approval failed | "
            f"thread_id: {thread_id} | "
            f"error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process hurricane approval: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the Weather AI Agent service is healthy and operational",
    tags=["System"]
)
async def health_check():
    """Health check endpoint.

    Returns service health status and current implementation level.

    Example Response:
        {
            "status": "healthy",
            "level": "3a+L5a",
            "timestamp": "2025-12-03T16:20:00Z"
        }

    Returns:
        HealthCheckResponse with status, level, and timestamp
    """
    logger.debug("Health check requested")

    return HealthCheckResponse(
        status="healthy",
        level="L4+L5a",  # Updated to Level 4 Multi-Agent + L5a Caching
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get(
    "/cache/stats",
    status_code=status.HTTP_200_OK,
    summary="Cache statistics",
    description="Get cache performance statistics for L1 (memory), L2 (Redis), and L3 (Anthropic)",
    tags=["System"]
)
async def cache_stats(
    l1_cache: QueryCache | None = Depends(get_l1_cache),
    l2_cache: RedisQueryCache | None = Depends(get_l2_cache),
    l3_metrics: AnthropicCacheMetrics | None = Depends(get_l3_metrics),
):
    """Get cache statistics for all three cache layers.

    Returns hit rates, misses, and cost savings for:
    - L1: In-process LRU cache
    - L2: Redis distributed cache
    - L3: Anthropic prompt cache

    Example Response:
        {
            "enabled": true,
            "l1": {
                "enabled": true,
                "hits": 150,
                "misses": 850,
                "evictions": 50,
                "hit_rate": 0.15,
                "size": 950,
                "max_size": 1000,
                "ttl_seconds": 300
            },
            "l2": {
                "enabled": true,
                "hits": 300,
                "misses": 700,
                "errors": 5,
                "hit_rate": 0.30,
                "ttl_seconds": 1800
            },
            "l3": {
                "enabled": true,
                "request_count": 500,
                "total_cache_read": 3500000,
                "total_cache_creation": 500000,
                "total_input": 1000000,
                "aggregate_hit_rate": 70.0,
                "aggregate_savings_percent": 63.0
            },
            "summary": {
                "total_requests": 1000,
                "total_cache_hits": 450,
                "overall_hit_rate": 0.45,
                "estimated_cost_savings_percent": 60.0
            }
        }

    Returns:
        dict with L1, L2, L3 statistics and summary
    """
    logger.debug("Cache stats requested")

    stats = {
        "enabled": cache_config.CACHE_ENABLED,
        "l1": None,
        "l2": None,
        "l3": None,
        "summary": None,
    }

    # L1 stats
    if l1_cache:
        stats["l1"] = {
            "enabled": True,
            **l1_cache.get_stats()
        }
    else:
        stats["l1"] = {"enabled": False}

    # L2 stats
    if l2_cache:
        stats["l2"] = {
            "enabled": True,
            **await l2_cache.get_stats()
        }
    else:
        stats["l2"] = {"enabled": False}

    # L3 stats
    if l3_metrics:
        stats["l3"] = {
            "enabled": True,
            **l3_metrics.get_stats()
        }
    else:
        stats["l3"] = {"enabled": False}

    # Calculate summary
    if cache_config.CACHE_ENABLED:
        total_requests = 0
        total_cache_hits = 0

        # L1 contribution
        if l1_cache:
            l1_stats = l1_cache.get_stats()
            total_requests += l1_stats["hits"] + l1_stats["misses"]
            total_cache_hits += l1_stats["hits"]

        # L2 contribution (only misses from L1 reach L2)
        if l2_cache:
            l2_stats = await l2_cache.get_stats()
            # L2 hits are already counted, just add L2-specific hits
            total_cache_hits += l2_stats["hits"]

        # Overall hit rate
        overall_hit_rate = (
            total_cache_hits / total_requests if total_requests > 0 else 0.0
        )

        # Estimated cost savings (simplified model)
        # L1/L2 hits: 100% savings
        # L3 hits: 90% savings (from L3 stats)
        l1_l2_savings = (total_cache_hits / total_requests * 100) if total_requests > 0 else 0.0
        l3_savings = stats["l3"].get("aggregate_savings_percent", 0.0) if stats["l3"]["enabled"] else 0.0

        # Weighted average (L1/L2 get priority, L3 fills gaps)
        estimated_savings = min(l1_l2_savings + (l3_savings * 0.5), 90.0)

        stats["summary"] = {
            "total_requests": total_requests,
            "total_cache_hits": total_cache_hits,
            "overall_hit_rate": round(overall_hit_rate, 3),
            "estimated_cost_savings_percent": round(estimated_savings, 1),
        }

    return stats


@app.post(
    "/cache/clear",
    status_code=status.HTTP_200_OK,
    summary="Clear all cache layers",
    description="Flush L1 (in-process memory) and L2 (Redis) caches. Useful for testing and development. L3 (Anthropic) cache cannot be cleared programmatically.",
    tags=["System"]
)
async def cache_clear(
    l1_cache: QueryCache | None = Depends(get_l1_cache),
    l2_cache: RedisQueryCache | None = Depends(get_l2_cache),
):
    """Clear all cache layers (L1 and L2).

    Clears:
    - L1: In-process LRU cache (Python dict)
    - L2: Redis distributed cache (FLUSHDB)

    Note: L3 (Anthropic prompt cache) cannot be cleared programmatically.
    It expires automatically based on cache_control parameters (5 minutes).

    Example Response:
        {
            "status": "success",
            "cleared_layers": ["L1", "L2"],
            "l1": {
                "cleared": true,
                "previous_size": 150
            },
            "l2": {
                "cleared": true,
                "previous_keys": 42
            },
            "timestamp": "2025-12-09T20:30:00Z"
        }

    Returns:
        dict with status and details of cleared caches
    """
    logger.info("🗑️  Cache clear requested")

    result = {
        "status": "success",
        "cleared_layers": [],
        "l1": None,
        "l2": None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Clear L1 cache
    if l1_cache:
        previous_size = len(l1_cache.cache)
        l1_cache.cache.clear()  # Python dict.clear()
        result["cleared_layers"].append("L1")
        result["l1"] = {
            "cleared": True,
            "previous_size": previous_size
        }
        logger.info(f"✅ L1 cache cleared (was: {previous_size} entries)")
    else:
        result["l1"] = {
            "cleared": False,
            "reason": "L1 cache not enabled"
        }
        logger.warning("⚠️  L1 cache not available")

    # Clear L2 Redis cache
    if l2_cache:
        try:
            # Get count before clearing
            keys_pattern = f"{l2_cache.key_prefix}*"
            keys = await l2_cache.client.keys(keys_pattern)
            previous_keys = len(keys) if keys else 0

            # Clear only weather cache keys (not entire Redis DB)
            if keys:
                await l2_cache.client.delete(*keys)

            result["cleared_layers"].append("L2")
            result["l2"] = {
                "cleared": True,
                "previous_keys": previous_keys
            }
            logger.info(f"✅ L2 cache cleared (was: {previous_keys} keys)")
        except Exception as e:
            result["l2"] = {
                "cleared": False,
                "error": str(e)
            }
            logger.error(f"❌ L2 cache clear failed: {e}")
            # Don't fail the whole request, just log the error
    else:
        result["l2"] = {
            "cleared": False,
            "reason": "L2 cache not enabled"
        }
        logger.warning("⚠️  L2 cache not available")

    # Update status based on results
    if not result["cleared_layers"]:
        result["status"] = "no_caches_available"
    elif len(result["cleared_layers"]) < 2:
        result["status"] = "partial"

    logger.info(f"🗑️  Cache clear complete: {result['cleared_layers']}")
    return result


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # For development
        log_level="info"
    )
