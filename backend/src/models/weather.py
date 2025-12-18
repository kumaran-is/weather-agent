"""Weather Query and Response Models

This module defines Pydantic models for weather-related API requests and responses.

PROGRESSIVE ENHANCEMENT PATTERN (from Level 2 learnings):
- Level 2: Basic query model with user tracking
- Level 3a: Add memory support (use_memory, enable_rag, enable_cot)
- Level 3b: Add advanced reasoning (enable_tot, enable_got)
- Level 4: Multi-agent orchestration (AUTO-ROUTED based on query intent)

AUTO-ROUTING (v0.6.0+):
- Removed: use_multi_agent, agent_level flags
- Added: Intelligent query classification based on intent
- Routing is automatic - no user configuration needed

CRITICAL RULE: ENHANCE existing models, NEVER create versioned models (WeatherQueryV2)
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


# NOTE: AgentLevel enum kept for internal use and backward compatibility
# It is NOT exposed in the API anymore - routing is automatic
class AgentLevel(str, Enum):
    """Agent orchestration level for Level 4 multi-agent system.

    NOTE: This enum is for INTERNAL use only. The API no longer exposes
    agent level selection - routing is automatic based on query intent.

    Levels:
    - BASIC: Single agent (Level 1-3 behavior)
    - L4A: 3-agent system (Triage, Hurricane Specialist, Alert Manager)
    - L4B: 8-agent system with Supervisor orchestration and parallel execution
    - L4C: 15-agent production system with debate, reflection, and advanced features
    """

    BASIC = "basic"  # Single agent, Level 1-3 behavior
    L4A = "l4a"      # 3-agent: Triage, Hurricane Specialist, Alert Manager
    L4B = "l4b"      # 8-agent: + Supervisor, Forecaster, Historical, Research, Reflection
    L4C = "l4c"      # 15-agent: + Debate, Meta-Prompt, Emergency, Climate, Personalization, etc.


class WeatherQuery(BaseModel):
    """Request model for weather query endpoint.

    Progressive enhancement across levels:
    - Level 2: query, user_id, session_id
    - Level 3a: + location, enable_rag, enable_cot, use_memory
    - Level 3b: + enable_tot, enable_got
    - Level 4: AUTO-ROUTING (no explicit flags needed)

    AUTO-ROUTING (v0.6.0+):
    Multi-agent routing is now AUTOMATIC based on query intent:
    - Simple weather queries → Basic agent
    - Hurricane/storm queries → L4A (3-agent)
    - Complex analysis queries → L4B (8-agent)
    - Emergency/safety queries → L4C (15-agent with HITL)

    NOTE: Feature flags (enable_rag, enable_cot, enable_tot, enable_got, use_memory)
    can still be set via QUERY PARAMETERS for fine-tuning.

    Clean example for Swagger UI:
        {
            "query": "What's the weather in London?",
            "user_id": "user123",
            "session_id": "session456"
        }
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What's the weather in London?",
                    "user_id": "test_simple_001",
                    "session_id": "session_simple"
                },
                {
                    "query": "Is Hurricane Milton going to hit Tampa?",
                    "user_id": "test_hurricane_001",
                    "session_id": "session_hurricane"
                },
                {
                    "query": "Should I evacuate from Miami Beach?",
                    "user_id": "test_evac_001",
                    "session_id": "session_evac"
                },
                {
                    "query": "Compare Hurricane Milton to historical hurricanes that hit Tampa Bay",
                    "user_id": "test_complex_001",
                    "session_id": "session_complex"
                },
                {
                    "query": "Category 4 hurricane making landfall in 6 hours, should I evacuate?",
                    "user_id": "test_emergency_001",
                    "session_id": "session_emergency"
                },
                {
                    "query": "What hurricane safety precautions should I take?",
                    "user_id": "test_rag_001",
                    "session_id": "session_rag",
                    "enable_rag": True
                },
                {
                    "query": "What was the weather I asked about earlier?",
                    "user_id": "test_memory_001",
                    "session_id": "session_memory",
                    "use_memory": True
                }
            ]
        }
    }

    # Core query (Level 2)
    query: str = Field(
        ...,
        description="Weather question from the user",
        min_length=1,
        max_length=500,
        examples=[
            "What's the weather in London?",
            "Will it rain tomorrow in Seattle?",
            "Is Hurricane Milton going to hit Tampa?",
            "Should I evacuate?",
        ],
    )

    # User tracking (Level 2)
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        examples=["user123", "uuid-here"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for conversation context",
        examples=["session456", "uuid-here"],
    )

    # Level 3a enhancements (PROGRESSIVE - added fields)
    location: str | None = Field(
        default=None,
        description="Optional explicit location (if not in query)",
        examples=["London", "Seattle, WA", "90210"],
    )
    enable_rag: bool = Field(
        default=True,
        description="Enable RAG for historical weather knowledge retrieval",
    )
    enable_cot: bool = Field(
        default=False,
        description="Enable Chain-of-Thought reasoning for complex queries",
    )
    # Level 3b enhancements (PROGRESSIVE - added fields)
    enable_tot: bool = Field(
        default=False,
        description="Enable Tree of Thoughts reasoning for multi-path exploration",
    )
    enable_got: bool = Field(
        default=False,
        description="Enable Graph of Thoughts reasoning for shared sub-problems",
    )
    use_memory: bool = Field(
        default=True,
        description="Enable memory features (short-term + long-term)",
    )

    # NOTE: use_multi_agent and agent_level REMOVED in v0.6.0
    # Routing is now automatic based on query intent classification
    # See backend/src/routing/ for the classification logic


class EvaluationScores(BaseModel):
    """4-pillar evaluation scores for response quality.

    Level 5b: Evaluation Framework
    - Effectiveness (40%): Answer correctness (LLM-as-Judge)
    - Efficiency (20%): Optimal path taken (deterministic)
    - Robustness (20%): Edge case handling (heuristics)
    - Safety (20%): Zero-tolerance safety violations

    Overall score: Weighted average of 4 pillars (0.0-1.0)
    Pass threshold: >=0.80 AND safety == 1.0
    """

    effectiveness: float = Field(ge=0.0, le=1.0, description="Answer correctness score")
    efficiency: float = Field(ge=0.0, le=1.0, description="Optimal path score")
    robustness: float = Field(ge=0.0, le=1.0, description="Edge case handling score")
    safety: float = Field(ge=0.0, le=1.0, description="Safety validation score (1.0=safe)")
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted overall score")
    passed: bool = Field(description="Meets quality threshold (>=0.80, no safety violations)")


class WeatherResponse(BaseModel):
    """Response model for weather query endpoint.

    Progressive enhancement across levels:
    - Level 2: response, user_id, timestamp
    - Level 4: + agents_invoked, agent_level, query_complexity, execution_time_ms
    - Level 5a: + cache_hit, cache_layer (for observability)
    - Level 5b: + evaluation_scores (4-pillar quality assessment)

    Example (cache miss):
        {
            "response": "The weather in London is 15°C and rainy...",
            "user_id": "user123",
            "timestamp": "2025-12-03T16:20:00Z",
            "cache_hit": false,
            "cache_layer": null
        }

    Example (L1 cache hit with evaluation):
        {
            "response": "The weather in London is 15°C and rainy...",
            "user_id": "user123",
            "timestamp": "2025-12-03T16:20:00Z",
            "cache_hit": true,
            "cache_layer": "L1",
            "evaluation_scores": {
                "effectiveness": 0.92,
                "efficiency": 0.85,
                "robustness": 0.88,
                "safety": 1.0,
                "overall_score": 0.90,
                "passed": true
            }
        }
    """

    response: str = Field(
        ...,
        description="Agent's response to the weather query"
    )
    user_id: str = Field(
        ...,
        description="User identifier from the request"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 timestamp in UTC"
    )

    # 🆕 Level 4: Multi-agent orchestration metadata
    agents_invoked: list[str] = Field(
        default_factory=list,
        description="List of agents that participated in generating this response",
        examples=[["triage", "hurricane_specialist", "alert_manager"]],
    )
    agent_level: str | None = Field(
        default=None,
        description="Multi-agent orchestration level used (basic, l4a, l4b, l4c)",
        examples=["l4a", "l4b", "l4c", "basic"],
    )
    query_complexity: str | None = Field(
        default=None,
        description="Detected query complexity (simple, moderate, complex, emergency)",
        examples=["simple", "moderate", "complex", "emergency"],
    )
    execution_time_ms: float | None = Field(
        default=None,
        description="Total execution time in milliseconds for multi-agent processing",
        examples=[150.5, 892.3],
    )

    # 🆕 L5a: Cache observability fields
    cache_hit: bool = Field(
        default=False,
        description="Whether response was served from cache (L1, L2, or L3)",
    )
    cache_layer: str | None = Field(
        default=None,
        description="Cache layer that served the response (L1, L2, or None if cache miss)",
        examples=["L1", "L2", None],
    )

    # 🆕 L5b: Evaluation scores (4-pillar quality assessment)
    evaluation_scores: EvaluationScores | None = Field(
        default=None,
        description="4-pillar evaluation scores (effectiveness, efficiency, robustness, safety)",
    )
