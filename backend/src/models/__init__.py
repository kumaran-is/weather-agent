"""Pydantic Models for Weather AI Agent API - SINGLE SOURCE OF TRUTH

⚠️ CRITICAL ARCHITECTURAL RULES (STRICTLY ENFORCED FOR ALL LEVELS):

1. **SINGLE SOURCE OF TRUTH**:
   - ALL schemas/models MUST live in backend/src/models/ ONLY
   - NEVER create backend/src/api/schemas.py or any other model location
   - NEVER scatter models across multiple directories

2. **INCREMENTAL ENHANCEMENT OVER NEW MODELS**:
   - ALWAYS enhance existing models when possible
   - Create new models ONLY when genuinely needed (different domain/purpose)
   - Example: Add fields to WeatherResponse, NOT create WeatherResponseV2

3. **MINIMAL MODELS PATTERN**:
   - Keep only ACTIVELY USED models (delete unused immediately)
   - One domain per file (weather, hurricane, health, memory, agents, etc.)
   - No premature model creation (add when needed in that level, not before)

4. **PROGRESSIVE ENHANCEMENT STRATEGY**:
   Level 2 → Level 3: Enhance WeatherQuery with memory context, NOT create new model
   Level 3 → Level 4: Enhance models with agent fields, NOT create new models
   Level 4 → Level 5: Enhance models with observability, NOT create new models

When to Create NEW Model vs Enhance Existing:
✅ CREATE NEW when:
   - Genuinely different domain (weather vs hurricane vs memory)
   - Different lifecycle (request vs response vs internal state)
   - Different user persona (end-user vs admin vs system)

❌ ENHANCE EXISTING when:
   - Adding optional fields to existing model
   - Supporting new feature in same domain
   - Incremental improvements (L2→L3→L4)
   - Backward-compatible changes

Current Models:
- weather.py: WeatherQuery, WeatherResponse (2 models)
- hurricane.py: HurricaneAlertRequest, HurricaneAlertResponse, HurricaneApprovalRequest, HurricaneApprovalResponse (4 models)
- health.py: HealthCheckResponse (1 model)
- multi_agent.py: AgentRole, QueryComplexity, RoutingDecision, AgentResponse, AgentState, MultiAgentState (6 models - Level 4a)

Total: 13 models, 100% utilized, 0% unused

Future Enhancements (Level 3-6) - ENHANCE, DON'T PROLIFERATE:
- Level 3: ENHANCE WeatherQuery/Response with session_id, memory_context (NOT new models)
           ADD memory.py ONLY IF truly new domain (ConversationContext, UserProfile)
- Level 4: ENHANCE existing models with agent coordination fields (NOT new models)
           ADD agents.py ONLY IF truly new domain (AgentState, CoordinationMessage)
- Level 5: ENHANCE existing models with observability metadata (NOT new models)
           ADD observability.py ONLY IF truly new domain (MetricsSnapshot, TraceEvent)
- Level 6: ENHANCE existing models with evolution tracking (NOT new models)
           ADD evolution.py ONLY IF truly new domain (PerformanceStats, ExperimentConfig)

Design Principles (STRICTLY ENFORCED):
1. Single Source of Truth: ALL models in backend/src/models/ ONLY
2. Incremental Enhancement: Enhance existing > Create new
3. Domain Organization: One file per distinct domain
4. Minimal Code: Keep ONLY actively used models (0% unused)
5. Progressive Addition: Add when needed in that level, not prematurely
"""

# Weather models (v0.6.0: AgentLevel kept for internal use, not in API)
# Health check models (Level 5c: Comprehensive service health)
from backend.src.models.health import (
    HealthCheckResponse,
    ServiceHealth,
    ServicesHealth,
)

# Hurricane alert models
from backend.src.models.hurricane import (
    HurricaneAlertRequest,
    HurricaneAlertResponse,
    HurricaneApprovalRequest,
    HurricaneApprovalResponse,
)

# Multi-agent models (Level 4a)
from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    AgentState,
    MultiAgentState,
    QueryComplexity,
    RoutingDecision,
)
from backend.src.models.weather import (
    AgentLevel,  # Internal use only - API uses auto-routing now
    EvaluationScores,  # Level 5b: 4-pillar evaluation scores
    WeatherQuery,
    WeatherResponse,
)

# Tool registry models (Level 7 - migrated to langgraph-bigtool)
from backend.src.registry.bigtool_registry import (
    BigtoolStats,
    ToolCategory,
    ToolMetadata,
)

__all__ = [
    # Weather (v0.6.0: AgentLevel for internal use, API uses auto-routing)
    "AgentLevel",  # Internal - not exposed in API
    "EvaluationScores",  # Level 5b: 4-pillar evaluation scores
    "WeatherQuery",
    "WeatherResponse",
    # Hurricane
    "HurricaneAlertRequest",
    "HurricaneAlertResponse",
    "HurricaneApprovalRequest",
    "HurricaneApprovalResponse",
    # Health (Level 5c: Comprehensive service health)
    "ServiceHealth",
    "ServicesHealth",
    "HealthCheckResponse",
    # Multi-Agent (Level 4a)
    "AgentRole",
    "QueryComplexity",
    "RoutingDecision",
    "AgentResponse",
    "AgentState",
    "MultiAgentState",
    # Tool Registry (Level 7 - langgraph-bigtool)
    "BigtoolStats",
    "ToolCategory",
    "ToolMetadata",
]
