"""Multi-Agent Orchestration for Level 4a, Level 4b, and Level 4c.

This module provides LangGraph StateGraph-based orchestration for the
Weather AI multi-agent system.

Components:
- multi_agent_workflow: StateGraph workflow definition and compilation
- parallel_executor: Parallel agent execution utilities
- load_aware_router: Load-aware agent routing with cost optimization
- cost_optimizer: Budget management and cost tracking
- Routing logic: Confidence-based routing between agents
- Memory integration: PostgreSQL checkpointing for conversation persistence

Level 4a Architecture (3-Agent System):
    User Query
        ↓
    Triage Agent (Router)
        ├─→ Hurricane Specialist Agent (Domain Expert)
        ├─→ Alert Manager Agent (Notifications)
        └─→ Direct Response (Simple queries)
            ↓
    Response Synthesis
        ↓
    User

Level 4b Architecture (8-Agent Orchestration):
    User Query
        ↓
    Supervisor Agent (Orchestrator)
        ↓
    Triage Agent (Classification)
        ↓
    Parallel Agent Execution (up to 4 concurrent):
        ├─→ Hurricane Specialist
        ├─→ Forecaster Agent
        ├─→ Historical Analyst
        ├─→ Research Agent
        └─→ Alert Manager
            ↓
    Reflection/Critique (Quality Assurance)
        ↓
    Response Synthesis
        ↓
    User

Level 4c Architecture (15-Agent Production System):
    User Query
        ↓
    Meta-Prompt Agent (Dynamic Prompt Generation)
        ↓
    Load-Aware Router (Cost + Latency Optimization)
        ↓
    Supervisor Agent (Orchestration)
        ↓
    Parallel Agent Execution (15 agents available):
        ├─→ Specialist Agents (Hurricane, Forecaster, Historical, Research)
        ├─→ Production Agents (Emergency, Climate, Personalization)
        ├─→ Quality Agents (Reflection, Critique, Debate)
        └─→ System Agents (Self-Healing, Circuit Breaker)
            ↓
    Response Synthesis + Personalization
        ↓
    User
"""

# Level 4a: Multi-Agent Workflow
# Level 4c: Cost Optimization
from backend.src.orchestration.cost_optimizer import (
    AGENT_TOKEN_ESTIMATES,
    TOKEN_COSTS,
    CostOptimizer,
)

# Level 4c: Load-Aware Routing
from backend.src.orchestration.load_aware_router import (
    DEFAULT_AGENT_METRICS,
    AgentMetrics,
    CostTier,
    LoadAwareRouter,
    load_aware_router,
)

# Level 4b: 8-Agent Orchestration Workflow
from backend.src.orchestration.multi_agent_workflow import (
    compile_level4b_workflow,
    compile_workflow,
    create_level4b_workflow,
    create_multi_agent_workflow,
    forecaster_node,
    historical_analyst_node,
    invoke_workflow_v2,
    reflection_node,
    research_node,
    route_after_specialist,
    route_after_triage,
    supervisor_node,
)

# Level 4b: Parallel Execution Utilities
from backend.src.orchestration.parallel_executor import (
    ParallelExecutor,
    ParallelResult,
    TaskResult,
    execute_parallel,
    execute_with_fallback,
)

__all__ = [
    # Level 4a: Multi-Agent Workflow
    "create_multi_agent_workflow",
    "compile_workflow",
    "route_after_triage",
    "route_after_specialist",
    # Level 4b: 8-Agent Orchestration Workflow
    "create_level4b_workflow",
    "compile_level4b_workflow",
    "invoke_workflow_v2",
    "supervisor_node",
    "forecaster_node",
    "historical_analyst_node",
    "research_node",
    "reflection_node",
    # Level 4b: Parallel Execution Utilities
    "ParallelResult",
    "TaskResult",
    "execute_parallel",
    "execute_with_fallback",
    "ParallelExecutor",
    # Level 4c: Load-Aware Routing
    "LoadAwareRouter",
    "AgentMetrics",
    "CostTier",
    "DEFAULT_AGENT_METRICS",
    "load_aware_router",
    # Level 4c: Cost Optimization
    "CostOptimizer",
    "TOKEN_COSTS",
    "AGENT_TOKEN_ESTIMATES",
]
