"""Weather AI agents module.

This module provides ReAct pattern agents for weather queries and hurricane alerts.

Level 4a Additions:
- TriageAgent: Query classification and routing
- HurricaneSpecialistAgent: Domain expert for hurricane forecasts
- AlertManagerAgent: Weather alert generation and delivery

Level 4b Additions:
- SupervisorAgent: Workflow orchestration and agent coordination
- ReflectionAgent: Self-critique and response improvement
- CritiqueAgent: Generator → Critic → Refiner workflow
- ForecasterAgent: General weather forecasting specialist
- HistoricalAnalystAgent: Historical weather pattern analysis
- ResearchAgent: Deep data retrieval and research

Level 4c Additions:
- MetaPromptAgent: Dynamic prompt generation for other agents
- SelfHealingAgent: Automatic retry, fallback, and circuit breakers
- DebateAgent: Multi-proposal evaluation and selection
- EmergencyResponseAgent: Urgent weather emergency handling
- ClimateAnalystAgent: Long-term climate pattern analysis
- PersonalizationAgent: User preference-based customization
- CircuitBreaker: Resilience pattern for agent failures
"""

from backend.src.agents.state import WeatherAgentState
from backend.src.agents.weather_agent import create_weather_agent, query_weather
from backend.src.agents.prompts import WEATHER_ASSISTANT_SYSTEM_PROMPT

# Level 4a: Multi-Agent System
from backend.src.agents.triage_agent import TriageAgent
from backend.src.agents.hurricane_specialist import HurricaneSpecialistAgent
from backend.src.agents.alert_manager import AlertManagerAgent, AlertSeverity, AlertChannel

# Level 4b: 8-Agent Orchestration System
from backend.src.agents.supervisor_agent import SupervisorAgent, AgentCapabilityRegistry
from backend.src.agents.reflection_agent import ReflectionAgent
from backend.src.agents.critique_agent import CritiqueAgent
from backend.src.agents.forecaster_agent import ForecasterAgent
from backend.src.agents.historical_agent import HistoricalAnalystAgent
from backend.src.agents.research_agent import ResearchAgent

# Level 4c: 15-Agent Production System
from backend.src.agents.meta_prompt_agent import MetaPromptAgent, create_meta_prompt_agent
from backend.src.agents.self_healing_agent import (
    SelfHealingAgent,
    create_self_healing_agent,
    self_healing_agent,
)
from backend.src.agents.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_registry,
    get_circuit_status,
    get_system_health,
)
from backend.src.agents.debate_agent import DebateAgent, Proposal, DebateResult, create_debate_agent
from backend.src.agents.emergency_agent import EmergencyResponseAgent, create_emergency_agent
from backend.src.agents.climate_agent import ClimateAnalystAgent, create_climate_agent
from backend.src.agents.personalization_agent import (
    PersonalizationAgent,
    UserProfile,
    create_personalization_agent,
)

__all__ = [
    # Level 1-3: Core agents
    "WeatherAgentState",
    "create_weather_agent",
    "query_weather",
    "WEATHER_ASSISTANT_SYSTEM_PROMPT",
    # Level 4a: Multi-agent system
    "TriageAgent",
    "HurricaneSpecialistAgent",
    "AlertManagerAgent",
    "AlertSeverity",
    "AlertChannel",
    # Level 4b: 8-Agent orchestration system
    "SupervisorAgent",
    "AgentCapabilityRegistry",
    "ReflectionAgent",
    "CritiqueAgent",
    "ForecasterAgent",
    "HistoricalAnalystAgent",
    "ResearchAgent",
    # Level 4c: 15-Agent production system
    "MetaPromptAgent",
    "create_meta_prompt_agent",
    "SelfHealingAgent",
    "create_self_healing_agent",
    "self_healing_agent",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
    "circuit_registry",
    "get_circuit_status",
    "get_system_health",
    "DebateAgent",
    "Proposal",
    "DebateResult",
    "create_debate_agent",
    "EmergencyResponseAgent",
    "create_emergency_agent",
    "ClimateAnalystAgent",
    "create_climate_agent",
    "PersonalizationAgent",
    "UserProfile",
    "create_personalization_agent",
]
