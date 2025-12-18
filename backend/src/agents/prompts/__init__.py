"""Agent prompts for Weather AI Agent Service.

This package contains all prompt templates including:
- Base prompts (Level 1/2/3): Zero-shot, CoT, ToT, GoT prompts
- Triage Agent prompts (Level 4a)
- Hurricane Specialist prompts (Level 4a)
- Alert Manager prompts (Level 4a)
- Supervisor Agent prompts (Level 4b)
- Reflection/Critique Agent prompts (Level 4b)
- Meta-Prompt Agent prompts (Level 4c)
- Debate Agent prompts (Level 4c)
"""

# Base prompts (Level 1/2/3) - Re-exported for backward compatibility
from backend.src.agents.base_prompts import (
    COT_WEATHER_SYSTEM_PROMPT,
    GOT_WEATHER_SYSTEM_PROMPT,
    TOT_WEATHER_SYSTEM_PROMPT,
    WEATHER_ASSISTANT_SYSTEM_PROMPT,
)

# Alert Manager prompts (Level 4a)
from backend.src.agents.prompts.alert_prompts import (
    ALERT_GENERATION_PROMPT,
    ALERT_MANAGER_SYSTEM_PROMPT,
    ALERT_SEVERITY_CLASSIFICATION_PROMPT,
    ALERT_SEVERITY_LEVELS,
    EMAIL_BODY_TEMPLATE,
    EMAIL_SUBJECT_TEMPLATE,
    EVACUATION_ALERT_PROMPT,
    PUSH_NOTIFICATION_TEMPLATE,
    SMS_ALERT_TEMPLATE,
)

# Debate Agent prompts (Level 4c)
from backend.src.agents.prompts.debate_prompts import (
    CONFLICT_RESOLUTION_PROMPT,
    DEBATE_CONFIG,
    DEBATE_SYSTEM_PROMPT,
    JUDGE_EVALUATION_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    MIN_SCORE_FOR_WINNER,
    PROPOSAL_GENERATION_PROMPT,
    SYNTHESIS_PROMPT,
)

# Hurricane Specialist prompts (Level 4a)
from backend.src.agents.prompts.hurricane_prompts import (
    CONFIDENCE_SCORING_GUIDE,
    EMERGENCY_RESPONSE_TEMPLATE,
    HURRICANE_FORECAST_PROMPT,
    HURRICANE_SPECIALIST_SYSTEM_PROMPT,
    HURRICANE_VALIDATION_PROMPT,
    NHC_DATA_TEMPLATE,
    SAFFIR_SIMPSON_SCALE,
    USER_CONTEXT_TEMPLATE,
)

# Meta-Prompt Agent prompts (Level 4c)
from backend.src.agents.prompts.meta_prompt_templates import (
    AGENT_PROMPT_TEMPLATES,
    FEW_SHOT_EXAMPLES,
    META_PROMPT_SYSTEM,
    PROMPT_GENERATION_TEMPLATE,
)

# Reflection/Critique Agent prompts (Level 4b)
from backend.src.agents.prompts.reflection_prompts import (
    CRITIQUE_EVALUATION_PROMPT,
    CRITIQUE_SYSTEM_PROMPT,
    IMPROVEMENT_PROMPT,
    MAX_REFLECTION_ITERATIONS,
    QUALITY_THRESHOLDS,
    REFINEMENT_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    SAFFIR_SIMPSON_REFERENCE,
    SELF_CRITIQUE_PROMPT,
)

# Supervisor Agent prompts (Level 4b)
from backend.src.agents.prompts.supervisor_prompts import (
    AGENT_CAPABILITIES,
    SUPERVISOR_SYSTEM_PROMPT,
    WORKFLOW_PLANNING_PROMPT,
)

# Triage Agent prompts (Level 4a)
from backend.src.agents.prompts.triage_prompts import (
    TRIAGE_CLASSIFICATION_PROMPT,
    TRIAGE_FALLBACK_PROMPT,
    TRIAGE_SYSTEM_PROMPT,
)
from backend.src.agents.prompts.triage_prompts import (
    USER_CONTEXT_TEMPLATE as TRIAGE_USER_CONTEXT_TEMPLATE,
)

__all__ = [
    # Base prompts (Level 1/2/3)
    "WEATHER_ASSISTANT_SYSTEM_PROMPT",
    "COT_WEATHER_SYSTEM_PROMPT",
    "TOT_WEATHER_SYSTEM_PROMPT",
    "GOT_WEATHER_SYSTEM_PROMPT",
    # Triage prompts (Level 4a)
    "TRIAGE_SYSTEM_PROMPT",
    "TRIAGE_CLASSIFICATION_PROMPT",
    "TRIAGE_FALLBACK_PROMPT",
    "TRIAGE_USER_CONTEXT_TEMPLATE",
    # Hurricane Specialist prompts (Level 4a)
    "HURRICANE_SPECIALIST_SYSTEM_PROMPT",
    "HURRICANE_FORECAST_PROMPT",
    "HURRICANE_VALIDATION_PROMPT",
    "USER_CONTEXT_TEMPLATE",
    "NHC_DATA_TEMPLATE",
    "EMERGENCY_RESPONSE_TEMPLATE",
    "SAFFIR_SIMPSON_SCALE",
    "CONFIDENCE_SCORING_GUIDE",
    # Alert Manager prompts (Level 4a)
    "ALERT_MANAGER_SYSTEM_PROMPT",
    "ALERT_GENERATION_PROMPT",
    "ALERT_SEVERITY_LEVELS",
    "SMS_ALERT_TEMPLATE",
    "PUSH_NOTIFICATION_TEMPLATE",
    "EMAIL_SUBJECT_TEMPLATE",
    "EMAIL_BODY_TEMPLATE",
    "EVACUATION_ALERT_PROMPT",
    "ALERT_SEVERITY_CLASSIFICATION_PROMPT",
    # Supervisor prompts (Level 4b)
    "SUPERVISOR_SYSTEM_PROMPT",
    "WORKFLOW_PLANNING_PROMPT",
    "AGENT_CAPABILITIES",
    # Reflection/Critique prompts (Level 4b)
    "REFLECTION_SYSTEM_PROMPT",
    "SELF_CRITIQUE_PROMPT",
    "IMPROVEMENT_PROMPT",
    "CRITIQUE_SYSTEM_PROMPT",
    "CRITIQUE_EVALUATION_PROMPT",
    "REFINEMENT_PROMPT",
    "QUALITY_THRESHOLDS",
    "MAX_REFLECTION_ITERATIONS",
    "SAFFIR_SIMPSON_REFERENCE",
    # Meta-Prompt Agent prompts (Level 4c)
    "META_PROMPT_SYSTEM",
    "PROMPT_GENERATION_TEMPLATE",
    "FEW_SHOT_EXAMPLES",
    "AGENT_PROMPT_TEMPLATES",
    # Debate Agent prompts (Level 4c)
    "DEBATE_SYSTEM_PROMPT",
    "DEBATE_CONFIG",
    "PROPOSAL_GENERATION_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_EVALUATION_PROMPT",
    "SYNTHESIS_PROMPT",
    "CONFLICT_RESOLUTION_PROMPT",
    "MIN_SCORE_FOR_WINNER",
]
