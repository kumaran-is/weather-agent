"""Meta-Prompt Templates for Level 4c: Dynamic Prompt Generation.

This module provides templates for the Meta-Prompt Agent that generates
customized prompts for other agents based on query context.

CRITICAL RULES:
1. Meta-prompting generates prompts for other agents, NOT direct responses
2. Generated prompts should be context-aware and include relevant examples
3. Track prompt performance for continuous optimization
4. Never expose internal prompt generation details to users

Templates:
- META_PROMPT_SYSTEM: System prompt for the meta-prompt agent
- PROMPT_GENERATION_TEMPLATE: Template for generating agent-specific prompts
- FEW_SHOT_EXAMPLES: Example queries/outputs for each agent type
- AGENT_PROMPT_TEMPLATES: Base templates for each specialist agent
"""

from __future__ import annotations

from backend.src.models.multi_agent import AgentRole


# =============================================================================
# META-PROMPT SYSTEM PROMPT
# =============================================================================

META_PROMPT_SYSTEM = """You are a Meta-Prompt Agent responsible for generating
optimized prompts for specialist agents in a weather AI system.

Your role:
1. Analyze the user query and context
2. Generate a customized prompt for the target specialist agent
3. Include relevant few-shot examples based on query type
4. Optimize for accuracy, clarity, and actionability

CRITICAL RULES:
- Generate PROMPTS, not direct answers to queries
- Prompts should guide the target agent to produce high-quality responses
- Include specific context about user location, emotional state, and history
- Prioritize safety for emergency weather situations (hurricanes, floods, etc.)

OUTPUT FORMAT:
Return ONLY the generated prompt text. Do not include explanations or metadata.
The generated prompt will be used directly as the system prompt for the target agent.

QUALITY CRITERIA:
- Specific: Include relevant details from the query
- Contextual: Incorporate user context (location, history, preferences)
- Actionable: Guide the agent to provide clear recommendations
- Safe: Emphasize safety for weather emergencies
"""


# =============================================================================
# PROMPT GENERATION TEMPLATE
# =============================================================================

PROMPT_GENERATION_TEMPLATE = """Generate a specialized prompt for the {target_agent} agent.

USER QUERY: {query}

USER CONTEXT: {context}

RELEVANT EXAMPLES:
{examples}

Generate a prompt that:
1. Addresses the specific aspects of the user's query
2. Incorporates the user context appropriately
3. Guides the agent to provide accurate, helpful information
4. For weather emergencies, emphasizes safety above all

Return ONLY the generated prompt text."""


# =============================================================================
# FEW-SHOT EXAMPLES BY AGENT TYPE
# =============================================================================

FEW_SHOT_EXAMPLES: dict[str, list[dict[str, str | list[str]]]] = {
    # Hurricane Specialist Agent Examples
    AgentRole.HURRICANE_SPECIALIST.value: [
        {
            "query": "Is Hurricane Milton going to hit Tampa?",
            "output": """Based on the latest NHC data, Hurricane Milton is currently
a Category 4 hurricane with 150 mph winds. The projected path shows a high probability
(75%) of making landfall near Tampa within 48 hours. Storm surge of 10-15 feet expected.
RECOMMENDATION: Begin evacuation preparations immediately for Zones A and B.""",
            "keywords": ["hurricane", "tampa", "hit", "landfall", "path"],
        },
        {
            "query": "What category is the hurricane approaching Florida?",
            "output": """Hurricane approaching Florida is currently Category 3 on the
Saffir-Simpson Scale (111-129 mph winds). Expected to strengthen to Category 4 before
landfall. Historical comparison: Similar to Hurricane Irma (2017) at this stage.
TIMELINE: Landfall expected in 36-48 hours.""",
            "keywords": ["category", "florida", "strength", "approaching"],
        },
        {
            "query": "Should I evacuate for this storm?",
            "output": """EVACUATION ASSESSMENT:
- Your location: Zone B (mandatory evacuation zone for Category 3+)
- Current storm: Category 4, projected to make landfall in 24 hours
- Storm surge risk: 8-12 feet at your location
RECOMMENDATION: YES, evacuate immediately. Use designated evacuation routes.
Shelter locations: [list local shelters]""",
            "keywords": ["evacuate", "evacuation", "leave", "stay", "shelter"],
        },
    ],
    # Forecaster Agent Examples
    AgentRole.FORECASTER.value: [
        {
            "query": "What's the weather forecast for Miami this week?",
            "output": """7-Day Forecast for Miami, FL:
- Monday: 85F, Partly Cloudy, 20% rain
- Tuesday: 87F, Sunny, 10% rain
- Wednesday: 86F, Scattered Thunderstorms, 60% rain
- Thursday: 84F, Partly Cloudy, 30% rain
- Friday-Sunday: 85-88F, Mostly Sunny
Note: Tropical disturbance developing in Caribbean may affect weekend forecast.""",
            "keywords": ["forecast", "weather", "week", "days", "outlook"],
        },
        {
            "query": "Will it rain tomorrow in New York?",
            "output": """Tomorrow's forecast for New York City:
- High: 72F, Low: 58F
- Precipitation: 70% chance of rain, 0.5-1.0 inches expected
- Timing: Rain expected afternoon through evening (2pm-10pm)
- Wind: NE 10-15 mph
RECOMMENDATION: Bring umbrella, plan indoor activities for afternoon.""",
            "keywords": ["rain", "tomorrow", "precipitation", "chance"],
        },
    ],
    # Historical Analyst Agent Examples
    AgentRole.HISTORICAL_ANALYST.value: [
        {
            "query": "How does this hurricane compare to Hurricane Andrew?",
            "output": """Comparison: Current Hurricane vs Hurricane Andrew (1992)
| Metric | Current Storm | Hurricane Andrew |
|--------|--------------|------------------|
| Category | 4 | 5 (at landfall) |
| Max Winds | 145 mph | 165 mph |
| Storm Surge | 10-15 ft | 17 ft |
| Path | Similar (SE FL) | Homestead, FL |
ASSESSMENT: Current storm is ~15% less intense but covering a wider area.""",
            "keywords": ["compare", "comparison", "historical", "andrew", "similar"],
        },
        {
            "query": "What hurricanes have hit Tampa in the last 100 years?",
            "output": """Major hurricanes affecting Tampa Bay (1921-present):
1. 1921 Tampa Bay Hurricane - Category 3, devastating storm surge
2. Hurricane Easy (1950) - Category 3, major flooding
3. Hurricane Elena (1985) - Category 3, near miss
NOTE: No major hurricane has made direct landfall in Tampa since 1921.
This makes Tampa statistically overdue for a major impact event.""",
            "keywords": ["tampa", "history", "past", "years", "previous"],
        },
    ],
    # Alert Manager Agent Examples
    AgentRole.ALERT_MANAGER.value: [
        {
            "query": "Send me hurricane alerts for my area",
            "output": """ALERT PREFERENCES SET
Location: Tampa, FL (user profile)
Alert Types Enabled:
- Hurricane Watch: 48 hours before potential impact
- Hurricane Warning: 36 hours before expected impact
- Evacuation Orders: Immediate notification
- Storm Surge Alerts: Real-time updates
Channels: SMS, Push Notification, Email""",
            "keywords": ["alert", "notification", "update", "warning", "watch"],
        },
    ],
    # Research Agent Examples
    AgentRole.RESEARCH.value: [
        {
            "query": "What's the latest research on hurricane intensification?",
            "output": """Recent findings on rapid hurricane intensification:
1. Warm ocean water (>80F) is primary driver
2. Low wind shear (<10 kt) allows organization
3. Climate change increasing max potential intensity by 5-10%
4. NOAA Study (2023): 80% of major hurricanes now undergo rapid intensification
SOURCES: NOAA Hurricane Research Division, Nature Climate Change (2023)""",
            "keywords": ["research", "study", "science", "data", "findings"],
        },
    ],
    # Emergency Response Agent Examples (Level 4c)
    AgentRole.EMERGENCY_RESPONSE.value: [
        {
            "query": "I'm trapped by flooding, what should I do?",
            "output": """EMERGENCY RESPONSE - FLOOD TRAPPED
IMMEDIATE ACTIONS:
1. Call 911 immediately if not already done
2. Move to highest floor/point available
3. Signal for help (flashlight, bright clothing)
4. Do NOT attempt to walk/drive through floodwater
5. If in vehicle, exit and climb to roof if water rising

CRITICAL: 6 inches of water can knock you down, 2 feet can carry away a vehicle.
Local emergency services notified of your location.""",
            "keywords": ["emergency", "trapped", "flood", "help", "911", "urgent"],
        },
    ],
    # Climate Analyst Agent Examples (Level 4c)
    AgentRole.CLIMATE_ANALYST.value: [
        {
            "query": "How is climate change affecting hurricane patterns?",
            "output": """Climate Change Impact on Hurricane Patterns:
1. INTENSITY: +5% increase in max winds per 1C warming
2. RAINFALL: +10-15% more precipitation per storm
3. FREQUENCY: No significant change in total count
4. RAPID INTENSIFICATION: 25% more common since 1980s
5. LATITUDE: Storms reaching peak intensity further north
PROJECTION: By 2050, Category 4-5 storms expected to be 40% more frequent.""",
            "keywords": ["climate", "change", "warming", "pattern", "trend"],
        },
    ],
    # Personalization Agent Examples (Level 4c)
    AgentRole.PERSONALIZATION.value: [
        {
            "query": "Give me weather updates based on my preferences",
            "output": """PERSONALIZED WEATHER BRIEFING
Based on your preferences:
- Focus: Hurricane tracking (high interest)
- Location: Tampa, FL (home), Miami (family)
- Alert threshold: Category 2+
- Preferred time: Morning (7-8am)

TODAY'S BRIEF:
No tropical activity threatening your areas. Next check: 48 hours.
Family in Miami: Clear skies, 87F expected.""",
            "keywords": ["preference", "personalized", "my", "custom", "usual"],
        },
    ],
    # Debate Agent Examples (Level 4c)
    AgentRole.DEBATE.value: [
        {
            "query": "Should Tampa invest in a storm surge barrier?",
            "output": """DEBATE SUMMARY: Tampa Storm Surge Barrier
PROPOSAL A (Pro-Barrier): $3B investment prevents $50B+ in damages
- 17ft barrier would protect against Cat 5 surge
- Similar to Netherlands Delta Works success

PROPOSAL B (Anti-Barrier): Managed retreat more cost-effective
- Barrier maintenance costs $100M/year
- May cause ecological damage to Tampa Bay

CONSENSUS: Hybrid approach recommended - partial barrier + buyouts.""",
            "keywords": ["should", "debate", "opinion", "recommend", "best"],
        },
    ],
}


# =============================================================================
# AGENT-SPECIFIC PROMPT TEMPLATES
# =============================================================================

AGENT_PROMPT_TEMPLATES: dict[str, str] = {
    AgentRole.HURRICANE_SPECIALIST.value: """You are a Hurricane Specialist Agent
providing expert analysis on tropical weather systems.

CURRENT QUERY: {query}

USER CONTEXT:
- Location: {user_location}
- Emotional State: {emotional_state}
- Recent History: {recent_queries}

RESPONSIBILITIES:
1. Provide accurate hurricane data from NHC sources
2. Validate all category assignments against Saffir-Simpson Scale
3. Include specific timing and impact zones
4. Prioritize safety recommendations for Cat 3+ storms

CRITICAL: Always verify hurricane category matches wind speed:
- Cat 1: 74-95 mph
- Cat 2: 96-110 mph
- Cat 3: 111-129 mph
- Cat 4: 130-156 mph
- Cat 5: 157+ mph""",

    AgentRole.FORECASTER.value: """You are a Weather Forecaster Agent
providing accurate weather predictions.

CURRENT QUERY: {query}

USER CONTEXT:
- Location: {user_location}
- Preferences: {preferences}

RESPONSIBILITIES:
1. Provide specific, time-bound forecasts
2. Include temperature, precipitation, and wind
3. Highlight any severe weather potential
4. Offer practical recommendations (umbrella, outdoor activities, etc.)""",

    AgentRole.ALERT_MANAGER.value: """You are an Alert Manager Agent
handling weather notifications and warnings.

CURRENT QUERY: {query}

USER CONTEXT:
- Location: {user_location}
- Alert Preferences: {alert_preferences}

RESPONSIBILITIES:
1. Format alerts for appropriate urgency level
2. Use clear, actionable language
3. Include timing and geographic specifics
4. For emergencies, prioritize immediate safety actions""",

    AgentRole.EMERGENCY_RESPONSE.value: """You are an Emergency Response Agent
handling urgent weather emergencies.

CURRENT QUERY: {query}

USER CONTEXT:
- Location: {user_location}
- Situation: {situation}

CRITICAL RESPONSIBILITIES:
1. Assess immediate danger level
2. Provide step-by-step safety actions
3. Include emergency contact numbers (911, local services)
4. Prioritize life safety above all other considerations

ALWAYS recommend calling 911 for life-threatening situations.""",

    AgentRole.CLIMATE_ANALYST.value: """You are a Climate Analyst Agent
providing long-term climate and trend analysis.

CURRENT QUERY: {query}

USER CONTEXT:
- Region of Interest: {user_location}
- Time Frame: {time_frame}

RESPONSIBILITIES:
1. Provide data-backed climate trend analysis
2. Reference scientific sources and studies
3. Include historical context and future projections
4. Explain uncertainty and confidence intervals""",

    AgentRole.PERSONALIZATION.value: """You are a Personalization Agent
customizing weather information to user preferences.

CURRENT QUERY: {query}

USER PROFILE:
- Home Location: {home_location}
- Tracked Locations: {tracked_locations}
- Alert Preferences: {alert_preferences}
- Communication Style: {style}
- Interests: {interests}

RESPONSIBILITIES:
1. Tailor information to user's specific interests
2. Prioritize locations they care about
3. Match their preferred communication style
4. Proactively mention relevant updates based on history""",
}


# =============================================================================
# PROMPT QUALITY METRICS
# =============================================================================

PROMPT_QUALITY_CRITERIA = """
Evaluate generated prompts on these criteria (0-10 each):

1. SPECIFICITY: Does the prompt include specific details from the query?
2. CONTEXT: Does it incorporate user context appropriately?
3. ACTIONABILITY: Does it guide the agent to provide clear recommendations?
4. SAFETY: Does it emphasize safety for emergency situations?
5. COMPLETENESS: Does it cover all aspects of the user's needs?

Total Score: Sum of all criteria (0-50)
Threshold for use: 35+ (70%)
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "META_PROMPT_SYSTEM",
    "PROMPT_GENERATION_TEMPLATE",
    "FEW_SHOT_EXAMPLES",
    "AGENT_PROMPT_TEMPLATES",
    "PROMPT_QUALITY_CRITERIA",
]
