"""Hurricane Specialist Agent Prompts for Level 4a: Three-Agent Foundation System.

CRITICAL: This module defines the prompts used by the Hurricane Specialist Agent
to provide expert hurricane forecasts, storm tracking, and safety recommendations.

Prompts:
- HURRICANE_SPECIALIST_SYSTEM_PROMPT: System-level instructions for domain expertise
- HURRICANE_FORECAST_PROMPT: Template for generating comprehensive hurricane forecasts
- HURRICANE_VALIDATION_PROMPT: Template for validating hurricane data accuracy
- HURRICANE_CONTEXT_TEMPLATE: Template for formatting MCP server data

Level 4a Architecture:
- Triage Agent → Routes queries to Hurricane Specialist
- Hurricane Specialist Agent → Handles hurricane-specific queries using MCP server
- Alert Manager Agent → Generates emergency alerts based on specialist analysis

Design Principles:
- NHC data via Hurricane MCP Server (http://localhost:8081) as authoritative source
- Saffir-Simpson scale validation for all category assignments
- Confidence-based responses with clear uncertainty communication
- Life-safety priority for all evacuation recommendations
"""

# Saffir-Simpson Hurricane Wind Scale reference
# Used for validation and response generation
SAFFIR_SIMPSON_SCALE = {
    1: {"wind_mph": (74, 95), "surge_ft": (4, 5), "damage": "Some damage"},
    2: {"wind_mph": (96, 110), "surge_ft": (6, 8), "damage": "Extensive damage"},
    3: {"wind_mph": (111, 129), "surge_ft": (9, 12), "damage": "Devastating damage"},
    4: {"wind_mph": (130, 156), "surge_ft": (13, 18), "damage": "Catastrophic damage"},
    5: {"wind_mph": (157, 999), "surge_ft": (19, 999), "damage": "Catastrophic damage"},
}

# Hurricane Specialist System Prompt - Domain Expertise Definition
HURRICANE_SPECIALIST_SYSTEM_PROMPT = """You are a Hurricane Domain Expert Agent with specialized knowledge of tropical meteorology and emergency preparedness.

Your Expertise:
- Hurricane forecasting and track prediction (NHC cone of uncertainty)
- Saffir-Simpson Hurricane Wind Scale (Category 1-5 classification)
- Storm surge prediction and coastal flooding
- Evacuation zone guidance (Zone A, B, C designations)
- Historical storm analysis and pattern recognition
- Impact assessment (wind, surge, rainfall, tornadoes)

Primary Data Source:
- National Hurricane Center (NHC) data via Hurricane MCP Server
- URL: http://localhost:8081 (Hurricane Tracker MCP Server)
- Tools: get_active_storms, get_storm_details, get_forecast_cone, get_evacuation_zones

Critical Validation Rules (MANDATORY):
You MUST validate all hurricane categories against the Saffir-Simpson scale:

| Category | Wind Speed (mph) | Storm Surge (ft) | Damage Level |
|----------|------------------|------------------|--------------|
| Cat 1    | 74-95 mph        | 4-5 ft           | Some damage  |
| Cat 2    | 96-110 mph       | 6-8 ft           | Extensive    |
| Cat 3    | 111-129 mph      | 9-12 ft          | Devastating  |
| Cat 4    | 130-156 mph      | 13-18 ft         | Catastrophic |
| Cat 5    | 157+ mph         | 19+ ft           | Catastrophic |

⚠️ NEVER assign Category 5 to a storm with winds <157 mph!
⚠️ NEVER assign Category 4 to a storm with winds <130 mph!
⚠️ NEVER minimize hurricane risks - err on the side of caution

Response Guidelines:
1. **Accuracy First**: Use NHC data as authoritative source
2. **Time Specificity**: Use exact EDT/UTC times, NEVER "soon" or "later"
3. **Location Precision**: Specify cities, counties, evacuation zones
4. **Confidence Scoring**: State HIGH/MEDIUM/LOW certainty with reasoning
5. **Actionable Guidance**: Provide clear next steps for users
6. **Life-Safety Priority**: Conservative recommendations for evacuation

Response Structure:
1. Current Storm Status (name, category, location, wind speed)
2. Forecast Track (24h, 48h, 72h positions and intensity)
3. Potential Impacts (wind, surge, rainfall by location)
4. Recommended Actions (evacuation, preparation, shelter)
5. Confidence Assessment (HIGH/MEDIUM/LOW with rationale)

CRITICAL:
- ALWAYS reference NHC data when providing forecasts
- ALWAYS validate categories before responding
- NEVER use vague timing (use specific EDT/UTC times)
- NEVER minimize life-safety risks
- IF uncertain, recommend more conservative action
"""

# Hurricane Forecast Prompt Template
HURRICANE_FORECAST_PROMPT = """Provide an expert hurricane analysis for this query:

**User Query**: {query}

**NHC Data (from Hurricane MCP Server)**:
{nhc_data}

**Advanced Reasoning Results** (if applicable):
{reasoning_context}

**User Context**:
{user_context}

**Instructions**:
1. Analyze the query and NHC data to formulate response
2. VALIDATE all hurricane categories against Saffir-Simpson scale
3. Provide specific times (EDT/UTC), locations, and actionable guidance
4. Calculate confidence based on:
   - NHC data availability (+30% if present)
   - Forecast time horizon (shorter = higher confidence)
   - Storm predictability (tropical storms more predictable than rapidly intensifying)
5. Structure response with clear sections

**Response Format**:
## Current Status
[Storm name, category, location, wind speed, movement]

## Forecast Track
[24h, 48h, 72h positions and intensity changes]

## Potential Impacts
[Wind, surge, rainfall impacts by location]

## Recommended Actions
[Specific, actionable guidance with timing]

## Confidence Assessment
[HIGH/MEDIUM/LOW with specific reasoning]

REMEMBER:
- Validate category matches wind speed (Saffir-Simpson)
- Use exact times (e.g., "2:00 PM EDT Tuesday" not "tomorrow afternoon")
- Prioritize life-safety in all recommendations
"""

# Hurricane Validation Prompt - For checking response accuracy
HURRICANE_VALIDATION_PROMPT = """Validate this hurricane forecast for accuracy:

**Forecast to Validate**:
{forecast_text}

**NHC Reference Data**:
{nhc_data}

**Validation Checklist**:
1. [ ] Category matches wind speed (Saffir-Simpson scale)
   - Cat 1: 74-95 mph
   - Cat 2: 96-110 mph
   - Cat 3: 111-129 mph
   - Cat 4: 130-156 mph
   - Cat 5: 157+ mph

2. [ ] Times are specific (EDT/UTC), not vague ("soon", "later")

3. [ ] Locations are specific (cities, zones), not general

4. [ ] Recommendations are conservative for life-safety

5. [ ] Confidence level is appropriate for forecast certainty

**Return JSON**:
{{
    "is_valid": true/false,
    "errors": ["List of specific errors if any"],
    "warnings": ["List of warnings/suggestions"],
    "corrected_category": null or corrected value,
    "confidence_adjustment": null or adjusted value
}}
"""

# User Context Template - For memory integration
USER_CONTEXT_TEMPLATE = """**User Location**: {user_location}

**Previous Hurricane Queries**:
{previous_queries}

**User Preferences**:
{user_preferences}

**Emotional State**: {emotional_state}
{emotional_guidance}
"""

# NHC Data Formatting Template
NHC_DATA_TEMPLATE = """**Active Storms**:
{active_storms}

**Storm Details**:
{storm_details}

**Forecast Cone**:
{forecast_cone}

**Evacuation Zones**:
{evacuation_zones}
"""

# Emergency Response Template - For urgent queries
EMERGENCY_RESPONSE_TEMPLATE = """⚠️ **EMERGENCY HURRICANE ALERT** ⚠️

**Storm**: {storm_name} (Category {category})
**Location**: {location}
**Wind Speed**: {wind_speed} mph
**Movement**: {movement}

**IMMEDIATE ACTIONS REQUIRED**:
{immediate_actions}

**Evacuation Status**:
{evacuation_status}

**Time Window**: {time_window}

**Shelter Information**:
{shelter_info}

This is a LIFE-SAFETY situation. Follow all official evacuation orders.
"""

# Confidence Scoring Guide
CONFIDENCE_SCORING_GUIDE = """
**Confidence Scoring Criteria**:

HIGH Confidence (≥0.8):
- NHC data available and recent (<6 hours old)
- Forecast horizon <48 hours
- Storm track is stable and predictable
- Historical patterns support forecast

MEDIUM Confidence (0.5-0.79):
- NHC data available but may be older
- Forecast horizon 48-96 hours
- Some uncertainty in track or intensity
- Rapid intensification possible

LOW Confidence (<0.5):
- Limited or no NHC data
- Forecast horizon >96 hours
- High uncertainty in track or intensity
- Unusual storm behavior observed

**Confidence Adjustments**:
+0.3 if NHC data available
+0.1 if advanced reasoning (ToT/GoT) used
+0.1 if validation passed
-0.2 if forecast horizon >72 hours
-0.2 if rapid intensification possible
"""
