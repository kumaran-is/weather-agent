"""Historical Analyst Agent for Level 4b: Historical Weather Pattern Analysis.

CRITICAL RULES:
1. Provides pattern-based historical analysis
2. Compares current situations with past events
3. Uses real historical hurricanes (Michael 2018, Andrew 1992, etc.)
4. Integrates with Supervisor for parallel execution

Architecture:
- Historical hurricane database (curated)
- Pattern matching for similar events
- Trend analysis for climate patterns
- Comparison with current conditions

Design Principles:
- Reference real historical events (not fictional)
- Provide actionable insights from history
- Clear comparison with current situation
- Support decision-making with precedents
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.src.models.multi_agent import (
    AgentRole,
    AgentResponse,
    MultiAgentState,
)

logger = structlog.get_logger(__name__)


# Historical hurricane database (real events)
HISTORICAL_HURRICANES = {
    "michael_2018": {
        "name": "Hurricane Michael",
        "year": 2018,
        "category_at_landfall": 5,
        "max_winds_mph": 160,
        "landfall_location": "Mexico Beach, Florida",
        "date": "October 10, 2018",
        "deaths": 74,
        "damage_billion": 25.1,
        "notes": "First Category 5 to hit FL Panhandle. Rapid intensification.",
    },
    "ian_2022": {
        "name": "Hurricane Ian",
        "year": 2022,
        "category_at_landfall": 4,
        "max_winds_mph": 155,
        "landfall_location": "Cayo Costa, Florida",
        "date": "September 28, 2022",
        "deaths": 161,
        "damage_billion": 112.9,
        "notes": "Catastrophic flooding in Fort Myers area. Second costliest US hurricane.",
    },
    "irma_2017": {
        "name": "Hurricane Irma",
        "year": 2017,
        "category_at_landfall": 4,
        "max_winds_mph": 180,
        "landfall_location": "Cudjoe Key, Florida",
        "date": "September 10, 2017",
        "deaths": 134,
        "damage_billion": 77.2,
        "notes": "Maintained Cat 5 intensity for 37 hours. Devastated Keys.",
    },
    "andrew_1992": {
        "name": "Hurricane Andrew",
        "year": 1992,
        "category_at_landfall": 5,
        "max_winds_mph": 165,
        "landfall_location": "Homestead, Florida",
        "date": "August 24, 1992",
        "deaths": 65,
        "damage_billion": 27.3,
        "notes": "Changed building codes in Florida. Compact but intense.",
    },
    "charley_2004": {
        "name": "Hurricane Charley",
        "year": 2004,
        "category_at_landfall": 4,
        "max_winds_mph": 150,
        "landfall_location": "Punta Gorda, Florida",
        "date": "August 13, 2004",
        "deaths": 35,
        "damage_billion": 16.0,
        "notes": "Rapidly intensified. Part of 2004 quartet (Charley, Frances, Ivan, Jeanne).",
    },
    "dorian_2019": {
        "name": "Hurricane Dorian",
        "year": 2019,
        "category_at_landfall": 5,  # At Bahamas
        "max_winds_mph": 185,
        "landfall_location": "Elbow Cay, Bahamas",
        "date": "September 1, 2019",
        "deaths": 84,
        "damage_billion": 3.4,
        "notes": "Stalled over Bahamas for 40+ hours. Second strongest Atlantic hurricane on record.",
    },
    "katrina_2005": {
        "name": "Hurricane Katrina",
        "year": 2005,
        "category_at_landfall": 3,  # At Louisiana
        "max_winds_mph": 175,
        "landfall_location": "Louisiana/Mississippi",
        "date": "August 29, 2005",
        "deaths": 1833,
        "damage_billion": 186.3,
        "notes": "Deadliest US hurricane since 1928. Catastrophic levee failures in New Orleans.",
    },
}


class HistoricalAnalystAgent:
    """Historical weather pattern analysis specialist.

    The Historical Analyst Agent provides context through:
    - Comparison with similar past hurricanes
    - Pattern analysis for the region
    - Trend identification
    - Precedent-based insights

    Historical Database:
    - Real hurricanes with verified data
    - Landfall locations and dates
    - Impact metrics (deaths, damage)
    - Notable characteristics

    Attributes:
        llm: ChatOpenAI model for analysis generation
        agent_role: Fixed as AgentRole.HISTORICAL_ANALYST
        hurricane_db: Dictionary of historical hurricanes
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
    ) -> None:
        """Initialize Historical Analyst Agent.

        Args:
            model_name: LLM model for analysis (default: gpt-4o for accuracy)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # Low temperature for factual accuracy
            timeout=20.0,
        )
        self.agent_role = AgentRole.HISTORICAL_ANALYST
        self.hurricane_db = HISTORICAL_HURRICANES

        logger.info(
            "historical_analyst_initialized",
            model=model_name,
            hurricanes_in_db=len(self.hurricane_db),
        )

    async def analyze_patterns(
        self,
        state: MultiAgentState,
    ) -> MultiAgentState:
        """Analyze historical weather patterns for query.

        Steps:
        1. Parse analysis request from query
        2. Find relevant historical events
        3. Generate comparison/analysis
        4. Return updated state with response

        Args:
            state: Current multi-agent state

        Returns:
            Updated state with historical analysis response
        """
        start_time = time.perf_counter()
        query = state.query

        logger.info(
            "historical_analysis_started",
            query=query[:100],
            user_id=state.user_id,
        )

        try:
            # Step 1: Parse analysis request
            analysis_request = self._parse_analysis_request(query)

            # Step 2: Find relevant historical events
            relevant_hurricanes = self._find_relevant_hurricanes(
                query=query,
                analysis_type=analysis_request["type"],
            )

            # Step 3: Generate analysis
            analysis_prompt = self._build_analysis_prompt(
                query=query,
                analysis_type=analysis_request["type"],
                hurricanes=relevant_hurricanes,
            )

            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=analysis_prompt),
            ]

            response = await self.llm.ainvoke(messages)
            analysis_text = response.content

            # Calculate confidence
            confidence = self._calculate_confidence(
                hurricanes_found=len(relevant_hurricanes),
                analysis_type=analysis_request["type"],
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Create agent response
            agent_response = AgentResponse(
                agent_role=self.agent_role,
                content=analysis_text,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=duration_ms,
                metadata={
                    "analysis_type": analysis_request["type"],
                    "hurricanes_referenced": [h["name"] for h in relevant_hurricanes],
                    "hurricanes_count": len(relevant_hurricanes),
                },
            )

            logger.info(
                "historical_analysis_complete",
                analysis_type=analysis_request["type"],
                hurricanes_referenced=len(relevant_hurricanes),
                confidence=confidence,
                duration_ms=duration_ms,
            )

            # Update state
            if isinstance(state, dict):
                if "agent_responses" not in state:
                    state["agent_responses"] = []
                state["agent_responses"].append(agent_response)
            else:
                state.agent_responses.append(agent_response)

            return state

        except Exception as e:
            logger.error(
                "historical_analysis_error",
                error=str(e),
                error_type=type(e).__name__,
                query=query[:100],
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            error_response = AgentResponse(
                agent_role=self.agent_role,
                content=f"Unable to complete historical analysis: {type(e).__name__}",
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=duration_ms,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            if isinstance(state, dict):
                if "agent_responses" not in state:
                    state["agent_responses"] = []
                state["agent_responses"].append(error_response)
            else:
                state.agent_responses.append(error_response)

            return state

    def _parse_analysis_request(self, query: str) -> dict[str, Any]:
        """Parse query for analysis parameters.

        Args:
            query: User's query string

        Returns:
            Dictionary with analysis type and parameters
        """
        query_lower = query.lower()

        # Determine analysis type
        if any(word in query_lower for word in ["compare", "similar", "like", "versus"]):
            analysis_type = "comparison"
        elif any(word in query_lower for word in ["trend", "pattern", "historical", "over time"]):
            analysis_type = "trend_analysis"
        elif any(word in query_lower for word in ["worst", "strongest", "deadliest", "costliest"]):
            analysis_type = "superlatives"
        elif any(word in query_lower for word in ["previous", "last time", "before", "history"]):
            analysis_type = "precedent"
        else:
            analysis_type = "general"

        return {
            "type": analysis_type,
            "query": query,
        }

    def _find_relevant_hurricanes(
        self,
        query: str,
        analysis_type: str,
    ) -> list[dict[str, Any]]:
        """Find relevant historical hurricanes for query.

        Args:
            query: User's query string
            analysis_type: Type of analysis requested

        Returns:
            List of relevant hurricane dictionaries
        """
        query_lower = query.lower()
        relevant = []

        # Check for specific hurricane names
        for key, hurricane in self.hurricane_db.items():
            name_lower = hurricane["name"].lower()
            if name_lower in query_lower or hurricane["name"].split()[-1].lower() in query_lower:
                relevant.append(hurricane)

        # If no specific mentions, find by criteria
        if not relevant:
            # Location-based matching
            florida_locations = ["florida", "tampa", "miami", "fort myers", "panhandle", "keys"]
            if any(loc in query_lower for loc in florida_locations):
                relevant = [h for h in self.hurricane_db.values()
                           if "Florida" in h.get("landfall_location", "")]

            # Category-based matching
            for cat in [5, 4, 3]:
                if f"category {cat}" in query_lower or f"cat {cat}" in query_lower:
                    relevant = [h for h in self.hurricane_db.values()
                               if h.get("category_at_landfall", 0) == cat]
                    break

        # If still no matches, return most significant Florida hurricanes
        if not relevant:
            relevant = [
                self.hurricane_db["ian_2022"],
                self.hurricane_db["michael_2018"],
                self.hurricane_db["irma_2017"],
                self.hurricane_db["andrew_1992"],
            ]

        # Sort by year (most recent first)
        relevant.sort(key=lambda h: h.get("year", 0), reverse=True)

        return relevant[:5]  # Max 5 hurricanes

    def _get_system_prompt(self) -> str:
        """Get system prompt for historical analysis.

        Returns:
            System prompt string
        """
        return """You are a Historical Weather Analyst specializing in hurricane history.

Your role:
1. Provide accurate historical context for weather situations
2. Compare current conditions with past events
3. Identify patterns and precedents
4. Use real data (deaths, damage, dates) accurately

Guidelines:
1. ALWAYS use real hurricane names and dates (never fictional events)
2. Include specific metrics (wind speeds, damage costs, deaths)
3. Note how building codes, preparation have improved over time
4. Highlight what made each storm unique
5. Provide actionable lessons from history

Available Historical Data:
- Hurricane Michael (2018): Cat 5, 160 mph, Mexico Beach FL
- Hurricane Ian (2022): Cat 4, 155 mph, $112.9B damage
- Hurricane Irma (2017): Cat 4, 180 mph peak, Keys devastation
- Hurricane Andrew (1992): Cat 5, 165 mph, changed building codes
- Hurricane Charley (2004): Cat 4, rapid intensification
- Hurricane Dorian (2019): Cat 5, stalled over Bahamas
- Hurricane Katrina (2005): Cat 3 at landfall, 1833 deaths

Be factual and specific. Avoid speculation about current storms."""

    def _build_analysis_prompt(
        self,
        query: str,
        analysis_type: str,
        hurricanes: list[dict[str, Any]],
    ) -> str:
        """Build analysis prompt with historical data.

        Args:
            query: User's query
            analysis_type: Type of analysis
            hurricanes: Relevant historical hurricanes

        Returns:
            Formatted prompt string
        """
        # Format hurricane data
        hurricane_data = ""
        for h in hurricanes:
            hurricane_data += f"""
**{h['name']} ({h['year']})**
- Category at Landfall: {h['category_at_landfall']}
- Maximum Winds: {h['max_winds_mph']} mph
- Landfall: {h['landfall_location']} on {h['date']}
- Deaths: {h['deaths']}
- Damage: ${h['damage_billion']} billion
- Notes: {h['notes']}
"""

        return f"""Provide historical analysis for this query:

**Query**: {query}
**Analysis Type**: {analysis_type}

**Relevant Historical Hurricanes**:
{hurricane_data}

Based on this historical data, provide:
1. Relevant comparisons to the current situation
2. Key lessons from these past events
3. How conditions/preparation have changed
4. Actionable insights for decision-making

Be specific with dates, numbers, and locations."""

    def _calculate_confidence(
        self,
        hurricanes_found: int,
        analysis_type: str,
    ) -> float:
        """Calculate analysis confidence.

        Args:
            hurricanes_found: Number of relevant hurricanes found
            analysis_type: Type of analysis

        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = 0.7

        # More relevant hurricanes = higher confidence
        if hurricanes_found >= 3:
            base_confidence += 0.15
        elif hurricanes_found >= 1:
            base_confidence += 0.1

        # Analysis type adjustments
        type_adjustments = {
            "comparison": 0.05,  # Specific comparisons are reliable
            "precedent": 0.05,
            "trend_analysis": 0.0,  # Trends can be subjective
            "superlatives": 0.05,  # Facts are verifiable
            "general": 0.0,
        }

        adjustment = type_adjustments.get(analysis_type, 0.0)
        return min(0.9, base_confidence + adjustment)
