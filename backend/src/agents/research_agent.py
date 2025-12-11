"""Research Agent for Level 4b: Deep Data Retrieval.

CRITICAL RULES:
1. Performs deep research and data gathering
2. Integrates multiple data sources
3. Provides comprehensive background information
4. Supports specialist agents with additional context

Architecture:
- Multi-source data integration
- Extended search capabilities
- Comprehensive context building
- Support for complex queries

Design Principles:
- Thorough data gathering
- Multiple source validation
- Context enrichment for other agents
- Support decision-making with deep research
"""

from __future__ import annotations

import httpx
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
from backend.config.settings import settings

logger = structlog.get_logger(__name__)


class ResearchAgent:
    """Deep research and data retrieval agent.

    The Research Agent provides comprehensive background research:
    - Multi-source data gathering
    - Extended context building
    - Support for complex multi-factor queries
    - Deep analysis preparation

    Use Cases:
    - Complex weather phenomena explanation
    - Multi-day forecast research
    - Climate pattern analysis
    - Preparation guidance research

    Attributes:
        llm: ChatOpenAI model for research synthesis
        agent_role: Fixed as AgentRole.RESEARCH
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
    ) -> None:
        """Initialize Research Agent.

        Args:
            model_name: LLM model for research (default: gpt-4o for accuracy)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Some creativity for comprehensive research
            timeout=25.0,  # Longer timeout for deep research
        )
        self.agent_role = AgentRole.RESEARCH

        logger.info(
            "research_agent_initialized",
            model=model_name,
        )

    async def research_query(
        self,
        state: MultiAgentState,
    ) -> MultiAgentState:
        """Perform deep research for query.

        Steps:
        1. Analyze query for research areas
        2. Gather data from available sources
        3. Synthesize comprehensive research response
        4. Return updated state with response

        Args:
            state: Current multi-agent state

        Returns:
            Updated state with research response
        """
        start_time = time.perf_counter()
        query = state.query
        tool_calls: list[str] = []

        logger.info(
            "research_started",
            query=query[:100],
            user_id=state.user_id,
        )

        try:
            # Step 1: Analyze query for research areas
            research_areas = self._identify_research_areas(query)

            # Step 2: Gather data from available sources
            gathered_data = await self._gather_data(query, research_areas)
            if gathered_data.get("mcp_data"):
                tool_calls.append("mcp_servers")

            # Step 3: Generate research synthesis
            research_prompt = self._build_research_prompt(
                query=query,
                research_areas=research_areas,
                gathered_data=gathered_data,
            )

            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=research_prompt),
            ]

            response = await self.llm.ainvoke(messages)
            research_text = response.content

            # Calculate confidence
            confidence = self._calculate_confidence(
                research_areas=research_areas,
                data_sources_used=len(gathered_data.get("sources", [])),
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Create agent response
            agent_response = AgentResponse(
                agent_role=self.agent_role,
                content=research_text,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=duration_ms,
                metadata={
                    "research_areas": research_areas,
                    "data_sources": gathered_data.get("sources", []),
                    "tool_calls": tool_calls,
                },
            )

            logger.info(
                "research_complete",
                research_areas=research_areas,
                data_sources=len(gathered_data.get("sources", [])),
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
                "research_error",
                error=str(e),
                error_type=type(e).__name__,
                query=query[:100],
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            error_response = AgentResponse(
                agent_role=self.agent_role,
                content=f"Unable to complete research: {type(e).__name__}",
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=duration_ms,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "tool_calls": tool_calls,
                },
            )

            if isinstance(state, dict):
                if "agent_responses" not in state:
                    state["agent_responses"] = []
                state["agent_responses"].append(error_response)
            else:
                state.agent_responses.append(error_response)

            return state

    def _identify_research_areas(self, query: str) -> list[str]:
        """Identify research areas needed for query.

        Args:
            query: User's query string

        Returns:
            List of research area keywords
        """
        query_lower = query.lower()
        research_areas = []

        # Weather phenomena
        if any(word in query_lower for word in ["hurricane", "tropical", "storm", "cyclone"]):
            research_areas.append("tropical_meteorology")

        if any(word in query_lower for word in ["forecast", "prediction", "outlook"]):
            research_areas.append("forecast_methodology")

        if any(word in query_lower for word in ["evacuate", "evacuation", "shelter", "prepare"]):
            research_areas.append("emergency_preparedness")

        if any(word in query_lower for word in ["flood", "surge", "flooding", "water"]):
            research_areas.append("flood_risk")

        if any(word in query_lower for word in ["wind", "damage", "destruction", "impact"]):
            research_areas.append("impact_assessment")

        if any(word in query_lower for word in ["climate", "pattern", "trend", "change"]):
            research_areas.append("climate_patterns")

        if any(word in query_lower for word in ["historical", "past", "previous", "record"]):
            research_areas.append("historical_data")

        # Default if no specific areas identified
        if not research_areas:
            research_areas = ["general_weather"]

        return research_areas

    async def _gather_data(
        self,
        query: str,
        research_areas: list[str],
    ) -> dict[str, Any]:
        """Gather data from available sources.

        Args:
            query: User's query
            research_areas: Areas to research

        Returns:
            Dictionary with gathered data and sources
        """
        gathered_data: dict[str, Any] = {
            "sources": [],
            "mcp_data": {},
            "knowledge_base": {},
        }

        # Try MCP servers if available
        try:
            if settings.MCP_WEATHER_SERVER_ENABLED:
                weather_data = await self._fetch_from_weather_mcp()
                if weather_data:
                    gathered_data["mcp_data"]["weather"] = weather_data
                    gathered_data["sources"].append("weather_mcp_server")

            if settings.MCP_HURRICANE_SERVER_ENABLED:
                hurricane_data = await self._fetch_from_hurricane_mcp()
                if hurricane_data:
                    gathered_data["mcp_data"]["hurricane"] = hurricane_data
                    gathered_data["sources"].append("hurricane_mcp_server")

        except Exception as e:
            logger.warning("mcp_data_gather_error", error=str(e))

        # Add knowledge base data based on research areas
        if "tropical_meteorology" in research_areas:
            gathered_data["knowledge_base"]["tropical"] = self._get_tropical_knowledge()
            gathered_data["sources"].append("tropical_meteorology_kb")

        if "emergency_preparedness" in research_areas:
            gathered_data["knowledge_base"]["preparedness"] = self._get_preparedness_knowledge()
            gathered_data["sources"].append("emergency_preparedness_kb")

        if "flood_risk" in research_areas:
            gathered_data["knowledge_base"]["flood"] = self._get_flood_knowledge()
            gathered_data["sources"].append("flood_risk_kb")

        return gathered_data

    async def _fetch_from_weather_mcp(self) -> dict[str, Any] | None:
        """Fetch data from Weather MCP server.

        Returns:
            Weather data or None
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.MCP_WEATHER_SERVER_URL}/api/status")
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass
        return None

    async def _fetch_from_hurricane_mcp(self) -> dict[str, Any] | None:
        """Fetch data from Hurricane MCP server.

        Returns:
            Hurricane data or None
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.MCP_HURRICANE_SERVER_URL}/api/storms/active")
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass
        return None

    def _get_tropical_knowledge(self) -> dict[str, Any]:
        """Get tropical meteorology knowledge base data.

        Returns:
            Dictionary with tropical weather knowledge
        """
        return {
            "saffir_simpson_scale": {
                1: {"wind_mph": "74-95", "damage": "Minimal", "surge_ft": "4-5"},
                2: {"wind_mph": "96-110", "damage": "Moderate", "surge_ft": "6-8"},
                3: {"wind_mph": "111-129", "damage": "Extensive", "surge_ft": "9-12"},
                4: {"wind_mph": "130-156", "damage": "Extreme", "surge_ft": "13-18"},
                5: {"wind_mph": "157+", "damage": "Catastrophic", "surge_ft": "18+"},
            },
            "formation_requirements": [
                "Sea surface temperature ≥80°F (26.5°C)",
                "Distance from equator (usually >5°N or S)",
                "Low vertical wind shear",
                "Pre-existing disturbance",
                "Moist mid-levels",
            ],
            "intensification_factors": [
                "Warm ocean water depth (ocean heat content)",
                "Upper-level outflow",
                "Low wind shear",
                "High humidity",
                "Favorable upper-level pattern",
            ],
        }

    def _get_preparedness_knowledge(self) -> dict[str, Any]:
        """Get emergency preparedness knowledge base data.

        Returns:
            Dictionary with preparedness knowledge
        """
        return {
            "evacuation_zones": {
                "A": "Lowest elevation, first to evacuate for any hurricane",
                "B": "Next lowest, evacuate for Cat 2+",
                "C": "Higher ground, evacuate for Cat 3+",
                "D": "Highest elevation in coastal area",
                "E": "Inland, may evacuate for Cat 4-5",
            },
            "preparation_timeline": {
                "5_days_out": "Monitor forecast, review plans, check supplies",
                "3_days_out": "Finalize supplies, fuel vehicles, review routes",
                "2_days_out": "Begin securing property, final supply runs",
                "1_day_out": "Complete preparations, evacuate if ordered",
                "12_hours_out": "Final preparations, shelter in place if staying",
            },
            "essential_supplies": [
                "Water (1 gallon per person per day, 7 day supply)",
                "Non-perishable food (7 day supply)",
                "Medications (30 day supply)",
                "First aid kit",
                "Flashlights and batteries",
                "Battery-powered radio",
                "Cash and important documents",
            ],
        }

    def _get_flood_knowledge(self) -> dict[str, Any]:
        """Get flood risk knowledge base data.

        Returns:
            Dictionary with flood knowledge
        """
        return {
            "storm_surge_factors": [
                "Hurricane intensity (category)",
                "Forward speed (slower = more surge time)",
                "Angle of approach to coast",
                "Shape of coastline",
                "Bathymetry (underwater topography)",
                "Tide cycle",
            ],
            "flood_zones": {
                "AE": "Base flood elevation determined, mandatory insurance",
                "VE": "Coastal high hazard area with wave action",
                "X500": "500-year flood zone, 0.2% annual chance",
                "X": "Outside flood zone, but flooding still possible",
            },
            "safety_facts": [
                "6 inches of moving water can knock you down",
                "12 inches of moving water can carry away a vehicle",
                "2 feet of water can float most cars",
                "Most hurricane deaths are from drowning",
            ],
        }

    def _get_system_prompt(self) -> str:
        """Get system prompt for research synthesis.

        Returns:
            System prompt string
        """
        return """You are a Weather Research Agent providing comprehensive background information.

Your role:
1. Synthesize information from multiple sources
2. Provide thorough, well-organized research
3. Support decision-making with facts and context
4. Explain complex weather phenomena clearly

Guidelines:
1. Be comprehensive but organized
2. Use specific data points (numbers, dates, statistics)
3. Cite sources when available
4. Explain technical concepts clearly
5. Connect research to practical applications

Research areas you cover:
- Tropical meteorology (hurricane formation, intensification)
- Forecast methodology (how predictions work)
- Emergency preparedness (evacuation, supplies, planning)
- Flood risk (storm surge, inland flooding)
- Impact assessment (damage potential, risks)
- Climate patterns (seasonal trends, historical patterns)
- Historical data (past events, lessons learned)

Always provide actionable insights along with background information."""

    def _build_research_prompt(
        self,
        query: str,
        research_areas: list[str],
        gathered_data: dict[str, Any],
    ) -> str:
        """Build research synthesis prompt.

        Args:
            query: User's query
            research_areas: Research areas identified
            gathered_data: Data gathered from sources

        Returns:
            Formatted prompt string
        """
        # Format gathered data
        data_sections = []

        if gathered_data.get("mcp_data"):
            data_sections.append("**Real-Time Data Available**:")
            for source, data in gathered_data["mcp_data"].items():
                data_sections.append(f"- {source}: Data available")

        if gathered_data.get("knowledge_base"):
            for topic, kb_data in gathered_data["knowledge_base"].items():
                data_sections.append(f"\n**{topic.replace('_', ' ').title()} Knowledge**:")
                # Summarize key points from knowledge base
                if isinstance(kb_data, dict):
                    for key in list(kb_data.keys())[:3]:
                        data_sections.append(f"- {key.replace('_', ' ').title()}: Available")

        data_summary = "\n".join(data_sections) if data_sections else "Limited data sources available."

        return f"""Conduct comprehensive research for this query:

**Query**: {query}

**Research Areas to Cover**: {', '.join(research_areas)}

**Available Data Sources**:
{data_summary}

**Data Sources Used**: {', '.join(gathered_data.get('sources', ['LLM knowledge']))}

Provide thorough research that:
1. Directly addresses the query
2. Covers all relevant research areas
3. Includes specific facts and data
4. Explains technical concepts
5. Provides actionable insights

Organize your response with clear sections and bullet points where appropriate."""

    def _calculate_confidence(
        self,
        research_areas: list[str],
        data_sources_used: int,
    ) -> float:
        """Calculate research confidence.

        Args:
            research_areas: Areas researched
            data_sources_used: Number of data sources

        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = 0.7

        # More sources = higher confidence
        if data_sources_used >= 3:
            base_confidence += 0.15
        elif data_sources_used >= 1:
            base_confidence += 0.1

        # More research areas = more comprehensive
        if len(research_areas) >= 3:
            base_confidence += 0.05

        return min(0.9, base_confidence)
