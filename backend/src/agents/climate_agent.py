"""Climate Analyst Agent for Level 4c: Long-term Climate Pattern Analysis.

This agent specializes in analyzing long-term climate trends, historical
patterns, and climate change impacts on weather events.

Features:
- Long-term climate trend analysis
- Historical pattern identification
- Climate change impact assessment
- Seasonal forecasting
- Scientific source citation
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)

logger = structlog.get_logger()


# =============================================================================
# Climate Analyst Prompts
# =============================================================================

CLIMATE_SYSTEM_PROMPT = """You are a Climate Analyst Agent specializing in long-term
climate patterns, trends, and climate change impacts on weather.

Your expertise includes:
1. Multi-decadal climate trend analysis
2. Historical weather pattern identification
3. Climate change attribution science
4. Regional climate variations
5. Seasonal and multi-year forecasting

ANALYSIS GUIDELINES:
- Always cite scientific sources (NOAA, IPCC, peer-reviewed studies)
- Distinguish between weather (short-term) and climate (long-term)
- Quantify trends with percentages and statistical confidence
- Present uncertainty ranges where applicable
- Compare current conditions to historical baselines

OUTPUT STRUCTURE:
1. TREND SUMMARY: Key findings (2-3 bullet points)
2. HISTORICAL CONTEXT: How current compares to past
3. CLIMATE ATTRIBUTION: What factors are driving changes
4. PROJECTIONS: Future expectations with confidence levels
5. SOURCES: Scientific references

CRITICAL RULES:
- Base analysis on peer-reviewed science
- Present uncertainty honestly (HIGH/MEDIUM/LOW confidence)
- Avoid political commentary - focus on data
- Distinguish between correlation and causation
- Note when data is limited or inconclusive
"""


TREND_ANALYSIS_PROMPT = """Analyze climate trends for:

REGION: {region}
METRIC: {metric}
TIME PERIOD: {time_period}

Provide:
1. Historical baseline (pre-1990)
2. Current trend direction and magnitude
3. Rate of change (per decade)
4. Statistical confidence
5. Key drivers of the trend
6. Scientific sources
"""


CLIMATE_IMPACT_PROMPT = """Assess climate change impacts on:

WEATHER PHENOMENON: {phenomenon}
REGION: {region}

Include:
1. How has this phenomenon changed over time?
2. What is the scientific consensus on attribution?
3. What changes are projected for the future?
4. What is the uncertainty range?
5. Cite relevant scientific studies
"""


SEASONAL_OUTLOOK_PROMPT = """Provide seasonal climate outlook for:

REGION: {region}
SEASON: {season}
YEAR: {year}

Include:
1. Expected temperature departures from normal
2. Precipitation outlook
3. Confidence level for predictions
4. Major climate drivers (ENSO, NAO, etc.)
5. Comparison to recent years
"""


class ClimateAnalystAgent:
    """Agent specialized in long-term climate analysis.

    Provides data-driven climate trend analysis, historical context,
    and climate change impact assessments.

    Attributes:
        llm: Language model for climate analysis
        agent_role: Role identifier (CLIMATE_ANALYST)
        _analysis_count: Count of analyses performed
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,  # Low temp for factual analysis
    ):
        """Initialize the Climate Analyst Agent.

        Args:
            model_name: Model to use for climate analysis.
            temperature: Low temperature for factual accuracy.
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=30.0,  # Climate analysis may take longer
        )
        self.agent_role = AgentRole.CLIMATE_ANALYST
        self._analysis_count = 0

    async def process(self, state: MultiAgentState) -> MultiAgentState:
        """Process a climate-related query.

        Args:
            state: Current workflow state containing the query.

        Returns:
            Updated state with climate analysis.
        """
        start_time = time.perf_counter()

        try:
            # Detect analysis type from query
            analysis_type = self._detect_analysis_type(state.query)

            # Generate climate analysis
            response = await self._generate_climate_analysis(
                query=state.query,
                analysis_type=analysis_type,
                context=state.memory_context,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._analysis_count += 1

            # Add response to state
            state.agent_responses.append(
                AgentResponse(
                    agent_role=self.agent_role,
                    content=response,
                    confidence=0.85,
                    execution_time_ms=duration_ms,
                    metadata={
                        "analysis_type": analysis_type,
                        "analysis_count": self._analysis_count,
                    },
                )
            )

            state.current_agent = self.agent_role

            logger.info(
                "climate_analysis_complete",
                analysis_type=analysis_type,
                duration_ms=round(duration_ms, 2),
            )

            return state

        except Exception as e:
            logger.error("climate_analysis_error", error=str(e))

            duration_ms = (time.perf_counter() - start_time) * 1000
            state.agent_responses.append(
                AgentResponse(
                    agent_role=self.agent_role,
                    content=f"Climate analysis encountered an error: {str(e)}",
                    confidence=0.0,
                    execution_time_ms=duration_ms,
                    metadata={"error": str(e)},
                )
            )

            return state

    async def _generate_climate_analysis(
        self,
        query: str,
        analysis_type: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate climate analysis for the query.

        Args:
            query: User's climate query.
            analysis_type: Detected type of analysis needed.
            context: Additional context from memory.

        Returns:
            Climate analysis response.
        """
        location = context.get("user_location", "global") if context else "global"

        messages = [
            SystemMessage(content=CLIMATE_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""CLIMATE QUERY: {query}

ANALYSIS TYPE: {analysis_type}
REGION OF INTEREST: {location}

Provide a comprehensive climate analysis following the output structure.
Include scientific sources and quantify trends where possible.
Present uncertainty ranges and confidence levels."""
            ),
        ]

        response = await self.llm.ainvoke(messages)
        return str(response.content)

    def _detect_analysis_type(self, query: str) -> str:
        """Detect type of climate analysis needed.

        Args:
            query: User's query text.

        Returns:
            Analysis type classification.
        """
        query_lower = query.lower()

        # Analysis type keywords
        if any(kw in query_lower for kw in ["trend", "change", "changing", "over time"]):
            return "TREND_ANALYSIS"
        elif any(kw in query_lower for kw in ["climate change", "global warming", "greenhouse"]):
            return "CLIMATE_IMPACT"
        elif any(kw in query_lower for kw in ["season", "winter", "summer", "spring", "fall"]):
            return "SEASONAL_OUTLOOK"
        elif any(kw in query_lower for kw in ["history", "historical", "past", "record"]):
            return "HISTORICAL_ANALYSIS"
        elif any(kw in query_lower for kw in ["future", "projection", "forecast", "predict"]):
            return "FUTURE_PROJECTION"
        elif any(kw in query_lower for kw in ["compare", "comparison", "vs", "versus"]):
            return "COMPARATIVE_ANALYSIS"
        else:
            return "GENERAL_CLIMATE"

    async def analyze_trend(
        self,
        region: str,
        metric: str,
        time_period: str = "1980-present",
    ) -> dict[str, Any]:
        """Analyze climate trend for a specific metric.

        Args:
            region: Geographic region to analyze.
            metric: Climate metric (temperature, precipitation, etc.).
            time_period: Time period for analysis.

        Returns:
            Dictionary with trend analysis results.
        """
        messages = [
            SystemMessage(content=CLIMATE_SYSTEM_PROMPT),
            HumanMessage(
                content=TREND_ANALYSIS_PROMPT.format(
                    region=region,
                    metric=metric,
                    time_period=time_period,
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)

        return {
            "region": region,
            "metric": metric,
            "time_period": time_period,
            "analysis": str(response.content),
        }

    async def assess_climate_impact(
        self,
        phenomenon: str,
        region: str,
    ) -> dict[str, Any]:
        """Assess climate change impact on weather phenomenon.

        Args:
            phenomenon: Weather phenomenon to assess.
            region: Geographic region.

        Returns:
            Dictionary with impact assessment.
        """
        messages = [
            SystemMessage(content=CLIMATE_SYSTEM_PROMPT),
            HumanMessage(
                content=CLIMATE_IMPACT_PROMPT.format(
                    phenomenon=phenomenon,
                    region=region,
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)

        return {
            "phenomenon": phenomenon,
            "region": region,
            "assessment": str(response.content),
        }

    async def get_seasonal_outlook(
        self,
        region: str,
        season: str,
        year: int,
    ) -> dict[str, Any]:
        """Get seasonal climate outlook.

        Args:
            region: Geographic region.
            season: Season (Winter, Spring, Summer, Fall).
            year: Year for outlook.

        Returns:
            Dictionary with seasonal outlook.
        """
        messages = [
            SystemMessage(content=CLIMATE_SYSTEM_PROMPT),
            HumanMessage(
                content=SEASONAL_OUTLOOK_PROMPT.format(
                    region=region,
                    season=season,
                    year=year,
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)

        return {
            "region": region,
            "season": season,
            "year": year,
            "outlook": str(response.content),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about analyses performed.

        Returns:
            Dictionary with analysis statistics.
        """
        return {
            "total_analyses": self._analysis_count,
            "agent_role": self.agent_role.value,
        }


# =============================================================================
# Factory function
# =============================================================================


def create_climate_agent(
    model_name: str = "gpt-4o",
) -> ClimateAnalystAgent:
    """Create a Climate Analyst Agent instance.

    Args:
        model_name: Model to use for climate analysis.

    Returns:
        Configured ClimateAnalystAgent instance.
    """
    return ClimateAnalystAgent(model_name=model_name)


__all__ = [
    "ClimateAnalystAgent",
    "create_climate_agent",
    "CLIMATE_SYSTEM_PROMPT",
]
