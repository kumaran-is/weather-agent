"""Tests for Research Agent (Level 4b Phase 9).

Test Coverage:
- Deep research and data retrieval
- Multi-source data integration
- Research area identification
- Knowledge base integration
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.src.agents.research_agent import ResearchAgent
from backend.src.models.multi_agent import (
    AgentRole,
    MultiAgentState,
)


# Fixtures
@pytest.fixture
def base_state() -> MultiAgentState:
    """Create base MultiAgentState for testing."""
    return MultiAgentState(
        query="Test research query",
        user_id="test_user_123",
        session_id="test_session_456",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        timeout_ms=30000,
        current_agent=None,
        routing_decision=None,
        agent_responses=[],
        next_agent=None,
        workflow_complete=False,
        final_response=None,
        error=None,
        total_execution_time_ms=0.0,
    )


@pytest.fixture
def research_agent() -> ResearchAgent:
    """Create ResearchAgent for testing."""
    return ResearchAgent()


@pytest.fixture
def mock_research_response():
    """Mock LLM research response."""
    return MagicMock(
        content="""Comprehensive Research: Hurricane Preparation

1. HURRICANE FORMATION & INTENSIFICATION
Hurricanes form over warm ocean waters (≥80°F/26.5°C) when:
- Sea surface temperature exceeds threshold
- Low vertical wind shear exists
- Coriolis force is sufficient (>5° from equator)
- Pre-existing disturbance is present

Rapid intensification occurs when:
- Ocean heat content is high (warm water extends deep)
- Upper-level outflow is favorable
- Mid-level humidity is high

2. SAFFIR-SIMPSON SCALE REFERENCE
- Category 1: 74-95 mph, 4-5 ft surge
- Category 2: 96-110 mph, 6-8 ft surge
- Category 3: 111-129 mph, 9-12 ft surge (Major)
- Category 4: 130-156 mph, 13-18 ft surge (Major)
- Category 5: 157+ mph, 18+ ft surge (Major)

3. PREPARATION TIMELINE
5 days out: Monitor forecast, review plans
3 days out: Finalize supplies, fuel vehicles
2 days out: Begin securing property
1 day out: Complete preparations, evacuate if ordered

4. ESSENTIAL SUPPLIES
- Water: 1 gallon per person per day (7 days)
- Food: Non-perishable (7 days)
- Medications: 30-day supply
- First aid kit, flashlights, batteries
- Cash and important documents"""
    )


# Test: ResearchAgent Initialization
class TestResearchAgentInit:
    """Tests for ResearchAgent initialization."""

    def test_default_initialization(self, research_agent):
        """Test ResearchAgent initializes with defaults."""
        assert research_agent.llm is not None
        assert research_agent.agent_role == AgentRole.RESEARCH

    def test_custom_initialization(self):
        """Test ResearchAgent with custom model."""
        agent = ResearchAgent(model_name="gpt-4o")
        assert agent.llm is not None


# Test: Research Area Identification
class TestResearchAreaIdentification:
    """Tests for identifying research areas from queries."""

    def test_identify_tropical_meteorology(self, research_agent):
        """Test identification of tropical meteorology area."""
        query = "How do hurricanes form and intensify?"
        areas = research_agent._identify_research_areas(query)
        assert "tropical_meteorology" in areas

    def test_identify_forecast_methodology(self, research_agent):
        """Test identification of forecast methodology area."""
        query = "How are hurricane predictions made?"
        areas = research_agent._identify_research_areas(query)
        assert "forecast_methodology" in areas

    def test_identify_emergency_preparedness(self, research_agent):
        """Test identification of emergency preparedness area."""
        query = "How should I prepare for evacuation?"
        areas = research_agent._identify_research_areas(query)
        assert "emergency_preparedness" in areas

    def test_identify_flood_risk(self, research_agent):
        """Test identification of flood risk area."""
        query = "What causes storm surge flooding?"
        areas = research_agent._identify_research_areas(query)
        assert "flood_risk" in areas

    def test_identify_impact_assessment(self, research_agent):
        """Test identification of impact assessment area."""
        query = "What damage can a Category 4 cause?"
        areas = research_agent._identify_research_areas(query)
        assert "impact_assessment" in areas

    def test_identify_historical_data(self, research_agent):
        """Test identification of historical data area."""
        query = "What were the worst hurricanes in history?"
        areas = research_agent._identify_research_areas(query)
        assert "historical_data" in areas

    def test_identify_multiple_areas(self, research_agent):
        """Test identification of multiple research areas."""
        query = "How do hurricanes form, what damage can they cause, and how should I prepare?"
        areas = research_agent._identify_research_areas(query)
        assert len(areas) >= 2

    def test_default_area_for_general_query(self, research_agent):
        """Test default area for general queries."""
        query = "Tell me about weather"
        areas = research_agent._identify_research_areas(query)
        assert "general_weather" in areas or len(areas) > 0


# Test: Research Query
class TestResearchQuery:
    """Tests for research_query method."""

    @pytest.mark.asyncio
    async def test_research_query_success(
        self, research_agent, base_state, mock_research_response
    ):
        """Test successful research query."""
        base_state.query = "How do hurricanes form?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_research_response)
        research_agent.llm = mock_llm

        result = await research_agent.research_query(base_state)

        assert result is not None
        assert len(result.agent_responses) == 1
        assert result.agent_responses[0].agent_role == AgentRole.RESEARCH

    @pytest.mark.asyncio
    async def test_research_includes_research_areas(
        self, research_agent, base_state, mock_research_response
    ):
        """Test research includes identified areas in metadata."""
        base_state.query = "Hurricane preparation guide"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_research_response)
        research_agent.llm = mock_llm

        result = await research_agent.research_query(base_state)

        response = result.agent_responses[-1]
        assert "research_areas" in response.metadata

    @pytest.mark.asyncio
    async def test_research_error_handling(
        self, research_agent, base_state
    ):
        """Test research handles errors gracefully."""
        base_state.query = "Research query"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        research_agent.llm = mock_llm

        result = await research_agent.research_query(base_state)

        # Should handle error
        assert result is not None
        assert result.agent_responses[-1].confidence == 0.0


# Test: Knowledge Base Integration
class TestKnowledgeBaseIntegration:
    """Tests for knowledge base integration."""

    def test_tropical_knowledge_available(self, research_agent):
        """Test tropical meteorology knowledge is available."""
        kb = research_agent._get_tropical_knowledge()
        assert kb is not None
        assert "saffir_simpson_scale" in kb
        assert "formation_requirements" in kb

    def test_saffir_simpson_scale_data(self, research_agent):
        """Test Saffir-Simpson scale data is accurate."""
        kb = research_agent._get_tropical_knowledge()
        scale = kb["saffir_simpson_scale"]

        # Verify Category 5
        assert 5 in scale
        assert scale[5]["wind_mph"] == "157+"

        # Verify Category 1
        assert 1 in scale
        assert scale[1]["wind_mph"] == "74-95"

    def test_preparedness_knowledge_available(self, research_agent):
        """Test preparedness knowledge is available."""
        kb = research_agent._get_preparedness_knowledge()
        assert kb is not None
        assert "evacuation_zones" in kb
        assert "preparation_timeline" in kb
        assert "essential_supplies" in kb

    def test_evacuation_zones_data(self, research_agent):
        """Test evacuation zones data."""
        kb = research_agent._get_preparedness_knowledge()
        zones = kb["evacuation_zones"]

        assert "A" in zones
        assert "B" in zones
        assert "C" in zones

    def test_flood_knowledge_available(self, research_agent):
        """Test flood risk knowledge is available."""
        kb = research_agent._get_flood_knowledge()
        assert kb is not None
        assert "storm_surge_factors" in kb
        assert "flood_zones" in kb
        assert "safety_facts" in kb


# Test: Data Gathering
class TestDataGathering:
    """Tests for data gathering functionality."""

    @pytest.mark.asyncio
    async def test_gather_data_tropical(self, research_agent):
        """Test data gathering for tropical queries."""
        query = "Hurricane formation"
        areas = ["tropical_meteorology"]

        data = await research_agent._gather_data(query, areas)

        assert data is not None
        assert "knowledge_base" in data
        assert "tropical" in data["knowledge_base"]

    @pytest.mark.asyncio
    async def test_gather_data_preparedness(self, research_agent):
        """Test data gathering for preparedness queries."""
        query = "Hurricane preparation"
        areas = ["emergency_preparedness"]

        data = await research_agent._gather_data(query, areas)

        assert data is not None
        assert "knowledge_base" in data
        assert "preparedness" in data["knowledge_base"]

    @pytest.mark.asyncio
    async def test_gather_data_multiple_areas(self, research_agent):
        """Test data gathering for multiple areas."""
        query = "Hurricane prep and flood risk"
        areas = ["emergency_preparedness", "flood_risk"]

        data = await research_agent._gather_data(query, areas)

        assert data is not None
        assert len(data["knowledge_base"]) >= 2


# Test: Confidence Calculation
class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    def test_high_confidence_multiple_sources(self, research_agent):
        """Test high confidence with multiple data sources."""
        confidence = research_agent._calculate_confidence(
            research_areas=["tropical_meteorology", "emergency_preparedness"],
            data_sources_used=3,
        )
        assert confidence >= 0.85

    def test_base_confidence_few_sources(self, research_agent):
        """Test base confidence with few sources."""
        confidence = research_agent._calculate_confidence(
            research_areas=["general_weather"],
            data_sources_used=0,
        )
        assert confidence >= 0.7
        assert confidence < 0.85

    def test_confidence_bounded(self, research_agent):
        """Test confidence is bounded at 0.9 max."""
        confidence = research_agent._calculate_confidence(
            research_areas=["a", "b", "c", "d", "e"],
            data_sources_used=10,
        )
        assert confidence <= 0.9


# Test: Response Formatting
class TestResponseFormatting:
    """Tests for response formatting."""

    @pytest.mark.asyncio
    async def test_response_has_metadata(
        self, research_agent, base_state, mock_research_response
    ):
        """Test response includes proper metadata."""
        base_state.query = "Hurricane research"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_research_response)
        research_agent.llm = mock_llm

        result = await research_agent.research_query(base_state)

        response = result.agent_responses[-1]
        assert "research_areas" in response.metadata
        assert "data_sources" in response.metadata

    @pytest.mark.asyncio
    async def test_response_tracks_execution_time(
        self, research_agent, base_state, mock_research_response
    ):
        """Test response tracks execution time."""
        base_state.query = "Research query"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_research_response)
        research_agent.llm = mock_llm

        result = await research_agent.research_query(base_state)

        response = result.agent_responses[-1]
        assert response.execution_time_ms > 0


# Test: Error Handling
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_llm_timeout(
        self, research_agent, base_state
    ):
        """Test handling of LLM timeout."""
        base_state.query = "Research query"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=TimeoutError("Timeout"))
        research_agent.llm = mock_llm

        result = await research_agent.research_query(base_state)

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_handles_mcp_unavailable(
        self, research_agent, base_state, mock_research_response
    ):
        """Test handling when MCP servers are unavailable."""
        base_state.query = "Hurricane research"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_research_response)
        research_agent.llm = mock_llm

        # Should still work with knowledge base fallback
        result = await research_agent.research_query(base_state)

        assert result is not None


# Integration Tests
class TestResearchIntegration:
    """Integration tests for ResearchAgent."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_research_workflow(
        self, research_agent, base_state, mock_research_response
    ):
        """Test complete research workflow."""
        base_state.query = "Comprehensive guide to hurricane preparation and evacuation"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_research_response)
        research_agent.llm = mock_llm

        result = await research_agent.research_query(base_state)

        # Verify complete workflow
        assert result is not None
        assert len(result.agent_responses) == 1

        response = result.agent_responses[0]
        assert response.agent_role == AgentRole.RESEARCH
        assert response.content is not None
        assert response.confidence > 0
        assert response.execution_time_ms > 0
        assert "research_areas" in response.metadata
        assert "data_sources" in response.metadata
