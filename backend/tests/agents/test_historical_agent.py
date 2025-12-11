"""Tests for Historical Analyst Agent (Level 4b Phase 9).

Test Coverage:
- Historical hurricane pattern analysis
- Pattern detection and comparison
- Historical database queries
- Trend analysis functionality
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from backend.src.agents.historical_agent import HistoricalAnalystAgent
from backend.src.models.multi_agent import (
    AgentRole,
    AgentResponse,
    MultiAgentState,
)


# Fixtures
@pytest.fixture
def base_state() -> MultiAgentState:
    """Create base MultiAgentState for testing."""
    return MultiAgentState(
        query="Test historical query",
        user_id="test_user_123",
        session_id="test_session_456",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
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
def historical_agent() -> HistoricalAnalystAgent:
    """Create HistoricalAnalystAgent for testing."""
    return HistoricalAnalystAgent()


@pytest.fixture
def mock_analysis_response():
    """Mock LLM analysis response."""
    return MagicMock(
        content="""Historical Analysis: Hurricane Michael (2018)

OVERVIEW:
Hurricane Michael made landfall near Mexico Beach, Florida on October 10, 2018 as a Category 5 hurricane with maximum sustained winds of 160 mph.

KEY STATISTICS:
- Peak Intensity: Category 5 (160 mph)
- Landfall Category: Category 5
- Central Pressure: 919 mb (one of lowest on record for US landfall)
- Deaths: 16 direct, 43 indirect
- Damage: $25.1 billion

HISTORICAL CONTEXT:
- First Category 5 landfall in Florida Panhandle since records began
- Only fourth Category 5 to make US landfall since 1992
- Rapid intensification: Strengthened from Cat 2 to Cat 5 in 24 hours

COMPARISON TO SIMILAR STORMS:
- Intensity similar to Hurricane Andrew (1992): Cat 5, 165 mph
- Track similar to Hurricane Opal (1995): Panhandle landfall
- Damage exceeded Hurricane Ivan (2004) in same region"""
    )


# Test: HistoricalAnalystAgent Initialization
class TestHistoricalAgentInit:
    """Tests for HistoricalAnalystAgent initialization."""

    def test_default_initialization(self, historical_agent):
        """Test HistoricalAnalystAgent initializes with defaults."""
        assert historical_agent.llm is not None
        assert historical_agent.agent_role == AgentRole.HISTORICAL_ANALYST

    def test_has_historical_database(self, historical_agent):
        """Test agent has historical hurricane database."""
        assert hasattr(historical_agent, 'HISTORICAL_HURRICANES')
        assert len(historical_agent.HISTORICAL_HURRICANES) > 0

    def test_custom_initialization(self):
        """Test HistoricalAnalystAgent with custom model."""
        agent = HistoricalAnalystAgent(model_name="gpt-4o")
        assert agent.llm is not None


# Test: Historical Database
class TestHistoricalDatabase:
    """Tests for historical hurricane database."""

    def test_database_contains_michael(self, historical_agent):
        """Test database contains Hurricane Michael."""
        assert "michael_2018" in historical_agent.HISTORICAL_HURRICANES
        michael = historical_agent.HISTORICAL_HURRICANES["michael_2018"]
        assert michael["name"] == "Michael"
        assert michael["year"] == 2018
        assert michael["category"] == 5

    def test_database_contains_ian(self, historical_agent):
        """Test database contains Hurricane Ian."""
        assert "ian_2022" in historical_agent.HISTORICAL_HURRICANES
        ian = historical_agent.HISTORICAL_HURRICANES["ian_2022"]
        assert ian["name"] == "Ian"
        assert ian["year"] == 2022

    def test_database_contains_irma(self, historical_agent):
        """Test database contains Hurricane Irma."""
        assert "irma_2017" in historical_agent.HISTORICAL_HURRICANES
        irma = historical_agent.HISTORICAL_HURRICANES["irma_2017"]
        assert irma["name"] == "Irma"
        assert irma["category"] == 5

    def test_database_contains_andrew(self, historical_agent):
        """Test database contains Hurricane Andrew."""
        assert "andrew_1992" in historical_agent.HISTORICAL_HURRICANES
        andrew = historical_agent.HISTORICAL_HURRICANES["andrew_1992"]
        assert andrew["year"] == 1992
        assert andrew["category"] == 5


# Test: Storm Lookup
class TestStormLookup:
    """Tests for historical storm lookup."""

    def test_find_storm_by_name(self, historical_agent):
        """Test finding storm by name."""
        storm = historical_agent._find_storm("Michael")
        assert storm is not None
        assert storm["name"] == "Michael"

    def test_find_storm_by_name_year(self, historical_agent):
        """Test finding storm by name and year."""
        storm = historical_agent._find_storm("Michael", year=2018)
        assert storm is not None
        assert storm["year"] == 2018

    def test_find_nonexistent_storm(self, historical_agent):
        """Test finding non-existent storm returns None."""
        storm = historical_agent._find_storm("NonExistent")
        assert storm is None

    def test_find_storm_case_insensitive(self, historical_agent):
        """Test storm lookup is case insensitive."""
        storm = historical_agent._find_storm("MICHAEL")
        assert storm is not None
        assert storm["name"] == "Michael"


# Test: Pattern Analysis
class TestPatternAnalysis:
    """Tests for pattern analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_patterns_success(
        self, historical_agent, base_state, mock_analysis_response
    ):
        """Test successful pattern analysis."""
        base_state.query = "What was Hurricane Michael's impact?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_analysis_response)
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        assert result is not None
        assert len(result.agent_responses) == 1
        assert result.agent_responses[0].agent_role == AgentRole.HISTORICAL_ANALYST

    @pytest.mark.asyncio
    async def test_analyze_comparison_query(
        self, historical_agent, base_state
    ):
        """Test analysis with comparison query."""
        base_state.query = "Compare Hurricane Michael to Hurricane Andrew"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Comparison: Both were Category 5 hurricanes..."
        ))
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        assert result is not None
        response = result.agent_responses[-1]
        # Should identify multiple storms
        assert response.metadata.get("storms_analyzed", 0) >= 1

    @pytest.mark.asyncio
    async def test_analyze_trend_query(
        self, historical_agent, base_state
    ):
        """Test analysis of historical trends."""
        base_state.query = "What are the hurricane trends for Florida Panhandle?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Historical trends: Category 5 landfalls are rare but..."
        ))
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        assert result is not None


# Test: Storm Identification
class TestStormIdentification:
    """Tests for identifying storms in queries."""

    def test_identify_single_storm(self, historical_agent):
        """Test identifying single storm in query."""
        query = "Tell me about Hurricane Michael"
        storms = historical_agent._identify_storms_in_query(query)
        assert len(storms) >= 1
        assert any(s["name"] == "Michael" for s in storms)

    def test_identify_multiple_storms(self, historical_agent):
        """Test identifying multiple storms in query."""
        query = "Compare Hurricane Michael to Hurricane Ian"
        storms = historical_agent._identify_storms_in_query(query)
        assert len(storms) >= 2

    def test_identify_storm_with_year(self, historical_agent):
        """Test identifying storm with year specified."""
        query = "Hurricane Michael 2018 analysis"
        storms = historical_agent._identify_storms_in_query(query)
        assert len(storms) >= 1
        if storms:
            assert storms[0]["year"] == 2018


# Test: Confidence Calculation
class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    @pytest.mark.asyncio
    async def test_high_confidence_with_known_storm(
        self, historical_agent, base_state, mock_analysis_response
    ):
        """Test high confidence for known storm."""
        base_state.query = "What was Hurricane Michael's peak intensity?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_analysis_response)
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        response = result.agent_responses[-1]
        assert response.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_lower_confidence_for_general_query(
        self, historical_agent, base_state
    ):
        """Test lower confidence for general queries."""
        base_state.query = "What are some historical hurricane patterns?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="General historical patterns..."
        ))
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        response = result.agent_responses[-1]
        # Should still have reasonable confidence
        assert response.confidence >= 0.5


# Test: Data Retrieval
class TestDataRetrieval:
    """Tests for historical data retrieval."""

    def test_get_storm_statistics(self, historical_agent):
        """Test retrieving storm statistics."""
        stats = historical_agent._get_storm_statistics("michael_2018")
        assert stats is not None
        assert "category" in stats
        assert "peak_winds_mph" in stats

    def test_get_storms_by_category(self, historical_agent):
        """Test filtering storms by category."""
        cat5_storms = historical_agent._get_storms_by_category(5)
        assert len(cat5_storms) >= 3  # Michael, Andrew, Irma

    def test_get_storms_by_year_range(self, historical_agent):
        """Test filtering storms by year range."""
        recent_storms = historical_agent._get_storms_in_year_range(2015, 2023)
        assert len(recent_storms) >= 2  # Irma 2017, Michael 2018, Ian 2022


# Test: Error Handling
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_llm_error(
        self, historical_agent, base_state
    ):
        """Test handling of LLM errors."""
        base_state.query = "Hurricane analysis"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        # Should handle gracefully
        assert result is not None
        assert result.agent_responses[-1].confidence == 0.0

    @pytest.mark.asyncio
    async def test_handles_unknown_storm(
        self, historical_agent, base_state
    ):
        """Test handling of unknown storm queries."""
        base_state.query = "What was Hurricane Zzzz's impact?"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="No historical data found for that storm."
        ))
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        assert result is not None


# Test: Response Metadata
class TestResponseMetadata:
    """Tests for response metadata."""

    @pytest.mark.asyncio
    async def test_metadata_includes_storms_analyzed(
        self, historical_agent, base_state, mock_analysis_response
    ):
        """Test metadata includes storms analyzed."""
        base_state.query = "Hurricane Michael analysis"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_analysis_response)
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        response = result.agent_responses[-1]
        assert "storms_analyzed" in response.metadata

    @pytest.mark.asyncio
    async def test_metadata_includes_data_sources(
        self, historical_agent, base_state, mock_analysis_response
    ):
        """Test metadata includes data sources."""
        base_state.query = "Hurricane Michael historical data"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_analysis_response)
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        response = result.agent_responses[-1]
        assert "data_sources" in response.metadata


# Integration Tests
class TestHistoricalIntegration:
    """Integration tests for HistoricalAnalystAgent."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_analysis_workflow(
        self, historical_agent, base_state, mock_analysis_response
    ):
        """Test complete historical analysis workflow."""
        base_state.query = "Compare Hurricane Michael 2018 landfall patterns to Hurricane Andrew 1992"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_analysis_response)
        historical_agent.llm = mock_llm

        result = await historical_agent.analyze_patterns(base_state)

        # Verify complete workflow
        assert result is not None
        assert len(result.agent_responses) == 1

        response = result.agent_responses[0]
        assert response.agent_role == AgentRole.HISTORICAL_ANALYST
        assert response.content is not None
        assert response.confidence > 0
        assert response.execution_time_ms > 0
