"""Comprehensive tests for LongTermMemory (Graphiti + Neo4j).

Tests cover:
- Context manager usage
- Error handling for all Graphiti operations
- User profile creation and retrieval
- Temporal fact management
- Weather event tracking
- Connection initialization and cleanup
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.src.memory.exceptions import GraphitiMemoryError
from backend.src.memory.long_term import LongTermMemory
from backend.src.models.memory import TemporalFact, UserProfile


@pytest.fixture
def mock_graphiti():
    """Create a mocked Graphiti client."""
    mock = AsyncMock()
    mock.build_indices_and_constraints = AsyncMock()
    mock.add_episode = AsyncMock()
    mock.search = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def sample_user_profile():
    """Create a sample user profile."""
    return UserProfile(
        user_id="test_user",
        name="John Doe",
        home_location="London",
        preferred_units="celsius",
        preferred_detail_level="moderate",
        total_queries=10,
    )


@pytest.fixture
def sample_temporal_fact():
    """Create a sample temporal fact."""
    return TemporalFact(
        fact_id="fact_123",
        user_id="test_user",
        fact_type="preference",
        content="User prefers detailed forecasts",
        confidence=0.95,
        source="conversation",
        metadata={},
        valid_from=datetime.now(UTC),
        valid_to=None,
        created_at=datetime.now(UTC),
    )


class TestContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_entry_and_exit(self, mock_graphiti):
        """Test context manager properly enters and exits."""
        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            async with LongTermMemory() as memory:
                assert memory is not None
                assert isinstance(memory, LongTermMemory)
                assert memory._initialized is True  # Should be initialized on enter

            # Verify close() was called on exit
            mock_graphiti.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, mock_graphiti):
        """Test context manager properly closes even when exception occurs."""
        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            with pytest.raises(ValueError):
                async with LongTermMemory() as memory:
                    raise ValueError("Test exception")

            # Close should still be called
            mock_graphiti.close.assert_called_once()


class TestInitialization:
    """Test Graphiti initialization."""

    @pytest.mark.asyncio
    async def test_ensure_initialized_success(self, mock_graphiti):
        """Test successful Graphiti initialization."""
        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            await memory._ensure_initialized()

            assert memory._initialized is True
            mock_graphiti.build_indices_and_constraints.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_initialized_only_once(self, mock_graphiti):
        """Test initialization only happens once."""
        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            await memory._ensure_initialized()
            await memory._ensure_initialized()

            # Should only be called once
            assert mock_graphiti.build_indices_and_constraints.call_count == 1

    @pytest.mark.asyncio
    async def test_ensure_initialized_error(self, mock_graphiti):
        """Test initialization raises GraphitiMemoryError on failure."""
        mock_graphiti.build_indices_and_constraints.side_effect = Exception("Neo4j connection failed")

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()

            with pytest.raises(GraphitiMemoryError, match="Failed to initialize Graphiti"):
                await memory._ensure_initialized()


class TestCreateUserProfile:
    """Test create_user_profile method."""

    @pytest.mark.asyncio
    async def test_create_user_profile_success(self, mock_graphiti):
        """Test successful user profile creation."""
        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            await memory.create_user_profile(
                user_id="test_user",
                name="John Doe",
                location="London",
                preferences={"units": "celsius"},
            )

            # Verify add_episode was called
            mock_graphiti.add_episode.assert_called_once()
            call_args = mock_graphiti.add_episode.call_args
            assert "John Doe" in call_args[1]["episode_body"]
            assert "London" in call_args[1]["episode_body"]

    @pytest.mark.asyncio
    async def test_create_user_profile_error(self, mock_graphiti):
        """Test create_user_profile raises GraphitiMemoryError on failure."""
        mock_graphiti.add_episode.side_effect = Exception("Episode ingestion failed")

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()

            with pytest.raises(GraphitiMemoryError, match="Failed to create user profile"):
                await memory.create_user_profile(user_id="test_user", name="John")


class TestGetUserProfile:
    """Test get_user_profile method."""

    @pytest.mark.asyncio
    async def test_get_user_profile_found(self, mock_graphiti):
        """Test successful user profile retrieval."""
        # Mock search results
        mock_edge = Mock()
        mock_edge.fact = "User John Doe (ID: test_user) lives in London and prefers celsius units"
        mock_graphiti.search.return_value = [mock_edge]

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            profile = await memory.get_user_profile("test_user")

            assert profile is not None
            assert profile.user_id == "test_user"
            assert profile.home_location == "London"

    @pytest.mark.asyncio
    async def test_get_user_profile_not_found(self, mock_graphiti):
        """Test get_user_profile returns None when not found."""
        mock_graphiti.search.return_value = []

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            profile = await memory.get_user_profile("test_user")

            assert profile is None

    @pytest.mark.asyncio
    async def test_get_user_profile_error(self, mock_graphiti):
        """Test get_user_profile raises GraphitiMemoryError on failure."""
        mock_graphiti.search.side_effect = Exception("Search failed")

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()

            with pytest.raises(GraphitiMemoryError, match="Failed to search user profile"):
                await memory.get_user_profile("test_user")


class TestUpdateUserPreference:
    """Test update_user_preference method."""

    @pytest.mark.asyncio
    async def test_update_preference_success(self, mock_graphiti):
        """Test successful preference update."""
        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            await memory.update_user_preference("test_user", "units", "fahrenheit")

            # Verify add_episode was called
            mock_graphiti.add_episode.assert_called_once()
            call_args = mock_graphiti.add_episode.call_args
            assert "fahrenheit" in call_args[1]["episode_body"]
            assert "units" in call_args[1]["episode_body"]

    @pytest.mark.asyncio
    async def test_update_preference_error(self, mock_graphiti):
        """Test update_preference raises GraphitiMemoryError on failure."""
        mock_graphiti.add_episode.side_effect = Exception("Episode failed")

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()

            with pytest.raises(GraphitiMemoryError, match="Failed to update user preference"):
                await memory.update_user_preference("test_user", "units", "fahrenheit")


class TestTrackWeatherEvent:
    """Test track_weather_event method."""

    @pytest.mark.asyncio
    async def test_track_weather_event_success(self, mock_graphiti):
        """Test successful weather event tracking."""
        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            await memory.track_weather_event(
                location="London",
                weather_type="rain",
                conditions={"temp": 18, "humidity": 85},
                valid_from=datetime.now(UTC),
                valid_to=None,
            )

            # Verify add_episode was called
            mock_graphiti.add_episode.assert_called_once()
            call_args = mock_graphiti.add_episode.call_args
            assert "London" in call_args[1]["episode_body"]
            assert "rain" in call_args[1]["episode_body"]

    @pytest.mark.asyncio
    async def test_track_weather_event_error(self, mock_graphiti):
        """Test track_weather_event raises GraphitiMemoryError on failure."""
        mock_graphiti.add_episode.side_effect = Exception("Episode failed")

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()

            with pytest.raises(GraphitiMemoryError, match="Failed to track weather event"):
                await memory.track_weather_event(
                    location="London",
                    weather_type="rain",
                    conditions={},
                    valid_from=datetime.now(UTC),
                )


class TestAddTemporalFact:
    """Test add_temporal_fact method."""

    @pytest.mark.asyncio
    async def test_add_temporal_fact_success(self, mock_graphiti, sample_temporal_fact):
        """Test successful temporal fact addition."""
        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            await memory.add_temporal_fact(sample_temporal_fact)

            # Verify add_episode was called
            mock_graphiti.add_episode.assert_called_once()
            call_args = mock_graphiti.add_episode.call_args
            assert sample_temporal_fact.content in call_args[1]["episode_body"]

    @pytest.mark.asyncio
    async def test_add_temporal_fact_error(self, mock_graphiti, sample_temporal_fact):
        """Test add_temporal_fact raises GraphitiMemoryError on failure."""
        mock_graphiti.add_episode.side_effect = Exception("Episode failed")

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()

            with pytest.raises(GraphitiMemoryError, match="Failed to add temporal fact"):
                await memory.add_temporal_fact(sample_temporal_fact)


class TestGetTemporalFacts:
    """Test get_temporal_facts method."""

    @pytest.mark.asyncio
    async def test_get_temporal_facts_success(self, mock_graphiti):
        """Test successful temporal facts retrieval."""
        # Mock search results
        mock_edge = Mock()
        mock_edge.fact = "User test_user: prefers detailed forecasts (type: preference, confidence: 0.95, valid from 2025-01-21)"
        mock_edge.uuid = "fact_123"
        mock_edge.valid_at = datetime.now(UTC)
        mock_edge.invalid_at = None
        mock_edge.created_at = datetime.now(UTC)
        mock_graphiti.search.return_value = [mock_edge]

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            facts = await memory.get_temporal_facts("test_user")

            assert len(facts) == 1
            assert facts[0].user_id == "test_user"
            assert "forecasts" in facts[0].content

    @pytest.mark.asyncio
    async def test_get_temporal_facts_empty(self, mock_graphiti):
        """Test get_temporal_facts returns empty list when none found."""
        mock_graphiti.search.return_value = []

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            facts = await memory.get_temporal_facts("test_user")

            assert facts == []

    @pytest.mark.asyncio
    async def test_get_temporal_facts_error(self, mock_graphiti):
        """Test get_temporal_facts raises GraphitiMemoryError on failure."""
        mock_graphiti.search.side_effect = Exception("Search failed")

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()

            with pytest.raises(GraphitiMemoryError, match="Failed to get temporal facts"):
                await memory.get_temporal_facts("test_user")


class TestClose:
    """Test close method."""

    @pytest.mark.asyncio
    async def test_close_success(self, mock_graphiti):
        """Test successful Graphiti connection close."""
        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()
            await memory.close()

            mock_graphiti.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_error(self, mock_graphiti):
        """Test close raises GraphitiMemoryError on failure."""
        mock_graphiti.close.side_effect = Exception("Close failed")

        with patch("backend.src.memory.long_term.Graphiti", return_value=mock_graphiti):
            memory = LongTermMemory()

            with pytest.raises(GraphitiMemoryError, match="Failed to close Graphiti connection"):
                await memory.close()
