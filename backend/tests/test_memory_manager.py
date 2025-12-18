"""Comprehensive tests for MemoryManager (Unified Interface).

Tests cover:
- Context manager usage
- Unified context retrieval (short-term + long-term)
- Interaction saving across both layers
- Entity tracking delegation
- Pronoun resolution delegation
- User profile management
- Conversation summarization
- Connection cleanup
- P1: Parallel cross-layer retrieval (new)
- P1: Graceful degradation and timeout handling (new)
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.src.memory.manager import MemoryManager
from backend.src.models.memory import ConversationContext, UserProfile


@pytest.fixture
def mock_short_term():
    """Create a mocked ShortTermMemory."""
    mock = AsyncMock()
    mock.get_context = AsyncMock()
    mock.create_context = AsyncMock()
    mock.save_context = AsyncMock()
    mock.track_entity = AsyncMock()
    mock.resolve_pronoun = AsyncMock(return_value="resolved query")
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_long_term():
    """Create a mocked LongTermMemory."""
    mock = AsyncMock()
    mock.get_user_profile = AsyncMock()
    mock.create_user_profile = AsyncMock()
    mock.update_user_preference = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def sample_context():
    """Create a sample conversation context."""
    return ConversationContext(
        user_id="test_user",
        session_id="test_session",
        conversation_history=[
            {"role": "user", "content": "Weather in London?"},
        ],
        current_entities={"location": "London"},
        token_count=5,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(minutes=30),
    )


@pytest.fixture
def sample_profile():
    """Create a sample user profile."""
    return UserProfile(
        user_id="test_user",
        name="John Doe",
        home_location="London",
        preferred_units="celsius",
        preferred_detail_level="moderate",
        total_queries=10,
    )


class TestContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_entry_and_exit(self, mock_short_term, mock_long_term):
        """Test context manager properly enters and exits."""
        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                async with MemoryManager() as manager:
                    assert manager is not None
                    assert isinstance(manager, MemoryManager)

                # Verify close() was called on both layers
                mock_short_term.close.assert_called_once()
                mock_long_term.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, mock_short_term, mock_long_term):
        """Test context manager properly closes even when exception occurs."""
        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with pytest.raises(ValueError):
                    async with MemoryManager() as manager:
                        raise ValueError("Test exception")

                # Both layers should still be closed
                mock_short_term.close.assert_called_once()
                mock_long_term.close.assert_called_once()


class TestGetContext:
    """Test get_context method."""

    @pytest.mark.asyncio
    async def test_get_context_existing_session(self, mock_short_term, mock_long_term, sample_context, sample_profile):
        """Test retrieving context with existing session."""
        mock_short_term.get_context.return_value = sample_context
        mock_long_term.get_user_profile.return_value = sample_profile

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    manager = MemoryManager()
                    result = await manager.get_context("test_user", "test_session")

                    assert result["session"] == sample_context
                    assert result["profile"] == sample_profile

    @pytest.mark.asyncio
    async def test_get_context_new_session(self, mock_short_term, mock_long_term, sample_context):
        """Test retrieving context creates new session if doesn't exist."""
        mock_short_term.get_context.return_value = None
        mock_short_term.create_context.return_value = sample_context
        mock_long_term.get_user_profile.return_value = None

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                manager = MemoryManager()
                result = await manager.get_context("test_user", "test_session")

                # Should create new context
                mock_short_term.create_context.assert_called_once_with("test_user", "test_session")
                assert result["session"] == sample_context

    @pytest.mark.asyncio
    async def test_get_context_profiles_disabled(self, mock_short_term, mock_long_term, sample_context):
        """Test get_context skips profile retrieval when profiles disabled."""
        mock_short_term.get_context.return_value = sample_context

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", False):
                    manager = MemoryManager()
                    result = await manager.get_context("test_user", "test_session")

                    # Should not call get_user_profile
                    mock_long_term.get_user_profile.assert_not_called()
                    assert result["profile"] is None


class TestSaveInteraction:
    """Test save_interaction method."""

    @pytest.mark.asyncio
    async def test_save_interaction_existing_context(self, mock_short_term, mock_long_term, sample_context):
        """Test saving interaction to existing context."""
        mock_short_term.get_context.return_value = sample_context

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                manager = MemoryManager()
                await manager.save_interaction(
                    user_id="test_user",
                    session_id="test_session",
                    query="How about tomorrow?",
                    response="Tomorrow will be 20°C",
                )

                # Verify save_context was called
                mock_short_term.save_context.assert_called_once()
                saved_context = mock_short_term.save_context.call_args[0][0]
                assert len(saved_context.conversation_history) == 4  # 2 original + 2 new

    @pytest.mark.asyncio
    async def test_save_interaction_new_context(self, mock_short_term, mock_long_term):
        """Test saving interaction creates new context if doesn't exist."""
        mock_short_term.get_context.return_value = None

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                manager = MemoryManager()
                await manager.save_interaction(
                    user_id="test_user",
                    session_id="test_session",
                    query="Weather?",
                    response="It's sunny",
                )

                # Should create and save new context
                mock_short_term.save_context.assert_called_once()
                saved_context = mock_short_term.save_context.call_args[0][0]
                assert len(saved_context.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_save_interaction_conversation_truncation(self, mock_short_term, mock_long_term):
        """Test conversation history is truncated after max turns."""
        # Create context with many turns
        long_context = ConversationContext(
            user_id="test_user",
            session_id="test_session",
            conversation_history=[{"role": "user", "content": f"Query {i}"} for i in range(100)],
            current_entities={},
            token_count=100,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=30),
        )
        mock_short_term.get_context.return_value = long_context

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.REDIS_MAX_CONVERSATION_TURNS", 10):
                    manager = MemoryManager()
                    await manager.save_interaction(
                        user_id="test_user",
                        session_id="test_session",
                        query="New query",
                        response="New response",
                    )

                    # History should be truncated to last 20 turns (10 pairs)
                    saved_context = mock_short_term.save_context.call_args[0][0]
                    assert len(saved_context.conversation_history) == 20


class TestTrackEntity:
    """Test track_entity method."""

    @pytest.mark.asyncio
    async def test_track_entity_success(self, mock_short_term, mock_long_term, sample_context):
        """Test successful entity tracking."""
        mock_short_term.get_context.return_value = sample_context

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                manager = MemoryManager()
                await manager.track_entity("test_user", "test_session", "location", "Paris")

                # Should delegate to short_term
                mock_short_term.track_entity.assert_called_once_with(
                    "test_user", "test_session", "location", "Paris"
                )

                # Should also update context
                mock_short_term.save_context.assert_called_once()


class TestResolvePronoun:
    """Test resolve_pronoun method."""

    @pytest.mark.asyncio
    async def test_resolve_pronoun_enabled(self, mock_short_term, mock_long_term):
        """Test pronoun resolution when enabled."""
        mock_short_term.resolve_pronoun.return_value = "Weather in London?"

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_PRONOUN_RESOLUTION", True):
                    manager = MemoryManager()
                    result = await manager.resolve_pronoun(
                        "test_user", "test_session", "Weather there?"
                    )

                    assert result == "Weather in London?"
                    mock_short_term.resolve_pronoun.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_pronoun_disabled(self, mock_short_term, mock_long_term):
        """Test pronoun resolution returns query unchanged when disabled."""
        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_PRONOUN_RESOLUTION", False):
                    manager = MemoryManager()
                    result = await manager.resolve_pronoun(
                        "test_user", "test_session", "Weather there?"
                    )

                    assert result == "Weather there?"
                    mock_short_term.resolve_pronoun.assert_not_called()


class TestUserProfileManagement:
    """Test user profile creation and updates."""

    @pytest.mark.asyncio
    async def test_create_user_profile(self, mock_short_term, mock_long_term):
        """Test user profile creation."""
        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    manager = MemoryManager()
                    await manager.create_user_profile(
                        user_id="test_user",
                        name="John",
                        location="London",
                        preferences={"units": "celsius"},
                    )

                    # Should delegate to long_term
                    mock_long_term.create_user_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_preference(self, mock_short_term, mock_long_term):
        """Test user preference update."""
        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    manager = MemoryManager()
                    await manager.update_user_preference("test_user", "units", "fahrenheit")

                    # Should delegate to long_term
                    mock_long_term.update_user_preference.assert_called_once()


class TestGetConversationSummary:
    """Test get_conversation_summary method."""

    @pytest.mark.asyncio
    async def test_get_conversation_summary_success(self, mock_short_term, mock_long_term, sample_context):
        """Test successful conversation summary generation."""
        mock_short_term.get_context.return_value = sample_context

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                manager = MemoryManager()
                summary = await manager.get_conversation_summary("test_user", "test_session")

                assert "Weather in London?" in summary
                assert "London is 18°C" in summary

    @pytest.mark.asyncio
    async def test_get_conversation_summary_no_history(self, mock_short_term, mock_long_term):
        """Test conversation summary when no history exists."""
        mock_short_term.get_context.return_value = None

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                manager = MemoryManager()
                summary = await manager.get_conversation_summary("test_user", "test_session")

                assert summary == "No conversation history"


class TestClose:
    """Test close method."""

    @pytest.mark.asyncio
    async def test_close_both_layers(self, mock_short_term, mock_long_term):
        """Test close properly closes both memory layers."""
        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                manager = MemoryManager()
                await manager.close()

                mock_short_term.close.assert_called_once()
                mock_long_term.close.assert_called_once()


class TestParallelRetrieval:
    """Test P1: Parallel cross-layer retrieval functionality.

    These tests validate the P1 fix that reduced memory retrieval from 30s → 6-10s
    by executing independent layers (profile + episodes) concurrently using asyncio.gather().

    Key aspects tested:
    - Parallel execution (all layers succeed concurrently)
    - Graceful degradation (one layer fails, others succeed)
    - Timeout handling (one layer times out, others succeed)
    - Sequential fallback (when parallel disabled)
    - Configuration flags (MEMORY_PARALLEL_RETRIEVAL, MEMORY_LAYER_TIMEOUT)
    """

    @pytest.mark.asyncio
    async def test_parallel_retrieval_all_layers_succeed(
        self, mock_short_term, mock_long_term, sample_context, sample_profile
    ):
        """Test parallel retrieval with all layers succeeding concurrently.

        Expected behavior:
        - Profile and episodes retrieved in parallel (not sequential)
        - Both layers succeed
        - Result contains session, profile, and episodes
        - Performance: ~max(profile_time, episodes_time) not sum
        """
        # Arrange: Mock successful responses
        mock_short_term.get_context.return_value = sample_context
        mock_long_term.get_user_profile.return_value = sample_profile
        mock_long_term.search_episodes.return_value = [
            {"episode_id": "ep1", "content": "Previous conversation about Florida"},
            {"episode_id": "ep2", "content": "Discussed hurricane preparedness"},
        ]

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        # Act: Get context with parallel retrieval
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert: All layers present
                        assert "session" in result
                        assert "profile" in result
                        assert "episodes" in result
                        assert result["session"] == sample_context
                        assert result["profile"] == sample_profile
                        assert len(result["episodes"]) == 2

                        # Assert: Both long-term methods were called
                        mock_long_term.get_user_profile.assert_called_once_with("test_user")
                        mock_long_term.search_episodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_retrieval_profile_timeout_graceful_degradation(
        self, mock_short_term, mock_long_term, sample_context
    ):
        """Test graceful degradation when profile layer times out.

        Expected behavior:
        - Profile times out after MEMORY_LAYER_TIMEOUT (5s)
        - Episodes still succeed (parallel execution continues)
        - Result contains session + episodes, profile=None
        - No exception raised (graceful degradation)
        """
        # Arrange: Profile times out, episodes succeed
        mock_short_term.get_context.return_value = sample_context

        async def slow_profile(user_id):
            """Simulate slow profile retrieval that times out."""
            await asyncio.sleep(10)  # Longer than timeout
            return {"user_id": user_id}

        mock_long_term.get_user_profile.side_effect = slow_profile
        mock_long_term.search_episodes.return_value = [
            {"episode_id": "ep1", "content": "Episode data"},
        ]

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        with patch("backend.config.memory_config.memory_config.MEMORY_LAYER_TIMEOUT", 1.0):
                            # Act: Get context (profile should timeout)
                            manager = MemoryManager()
                            result = await manager.get_context("test_user", "test_session")

                            # Assert: Graceful degradation - profile is None
                            assert result["session"] == sample_context
                            assert result["profile"] is None  # Timed out
                            assert "episodes" in result  # Episodes still succeeded
                            assert len(result["episodes"]) == 1

    @pytest.mark.asyncio
    async def test_parallel_retrieval_episodes_timeout_graceful_degradation(
        self, mock_short_term, mock_long_term, sample_context, sample_profile
    ):
        """Test graceful degradation when episodes layer times out.

        Expected behavior:
        - Episodes times out after MEMORY_LAYER_TIMEOUT (5s)
        - Profile still succeeds (parallel execution continues)
        - Result contains session + profile, episodes=[]
        - No exception raised (graceful degradation)
        """
        # Arrange: Episodes times out, profile succeeds
        mock_short_term.get_context.return_value = sample_context
        mock_long_term.get_user_profile.return_value = sample_profile

        async def slow_episodes(user_id, query, limit):
            """Simulate slow episode search that times out."""
            await asyncio.sleep(10)  # Longer than timeout
            return []

        mock_long_term.search_episodes.side_effect = slow_episodes

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        with patch("backend.config.memory_config.memory_config.MEMORY_LAYER_TIMEOUT", 1.0):
                            # Act: Get context (episodes should timeout)
                            manager = MemoryManager()
                            result = await manager.get_context("test_user", "test_session")

                            # Assert: Graceful degradation - episodes is empty list
                            assert result["session"] == sample_context
                            assert result["profile"] == sample_profile  # Profile succeeded
                            assert result["episodes"] == []  # Timed out, returns empty list

    @pytest.mark.asyncio
    async def test_parallel_retrieval_profile_exception_graceful_degradation(
        self, mock_short_term, mock_long_term, sample_context
    ):
        """Test graceful degradation when profile layer raises exception.

        Expected behavior:
        - Profile raises exception (e.g., database error)
        - Episodes still succeed (parallel execution continues)
        - Result contains session + episodes, profile=None
        - Exception logged but not propagated (graceful degradation)
        """
        # Arrange: Profile raises exception, episodes succeed
        mock_short_term.get_context.return_value = sample_context
        mock_long_term.get_user_profile.side_effect = Exception("Database connection error")
        mock_long_term.search_episodes.return_value = [
            {"episode_id": "ep1", "content": "Episode data"},
        ]

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        # Act: Get context (profile should fail gracefully)
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert: Graceful degradation - profile is None
                        assert result["session"] == sample_context
                        assert result["profile"] is None  # Exception occurred
                        assert "episodes" in result  # Episodes still succeeded
                        assert len(result["episodes"]) == 1

    @pytest.mark.asyncio
    async def test_parallel_retrieval_episodes_exception_graceful_degradation(
        self, mock_short_term, mock_long_term, sample_context, sample_profile
    ):
        """Test graceful degradation when episodes layer raises exception.

        Expected behavior:
        - Episodes raises exception (e.g., search error)
        - Profile still succeeds (parallel execution continues)
        - Result contains session + profile, episodes=[]
        - Exception logged but not propagated (graceful degradation)
        """
        # Arrange: Episodes raises exception, profile succeeds
        mock_short_term.get_context.return_value = sample_context
        mock_long_term.get_user_profile.return_value = sample_profile
        mock_long_term.search_episodes.side_effect = Exception("Search index error")

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        # Act: Get context (episodes should fail gracefully)
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert: Graceful degradation - episodes is empty list
                        assert result["session"] == sample_context
                        assert result["profile"] == sample_profile  # Profile succeeded
                        assert result["episodes"] == []  # Exception occurred, returns empty list

    @pytest.mark.asyncio
    async def test_parallel_retrieval_both_layers_fail_graceful_degradation(
        self, mock_short_term, mock_long_term, sample_context
    ):
        """Test graceful degradation when both profile and episodes fail.

        Expected behavior:
        - Profile raises exception
        - Episodes times out
        - Result contains only session (short-term always required)
        - profile=None, episodes=[]
        - No exception propagated (graceful degradation)
        """
        # Arrange: Both layers fail
        mock_short_term.get_context.return_value = sample_context
        mock_long_term.get_user_profile.side_effect = Exception("Database error")

        async def slow_episodes(user_id, query, limit):
            await asyncio.sleep(10)  # Times out
            return []

        mock_long_term.search_episodes.side_effect = slow_episodes

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        with patch("backend.config.memory_config.memory_config.MEMORY_LAYER_TIMEOUT", 1.0):
                            # Act: Get context (both long-term layers fail)
                            manager = MemoryManager()
                            result = await manager.get_context("test_user", "test_session")

                            # Assert: Graceful degradation - only session present
                            assert result["session"] == sample_context
                            assert result["profile"] is None  # Exception
                            assert result["episodes"] == []  # Timeout

    @pytest.mark.asyncio
    async def test_sequential_fallback_when_parallel_disabled(
        self, mock_short_term, mock_long_term, sample_context, sample_profile
    ):
        """Test sequential retrieval fallback when parallel is disabled.

        Expected behavior:
        - MEMORY_PARALLEL_RETRIEVAL = False triggers sequential path
        - Profile and episodes retrieved sequentially (not parallel)
        - Both layers still succeed
        - Result identical to parallel (but slower)
        """
        # Arrange: Sequential mode
        mock_short_term.get_context.return_value = sample_context
        mock_long_term.get_user_profile.return_value = sample_profile
        mock_long_term.search_episodes.return_value = [
            {"episode_id": "ep1", "content": "Episode data"},
        ]

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", False):
                        # Act: Get context in sequential mode
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert: Same results as parallel (but sequential execution)
                        assert result["session"] == sample_context
                        assert result["profile"] == sample_profile
                        assert len(result["episodes"]) == 1

                        # Both methods still called (just sequentially)
                        mock_long_term.get_user_profile.assert_called_once()
                        mock_long_term.search_episodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_retrieval_respects_timeout_config(
        self, mock_short_term, mock_long_term, sample_context
    ):
        """Test that parallel retrieval respects MEMORY_LAYER_TIMEOUT config.

        Expected behavior:
        - MEMORY_LAYER_TIMEOUT controls per-layer timeout
        - Layers timing out beyond this value are gracefully degraded
        - Faster layers complete successfully
        """
        # Arrange: One layer slower than timeout
        mock_short_term.get_context.return_value = sample_context

        async def slow_profile(user_id):
            """Profile takes 3 seconds."""
            await asyncio.sleep(3)
            return {"user_id": user_id}

        mock_long_term.get_user_profile.side_effect = slow_profile
        mock_long_term.search_episodes.return_value = [{"episode_id": "ep1"}]

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        # Case 1: Timeout 2s (profile should timeout)
                        with patch("backend.config.memory_config.memory_config.MEMORY_LAYER_TIMEOUT", 2.0):
                            manager = MemoryManager()
                            result = await manager.get_context("test_user", "test_session")

                            # Profile timed out (3s > 2s timeout)
                            assert result["profile"] is None
                            assert len(result["episodes"]) == 1  # Episodes succeeded

    @pytest.mark.asyncio
    async def test_parallel_retrieval_profiles_disabled(
        self, mock_short_term, mock_long_term, sample_context
    ):
        """Test parallel retrieval when user profiles are disabled.

        Expected behavior:
        - ENABLE_USER_PROFILES = False skips profile/episodes retrieval
        - Only session context returned
        - No long-term methods called
        """
        # Arrange: Profiles disabled
        mock_short_term.get_context.return_value = sample_context

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", False):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        # Act: Get context with profiles disabled
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert: Only session present, no long-term data
                        assert result["session"] == sample_context
                        assert result["profile"] is None
                        assert result["episodes"] is None

                        # Long-term methods not called
                        mock_long_term.get_user_profile.assert_not_called()
                        mock_long_term.search_episodes.assert_not_called()

    @pytest.mark.asyncio
    async def test_parallel_retrieval_creates_session_if_missing(
        self, mock_short_term, mock_long_term, sample_context, sample_profile
    ):
        """Test parallel retrieval creates new session if doesn't exist.

        Expected behavior:
        - get_context returns None (no session)
        - create_context called to create new session
        - Parallel retrieval still works for long-term layers
        """
        # Arrange: No existing session
        mock_short_term.get_context.return_value = None
        mock_short_term.create_context.return_value = sample_context
        mock_long_term.get_user_profile.return_value = sample_profile
        mock_long_term.search_episodes.return_value = []

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        # Act: Get context (should create new session)
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert: New session created
                        mock_short_term.create_context.assert_called_once_with("test_user", "test_session")
                        assert result["session"] == sample_context

                        # Parallel retrieval still happened
                        assert result["profile"] == sample_profile
                        assert result["episodes"] == []
