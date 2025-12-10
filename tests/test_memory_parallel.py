"""Unit tests for P1: Parallel cross-layer memory retrieval.

Tests validate the P1 fix that reduced memory retrieval from 30s → 6-10s
by executing independent layers (profile + episodes) concurrently using asyncio.gather().

Test coverage:
- Parallel execution (all layers succeed concurrently)
- Graceful degradation (one layer fails, others succeed)
- Timeout handling (one layer times out, others succeed)
- Sequential fallback (when parallel disabled)
- Configuration flags (MEMORY_PARALLEL_RETRIEVAL, MEMORY_LAYER_TIMEOUT)
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from backend.src.memory.manager import MemoryManager
from backend.src.models.memory import ConversationContext, UserProfile


@pytest.fixture
def mock_short_term():
    """Create a mocked ShortTermMemory."""
    mock = AsyncMock()
    mock.get_context = AsyncMock()
    mock.create_context = AsyncMock()
    mock.save_context = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_long_term():
    """Create a mocked LongTermMemory."""
    mock = AsyncMock()
    mock.get_user_profile = AsyncMock()
    mock.search_episodes = AsyncMock()
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


class TestParallelRetrieval:
    """Test P1: Parallel cross-layer retrieval functionality."""

    @pytest.mark.asyncio
    async def test_parallel_retrieval_all_layers_succeed(
        self, mock_short_term, mock_long_term, sample_context, sample_profile
    ):
        """Test parallel retrieval with all layers succeeding concurrently."""
        # Arrange
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
                        # Act
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert
                        assert "session" in result
                        assert "profile" in result
                        assert "episodes" in result
                        assert result["session"] == sample_context
                        assert result["profile"] == sample_profile
                        assert len(result["episodes"]) == 2
                        mock_long_term.get_user_profile.assert_called_once_with("test_user")
                        mock_long_term.search_episodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_retrieval_profile_timeout_graceful_degradation(
        self, mock_short_term, mock_long_term, sample_context
    ):
        """Test graceful degradation when profile layer times out."""
        # Arrange
        mock_short_term.get_context.return_value = sample_context

        async def slow_profile(user_id):
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
                            # Act
                            manager = MemoryManager()
                            result = await manager.get_context("test_user", "test_session")

                            # Assert: Graceful degradation
                            assert result["session"] == sample_context
                            assert result["profile"] is None  # Timed out
                            assert "episodes" in result
                            assert len(result["episodes"]) == 1

    @pytest.mark.asyncio
    async def test_parallel_retrieval_episodes_exception_graceful_degradation(
        self, mock_short_term, mock_long_term, sample_context, sample_profile
    ):
        """Test graceful degradation when episodes layer raises exception."""
        # Arrange
        mock_short_term.get_context.return_value = sample_context
        mock_long_term.get_user_profile.return_value = sample_profile
        mock_long_term.search_episodes.side_effect = Exception("Search index error")

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        # Act
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert: Graceful degradation
                        assert result["session"] == sample_context
                        assert result["profile"] == sample_profile  # Profile succeeded
                        assert result["episodes"] == []  # Exception, returns empty list

    @pytest.mark.asyncio
    async def test_sequential_fallback_when_parallel_disabled(
        self, mock_short_term, mock_long_term, sample_context, sample_profile
    ):
        """Test sequential retrieval fallback when parallel is disabled."""
        # Arrange
        mock_short_term.get_context.return_value = sample_context
        mock_long_term.get_user_profile.return_value = sample_profile
        mock_long_term.search_episodes.return_value = [
            {"episode_id": "ep1", "content": "Episode data"},
        ]

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", False):
                        # Act
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert: Same results as parallel
                        assert result["session"] == sample_context
                        assert result["profile"] == sample_profile
                        assert len(result["episodes"]) == 1
                        mock_long_term.get_user_profile.assert_called_once()
                        mock_long_term.search_episodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_retrieval_respects_timeout_config(
        self, mock_short_term, mock_long_term, sample_context
    ):
        """Test that parallel retrieval respects MEMORY_LAYER_TIMEOUT config."""
        # Arrange
        mock_short_term.get_context.return_value = sample_context

        async def slow_profile(user_id):
            await asyncio.sleep(3)  # 3 seconds
            return {"user_id": user_id}

        mock_long_term.get_user_profile.side_effect = slow_profile
        mock_long_term.search_episodes.return_value = [{"episode_id": "ep1"}]

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", True):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        # Timeout 2s (profile should timeout at 3s)
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
        """Test parallel retrieval when user profiles are disabled."""
        # Arrange
        mock_short_term.get_context.return_value = sample_context

        with patch("backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term):
            with patch("backend.src.memory.manager.LongTermMemory", return_value=mock_long_term):
                with patch("backend.config.memory_config.memory_config.ENABLE_USER_PROFILES", False):
                    with patch("backend.config.memory_config.memory_config.MEMORY_PARALLEL_RETRIEVAL", True):
                        # Act
                        manager = MemoryManager()
                        result = await manager.get_context("test_user", "test_session")

                        # Assert: Only session present
                        assert result["session"] == sample_context
                        assert result["profile"] is None
                        assert result["episodes"] is None
                        mock_long_term.get_user_profile.assert_not_called()
                        mock_long_term.search_episodes.assert_not_called()
