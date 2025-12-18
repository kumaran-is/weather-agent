"""Comprehensive tests for ShortTermMemory (Redis-based).

Tests cover:
- Context manager usage
- Error handling for all Redis operations
- Context lifecycle (create, save, retrieve)
- Entity tracking and retrieval
- Pronoun resolution
- Session state management
- Connection cleanup
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import redis.asyncio as redis

from backend.src.memory.exceptions import RedisMemoryError
from backend.src.memory.short_term import ShortTermMemory
from backend.src.models.memory import ConversationContext, SessionState


@pytest.fixture
def mock_redis():
    """Create a mocked Redis client."""
    mock = AsyncMock(spec=redis.Redis)
    mock.get = AsyncMock()
    mock.setex = AsyncMock()
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
            {"role": "assistant", "content": "London is 18°C"},
        ],
        current_entities={"location": "London"},
        token_count=10,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(minutes=30),
    )


@pytest.fixture
def sample_session_state():
    """Create a sample session state."""
    return SessionState(
        session_id="test_session",
        user_id="test_user",
        start_time=datetime.now(),
        last_activity=datetime.now(),
        query_count=5,
    )


class TestContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_entry_and_exit(self, mock_redis):
        """Test context manager properly enters and exits."""
        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            async with ShortTermMemory() as memory:
                assert memory is not None
                assert isinstance(memory, ShortTermMemory)

            # Verify close() was called on exit
            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, mock_redis):
        """Test context manager properly closes even when exception occurs."""
        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            with pytest.raises(ValueError):
                async with ShortTermMemory() as memory:
                    raise ValueError("Test exception")

            # Close should still be called
            mock_redis.close.assert_called_once()


class TestSaveContext:
    """Test save_context method."""

    @pytest.mark.asyncio
    async def test_save_context_success(self, mock_redis, sample_context):
        """Test successful context save."""
        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            await memory.save_context(sample_context)

            # Verify setex was called with correct parameters
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            assert call_args[1]["name"] == "stm:test_user:test_session"
            assert call_args[1]["time"] == memory.ttl

    @pytest.mark.asyncio
    async def test_save_context_redis_error(self, mock_redis, sample_context):
        """Test save_context raises RedisMemoryError on Redis failure."""
        mock_redis.setex.side_effect = redis.RedisError("Connection failed")

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            with pytest.raises(RedisMemoryError, match="Failed to save context"):
                await memory.save_context(sample_context)


class TestGetContext:
    """Test get_context method."""

    @pytest.mark.asyncio
    async def test_get_context_found(self, mock_redis, sample_context):
        """Test successful context retrieval."""
        mock_redis.get.return_value = sample_context.model_dump_json()

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            result = await memory.get_context("test_user", "test_session")

            assert result is not None
            assert result.user_id == "test_user"
            assert result.session_id == "test_session"
            assert len(result.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_get_context_not_found(self, mock_redis):
        """Test get_context returns None when context doesn't exist."""
        mock_redis.get.return_value = None

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            result = await memory.get_context("test_user", "test_session")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_context_redis_error(self, mock_redis):
        """Test get_context raises RedisMemoryError on Redis failure."""
        mock_redis.get.side_effect = redis.RedisError("Connection timeout")

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            with pytest.raises(RedisMemoryError, match="Failed to get context"):
                await memory.get_context("test_user", "test_session")


class TestCreateContext:
    """Test create_context method."""

    @pytest.mark.asyncio
    async def test_create_context_success(self, mock_redis):
        """Test successful context creation."""
        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            context = await memory.create_context("test_user", "test_session")

            assert context.user_id == "test_user"
            assert context.session_id == "test_session"
            assert context.conversation_history == []
            assert context.current_entities == {}
            assert context.token_count == 0

            # Verify save_context was called
            mock_redis.setex.assert_called_once()


class TestTrackEntity:
    """Test track_entity method."""

    @pytest.mark.asyncio
    async def test_track_entity_success(self, mock_redis):
        """Test successful entity tracking."""
        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            await memory.track_entity("test_user", "test_session", "location", "London")

            # Verify setex was called with correct key
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            assert call_args[1]["name"] == "stm:test_user:test_session:entities:location"
            assert call_args[1]["value"] == "London"

    @pytest.mark.asyncio
    async def test_track_entity_redis_error(self, mock_redis):
        """Test track_entity raises RedisMemoryError on Redis failure."""
        mock_redis.setex.side_effect = redis.RedisError("Connection failed")

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            with pytest.raises(RedisMemoryError, match="Failed to track entity"):
                await memory.track_entity("test_user", "test_session", "location", "London")


class TestGetEntity:
    """Test get_entity method."""

    @pytest.mark.asyncio
    async def test_get_entity_found(self, mock_redis):
        """Test successful entity retrieval."""
        mock_redis.get.return_value = "London"

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            result = await memory.get_entity("test_user", "test_session", "location")

            assert result == "London"

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, mock_redis):
        """Test get_entity returns None when entity doesn't exist."""
        mock_redis.get.return_value = None

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            result = await memory.get_entity("test_user", "test_session", "location")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_entity_redis_error(self, mock_redis):
        """Test get_entity raises RedisMemoryError on Redis failure."""
        mock_redis.get.side_effect = redis.RedisError("Connection timeout")

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            with pytest.raises(RedisMemoryError, match="Failed to get entity"):
                await memory.get_entity("test_user", "test_session", "location")


class TestResolvePronoun:
    """Test resolve_pronoun method."""

    @pytest.mark.asyncio
    async def test_resolve_pronoun_with_location(self, mock_redis):
        """Test pronoun resolution when location is tracked."""
        mock_redis.get.return_value = "London"

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            result = await memory.resolve_pronoun(
                "test_user", "test_session", "How about there tomorrow?"
            )

            assert result == "How about London tomorrow?"

    @pytest.mark.asyncio
    async def test_resolve_pronoun_no_location(self, mock_redis):
        """Test pronoun resolution when no location is tracked."""
        mock_redis.get.return_value = None

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            result = await memory.resolve_pronoun(
                "test_user", "test_session", "How about there tomorrow?"
            )

            # Should return query unchanged
            assert result == "How about there tomorrow?"

    @pytest.mark.asyncio
    async def test_resolve_pronoun_multiple_pronouns(self, mock_redis):
        """Test pronoun resolution with multiple pronouns."""
        mock_redis.get.return_value = "Paris"

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            result = await memory.resolve_pronoun(
                "test_user",
                "test_session",
                "How about there and that place tomorrow?",
            )

            assert "Paris" in result
            assert "there" not in result or "that place" not in result

    @pytest.mark.asyncio
    async def test_resolve_pronoun_redis_error(self, mock_redis):
        """Test resolve_pronoun raises RedisMemoryError on Redis failure."""
        mock_redis.get.side_effect = redis.RedisError("Connection failed")

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            with pytest.raises(RedisMemoryError, match="Failed to resolve pronoun"):
                await memory.resolve_pronoun("test_user", "test_session", "Weather there?")


class TestSessionState:
    """Test session state management."""

    @pytest.mark.asyncio
    async def test_save_session_state_success(self, mock_redis, sample_session_state):
        """Test successful session state save."""
        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            await memory.save_session_state(sample_session_state)

            # Verify setex was called
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            assert call_args[1]["name"] == "stm:session:test_session"

    @pytest.mark.asyncio
    async def test_save_session_state_redis_error(self, mock_redis, sample_session_state):
        """Test save_session_state raises RedisMemoryError on Redis failure."""
        mock_redis.setex.side_effect = redis.RedisError("Connection failed")

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            with pytest.raises(RedisMemoryError, match="Failed to save session state"):
                await memory.save_session_state(sample_session_state)

    @pytest.mark.asyncio
    async def test_get_session_state_found(self, mock_redis, sample_session_state):
        """Test successful session state retrieval."""
        mock_redis.get.return_value = sample_session_state.model_dump_json()

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            result = await memory.get_session_state("test_session")

            assert result is not None
            assert result.session_id == "test_session"
            assert result.query_count == 5

    @pytest.mark.asyncio
    async def test_get_session_state_not_found(self, mock_redis):
        """Test get_session_state returns None when state doesn't exist."""
        mock_redis.get.return_value = None

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            result = await memory.get_session_state("test_session")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_session_state_redis_error(self, mock_redis):
        """Test get_session_state raises RedisMemoryError on Redis failure."""
        mock_redis.get.side_effect = redis.RedisError("Connection timeout")

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            with pytest.raises(RedisMemoryError, match="Failed to get session state"):
                await memory.get_session_state("test_session")


class TestClose:
    """Test close method."""

    @pytest.mark.asyncio
    async def test_close_success(self, mock_redis):
        """Test successful Redis connection close."""
        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()
            await memory.close()

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_redis_error(self, mock_redis):
        """Test close raises RedisMemoryError on Redis failure."""
        mock_redis.close.side_effect = redis.RedisError("Close failed")

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            with pytest.raises(RedisMemoryError, match="Failed to close Redis connection"):
                await memory.close()


class TestFullLifecycle:
    """Test complete lifecycle workflows."""

    @pytest.mark.asyncio
    async def test_create_save_retrieve_lifecycle(self, mock_redis):
        """Test full lifecycle: create -> save -> retrieve."""
        context_data = None

        async def mock_setex(name, time, value):
            nonlocal context_data
            context_data = value

        async def mock_get(key):
            return context_data

        mock_redis.setex.side_effect = mock_setex
        mock_redis.get.side_effect = mock_get

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            # Create
            context = await memory.create_context("user_123", "session_abc")
            assert context is not None

            # Retrieve
            retrieved = await memory.get_context("user_123", "session_abc")
            assert retrieved is not None
            assert retrieved.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_track_and_resolve_workflow(self, mock_redis):
        """Test track entity -> resolve pronoun workflow."""
        entity_data = {}

        async def mock_setex_entity(name, time, value):
            entity_data[name] = value

        async def mock_get_entity(key):
            return entity_data.get(key)

        mock_redis.setex.side_effect = mock_setex_entity
        mock_redis.get.side_effect = mock_get_entity

        with patch("backend.src.memory.short_term.redis.from_url", return_value=mock_redis):
            memory = ShortTermMemory()

            # Track entity
            await memory.track_entity("user_123", "session_abc", "location", "Tokyo")

            # Resolve pronoun
            resolved = await memory.resolve_pronoun(
                "user_123", "session_abc", "Weather there?"
            )

            assert "Tokyo" in resolved
            assert "there" not in resolved
