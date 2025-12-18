"""Unit tests for Level 3a memory system components.

Tests:
- Short-term memory (Redis): Session context, entity tracking, TTL
- Long-term memory (Graphiti): User profiles, temporal facts, persistence
- Memory manager: Unified interface, context retrieval, interaction saving
- Pronoun resolution: Entity tracking, query rewriting
- Context management: Creation, retrieval, expiration

Test Strategy:
- Mock Redis and Graphiti for unit tests
- Test each memory component in isolation
- Verify data models and serialization
- Test error handling and edge cases

Coverage Target: 90%+
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.src.memory.long_term import LongTermMemory
from backend.src.memory.manager import MemoryManager
from backend.src.memory.short_term import ShortTermMemory
from backend.src.models.memory import (
    ConversationContext,
    SessionState,
    TemporalFact,
    UserProfile,
)


class TestShortTermMemory:
    """Test suite for Redis-based short-term memory."""

    @pytest.fixture
    async def mock_redis(self):
        """Mock Redis client."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis
            yield mock_redis

    @pytest.fixture
    async def short_term_memory(self, mock_redis):
        """Create ShortTermMemory instance with mocked Redis."""
        return ShortTermMemory()

    @pytest.mark.asyncio
    async def test_save_context(self, short_term_memory, mock_redis):
        """Test saving conversation context to Redis."""
        context = ConversationContext(
            user_id="user_123",
            session_id="session_abc",
            conversation_history=[{"role": "user", "content": "Weather in London?"}],
            current_entities={"location": "London"},
            token_count=10,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=30),
        )

        await short_term_memory.save_context(context)

        # Verify Redis setex was called with correct key and TTL
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[1]["name"] == "stm:user_123:session_abc"
        assert call_args[1]["time"] == 1800  # 30 minutes

    @pytest.mark.asyncio
    async def test_get_context_exists(self, short_term_memory, mock_redis):
        """Test retrieving existing context from Redis."""
        context_data = {
            "user_id": "user_123",
            "session_id": "session_abc",
            "conversation_history": [{"role": "user", "content": "Hello"}],
            "current_entities": {},
            "token_count": 5,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=30)).isoformat(),
        }

        mock_redis.get.return_value = json.dumps(context_data)

        context = await short_term_memory.get_context("user_123", "session_abc")

        assert context is not None
        assert context.user_id == "user_123"
        assert context.session_id == "session_abc"
        assert len(context.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_get_context_not_exists(self, short_term_memory, mock_redis):
        """Test retrieving non-existent context returns None."""
        mock_redis.get.return_value = None

        context = await short_term_memory.get_context("user_123", "session_abc")

        assert context is None

    @pytest.mark.asyncio
    async def test_create_context(self, short_term_memory, mock_redis):
        """Test creating new conversation context."""
        context = await short_term_memory.create_context("user_123", "session_abc")

        assert context.user_id == "user_123"
        assert context.session_id == "session_abc"
        assert context.conversation_history == []
        assert context.current_entities == {}
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_entity(self, short_term_memory, mock_redis):
        """Test entity tracking for pronoun resolution."""
        await short_term_memory.track_entity(
            "user_123", "session_abc", "location", "London"
        )

        # Verify entity stored in Redis
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[1]["name"] == "stm:user_123:session_abc:entities:location"
        assert call_args[1]["value"] == "London"

    @pytest.mark.asyncio
    async def test_get_entity(self, short_term_memory, mock_redis):
        """Test retrieving tracked entity."""
        mock_redis.get.return_value = "London"

        location = await short_term_memory.get_entity(
            "user_123", "session_abc", "location"
        )

        assert location == "London"

    @pytest.mark.asyncio
    async def test_resolve_pronoun_with_location(self, short_term_memory, mock_redis):
        """Test pronoun resolution with tracked location."""
        mock_redis.get.return_value = "London"

        resolved = await short_term_memory.resolve_pronoun(
            "user_123", "session_abc", "How about there tomorrow?"
        )

        assert resolved == "How about London tomorrow?"

    @pytest.mark.asyncio
    async def test_resolve_pronoun_no_location(self, short_term_memory, mock_redis):
        """Test pronoun resolution without tracked location."""
        mock_redis.get.return_value = None

        resolved = await short_term_memory.resolve_pronoun(
            "user_123", "session_abc", "How about there tomorrow?"
        )

        # Should return original query
        assert resolved == "How about there tomorrow?"

    @pytest.mark.asyncio
    async def test_resolve_multiple_pronouns(self, short_term_memory, mock_redis):
        """Test resolving multiple pronouns in one query."""
        mock_redis.get.return_value = "Paris"

        resolved = await short_term_memory.resolve_pronoun(
            "user_123",
            "session_abc",
            "What about there? Is it raining in that city?",
        )

        assert "Paris" in resolved
        assert "there" not in resolved.lower()

    @pytest.mark.asyncio
    async def test_save_session_state(self, short_term_memory, mock_redis):
        """Test saving session metadata."""
        state = SessionState(
            session_id="session_abc",
            user_id="user_123",
            query_count=5,
            started_at=datetime.now(),
        )

        await short_term_memory.save_session_state(state)

        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, short_term_memory, mock_redis):
        """Test closing Redis connection."""
        await short_term_memory.close()

        mock_redis.close.assert_called_once()


class TestLongTermMemory:
    """Test suite for Graphiti-based long-term memory."""

    @pytest.fixture
    async def mock_graphiti(self):
        """Mock Graphiti client."""
        with patch("graphiti_core.Graphiti") as mock_graphiti_class:
            mock_graphiti = AsyncMock()
            mock_graphiti_class.return_value = mock_graphiti
            yield mock_graphiti

    @pytest.fixture
    async def long_term_memory(self, mock_graphiti):
        """Create LongTermMemory instance with mocked Graphiti."""
        return LongTermMemory()

    @pytest.mark.asyncio
    async def test_create_user_profile(self, long_term_memory, mock_graphiti):
        """Test creating user profile in Neo4j."""
        await long_term_memory.create_user_profile(
            user_id="user_123",
            name="John Doe",
            location="London",
            preferences={"units": "celsius"},
        )

        # Verify user node created
        assert mock_graphiti.add_node.call_count == 2  # User + Location
        assert mock_graphiti.add_edge.call_count == 1  # LOCATED_IN

    @pytest.mark.asyncio
    async def test_get_user_profile_exists(self, long_term_memory, mock_graphiti):
        """Test retrieving existing user profile."""
        # Mock Graphiti search result
        mock_node = MagicMock()
        mock_node.properties = {
            "user_id": "user_123",
            "name": "John Doe",
            "home_location": "London",
            "preferences": {"units": "celsius", "detail": "moderate"},
            "total_queries": 10,
        }

        mock_search_result = MagicMock()
        mock_search_result.nodes = [mock_node]
        mock_graphiti.search.return_value = mock_search_result

        profile = await long_term_memory.get_user_profile("user_123")

        assert profile is not None
        assert profile.user_id == "user_123"
        assert profile.name == "John Doe"
        assert profile.home_location == "London"
        assert profile.preferred_units == "celsius"

    @pytest.mark.asyncio
    async def test_get_user_profile_not_exists(self, long_term_memory, mock_graphiti):
        """Test retrieving non-existent profile returns None."""
        mock_search_result = MagicMock()
        mock_search_result.nodes = []
        mock_graphiti.search.return_value = mock_search_result

        profile = await long_term_memory.get_user_profile("user_123")

        assert profile is None

    @pytest.mark.asyncio
    async def test_add_temporal_fact(self, long_term_memory, mock_graphiti):
        """Test adding temporal fact to knowledge graph."""
        fact = TemporalFact(
            fact_id="fact_123",
            user_id="user_123",
            fact_type="preference",
            content="User prefers detailed forecasts",
            confidence=0.95,
            valid_from=datetime.now(),
        )

        await long_term_memory.add_temporal_fact(fact)

        # Verify fact node and KNOWS edge created
        assert mock_graphiti.add_node.call_count == 1
        assert mock_graphiti.add_edge.call_count == 1

    @pytest.mark.asyncio
    async def test_close(self, long_term_memory, mock_graphiti):
        """Test closing Graphiti connection."""
        await long_term_memory.close()

        mock_graphiti.close.assert_called_once()


class TestMemoryManager:
    """Test suite for unified memory manager."""

    @pytest.fixture
    async def mock_short_term(self):
        """Mock ShortTermMemory."""
        return AsyncMock(spec=ShortTermMemory)

    @pytest.fixture
    async def mock_long_term(self):
        """Mock LongTermMemory."""
        return AsyncMock(spec=LongTermMemory)

    @pytest.fixture
    async def memory_manager(self, mock_short_term, mock_long_term):
        """Create MemoryManager with mocked components."""
        with patch(
            "backend.src.memory.manager.ShortTermMemory", return_value=mock_short_term
        ):
            with patch(
                "backend.src.memory.manager.LongTermMemory",
                return_value=mock_long_term,
            ):
                return MemoryManager()

    @pytest.mark.asyncio
    async def test_get_context_new_session(
        self, memory_manager, mock_short_term, mock_long_term
    ):
        """Test getting context for new session."""
        mock_short_term.get_context.return_value = None
        mock_short_term.create_context.return_value = ConversationContext(
            user_id="user_123",
            session_id="session_abc",
            conversation_history=[],
            current_entities={},
            token_count=0,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=30),
        )

        context = await memory_manager.get_context("user_123", "session_abc")

        assert context["session"] is not None
        mock_short_term.create_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_interaction(
        self, memory_manager, mock_short_term, mock_long_term
    ):
        """Test saving interaction to both memory layers."""
        # Mock existing context
        existing_context = ConversationContext(
            user_id="user_123",
            session_id="session_abc",
            conversation_history=[],
            current_entities={},
            token_count=0,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=30),
        )
        mock_short_term.get_context.return_value = existing_context

        await memory_manager.save_interaction(
            user_id="user_123",
            session_id="session_abc",
            query="Weather in London?",
            response="London is 18°C, sunny",
        )

        # Verify conversation history updated
        mock_short_term.save_context.assert_called_once()
        saved_context = mock_short_term.save_context.call_args[0][0]
        assert len(saved_context.conversation_history) == 2
        assert saved_context.conversation_history[0]["role"] == "user"
        assert saved_context.conversation_history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_resolve_pronoun(self, memory_manager, mock_short_term):
        """Test pronoun resolution via manager."""
        mock_short_term.resolve_pronoun.return_value = "Weather in London tomorrow?"

        resolved = await memory_manager.resolve_pronoun(
            "user_123", "session_abc", "Weather there tomorrow?"
        )

        assert resolved == "Weather in London tomorrow?"
        mock_short_term.resolve_pronoun.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, memory_manager, mock_short_term, mock_long_term):
        """Test closing all memory connections."""
        await memory_manager.close()

        mock_short_term.close.assert_called_once()
        mock_long_term.close.assert_called_once()


class TestMemoryModels:
    """Test suite for memory data models."""

    def test_conversation_context_model(self):
        """Test ConversationContext model validation."""
        context = ConversationContext(
            user_id="user_123",
            session_id="session_abc",
            conversation_history=[{"role": "user", "content": "Hello"}],
            current_entities={"location": "London"},
            token_count=10,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=30),
        )

        assert context.user_id == "user_123"
        assert len(context.conversation_history) == 1
        assert context.current_entities["location"] == "London"

    def test_user_profile_model(self):
        """Test UserProfile model validation."""
        profile = UserProfile(
            user_id="user_123",
            name="John Doe",
            home_location="London",
            preferred_units="celsius",
            preferred_detail_level="moderate",
            total_queries=10,
        )

        assert profile.user_id == "user_123"
        assert profile.name == "John Doe"
        assert profile.preferred_units == "celsius"

    def test_temporal_fact_model(self):
        """Test TemporalFact model validation."""
        fact = TemporalFact(
            fact_id="fact_123",
            user_id="user_123",
            fact_type="preference",
            content="User prefers detailed forecasts",
            confidence=0.95,
            valid_from=datetime.now(),
        )

        assert fact.fact_id == "fact_123"
        assert fact.confidence == 0.95
        assert fact.valid_to is None

    def test_session_state_model(self):
        """Test SessionState model validation."""
        state = SessionState(
            session_id="session_abc",
            user_id="user_123",
            query_count=5,
            started_at=datetime.now(),
        )

        assert state.session_id == "session_abc"
        assert state.query_count == 5
