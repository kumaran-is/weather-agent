"""End-to-end integration tests for Level 3a memory system.

Tests:
- Complete memory flow: Session creation → Interaction → Context recall
- Pronoun resolution across conversation turns
- User profile persistence across sessions
- Entity tracking and resolution
- Memory manager orchestration

Test Strategy:
- Use Docker containers for Redis and Neo4j (via pytest fixtures)
- Test realistic user workflows
- Verify cross-component integration
- Test memory persistence and TTL

Coverage Target: 80%+

Prerequisites:
- Docker running with Redis container
- Docker running with Neo4j container
- Or use in-memory alternatives for CI/CD
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from backend.src.memory.manager import MemoryManager


@pytest.mark.asyncio
@pytest.mark.integration
class TestMemoryE2E:
    """End-to-end integration tests for memory system."""

    @pytest.fixture
    async def memory_manager(self):
        """Create real MemoryManager instance.

        Note: Requires Redis and Neo4j running via Docker.
        Set environment variables for connection:
        - REDIS_URL=redis://localhost:6379/0
        - GRAPHITI_URL=bolt://localhost:7687
        - GRAPHITI_USER=neo4j
        - GRAPHITI_PASSWORD=password
        """
        manager = MemoryManager()
        yield manager
        # Cleanup after test
        await manager.close()

    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self, memory_manager):
        """Test complete conversation with memory recall.

        Flow:
        1. User asks: "Weather in London?"
        2. Agent responds with weather
        3. User asks: "How about there tomorrow?"
        4. Agent resolves "there" → "London" and responds
        """
        user_id = "test_user_001"
        session_id = "test_session_001"

        # Step 1: Get initial context (should create new)
        context1 = await memory_manager.get_context(user_id, session_id)
        assert context1["session"] is not None
        assert len(context1["session"].conversation_history) == 0

        # Step 2: Save first interaction
        await memory_manager.save_interaction(
            user_id=user_id,
            session_id=session_id,
            query="What's the weather in London?",
            response="London is currently 18°C and sunny.",
        )

        # Track location entity
        await memory_manager.track_entity(
            user_id, session_id, "location", "London"
        )

        # Step 3: Get context again (should have conversation history)
        context2 = await memory_manager.get_context(user_id, session_id)
        assert len(context2["session"].conversation_history) == 2
        assert context2["session"].current_entities["location"] == "London"

        # Step 4: Resolve pronoun for follow-up query
        resolved_query = await memory_manager.resolve_pronoun(
            user_id, session_id, "How about there tomorrow?"
        )

        assert "London" in resolved_query
        assert "there" not in resolved_query.lower()

        # Step 5: Save follow-up interaction
        await memory_manager.save_interaction(
            user_id=user_id,
            session_id=session_id,
            query=resolved_query,
            response="London tomorrow: 16°C, partly cloudy.",
        )

        # Step 6: Verify complete conversation history
        context3 = await memory_manager.get_context(user_id, session_id)
        assert len(context3["session"].conversation_history) == 4
        assert context3["session"].conversation_history[0]["content"] == (
            "What's the weather in London?"
        )
        assert "London" in context3["session"].conversation_history[2]["content"]

    @pytest.mark.asyncio
    async def test_user_profile_persistence(self, memory_manager):
        """Test user profile creation and retrieval across sessions."""
        user_id = "test_user_002"

        # Create user profile
        await memory_manager.create_user_profile(
            user_id=user_id,
            name="Jane Smith",
            location="Paris",
            preferences={"units": "celsius", "detail": "high"},
        )

        # Retrieve profile in new session
        session_id_1 = "session_001"
        context1 = await memory_manager.get_context(user_id, session_id_1)

        assert context1["profile"] is not None
        assert context1["profile"].name == "Jane Smith"
        assert context1["profile"].home_location == "Paris"
        assert context1["profile"].preferred_units == "celsius"

        # Verify profile persists in different session
        session_id_2 = "session_002"
        context2 = await memory_manager.get_context(user_id, session_id_2)

        assert context2["profile"] is not None
        assert context2["profile"].name == "Jane Smith"

    @pytest.mark.asyncio
    async def test_multi_entity_tracking(self, memory_manager):
        """Test tracking multiple entity types."""
        user_id = "test_user_003"
        session_id = "test_session_003"

        # Track multiple entities
        await memory_manager.track_entity(user_id, session_id, "location", "Tokyo")
        await memory_manager.track_entity(
            user_id, session_id, "date", "2025-12-25"
        )
        await memory_manager.track_entity(
            user_id, session_id, "hurricane", "Milton"
        )

        # Retrieve context with all entities
        context = await memory_manager.get_context(user_id, session_id)

        assert context["session"].current_entities["location"] == "Tokyo"
        assert context["session"].current_entities["date"] == "2025-12-25"
        assert context["session"].current_entities["hurricane"] == "Milton"

    @pytest.mark.asyncio
    async def test_conversation_history_truncation(self, memory_manager):
        """Test conversation history truncation to prevent context explosion."""
        user_id = "test_user_004"
        session_id = "test_session_004"

        # Add 15 Q&A pairs (30 messages total)
        for i in range(15):
            await memory_manager.save_interaction(
                user_id=user_id,
                session_id=session_id,
                query=f"Query {i}",
                response=f"Response {i}",
            )

        # Verify history is truncated to max turns (default 10 turns = 20 messages)
        context = await memory_manager.get_context(user_id, session_id)
        # Should keep only last 10 turns (20 messages)
        assert len(context["session"].conversation_history) <= 20

    @pytest.mark.asyncio
    async def test_session_isolation(self, memory_manager):
        """Test that sessions are isolated from each other."""
        user_id = "test_user_005"
        session_1 = "session_001"
        session_2 = "session_002"

        # Add interaction to session 1
        await memory_manager.save_interaction(
            user_id=user_id,
            session_id=session_1,
            query="Weather in London?",
            response="18°C, sunny",
        )
        await memory_manager.track_entity(user_id, session_1, "location", "London")

        # Add different interaction to session 2
        await memory_manager.save_interaction(
            user_id=user_id,
            session_id=session_2,
            query="Weather in Paris?",
            response="15°C, rainy",
        )
        await memory_manager.track_entity(user_id, session_2, "location", "Paris")

        # Verify sessions are isolated
        context1 = await memory_manager.get_context(user_id, session_1)
        context2 = await memory_manager.get_context(user_id, session_2)

        assert context1["session"].current_entities["location"] == "London"
        assert context2["session"].current_entities["location"] == "Paris"
        assert context1["session"].conversation_history != (
            context2["session"].conversation_history
        )

    @pytest.mark.asyncio
    async def test_conversation_summary(self, memory_manager):
        """Test conversation summary generation."""
        user_id = "test_user_006"
        session_id = "test_session_006"

        # Add several interactions
        await memory_manager.save_interaction(
            user_id=user_id,
            session_id=session_id,
            query="Weather in Berlin?",
            response="Berlin is 12°C, overcast",
        )
        await memory_manager.save_interaction(
            user_id=user_id,
            session_id=session_id,
            query="Will it rain tomorrow?",
            response="Yes, 70% chance of rain tomorrow",
        )

        # Get conversation summary
        summary = await memory_manager.get_conversation_summary(user_id, session_id)

        assert summary is not None
        assert "User:" in summary or "Agent:" in summary

    @pytest.mark.asyncio
    async def test_pronoun_resolution_edge_cases(self, memory_manager):
        """Test pronoun resolution with edge cases."""
        user_id = "test_user_007"
        session_id = "test_session_007"

        # Test 1: No tracked location
        resolved = await memory_manager.resolve_pronoun(
            user_id, session_id, "Weather there?"
        )
        assert resolved == "Weather there?"  # No change

        # Test 2: Track location and resolve
        await memory_manager.track_entity(
            user_id, session_id, "location", "New York"
        )
        resolved = await memory_manager.resolve_pronoun(
            user_id, session_id, "How about there tomorrow?"
        )
        assert "New York" in resolved

        # Test 3: Multiple pronouns
        resolved = await memory_manager.resolve_pronoun(
            user_id,
            session_id,
            "What about there? Is it sunny in that place?",
        )
        assert resolved.count("New York") >= 1
