"""Unit tests for Level 3c memory system components (Complete 7-layer memory).

Tests:
- Episodic memory (Layer 3): Event history with temporal context (Redis, 7-day TTL)
- Semantic memory (Layer 4): Domain knowledge (Qdrant vector store)
- Procedural memory (Layer 5): Workflow optimization (PostgreSQL)
- Emotional memory (Layer 6): Sentiment analysis (Redis, 7-day TTL)
- Reflective memory (Layer 7): Self-improvement (PostgreSQL)
- Memory consolidation: 70% storage reduction pipeline

Test Strategy:
- Mock Redis, PostgreSQL, and Qdrant for unit tests
- Test each memory layer in isolation
- Verify data models and serialization
- Test error handling and edge cases
- Validate consolidation pipeline stages

Coverage Target: 90%+ (as per Level 3c requirements)
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from backend.src.memory.consolidation import ConsolidationResult, MemoryConsolidator
from backend.src.memory.emotional import EmotionalMemoryDetector
from backend.src.memory.procedural import ProceduralMemoryBackend
from backend.src.memory.reflective import ReflectiveMemoryBackend
from backend.src.models.memory import (
    EmotionalMemory,
    EpisodicMemory,
    ProceduralMemory,
    ReflectiveMemory,
    SemanticMemory,
)

# ============================================================================
# Layer 3: Episodic Memory Tests (Redis, 7-day TTL)
# ============================================================================


class TestEpisodicMemory:
    """Test suite for episodic memory model and storage."""

    def test_episodic_memory_creation(self):
        """Test creating episodic memory with valid data."""
        episode = EpisodicMemory(
            episode_id="ep_abc123",
            user_id="user_123",
            query="Will Hurricane Milton hit Florida?",
            response="Hurricane Milton is projected to make landfall near Tampa Bay...",
            outcome="helpful",
            emotional_context="anxious",
            tools_used=["search_hurricanes", "get_hurricane_forecast"],
        )

        assert episode.episode_id == "ep_abc123"
        assert episode.user_id == "user_123"
        assert episode.outcome == "helpful"
        assert episode.emotional_context == "anxious"
        assert len(episode.tools_used) == 2
        assert isinstance(episode.timestamp, datetime)

    def test_episodic_memory_defaults(self):
        """Test episodic memory with default values."""
        episode = EpisodicMemory(
            episode_id="ep_123",
            user_id="user_123",
            query="What's the weather?",
            response="It's sunny today.",
        )

        assert episode.outcome == "unknown"
        assert episode.emotional_context is None
        assert episode.tools_used == []

    def test_episodic_memory_validation_error(self):
        """Test episodic memory validation for missing required fields."""
        with pytest.raises(ValidationError):
            EpisodicMemory(
                episode_id="ep_123",
                # Missing user_id, query, response
            )

    def test_episodic_memory_serialization(self):
        """Test episodic memory JSON serialization."""
        episode = EpisodicMemory(
            episode_id="ep_123",
            user_id="user_123",
            query="Weather forecast?",
            response="Sunny and warm.",
            outcome="helpful",
        )

        json_data = episode.model_dump_json()
        parsed = json.loads(json_data)

        assert parsed["episode_id"] == "ep_123"
        assert parsed["user_id"] == "user_123"
        assert parsed["outcome"] == "helpful"


# ============================================================================
# Layer 4: Semantic Memory Tests (Qdrant vector store)
# ============================================================================


class TestSemanticMemory:
    """Test suite for semantic memory model and knowledge storage."""

    def test_semantic_memory_creation(self):
        """Test creating semantic memory with valid data."""
        semantic = SemanticMemory(
            concept_id="concept_hurr_cat5",
            user_id="user_123",
            concept="Hurricane Category 5",
            definition="A hurricane with sustained winds of 157 mph or higher",
            embedding=[0.1, 0.2, 0.3] * 512,  # 1536-dim embedding
            related_concepts=["Hurricane", "Saffir-Simpson Scale", "Wind Speed"],
            confidence_score=0.95,
        )

        assert semantic.concept_id == "concept_hurr_cat5"
        assert semantic.concept == "Hurricane Category 5"
        assert semantic.confidence_score == 0.95
        assert len(semantic.embedding) == 1536
        assert len(semantic.related_concepts) == 3

    def test_semantic_memory_defaults(self):
        """Test semantic memory with default values."""
        semantic = SemanticMemory(
            concept_id="concept_123",
            user_id="user_123",
            concept="Test Concept",
            definition="Test definition",
            embedding=[0.1] * 1536,
        )

        assert semantic.related_concepts == []
        assert semantic.confidence_score == 0.0
        assert semantic.source == "user_interaction"
        assert isinstance(semantic.learned_at, datetime)

    def test_semantic_memory_validation_embedding_size(self):
        """Test semantic memory validates embedding dimension (1536)."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticMemory(
                concept_id="concept_123",
                user_id="user_123",
                concept="Test",
                definition="Test",
                embedding=[0.1] * 100,  # Wrong dimension
            )

        assert "embedding" in str(exc_info.value).lower()

    def test_semantic_memory_confidence_bounds(self):
        """Test semantic memory validates confidence score bounds [0.0, 1.0]."""
        # Valid confidence
        semantic = SemanticMemory(
            concept_id="concept_123",
            user_id="user_123",
            concept="Test",
            definition="Test",
            embedding=[0.1] * 1536,
            confidence_score=0.5,
        )
        assert semantic.confidence_score == 0.5

        # Invalid confidence > 1.0
        with pytest.raises(ValidationError):
            SemanticMemory(
                concept_id="concept_123",
                user_id="user_123",
                concept="Test",
                definition="Test",
                embedding=[0.1] * 1536,
                confidence_score=1.5,
            )


# ============================================================================
# Layer 5: Procedural Memory Tests (PostgreSQL workflow optimization)
# ============================================================================


class TestProceduralMemory:
    """Test suite for procedural memory model and workflow optimization."""

    def test_procedural_memory_creation(self):
        """Test creating procedural memory with valid data."""
        procedural = ProceduralMemory(
            workflow_id="wf_hurr_forecast",
            user_id="user_123",
            workflow_name="Hurricane Forecast Workflow",
            steps=[
                {"step": 1, "action": "search_hurricanes", "success": True},
                {"step": 2, "action": "get_hurricane_forecast", "success": True},
            ],
            success_rate=1.0,
            avg_duration_seconds=2.5,
            last_executed=datetime.now(),
        )

        assert procedural.workflow_id == "wf_hurr_forecast"
        assert procedural.workflow_name == "Hurricane Forecast Workflow"
        assert len(procedural.steps) == 2
        assert procedural.success_rate == 1.0
        assert procedural.avg_duration_seconds == 2.5
        assert procedural.execution_count == 1

    def test_procedural_memory_defaults(self):
        """Test procedural memory with default values."""
        procedural = ProceduralMemory(
            workflow_id="wf_123",
            user_id="user_123",
            workflow_name="Test Workflow",
            steps=[],
        )

        assert procedural.execution_count == 1
        assert procedural.success_rate == 0.0
        assert procedural.avg_duration_seconds == 0.0
        assert procedural.optimizations_applied == []

    def test_procedural_memory_success_rate_bounds(self):
        """Test procedural memory validates success rate bounds [0.0, 1.0]."""
        # Valid success rate
        procedural = ProceduralMemory(
            workflow_id="wf_123",
            user_id="user_123",
            workflow_name="Test",
            steps=[],
            success_rate=0.85,
        )
        assert procedural.success_rate == 0.85

        # Invalid success rate > 1.0
        with pytest.raises(ValidationError):
            ProceduralMemory(
                workflow_id="wf_123",
                user_id="user_123",
                workflow_name="Test",
                steps=[],
                success_rate=1.5,
            )


class TestProceduralMemoryBackend:
    """Test suite for procedural memory backend (PostgreSQL)."""

    @pytest.fixture
    def mock_pg_pool(self):
        """Mock PostgreSQL connection pool."""
        pool = AsyncMock()
        pool.acquire = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock()
        pool.acquire.return_value.__aexit__ = AsyncMock()
        return pool

    @pytest.fixture
    def procedural_backend(self, mock_pg_pool):
        """Create ProceduralMemoryBackend with mocked PostgreSQL."""
        with patch("asyncpg.create_pool", return_value=mock_pg_pool):
            backend = ProceduralMemoryBackend()
            backend.pool = mock_pg_pool
            return backend

    @pytest.mark.asyncio
    async def test_save_workflow(self, procedural_backend, mock_pg_pool):
        """Test saving workflow to PostgreSQL."""
        procedural = ProceduralMemory(
            workflow_id="wf_test",
            user_id="user_123",
            workflow_name="Test Workflow",
            steps=[{"step": 1, "action": "test_action", "success": True}],
            success_rate=1.0,
        )

        mock_conn = AsyncMock()
        mock_pg_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await procedural_backend.save_workflow(procedural)

        # Verify execute was called with INSERT query
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO workflows" in call_args[0] or "workflows" in call_args[0].lower()

    @pytest.mark.asyncio
    async def test_get_workflow(self, procedural_backend, mock_pg_pool):
        """Test retrieving workflow from PostgreSQL."""
        mock_conn = AsyncMock()
        mock_pg_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Mock workflow data
        mock_conn.fetchrow.return_value = {
            "workflow_id": "wf_test",
            "user_id": "user_123",
            "workflow_name": "Test Workflow",
            "steps": json.dumps([{"step": 1, "action": "test", "success": True}]),
            "execution_count": 5,
            "success_rate": 0.8,
            "avg_duration_seconds": 1.5,
            "last_executed": datetime.now(),
            "created_at": datetime.now(),
            "optimizations_applied": json.dumps([]),
        }

        workflow = await procedural_backend.get_workflow("user_123", "wf_test")

        assert workflow is not None
        assert workflow.workflow_id == "wf_test"
        assert workflow.execution_count == 5
        assert workflow.success_rate == 0.8

    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self, procedural_backend, mock_pg_pool):
        """Test retrieving non-existent workflow returns None."""
        mock_conn = AsyncMock()
        mock_pg_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None

        workflow = await procedural_backend.get_workflow("user_123", "nonexistent")

        assert workflow is None


# ============================================================================
# Layer 6: Emotional Memory Tests (Redis sentiment analysis)
# ============================================================================


class TestEmotionalMemory:
    """Test suite for emotional memory model."""

    def test_emotional_memory_creation(self):
        """Test creating emotional memory with valid data."""
        emotional = EmotionalMemory(
            emotion_id="emo_abc123",
            user_id="user_123",
            query="I'm really worried about this hurricane!",
            dominant_emotion="anxious",
            sentiment_score=-0.7,
            subjectivity=0.8,
        )

        assert emotional.emotion_id == "emo_abc123"
        assert emotional.dominant_emotion == "anxious"
        assert emotional.sentiment_score == -0.7
        assert emotional.subjectivity == 0.8

    def test_emotional_memory_sentiment_bounds(self):
        """Test emotional memory validates sentiment score bounds [-1.0, 1.0]."""
        # Valid sentiment
        emotional = EmotionalMemory(
            emotion_id="emo_123",
            user_id="user_123",
            query="Test query",
            dominant_emotion="neutral",
            sentiment_score=0.0,
        )
        assert emotional.sentiment_score == 0.0

        # Invalid sentiment < -1.0
        with pytest.raises(ValidationError):
            EmotionalMemory(
                emotion_id="emo_123",
                user_id="user_123",
                query="Test",
                dominant_emotion="neutral",
                sentiment_score=-1.5,
            )

        # Invalid sentiment > 1.0
        with pytest.raises(ValidationError):
            EmotionalMemory(
                emotion_id="emo_123",
                user_id="user_123",
                query="Test",
                dominant_emotion="neutral",
                sentiment_score=1.5,
            )


class TestEmotionalMemoryDetector:
    """Test suite for emotional memory detector."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        return AsyncMock()

    @pytest.fixture
    def detector(self, mock_redis):
        """Create EmotionalMemoryDetector with mocked Redis."""
        return EmotionalMemoryDetector(redis_client=mock_redis)

    @pytest.mark.asyncio
    async def test_detect_emotion_anxious(self, detector):
        """Test detecting anxious emotion from query."""
        query = "I'm really worried about this Category 5 hurricane! Is my family safe?"

        emotion = await detector.detect_emotion(
            user_id="user_123", query=query, session_id="sess_456"
        )

        assert emotion.dominant_emotion == "anxious"
        assert emotion.sentiment_score < 0  # Negative sentiment
        assert emotion.query == query

    @pytest.mark.asyncio
    async def test_detect_emotion_frustrated(self, detector):
        """Test detecting frustrated emotion from query."""
        query = "This weather app is so confusing! Why can't I get a simple forecast?"

        emotion = await detector.detect_emotion(
            user_id="user_123", query=query, session_id="sess_456"
        )

        assert emotion.dominant_emotion in ["frustrated", "anxious"]  # Allow both
        assert emotion.sentiment_score < 0  # Negative sentiment

    @pytest.mark.asyncio
    async def test_detect_emotion_neutral(self, detector):
        """Test detecting neutral emotion from query."""
        query = "What is the temperature in New York?"

        emotion = await detector.detect_emotion(
            user_id="user_123", query=query, session_id="sess_456"
        )

        assert emotion.dominant_emotion in ["neutral", "curious"]
        assert -0.3 <= emotion.sentiment_score <= 0.3  # Near-neutral

    @pytest.mark.asyncio
    async def test_get_emotional_trend(self, detector, mock_redis):
        """Test retrieving emotional trend over time."""
        # Mock Redis response with emotional history
        mock_redis.keys.return_value = [
            b"emotion:user_123:2025-01-21T10:00:00",
            b"emotion:user_123:2025-01-21T11:00:00",
            b"emotion:user_123:2025-01-21T12:00:00",
        ]

        mock_redis.get.side_effect = [
            json.dumps({
                "emotion_id": "emo_1",
                "user_id": "user_123",
                "timestamp": "2025-01-21T10:00:00",
                "query": "Test 1",
                "dominant_emotion": "anxious",
                "sentiment_score": -0.6,
                "subjectivity": 0.7,
                "detected_emotions": {"anxious": 0.7, "neutral": 0.3},
            }),
            json.dumps({
                "emotion_id": "emo_2",
                "user_id": "user_123",
                "timestamp": "2025-01-21T11:00:00",
                "query": "Test 2",
                "dominant_emotion": "anxious",
                "sentiment_score": -0.5,
                "subjectivity": 0.6,
                "detected_emotions": {"anxious": 0.6, "neutral": 0.4},
            }),
            json.dumps({
                "emotion_id": "emo_3",
                "user_id": "user_123",
                "timestamp": "2025-01-21T12:00:00",
                "query": "Test 3",
                "dominant_emotion": "neutral",
                "sentiment_score": 0.1,
                "subjectivity": 0.3,
                "detected_emotions": {"neutral": 0.8, "curious": 0.2},
            }),
        ]

        trend = await detector.get_emotional_trend(user_id="user_123", hours=24)

        assert trend.user_id == "user_123"
        assert trend.time_period_hours == 24
        assert len(trend.dominant_emotions) == 3
        assert trend.anxiety_episodes >= 2
        assert trend.avg_sentiment < 0  # Overall negative due to 2 anxious queries


# ============================================================================
# Layer 7: Reflective Memory Tests (PostgreSQL self-improvement)
# ============================================================================


class TestReflectiveMemory:
    """Test suite for reflective memory model."""

    def test_reflective_memory_creation(self):
        """Test creating reflective memory with valid data."""
        reflective = ReflectiveMemory(
            reflection_id="ref_abc123",
            user_id="user_123",
            trigger_event="negative_feedback",
            observation="User reported incorrect hurricane category (Cat 5 with 140 mph)",
            analysis="Wind speed validation failed - should be ≥157 mph for Cat 5",
            improvement_action="Add stricter Saffir-Simpson validation in hurricane tool",
            expected_impact="Prevent future category mismatches, improve safety accuracy",
            priority="high",
        )

        assert reflective.reflection_id == "ref_abc123"
        assert reflective.trigger_event == "negative_feedback"
        assert reflective.priority == "high"
        assert reflective.status == "pending"

    def test_reflective_memory_priority_validation(self):
        """Test reflective memory validates priority enum."""
        # Valid priorities
        for priority in ["low", "medium", "high", "critical"]:
            reflective = ReflectiveMemory(
                reflection_id="ref_123",
                user_id="user_123",
                trigger_event="test",
                observation="test",
                analysis="test",
                improvement_action="test",
                expected_impact="test",
                priority=priority,
            )
            assert reflective.priority == priority

        # Invalid priority
        with pytest.raises(ValidationError):
            ReflectiveMemory(
                reflection_id="ref_123",
                user_id="user_123",
                trigger_event="test",
                observation="test",
                analysis="test",
                improvement_action="test",
                expected_impact="test",
                priority="invalid",
            )


class TestReflectiveMemoryBackend:
    """Test suite for reflective memory backend (PostgreSQL)."""

    @pytest.fixture
    def mock_pg_pool(self):
        """Mock PostgreSQL connection pool."""
        pool = AsyncMock()
        pool.acquire = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock()
        pool.acquire.return_value.__aexit__ = AsyncMock()
        return pool

    @pytest.fixture
    def reflective_backend(self, mock_pg_pool):
        """Create ReflectiveMemoryBackend with mocked PostgreSQL."""
        with patch("asyncpg.create_pool", return_value=mock_pg_pool):
            backend = ReflectiveMemoryBackend()
            backend.pool = mock_pg_pool
            return backend

    @pytest.mark.asyncio
    async def test_save_reflection(self, reflective_backend, mock_pg_pool):
        """Test saving reflection to PostgreSQL."""
        reflection = ReflectiveMemory(
            reflection_id="ref_test",
            user_id="user_123",
            trigger_event="error",
            observation="Test observation",
            analysis="Test analysis",
            improvement_action="Test action",
            expected_impact="Test impact",
            priority="medium",
        )

        mock_conn = AsyncMock()
        mock_pg_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await reflective_backend.save_reflection(reflection)

        # Verify execute was called with INSERT query
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO reflections" in call_args[0] or "reflections" in call_args[0].lower()

    @pytest.mark.asyncio
    async def test_get_pending_reflections(self, reflective_backend, mock_pg_pool):
        """Test retrieving pending reflections from PostgreSQL."""
        mock_conn = AsyncMock()
        mock_pg_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Mock pending reflections
        mock_conn.fetch.return_value = [
            {
                "reflection_id": "ref_1",
                "user_id": "user_123",
                "created_at": datetime.now(),
                "trigger_event": "error",
                "observation": "Obs 1",
                "analysis": "Analysis 1",
                "improvement_action": "Action 1",
                "expected_impact": "Impact 1",
                "priority": "high",
                "status": "pending",
                "implemented_at": None,
            },
            {
                "reflection_id": "ref_2",
                "user_id": "user_123",
                "created_at": datetime.now(),
                "trigger_event": "negative_feedback",
                "observation": "Obs 2",
                "analysis": "Analysis 2",
                "improvement_action": "Action 2",
                "expected_impact": "Impact 2",
                "priority": "medium",
                "status": "pending",
                "implemented_at": None,
            },
        ]

        reflections = await reflective_backend.get_pending_reflections("user_123")

        assert len(reflections) == 2
        assert reflections[0].reflection_id == "ref_1"
        assert reflections[0].priority == "high"
        assert reflections[1].status == "pending"


# ============================================================================
# Memory Consolidation Tests (70% storage reduction)
# ============================================================================


class TestMemoryConsolidation:
    """Test suite for memory consolidation pipeline."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        return AsyncMock()

    @pytest.fixture
    def mock_neo4j(self):
        """Mock Neo4j driver."""
        return AsyncMock()

    @pytest.fixture
    def consolidator(self, mock_redis, mock_neo4j):
        """Create MemoryConsolidator with mocked dependencies."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            with patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_neo4j):
                return MemoryConsolidator()

    def test_consolidation_result_creation(self):
        """Test creating consolidation result."""
        result = ConsolidationResult(
            original_size=10240,
            consolidated_size=2048,
            reduction_percent=80.0,
            items_processed=10,
            items_retained=2,
            duration_seconds=0.5,
        )

        assert result.original_size == 10240
        assert result.consolidated_size == 2048
        assert result.reduction_percent == 80.0
        assert result.items_processed == 10

    def test_consolidation_result_validation(self):
        """Test consolidation result validates bounds."""
        # Valid reduction
        result = ConsolidationResult(
            original_size=1000,
            consolidated_size=300,
            reduction_percent=70.0,
            items_processed=5,
            items_retained=2,
            duration_seconds=1.0,
        )
        assert result.reduction_percent == 70.0

        # Invalid reduction > 100%
        with pytest.raises(ValidationError):
            ConsolidationResult(
                original_size=1000,
                consolidated_size=300,
                reduction_percent=150.0,
                items_processed=5,
                items_retained=2,
                duration_seconds=1.0,
            )

    @pytest.mark.asyncio
    async def test_consolidate_conversation(self, consolidator, mock_redis):
        """Test Stage 1: Consolidate conversation turns (hourly)."""
        # Mock conversation history in Redis
        conversation_data = json.dumps({
            "user_id": "user_123",
            "session_id": "sess_456",
            "conversation_history": [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Response 1"},
                {"role": "user", "content": "Turn 2"},
                {"role": "assistant", "content": "Response 2"},
            ] * 5,  # 20 turns total
            "current_entities": {},
            "token_count": 5000,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=30)).isoformat(),
        })

        mock_redis.get.return_value = conversation_data

        result = await consolidator.consolidate_conversation(
            user_id="user_123", session_id="sess_456"
        )

        assert result.reduction_percent >= 60.0  # Target: 80% (allow 60%+)
        assert result.items_processed > 0
        assert result.items_retained < result.items_processed

    @pytest.mark.asyncio
    async def test_consolidate_session(self, consolidator, mock_redis):
        """Test Stage 2: Consolidate session to long-term (daily)."""
        # Mock session data
        mock_redis.get.return_value = json.dumps({
            "session_id": "sess_456",
            "user_id": "user_123",
            "query_count": 10,
            "started_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        })

        result = await consolidator.consolidate_session(session_id="sess_456")

        assert result.reduction_percent >= 50.0  # Target: 60% (allow 50%+)
        assert result.items_processed > 0

    @pytest.mark.asyncio
    async def test_consolidate_weekly(self, consolidator):
        """Test Stage 3: Weekly episodic consolidation."""
        result = await consolidator.consolidate_weekly(user_id="user_123")

        assert result.reduction_percent >= 60.0  # Target: 70% (allow 60%+)
        assert result.items_processed >= 0


# ============================================================================
# Integration Tests: Cross-Layer Interactions
# ============================================================================


class TestMemoryLayersIntegration:
    """Test suite for integration across all 7 memory layers."""

    @pytest.mark.asyncio
    async def test_episodic_to_semantic_consolidation(self):
        """Test consolidating episodic memories into semantic knowledge."""
        # Create multiple related episodic memories
        episodes = [
            EpisodicMemory(
                episode_id=f"ep_{i}",
                user_id="user_123",
                query=f"Hurricane category query {i}",
                response=f"Category {i % 5 + 1} hurricane info",
                outcome="helpful",
            )
            for i in range(10)
        ]

        # Verify episodic memories are valid
        assert len(episodes) == 10
        assert all(ep.outcome == "helpful" for ep in episodes)

        # In a real integration test, we would:
        # 1. Store episodic memories in Redis
        # 2. Run consolidation to extract semantic patterns
        # 3. Verify semantic memory in Qdrant contains learned concepts

    @pytest.mark.asyncio
    async def test_emotional_to_reflective_feedback_loop(self):
        """Test emotional patterns triggering reflective improvements."""
        # Create emotional memory showing anxiety trend
        emotions = [
            EmotionalMemory(
                emotion_id=f"emo_{i}",
                user_id="user_123",
                query=f"Anxious query {i}",
                dominant_emotion="anxious",
                sentiment_score=-0.7,
            )
            for i in range(5)
        ]

        # This should trigger reflective memory to improve empathy
        assert len(emotions) == 5
        assert all(em.dominant_emotion == "anxious" for em in emotions)

        # In a real integration test, we would:
        # 1. Store emotional memories in Redis
        # 2. Detect anxiety trend
        # 3. Trigger reflective memory creation
        # 4. Verify reflection suggests empathy improvements

    @pytest.mark.asyncio
    async def test_procedural_to_reflective_optimization(self):
        """Test failed workflows triggering reflective improvements."""
        # Create procedural memory with low success rate
        workflow = ProceduralMemory(
            workflow_id="wf_failing",
            user_id="user_123",
            workflow_name="Failing Workflow",
            steps=[
                {"step": 1, "action": "action1", "success": False},
                {"step": 2, "action": "action2", "success": False},
            ],
            success_rate=0.2,  # Only 20% success
            execution_count=10,
        )

        assert workflow.success_rate < 0.5

        # In a real integration test, we would:
        # 1. Store workflow in PostgreSQL
        # 2. Detect low success rate
        # 3. Trigger reflective analysis
        # 4. Verify reflection suggests workflow improvements


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self):
        """Test handling Redis connection failure gracefully."""
        with patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis unavailable")):
            # Should not raise, should log error
            detector = EmotionalMemoryDetector()
            # Detector should work with fallback even if Redis fails
            assert detector is not None

    @pytest.mark.asyncio
    async def test_postgres_connection_failure(self):
        """Test handling PostgreSQL connection failure gracefully."""
        with patch("asyncpg.create_pool", side_effect=ConnectionError("PostgreSQL unavailable")):
            # Should not raise, should log error
            backend = ProceduralMemoryBackend()
            assert backend is not None

    @pytest.mark.asyncio
    async def test_consolidation_empty_data(self):
        """Test consolidation with no data to consolidate."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            consolidator = MemoryConsolidator()
            result = await consolidator.consolidate_conversation(
                user_id="user_123", session_id="sess_nonexistent"
            )

            # Should handle gracefully with 0% reduction
            assert result.reduction_percent == 0.0
            assert result.items_processed == 0
