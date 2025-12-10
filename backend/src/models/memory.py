"""Memory models for Level 3a/3b/3c: Complete 7-layer memory architecture.

CRITICAL RULES (from Level 2 learnings):
- SINGLE SOURCE OF TRUTH: All models in backend/src/models/ ONLY
- PROGRESSIVE ENHANCEMENT: Add fields to existing models when possible
- NO PREMATURE CREATION: Create only what's needed for current level

7-Layer Memory Architecture:
- Layer 1 (Short-term): Redis (TTL 30min, conversation context)
- Layer 2 (Long-term): Graphiti + Neo4j (user profiles, temporal facts)
- Layer 3 (Episodic): Redis (TTL 7 days, event history)
- Layer 4 (Semantic): Qdrant (domain knowledge, concepts)
- Layer 5 (Procedural): PostgreSQL (workflow optimization)
- Layer 6 (Emotional): Redis (TTL 7 days, sentiment analysis)
- Layer 7 (Reflective): PostgreSQL (self-improvement, learnings)

Models:
Level 3a (Layers 1-2):
1. ConversationContext - Session conversation state (Redis)
2. SessionState - Session management metadata
3. UserProfile - Persistent user preferences (Graphiti + Neo4j)
4. TemporalFact - Time-bound knowledge with valid_from/valid_to

Level 3c (Layers 3-7):
5. EpisodicMemory - Specific past events with temporal context
6. SemanticMemory - General knowledge and facts
7. ProceduralMemory - How-to knowledge (workflows learned)
8. EmotionalMemory - Sentiment and emotional state
9. ReflectiveMemory - Self-improvement and learning
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ==================== SHORT-TERM MEMORY ====================


class ConversationContext(BaseModel):
    """Short-term memory: Session conversation state (Redis).

    Stores recent conversation history, entity tracking, and session metadata.
    TTL: 30 minutes (configurable via REDIS_TTL_SECONDS)

    Example:
        >>> context = ConversationContext(
        ...     user_id="user_123",
        ...     session_id="session_abc",
        ...     conversation_history=[
        ...         {"role": "user", "content": "Weather in London?"},
        ...         {"role": "assistant", "content": "London is 18°C, sunny"}
        ...     ],
        ...     current_entities={"location": "London"}
        ... )
    """

    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Unique session identifier")
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of conversation turns (role, content)",
    )
    current_entities: dict[str, str] = Field(
        default_factory=dict,
        description="Tracked entities from conversation (e.g., {'location': 'London'})",
    )
    token_count: int = Field(
        default=0,
        description="Total tokens used in this session",
        ge=0,
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Session creation timestamp",
    )
    expires_at: datetime = Field(
        ...,
        description="Session expiration timestamp (TTL)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_123",
                    "session_id": "session_abc",
                    "conversation_history": [
                        {"role": "user", "content": "Weather in London?"},
                        {"role": "assistant", "content": "London is 18°C, sunny"},
                    ],
                    "current_entities": {"location": "London"},
                    "token_count": 450,
                    "created_at": "2025-01-21T10:00:00Z",
                    "expires_at": "2025-01-21T10:30:00Z",
                }
            ]
        }
    }


class SessionState(BaseModel):
    """Session management metadata.

    Tracks session lifecycle, activity, and usage statistics.

    Example:
        >>> state = SessionState(
        ...     session_id="session_abc",
        ...     user_id="user_123",
        ...     query_count=5
        ... )
    """

    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User ID for this session")
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="Session start time",
    )
    last_activity: datetime = Field(
        default_factory=datetime.now,
        description="Last activity timestamp",
    )
    query_count: int = Field(
        default=0,
        description="Number of queries in this session",
        ge=0,
    )
    is_active: bool = Field(
        default=True,
        description="Session active status",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "session_abc",
                    "user_id": "user_123",
                    "started_at": "2025-01-21T10:00:00Z",
                    "last_activity": "2025-01-21T10:15:00Z",
                    "query_count": 5,
                    "is_active": True,
                }
            ]
        }
    }


# ==================== LONG-TERM MEMORY ====================


class UserProfile(BaseModel):
    """Long-term memory: Persistent user preferences (Graphiti + Neo4j).

    Stores user preferences, interests, and historical context.
    Persists across sessions with temporal validity.

    Example:
        >>> profile = UserProfile(
        ...     user_id="user_123",
        ...     name="John Doe",
        ...     home_location="London",
        ...     preferred_units="celsius"
        ... )
    """

    user_id: str = Field(..., description="Unique user identifier")
    name: str | None = Field(
        default=None,
        description="User's name (optional)",
    )
    home_location: str | None = Field(
        default=None,
        description="User's home location (city or zip)",
    )
    zip_code: str | None = Field(
        default=None,
        description="User's zip code (optional)",
    )
    timezone: str = Field(
        default="UTC",
        description="User's timezone (IANA format, e.g., 'America/New_York')",
    )

    # Preferences
    preferred_units: str = Field(
        default="celsius",
        description="Temperature units preference (celsius or fahrenheit)",
    )
    preferred_detail_level: str = Field(
        default="moderate",
        description="Response detail level (minimal, moderate, detailed)",
    )
    preferred_response_style: str = Field(
        default="conversational",
        description="Response style (conversational, technical, casual)",
    )

    # Interests
    interests: list[str] = Field(
        default_factory=list,
        description="User interests and topics (e.g., ['hurricanes', 'skiing'])",
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Profile creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last profile update timestamp",
    )
    total_queries: int = Field(
        default=0,
        description="Total queries by this user",
        ge=0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_123",
                    "name": "John Doe",
                    "home_location": "London",
                    "zip_code": "SW1A 1AA",
                    "timezone": "Europe/London",
                    "preferred_units": "celsius",
                    "preferred_detail_level": "moderate",
                    "preferred_response_style": "conversational",
                    "interests": ["hurricanes", "skiing", "beach weather"],
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-21T10:00:00Z",
                    "total_queries": 127,
                }
            ]
        }
    }


class TemporalFact(BaseModel):
    """Time-bound knowledge with valid_from/valid_to timestamps.

    Stores facts about users or weather patterns with temporal validity.
    Used by Graphiti for temporal knowledge graphs.

    Example:
        >>> fact = TemporalFact(
        ...     user_id="user_123",
        ...     fact_type="preference",
        ...     content="User prefers detailed hurricane forecasts",
        ...     valid_from="2025-01-01T00:00:00Z"
        ... )
    """

    fact_id: str = Field(..., description="Unique fact identifier")
    user_id: str = Field(..., description="User this fact relates to")
    fact_type: str = Field(
        ...,
        description="Type of fact (preference, location, interest, pattern)",
    )
    content: str = Field(
        ...,
        description="The actual fact content",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    source: str = Field(
        default="conversation",
        description="Source of this fact (conversation, explicit, inferred)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    # Temporal validity
    valid_from: datetime = Field(
        default_factory=datetime.now,
        description="Fact validity start time",
    )
    valid_to: datetime | None = Field(
        default=None,
        description="Fact validity end time (None = indefinite)",
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Fact creation timestamp",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fact_id": "fact_abc123",
                    "user_id": "user_123",
                    "fact_type": "preference",
                    "content": "User prefers detailed hurricane forecasts",
                    "confidence": 0.95,
                    "source": "conversation",
                    "metadata": {"inferred_from": "repeated detailed questions"},
                    "valid_from": "2025-01-01T00:00:00Z",
                    "valid_to": None,
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ]
        }
    }


# ==================== EPISODIC MEMORY (Layer 3) ====================


class EpisodicMemory(BaseModel):
    """Layer 3: Episodic memory - Specific past events with temporal context (Redis).

    Stores individual query-response episodes with outcomes and emotional context.
    TTL: 7 days (configurable via REDIS_EPISODIC_TTL_SECONDS)

    Example:
        >>> episode = EpisodicMemory(
        ...     episode_id="ep_abc123",
        ...     user_id="user_123",
        ...     query="Will Hurricane Milton hit Florida?",
        ...     response="Hurricane Milton is projected to make landfall...",
        ...     outcome="helpful"
        ... )
    """

    episode_id: str = Field(..., description="Unique episode identifier")
    user_id: str = Field(..., description="User who made this query")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this episode occurred"
    )
    query: str = Field(..., description="User's query")
    response: str = Field(..., description="Agent's response")
    outcome: str = Field(
        default="unknown",
        description="Query outcome (helpful, confusing, error, unknown)"
    )
    emotional_context: str | None = Field(
        default=None,
        description="User's emotional state during query (anxious, curious, frustrated, etc.)"
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="List of tools used to answer this query"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "episode_id": "ep_abc123",
                    "user_id": "user_123",
                    "timestamp": "2025-01-21T10:00:00Z",
                    "query": "Will Hurricane Milton hit Florida?",
                    "response": "Hurricane Milton is projected to make landfall near Tampa Bay...",
                    "outcome": "helpful",
                    "emotional_context": "anxious",
                    "tools_used": ["search_hurricanes", "get_hurricane_forecast"]
                }
            ]
        }
    }


# ==================== SEMANTIC MEMORY (Layer 4) ====================


class SemanticMemory(BaseModel):
    """Layer 4: Semantic memory - General knowledge and facts (Qdrant).

    Stores domain knowledge about weather concepts, terminology, and relationships.
    Persists indefinitely, embedded as vectors for semantic search.

    Example:
        >>> concept = SemanticMemory(
        ...     concept="cold front",
        ...     definition="A boundary where cold air replaces warm air...",
        ...     category="meteorology",
        ...     related_concepts=["warm front", "occluded front"]
        ... )
    """

    concept: str = Field(..., description="Weather concept or term")
    definition: str = Field(..., description="Clear definition of the concept")
    category: str = Field(
        default="general",
        description="Category (meteorology, safety, forecasting, instruments, etc.)"
    )
    related_concepts: list[str] = Field(
        default_factory=list,
        description="Related concepts for knowledge graph"
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Real-world examples of this concept"
    )
    source: str = Field(
        default="curated",
        description="Source of this knowledge (curated, learned, user_contributed)"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence in this definition (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "concept": "cold front",
                    "definition": "A boundary where a mass of cold air replaces a mass of warm air, often bringing precipitation and temperature drops.",
                    "category": "meteorology",
                    "related_concepts": ["warm front", "occluded front", "stationary front"],
                    "examples": ["Cold front brings thunderstorms to Midwest"],
                    "source": "curated",
                    "confidence": 1.0
                }
            ]
        }
    }


# ==================== PROCEDURAL MEMORY (Layer 5) ====================


class ProceduralMemory(BaseModel):
    """Layer 5: Procedural memory - How-to knowledge and workflows (PostgreSQL).

    Stores learned workflows and tool sequences with success rates.
    Used for workflow optimization and intelligent tool selection.

    Example:
        >>> workflow = ProceduralMemory(
        ...     task_name="hurricane_forecast",
        ...     steps=["search_hurricanes", "get_hurricane_forecast", "analyze_storm_impact"],
        ...     success_rate=0.94
        ... )
    """

    task_name: str = Field(..., description="Type of task (hurricane_forecast, city_weather, etc.)")
    steps: list[str] = Field(..., description="Ordered list of tools/steps in workflow")
    success_rate: float = Field(
        default=0.0,
        description="Success rate for this workflow (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    avg_response_time: float = Field(
        default=0.0,
        description="Average response time in seconds",
        ge=0.0
    )
    usage_count: int = Field(
        default=0,
        description="Number of times this workflow was used",
        ge=0
    )
    last_used: datetime = Field(
        default_factory=datetime.now,
        description="Last time this workflow was executed"
    )
    feedback_score: float | None = Field(
        default=None,
        description="User feedback score (1-5 stars)",
        ge=1.0,
        le=5.0
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task_name": "hurricane_forecast",
                    "steps": ["search_hurricanes", "get_hurricane_forecast", "analyze_storm_impact"],
                    "success_rate": 0.94,
                    "avg_response_time": 5.2,
                    "usage_count": 127,
                    "last_used": "2025-01-21T10:00:00Z",
                    "feedback_score": 4.5
                }
            ]
        }
    }


# ==================== EMOTIONAL MEMORY (Layer 6) ====================


class EmotionalMemory(BaseModel):
    """Layer 6: Emotional memory - Sentiment and emotional state (Redis).

    Tracks user emotions and sentiment over time to adapt responses.
    TTL: 7 days (configurable via REDIS_EMOTIONAL_TTL_SECONDS)

    Example:
        >>> emotion = EmotionalMemory(
        ...     user_id="user_123",
        ...     dominant_emotion="anxious",
        ...     sentiment_score=-0.4,
        ...     triggers=["hurricane approaching", "evacuation zone mentioned"]
        ... )
    """

    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this emotion was detected"
    )
    dominant_emotion: str = Field(
        default="neutral",
        description="Primary detected emotion (anxious, curious, frustrated, excited, neutral)"
    )
    sentiment_score: float = Field(
        default=0.0,
        description="Sentiment polarity score (-1.0 negative to +1.0 positive)",
        ge=-1.0,
        le=1.0
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence in emotion detection (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    triggers: list[str] = Field(
        default_factory=list,
        description="What caused this emotion (keywords, topics)"
    )
    query_context: str | None = Field(
        default=None,
        description="The query that triggered this emotion"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_123",
                    "timestamp": "2025-01-21T10:00:00Z",
                    "dominant_emotion": "anxious",
                    "sentiment_score": -0.4,
                    "confidence": 0.85,
                    "triggers": ["hurricane approaching", "evacuation zone B"],
                    "query_context": "Should I evacuate for Hurricane Milton?"
                }
            ]
        }
    }


# ==================== REFLECTIVE MEMORY (Layer 7) ====================


class ReflectiveMemory(BaseModel):
    """Layer 7: Reflective memory - Self-improvement and learning (PostgreSQL).

    Stores agent learnings from mistakes, feedback, and patterns.
    Used for meta-cognitive self-improvement and knowledge evolution.

    Example:
        >>> reflection = ReflectiveMemory(
        ...     insight="Always verify hurricane category matches wind speed",
        ...     learned_from="ep_abc123",
        ...     applied_count=15,
        ...     effectiveness=0.92
        ... )
    """

    reflection_id: str = Field(..., description="Unique reflection identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this learning occurred"
    )
    trigger: str = Field(
        ...,
        description="What triggered this reflection (user_feedback_negative, error_pattern, etc.)"
    )
    context: str = Field(
        ...,
        description="Situation that led to this learning"
    )
    analysis: str = Field(
        ...,
        description="Analysis of what went wrong or could improve"
    )
    insight: str = Field(
        ...,
        description="Key learning or takeaway"
    )
    learned_from: str = Field(
        ...,
        description="Episode ID or pattern that triggered learning"
    )
    action_taken: str | None = Field(
        default=None,
        description="Action taken to apply this learning"
    )
    applied_count: int = Field(
        default=0,
        description="Number of times this learning was applied",
        ge=0
    )
    effectiveness: float = Field(
        default=0.0,
        description="Measured effectiveness of this learning (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    improvement_score: float | None = Field(
        default=None,
        description="Measurable improvement from applying this learning (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    applied: bool = Field(
        default=False,
        description="Whether this learning has been applied to the system"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "reflection_id": "refl_abc123",
                    "timestamp": "2025-01-21T10:00:00Z",
                    "trigger": "user_feedback_negative",
                    "context": "User said forecast was wrong for London yesterday",
                    "analysis": "Forecast showed 'sunny' but it rained. MCP data was outdated.",
                    "insight": "Always check forecast timestamp, prefer newer data sources",
                    "learned_from": "ep_xyz789",
                    "action_taken": "Add data freshness check to get_forecast tool",
                    "applied_count": 15,
                    "effectiveness": 0.92,
                    "improvement_score": 0.15,
                    "applied": True
                }
            ]
        }
    }
