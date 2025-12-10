"""Emotional memory detection and analysis for Level 3c (Layer 6).

This module implements emotional memory detection using sentiment analysis
to track user emotional states over time. This enables:
- Empathetic responses based on emotional context
- Anxiety detection for safety-critical warnings
- Frustration detection for adaptive UX
- Excitement detection for engagement optimization

Emotional States Tracked:
- anxious: User shows worry or concern (e.g., hurricane approaching)
- frustrated: User shows annoyance or confusion
- excited: User shows enthusiasm or interest
- curious: User asks exploratory questions
- neutral: Default state, no strong emotion

Storage Layer:
- Redis with 7-day TTL (Layer 6)
- Key pattern: emotion:{user_id}:{timestamp_ms}

Sentiment Analysis:
- TextBlob for polarity and subjectivity
- Custom rules for weather-specific emotions
- Temporal aggregation for emotional trends

Example:
    >>> from backend.src.memory.emotional import EmotionalMemoryDetector
    >>> detector = EmotionalMemoryDetector()
    >>>
    >>> # Detect emotion from query
    >>> emotion = await detector.detect_emotion(
    ...     user_id="user_123",
    ...     query="Is the hurricane going to hit us? I'm really worried!",
    ...     session_id="sess_456"
    ... )
    >>> print(emotion.dominant_emotion)
    'anxious'
    >>> print(emotion.sentiment_score)
    -0.6  # Negative sentiment
    >>>
    >>> # Get emotional trend
    >>> trend = await detector.get_emotional_trend(user_id="user_123", hours=24)
    >>> print(trend.dominant_emotions)
    ['anxious', 'anxious', 'curious', 'neutral']
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from backend.config.settings import Settings
from backend.src.models.memory import EmotionalMemory

logger = logging.getLogger(__name__)

# Try importing TextBlob (optional dependency)
try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning(
        "TextBlob not installed. Sentiment analysis will use basic rule-based fallback. "
        "Install with: pip install textblob"
    )


# ============================================================================
# Pydantic Models
# ============================================================================


class EmotionalTrend(BaseModel):
    """Emotional trend over time period."""

    user_id: str
    time_period_hours: int
    dominant_emotions: list[str] = Field(
        ..., description="List of dominant emotions in chronological order"
    )
    avg_sentiment: float = Field(
        ..., description="Average sentiment score (-1.0 to +1.0)", ge=-1.0, le=1.0
    )
    emotional_volatility: float = Field(
        ...,
        description="Standard deviation of sentiment (0.0 = stable, 1.0 = volatile)",
        ge=0.0,
        le=1.0,
    )
    anxiety_episodes: int = Field(..., description="Number of anxiety episodes", ge=0)
    frustration_episodes: int = Field(
        ..., description="Number of frustration episodes", ge=0
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_123",
                    "time_period_hours": 24,
                    "dominant_emotions": ["anxious", "anxious", "curious", "neutral"],
                    "avg_sentiment": -0.2,
                    "emotional_volatility": 0.4,
                    "anxiety_episodes": 2,
                    "frustration_episodes": 0,
                }
            ]
        }
    }


# ============================================================================
# Emotional Memory Detector
# ============================================================================


class EmotionalMemoryDetector:
    """Emotional memory detection using sentiment analysis.

    This class provides sentiment analysis and emotional state detection
    for user queries, enabling empathetic and context-aware responses.

    Attributes:
        settings: Application settings
        redis_client: Redis client for emotional memory storage
    """

    def __init__(
        self, settings: Settings | None = None, redis_client: Any | None = None
    ) -> None:
        """Initialize emotional memory detector.

        Args:
            settings: Application settings (auto-loaded if None)
            redis_client: Redis client (optional, for testing)
        """
        self.settings = settings or Settings()

        # Initialize Redis client
        if redis_client is not None:
            self.redis_client = redis_client
        else:
            import redis.asyncio as redis
            from backend.config.memory_config import memory_config

            self.redis_client = redis.from_url(
                memory_config.REDIS_URL,
                decode_responses=True,
                max_connections=memory_config.REDIS_MAX_CONNECTIONS,
            )

        if not TEXTBLOB_AVAILABLE:
            logger.warning(
                "Running with rule-based sentiment analysis (TextBlob not available)"
            )

        logger.info("Initialized EmotionalMemoryDetector with Redis")

    # ========================================================================
    # Emotion Detection
    # ========================================================================

    async def detect_emotion(
        self,
        user_id: str,
        query: str,
        session_id: str | None = None,
        context: str | None = None,
    ) -> EmotionalMemory:
        """Detect emotional state from user query.

        Analyzes the query text using:
        1. TextBlob sentiment analysis (polarity and subjectivity)
        2. Weather-specific emotional keywords
        3. Punctuation and capitalization patterns
        4. Contextual signals (previous emotional state)

        Args:
            user_id: User identifier
            query: User's query text
            session_id: Optional session identifier
            context: Optional additional context

        Returns:
            EmotionalMemory object with detected emotion

        Example:
            >>> emotion = await detector.detect_emotion(
            ...     user_id="user_123",
            ...     query="Is the hurricane going to hit us? I'm really worried!",
            ...     session_id="sess_456"
            ... )
            >>> print(emotion.dominant_emotion)
            'anxious'
        """
        timestamp = datetime.now()

        # Step 1: Get sentiment score using TextBlob or fallback
        sentiment_score, confidence = self._analyze_sentiment(query)

        # Step 2: Detect emotion from keywords and patterns
        dominant_emotion, triggers = self._classify_emotion(query, sentiment_score)

        # Step 3: Create EmotionalMemory object
        emotional_memory = EmotionalMemory(
            user_id=user_id,
            timestamp=timestamp,
            dominant_emotion=dominant_emotion,
            sentiment_score=sentiment_score,
            confidence=confidence,
            triggers=triggers,
            query_context=context or query[:200],  # Store first 200 chars
        )

        # Step 4: Store in Redis (Layer 6)
        await self._store_emotional_memory(emotional_memory)

        logger.info(
            f"Detected emotion: {dominant_emotion} (sentiment={sentiment_score:.2f}, confidence={confidence:.2f})"
        )

        return emotional_memory

    def _analyze_sentiment(self, text: str) -> tuple[float, float]:
        """Analyze sentiment using TextBlob or rule-based fallback.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (sentiment_score, confidence)
            - sentiment_score: -1.0 (negative) to +1.0 (positive)
            - confidence: 0.0 (low) to 1.0 (high)
        """
        if TEXTBLOB_AVAILABLE:
            # Use TextBlob sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1.0 to +1.0
            subjectivity = blob.sentiment.subjectivity  # 0.0 to 1.0

            # Confidence = subjectivity (higher subjectivity = more confident)
            confidence = min(0.9, 0.5 + subjectivity / 2.0)

            return polarity, confidence

        else:
            # Rule-based fallback
            return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> tuple[float, float]:
        """Rule-based sentiment analysis (fallback when TextBlob unavailable).

        Args:
            text: Input text

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        text_lower = text.lower()

        # Positive keywords
        positive_keywords = [
            "thank",
            "great",
            "awesome",
            "excellent",
            "perfect",
            "love",
            "amazing",
            "wonderful",
        ]
        positive_count = sum(
            1 for keyword in positive_keywords if keyword in text_lower
        )

        # Negative keywords
        negative_keywords = [
            "worried",
            "concerned",
            "scared",
            "anxious",
            "dangerous",
            "bad",
            "terrible",
            "awful",
            "frustrated",
            "confused",
        ]
        negative_count = sum(
            1 for keyword in negative_keywords if keyword in text_lower
        )

        # Calculate sentiment
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0, 0.3  # Neutral with low confidence

        sentiment_score = (positive_count - negative_count) / max(
            1, total_keywords
        ) * 0.8
        confidence = min(0.8, 0.4 + (total_keywords / 10.0))

        return sentiment_score, confidence

    def _classify_emotion(
        self, query: str, sentiment_score: float
    ) -> tuple[str, list[str]]:
        """Classify emotion from query text and sentiment.

        Args:
            query: User query text
            sentiment_score: Sentiment score from analysis

        Returns:
            Tuple of (dominant_emotion, triggers)
        """
        query_lower = query.lower()
        triggers: list[str] = []

        # Emotion detection rules (weather-specific)
        emotion_rules = {
            "anxious": [
                "worried",
                "concerned",
                "scared",
                "anxious",
                "nervous",
                "safe",
                "danger",
                "evacuate",
                "hurricane",
                "storm",
            ],
            "frustrated": [
                "frustrated",
                "confused",
                "don't understand",
                "doesn't make sense",
                "wrong",
                "error",
                "not working",
            ],
            "excited": [
                "excited",
                "amazing",
                "awesome",
                "cool",
                "interesting",
                "fascinating",
                "wow",
            ],
            "curious": [
                "how",
                "why",
                "what",
                "when",
                "where",
                "tell me more",
                "explain",
                "understand",
            ],
        }

        # Find matching emotions
        emotion_scores: dict[str, int] = {}
        for emotion, keywords in emotion_rules.items():
            matches = [keyword for keyword in keywords if keyword in query_lower]
            if matches:
                emotion_scores[emotion] = len(matches)
                triggers.extend(matches)

        # Determine dominant emotion
        if emotion_scores:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        elif sentiment_score < -0.3:
            dominant_emotion = "anxious"
            triggers.append("negative_sentiment")
        elif sentiment_score > 0.3:
            dominant_emotion = "excited"
            triggers.append("positive_sentiment")
        else:
            dominant_emotion = "neutral"

        # Check for exclamation marks and question marks
        if "!" in query:
            if dominant_emotion == "neutral":
                dominant_emotion = "excited" if sentiment_score > 0 else "anxious"
            triggers.append("exclamation_mark")

        if query.count("?") >= 2:
            if dominant_emotion == "neutral":
                dominant_emotion = "curious"
            triggers.append("multiple_questions")

        return dominant_emotion, triggers

    # ========================================================================
    # Redis Storage
    # ========================================================================

    async def _store_emotional_memory(self, memory: EmotionalMemory) -> None:
        """Store emotional memory in Redis with 7-day TTL.

        Args:
            memory: EmotionalMemory object to store
        """
        key = f"emotion:{memory.user_id}:{int(memory.timestamp.timestamp() * 1000)}"
        ttl_seconds = 7 * 24 * 60 * 60  # 7 days

        try:
            # Store emotional memory in Redis with TTL
            await self.redis_client.setex(
                name=key,
                time=ttl_seconds,
                value=memory.model_dump_json()
            )
            logger.debug(f"✅ Stored emotional memory: {key} (TTL={ttl_seconds}s)")
        except Exception as e:
            logger.error(f"Failed to store emotional memory {key}: {e}")

    async def get_recent_emotions(
        self, user_id: str, limit: int = 10
    ) -> list[EmotionalMemory]:
        """Get recent emotional memories for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of memories to return

        Returns:
            List of EmotionalMemory objects (most recent first)
        """
        try:
            # Scan for emotion keys matching pattern
            pattern = f"emotion:{user_id}:*"
            cursor = 0
            keys = []

            while True:
                cursor, batch = await self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                keys.extend(batch)
                if cursor == 0:
                    break

            if not keys:
                logger.debug(f"No emotional memories found for user={user_id}")
                return []

            # Sort keys by timestamp (descending) - timestamp is in key
            # Key format: emotion:{user_id}:{timestamp_ms}
            sorted_keys = sorted(keys, key=lambda k: int(k.split(":")[-1]), reverse=True)

            # Fetch memories (up to limit)
            memories = []
            for key in sorted_keys[:limit]:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        memory = EmotionalMemory.model_validate_json(data)
                        memories.append(memory)
                except Exception as e:
                    logger.warning(f"Failed to parse emotional memory from {key}: {e}")

            logger.debug(f"✅ Fetched {len(memories)} emotional memories for user={user_id}")
            return memories

        except Exception as e:
            logger.error(f"Failed to get recent emotions for {user_id}: {e}")
            return []

    # ========================================================================
    # Emotional Trends
    # ========================================================================

    async def get_emotional_trend(
        self, user_id: str, hours: int = 24
    ) -> EmotionalTrend:
        """Calculate emotional trend over time period.

        Args:
            user_id: User identifier
            hours: Time period in hours (default 24)

        Returns:
            EmotionalTrend object with aggregated metrics

        Example:
            >>> trend = await detector.get_emotional_trend("user_123", hours=24)
            >>> if trend.anxiety_episodes > 3:
            ...     print("User showing high anxiety")
        """
        # Fetch emotional memories from last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_timestamp_ms = int(cutoff_time.timestamp() * 1000)

        try:
            # Scan for emotion keys matching pattern
            pattern = f"emotion:{user_id}:*"
            cursor = 0
            keys = []

            while True:
                cursor, batch = await self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                keys.extend(batch)
                if cursor == 0:
                    break

            # Filter keys by timestamp (only last N hours)
            # Key format: emotion:{user_id}:{timestamp_ms}
            recent_keys = [
                key for key in keys
                if int(key.split(":")[-1]) >= cutoff_timestamp_ms
            ]

            # Fetch memories
            memories: list[EmotionalMemory] = []
            for key in recent_keys:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        memory = EmotionalMemory.model_validate_json(data)
                        memories.append(memory)
                except Exception as e:
                    logger.warning(f"Failed to parse emotional memory from {key}: {e}")

            if not memories:
                # No data available, return neutral trend
                logger.debug(f"No emotional data for user={user_id} in last {hours}h")
                return EmotionalTrend(
                    user_id=user_id,
                    time_period_hours=hours,
                    dominant_emotions=["neutral"],
                    avg_sentiment=0.0,
                    emotional_volatility=0.0,
                    anxiety_episodes=0,
                    frustration_episodes=0,
                )

            # Sort by timestamp
            memories.sort(key=lambda m: m.timestamp)

            # Calculate metrics
            dominant_emotions = [m.dominant_emotion for m in memories]
            sentiments = [m.sentiment_score for m in memories]

            avg_sentiment = sum(sentiments) / len(sentiments)
            emotional_volatility = self._calculate_volatility(sentiments)

            anxiety_episodes = sum(
                1 for emotion in dominant_emotions if emotion == "anxious"
            )
            frustration_episodes = sum(
                1 for emotion in dominant_emotions if emotion == "frustrated"
            )

            logger.debug(
                f"✅ Emotional trend: {len(memories)} memories, "
                f"avg_sentiment={avg_sentiment:.2f}, anxiety={anxiety_episodes}"
            )

            return EmotionalTrend(
                user_id=user_id,
                time_period_hours=hours,
                dominant_emotions=dominant_emotions,
                avg_sentiment=avg_sentiment,
                emotional_volatility=emotional_volatility,
                anxiety_episodes=anxiety_episodes,
                frustration_episodes=frustration_episodes,
            )

        except Exception as e:
            logger.error(f"Failed to calculate emotional trend for {user_id}: {e}")
            # Return neutral trend on error
            return EmotionalTrend(
                user_id=user_id,
                time_period_hours=hours,
                dominant_emotions=["neutral"],
                avg_sentiment=0.0,
                emotional_volatility=0.0,
                anxiety_episodes=0,
                frustration_episodes=0,
            )

    def _calculate_volatility(self, sentiments: list[float]) -> float:
        """Calculate emotional volatility (standard deviation of sentiment).

        Args:
            sentiments: List of sentiment scores

        Returns:
            Volatility score (0.0 = stable, 1.0 = very volatile)
        """
        if len(sentiments) < 2:
            return 0.0

        # Calculate standard deviation
        mean = sum(sentiments) / len(sentiments)
        variance = sum((x - mean) ** 2 for x in sentiments) / len(sentiments)
        std_dev = variance**0.5

        # Normalize to 0-1 scale (sentiment range is -1 to +1, so max std_dev ≈ 1)
        volatility = min(1.0, std_dev)

        return volatility

    # ========================================================================
    # Emotional Context for Agent
    # ========================================================================

    async def get_emotional_context(self, user_id: str) -> str:
        """Get emotional context string for agent prompt.

        This provides the agent with recent emotional state to enable
        empathetic responses.

        Args:
            user_id: User identifier

        Returns:
            Emotional context string for agent prompt

        Example:
            >>> context = await detector.get_emotional_context("user_123")
            >>> print(context)
            "User emotional state: anxious (last 3 queries). Show extra empathy and provide reassurance."
        """
        trend = await self.get_emotional_trend(user_id, hours=1)

        if not trend.dominant_emotions or trend.dominant_emotions == ["neutral"]:
            return "User emotional state: neutral. Use standard tone."

        # Get most recent emotion
        recent_emotion = trend.dominant_emotions[-1]

        # Count consecutive occurrences
        consecutive_count = 1
        for emotion in reversed(trend.dominant_emotions[:-1]):
            if emotion == recent_emotion:
                consecutive_count += 1
            else:
                break

        # Generate context message
        emotion_guidance = {
            "anxious": "Show extra empathy and provide clear, reassuring information. Avoid alarming language.",
            "frustrated": "Be patient and provide clearer explanations. Simplify complex information.",
            "excited": "Match enthusiasm while maintaining accuracy. Encourage further exploration.",
            "curious": "Provide detailed explanations and encourage follow-up questions.",
            "neutral": "Use standard professional tone.",
        }

        guidance = emotion_guidance.get(recent_emotion, "Use standard tone.")

        context = f"User emotional state: {recent_emotion}"
        if consecutive_count > 1:
            context += f" (last {consecutive_count} queries)"
        context += f". {guidance}"

        return context


# ============================================================================
# Helper Functions
# ============================================================================


async def detect_emotion_from_query(
    user_id: str, query: str, session_id: str | None = None
) -> EmotionalMemory:
    """Convenience function to detect emotion from a query.

    Args:
        user_id: User identifier
        query: User query text
        session_id: Optional session identifier

    Returns:
        EmotionalMemory object

    Example:
        >>> emotion = await detect_emotion_from_query("user_123", "Is it safe to travel?")
        >>> print(emotion.dominant_emotion)
        'anxious'
    """
    detector = EmotionalMemoryDetector()
    return await detector.detect_emotion(user_id, query, session_id)
