"""Short-term memory implementation using Redis.

Provides session-based memory with automatic expiration (TTL).
Stores:
- Conversation history (last 10 turns)
- Entity tracking (location, date, hurricane names)
- Pronoun resolution (there → London)

Architecture:
- Redis key pattern: stm:{user_id}:{session_id}
- TTL: 30 minutes (configurable via REDIS_TTL_SECONDS)
- Async Redis client (redis.asyncio)

Example:
    >>> memory = ShortTermMemory()
    >>> await memory.save_context(context)
    >>> context = await memory.get_context(user_id, session_id)
    >>> resolved = await memory.resolve_pronoun(user_id, session_id, "Weather there?")
    # Result: "Weather in London?" (if London was mentioned)
"""

import json
from datetime import datetime, timedelta

import redis.asyncio as redis

from backend.config.memory_config import memory_config
from backend.src.memory.exceptions import RedisMemoryError
from backend.src.models.memory import ConversationContext, SessionState


class ShortTermMemory:
    """Redis-based short-term memory for session context.

    Manages conversation state with automatic expiration.
    Uses Redis for fast, ephemeral storage (30min TTL).
    """

    def __init__(self) -> None:
        """Initialize Redis connection."""
        self.redis: redis.Redis = redis.from_url(
            memory_config.REDIS_URL,
            decode_responses=True,
            max_connections=memory_config.REDIS_MAX_CONNECTIONS,
        )
        self.ttl = memory_config.REDIS_TTL_SECONDS

    async def __aenter__(self) -> "ShortTermMemory":
        """Context manager entry.

        Returns:
            Self for use in async with statement

        Example:
            >>> async with ShortTermMemory() as memory:
            ...     await memory.save_context(context)
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit. Ensures Redis connection is closed.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        await self.close()

    async def save_context(self, context: ConversationContext) -> None:
        """Save conversation context to Redis with TTL.

        Args:
            context: ConversationContext to save

        Raises:
            RedisMemoryError: If Redis operation fails

        Example:
            >>> context = ConversationContext(
            ...     user_id="user_123",
            ...     session_id="session_abc",
            ...     conversation_history=[
            ...         {"role": "user", "content": "Weather in London?"}
            ...     ]
            ... )
            >>> await memory.save_context(context)
        """
        key = f"stm:{context.user_id}:{context.session_id}"
        value = context.model_dump_json()

        try:
            await self.redis.setex(
                name=key,
                time=self.ttl,
                value=value,
            )
        except redis.RedisError as e:
            raise RedisMemoryError(f"Failed to save context for {key}: {e}") from e

    async def get_context(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationContext | None:
        """Retrieve conversation context from Redis.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            ConversationContext if found, None otherwise

        Raises:
            RedisMemoryError: If Redis operation fails

        Example:
            >>> context = await memory.get_context("user_123", "session_abc")
            >>> if context:
            ...     print(context.conversation_history)
        """
        key = f"stm:{user_id}:{session_id}"

        try:
            data = await self.redis.get(key)
        except redis.RedisError as e:
            raise RedisMemoryError(f"Failed to get context for {key}: {e}") from e

        if not data:
            return None

        return ConversationContext.model_validate_json(data)

    async def create_context(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationContext:
        """Create new conversation context with TTL.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            New ConversationContext

        Example:
            >>> context = await memory.create_context("user_123", "session_abc")
        """
        expires_at = datetime.now() + timedelta(seconds=self.ttl)

        context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            conversation_history=[],
            current_entities={},
            token_count=0,
            created_at=datetime.now(),
            expires_at=expires_at,
        )

        await self.save_context(context)
        return context

    async def track_entity(
        self,
        user_id: str,
        session_id: str,
        entity_type: str,
        entity_value: str,
    ) -> None:
        """Track entities mentioned in conversation.

        Args:
            user_id: User identifier
            session_id: Session identifier
            entity_type: Type of entity (location, date, hurricane)
            entity_value: Value of entity (London, 2025-01-21, Milton)

        Raises:
            RedisMemoryError: If Redis operation fails

        Example:
            >>> await memory.track_entity(
            ...     "user_123", "session_abc", "location", "London"
            ... )
        """
        key = f"stm:{user_id}:{session_id}:entities:{entity_type}"

        try:
            await self.redis.setex(
                name=key,
                time=self.ttl,
                value=entity_value,
            )
        except redis.RedisError as e:
            raise RedisMemoryError(f"Failed to track entity {entity_type} for {key}: {e}") from e

    async def get_entity(
        self,
        user_id: str,
        session_id: str,
        entity_type: str,
    ) -> str | None:
        """Retrieve tracked entity.

        Args:
            user_id: User identifier
            session_id: Session identifier
            entity_type: Type of entity (location, date, hurricane)

        Returns:
            Entity value if found, None otherwise

        Raises:
            RedisMemoryError: If Redis operation fails

        Example:
            >>> location = await memory.get_entity(
            ...     "user_123", "session_abc", "location"
            ... )
            >>> print(location)  # "London"
        """
        key = f"stm:{user_id}:{session_id}:entities:{entity_type}"

        try:
            return await self.redis.get(key)
        except redis.RedisError as e:
            raise RedisMemoryError(f"Failed to get entity {entity_type} for {key}: {e}") from e

    async def resolve_pronoun(
        self,
        user_id: str,
        session_id: str,
        query: str,
    ) -> str:
        """Resolve pronouns using entity tracking.

        Replaces location pronouns with actual locations from session context.

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Original query with pronouns

        Returns:
            Query with pronouns resolved to entities

        Raises:
            RedisMemoryError: If Redis operation fails

        Example:
            >>> # User previously asked: "Weather in London?"
            >>> resolved = await memory.resolve_pronoun(
            ...     "user_123", "session_abc", "How about there tomorrow?"
            ... )
            >>> print(resolved)  # "How about London tomorrow?"
        """
        # Get last mentioned location
        location_key = f"stm:{user_id}:{session_id}:entities:location"

        try:
            location = await self.redis.get(location_key)
        except redis.RedisError as e:
            raise RedisMemoryError(f"Failed to resolve pronoun for {location_key}: {e}") from e

        if not location:
            return query

        # Replace pronouns with actual location
        pronouns = [
            "there",
            "that place",
            "that city",
            "it",
            "the same place",
            "same location",
        ]
        resolved = query

        for pronoun in pronouns:
            if pronoun in query.lower():
                # Case-insensitive replacement
                resolved = resolved.replace(pronoun, location)
                resolved = resolved.replace(pronoun.title(), location)
                resolved = resolved.replace(pronoun.upper(), location.upper())

        return resolved

    async def save_session_state(self, state: SessionState) -> None:
        """Save session metadata.

        Args:
            state: SessionState to save

        Raises:
            RedisMemoryError: If Redis operation fails

        Example:
            >>> state = SessionState(
            ...     session_id="session_abc",
            ...     user_id="user_123",
            ...     query_count=5
            ... )
            >>> await memory.save_session_state(state)
        """
        key = f"stm:session:{state.session_id}"
        value = state.model_dump_json()

        try:
            await self.redis.setex(
                name=key,
                time=self.ttl,
                value=value,
            )
        except redis.RedisError as e:
            raise RedisMemoryError(f"Failed to save session state for {key}: {e}") from e

    async def get_session_state(self, session_id: str) -> SessionState | None:
        """Retrieve session metadata.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found, None otherwise

        Raises:
            RedisMemoryError: If Redis operation fails

        Example:
            >>> state = await memory.get_session_state("session_abc")
            >>> if state:
            ...     print(state.query_count)
        """
        key = f"stm:session:{session_id}"

        try:
            data = await self.redis.get(key)
        except redis.RedisError as e:
            raise RedisMemoryError(f"Failed to get session state for {key}: {e}") from e

        if not data:
            return None

        return SessionState.model_validate_json(data)

    async def close(self) -> None:
        """Close Redis connection.

        Raises:
            RedisMemoryError: If closing connection fails

        Example:
            >>> await memory.close()
        """
        try:
            await self.redis.close()
        except redis.RedisError as e:
            raise RedisMemoryError(f"Failed to close Redis connection: {e}") from e
