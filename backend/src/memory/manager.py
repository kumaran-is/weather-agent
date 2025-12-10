"""Unified memory manager interface.

Provides a single interface for both short-term (Redis) and long-term (Graphiti) memory.
Handles:
- Context retrieval (session + profile) with parallel multi-layer retrieval
- Interaction saving (conversation + profile updates)
- Entity tracking and pronoun resolution
- Query count updates

Architecture:
- Short-term: Redis (session context, 30min TTL)
- Long-term: Graphiti + Neo4j (user profiles, persistent)
- Unified interface: Single entry point for memory operations
- Parallel retrieval: Independent layers retrieved concurrently (2-3x faster)

Performance:
- Sequential retrieval: 15-30s (old approach)
- Parallel retrieval: 6-10s (new approach with asyncio.gather)

Example:
    >>> manager = MemoryManager()
    >>> context = await manager.get_context("user_123", "session_abc")
    >>> await manager.save_interaction(
    ...     user_id="user_123",
    ...     session_id="session_abc",
    ...     query="Weather in London?",
    ...     response="London is 18°C, sunny"
    ... )
"""

import asyncio  # ✅ P0 fix: Required for asyncio.wait_for() and asyncio.gather()
import logging
import re
from datetime import datetime

from backend.config.memory_config import memory_config
from backend.src.memory.long_term import LongTermMemory
from backend.src.memory.short_term import ShortTermMemory
from backend.src.models.memory import ConversationContext, UserProfile

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified interface for short-term and long-term memory.

    Orchestrates memory operations across both layers:
    - Short-term: Session context, entity tracking
    - Long-term: User profiles, preferences, temporal facts
    """

    def __init__(self) -> None:
        """Initialize memory manager with both memory layers."""
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    async def __aenter__(self) -> "MemoryManager":
        """Context manager entry.

        Returns:
            Self for use in async with statement

        Example:
            >>> async with MemoryManager() as manager:
            ...     context = await manager.get_context("user_123", "session_abc")
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit. Ensures all memory connections are closed.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        await self.close()

    def _extract_entities(self, text: str) -> dict[str, str]:
        """Extract entities from text using regex patterns.

        Extracts:
        - Locations: City names, place names (e.g., "London", "Tokyo", "Orlando")
        - Dates: Relative dates (e.g., "tomorrow", "next week", "two weeks")
        - Units: Temperature preferences (e.g., "Fahrenheit", "Celsius")

        Args:
            text: Text to extract entities from

        Returns:
            Dictionary of entity_type -> entity_value

        Example:
            >>> entities = self._extract_entities("Weather in London tomorrow?")
            >>> print(entities)
            # {"location": "London", "date": "tomorrow"}
        """
        entities: dict[str, str] = {}

        # Common city names and patterns
        # Format: "in/at/for CITY", "CITY weather", "trip to CITY"
        location_patterns = [
            r"(?:in|at|for|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:,\s*[A-Z][a-z]+)?)",  # "in London", "to New York"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:weather|forecast|temperature)",  # "London weather"
            r"(?:trip|vacation|visit)\s+(?:to|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # "trip to Orlando"
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                location = match.group(1).strip()
                # Filter out common false positives (including temperature units)
                if location and location.lower() not in ["weather", "forecast", "temperature", "fahrenheit", "celsius", "i", "you", "what", "how"]:
                    entities["location"] = location
                    break  # Use first match

        # Relative date patterns
        date_patterns = [
            r"\b(tomorrow|today|tonight)\b",
            r"\b(next\s+(?:week|month|weekend|year))\b",
            r"\b(this\s+(?:week|weekend|month|year|saturday|sunday|monday|tuesday|wednesday|thursday|friday))\b",
            r"\b(in\s+(?:two|three|four|five|six|seven)\s+(?:days|weeks|months))\b",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["date"] = match.group(1).lower()
                break  # Use first match

        # Temperature units
        if re.search(r"\bfahrenheit\b", text, re.IGNORECASE):
            entities["units"] = "fahrenheit"
        elif re.search(r"\bcelsius\b", text, re.IGNORECASE):
            entities["units"] = "celsius"

        return entities

    async def get_context(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, any]:
        """Get complete memory context with parallel multi-layer retrieval.

        Performance:
        - Sequential (old): 15-30s (each layer waits for previous)
        - Parallel (new): 6-10s (independent layers run concurrently)

        Architecture:
        - Step 1: Short-term context (sequential, required for session creation)
        - Step 2: Parallel long-term retrieval (profile + episodes run concurrently)
        - Future: Supports 7-layer architecture (semantic, procedural, emotional, reflective)

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Dictionary with session context, user profile, and episodes

        Example:
            >>> context = await manager.get_context("user_123", "session_abc")
            >>> print(context["session"].conversation_history)
            >>> print(context["profile"].home_location)
            >>> print(context["episodes"])  # Previous conversations
        """
        # ============================================================================
        # STEP 1: Short-term context (SEQUENTIAL - required for session creation)
        # ============================================================================
        session_context = await self.short_term.get_context(user_id, session_id)

        if not session_context:
            session_context = await self.short_term.create_context(user_id, session_id)

        # ============================================================================
        # STEP 2: Parallel long-term retrieval (CONCURRENT - independent operations)
        # ============================================================================

        # Check if parallel retrieval is enabled (can be disabled for debugging)
        if not memory_config.MEMORY_PARALLEL_RETRIEVAL:
            # Fallback to sequential retrieval (old behavior)
            logger.debug("Parallel retrieval disabled - using sequential retrieval")
            return await self._get_context_sequential(user_id, session_context)

        # Build list of parallel retrieval tasks
        parallel_tasks: list[tuple[str, any]] = []

        # Layer 2: Long-term user profile (Graphiti)
        if memory_config.ENABLE_USER_PROFILES:
            parallel_tasks.append((
                "profile",
                self._get_user_profile_with_timeout(
                    user_id,
                    timeout=memory_config.MEMORY_LAYER_TIMEOUT
                )
            ))

        # Layer 3: Episodic memory (Graphiti search)
        if memory_config.ENABLE_USER_PROFILES:  # Reuse same flag for now
            parallel_tasks.append((
                "episodes",
                self._search_episodes_with_timeout(
                    user_id,
                    timeout=memory_config.MEMORY_LAYER_TIMEOUT
                )
            ))

        # Future: Layer 4 - Semantic memory (RAG from Qdrant)
        # if memory_config.ENABLE_SEMANTIC_MEMORY:
        #     parallel_tasks.append((
        #         "semantic",
        #         self._search_semantic_with_timeout(user_id, timeout=5.0)
        #     ))

        # Future: Layer 5 - Procedural memory (PostgreSQL)
        # if memory_config.ENABLE_PROCEDURAL_MEMORY:
        #     parallel_tasks.append((
        #         "procedural",
        #         self._get_procedural_with_timeout(user_id, timeout=5.0)
        #     ))

        # Future: Layer 6 - Emotional memory (Redis sentiment)
        # if memory_config.ENABLE_EMOTIONAL_MEMORY:
        #     parallel_tasks.append((
        #         "emotional",
        #         self._get_emotional_with_timeout(user_id, session_id, timeout=5.0)
        #     ))

        # Future: Layer 7 - Reflective memory (PostgreSQL learnings)
        # if memory_config.ENABLE_REFLECTIVE_MEMORY:
        #     parallel_tasks.append((
        #         "reflective",
        #         self._get_reflective_with_timeout(user_id, timeout=5.0)
        #     ))

        # If no parallel tasks, return early
        if not parallel_tasks:
            logger.debug("No parallel tasks - memory disabled or no layers enabled")
            return {"session": session_context}

        # Execute all layers in parallel with graceful degradation
        task_names = [name for name, _ in parallel_tasks]
        task_coros = [task for _, task in parallel_tasks]

        logger.debug(
            f"Starting parallel retrieval for {len(parallel_tasks)} layers: {task_names}"
        )

        # ✅ P1 FIX: Run all layers concurrently (2-3x faster than sequential)
        results = await asyncio.gather(*task_coros, return_exceptions=True)

        # Build response dict with graceful degradation
        response = {"session": session_context}

        for i, (task_name, result) in enumerate(zip(task_names, results)):
            if isinstance(result, Exception):
                # Log error but continue with other layers (graceful degradation)
                logger.warning(
                    f"⚠️ Layer '{task_name}' retrieval failed: {result}. "
                    f"Continuing with partial memory context."
                )
                # Set safe default based on expected type
                if task_name == "episodes":
                    response[task_name] = []
                else:
                    response[task_name] = None
            else:
                response[task_name] = result
                logger.debug(f"✅ Layer '{task_name}' retrieved successfully")

        logger.debug(
            f"Parallel retrieval complete - {len([r for r in results if not isinstance(r, Exception)])}/{len(results)} layers succeeded"
        )

        return response

    async def _get_context_sequential(
        self,
        user_id: str,
        session_context: ConversationContext,
    ) -> dict[str, any]:
        """Fallback to sequential retrieval (old behavior).

        Used when MEMORY_PARALLEL_RETRIEVAL is disabled for debugging.

        Args:
            user_id: User identifier
            session_context: Already retrieved session context

        Returns:
            Dictionary with session context, profile, and episodes
        """
        # Get long-term profile (optional)
        user_profile: UserProfile | None = None
        if memory_config.ENABLE_USER_PROFILES:
            user_profile = await self.long_term.get_user_profile(user_id)

        # Get previous episodes
        previous_episodes: list[dict[str, any]] = []
        if memory_config.ENABLE_USER_PROFILES:
            previous_episodes = await self.long_term.search_episodes(
                user_id=user_id,
                query=None,
                limit=3,
            )

        return {
            "session": session_context,
            "profile": user_profile,
            "episodes": previous_episodes,
        }

    async def _get_user_profile_with_timeout(
        self,
        user_id: str,
        timeout: float = 5.0,
    ) -> UserProfile | None:
        """Get user profile with timeout protection.

        Prevents slow Graphiti queries from blocking entire memory retrieval.
        Uses graceful degradation: returns None on timeout/error.

        Args:
            user_id: User identifier
            timeout: Timeout in seconds (default: 5s)

        Returns:
            User profile or None if timeout/error

        Example:
            >>> profile = await self._get_user_profile_with_timeout("user_123", timeout=5.0)
            >>> if profile:
            ...     print(f"Home: {profile.home_location}")
        """
        try:
            logger.debug(f"Retrieving user profile for user={user_id} with {timeout}s timeout")

            profile = await asyncio.wait_for(
                self.long_term.get_user_profile(user_id),
                timeout=timeout,
            )

            logger.debug(f"✅ User profile retrieved for user={user_id}")
            return profile

        except asyncio.TimeoutError:
            logger.warning(
                f"⚠️ User profile retrieval timed out after {timeout}s for user={user_id}. "
                f"Continuing without profile data (graceful degradation)."
            )
            return None

        except Exception as e:
            logger.error(
                f"❌ Failed to get user profile for user={user_id}: {e}. "
                f"Continuing without profile data."
            )
            return None

    async def _search_episodes_with_timeout(
        self,
        user_id: str,
        timeout: float = 5.0,
    ) -> list[dict[str, any]]:
        """Search episodes with timeout protection.

        Prevents slow Graphiti searches from blocking entire memory retrieval.
        Uses graceful degradation: returns empty list on timeout/error.

        Args:
            user_id: User identifier
            timeout: Timeout in seconds (default: 5s)

        Returns:
            List of episodes or empty list if timeout/error

        Example:
            >>> episodes = await self._search_episodes_with_timeout("user_123", timeout=5.0)
            >>> if episodes:
            ...     print(f"Found {len(episodes)} previous conversations")
        """
        try:
            logger.debug(f"Searching episodes for user={user_id} with {timeout}s timeout")

            episodes = await asyncio.wait_for(
                self.long_term.search_episodes(
                    user_id=user_id,
                    query=None,  # Get recent episodes regardless of content
                    limit=3,  # Last 3 conversations
                ),
                timeout=timeout,
            )

            logger.debug(f"✅ Retrieved {len(episodes)} episodes for user={user_id}")
            return episodes

        except asyncio.TimeoutError:
            logger.warning(
                f"⚠️ Episodes search timed out after {timeout}s for user={user_id}. "
                f"Continuing without episode data (graceful degradation)."
            )
            return []

        except Exception as e:
            logger.error(
                f"❌ Failed to search episodes for user={user_id}: {e}. "
                f"Continuing without episode data."
            )
            return []

    async def save_interaction(
        self,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
    ) -> None:
        """Save interaction to both memory layers.

        Updates:
        - Short-term: Conversation history
        - Short-term: Automatically extract and track entities from query 🆕
        - Long-term: Query count

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: User query
            response: Agent response

        Example:
            >>> await manager.save_interaction(
            ...     user_id="user_123",
            ...     session_id="session_abc",
            ...     query="Weather in London?",
            ...     response="London is 18°C, sunny"
            ... )
            # Automatically tracks location="London"
        """
        # 🆕 CRITICAL FIX: Extract entities from query automatically
        if memory_config.ENABLE_ENTITY_TRACKING:
            extracted_entities = self._extract_entities(query)
            # Track each extracted entity
            for entity_type, entity_value in extracted_entities.items():
                await self.short_term.track_entity(
                    user_id,
                    session_id,
                    entity_type,
                    entity_value,
                )

        # Update short-term context
        context = await self.short_term.get_context(user_id, session_id)

        if context:
            # Add new conversation turn
            context.conversation_history.append(
                {"role": "user", "content": query}
            )
            context.conversation_history.append(
                {"role": "assistant", "content": response}
            )

            # 🆕 Update tracked entities in context
            if memory_config.ENABLE_ENTITY_TRACKING:
                extracted_entities = self._extract_entities(query)
                for entity_type, entity_value in extracted_entities.items():
                    context.current_entities[entity_type] = entity_value

            # Keep only last N turns (prevent context explosion)
            max_turns = memory_config.REDIS_MAX_CONVERSATION_TURNS
            if len(context.conversation_history) > max_turns * 2:
                context.conversation_history = context.conversation_history[
                    -max_turns * 2 :
                ]

            # Update token count (approximate)
            context.token_count += len(query.split()) + len(response.split())

            await self.short_term.save_context(context)
        else:
            # Create new context
            expires_at = datetime.now()
            context = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                conversation_history=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ],
                expires_at=expires_at,
            )
            await self.short_term.save_context(context)

        # 🆕 CRITICAL FIX: Auto-create/update user profile when preferences are detected
        if memory_config.ENABLE_USER_PROFILES and memory_config.ENABLE_ENTITY_TRACKING:
            extracted_entities = self._extract_entities(query)

            # Check if user is setting preferences (location or units)
            has_location = "location" in extracted_entities
            has_units = "units" in extracted_entities

            if has_location or has_units:
                # Get existing profile or create new one
                profile = await self.long_term.get_user_profile(user_id)

                # Build preferences dict
                preferences = {}
                if has_units:
                    preferences["units"] = extracted_entities["units"]

                # Determine if this is a "home location" or "trip location"
                # For "trip to X" or "planning X", treat as temporary preference
                location = extracted_entities.get("location")
                is_trip = any(keyword in query.lower() for keyword in ["trip", "vacation", "visit", "planning"])

                if profile:
                    # Update existing profile
                    if has_units and preferences["units"] != getattr(profile, "preferred_units", None):
                        await self.long_term.update_user_preference(
                            user_id,
                            "units",
                            preferences["units"]
                        )

                    # If user mentions a location (trip or otherwise), update profile location
                    if has_location and location:
                        # For trips, we still want to remember the destination for the session
                        # Store as home_location if not a trip, otherwise just track in session
                        if not is_trip:
                            await self.long_term.update_user_preference(
                                user_id,
                                "location",
                                location
                            )
                        # For trips, we've already tracked it in session context above
                else:
                    # Create new profile
                    # 🔧 CRITICAL FIX: Pass is_trip_destination flag to Graphiti
                    await self.long_term.create_user_profile(
                        user_id=user_id,
                        name=None,  # Will be extracted from episodes if available
                        location=location if has_location else None,
                        preferences=preferences if preferences else None,
                        is_trip_destination=is_trip,  # 🆕 Pass trip flag to Graphiti
                    )

        # 🆕 CRITICAL FIX FOR LEVEL 3C: Save episode to Graphiti for cross-session recall
        # This enables episodic memory (Layer 3) by storing conversational episodes
        # in the long-term knowledge graph for retrieval across sessions
        try:
            await self.long_term.save_episode(
                user_id=user_id,
                session_id=session_id,
                query=query,
                response=response,
            )
        except Exception as e:
            # Log but don't fail - episode saving is non-critical
            logger.warning(f"Failed to save episode for cross-session recall: {e}")

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
            entity_value: Value of entity

        Example:
            >>> await manager.track_entity(
            ...     "user_123", "session_abc", "location", "London"
            ... )
        """
        await self.short_term.track_entity(
            user_id,
            session_id,
            entity_type,
            entity_value,
        )

        # Also update session context
        context = await self.short_term.get_context(user_id, session_id)
        if context:
            context.current_entities[entity_type] = entity_value
            await self.short_term.save_context(context)

    async def resolve_pronoun(
        self,
        user_id: str,
        session_id: str,
        query: str,
    ) -> str:
        """Resolve pronouns using entity tracking.

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Query with potential pronouns

        Returns:
            Query with pronouns resolved

        Example:
            >>> # User asked: "Weather in London?"
            >>> resolved = await manager.resolve_pronoun(
            ...     "user_123", "session_abc", "How about there?"
            ... )
            >>> print(resolved)  # "How about London?"
        """
        if not memory_config.ENABLE_PRONOUN_RESOLUTION:
            return query

        return await self.short_term.resolve_pronoun(
            user_id,
            session_id,
            query,
        )

    async def create_user_profile(
        self,
        user_id: str,
        name: str | None = None,
        location: str | None = None,
        preferences: dict[str, any] | None = None,
    ) -> None:
        """Create user profile in long-term memory.

        Args:
            user_id: User identifier
            name: User's name
            location: User's home location
            preferences: User preferences

        Example:
            >>> await manager.create_user_profile(
            ...     user_id="user_123",
            ...     name="John Doe",
            ...     location="London",
            ...     preferences={"units": "celsius"}
            ... )
        """
        if not memory_config.ENABLE_USER_PROFILES:
            return

        await self.long_term.create_user_profile(
            user_id=user_id,
            name=name,
            location=location,
            preferences=preferences,
        )

    async def update_user_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: str,
    ) -> None:
        """Update user preference in long-term memory.

        Args:
            user_id: User identifier
            preference_key: Preference key
            preference_value: New preference value

        Example:
            >>> await manager.update_user_preference(
            ...     "user_123", "units", "fahrenheit"
            ... )
        """
        if not memory_config.ENABLE_USER_PROFILES:
            return

        await self.long_term.update_user_preference(
            user_id,
            preference_key,
            preference_value,
        )

    async def get_conversation_summary(
        self,
        user_id: str,
        session_id: str,
    ) -> str:
        """Get summary of current conversation.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Formatted conversation summary

        Example:
            >>> summary = await manager.get_conversation_summary(
            ...     "user_123", "session_abc"
            ... )
            >>> print(summary)
            # "User asked about London weather (18°C, sunny)."
        """
        context = await self.short_term.get_context(user_id, session_id)

        if not context or not context.conversation_history:
            return "No conversation history"

        # Create simple summary from last few turns
        last_turns = context.conversation_history[-4:]  # Last 2 Q&A pairs
        summary_parts: list[str] = []

        for turn in last_turns:
            role = turn["role"]
            content = turn["content"][:100]  # Truncate long messages

            if role == "user":
                summary_parts.append(f"User: {content}")
            else:
                summary_parts.append(f"Agent: {content}")

        return " | ".join(summary_parts)

    async def close(self) -> None:
        """Close all memory connections.

        Example:
            >>> await manager.close()
        """
        await self.short_term.close()
        await self.long_term.close()
