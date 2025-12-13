"""Long-term memory implementation using Graphiti + Neo4j.

Provides persistent user profiles and temporal knowledge graphs.
Stores:
- User profiles (name, location, preferences)
- Temporal facts (valid_from/valid_to)
- Weather events with time bounds
- User-location relationships

Architecture:
- Graphiti: Temporal knowledge graph management (episode-based ingestion)
- Neo4j: Graph database backend
- Temporal validity: Automatic extraction from episodes

Example:
    >>> memory = LongTermMemory()
    >>> await memory.create_user_profile(
    ...     user_id="user_123",
    ...     name="John Doe",
    ...     location="London",
    ...     preferences={"units": "celsius"}
    ... )
    >>> profile = await memory.get_user_profile("user_123")
"""

import asyncio
import logging
from datetime import datetime, timezone

from graphiti_core import Graphiti

from backend.config.memory_config import memory_config
from backend.src.memory.exceptions import GraphitiMemoryError
from backend.src.models.memory import TemporalFact, UserProfile

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Graphiti-based long-term memory for user profiles.

    Manages persistent user data with temporal validity.
    Uses Graphiti + Neo4j for graph-based storage.

    CRITICAL: Uses episode-based ingestion (official Graphiti pattern).
    Graphiti automatically extracts entities and relationships from natural language.
    """

    def __init__(self) -> None:
        """Initialize Graphiti connection."""
        self.graphiti = Graphiti(
            uri=memory_config.GRAPHITI_URL,
            user=memory_config.GRAPHITI_USER,
            password=memory_config.GRAPHITI_PASSWORD,
            # NOTE: Graphiti v0.24+ removed 'database' parameter
            # Database selection handled via URI: bolt://localhost:7687/neo4j
        )
        self._initialized = False

    async def __aenter__(self) -> "LongTermMemory":
        """Context manager entry.

        Returns:
            Self for use in async with statement

        Example:
            >>> async with LongTermMemory() as memory:
            ...     await memory.create_user_profile(...)
        """
        await self._ensure_initialized()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit. Ensures Graphiti connection is closed.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        await self.close()

    async def _ensure_initialized(self) -> None:
        """Ensure Graphiti indices and constraints are built.

        MUST be called before any operations.
        Creates required Neo4j indices and constraints for optimal performance.

        Raises:
            GraphitiMemoryError: If initialization fails
        """
        if not self._initialized:
            try:
                await self.graphiti.build_indices_and_constraints()
                self._initialized = True
            except Exception as e:
                raise GraphitiMemoryError(f"Failed to initialize Graphiti: {e}") from e

    async def create_user_profile(
        self,
        user_id: str,
        name: str | None = None,
        location: str | None = None,
        preferences: dict[str, any] | None = None,
        is_trip_destination: bool = False,
    ) -> None:
        """Store persistent user profile in Neo4j using episode-based ingestion.

        Args:
            user_id: Unique user identifier
            name: User's name
            location: User's home location or trip destination
            preferences: User preferences (units, detail level, etc.)
            is_trip_destination: If True, location is a trip destination, not home

        Example:
            >>> await memory.create_user_profile(
            ...     user_id="user_123",
            ...     name="John Doe",
            ...     location="London",
            ...     preferences={"units": "celsius", "detail": "moderate"}
            ... )
        """
        await self._ensure_initialized()

        if preferences is None:
            preferences = {}

        # 🔧 OPTIMIZED: Create ONE well-structured episode instead of multiple
        # This is faster (1 Graphiti call vs 3) and still prevents ambiguity
        # Following official Graphiti best practice: Clear, unambiguous natural language

        user_name = name or user_id
        group_id = f"user_profile_{user_id}"

        # Construct clear, structured episode body
        episode_parts = []

        # Part 1: Identity
        episode_parts.append(f"User {user_name} (ID: {user_id})")

        # Part 2: Location - EXPLICIT phrasing to distinguish home vs trip vs temperature units
        if location:
            if is_trip_destination:
                # 🔧 CRITICAL: Use "trip to" not "lives in" for trip destinations
                episode_parts.append(f"is planning a trip to the city of {location}")
            else:
                # 🔧 CRITICAL: Use "lives in the city of" to make it explicit it's a location, not a unit
                episode_parts.append(f"lives in the city of {location}")

        # Part 3: Temperature unit preference - EXPLICIT phrasing to avoid confusion with location
        if preferences and "units" in preferences:
            units = preferences["units"]
            # 🔧 CRITICAL: "temperature readings in X" makes it impossible to confuse with location
            episode_parts.append(f"and prefers weather temperature readings in {units}")

        # Part 4: Detail level preference
        if preferences and "detail" in preferences:
            detail = preferences["detail"]
            episode_parts.append(f"and prefers {detail} detail level for forecasts")

        # Combine with proper separators
        episode_body = " ".join(episode_parts) + "."

        logger.info(f"[GRAPHITI] Creating user profile episode: '{episode_body}'")

        try:
            await self.graphiti.add_episode(
                name=f"User Profile: {user_id}",
                episode_body=episode_body,
                source_description="User profile creation",
                reference_time=datetime.now(timezone.utc),
                group_id=group_id,
            )
            logger.info(f"[GRAPHITI] ✅ Successfully created profile episode for {user_id}")
        except Exception as e:
            logger.error(f"[GRAPHITI] ❌ Failed to create profile episode: {e}")
            raise GraphitiMemoryError(f"Failed to create user profile for {user_id}: {e}") from e

    async def get_user_profile(self, user_id: str) -> UserProfile | None:
        """Retrieve user profile from Neo4j.

        Args:
            user_id: User identifier

        Returns:
            UserProfile if found, None otherwise

        Example:
            >>> profile = await memory.get_user_profile("user_123")
            >>> if profile:
            ...     print(profile.name)
            ...     print(profile.home_location)
        """
        await self._ensure_initialized()

        # 🔧 CRITICAL FIX: Use user-specific group_id to prevent cross-user contamination
        user_group_id = f"user_profile_{user_id}"

        # 🔍 DEBUG: Log search query
        logger.info(f"[S3B DEBUG] Searching for user profile: user_id={user_id}")
        logger.info(f"[S3B DEBUG] Graphiti search query: 'user {user_id}'")
        logger.info(f"[S3B DEBUG] Graphiti group_ids: ['{user_group_id}']")

        # Search for user facts using Graphiti
        # search() returns List[EntityEdge] (facts/relationships)
        try:
            results = await self.graphiti.search(
                query=f"user {user_id}",
                group_ids=[user_group_id],  # 🆕 User-specific group ID
                num_results=10,
            )
        except Exception as e:
            logger.error(f"[S3B DEBUG] Graphiti search failed: {e}")
            raise GraphitiMemoryError(f"Failed to search user profile for {user_id}: {e}") from e

        # 🔍 DEBUG: Log search results
        logger.info(f"[S3B DEBUG] Graphiti search returned {len(results)} results")
        if results:
            for idx, edge in enumerate(results):
                logger.info(f"[S3B DEBUG] Result {idx+1}: fact='{edge.fact}', uuid={edge.uuid}")
        else:
            logger.warning(f"[S3B DEBUG] No results found for user_id={user_id}")

        if not results:
            return None

        # Extract user information from facts
        user_data: dict[str, any] = {
            "user_id": user_id,
            "name": None,
            "home_location": None,
            "preferred_units": "celsius",
            "preferred_detail_level": "moderate",
            "total_queries": 0,
        }

        # 🔍 DEBUG: Track extraction process
        logger.info(f"[S3B DEBUG] Starting profile extraction for {len(results)} facts")

        # 🆕 Track if we've found a high-priority location (to prevent overwrites)
        found_preferred_location = False

        for idx, edge in enumerate(results):
            fact_original = edge.fact  # Keep original for extraction
            fact = edge.fact.lower()  # Lowercase for pattern matching
            logger.info(f"[S3B DEBUG] Processing fact {idx+1}: '{fact}'")

            # 🔧 IMPROVED: Extract location with better patterns matching new episode format
            # New format: "lives in the city of London" or "planning a trip to the city of Paris"
            # PRIORITY ORDER: Recent preferences > Trip destinations > Home locations
            location_patterns = [
                ("now prefers", False),  # 🆕 HIGHEST PRIORITY - Recent preference
                ("prefers", False),  # 🆕 Recent preference (fallback)
                ("planning a trip to the city of", True),  # Trip destination
                ("planning a trip to", True),
                ("trip to the city of", True),
                ("vacation to", True),
                ("visit to", True),
                ("visiting", True),
                ("lives in the city of", False),  # Home location (new format)
                ("lives in", False),  # Home location (old format fallback)
                ("located in", False),
                ("based in", False),
            ]

            # 🆕 CRITICAL: Skip location extraction if we already found a high-priority one
            if not found_preferred_location:
                for pattern, is_trip in location_patterns:
                    if pattern in fact:
                        logger.info(f"[S3B DEBUG] Found pattern '{pattern}' in fact (is_trip={is_trip})")
                        # Find same position in original fact
                        pattern_pos = fact.find(pattern)
                        if pattern_pos >= 0:
                            # Extract text after pattern from original fact
                            after_pattern = fact_original[pattern_pos + len(pattern):].strip()
                            words = after_pattern.split()
                            logger.info(f"[S3B DEBUG] Words after pattern: {words}")
                            for word in words:
                                # Look for capitalized words (likely city names)
                                # Clean punctuation first
                                clean_word = word.rstrip(".,!?;:")
                                # Check if it's a valid location (capitalized, not a common word)
                                # Exclude words related to preferences/units/descriptors
                                excluded_words = ["and", "or", "the", "a", "location", "units", "user", "fahrenheit", "celsius", "temperature", "readings", "prefers", "in", "of", "city", "weather"]
                                if clean_word and clean_word[0].isupper() and clean_word.lower() not in excluded_words:
                                    extracted_location = clean_word
                                    user_data["home_location"] = extracted_location
                                    logger.info(f"[S3B DEBUG] ✅ Extracted location: {extracted_location} (is_trip={is_trip})")
                                    # 🆕 Mark as found if it's a high-priority pattern (prefers)
                                    if "prefers" in pattern:
                                        found_preferred_location = True
                                        logger.info(f"[S3B DEBUG] 🔒 Locked to preferred location (won't overwrite)")
                                    break
                        if user_data["home_location"]:
                            break  # Found location, stop looking at patterns for this fact

            # Extract name
            if "user" in fact and ("has id" in fact or "id:" in fact):
                if "has id" in fact:
                    # New format: "User John has ID user_123"
                    parts = fact.split("user")[1].split("has id")
                    name_part = parts[0].strip()
                    if name_part and name_part != user_id.lower():
                        user_data["name"] = name_part.title()  # Capitalize properly
                        logger.info(f"[S3B DEBUG] ✅ Extracted name: {name_part}")
                elif "id:" in fact:
                    # Old format: "User John (ID: user_123)"
                    parts = fact.split("user")
                    if len(parts) > 1:
                        name_part = parts[1].split("(")[0].strip()
                        if name_part and name_part != user_id:
                            user_data["name"] = name_part
                            logger.info(f"[S3B DEBUG] ✅ Extracted name: {name_part}")

            # 🔧 IMPROVED: Extract temperature unit preferences matching new format
            # New format: "prefers weather temperature readings in fahrenheit"
            if "prefers weather temperature readings in" in fact or "prefers temperature readings in" in fact:
                # New format: exact match
                if "celsius" in fact:
                    user_data["preferred_units"] = "celsius"
                    logger.info(f"[S3B DEBUG] ✅ Extracted units: celsius")
                elif "fahrenheit" in fact:
                    user_data["preferred_units"] = "fahrenheit"
                    logger.info(f"[S3B DEBUG] ✅ Extracted units: fahrenheit")
            elif "prefers" in fact and ("celsius" in fact or "fahrenheit" in fact):
                # Fallback for old formats
                # BUT: Make sure it's not "lives in fahrenheit" or "trip to fahrenheit" (nonsense)
                # Only extract if "temperature" or "units" or "readings" or "weather" is nearby
                if any(word in fact for word in ["temperature", "units", "readings", "weather"]):
                    unit = "celsius" if "celsius" in fact else "fahrenheit"
                    user_data["preferred_units"] = unit
                    logger.info(f"[S3B DEBUG] ✅ Extracted units (fallback): {unit}")

            # Extract detail level preferences
            if "prefers" in fact and "level of detail" in fact:
                # New format: "User X prefers moderate level of detail in weather forecasts."
                if "moderate" in fact:
                    user_data["preferred_detail_level"] = "moderate"
                    logger.info(f"[S3B DEBUG] ✅ Extracted detail level: moderate")
                elif "detailed" in fact:
                    user_data["preferred_detail_level"] = "detailed"
                    logger.info(f"[S3B DEBUG] ✅ Extracted detail level: detailed")
                elif "brief" in fact:
                    user_data["preferred_detail_level"] = "brief"
                    logger.info(f"[S3B DEBUG] ✅ Extracted detail level: brief")

        # 🔍 DEBUG: Log final extracted data
        logger.info(f"[S3B DEBUG] Final extracted data: {user_data}")

        # Only return profile if we found at least some data
        if user_data["name"] or user_data["home_location"]:
            logger.info(f"[S3B DEBUG] ✅ Returning profile: name={user_data['name']}, location={user_data['home_location']}")
            return UserProfile(**user_data)

        logger.warning(f"[S3B DEBUG] ❌ No profile data found, returning None")
        return None

    async def update_user_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: str,
    ) -> None:
        """Update user preference using episode-based ingestion.

        Args:
            user_id: User identifier
            preference_key: Preference key (units, detail_level, etc.)
            preference_value: New preference value

        Example:
            >>> await memory.update_user_preference(
            ...     "user_123", "units", "fahrenheit"
            ... )
        """
        await self._ensure_initialized()

        # Create natural language episode for preference update
        episode_body = f"User {user_id} changed their preference: now prefers {preference_value} {preference_key}."

        # 🔧 CRITICAL FIX: Use user-specific group_id to prevent cross-user contamination
        try:
            await self.graphiti.add_episode(
                name=f"Preference Update: {user_id}",
                episode_body=episode_body,
                source_description="User preference update",
                reference_time=datetime.now(timezone.utc),
                group_id=f"user_profile_{user_id}",  # 🆕 User-specific group ID
            )
        except Exception as e:
            raise GraphitiMemoryError(f"Failed to update user preference for {user_id}: {e}") from e

    async def track_weather_event(
        self,
        location: str,
        weather_type: str,
        conditions: dict[str, any],
        valid_from: datetime,
        valid_to: datetime | None = None,
    ) -> None:
        """Track weather conditions with temporal validity using episode-based ingestion.

        Args:
            location: Location identifier
            weather_type: Type of weather (rain, hurricane, snow, etc.)
            conditions: Weather condition details
            valid_from: Event start time
            valid_to: Event end time (None = ongoing)

        Example:
            >>> await memory.track_weather_event(
            ...     location="London",
            ...     weather_type="rain",
            ...     conditions={"temp": 18, "humidity": 85},
            ...     valid_from=datetime(2025, 1, 21, 14, 0),
            ...     valid_to=datetime(2025, 1, 21, 16, 0)
            ... )
        """
        await self._ensure_initialized()

        # Construct natural language episode
        temp = conditions.get("temp", "unknown")
        humidity = conditions.get("humidity", "unknown")

        time_desc = f"from {valid_from.isoformat()}"
        if valid_to:
            time_desc += f" to {valid_to.isoformat()}"
        else:
            time_desc += " (ongoing)"

        episode_body = (
            f"Weather event in {location}: {weather_type} "
            f"with temperature {temp}°C and humidity {humidity}% "
            f"{time_desc}."
        )

        try:
            await self.graphiti.add_episode(
                name=f"Weather: {location} - {weather_type}",
                episode_body=episode_body,
                source_description="Weather event tracking",
                reference_time=valid_from,
                group_id="weather_events",
            )
        except Exception as e:
            raise GraphitiMemoryError(f"Failed to track weather event for {location}: {e}") from e

    async def add_temporal_fact(self, fact: TemporalFact) -> None:
        """Add a temporal fact to the knowledge graph using episode-based ingestion.

        Args:
            fact: TemporalFact to store

        Example:
            >>> fact = TemporalFact(
            ...     fact_id="fact_123",
            ...     user_id="user_123",
            ...     fact_type="preference",
            ...     content="User prefers detailed hurricane forecasts",
            ...     confidence=0.95,
            ...     valid_from=datetime.now()
            ... )
            >>> await memory.add_temporal_fact(fact)
        """
        await self._ensure_initialized()

        # Construct natural language episode
        time_desc = f"from {fact.valid_from.isoformat()}"
        if fact.valid_to:
            time_desc += f" to {fact.valid_to.isoformat()}"

        episode_body = (
            f"User {fact.user_id}: {fact.content} "
            f"(type: {fact.fact_type}, confidence: {fact.confidence}, "
            f"valid {time_desc})."
        )

        try:
            await self.graphiti.add_episode(
                name=f"Fact: {fact.fact_id}",
                episode_body=episode_body,
                source_description=fact.source,
                reference_time=fact.valid_from,
                group_id=f"user_facts_{fact.user_id}",
            )
        except Exception as e:
            raise GraphitiMemoryError(f"Failed to add temporal fact {fact.fact_id}: {e}") from e

    async def get_temporal_facts(
        self,
        user_id: str,
        fact_type: str | None = None,
        valid_at: datetime | None = None,
    ) -> list[TemporalFact]:
        """Retrieve temporal facts for a user.

        Args:
            user_id: User identifier
            fact_type: Optional fact type filter
            valid_at: Optional time filter (only facts valid at this time)

        Returns:
            List of TemporalFacts

        Example:
            >>> facts = await memory.get_temporal_facts(
            ...     user_id="user_123",
            ...     fact_type="preference"
            ... )
            >>> for fact in facts:
            ...     print(fact.content)
        """
        await self._ensure_initialized()

        query = f"user {user_id}"
        if fact_type:
            query += f" {fact_type}"

        # Search for facts
        try:
            results = await self.graphiti.search(
                query=query,
                group_ids=[f"user_facts_{user_id}"],
                num_results=20,
            )
        except Exception as e:
            raise GraphitiMemoryError(f"Failed to get temporal facts for {user_id}: {e}") from e

        facts: list[TemporalFact] = []

        for edge in results:
            # Parse fact from edge.fact string
            # Format: "User {user_id}: {content} (type: {type}, confidence: {conf}, valid ...)"
            fact_text = edge.fact

            try:
                # Extract content (between ": " and " (type:")
                if ": " in fact_text and " (type:" in fact_text:
                    content = fact_text.split(": ", 1)[1].split(" (type:")[0]

                    # Extract fact_type
                    if "type: " in fact_text and ", confidence:" in fact_text:
                        extracted_type = fact_text.split("type: ")[1].split(",")[0].strip()
                    else:
                        extracted_type = "general"

                    # Extract confidence
                    if "confidence: " in fact_text:
                        conf_str = fact_text.split("confidence: ")[1].split(",")[0].strip()
                        confidence = float(conf_str)
                    else:
                        confidence = 1.0

                    # Use edge timestamps
                    valid_from = edge.valid_at if edge.valid_at else datetime.now(timezone.utc)
                    valid_to = edge.invalid_at if edge.invalid_at else None

                    # Check temporal validity if requested
                    if valid_at:
                        if valid_from and valid_at < valid_from:
                            continue
                        if valid_to and valid_at > valid_to:
                            continue

                    fact = TemporalFact(
                        fact_id=edge.uuid,
                        user_id=user_id,
                        fact_type=extracted_type,
                        content=content,
                        confidence=confidence,
                        source="conversation",
                        metadata={},
                        valid_from=valid_from,
                        valid_to=valid_to,
                        created_at=edge.created_at,
                    )

                    facts.append(fact)
            except (ValueError, IndexError, AttributeError):
                # Skip malformed facts
                continue

        return facts

    async def save_episode(
        self,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        timeout_seconds: float | None = None,
    ) -> None:
        """Save query-response episode to Graphiti for cross-session recall.

        This enables episodic memory (Layer 3) by storing conversational episodes
        in the long-term knowledge graph. Graphiti automatically extracts:
        - Entities mentioned (locations, dates, preferences)
        - User facts and relationships
        - Temporal validity

        ⚠️ CRITICAL FIX: Added timeout to prevent hangs during episode saving.
        If save takes longer than configured timeout, logs warning and continues (non-blocking).

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: User's query
            response: Agent's response
            timeout_seconds: Maximum time to wait for save (default: from GRAPHITI_SAVE_TIMEOUT env var, 10.0s)

        Raises:
            GraphitiMemoryError: If episode saving fails

        Example:
            >>> await memory.save_episode(
            ...     user_id="user_123",
            ...     session_id="session_abc",
            ...     query="Is it safe to drive to work in Miami?",
            ...     response="Current weather in Miami is 24°C..."
            ... )
        """
        await self._ensure_initialized()

        # Use configured timeout if not specified
        from backend.config.memory_config import memory_config
        effective_timeout = timeout_seconds if timeout_seconds is not None else memory_config.GRAPHITI_SAVE_TIMEOUT

        try:
            # Construct episode content from query-response pair
            episode_content = f"""User query: {query}

Agent response: {response}

Context: User '{user_id}' in session '{session_id}'"""

            # ✅ REAL FIX: Use group_id to isolate user data (prevents cross-user contamination)
            # This is the ROOT CAUSE fix - missing group_id caused all episodes to go into global graph!
            user_group_id = f"user_profile_{user_id}"
            logger.debug(f"Saving episode for user={user_id} in group={user_group_id} with {effective_timeout}s timeout")

            await asyncio.wait_for(
                self.graphiti.add_episode(
                    name=f"conversation_{user_id}_{session_id}_{datetime.now(timezone.utc).isoformat()}",
                    episode_body=episode_content,
                    reference_time=datetime.now(timezone.utc),
                    source_description=f"Weather AI conversation with user {user_id}",
                    group_id=user_group_id,  # ✅ CRITICAL FIX: Isolate user's episodes in dedicated group
                ),
                timeout=effective_timeout
            )

            logger.info(
                f"Saved episode to Graphiti for user={user_id}, session={session_id}"
            )

        except asyncio.TimeoutError:
            logger.warning(
                f"⚠️ Episode save timed out after {effective_timeout}s for user={user_id}. "
                f"Episode NOT saved. Continuing without blocking query. "
                f"Consider increasing GRAPHITI_SAVE_TIMEOUT if this occurs frequently."
            )
            # Don't raise - episode saving is non-critical, continue operation

        except Exception as e:
            logger.error(
                f"Failed to save episode for user={user_id}, session={session_id}: {e}"
            )
            # Don't raise - episode saving is non-critical, continue operation
            # raise GraphitiMemoryError(f"Failed to save episode: {e}") from e

    async def search_episodes(
        self,
        user_id: str,
        query: str | None = None,
        limit: int = 5,
        timeout_seconds: float = 5.0,
    ) -> list[dict[str, any]]:
        """Search for relevant previous episodes from Graphiti.

        Enables cross-session memory recall by retrieving previous conversations
        and decisions from the knowledge graph.

        ⚠️ CRITICAL FIX: Added timeout to prevent 50-minute hangs in Graphiti search.
        If search takes >5s, returns empty list gracefully instead of blocking.

        Args:
            user_id: User identifier
            query: Optional search query (if None, retrieves recent episodes)
            limit: Maximum number of episodes to return
            timeout_seconds: Maximum time to wait for search (default: 5.0s)

        Returns:
            List of episode dictionaries with content and metadata

        Example:
            >>> episodes = await memory.search_episodes(
            ...     user_id="user_123",
            ...     query="Miami weather decision",
            ...     limit=3
            ... )
        """
        await self._ensure_initialized()

        try:
            # Build search query
            if query:
                search_query = f"user {user_id} {query}"
            else:
                search_query = f"user {user_id} conversation"

            # ✅ REAL FIX: Use group_ids to scope search to specific user (prevents graph-wide search)
            # This is the ROOT CAUSE fix - missing group_ids caused searching entire graph!
            user_group_id = f"user_profile_{user_id}"
            logger.debug(f"Searching episodes for user={user_id} in group={user_group_id} with {timeout_seconds}s timeout")

            search_results = await asyncio.wait_for(
                self.graphiti.search(
                    query=search_query,
                    group_ids=[user_group_id],  # ✅ CRITICAL FIX: Scope search to user's data only
                    num_results=limit,
                ),
                timeout=timeout_seconds
            )

            # Format results as episode summaries
            episodes: list[dict[str, any]] = []
            for result in search_results:
                # Each result is a SearchResult with fact and score
                fact_text = result.fact if hasattr(result, 'fact') else str(result)
                episodes.append({
                    "content": fact_text,
                    "score": result.score if hasattr(result, 'score') else 0.0,
                })

            logger.info(
                f"Retrieved {len(episodes)} episodes for user={user_id} in "
                f"{timeout_seconds}s (success)"
            )
            return episodes

        except asyncio.TimeoutError:
            logger.warning(
                f"⚠️ Episode search timed out after {timeout_seconds}s for user={user_id}. "
                f"Returning empty list to prevent query blocking. "
                f"This is a known issue with Graphiti search deadlock."
            )
            # Gracefully degrade - return empty episodes instead of blocking forever
            return []

        except Exception as e:
            logger.error(
                f"Failed to search episodes for user={user_id}: {e}"
            )
            # Return empty list on error - don't break the query
            return []

    async def close(self) -> None:
        """Close Graphiti connection.

        Raises:
            GraphitiMemoryError: If closing connection fails

        Example:
            >>> await memory.close()
        """
        try:
            await self.graphiti.close()
        except Exception as e:
            raise GraphitiMemoryError(f"Failed to close Graphiti connection: {e}") from e
