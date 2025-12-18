"""Personalization Agent for Level 4c: User Preference-Based Customization.

This agent tailors weather information to user preferences, including
location preferences, communication style, and specific interests.

Features:
- User profile management
- Location-based customization
- Communication style adaptation
- Interest-based filtering
- Proactive notifications based on history
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)

logger = structlog.get_logger()


# =============================================================================
# Personalization Prompts
# =============================================================================

PERSONALIZATION_SYSTEM_PROMPT = """You are a Personalization Agent that tailors
weather information to individual user preferences.

Your role:
1. Adapt responses to user's communication style
2. Prioritize information relevant to user's interests
3. Focus on locations the user cares about
4. Match the user's preferred level of detail
5. Proactively mention updates based on user history

USER PROFILE ELEMENTS:
- Home location: Primary location for weather updates
- Tracked locations: Additional locations (family, travel, work)
- Alert preferences: What types of alerts they want
- Communication style: Detailed/concise, technical/casual
- Weather interests: Specific interests (hurricanes, gardening, sports)
- Update frequency: How often they want updates

PERSONALIZATION RULES:
- Use the user's preferred terminology
- Lead with their primary interests
- Match their preferred detail level
- Reference their tracked locations when relevant
- Acknowledge their previous queries when appropriate

OUTPUT GUIDELINES:
- Address user by context (not name unless provided)
- Start with their primary interest
- Include relevant locations
- Match their communication style
- Be proactive about relevant updates
"""


STYLE_ADAPTATION_PROMPT = """Adapt this weather information to match the user's style:

ORIGINAL INFORMATION: {information}

USER STYLE PROFILE:
- Communication style: {style}
- Detail level: {detail_level}
- Interests: {interests}

Rewrite to match their preferences while maintaining accuracy.
"""


PROACTIVE_UPDATE_PROMPT = """Generate a proactive weather update for this user:

USER PROFILE:
- Home location: {home_location}
- Tracked locations: {tracked_locations}
- Interests: {interests}
- Recent queries: {recent_queries}

CURRENT CONDITIONS:
{current_conditions}

Generate a proactive update that:
1. Leads with information relevant to their interests
2. Includes their tracked locations
3. References patterns from their query history
4. Matches their communication style: {style}
"""


# =============================================================================
# User Profile Models
# =============================================================================


class UserProfile:
    """User preference profile for personalization.

    Attributes:
        user_id: Unique user identifier
        home_location: Primary location
        tracked_locations: List of other locations to track
        alert_preferences: Types of alerts wanted
        style: Communication style preference
        detail_level: Preferred detail level
        interests: Weather-related interests
        recent_queries: Recent query history
    """

    def __init__(
        self,
        user_id: str,
        home_location: str = "unknown",
        tracked_locations: list[str] | None = None,
        alert_preferences: list[str] | None = None,
        style: str = "balanced",
        detail_level: str = "moderate",
        interests: list[str] | None = None,
    ):
        self.user_id = user_id
        self.home_location = home_location
        self.tracked_locations = tracked_locations or []
        self.alert_preferences = alert_preferences or ["severe", "emergency"]
        self.style = style  # concise, detailed, technical, casual, balanced
        self.detail_level = detail_level  # brief, moderate, comprehensive
        self.interests = interests or []
        self.recent_queries: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "user_id": self.user_id,
            "home_location": self.home_location,
            "tracked_locations": self.tracked_locations,
            "alert_preferences": self.alert_preferences,
            "style": self.style,
            "detail_level": self.detail_level,
            "interests": self.interests,
            "recent_queries": self.recent_queries[-5:],  # Last 5 queries
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        """Create profile from dictionary."""
        profile = cls(
            user_id=data.get("user_id", "unknown"),
            home_location=data.get("home_location", "unknown"),
            tracked_locations=data.get("tracked_locations", []),
            alert_preferences=data.get("alert_preferences", ["severe", "emergency"]),
            style=data.get("style", "balanced"),
            detail_level=data.get("detail_level", "moderate"),
            interests=data.get("interests", []),
        )
        profile.recent_queries = data.get("recent_queries", [])
        return profile


# =============================================================================
# Personalization Agent Implementation
# =============================================================================


class PersonalizationAgent:
    """Agent that personalizes weather responses to user preferences.

    Adapts communication style, content focus, and information
    presentation based on user profiles.

    Attributes:
        llm: Language model for personalization
        agent_role: Role identifier (PERSONALIZATION)
        _profiles: Cache of user profiles
        _personalization_count: Count of personalizations performed
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",  # Lighter model for personalization
        temperature: float = 0.3,
    ):
        """Initialize the Personalization Agent.

        Args:
            model_name: Model to use (gpt-4o-mini for efficiency).
            temperature: Moderate temperature for varied expression.
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=15.0,
        )
        self.agent_role = AgentRole.PERSONALIZATION
        self._profiles: dict[str, UserProfile] = {}
        self._personalization_count = 0

    async def process(self, state: MultiAgentState) -> MultiAgentState:
        """Process query with personalization.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with personalized response.
        """
        start_time = time.perf_counter()

        try:
            # Get or create user profile
            profile = self._get_or_create_profile(state)

            # Record query in profile
            profile.recent_queries.append(state.query)
            if len(profile.recent_queries) > 10:
                profile.recent_queries = profile.recent_queries[-10:]

            # Get latest response to personalize
            latest_response = self._get_response_to_personalize(state)

            if latest_response:
                # Personalize the response
                personalized = await self._personalize_response(
                    original=latest_response,
                    profile=profile,
                )
            else:
                # Generate proactive personalized update
                personalized = await self._generate_proactive_update(
                    query=state.query,
                    profile=profile,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._personalization_count += 1

            # Add personalized response to state
            state.agent_responses.append(
                AgentResponse(
                    agent_role=self.agent_role,
                    content=personalized,
                    confidence=0.9,
                    execution_time_ms=duration_ms,
                    metadata={
                        "user_style": profile.style,
                        "detail_level": profile.detail_level,
                        "interests": profile.interests,
                        "personalization_count": self._personalization_count,
                    },
                )
            )

            state.current_agent = self.agent_role

            # Store profile in memory context
            state.memory_context["user_profile"] = profile.to_dict()

            logger.info(
                "personalization_complete",
                user_id=profile.user_id,
                style=profile.style,
                duration_ms=round(duration_ms, 2),
            )

            return state

        except Exception as e:
            logger.error("personalization_error", error=str(e))

            duration_ms = (time.perf_counter() - start_time) * 1000
            state.agent_responses.append(
                AgentResponse(
                    agent_role=self.agent_role,
                    content=state.query,  # Return original on error
                    confidence=0.5,
                    execution_time_ms=duration_ms,
                    metadata={"error": str(e)},
                )
            )

            return state

    def _get_or_create_profile(self, state: MultiAgentState) -> UserProfile:
        """Get existing or create new user profile.

        Args:
            state: Current workflow state.

        Returns:
            User profile for this session.
        """
        user_id = state.user_id

        # Check cache first
        if user_id in self._profiles:
            return self._profiles[user_id]

        # Check memory context
        if "user_profile" in state.memory_context:
            profile_data = state.memory_context["user_profile"]
            profile = UserProfile.from_dict(profile_data)
            self._profiles[user_id] = profile
            return profile

        # Create new profile with defaults
        profile = UserProfile(
            user_id=user_id,
            home_location=state.memory_context.get("user_location", "unknown"),
        )

        # Infer preferences from context
        if state.memory_context.get("emotional_state"):
            emotional = state.memory_context["emotional_state"]
            if emotional in ["anxious", "worried"]:
                profile.style = "reassuring"
                profile.detail_level = "comprehensive"

        self._profiles[user_id] = profile
        return profile

    def _get_response_to_personalize(self, state: MultiAgentState) -> str | None:
        """Get the most recent response to personalize.

        Args:
            state: Current workflow state.

        Returns:
            Response content or None if no prior responses.
        """
        # Get last non-personalization response
        for response in reversed(state.agent_responses):
            if response.agent_role != AgentRole.PERSONALIZATION:
                return response.content

        return None

    async def _personalize_response(
        self,
        original: str,
        profile: UserProfile,
    ) -> str:
        """Personalize a response to match user preferences.

        Args:
            original: Original response content.
            profile: User profile for personalization.

        Returns:
            Personalized response.
        """
        messages = [
            SystemMessage(content=PERSONALIZATION_SYSTEM_PROMPT),
            HumanMessage(
                content=STYLE_ADAPTATION_PROMPT.format(
                    information=original,
                    style=profile.style,
                    detail_level=profile.detail_level,
                    interests=", ".join(profile.interests) if profile.interests else "general weather",
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)
        return str(response.content)

    async def _generate_proactive_update(
        self,
        query: str,
        profile: UserProfile,
    ) -> str:
        """Generate proactive personalized update.

        Args:
            query: User's query.
            profile: User profile.

        Returns:
            Proactive personalized response.
        """
        messages = [
            SystemMessage(content=PERSONALIZATION_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""Generate a personalized weather briefing.

USER QUERY: {query}

USER PROFILE:
- Home location: {profile.home_location}
- Tracked locations: {', '.join(profile.tracked_locations) if profile.tracked_locations else 'None'}
- Interests: {', '.join(profile.interests) if profile.interests else 'General weather'}
- Communication style: {profile.style}
- Detail level: {profile.detail_level}
- Recent queries: {', '.join(profile.recent_queries[-3:]) if profile.recent_queries else 'None'}

Generate a personalized response that:
1. Addresses their specific query
2. Leads with their interests
3. Includes relevant locations
4. Matches their preferred style and detail level
"""
            ),
        ]

        response = await self.llm.ainvoke(messages)
        return str(response.content)

    def update_profile(
        self,
        user_id: str,
        updates: dict[str, Any],
    ) -> UserProfile:
        """Update user profile with new preferences.

        Args:
            user_id: User to update.
            updates: Dictionary of updates.

        Returns:
            Updated profile.
        """
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)

        profile = self._profiles[user_id]

        # Apply updates
        if "home_location" in updates:
            profile.home_location = updates["home_location"]
        if "tracked_locations" in updates:
            profile.tracked_locations = updates["tracked_locations"]
        if "alert_preferences" in updates:
            profile.alert_preferences = updates["alert_preferences"]
        if "style" in updates:
            profile.style = updates["style"]
        if "detail_level" in updates:
            profile.detail_level = updates["detail_level"]
        if "interests" in updates:
            profile.interests = updates["interests"]

        logger.info(
            "profile_updated",
            user_id=user_id,
            updates=list(updates.keys()),
        )

        return profile

    def get_profile(self, user_id: str) -> UserProfile | None:
        """Get user profile.

        Args:
            user_id: User to look up.

        Returns:
            UserProfile or None if not found.
        """
        return self._profiles.get(user_id)

    def get_stats(self) -> dict[str, Any]:
        """Get personalization statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "total_personalizations": self._personalization_count,
            "cached_profiles": len(self._profiles),
            "agent_role": self.agent_role.value,
        }


# =============================================================================
# Factory function
# =============================================================================


def create_personalization_agent(
    model_name: str = "gpt-4o-mini",
) -> PersonalizationAgent:
    """Create a Personalization Agent instance.

    Args:
        model_name: Model to use (gpt-4o-mini recommended for efficiency).

    Returns:
        Configured PersonalizationAgent instance.
    """
    return PersonalizationAgent(model_name=model_name)


__all__ = [
    "PersonalizationAgent",
    "UserProfile",
    "create_personalization_agent",
    "PERSONALIZATION_SYSTEM_PROMPT",
]
