"""Meta-Prompt Agent for Level 4c: Dynamic Prompt Generation.

This agent generates customized prompts for specialist agents based on
query context, user history, and performance optimization.

CRITICAL RULES:
1. Meta-prompting generates prompts for other agents, not responses
2. Track which generated prompts perform best (for optimization)
3. Include few-shot examples based on query type
4. Never expose internal prompt generation to users

Features:
- Context-aware prompt generation
- Few-shot example injection
- Performance tracking and optimization
- Agent-specific template customization
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.src.agents.prompts.meta_prompt_templates import (
    AGENT_PROMPT_TEMPLATES,
    FEW_SHOT_EXAMPLES,
    META_PROMPT_SYSTEM,
    PROMPT_GENERATION_TEMPLATE,
)
from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)

logger = structlog.get_logger()


class MetaPromptAgent:
    """Generate dynamic prompts for specialist agents.

    The Meta-Prompt Agent analyzes incoming queries and user context to
    generate optimized prompts for specialist agents. This enables:
    - Context-aware agent behavior
    - Few-shot learning from successful examples
    - Continuous prompt optimization based on performance

    Attributes:
        llm: Language model for prompt generation
        agent_role: Role identifier (META_PROMPT)
        _prompt_performance: Performance tracking for generated prompts
        _generation_count: Total number of prompts generated
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        timeout: float = 15.0,
    ):
        """Initialize the Meta-Prompt Agent.

        Args:
            model_name: Model to use for prompt generation.
                Default gpt-4o for quality prompt generation.
            temperature: Sampling temperature (0.3 allows some creativity
                while maintaining coherence).
            timeout: Request timeout in seconds.
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
        )
        self.agent_role = AgentRole.META_PROMPT

        # Performance tracking
        self._prompt_performance: dict[str, list[float]] = {}
        self._generation_count: int = 0

    async def generate_agent_prompt(
        self,
        target_agent: AgentRole,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate customized prompt for target agent.

        Analyzes the query and context to produce an optimized prompt
        that guides the target agent's response generation.

        Args:
            target_agent: Which agent this prompt is for.
            query: User's original query.
            context: Additional context including:
                - user_location: User's location
                - emotional_state: Detected emotional state
                - recent_queries: Recent query history
                - preferences: User preferences

        Returns:
            Generated prompt string for the target agent.
        """
        start_time = time.perf_counter()

        # Get few-shot examples for this agent type
        examples = self._get_relevant_examples(target_agent, query)

        # Build the generation request
        generation_messages = [
            SystemMessage(content=META_PROMPT_SYSTEM),
            HumanMessage(
                content=PROMPT_GENERATION_TEMPLATE.format(
                    target_agent=target_agent.value,
                    query=query,
                    context=self._format_context(context),
                    examples=examples,
                )
            ),
        ]

        # Generate the prompt
        response = await self.llm.ainvoke(generation_messages)
        generated_prompt = str(response.content)

        duration_ms = (time.perf_counter() - start_time) * 1000
        self._generation_count += 1

        logger.info(
            "meta_prompt_generated",
            target_agent=target_agent.value,
            query_preview=query[:50] if len(query) > 50 else query,
            prompt_length=len(generated_prompt),
            duration_ms=round(duration_ms, 2),
            generation_count=self._generation_count,
        )

        return generated_prompt

    async def process(self, state: MultiAgentState) -> MultiAgentState:
        """Process state and generate prompts for next agents.

        Analyzes the workflow plan and generates customized prompts
        for each agent that will be executed.

        Args:
            state: Current multi-agent workflow state.

        Returns:
            Updated state with generated prompts in metadata.
        """
        start_time = time.perf_counter()

        try:
            # Build context from state
            context = self._build_context_from_state(state)

            # Generate prompts for planned agents
            generated_prompts: dict[str, str] = {}

            for agent_role in state.workflow_plan:
                if agent_role == AgentRole.META_PROMPT:
                    continue  # Don't generate prompt for self

                prompt = await self.generate_agent_prompt(
                    target_agent=agent_role,
                    query=state.query,
                    context=context,
                )
                generated_prompts[agent_role.value] = prompt

            # Store generated prompts in state metadata
            state.memory_context["generated_prompts"] = generated_prompts

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Add response to state
            state.agent_responses.append(
                AgentResponse(
                    agent_role=self.agent_role,
                    content=f"Generated {len(generated_prompts)} customized prompts",
                    confidence=0.9,
                    execution_time_ms=duration_ms,
                    metadata={
                        "prompt_count": len(generated_prompts),
                        "target_agents": list(generated_prompts.keys()),
                    },
                )
            )

            state.current_agent = self.agent_role

            logger.info(
                "meta_prompt_processing_complete",
                prompt_count=len(generated_prompts),
                duration_ms=round(duration_ms, 2),
            )

            return state

        except Exception as e:
            logger.error("meta_prompt_error", error=str(e))

            duration_ms = (time.perf_counter() - start_time) * 1000
            state.agent_responses.append(
                AgentResponse(
                    agent_role=self.agent_role,
                    content="",
                    confidence=0.0,
                    execution_time_ms=duration_ms,
                    metadata={"error": str(e)},
                )
            )

            return state

    def _get_relevant_examples(
        self, target_agent: AgentRole, query: str
    ) -> str:
        """Get few-shot examples relevant to query and agent.

        Uses keyword matching to select the most relevant examples
        for the given query. Falls back to first 2 examples if no
        keyword matches found.

        Args:
            target_agent: Agent type to get examples for.
            query: User query to match against.

        Returns:
            Formatted string of relevant examples.
        """
        agent_examples = FEW_SHOT_EXAMPLES.get(target_agent.value, [])

        if not agent_examples:
            return "No specific examples available for this agent type."

        # Select most relevant examples via keyword matching
        query_lower = query.lower()
        relevant: list[dict[str, Any]] = []

        for example in agent_examples:
            keywords = example.get("keywords", [])
            if any(kw in query_lower for kw in keywords):
                relevant.append(example)

        # Fall back to first 2 examples if no keyword matches
        if not relevant:
            relevant = agent_examples[:2]

        # Format examples (max 3)
        formatted: list[str] = []
        for ex in relevant[:3]:
            formatted.append(
                f"Query: {ex['query']}\n"
                f"Expected Output: {ex['output']}"
            )

        return "\n\n".join(formatted) if formatted else "No examples available."

    def _format_context(self, context: dict[str, Any] | None) -> str:
        """Format context dictionary for prompt generation.

        Extracts relevant context fields and formats them into
        a readable string for the LLM.

        Args:
            context: Context dictionary with user information.

        Returns:
            Formatted context string.
        """
        if not context:
            return "No additional context available."

        parts: list[str] = []

        if user_loc := context.get("user_location"):
            parts.append(f"User location: {user_loc}")

        if emotional_state := context.get("emotional_state"):
            parts.append(f"Emotional state: {emotional_state}")

        if prev_queries := context.get("recent_queries"):
            if isinstance(prev_queries, list):
                parts.append(f"Recent queries: {', '.join(prev_queries[:3])}")

        if preferences := context.get("preferences"):
            parts.append(f"Preferences: {preferences}")

        return " | ".join(parts) if parts else "No additional context."

    def _build_context_from_state(
        self, state: MultiAgentState
    ) -> dict[str, Any]:
        """Build context dictionary from workflow state.

        Extracts relevant information from the MultiAgentState
        to provide context for prompt generation.

        Args:
            state: Current workflow state.

        Returns:
            Context dictionary for prompt generation.
        """
        context: dict[str, Any] = {}

        # Extract from memory context if available
        if state.memory_context:
            if "user_location" in state.memory_context:
                context["user_location"] = state.memory_context["user_location"]
            if "emotional_state" in state.memory_context:
                context["emotional_state"] = state.memory_context["emotional_state"]
            if "preferences" in state.memory_context:
                context["preferences"] = state.memory_context["preferences"]

        # Extract recent queries from agent responses
        recent_queries: list[str] = []
        for response in state.agent_responses[-5:]:
            if response.metadata.get("query"):
                recent_queries.append(response.metadata["query"])

        if recent_queries:
            context["recent_queries"] = recent_queries

        return context

    def track_prompt_performance(
        self,
        prompt_hash: str,
        quality_score: float,
    ) -> None:
        """Track prompt performance for optimization.

        Records quality scores for generated prompts to enable
        future optimization decisions.

        Args:
            prompt_hash: Hash identifying the prompt.
            quality_score: Quality score from 0.0 to 1.0.
        """
        if prompt_hash not in self._prompt_performance:
            self._prompt_performance[prompt_hash] = []

        self._prompt_performance[prompt_hash].append(quality_score)

        # Log when we have enough data for analysis
        scores = self._prompt_performance[prompt_hash]
        if len(scores) >= 10:
            avg_score = sum(scores) / len(scores)
            logger.info(
                "prompt_performance_tracked",
                prompt_hash=prompt_hash[:8],
                avg_score=round(avg_score, 3),
                sample_count=len(scores),
            )

    def get_prompt_hash(self, prompt: str) -> str:
        """Generate hash for a prompt string.

        Args:
            prompt: Prompt text to hash.

        Returns:
            MD5 hash of the prompt.
        """
        return hashlib.md5(prompt.encode()).hexdigest()

    def get_base_template(self, agent_role: AgentRole) -> str | None:
        """Get base template for an agent type.

        Args:
            agent_role: Agent type to get template for.

        Returns:
            Base template string or None if not found.
        """
        return AGENT_PROMPT_TEMPLATES.get(agent_role.value)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for generated prompts.

        Returns:
            Dictionary with performance statistics.
        """
        stats: dict[str, Any] = {
            "total_generated": self._generation_count,
            "tracked_prompts": len(self._prompt_performance),
            "prompt_scores": {},
        }

        for prompt_hash, scores in self._prompt_performance.items():
            if scores:
                stats["prompt_scores"][prompt_hash[:8]] = {
                    "avg": round(sum(scores) / len(scores), 3),
                    "count": len(scores),
                    "min": round(min(scores), 3),
                    "max": round(max(scores), 3),
                }

        return stats


# =============================================================================
# Factory function for easy instantiation
# =============================================================================


def create_meta_prompt_agent(
    model_name: str = "gpt-4o",
    temperature: float = 0.3,
) -> MetaPromptAgent:
    """Create a Meta-Prompt Agent instance.

    Factory function for consistent agent creation.

    Args:
        model_name: LLM model to use.
        temperature: Sampling temperature.

    Returns:
        Configured MetaPromptAgent instance.
    """
    return MetaPromptAgent(
        model_name=model_name,
        temperature=temperature,
    )
