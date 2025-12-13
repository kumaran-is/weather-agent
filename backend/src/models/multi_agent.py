"""Multi-Agent Models for Level 4a/4b/4c: Multi-Agent Orchestration System.

CRITICAL: This module defines the base models for multi-agent orchestration
in the Weather AI Agent Service.

Models:
- AgentRole: Enumeration of agent types in the system
- RoutingDecision: Decision model for agent-to-agent routing
- AgentResponse: Response model from individual agents
- AgentState: Base state model for agent orchestration (NEW - Level 4)
- MultiAgentState: Extended state for multi-agent workflows

Level 4a Architecture (3 Agents):
- Triage Agent → Routes based on query analysis
- Hurricane Specialist Agent → Handles hurricane-specific queries
  (NOTE: Always refers to Hurricane MCP Server at http://localhost:8081 for hurricane data)
- Alert Manager Agent → Generates and formats user-facing alerts

Level 4b Architecture (8 Agents):
- Supervisor Agent → Orchestrates complex multi-agent workflows
- Triage Agent → Query classification and routing
- Hurricane Specialist Agent → Hurricane domain expertise
- Forecaster Agent → General weather forecasting
- Historical Analyst Agent → Historical pattern analysis
- Research Agent → Deep data retrieval
- Verification Agent → Response accuracy checking
- Synthesis Agent → Combines multiple agent outputs

Level 4c Architecture (15 Agents):
- All Level 4b agents (8)
- Meta-Prompt Agent → Dynamic prompt generation for other agents
- Self-Healing Agent → Automatic retry, fallback, and circuit breakers
- Debate Agent → Multi-proposal evaluation and selection
- Emergency Response Agent → Urgent weather emergency handling
- Climate Analyst Agent → Long-term climate pattern analysis
- Personalization Agent → User preference-based response customization
- Reflection Agent → Self-critique and improvement (from L4b, enhanced)

Design Principles:
- Confidence-based routing (track accuracy for continuous improvement)
- Max 1s timeout per agent call (configurable)
- Parallel execution for independent agents (40% latency reduction)
- Async-compatible state management
- Pydantic v2 validation with field validators
- Self-healing with circuit breakers (Level 4c)
- Load-aware routing with cost optimization (Level 4c)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AgentRole(str, Enum):
    """Agent roles in the multi-agent system.

    Level 4a Foundation (3 agents):
    - TRIAGE: Initial query analysis and routing
    - HURRICANE_SPECIALIST: Hurricane-specific expertise
      (Uses Hurricane MCP Server at http://localhost:8081)
    - ALERT_MANAGER: User-facing alert generation and formatting

    Level 4b Expansion (8 agents total):
    - SUPERVISOR: Orchestrates complex multi-agent workflows
    - FORECASTER: General weather forecasting
    - HISTORICAL_ANALYST: Historical pattern analysis
    - VERIFICATION: Response accuracy checking
    - RESEARCH: Deep data retrieval
    - SYNTHESIS: Combines multiple agent outputs

    Level 4c Production (15 agents total):
    - META_PROMPT: Dynamic prompt generation for other agents
    - SELF_HEALING: Automatic retry, fallback, circuit breakers
    - DEBATE: Multi-proposal evaluation and selection
    - EMERGENCY_RESPONSE: Urgent weather emergency handling
    - CLIMATE_ANALYST: Long-term climate pattern analysis
    - PERSONALIZATION: User preference-based customization
    - REFLECTION: Self-critique and improvement (enhanced from L4b)
    """

    # Level 4a agents (Foundation)
    TRIAGE = "triage"
    HURRICANE_SPECIALIST = "hurricane_specialist"
    ALERT_MANAGER = "alert_manager"
    DIRECT_RESPONSE = "direct_response"  # For simple queries and educational questions

    # Level 4b agents (Expansion)
    SUPERVISOR = "supervisor"
    FORECASTER = "forecaster"
    HISTORICAL_ANALYST = "historical_analyst"
    VERIFICATION = "verification"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"

    # Level 4c agents (Production)
    META_PROMPT = "meta_prompt"
    SELF_HEALING = "self_healing"
    DEBATE = "debate"
    EMERGENCY_RESPONSE = "emergency_response"
    CLIMATE_ANALYST = "climate_analyst"
    PERSONALIZATION = "personalization"
    REFLECTION = "reflection"


class QueryComplexity(str, Enum):
    """Query complexity levels for routing decisions.

    Four complexity levels guide agent routing:
    - SIMPLE: Direct response, no specialist needed (e.g., "Weather in London?")
    - MODERATE: Single specialist agent required (e.g., "Hurricane forecast?")
    - COMPLEX: Multi-agent coordination required (e.g., "Prepare for Cat 4?")
    - EMERGENCY: Immediate alert required (e.g., "Evacuate now?")
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EMERGENCY = "emergency"


class RoutingDecision(BaseModel):
    """Routing decision from one agent to another.

    Contains the logic for agent-to-agent handoff including:
    - Next agent selection
    - Confidence score for the routing decision
    - Rationale explaining why this route was chosen
    - Timestamp for decision tracking

    Used by Triage Agent to route queries to specialized agents.
    """

    next_agent: AgentRole = Field(
        description="Agent to route to next"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in routing decision (0.0 to 1.0)"
    )
    rationale: str = Field(
        description="Explanation for routing decision"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When routing decision was made"
    )
    query_category: str = Field(
        description="Categorization of the query (e.g., 'hurricane_forecast', 'general_weather', 'emergency_alert')"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range.

        Args:
            v: Confidence score to validate

        Returns:
            Validated confidence score

        Raises:
            ValueError: If confidence is not between 0.0 and 1.0
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        """Ensure rationale is not empty.

        Args:
            v: Rationale string to validate

        Returns:
            Validated rationale string

        Raises:
            ValueError: If rationale is empty
        """
        if not v or not v.strip():
            raise ValueError("Rationale cannot be empty")
        return v.strip()


class AgentResponse(BaseModel):
    """Response from an individual agent in the workflow.

    Captures the output from a single agent execution including:
    - Agent role that generated the response
    - Response content
    - Confidence score
    - Timestamp and execution metadata

    Used to track agent-by-agent execution in multi-agent workflows.
    """

    agent_role: AgentRole = Field(
        description="Role of the agent that generated this response"
    )
    content: str = Field(
        description="Response content from the agent"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the response (0.0 to 1.0)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When response was generated"
    )
    execution_time_ms: float | None = Field(
        default=None,
        description="Time taken to execute agent (milliseconds)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tool calls, tokens used, etc.)"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range.

        Args:
            v: Confidence score to validate

        Returns:
            Validated confidence score

        Raises:
            ValueError: If confidence is not between 0.0 and 1.0
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is not empty.

        Args:
            v: Content string to validate

        Returns:
            Validated content string

        Raises:
            ValueError: If content is empty
        """
        if not v or not v.strip():
            raise ValueError("Response content cannot be empty")
        return v.strip()

    @field_validator("execution_time_ms")
    @classmethod
    def validate_execution_time(cls, v: float | None) -> float | None:
        """Ensure execution time is positive if provided.

        Args:
            v: Execution time to validate

        Returns:
            Validated execution time or None

        Raises:
            ValueError: If execution time is negative
        """
        if v is not None and v < 0:
            raise ValueError(f"Execution time cannot be negative, got {v}")
        return v


class AgentState(BaseModel):
    """Base state model for agent orchestration (NEW - Level 4).

    This is the foundational state model for all agent workflows.
    Contains core fields required for agent execution:
    - User context (user_id, session_id)
    - Query information
    - Timestamps
    - Basic workflow metadata

    Extended by MultiAgentState for multi-agent orchestration.
    """

    query: str = Field(
        description="User's original query"
    )
    user_id: str = Field(
        description="Unique identifier for the user"
    )
    session_id: str = Field(
        description="Session identifier for conversation continuity"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the agent state was created"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the agent state was last updated"
    )
    timeout_ms: int = Field(
        default=1000,
        ge=100,
        le=30000,
        description="Maximum execution time per agent (milliseconds, max 1s default)"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is not empty.

        Args:
            v: Query string to validate

        Returns:
            Validated query string

        Raises:
            ValueError: If query is empty
        """
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @field_validator("user_id", "session_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Ensure IDs are not empty.

        Args:
            v: ID string to validate

        Returns:
            Validated ID string

        Raises:
            ValueError: If ID is empty
        """
        if not v or not v.strip():
            raise ValueError("ID fields cannot be empty")
        return v.strip()

    @field_validator("timeout_ms")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Ensure timeout is within acceptable range.

        Args:
            v: Timeout value to validate

        Returns:
            Validated timeout value

        Raises:
            ValueError: If timeout is outside acceptable range (100ms to 30s)
        """
        if v < 100:
            raise ValueError(f"Timeout too short: {v}ms (minimum 100ms)")
        if v > 30000:
            raise ValueError(f"Timeout too long: {v}ms (maximum 30000ms)")
        return v


class MultiAgentState(AgentState):
    """Extended state for multi-agent workflows (Level 4a/4b).

    Extends AgentState with multi-agent orchestration fields:
    - Current agent tracking
    - Routing decisions
    - Agent response history
    - Workflow completion status

    Used by LangGraph StateGraph for agent-to-agent coordination.

    Level 4a Workflow (3 Agents):
    1. Triage Agent → Analyzes query, routes to specialist
    2. Specialist Agent → Processes query (e.g., Hurricane Specialist)
    3. Alert Manager Agent → Formats response for user

    Level 4b Workflow (8 Agents):
    1. Supervisor Agent → Plans workflow, orchestrates execution
    2. Triage Agent → Classifies and routes query
    3. Specialist Agents (Hurricane, Forecaster, Historical, Research) → Parallel execution
    4. Verification Agent → Validates response accuracy
    5. Synthesis Agent → Combines multiple responses
    6. Alert Manager Agent → Generates user-facing alerts

    NOTE: Hurricane Specialist Agent uses Hurricane MCP Server at http://localhost:8081
    for all hurricane-related data.
    """

    # Level 4a fields (Foundation)
    current_agent: AgentRole | None = Field(
        default=None,
        description="Currently executing agent"
    )
    routing_decision: RoutingDecision | None = Field(
        default=None,
        description="Most recent routing decision"
    )
    agent_responses: list[AgentResponse] = Field(
        default_factory=list,
        description="Response history from all agents in workflow"
    )
    next_agent: AgentRole | None = Field(
        default=None,
        description="Next agent to execute (from routing decision)"
    )
    workflow_complete: bool = Field(
        default=False,
        description="Whether the multi-agent workflow is complete"
    )
    final_response: str | None = Field(
        default=None,
        description="Final response to return to user (after all agents complete)"
    )
    error: str | None = Field(
        default=None,
        description="Error message if workflow failed"
    )
    total_execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total execution time across all agents (milliseconds)"
    )

    # Level 4b fields (Expansion)
    memory_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Memory context for the workflow (user history, preferences)"
    )
    workflow_plan: list[AgentRole] = Field(
        default_factory=list,
        description="Planned sequence of agents to execute"
    )
    parallel_groups: list[list[AgentRole]] = Field(
        default_factory=list,
        description="Groups of agents that can run in parallel"
    )
    reflection_iterations: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Number of reflection iterations completed (max 3)"
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality score from reflection/verification (0.0 to 1.0)"
    )
    requires_verification: bool = Field(
        default=False,
        description="Whether response requires verification before completion"
    )
    supervisor_notes: str | None = Field(
        default=None,
        description="Notes from supervisor about workflow execution"
    )

    @field_validator("agent_responses")
    @classmethod
    def validate_agent_responses(cls, v: list[AgentResponse]) -> list[AgentResponse]:
        """Ensure agent responses list is valid.

        Args:
            v: List of agent responses to validate

        Returns:
            Validated list of agent responses
        """
        # Check for duplicate agents in response history (warning, not error)
        agent_counts = {}
        for response in v:
            agent_counts[response.agent_role] = agent_counts.get(response.agent_role, 0) + 1

        # Allow multiple responses from same agent (e.g., retries)
        # Just ensure responses are ordered by timestamp
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i].timestamp < v[i-1].timestamp:
                    # Sort by timestamp if not already sorted
                    v.sort(key=lambda r: r.timestamp)
                    break

        return v

    @field_validator("total_execution_time_ms")
    @classmethod
    def validate_total_execution_time(cls, v: float) -> float:
        """Ensure total execution time is non-negative.

        Args:
            v: Total execution time to validate

        Returns:
            Validated execution time

        Raises:
            ValueError: If execution time is negative
        """
        if v < 0:
            raise ValueError(f"Total execution time cannot be negative, got {v}")
        return v

    def add_agent_response(self, response: AgentResponse) -> None:
        """Add an agent response to the workflow history.

        Updates:
        - Appends response to agent_responses list
        - Updates total_execution_time_ms
        - Updates updated_at timestamp

        Args:
            response: AgentResponse to add to history
        """
        self.agent_responses.append(response)
        if response.execution_time_ms:
            self.total_execution_time_ms += response.execution_time_ms
        self.updated_at = datetime.now(timezone.utc)

    def get_latest_response(self) -> AgentResponse | None:
        """Get the most recent agent response.

        Returns:
            Most recent AgentResponse or None if no responses yet
        """
        if not self.agent_responses:
            return None
        return self.agent_responses[-1]

    def get_responses_by_agent(self, agent_role: AgentRole) -> list[AgentResponse]:
        """Get all responses from a specific agent.

        Args:
            agent_role: Agent role to filter by

        Returns:
            List of responses from the specified agent
        """
        return [r for r in self.agent_responses if r.agent_role == agent_role]

    def mark_complete(self, final_response: str) -> None:
        """Mark the workflow as complete with a final response.

        Args:
            final_response: Final response to return to user
        """
        self.workflow_complete = True
        self.final_response = final_response
        self.updated_at = datetime.now(timezone.utc)

    def mark_error(self, error_message: str) -> None:
        """Mark the workflow as failed with an error message.

        Args:
            error_message: Error message describing the failure
        """
        self.workflow_complete = True
        self.error = error_message
        self.updated_at = datetime.now(timezone.utc)
