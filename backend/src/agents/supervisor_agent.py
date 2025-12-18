"""Supervisor Agent for Level 4b: 8-Agent Orchestration System.

CRITICAL RULES:
1. Supervisor coordinates workflow, does NOT process queries directly
2. Use parallel execution for independent agent tasks (40% latency reduction)
3. Implement agent capability registry for dynamic selection
4. Track all agent invocations for observability
5. Enforce maximum 4 parallel agents (resource constraint)

Architecture:
- AgentCapabilityRegistry: Registry of agent capabilities for dynamic selection
- SupervisorAgent: Orchestrates multi-agent workflows with parallel execution

Design Principles:
- LLM-based workflow planning for flexibility
- Parallel execution of independent specialists
- Quality assurance through verification
- Graceful error handling with fallbacks
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.src.agents.prompts.supervisor_prompts import (
    AGENT_CAPABILITIES,
    SUPERVISOR_SYSTEM_PROMPT,
    WORKFLOW_PLANNING_PROMPT,
)
from backend.src.models.multi_agent import (
    AgentResponse,
    AgentRole,
    MultiAgentState,
)

logger = structlog.get_logger(__name__)


class AgentCapabilityRegistry:
    """Registry of agent capabilities for dynamic selection.

    The registry maintains information about each agent's:
    - Description and purpose
    - Domain expertise (keywords)
    - Maximum expected latency
    - Cost tier (low/medium/high)
    - Model configuration

    Used by SupervisorAgent to:
    - Find agents matching query domains
    - Identify parallel execution opportunities
    - Estimate workflow cost and latency
    """

    def __init__(self) -> None:
        """Initialize the agent capability registry."""
        self.capabilities: dict[AgentRole, dict[str, Any]] = {
            AgentRole.TRIAGE: AGENT_CAPABILITIES["triage"],
            AgentRole.HURRICANE_SPECIALIST: AGENT_CAPABILITIES["hurricane_specialist"],
            AgentRole.ALERT_MANAGER: AGENT_CAPABILITIES["alert_manager"],
            AgentRole.FORECASTER: AGENT_CAPABILITIES["forecaster"],
            AgentRole.HISTORICAL_ANALYST: AGENT_CAPABILITIES["historical_analyst"],
            AgentRole.VERIFICATION: AGENT_CAPABILITIES["verification"],
            AgentRole.RESEARCH: AGENT_CAPABILITIES["research"],
            AgentRole.SYNTHESIS: AGENT_CAPABILITIES["synthesis"],
        }

        # Define which agents can run in parallel (no dependencies)
        self._parallel_capable: set[AgentRole] = {
            AgentRole.HURRICANE_SPECIALIST,
            AgentRole.FORECASTER,
            AgentRole.HISTORICAL_ANALYST,
            AgentRole.RESEARCH,
        }

        # Define agent dependencies
        self._dependencies: dict[AgentRole, set[AgentRole]] = {
            AgentRole.TRIAGE: set(),  # No dependencies
            AgentRole.HURRICANE_SPECIALIST: {AgentRole.TRIAGE},
            AgentRole.FORECASTER: {AgentRole.TRIAGE},
            AgentRole.HISTORICAL_ANALYST: {AgentRole.TRIAGE},
            AgentRole.RESEARCH: {AgentRole.TRIAGE},
            AgentRole.ALERT_MANAGER: {AgentRole.TRIAGE},
            AgentRole.SYNTHESIS: {AgentRole.HURRICANE_SPECIALIST, AgentRole.FORECASTER},
            AgentRole.VERIFICATION: {AgentRole.SYNTHESIS, AgentRole.HURRICANE_SPECIALIST},
        }

        logger.debug("agent_registry_initialized", agent_count=len(self.capabilities))

    def get_agents_for_domains(self, domains: list[str]) -> list[AgentRole]:
        """Find agents that can handle given domains.

        Args:
            domains: List of domain keywords (e.g., ["hurricane", "forecast"])

        Returns:
            List of AgentRole that match any of the domains
        """
        matching_agents = []
        domains_lower = [d.lower() for d in domains]

        for role, caps in self.capabilities.items():
            agent_domains = caps.get("domains", [])
            if any(d in agent_domains for d in domains_lower):
                matching_agents.append(role)

        logger.debug(
            "agents_for_domains",
            domains=domains,
            matching_agents=[a.value for a in matching_agents],
        )

        return matching_agents

    def get_independent_agents(self, agents: list[AgentRole]) -> list[set[AgentRole]]:
        """Identify agents that can run in parallel (no dependencies).

        Groups agents into parallel execution sets based on:
        - Agent is in _parallel_capable set
        - Agent has no pending dependencies

        Args:
            agents: List of agents planned for execution

        Returns:
            List of sets, where each set contains agents that can run in parallel
        """
        # Filter to parallel-capable agents
        parallel_group = {a for a in agents if a in self._parallel_capable}
        sequential_agents = [a for a in agents if a not in self._parallel_capable]

        groups: list[set[AgentRole]] = []

        # Add parallel group first (if any)
        if parallel_group:
            groups.append(parallel_group)

        # Add sequential agents as individual groups
        for agent in sequential_agents:
            groups.append({agent})

        logger.debug(
            "independent_agents_grouped",
            total_agents=len(agents),
            parallel_count=len(parallel_group),
            sequential_count=len(sequential_agents),
            groups_count=len(groups),
        )

        return groups

    def get_capability(self, role: AgentRole) -> dict[str, Any]:
        """Get capability info for a specific agent.

        Args:
            role: Agent role to look up

        Returns:
            Dictionary containing agent capabilities
        """
        return self.capabilities.get(role, {})

    def estimate_latency(self, agents: list[AgentRole]) -> float:
        """Estimate total latency for a list of agents.

        Considers parallel execution to reduce total latency.

        Args:
            agents: List of agents planned for execution

        Returns:
            Estimated total latency in milliseconds
        """
        groups = self.get_independent_agents(agents)
        total_latency = 0.0

        for group in groups:
            # For parallel groups, take the max latency
            group_latencies = [
                self.capabilities.get(a, {}).get("max_latency_ms", 10000)
                for a in group
            ]
            total_latency += max(group_latencies) if group_latencies else 0

        return total_latency


class SupervisorAgent:
    """Orchestrate multi-agent workflows with parallel execution.

    The Supervisor Agent coordinates up to 8 specialist agents:
    1. Plans workflow based on query analysis
    2. Executes agents in optimal order (parallel where possible)
    3. Synthesizes responses when multiple agents contribute
    4. Verifies accuracy for complex/emergency queries

    Key Features:
    - LLM-based workflow planning for flexibility
    - Parallel execution reduces latency by ~40%
    - Agent capability registry for dynamic selection
    - Quality assurance through verification agent
    - Comprehensive logging for observability

    Attributes:
        llm: ChatOpenAI model for workflow planning
        agent_role: Fixed as AgentRole.SUPERVISOR
        registry: AgentCapabilityRegistry for agent selection
        max_parallel_agents: Maximum concurrent agent executions (default: 4)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_parallel_agents: int = 4,
    ) -> None:
        """Initialize Supervisor Agent.

        Args:
            model_name: LLM model for workflow planning (default: gpt-4o-mini for speed)
            max_parallel_agents: Maximum concurrent agents (default: 4)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,  # Deterministic planning
            timeout=10.0,  # Fast planning
        )
        self.agent_role = AgentRole.SUPERVISOR
        self.registry = AgentCapabilityRegistry()
        self.max_parallel_agents = max_parallel_agents

        # Agent instances (lazy initialization)
        self._agents: dict[AgentRole, Any] = {}

        logger.info(
            "supervisor_agent_initialized",
            model=model_name,
            max_parallel=max_parallel_agents,
        )

    def _get_agent(self, role: AgentRole) -> Any | None:
        """Get or create agent instance (lazy initialization).

        Args:
            role: Agent role to get/create

        Returns:
            Agent instance or None if not available
        """
        if role in self._agents:
            return self._agents[role]

        try:
            if role == AgentRole.TRIAGE:
                from backend.src.agents.triage_agent import TriageAgent
                self._agents[role] = TriageAgent()

            elif role == AgentRole.HURRICANE_SPECIALIST:
                from backend.src.agents.hurricane_specialist import HurricaneSpecialistAgent
                self._agents[role] = HurricaneSpecialistAgent()

            elif role == AgentRole.ALERT_MANAGER:
                from backend.src.agents.alert_manager import AlertManagerAgent
                self._agents[role] = AlertManagerAgent()

            elif role == AgentRole.FORECASTER:
                from backend.src.agents.forecaster_agent import ForecasterAgent
                self._agents[role] = ForecasterAgent()

            elif role == AgentRole.HISTORICAL_ANALYST:
                from backend.src.agents.historical_agent import HistoricalAnalystAgent
                self._agents[role] = HistoricalAnalystAgent()

            elif role == AgentRole.RESEARCH:
                from backend.src.agents.research_agent import ResearchAgent
                self._agents[role] = ResearchAgent()

            elif role == AgentRole.VERIFICATION:
                from backend.src.agents.reflection_agent import ReflectionAgent
                # Verification uses reflection agent for quality checking
                self._agents[role] = ReflectionAgent()

            elif role == AgentRole.SYNTHESIS:
                # Synthesis is handled inline by supervisor
                self._agents[role] = None

            logger.debug("agent_initialized", role=role.value)
            return self._agents.get(role)

        except ImportError as e:
            logger.warning(
                "agent_import_failed",
                role=role.value,
                error=str(e),
            )
            return None

    async def orchestrate(
        self,
        state: MultiAgentState,
    ) -> MultiAgentState:
        """Orchestrate multi-agent workflow.

        Steps:
        1. Analyze query and plan workflow
        2. Execute triage first (always sequential)
        3. Execute specialist agents (parallel where possible)
        4. Synthesize responses if multiple agents contributed
        5. Run verification for complex/emergency queries
        6. Return updated state with final response

        Args:
            state: Current multi-agent state with query

        Returns:
            Updated state with agent responses and final response
        """
        start_time = time.perf_counter()

        logger.info(
            "supervisor_orchestration_started",
            query=state.query[:100],
            user_id=state.user_id,
            session_id=state.session_id,
        )

        try:
            # Step 1: Plan workflow
            workflow_plan = await self._plan_workflow(state)

            state.workflow_plan = workflow_plan["agents"]
            state.parallel_groups = workflow_plan["parallel_groups"]
            state.requires_verification = workflow_plan["requires_verification"]

            logger.info(
                "workflow_planned",
                query=state.query[:50],
                agents=[a.value for a in workflow_plan["agents"]],
                parallel_groups=len(workflow_plan["parallel_groups"]),
                requires_verification=workflow_plan["requires_verification"],
            )

            # Step 2: Execute triage first (always sequential)
            triage_agent = self._get_agent(AgentRole.TRIAGE)
            if triage_agent:
                state = await triage_agent.classify_and_route(state)

            # Step 3: Execute specialist agents
            specialist_agents = [
                a for a in workflow_plan["agents"]
                if a not in {AgentRole.TRIAGE, AgentRole.VERIFICATION, AgentRole.SYNTHESIS}
            ]

            # Get parallel execution groups
            execution_groups = self.registry.get_independent_agents(specialist_agents)

            for group in execution_groups:
                group_list = list(group)

                if len(group_list) > 1 and len(group_list) <= self.max_parallel_agents:
                    # Parallel execution
                    state = await self._execute_parallel(state, group_list)
                else:
                    # Sequential execution
                    for agent_role in group_list:
                        state = await self._execute_sequential(state, agent_role)

            # Step 4: Synthesize responses if multiple agents contributed
            specialist_responses = [
                r for r in state.agent_responses
                if r.agent_role not in {AgentRole.TRIAGE, AgentRole.SUPERVISOR}
                and r.content
                and not r.metadata.get("error")
            ]

            if len(specialist_responses) >= 2 or workflow_plan.get("requires_synthesis", False):
                state = await self._synthesize_responses(state)

            # Step 5: Run verification for complex/emergency queries
            if state.requires_verification:
                state = await self._verify_response(state)

            # Step 6: Set final response
            if not state.final_response:
                # Use latest non-error response as final
                valid_responses = [
                    r for r in state.agent_responses
                    if r.content and not r.metadata.get("error")
                ]
                if valid_responses:
                    state.final_response = valid_responses[-1].content

            duration_ms = (time.perf_counter() - start_time) * 1000
            state.total_execution_time_ms = duration_ms
            state.workflow_complete = True
            state.current_agent = AgentRole.SUPERVISOR  # Supervisor remains orchestrator

            logger.info(
                "supervisor_orchestration_complete",
                query=state.query[:50],
                total_agents=len(state.agent_responses),
                duration_ms=duration_ms,
                quality_score=state.quality_score,
            )

            return state

        except Exception as e:
            logger.error(
                "supervisor_orchestration_error",
                error=str(e),
                error_type=type(e).__name__,
                query=state.query[:100],
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Add error response
            error_response = AgentResponse(
                agent_role=AgentRole.SUPERVISOR,
                content=f"Workflow orchestration failed: {type(e).__name__}",
                confidence=0.0,
                timestamp=datetime.now(UTC),
                execution_time_ms=duration_ms,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            state.agent_responses.append(error_response)
            state.error = str(e)
            state.workflow_complete = True
            state.total_execution_time_ms = duration_ms
            state.current_agent = AgentRole.SUPERVISOR  # Supervisor remains orchestrator

            return state

    async def _plan_workflow(self, state: MultiAgentState) -> dict[str, Any]:
        """Use LLM to plan agent workflow.

        Args:
            state: Current multi-agent state

        Returns:
            Dictionary with workflow plan:
            - agents: List of AgentRole to execute
            - parallel_groups: List of agent groups for parallel execution
            - requires_verification: Whether verification is needed
            - requires_synthesis: Whether synthesis is needed
        """
        # Determine complexity hint from query
        complexity_hint = self._analyze_complexity(state.query)

        planning_messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(
                content=WORKFLOW_PLANNING_PROMPT.format(
                    query=state.query,
                    available_agents=[r.value for r in AgentRole if r != AgentRole.SUPERVISOR],
                    complexity_hint=complexity_hint,
                )
            ),
        ]

        try:
            response = await self.llm.ainvoke(planning_messages)
            plan = json.loads(response.content)

            # Parse planned agents
            agents = [AgentRole(a) for a in plan.get("agents", ["triage", "hurricane_specialist"])]

            # Ensure triage is first
            if AgentRole.TRIAGE not in agents:
                agents.insert(0, AgentRole.TRIAGE)

            # Parse parallel groups
            parallel_groups_raw = plan.get("parallel_groups", [])
            parallel_groups = []
            for group in parallel_groups_raw:
                try:
                    parallel_groups.append([AgentRole(a) for a in group])
                except ValueError:
                    pass

            # If no parallel groups defined, use registry to determine
            if not parallel_groups:
                groups = self.registry.get_independent_agents(agents)
                parallel_groups = [list(g) for g in groups]

            return {
                "agents": agents,
                "parallel_groups": parallel_groups,
                "requires_verification": plan.get("requires_verification", complexity_hint in ["complex", "emergency"]),
                "requires_synthesis": plan.get("requires_synthesis", len(agents) >= 3),
                "reasoning": plan.get("reasoning", ""),
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "workflow_planning_fallback",
                error=str(e),
            )

            # Fallback: triage + hurricane specialist
            return {
                "agents": [AgentRole.TRIAGE, AgentRole.HURRICANE_SPECIALIST],
                "parallel_groups": [[AgentRole.TRIAGE], [AgentRole.HURRICANE_SPECIALIST]],
                "requires_verification": False,
                "requires_synthesis": False,
                "reasoning": "Fallback plan due to LLM parsing error",
            }

    def _analyze_complexity(self, query: str) -> str:
        """Analyze query to determine complexity hint.

        Args:
            query: User's query string

        Returns:
            Complexity hint: "simple", "moderate", "complex", or "emergency"
        """
        query_lower = query.lower()

        # Emergency indicators
        emergency_keywords = [
            "evacuate now", "immediate danger", "life threatening",
            "category 5", "cat 5", "mandatory evacuation", "emergency",
        ]
        if any(kw in query_lower for kw in emergency_keywords):
            return "emergency"

        # Complex indicators
        complex_keywords = [
            "should i evacuate", "prepare for", "multiple", "compare",
            "category 4", "cat 4", "major hurricane", "long-term",
            "extended forecast", "next week", "versus",
        ]
        if any(kw in query_lower for kw in complex_keywords):
            return "complex"

        # Simple indicators
        simple_keywords = [
            "what is", "temperature", "rain today", "current weather",
            "humidity", "wind speed",
        ]
        if any(kw in query_lower for kw in simple_keywords):
            return "simple"

        return "moderate"

    async def _execute_parallel(
        self,
        state: MultiAgentState,
        agents: list[AgentRole],
    ) -> MultiAgentState:
        """Execute multiple agents in parallel.

        Args:
            state: Current multi-agent state
            agents: List of agents to execute in parallel

        Returns:
            Updated state with agent responses
        """
        logger.info(
            "parallel_execution_started",
            agents=[a.value for a in agents],
        )

        tasks: list[asyncio.Task] = []
        task_agents: list[AgentRole] = []

        for role in agents:
            agent = self._get_agent(role)
            if agent:
                # Create task for parallel execution
                if role == AgentRole.HURRICANE_SPECIALIST:
                    task = asyncio.create_task(agent.process_query(state.model_copy()))
                elif role == AgentRole.ALERT_MANAGER:
                    task = asyncio.create_task(agent.generate_alert(state.model_copy()))
                elif role == AgentRole.FORECASTER:
                    task = asyncio.create_task(agent.generate_forecast(state.model_copy()))
                elif role == AgentRole.HISTORICAL_ANALYST:
                    task = asyncio.create_task(agent.analyze_patterns(state.model_copy()))
                elif role == AgentRole.RESEARCH:
                    task = asyncio.create_task(agent.research_query(state.model_copy()))
                else:
                    continue

                tasks.append(task)
                task_agents.append(role)

        if not tasks:
            return state

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results into state
        for i, result in enumerate(results):
            agent_role = task_agents[i]

            if isinstance(result, Exception):
                logger.error(
                    "parallel_agent_failed",
                    agent=agent_role.value,
                    error=str(result),
                )
                # Add error response
                state.agent_responses.append(
                    AgentResponse(
                        agent_role=agent_role,
                        content=f"Agent {agent_role.value} failed: {type(result).__name__}",
                        confidence=0.0,
                        timestamp=datetime.now(UTC),
                        execution_time_ms=0,
                        metadata={"error": str(result)},
                    )
                )
            elif isinstance(result, dict):
                # Result is a state dict - extract responses
                if "agent_responses" in result:
                    state.agent_responses.extend(result["agent_responses"])
            elif hasattr(result, "agent_responses"):
                # Result is a state object
                state.agent_responses.extend(result.agent_responses)

        logger.info(
            "parallel_execution_complete",
            agents=[a.value for a in agents],
            successes=len([r for r in results if not isinstance(r, Exception)]),
            failures=len([r for r in results if isinstance(r, Exception)]),
        )

        return state

    async def _execute_sequential(
        self,
        state: MultiAgentState,
        role: AgentRole,
    ) -> MultiAgentState:
        """Execute single agent sequentially.

        Args:
            state: Current multi-agent state
            role: Agent role to execute

        Returns:
            Updated state with agent response
        """
        agent = self._get_agent(role)

        if not agent:
            logger.warning("agent_not_found", role=role.value)
            return state

        logger.debug("sequential_execution_started", agent=role.value)

        try:
            if role == AgentRole.HURRICANE_SPECIALIST:
                result = await agent.process_query(state)
            elif role == AgentRole.ALERT_MANAGER:
                result = await agent.generate_alert(state)
            elif role == AgentRole.FORECASTER:
                result = await agent.generate_forecast(state)
            elif role == AgentRole.HISTORICAL_ANALYST:
                result = await agent.analyze_patterns(state)
            elif role == AgentRole.RESEARCH:
                result = await agent.research_query(state)
            else:
                return state

            # Merge result into state
            if isinstance(result, dict):
                for key, value in result.items():
                    if key == "agent_responses":
                        state.agent_responses.extend(value)
                    elif hasattr(state, key):
                        setattr(state, key, value)
            elif hasattr(result, "agent_responses"):
                state.agent_responses.extend(result.agent_responses)

            return state

        except Exception as e:
            logger.error(
                "sequential_agent_failed",
                agent=role.value,
                error=str(e),
            )
            state.agent_responses.append(
                AgentResponse(
                    agent_role=role,
                    content=f"Agent {role.value} failed: {type(e).__name__}",
                    confidence=0.0,
                    timestamp=datetime.now(UTC),
                    execution_time_ms=0,
                    metadata={"error": str(e)},
                )
            )
            return state

    async def _synthesize_responses(self, state: MultiAgentState) -> MultiAgentState:
        """Synthesize multiple agent responses into coherent output.

        Args:
            state: Current multi-agent state with multiple responses

        Returns:
            Updated state with synthesized response
        """
        start_time = time.perf_counter()

        # Get valid responses for synthesis
        responses_to_synthesize = [
            r for r in state.agent_responses
            if r.agent_role not in {AgentRole.TRIAGE, AgentRole.SUPERVISOR}
            and r.content
            and not r.metadata.get("error")
        ]

        if len(responses_to_synthesize) < 2:
            return state

        # Format responses for synthesis
        responses_text = "\n\n".join([
            f"**{r.agent_role.value}** (confidence: {r.confidence:.2f}):\n{r.content}"
            for r in responses_to_synthesize
        ])

        synthesis_messages = [
            SystemMessage(
                content="""You are a synthesis agent for a Weather AI system.

Your task is to combine multiple agent responses into a single, coherent answer.

Guidelines:
1. Preserve all factual information from each agent
2. Eliminate redundancy without losing information
3. Maintain consistent formatting and tone
4. Prioritize life-safety information (put it first)
5. Include confidence levels when relevant
6. Note any contradictions between agents

Output should be comprehensive but concise."""
            ),
            HumanMessage(
                content=f"""Synthesize these weather expert responses into a single comprehensive answer:

**Original Query**: {state.query}

**Agent Responses**:
{responses_text}

Provide a synthesized response that combines all relevant information."""
            ),
        ]

        try:
            response = await self.llm.ainvoke(synthesis_messages)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Calculate synthesis confidence (average of source confidences)
            avg_confidence = sum(r.confidence for r in responses_to_synthesize) / len(responses_to_synthesize)

            synthesis_response = AgentResponse(
                agent_role=AgentRole.SYNTHESIS,
                content=response.content,
                confidence=min(avg_confidence + 0.05, 0.95),  # Slight boost for synthesis
                timestamp=datetime.now(UTC),
                execution_time_ms=duration_ms,
                metadata={
                    "source_agents": [r.agent_role.value for r in responses_to_synthesize],
                    "source_count": len(responses_to_synthesize),
                },
            )

            state.agent_responses.append(synthesis_response)
            state.final_response = response.content

            logger.info(
                "synthesis_complete",
                source_count=len(responses_to_synthesize),
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error("synthesis_failed", error=str(e))

        return state

    async def _verify_response(self, state: MultiAgentState) -> MultiAgentState:
        """Verify final response for accuracy.

        Focuses on:
        - Saffir-Simpson scale accuracy
        - Time specificity (EDT/UTC, not "soon")
        - Evacuation zone accuracy
        - Wind speed / category consistency

        Args:
            state: Current multi-agent state

        Returns:
            Updated state with verification response and quality score
        """
        start_time = time.perf_counter()

        # Get latest response to verify
        valid_responses = [
            r for r in state.agent_responses
            if r.content and not r.metadata.get("error")
        ]

        if not valid_responses:
            return state

        latest = valid_responses[-1]

        verification_messages = [
            SystemMessage(
                content="""You are a verification agent for a Weather AI system.

Your task is to verify the accuracy of weather responses, especially:

1. **Saffir-Simpson Scale Accuracy**:
   - Cat 1: 74-95 mph
   - Cat 2: 96-110 mph
   - Cat 3: 111-129 mph
   - Cat 4: 130-156 mph
   - Cat 5: 157+ mph
   - CRITICAL: Category MUST match wind speed range

2. **Time Specificity**:
   - Use exact times (e.g., "3:00 PM EDT", "1500 UTC")
   - Never use vague terms like "soon", "later", "eventually"

3. **Evacuation Guidance**:
   - Zones should be letter-based (A, B, C)
   - Never subjective descriptions

4. **Life-Safety Priority**:
   - Evacuation info should be prominent
   - Err on side of caution

Respond with JSON:
{
    "is_accurate": true/false,
    "quality_score": 0.0-1.0,
    "issues": ["Issue 1", "Issue 2"],
    "corrections": "Corrected text if needed",
    "verification_notes": "Summary of verification"
}"""
            ),
            HumanMessage(
                content=f"""Verify this weather response:

**Original Query**: {state.query}

**Response to Verify**:
{latest.content}

Check for accuracy and provide quality score."""
            ),
        ]

        try:
            response = await self.llm.ainvoke(verification_messages)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Parse verification result
            try:
                verification = json.loads(response.content)
                quality_score = float(verification.get("quality_score", 0.8))
                issues = verification.get("issues", [])
                is_accurate = verification.get("is_accurate", True)
            except (json.JSONDecodeError, ValueError):
                quality_score = 0.8
                issues = []
                is_accurate = True

            state.quality_score = quality_score

            verification_response = AgentResponse(
                agent_role=AgentRole.VERIFICATION,
                content=response.content,
                confidence=0.95,
                timestamp=datetime.now(UTC),
                execution_time_ms=duration_ms,
                metadata={
                    "verified_agent": latest.agent_role.value,
                    "quality_score": quality_score,
                    "issues_found": len(issues),
                    "is_accurate": is_accurate,
                },
            )

            state.agent_responses.append(verification_response)

            logger.info(
                "verification_complete",
                quality_score=quality_score,
                issues_found=len(issues),
                is_accurate=is_accurate,
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error("verification_failed", error=str(e))
            state.quality_score = 0.7  # Default quality on error

        return state
