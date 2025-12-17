"""LangChain Callback Handler for Auto-Instrumentation.

Level 9: Automatic observability for LangChain/LangGraph operations:
- LLM calls (tokens, latency, cost)
- Tool invocations (input/output, success/failure)
- Chain executions (steps, state changes)
- Agent actions (decisions, reasoning)

Usage:
    >>> from backend.src.observability.callbacks import create_langchain_callbacks
    >>> callbacks = create_langchain_callbacks()
    >>> result = await chain.ainvoke(input, config={"callbacks": callbacks})

Integration with LangGraph:
    >>> from langgraph.graph import StateGraph
    >>> workflow = StateGraph(State)
    >>> app = workflow.compile()
    >>> result = await app.ainvoke(input, config={"callbacks": callbacks})
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.outputs import LLMResult

from backend.src.observability.tracing import (
    SpanAttributes,
    SpanNames,
    create_span,
    get_tracer,
)
from backend.src.observability.metrics import get_metrics

# Use centralized metrics module for consistent metric naming
_PROMETHEUS_AVAILABLE = True

logger = logging.getLogger(__name__)

# Pricing per 1K tokens (approximate, update as needed)
LLM_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    # Default fallback
    "default": {"input": 0.001, "output": 0.002},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate LLM API cost in dollars.

    Args:
        model: Model name/ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    pricing = LLM_PRICING.get(model, LLM_PRICING["default"])
    cost = (input_tokens / 1000 * pricing["input"]) + (
        output_tokens / 1000 * pricing["output"]
    )
    return cost


class ObservabilityCallbackHandler(AsyncCallbackHandler, BaseCallbackHandler):
    """Callback handler for LangChain/LangGraph observability.

    Automatically instruments:
    - LLM calls: Start/end times, tokens, costs
    - Tool invocations: Input/output, latency, success/failure
    - Chain executions: Steps, state changes
    - Agent actions: Decisions, reasoning traces

    Integrates with:
    - OpenTelemetry: Creates spans for distributed tracing
    - Prometheus: Records metrics for monitoring
    - Structured logging: Logs with trace context
    """

    def __init__(
        self,
        run_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        """Initialize observability callback handler.

        Args:
            run_id: Unique identifier for this run/request
            user_id: User identifier for attribution
            session_id: Session identifier for grouping
        """
        super().__init__()
        self.run_id = run_id
        self.user_id = user_id
        self.session_id = session_id

        # Track timing for latency calculation
        self._llm_start_times: dict[str, float] = {}
        self._tool_start_times: dict[str, float] = {}
        self._chain_start_times: dict[str, float] = {}

        # Track active spans for hierarchical tracing
        self._active_spans: dict[str, Any] = {}

        # Track token counts per run
        self._token_counts: dict[str, dict[str, int]] = {}

    # ==================== LLM Callbacks ====================

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts processing."""
        run_id_str = str(run_id)
        self._llm_start_times[run_id_str] = time.time()

        # Extract model info
        model = serialized.get("kwargs", {}).get("model_name", "unknown")
        provider = self._get_provider_from_model(model)

        # Create span for LLM call
        tracer = get_tracer()
        span = tracer.start_span(
            SpanNames.LANGCHAIN_LLM,
            attributes={
                SpanAttributes.LLM_PROVIDER: provider,
                SpanAttributes.LLM_MODEL: model,
                SpanAttributes.REQUEST_USER_ID: self.user_id or "",
                SpanAttributes.REQUEST_SESSION_ID: self.session_id or "",
                "llm.prompt.length": sum(len(p) for p in prompts),
                "llm.prompts.count": len(prompts),
            },
        )
        self._active_spans[f"llm_{run_id_str}"] = span

        logger.debug(
            f"🤖 LLM call started | model: {model} | run_id: {run_id_str[:8]}"
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM completes processing."""
        run_id_str = str(run_id)
        start_time = self._llm_start_times.pop(run_id_str, time.time())
        latency = time.time() - start_time

        # Extract token usage
        token_usage = response.llm_output or {}
        if hasattr(response, "generations") and response.generations:
            if hasattr(response.generations[0][0], "generation_info"):
                gen_info = response.generations[0][0].generation_info or {}
                token_usage = gen_info.get("usage", token_usage)

        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

        # Extract model info
        model = token_usage.get("model", "unknown")
        provider = self._get_provider_from_model(model)

        # Calculate cost
        cost = estimate_cost(model, input_tokens, output_tokens)

        # End span with attributes
        span_key = f"llm_{run_id_str}"
        if span_key in self._active_spans:
            span = self._active_spans.pop(span_key)
            span.set_attribute(SpanAttributes.LLM_INPUT_TOKENS, input_tokens)
            span.set_attribute(SpanAttributes.LLM_OUTPUT_TOKENS, output_tokens)
            span.set_attribute(SpanAttributes.LLM_TOTAL_TOKENS, total_tokens)
            span.set_attribute(SpanAttributes.LLM_COST_USD, cost)
            span.set_attribute(SpanAttributes.LLM_LATENCY_MS, latency * 1000)
            span.end()

        # Record Prometheus metrics using centralized metrics module
        if _PROMETHEUS_AVAILABLE:
            metrics = get_metrics()
            metrics.record_llm_call(
                provider=provider,
                model=model,
                status="success",
                latency=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )

        logger.debug(
            f"🤖 LLM call completed | model: {model} | "
            f"tokens: {total_tokens} | latency: {latency:.2f}s | cost: ${cost:.4f}"
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM encounters an error."""
        run_id_str = str(run_id)
        start_time = self._llm_start_times.pop(run_id_str, time.time())
        latency = time.time() - start_time

        # End span with error status
        span_key = f"llm_{run_id_str}"
        if span_key in self._active_spans:
            span = self._active_spans.pop(span_key)
            span.set_attribute(SpanAttributes.ERROR_TYPE, type(error).__name__)
            span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(error))
            span.record_exception(error)
            span.end()

        # Record Prometheus metrics using centralized metrics module
        if _PROMETHEUS_AVAILABLE:
            metrics = get_metrics()
            metrics.record_llm_call(
                provider="unknown",
                model="unknown",
                status="error",
                latency=latency,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
            )

        logger.error(f"🤖 LLM call failed | error: {error} | latency: {latency:.2f}s")

    # ==================== Tool Callbacks ====================

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts executing."""
        run_id_str = str(run_id)
        self._tool_start_times[run_id_str] = time.time()

        tool_name = serialized.get("name", "unknown")

        # Create span for tool call
        tracer = get_tracer()
        span = tracer.start_span(
            SpanNames.LANGCHAIN_TOOL.format(name=tool_name),
            attributes={
                SpanAttributes.TOOL_NAME: tool_name,
                SpanAttributes.TOOL_INPUT: input_str[:500],  # Truncate for safety
                SpanAttributes.REQUEST_USER_ID: self.user_id or "",
            },
        )
        self._active_spans[f"tool_{run_id_str}"] = span

        logger.debug(f"🔧 Tool started | name: {tool_name} | run_id: {run_id_str[:8]}")

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool completes execution."""
        run_id_str = str(run_id)
        start_time = self._tool_start_times.pop(run_id_str, time.time())
        latency = time.time() - start_time

        # Get tool name from span attributes
        tool_name = "unknown"
        span_key = f"tool_{run_id_str}"
        if span_key in self._active_spans:
            span = self._active_spans.pop(span_key)
            tool_name = span.attributes.get(SpanAttributes.TOOL_NAME, "unknown")
            span.set_attribute(SpanAttributes.TOOL_OUTPUT, str(output)[:500])
            span.set_attribute(SpanAttributes.TOOL_SUCCESS, True)
            span.set_attribute(SpanAttributes.TOOL_LATENCY_MS, latency * 1000)
            span.end()

        # Record Prometheus metrics using centralized metrics module
        if _PROMETHEUS_AVAILABLE:
            metrics = get_metrics()
            metrics.record_tool_call(tool_name=tool_name, status="success", latency=latency)

        logger.debug(
            f"🔧 Tool completed | name: {tool_name} | latency: {latency:.3f}s"
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool encounters an error."""
        run_id_str = str(run_id)
        start_time = self._tool_start_times.pop(run_id_str, time.time())
        latency = time.time() - start_time

        tool_name = "unknown"
        span_key = f"tool_{run_id_str}"
        if span_key in self._active_spans:
            span = self._active_spans.pop(span_key)
            tool_name = span.attributes.get(SpanAttributes.TOOL_NAME, "unknown")
            span.set_attribute(SpanAttributes.TOOL_SUCCESS, False)
            span.set_attribute(SpanAttributes.ERROR_TYPE, type(error).__name__)
            span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(error))
            span.record_exception(error)
            span.end()

        # Record Prometheus metrics using centralized metrics module
        if _PROMETHEUS_AVAILABLE:
            metrics = get_metrics()
            metrics.record_tool_call(tool_name=tool_name, status="error", latency=latency)

        logger.error(
            f"🔧 Tool failed | name: {tool_name} | error: {error} | latency: {latency:.3f}s"
        )

    # ==================== Chain Callbacks ====================

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts processing."""
        run_id_str = str(run_id)
        self._chain_start_times[run_id_str] = time.time()

        chain_type = serialized.get("id", ["unknown"])[-1]

        # Create span for chain
        tracer = get_tracer()
        span = tracer.start_span(
            SpanNames.LANGCHAIN_CHAIN.format(name=chain_type),
            attributes={
                SpanAttributes.LANGCHAIN_CHAIN: chain_type,
                SpanAttributes.REQUEST_USER_ID: self.user_id or "",
                SpanAttributes.REQUEST_SESSION_ID: self.session_id or "",
                "chain.input_keys": list(inputs.keys()),
            },
        )
        self._active_spans[f"chain_{run_id_str}"] = span

        logger.debug(
            f"⛓️ Chain started | type: {chain_type} | run_id: {run_id_str[:8]}"
        )

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain completes processing."""
        run_id_str = str(run_id)
        start_time = self._chain_start_times.pop(run_id_str, time.time())
        latency = time.time() - start_time

        chain_type = "unknown"
        span_key = f"chain_{run_id_str}"
        if span_key in self._active_spans:
            span = self._active_spans.pop(span_key)
            chain_type = span.attributes.get(SpanAttributes.LANGCHAIN_CHAIN, "unknown")
            span.set_attribute("chain.output_keys", list(outputs.keys()))
            span.end()

        # Record Prometheus metrics using centralized metrics module
        # Note: Using agent_operation for chain tracking since chains are similar
        if _PROMETHEUS_AVAILABLE:
            metrics = get_metrics()
            metrics.record_agent_operation(
                agent_type=f"chain_{chain_type}",
                operation="execute",
                status="success",
                latency=latency,
            )

        logger.debug(
            f"⛓️ Chain completed | type: {chain_type} | latency: {latency:.3f}s"
        )

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain encounters an error."""
        run_id_str = str(run_id)
        start_time = self._chain_start_times.pop(run_id_str, time.time())
        latency = time.time() - start_time

        chain_type = "unknown"
        span_key = f"chain_{run_id_str}"
        if span_key in self._active_spans:
            span = self._active_spans.pop(span_key)
            chain_type = span.attributes.get(SpanAttributes.LANGCHAIN_CHAIN, "unknown")
            span.set_attribute(SpanAttributes.ERROR_TYPE, type(error).__name__)
            span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(error))
            span.record_exception(error)
            span.end()

        # Record Prometheus metrics using centralized metrics module
        # Note: Using agent_operation for chain tracking since chains are similar
        if _PROMETHEUS_AVAILABLE:
            metrics = get_metrics()
            metrics.record_agent_operation(
                agent_type=f"chain_{chain_type}",
                operation="execute",
                status="error",
                latency=latency,
            )

        logger.error(
            f"⛓️ Chain failed | type: {chain_type} | error: {error} | latency: {latency:.3f}s"
        )

    # ==================== Agent Callbacks ====================

    async def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent takes an action."""
        tool = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", {})

        logger.debug(
            f"🎯 Agent action | tool: {tool} | input: {str(tool_input)[:100]}..."
        )

    async def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent finishes."""
        output = getattr(finish, "return_values", {})

        logger.debug(f"🎯 Agent finished | output_keys: {list(output.keys())}")

    # ==================== Helper Methods ====================

    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider name from model identifier."""
        model_lower = model.lower()
        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "google" in model_lower:
            return "google"
        elif "llama" in model_lower:
            return "meta"
        else:
            return "unknown"


def create_langchain_callbacks(
    run_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> list[ObservabilityCallbackHandler]:
    """Create LangChain callbacks for observability.

    Factory function to create callback handlers with optional context.

    Args:
        run_id: Unique identifier for this run
        user_id: User identifier for attribution
        session_id: Session identifier for grouping

    Returns:
        List containing ObservabilityCallbackHandler

    Example:
        >>> callbacks = create_langchain_callbacks(
        ...     run_id="req-123",
        ...     user_id="user-456",
        ...     session_id="session-789"
        ... )
        >>> result = await chain.ainvoke(input, config={"callbacks": callbacks})
    """
    return [
        ObservabilityCallbackHandler(
            run_id=run_id,
            user_id=user_id,
            session_id=session_id,
        )
    ]
