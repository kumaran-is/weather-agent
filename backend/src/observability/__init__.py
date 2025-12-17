"""Observability module for Weather AI Agent Service.

Level 9: Comprehensive Observability Integration
- OpenTelemetry distributed tracing
- LangChain/LangGraph auto-instrumentation callbacks
- Enhanced Prometheus metrics with SLO support
- Structured logging with trace context

Components:
- tracing: OpenTelemetry span management
- callbacks: LangChain callback handlers for auto-instrumentation
- metrics: Enhanced Prometheus metrics with SLO support
- logging: Structured logging with trace context

Usage:
    # Initialize at application startup
    >>> from backend.src.observability import TracingManager, configure_structured_logging
    >>> tracing = TracingManager(service_name="weather-ai-agent")
    >>> tracing.initialize()
    >>> configure_structured_logging(level="INFO", json_format=True)

    # Create callbacks for LangChain
    >>> from backend.src.observability import create_langchain_callbacks
    >>> callbacks = create_langchain_callbacks(user_id="123", session_id="abc")
    >>> result = await chain.ainvoke(input, config={"callbacks": callbacks})

    # Record metrics
    >>> from backend.src.observability import get_metrics
    >>> metrics = get_metrics()
    >>> metrics.record_request(tier="standard", status="success", latency=0.5)
"""

from backend.src.observability.tracing import (
    TracingManager,
    get_tracer,
    create_span,
    inject_trace_context,
    extract_trace_context,
    SpanAttributes,
    SpanNames,
)
from backend.src.observability.callbacks import (
    ObservabilityCallbackHandler,
    create_langchain_callbacks,
)
from backend.src.observability.logging import (
    get_structured_logger,
    add_trace_context,
    set_request_context,
    clear_request_context,
    configure_structured_logging,
)
from backend.src.observability.metrics import (
    MetricsRegistry,
    get_metrics,
)

__all__ = [
    # Tracing
    "TracingManager",
    "get_tracer",
    "create_span",
    "inject_trace_context",
    "extract_trace_context",
    "SpanAttributes",
    "SpanNames",
    # Callbacks
    "ObservabilityCallbackHandler",
    "create_langchain_callbacks",
    # Logging
    "get_structured_logger",
    "add_trace_context",
    "set_request_context",
    "clear_request_context",
    "configure_structured_logging",
    # Metrics
    "MetricsRegistry",
    "get_metrics",
]
