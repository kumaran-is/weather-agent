"""OpenTelemetry Tracing Infrastructure for Weather AI Agent.

Level 9: Distributed tracing with span hierarchy for:
- HTTP requests (FastAPI middleware)
- LangGraph workflow execution
- Agent invocations
- Tool calls
- MCP server interactions
- Cache operations

Span Hierarchy:
```
http.request (parent)
└── langgraph.workflow
    ├── langgraph.node (supervisor)
    │   └── langchain.llm
    ├── langgraph.node (agent)
    │   ├── langchain.tool (weather_api)
    │   │   └── mcp.call
    │   └── langchain.tool (rag_search)
    │       └── vectorstore.query
    └── langgraph.node (synthesizer)
        └── langchain.llm
```

Usage:
    >>> from backend.src.observability.tracing import get_tracer, create_span
    >>> tracer = get_tracer()
    >>> with create_span("langgraph.workflow", {"workflow": "weather"}) as span:
    ...     result = await workflow.ainvoke(input)
    ...     span.set_attribute("output.tokens", len(result))
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

# Global tracer provider (singleton)
_tracer_provider: TracerProvider | None = None
_propagator = TraceContextTextMapPropagator()


class TracingManager:
    """Manages OpenTelemetry tracing configuration and lifecycle.

    Provides centralized configuration for distributed tracing with support for:
    - OTLP exporter (for Jaeger, Tempo, etc.)
    - Console exporter (for development/debugging)
    - W3C TraceContext propagation
    - Batch span processing

    Attributes:
        service_name: Name of the service for trace identification
        environment: Deployment environment (development, staging, production)
        otlp_endpoint: OTLP collector endpoint URL
    """

    def __init__(
        self,
        service_name: str = "weather-ai-agent",
        environment: str | None = None,
        otlp_endpoint: str | None = None,
        enable_console_export: bool = False,
    ):
        """Initialize tracing manager.

        Args:
            service_name: Name of the service
            environment: Deployment environment (default: from ENVIRONMENT env var)
            otlp_endpoint: OTLP exporter endpoint (default: from OTEL_EXPORTER_OTLP_ENDPOINT)
            enable_console_export: Enable console span export for debugging
        """
        self.service_name = service_name
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.otlp_endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        self.enable_console_export = enable_console_export
        self._initialized = False

    def initialize(self) -> None:
        """Initialize OpenTelemetry tracer provider.

        Sets up the global tracer provider with appropriate exporters and processors.
        This should be called once at application startup.
        """
        global _tracer_provider

        if self._initialized:
            logger.warning("Tracing already initialized, skipping...")
            return

        # Create resource with service metadata
        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: os.getenv("SERVICE_VERSION", "1.6.0"),
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.environment,
                "service.namespace": "weather-ai",
                "service.instance.id": os.getenv("HOSTNAME", "local"),
            }
        )

        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)

        # Add OTLP exporter (for production - Jaeger, Tempo, etc.)
        if os.getenv("OTEL_TRACES_ENABLED", "true").lower() == "true":
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint,
                    insecure=True,  # Use TLS in production
                )
                _tracer_provider.add_span_processor(
                    BatchSpanProcessor(
                        otlp_exporter,
                        max_queue_size=2048,
                        max_export_batch_size=512,
                        schedule_delay_millis=5000,
                    )
                )
                logger.info(
                    f"🔍 OpenTelemetry OTLP exporter initialized | endpoint: {self.otlp_endpoint}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OTLP exporter: {e}")

        # Add console exporter (for development/debugging)
        if self.enable_console_export:
            _tracer_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
            logger.info("🔍 OpenTelemetry console exporter enabled")

        # Set global tracer provider
        trace.set_tracer_provider(_tracer_provider)
        self._initialized = True

        logger.info(
            f"🔍 OpenTelemetry tracing initialized | "
            f"service: {self.service_name} | "
            f"environment: {self.environment}"
        )

    def shutdown(self) -> None:
        """Shutdown tracer provider and flush pending spans."""
        global _tracer_provider

        if _tracer_provider:
            _tracer_provider.shutdown()
            logger.info("🔍 OpenTelemetry tracing shutdown complete")
            self._initialized = False


@lru_cache(maxsize=1)
def get_tracer(name: str = "weather-ai-agent") -> trace.Tracer:
    """Get or create OpenTelemetry tracer.

    Args:
        name: Tracer name (typically service/module name)

    Returns:
        OpenTelemetry Tracer instance
    """
    provider = trace.get_tracer_provider()
    return provider.get_tracer(name, schema_url="https://opentelemetry.io/schemas/1.21.0")


@contextmanager
def create_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    parent: Context | None = None,
) -> Generator[Span, None, None]:
    """Create and manage an OpenTelemetry span.

    Creates a span with automatic error recording and timing. Supports
    hierarchical span creation via parent context.

    Args:
        name: Span name (e.g., "langgraph.workflow", "langchain.tool")
        attributes: Initial span attributes
        kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
        parent: Optional parent context for distributed tracing

    Yields:
        Active span context

    Example:
        >>> with create_span("langgraph.node", {"node": "supervisor"}) as span:
        ...     result = await node_function()
        ...     span.set_attribute("result.status", "success")
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(
        name,
        context=parent,
        kind=kind,
        attributes=attributes or {},
    ) as span:
        try:
            yield span
        except Exception as e:
            # Record exception details
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


class DictSetter:
    """Simple setter for injecting values into dict-like carriers."""

    def set(self, carrier: dict, key: str, value: str) -> None:
        """Set a value in the carrier."""
        carrier[key] = value


class DictGetter:
    """Simple getter for extracting values from dict-like carriers."""

    def get(self, carrier: dict, key: str) -> str | None:
        """Get a value from the carrier."""
        return carrier.get(key)

    def keys(self, carrier: dict) -> list[str]:
        """Get all keys from the carrier."""
        return list(carrier.keys())


_dict_setter = DictSetter()
_dict_getter = DictGetter()


def inject_trace_context(carrier: dict[str, str]) -> dict[str, str]:
    """Inject trace context into carrier for propagation.

    Injects W3C TraceContext headers into the carrier dict for
    distributed tracing across service boundaries.

    Args:
        carrier: Mutable dict to inject trace headers into

    Returns:
        Carrier dict with trace headers

    Example:
        >>> headers = {}
        >>> inject_trace_context(headers)
        >>> # headers now contains 'traceparent' and 'tracestate'
    """
    inject(carrier, setter=_dict_setter)
    return carrier


def extract_trace_context(carrier: dict[str, str]) -> Context:
    """Extract trace context from carrier.

    Extracts W3C TraceContext from incoming request headers for
    distributed tracing continuity.

    Args:
        carrier: Dict containing trace headers

    Returns:
        OpenTelemetry Context with trace information

    Example:
        >>> context = extract_trace_context(request.headers)
        >>> with create_span("my.operation", parent=context) as span:
        ...     pass
    """
    return extract(carrier, getter=_dict_getter)


# Span attribute constants for consistency
class SpanAttributes:
    """Standard span attribute names for Weather AI Agent.

    Provides consistent attribute naming across all spans for
    easier querying and dashboard creation.
    """

    # Request attributes
    REQUEST_ID = "request.id"
    REQUEST_USER_ID = "request.user_id"
    REQUEST_SESSION_ID = "request.session_id"
    REQUEST_QUERY = "request.query"
    REQUEST_TIER = "request.tier"

    # LangGraph attributes
    LANGGRAPH_WORKFLOW = "langgraph.workflow"
    LANGGRAPH_NODE = "langgraph.node"
    LANGGRAPH_EDGE = "langgraph.edge"
    LANGGRAPH_STATE_KEYS = "langgraph.state.keys"
    LANGGRAPH_CHECKPOINT_ID = "langgraph.checkpoint.id"

    # LangChain attributes
    LANGCHAIN_MODEL = "langchain.model"
    LANGCHAIN_TOOL = "langchain.tool"
    LANGCHAIN_CHAIN = "langchain.chain"
    LANGCHAIN_PROMPT_TEMPLATE = "langchain.prompt.template"

    # LLM attributes
    LLM_PROVIDER = "llm.provider"
    LLM_MODEL = "llm.model"
    LLM_TEMPERATURE = "llm.temperature"
    LLM_INPUT_TOKENS = "llm.input_tokens"
    LLM_OUTPUT_TOKENS = "llm.output_tokens"
    LLM_TOTAL_TOKENS = "llm.total_tokens"
    LLM_COST_USD = "llm.cost.usd"
    LLM_LATENCY_MS = "llm.latency_ms"

    # Tool attributes
    TOOL_NAME = "tool.name"
    TOOL_INPUT = "tool.input"
    TOOL_OUTPUT = "tool.output"
    TOOL_SUCCESS = "tool.success"
    TOOL_LATENCY_MS = "tool.latency_ms"

    # MCP attributes
    MCP_SERVER = "mcp.server"
    MCP_OPERATION = "mcp.operation"
    MCP_RESOURCE = "mcp.resource"
    MCP_STATUS = "mcp.status"
    MCP_LATENCY_MS = "mcp.latency_ms"

    # Cache attributes
    CACHE_LAYER = "cache.layer"
    CACHE_HIT = "cache.hit"
    CACHE_KEY = "cache.key"
    CACHE_TTL = "cache.ttl"

    # Agent attributes
    AGENT_NAME = "agent.name"
    AGENT_TYPE = "agent.type"
    AGENT_LEVEL = "agent.level"
    AGENTS_INVOKED = "agents.invoked"

    # RAG attributes
    RAG_QUERY = "rag.query"
    RAG_DOCUMENTS_RETRIEVED = "rag.documents.retrieved"
    RAG_RELEVANCE_SCORE = "rag.relevance.score"

    # Context optimization attributes
    CONTEXT_QUERY_TYPE = "context.query_type"
    CONTEXT_ORIGINAL_TOKENS = "context.original_tokens"
    CONTEXT_OPTIMIZED_TOKENS = "context.optimized_tokens"
    CONTEXT_REDUCTION_PCT = "context.reduction_pct"

    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_STACK = "error.stack"


# Semantic conventions for span names
class SpanNames:
    """Standard span names for Weather AI Agent.

    Provides consistent span naming for easier trace visualization
    and querying.
    """

    # HTTP spans
    HTTP_REQUEST = "http.request"
    HTTP_RESPONSE = "http.response"

    # LangGraph spans
    LANGGRAPH_WORKFLOW = "langgraph.workflow"
    LANGGRAPH_NODE = "langgraph.node.{name}"
    LANGGRAPH_EDGE = "langgraph.edge.{from_node}.{to_node}"

    # LangChain spans
    LANGCHAIN_LLM = "langchain.llm"
    LANGCHAIN_TOOL = "langchain.tool.{name}"
    LANGCHAIN_CHAIN = "langchain.chain.{name}"

    # MCP spans
    MCP_CALL = "mcp.call.{server}.{operation}"

    # Cache spans
    CACHE_GET = "cache.get.{layer}"
    CACHE_SET = "cache.set.{layer}"

    # Agent spans
    AGENT_INVOKE = "agent.invoke.{name}"

    # RAG spans
    RAG_RETRIEVE = "rag.retrieve"
    RAG_RERANK = "rag.rerank"

    # Context optimization spans
    CONTEXT_OPTIMIZE = "context.optimize"
    CONTEXT_DETECT_TYPE = "context.detect_type"
