"""Enhanced Prometheus Metrics with SLO Support for Weather AI Agent.

Level 9: Production-grade metrics with:
- SLO-aligned metrics (availability, latency, correctness)
- Multiprocess-safe registry
- Custom collectors for agent metrics
- Cost tracking
- Error budget calculations
- Exemplar support for trace correlation (Level 8)

SLO Targets:
- Availability: 99.9% (3 nines)
- Latency P95: <2s for standard queries, <5s for complex
- Correctness: 95% for weather data accuracy
- Safety: 100% for hurricane category validation

Signal Correlation (Level 8):
- Metrics → Traces: Exemplars with trace_id on all histograms
- Traces → Metrics: Tempo tracesToMetrics configuration
- Logs → Traces: Loki derivedFields for trace_id extraction
- Traces → Logs: Tempo tracesToLogsV2 configuration

Usage:
    >>> from backend.src.observability.metrics import get_metrics
    >>> metrics = get_metrics()
    >>> metrics.record_request(tier="standard", status="success", latency=0.5)

Exemplar Usage:
    >>> from backend.src.observability.metrics import observe_with_exemplar, get_exemplar_labels
    >>> exemplar = get_exemplar_labels()  # Gets current trace_id
    >>> histogram.observe(value, exemplar)  # Records with trace correlation
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import trace
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
)

# ============================================================================
# Exemplar Support for Trace Correlation (Level 8)
# ============================================================================
# Prometheus exemplars allow linking metrics to traces by including trace_id
# in histogram observations. This enables clicking on metric data points in
# Grafana and jumping directly to the corresponding trace in Tempo.
#
# Requirements:
#   - prometheus-client >= 0.16.0 (supports exemplars)
#   - Prometheus --enable-feature=exemplar-storage
#   - Grafana datasource with exemplarTraceIdDestinations
# ============================================================================


def get_exemplar_labels() -> dict[str, str]:
    """Get current trace context as exemplar labels.

    Extracts the current OpenTelemetry trace context and formats it
    for use as Prometheus exemplar labels. The trace_id is formatted
    as a 32-character lowercase hex string (W3C TraceContext format).

    Note: Prometheus exemplars typically only support trace_id.
    span_id can be added but may not render in all UIs.

    Returns:
        dict[str, str]: Exemplar labels with trace_id, empty if no valid context

    Example:
        >>> exemplar = get_exemplar_labels()
        >>> if exemplar:
        ...     histogram.observe(1.5, exemplar)
        ... else:
        ...     histogram.observe(1.5)
    """
    try:
        span = trace.get_current_span()
        ctx = span.get_span_context()

        if ctx.is_valid:
            return {
                "trace_id": format(ctx.trace_id, "032x"),
            }
    except Exception:
        # Fail silently - exemplars are optional enhancement
        pass
    return {}


def observe_with_exemplar(
    histogram: Histogram,
    value: float,
    labels: dict[str, str] | None = None,
) -> None:
    """Observe a histogram value with exemplar containing trace context.

    Records a histogram observation with automatic trace correlation.
    If a valid OpenTelemetry trace context exists, the trace_id is
    included as an exemplar, enabling metrics→traces correlation in Grafana.

    Args:
        histogram: Prometheus Histogram (with labels already applied if needed)
        value: Value to observe (e.g., latency in seconds)
        labels: Optional additional labels for logging/debugging

    Example:
        >>> # Without labels (direct histogram)
        >>> observe_with_exemplar(REQUEST_LATENCY, 1.5)

        >>> # With labels (labeled histogram)
        >>> observe_with_exemplar(
        ...     REQUEST_LATENCY.labels(tier="standard", endpoint="/query"),
        ...     1.5,
        ...     {"tier": "standard", "endpoint": "/query"}
        ... )

    Note:
        The `labels` parameter is for debugging/logging only and does not
        affect the histogram observation. Use histogram.labels() before
        passing to this function for labeled metrics.
    """
    exemplar = get_exemplar_labels()
    if exemplar:
        histogram.observe(value, exemplar)
    else:
        histogram.observe(value)

# Check if running in multiprocess mode (e.g., Gunicorn)
_MULTIPROCESS_MODE = "prometheus_multiproc_dir" in os.environ


class MetricsRegistry:
    """Centralized metrics registry for Weather AI Agent.

    Provides thread-safe access to Prometheus metrics with SLO tracking
    and multiprocess support. Uses lazy initialization to avoid duplicate
    registration issues.
    """

    _instance: MetricsRegistry | None = None
    _metrics_initialized: bool = False

    def __init__(self, registry: CollectorRegistry | None = None):
        """Initialize metrics registry.

        Args:
            registry: Optional custom registry for testing
        """
        self._registry = registry or REGISTRY
        self._initialized = False
        self._slo_targets = {
            "availability": 0.999,  # 99.9%
            "latency_p95_standard": 2.0,  # 2 seconds
            "latency_p95_complex": 5.0,  # 5 seconds
            "correctness": 0.95,  # 95%
            "safety": 1.0,  # 100%
            "safety_hurricane_validation": 1.0,  # 100% for hurricane validation
            "cache_hit_rate": 0.80,  # 80%
            "cost_per_query": 0.10,  # $0.10 per query
            "token_reduction": 0.40,  # 40% token reduction
        }

        # Metrics are lazily initialized
        self._request_total = None
        self._request_latency = None
        self._slo_error_budget = None
        self._slo_violations = None
        self._llm_request_total = None
        self._llm_tokens = None
        self._llm_latency = None
        self._llm_cost = None
        self._tool_invocations = None
        self._tool_latency = None
        self._mcp_requests = None
        self._mcp_latency = None
        self._mcp_data_freshness = None
        self._mcp_health_status = None
        self._cache_operations = None
        self._cache_latency = None
        self._cache_size = None
        self._cache_hit_rate = None
        self._agent_invocations = None
        self._agent_latency = None
        self._active_agents = None
        self._guardrail_checks = None
        self._guardrail_violations = None
        self._hurricane_validations = None
        self._pii_detections = None
        self._context_optimizations = None
        self._context_token_reduction = None
        self._context_optimization_latency = None
        self._service_info = None

    def _ensure_metrics_initialized(self) -> None:
        """Lazily initialize all metrics on first use."""
        if MetricsRegistry._metrics_initialized and self._registry is REGISTRY:
            return

        # Request metrics
        self._request_total = self._get_or_create_counter(
            "weather_ai_requests_total",
            "Total requests by tier and status",
            ["tier", "status", "endpoint"],
        )

        self._request_latency = self._get_or_create_histogram(
            "weather_ai_request_latency_seconds",
            "Request latency distribution",
            ["tier", "endpoint"],
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0],
        )

        # SLO metrics
        self._slo_error_budget = self._get_or_create_gauge(
            "weather_ai_slo_error_budget_remaining",
            "Remaining error budget for SLO",
            ["slo_name"],
        )

        self._slo_violations = self._get_or_create_counter(
            "weather_ai_slo_violations_total",
            "Total SLO violations",
            ["slo_name", "window"],
        )

        # LLM metrics
        self._llm_request_total = self._get_or_create_counter(
            "weather_ai_llm_requests_total",
            "Total LLM API requests",
            ["provider", "model", "status"],
        )

        self._llm_tokens = self._get_or_create_counter(
            "weather_ai_llm_tokens_total",
            "Total tokens processed",
            ["provider", "model", "direction"],
        )

        self._llm_latency = self._get_or_create_histogram(
            "weather_ai_llm_latency_seconds",
            "LLM API latency",
            ["provider", "model"],
            buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0],
        )

        self._llm_cost = self._get_or_create_counter(
            "weather_ai_llm_cost_dollars_total",
            "Total LLM API cost in dollars",
            ["provider", "model"],
        )

        # Tool metrics
        self._tool_invocations = self._get_or_create_counter(
            "weather_ai_tool_invocations_total",
            "Total tool invocations",
            ["tool_name", "status"],
        )

        self._tool_latency = self._get_or_create_histogram(
            "weather_ai_tool_latency_seconds",
            "Tool execution latency",
            ["tool_name"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # MCP metrics
        self._mcp_requests = self._get_or_create_counter(
            "weather_ai_mcp_requests_total",
            "Total MCP server requests",
            ["server", "operation", "status"],
        )

        self._mcp_latency = self._get_or_create_histogram(
            "weather_ai_mcp_latency_seconds",
            "MCP request latency",
            ["server", "operation"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        )

        self._mcp_data_freshness = self._get_or_create_gauge(
            "weather_ai_mcp_data_freshness_seconds",
            "Age of MCP data in seconds",
            ["server", "data_type"],
        )

        self._mcp_health_status = self._get_or_create_gauge(
            "weather_ai_mcp_health_status",
            "MCP server health status (1=healthy, 0=unhealthy)",
            ["server"],
        )

        # Cache metrics
        self._cache_operations = self._get_or_create_counter(
            "weather_ai_cache_operations_total",
            "Total cache operations",
            ["layer", "operation", "result"],
        )

        self._cache_latency = self._get_or_create_histogram(
            "weather_ai_cache_latency_seconds",
            "Cache operation latency",
            ["layer", "operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
        )

        self._cache_size = self._get_or_create_gauge(
            "weather_ai_cache_size_entries",
            "Current cache size in entries",
            ["layer"],
        )

        self._cache_hit_rate = self._get_or_create_gauge(
            "weather_ai_cache_hit_rate",
            "Cache hit rate (0-1)",
            ["layer"],
        )

        # Agent metrics
        self._agent_invocations = self._get_or_create_counter(
            "weather_ai_agent_invocations_total",
            "Total agent invocations",
            ["agent_name", "level", "status"],
        )

        self._agent_latency = self._get_or_create_histogram(
            "weather_ai_agent_latency_seconds",
            "Agent execution latency",
            ["agent_name", "level"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
        )

        self._active_agents = self._get_or_create_gauge(
            "weather_ai_active_agents",
            "Number of currently active agent executions",
            ["level"],
        )

        # Safety metrics
        self._guardrail_checks = self._get_or_create_counter(
            "weather_ai_guardrail_checks_total",
            "Total guardrail checks",
            ["guardrail", "result"],
        )

        self._guardrail_violations = self._get_or_create_counter(
            "weather_ai_guardrail_violations_total",
            "Total guardrail violations (safety-critical)",
            ["guardrail", "severity"],
        )

        self._hurricane_validations = self._get_or_create_counter(
            "weather_ai_hurricane_validations_total",
            "Total hurricane data validations",
            ["category", "result"],
        )

        self._pii_detections = self._get_or_create_counter(
            "weather_ai_pii_detections_total",
            "Total PII detections",
            ["pii_type", "action"],
        )

        # Context optimization metrics
        self._context_optimizations = self._get_or_create_counter(
            "weather_ai_context_optimizations_total",
            "Total context window optimizations",
            ["query_type", "status"],
        )

        self._context_token_reduction = self._get_or_create_histogram(
            "weather_ai_context_token_reduction_percent",
            "Context window token reduction percentage",
            ["query_type"],
            buckets=[10, 20, 30, 40, 50, 60, 70, 80],
        )

        self._context_optimization_latency = self._get_or_create_histogram(
            "weather_ai_context_optimization_latency_seconds",
            "Context optimization processing time",
            ["query_type"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3],
        )

        # Service info
        self._service_info = self._get_or_create_info(
            "weather_ai_service",
            "Weather AI Agent service information",
        )

        if self._registry is REGISTRY:
            MetricsRegistry._metrics_initialized = True

    def _get_or_create_counter(
        self, name: str, documentation: str, labelnames: list[str]
    ) -> Counter:
        """Get existing counter or create new one."""
        try:
            return Counter(
                name, documentation, labelnames, registry=self._registry
            )
        except ValueError:
            # Already registered, get from registry
            return self._registry._names_to_collectors.get(name)

    def _get_or_create_histogram(
        self,
        name: str,
        documentation: str,
        labelnames: list[str],
        buckets: list[float] | None = None,
    ) -> Histogram:
        """Get existing histogram or create new one."""
        try:
            kwargs = {"registry": self._registry}
            if buckets:
                kwargs["buckets"] = buckets
            return Histogram(name, documentation, labelnames, **kwargs)
        except ValueError:
            return self._registry._names_to_collectors.get(name)

    def _get_or_create_gauge(
        self, name: str, documentation: str, labelnames: list[str]
    ) -> Gauge:
        """Get existing gauge or create new one."""
        try:
            return Gauge(
                name, documentation, labelnames, registry=self._registry
            )
        except ValueError:
            return self._registry._names_to_collectors.get(name)

    def _get_or_create_info(self, name: str, documentation: str) -> Info:
        """Get existing info or create new one."""
        try:
            return Info(name, documentation, registry=self._registry)
        except ValueError:
            return self._registry._names_to_collectors.get(name)

    @classmethod
    def get_instance(cls) -> MetricsRegistry:
        """Get singleton metrics registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset singleton for testing purposes."""
        cls._instance = None
        cls._metrics_initialized = False

    def initialize(
        self, version: str = "1.6.0", environment: str = "production"
    ) -> None:
        """Initialize service info metrics.

        Args:
            version: Service version
            environment: Deployment environment
        """
        self._ensure_metrics_initialized()

        if self._initialized:
            return

        if self._service_info:
            self._service_info.info(
                {
                    "version": version,
                    "environment": environment,
                    "level": "L9",
                }
            )
        self._initialized = True

    # ==================== Request Metrics ====================

    def record_request(
        self,
        tier: str,
        status: str,
        latency: float,
        endpoint: str | None = "/weather/query",
    ) -> None:
        """Record a request with SLO tracking.

        Args:
            tier: Query tier (simple/standard/complex/emergency)
            status: Result status (success/error)
            latency: Request latency in seconds
            endpoint: API endpoint
        """
        self._ensure_metrics_initialized()

        endpoint = endpoint or "/weather/query"

        if self._request_total:
            self._request_total.labels(
                tier=tier, status=status, endpoint=endpoint
            ).inc()
        if self._request_latency:
            # Use exemplar for trace correlation (Level 8)
            observe_with_exemplar(
                self._request_latency.labels(tier=tier, endpoint=endpoint),
                latency,
            )

        # Check SLO violations
        if status == "error" and self._slo_violations:
            self._slo_violations.labels(
                slo_name="availability", window="5m"
            ).inc()

        latency_target = (
            self._slo_targets["latency_p95_complex"]
            if tier in ("complex", "emergency")
            else self._slo_targets["latency_p95_standard"]
        )
        if latency > latency_target and self._slo_violations:
            self._slo_violations.labels(
                slo_name=f"latency_{tier}", window="5m"
            ).inc()

    @contextmanager
    def track_request(
        self, tier: str, endpoint: str = "/weather/query"
    ) -> Generator[None]:
        """Context manager for tracking request metrics.

        Args:
            tier: Query tier
            endpoint: API endpoint

        Example:
            >>> with metrics.track_request("standard") as m:
            ...     result = await process_query()
        """
        start = time.time()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            latency = time.time() - start
            self.record_request(tier, status, latency, endpoint)

    # ==================== LLM Metrics ====================

    def record_llm_call(
        self,
        provider: str,
        model: str,
        status: str,
        latency: float,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Record LLM API call metrics.

        Args:
            provider: LLM provider (openai/anthropic)
            model: Model name
            status: Call status (success/error)
            latency: Call latency in seconds
            input_tokens: Input token count
            output_tokens: Output token count
            cost: Estimated cost in USD
        """
        self._ensure_metrics_initialized()

        if self._llm_request_total:
            self._llm_request_total.labels(
                provider=provider, model=model, status=status
            ).inc()
        if self._llm_latency:
            # Use exemplar for trace correlation (Level 8)
            observe_with_exemplar(
                self._llm_latency.labels(provider=provider, model=model),
                latency,
            )
        if self._llm_tokens:
            self._llm_tokens.labels(
                provider=provider, model=model, direction="input"
            ).inc(input_tokens)
            self._llm_tokens.labels(
                provider=provider, model=model, direction="output"
            ).inc(output_tokens)
        if self._llm_cost:
            self._llm_cost.labels(provider=provider, model=model).inc(cost)

    # ==================== Tool Metrics ====================

    def record_tool_call(
        self,
        tool_name: str,
        status: str,
        latency: float,
    ) -> None:
        """Record tool invocation metrics.

        Args:
            tool_name: Name of the tool
            status: Call status (success/error)
            latency: Call latency in seconds
        """
        self._ensure_metrics_initialized()

        if self._tool_invocations:
            self._tool_invocations.labels(
                tool_name=tool_name, status=status
            ).inc()
        if self._tool_latency:
            # Use exemplar for trace correlation (Level 8)
            observe_with_exemplar(
                self._tool_latency.labels(tool_name=tool_name),
                latency,
            )

    # ==================== MCP Metrics ====================

    def record_mcp_call(
        self,
        server: str,
        operation: str,
        status: str,
        latency: float,
    ) -> None:
        """Record MCP server call metrics.

        Args:
            server: MCP server name (weather/hurricane)
            operation: Operation type (get_weather/get_forecast)
            status: Call status (success/error)
            latency: Call latency in seconds
        """
        self._ensure_metrics_initialized()

        if self._mcp_requests:
            self._mcp_requests.labels(
                server=server, operation=operation, status=status
            ).inc()
        if self._mcp_latency:
            # Use exemplar for trace correlation (Level 8)
            observe_with_exemplar(
                self._mcp_latency.labels(server=server, operation=operation),
                latency,
            )

    def set_mcp_data_freshness(
        self,
        server: str,
        data_type: str,
        age_seconds: float,
    ) -> None:
        """Set MCP data freshness metric.

        Args:
            server: MCP server name
            data_type: Type of data (weather/hurricane)
            age_seconds: Age of data in seconds
        """
        self._ensure_metrics_initialized()

        if self._mcp_data_freshness:
            self._mcp_data_freshness.labels(
                server=server, data_type=data_type
            ).set(age_seconds)

    def set_mcp_health(self, server: str, healthy: bool) -> None:
        """Set MCP server health status.

        Args:
            server: MCP server name
            healthy: Whether server is healthy
        """
        self._ensure_metrics_initialized()

        if self._mcp_health_status:
            self._mcp_health_status.labels(server=server).set(1 if healthy else 0)

    # ==================== Cache Metrics ====================

    def record_cache_operation(
        self,
        cache_level: str,
        operation: str,
        hit: bool,
        latency: float,
    ) -> None:
        """Record cache operation metrics.

        Args:
            cache_level: Cache layer (L1/L2/L3)
            operation: Operation type (get/set)
            hit: Whether it was a cache hit
            latency: Operation latency in seconds
        """
        self._ensure_metrics_initialized()

        result = "hit" if hit else "miss"
        if self._cache_operations:
            self._cache_operations.labels(
                layer=cache_level, operation=operation, result=result
            ).inc()
        if self._cache_latency:
            # Use exemplar for trace correlation (Level 8)
            observe_with_exemplar(
                self._cache_latency.labels(layer=cache_level, operation=operation),
                latency,
            )

    def set_cache_size(self, layer: str, size: int) -> None:
        """Set cache size metric.

        Args:
            layer: Cache layer
            size: Current cache size in entries
        """
        self._ensure_metrics_initialized()

        if self._cache_size:
            self._cache_size.labels(layer=layer).set(size)

    def set_cache_hit_rate(self, layer: str, hit_rate: float) -> None:
        """Set cache hit rate metric.

        Args:
            layer: Cache layer
            hit_rate: Hit rate (0-1)
        """
        self._ensure_metrics_initialized()

        if self._cache_hit_rate:
            self._cache_hit_rate.labels(layer=layer).set(hit_rate)

    # ==================== Agent Metrics ====================

    def record_agent_operation(
        self,
        agent_type: str,
        operation: str,
        status: str,
        latency: float,
    ) -> None:
        """Record agent operation metrics.

        Args:
            agent_type: Type of agent
            operation: Operation type
            status: Operation status (success/error)
            latency: Operation latency in seconds
        """
        self._ensure_metrics_initialized()

        if self._agent_invocations:
            self._agent_invocations.labels(
                agent_name=agent_type, level="l4", status=status
            ).inc()
        if self._agent_latency:
            # Use exemplar for trace correlation (Level 8)
            observe_with_exemplar(
                self._agent_latency.labels(agent_name=agent_type, level="l4"),
                latency,
            )

    def record_agent_invocation(
        self,
        agent_name: str,
        level: str,
        status: str,
        latency: float,
    ) -> None:
        """Record agent invocation metrics.

        Args:
            agent_name: Name of the agent
            level: Agent level (l4a/l4b/l4c)
            status: Invocation status (success/error)
            latency: Invocation latency in seconds
        """
        self._ensure_metrics_initialized()

        if self._agent_invocations:
            self._agent_invocations.labels(
                agent_name=agent_name, level=level, status=status
            ).inc()
        if self._agent_latency:
            # Use exemplar for trace correlation (Level 8)
            observe_with_exemplar(
                self._agent_latency.labels(agent_name=agent_name, level=level),
                latency,
            )

    def set_active_agents(self, level: str, count: int) -> None:
        """Set active agents count.

        Args:
            level: Agent level
            count: Number of active agents
        """
        self._ensure_metrics_initialized()

        if self._active_agents:
            self._active_agents.labels(level=level).set(count)

    # ==================== Safety Metrics ====================

    def record_guardrail_check(
        self,
        guardrail: str,
        passed: bool,
        severity: str = "warning",
    ) -> None:
        """Record guardrail check result.

        Args:
            guardrail: Guardrail name
            passed: Whether check passed
            severity: Violation severity (warning/critical)
        """
        self._ensure_metrics_initialized()

        result = "passed" if passed else "blocked"
        if self._guardrail_checks:
            self._guardrail_checks.labels(
                guardrail=guardrail, result=result
            ).inc()

        if not passed:
            if self._guardrail_violations:
                self._guardrail_violations.labels(
                    guardrail=guardrail, severity=severity
                ).inc()
            if self._slo_violations:
                self._slo_violations.labels(slo_name="safety", window="1h").inc()

    def record_hurricane_validation(
        self,
        category: int,
        valid: bool,
    ) -> None:
        """Record hurricane data validation.

        Args:
            category: Hurricane category (1-5)
            valid: Whether validation passed
        """
        self._ensure_metrics_initialized()

        result = "passed" if valid else "failed"
        if self._hurricane_validations:
            self._hurricane_validations.labels(
                category=str(category), result=result
            ).inc()

        if not valid and self._guardrail_violations:
            self._guardrail_violations.labels(
                guardrail="hurricane_saffir_simpson", severity="critical"
            ).inc()

    def record_pii_detection(
        self,
        pii_type: str,
        action: str = "redacted",
    ) -> None:
        """Record PII detection.

        Args:
            pii_type: Type of PII (ssn/credit_card/phone/email)
            action: Action taken (redacted/blocked)
        """
        self._ensure_metrics_initialized()

        if self._pii_detections:
            self._pii_detections.labels(pii_type=pii_type, action=action).inc()
        if self._guardrail_violations:
            self._guardrail_violations.labels(
                guardrail="pii_detection", severity="critical"
            ).inc()

    # ==================== Context Optimization Metrics ====================

    def record_context_optimization(
        self,
        query_type: str,
        status: str,
        reduction_pct: float,
        latency: float,
    ) -> None:
        """Record context optimization metrics.

        Args:
            query_type: Query type (SIMPLE/STANDARD/COMPLEX/EMERGENCY)
            status: Optimization status (success/error)
            reduction_pct: Token reduction percentage
            latency: Optimization latency in seconds
        """
        self._ensure_metrics_initialized()

        if self._context_optimizations:
            self._context_optimizations.labels(
                query_type=query_type, status=status
            ).inc()
        if self._context_token_reduction:
            # Use exemplar for trace correlation (Level 8)
            observe_with_exemplar(
                self._context_token_reduction.labels(query_type=query_type),
                reduction_pct,
            )
        if self._context_optimization_latency:
            # Use exemplar for trace correlation (Level 8)
            observe_with_exemplar(
                self._context_optimization_latency.labels(query_type=query_type),
                latency,
            )

    # ==================== SLO Methods ====================

    def update_error_budget(self, slo_name: str, remaining: float) -> None:
        """Update error budget for an SLO.

        Args:
            slo_name: Name of the SLO
            remaining: Remaining error budget (0-1)
        """
        self._ensure_metrics_initialized()

        if self._slo_error_budget:
            self._slo_error_budget.labels(slo_name=slo_name).set(remaining)

    def get_slo_target(self, slo_name: str) -> float:
        """Get SLO target value.

        Args:
            slo_name: Name of the SLO

        Returns:
            Target value for the SLO
        """
        return self._slo_targets.get(slo_name, 0.99)


# Singleton instance
_metrics_instance: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    """Get global metrics registry instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsRegistry.get_instance()
    return _metrics_instance


def reset_metrics_for_testing() -> None:
    """Reset metrics singleton for testing."""
    global _metrics_instance
    _metrics_instance = None
    MetricsRegistry.reset_for_testing()
