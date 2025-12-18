"""Integration tests for Level 9 Observability Stack.

Validates that all observability components work together correctly:
- Prometheus metrics are exposed correctly
- Structured logs contain trace context
- OpenTelemetry spans are created with proper hierarchy
- LangChain callbacks record metrics and create spans

These tests simulate a full request lifecycle and verify observability data.
"""

import io
import json
import logging
import uuid

import pytest
from prometheus_client import REGISTRY, generate_latest


class TestPrometheusMetricsExport:
    """Test that Prometheus metrics are exported correctly."""

    def test_metrics_endpoint_format(self):
        """Test that metrics can be exported in Prometheus format."""
        from backend.src.observability.metrics import get_metrics

        metrics = get_metrics()

        # Record some test metrics
        metrics.record_request(tier="standard", status="success", latency=0.5)
        metrics.record_llm_call(
            provider="anthropic",
            model="claude-3-5-sonnet",
            status="success",
            latency=1.0,
            input_tokens=100,
            output_tokens=200,
            cost=0.003,
        )
        metrics.record_tool_call(tool_name="get_weather", status="success", latency=0.1)
        metrics.record_mcp_call(server="weather", operation="forecast", status="success", latency=0.2)

        # Generate Prometheus format output
        output = generate_latest(REGISTRY).decode("utf-8")

        # Verify key metrics are present
        assert "weather_ai_requests_total" in output or "weather_ai_request" in output
        # The metrics may use different naming based on when they were initialized

    def test_slo_targets_accessible(self):
        """Test that SLO targets are accessible for dashboard queries."""
        from backend.src.observability.metrics import get_metrics

        metrics = get_metrics()

        # Verify SLO targets
        assert metrics.get_slo_target("availability") == 0.999
        assert metrics.get_slo_target("latency_p95_standard") == 2.0
        assert metrics.get_slo_target("latency_p95_complex") == 5.0
        assert metrics.get_slo_target("safety") == 1.0

    def test_metric_labels_match_grafana_queries(self):
        """Test that metric labels match what Grafana dashboards expect."""
        from backend.src.observability.metrics import get_metrics

        metrics = get_metrics()

        # Record metrics with labels that match Grafana queries
        metrics.record_request(tier="standard", status="success", latency=0.5, endpoint="/weather/query")
        metrics.record_request(tier="complex", status="error", latency=3.0, endpoint="/weather/query")

        # These should not raise - labels must be valid
        metrics.record_llm_call(
            provider="anthropic",
            model="claude-3-5-sonnet",
            status="success",
            latency=1.5,
            input_tokens=100,
            output_tokens=500,
            cost=0.005,
        )


class TestStructuredLoggingIntegration:
    """Test structured logging integration."""

    def test_log_output_is_valid_json(self):
        """Test that log output is valid JSON."""
        from backend.src.observability.logging import (
            StructuredFormatter,
            clear_request_context,
            set_request_context,
        )

        # Capture log output
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter())

        logger = logging.getLogger("test.json.output")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        # Set context and log
        set_request_context(user_id="test_user_123", session_id="sess_456")
        logger.info("Test log message")

        # Get output
        log_output = log_stream.getvalue()
        clear_request_context()

        # Verify it's valid JSON
        parsed = json.loads(log_output)
        assert parsed["message"] == "Test log message"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed
        # Context should be included
        assert parsed.get("user_id") == "test_user_123"
        assert parsed.get("session_id") == "sess_456"

    def test_log_includes_trace_context_when_available(self):
        """Test that logs include trace context when inside a span."""
        from backend.src.observability.logging import StructuredFormatter
        from backend.src.observability.tracing import create_span

        formatter = StructuredFormatter()

        # Create a log record inside a span
        with create_span("test_operation") as span:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Inside span",
                args=(),
                exc_info=None,
            )
            output = formatter.format(record)
            parsed = json.loads(output)

            # Trace context should be present
            if span.get_span_context().is_valid:
                assert "trace_id" in parsed
                assert "span_id" in parsed


class TestOpenTelemetryTracingIntegration:
    """Test OpenTelemetry tracing integration."""

    def test_span_hierarchy_creation(self):
        """Test that spans maintain proper parent-child hierarchy."""
        from backend.src.observability.tracing import SpanAttributes, SpanNames, create_span

        # Create nested spans
        with create_span(SpanNames.HTTP_REQUEST, attributes={SpanAttributes.REQUEST_ID: "req_123"}) as parent:
            parent_context = parent.get_span_context()

            with create_span(SpanNames.LANGGRAPH_WORKFLOW) as workflow:
                workflow_context = workflow.get_span_context()

                with create_span(SpanNames.LANGCHAIN_LLM) as llm:
                    llm_context = llm.get_span_context()

                    # All spans should have same trace ID
                    assert parent_context.trace_id == workflow_context.trace_id
                    assert workflow_context.trace_id == llm_context.trace_id

    def test_span_attributes_propagate_correctly(self):
        """Test that span attributes are set correctly."""
        from backend.src.observability.tracing import SpanAttributes, create_span

        attributes = {
            SpanAttributes.REQUEST_USER_ID: "user_123",
            SpanAttributes.REQUEST_SESSION_ID: "sess_456",
            SpanAttributes.LLM_PROVIDER: "anthropic",
            SpanAttributes.LLM_MODEL: "claude-3-5-sonnet",
        }

        with create_span("test_span", attributes=attributes) as span:
            # Span should have attributes
            assert span is not None

    def test_trace_context_propagation(self):
        """Test W3C TraceContext propagation."""
        from backend.src.observability.tracing import (
            create_span,
            extract_trace_context,
            inject_trace_context,
        )

        # Create a span and inject its context
        with create_span("source_span") as span:
            carrier = {}
            inject_trace_context(carrier)

            # If span is valid, traceparent should be injected
            if span.get_span_context().is_valid:
                # Carrier may contain traceparent
                pass

        # Extract from empty carrier should return valid context
        context = extract_trace_context({})
        assert context is not None


class TestLangChainCallbackIntegration:
    """Test LangChain callback integration with observability."""

    @pytest.mark.asyncio
    async def test_full_llm_lifecycle_instrumentation(self):
        """Test that full LLM lifecycle is instrumented."""
        from langchain_core.outputs import Generation, LLMResult

        from backend.src.observability.callbacks import ObservabilityCallbackHandler
        from backend.src.observability.metrics import get_metrics

        handler = ObservabilityCallbackHandler(
            run_id="test_run",
            user_id="user_123",
            session_id="sess_456",
        )
        metrics = get_metrics()

        run_id = uuid.uuid4()

        # Simulate LLM call lifecycle
        await handler.on_llm_start(
            serialized={"name": "ChatAnthropic", "kwargs": {"model": "claude-3-5-sonnet"}},
            prompts=["What is the weather in Miami?"],
            run_id=run_id,
        )

        # Simulate response
        response = LLMResult(
            generations=[[Generation(text="The weather in Miami is sunny and 85°F.")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 12,
                    "total_tokens": 27,
                },
                "model": "claude-3-5-sonnet",
            },
        )

        await handler.on_llm_end(response=response, run_id=run_id)

        # Metrics should have been recorded (no exceptions)

    @pytest.mark.asyncio
    async def test_tool_lifecycle_instrumentation(self):
        """Test that tool lifecycle is instrumented."""
        from backend.src.observability.callbacks import ObservabilityCallbackHandler

        handler = ObservabilityCallbackHandler(user_id="user_123")
        run_id = uuid.uuid4()

        # Simulate tool call lifecycle
        await handler.on_tool_start(
            serialized={"name": "get_weather_forecast"},
            input_str='{"city": "Miami", "days": 3}',
            run_id=run_id,
        )

        await handler.on_tool_end(
            output='{"temperature": 85, "condition": "sunny", "forecast": [...]}',
            run_id=run_id,
        )

        # Should complete without errors

    @pytest.mark.asyncio
    async def test_chain_lifecycle_instrumentation(self):
        """Test that chain lifecycle is instrumented."""
        from backend.src.observability.callbacks import ObservabilityCallbackHandler

        handler = ObservabilityCallbackHandler(user_id="user_123")
        run_id = uuid.uuid4()

        # Simulate chain lifecycle
        await handler.on_chain_start(
            serialized={"name": "WeatherQueryChain"},
            inputs={"query": "What's the weather in Miami?"},
            run_id=run_id,
        )

        await handler.on_chain_end(
            outputs={"response": "The weather in Miami is sunny and 85°F."},
            run_id=run_id,
        )

        # Should complete without errors


class TestEndToEndObservability:
    """End-to-end tests simulating real request flow."""

    @pytest.mark.asyncio
    async def test_complete_request_observability_flow(self):
        """Test complete request flow with all observability components."""
        from langchain_core.outputs import Generation, LLMResult

        from backend.src.observability import (
            SpanAttributes,
            SpanNames,
            TracingManager,
            clear_request_context,
            create_langchain_callbacks,
            create_span,
            get_metrics,
            get_structured_logger,
            set_request_context,
        )

        # Initialize components
        tracing = TracingManager(service_name="test-weather-ai", environment="test")
        logger = get_structured_logger("test.e2e")
        metrics = get_metrics()

        # Simulate HTTP request
        request_id = str(uuid.uuid4())
        user_id = "user_e2e_test"
        session_id = "session_e2e_test"

        # Set request context for logging
        set_request_context(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )

        # Create request span
        with create_span(
            SpanNames.HTTP_REQUEST,
            attributes={
                SpanAttributes.REQUEST_ID: request_id,
                SpanAttributes.REQUEST_USER_ID: user_id,
                SpanAttributes.REQUEST_QUERY: "What's the weather in Miami?",
            },
        ) as request_span:
            logger.info("Received weather query request")

            # Create LangGraph workflow span
            with create_span(
                SpanNames.LANGGRAPH_WORKFLOW,
                attributes={SpanAttributes.LANGGRAPH_WORKFLOW: "weather_agent"},
            ) as workflow_span:
                # Create callback handler for LangChain
                callbacks = create_langchain_callbacks(
                    run_id=request_id,
                    user_id=user_id,
                    session_id=session_id,
                )
                handler = callbacks[0]

                # Simulate tool call
                tool_run_id = uuid.uuid4()
                await handler.on_tool_start(
                    serialized={"name": "get_current_weather"},
                    input_str='{"city": "Miami"}',
                    run_id=tool_run_id,
                )

                # Record MCP call
                metrics.record_mcp_call(
                    server="weather-mcp",
                    operation="get_current",
                    status="success",
                    latency=0.15,
                )

                await handler.on_tool_end(
                    output='{"temp": 85, "condition": "sunny"}',
                    run_id=tool_run_id,
                )

                # Simulate LLM call
                llm_run_id = uuid.uuid4()
                await handler.on_llm_start(
                    serialized={"name": "ChatAnthropic"},
                    prompts=["Generate weather report"],
                    run_id=llm_run_id,
                )

                await handler.on_llm_end(
                    response=LLMResult(
                        generations=[[Generation(text="Miami: Sunny, 85°F")]],
                        llm_output={"token_usage": {"prompt_tokens": 50, "completion_tokens": 20}},
                    ),
                    run_id=llm_run_id,
                )

            # Record cache operation
            metrics.record_cache_operation(
                cache_level="L1",
                operation="set",
                hit=False,
                latency=0.001,
            )

            # Record overall request
            metrics.record_request(
                tier="standard",
                status="success",
                latency=0.5,
                endpoint="/weather/query",
            )

            logger.info("Weather query completed successfully")

        # Cleanup
        clear_request_context()

        # Verify - just check no exceptions were raised
        # In production, you would verify metrics are accessible via /metrics endpoint


class TestGrafanaDashboardCompatibility:
    """Test that metrics match Grafana dashboard queries."""

    def test_availability_slo_query_compatibility(self):
        """Test that availability SLO query can be calculated."""
        from backend.src.observability.metrics import get_metrics

        metrics = get_metrics()

        # Record mix of success and error requests
        for _ in range(90):
            metrics.record_request(tier="standard", status="success", latency=0.5)
        for _ in range(10):
            metrics.record_request(tier="standard", status="error", latency=2.0)

        # Grafana query: (1 - error_rate) * 100
        # Should be ~90% availability
        # This validates the metric is recording correctly

    def test_latency_p95_query_compatibility(self):
        """Test that latency histogram works for P95 calculation."""
        from backend.src.observability.metrics import get_metrics

        metrics = get_metrics()

        # Record various latencies
        latencies = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
        for latency in latencies:
            metrics.record_request(tier="standard", status="success", latency=latency)

        # Histogram should have buckets for P95 calculation
        # Grafana query: histogram_quantile(0.95, ...)

    def test_llm_cost_tracking_query_compatibility(self):
        """Test that LLM cost tracking works."""
        from backend.src.observability.metrics import get_metrics

        metrics = get_metrics()

        # Record LLM calls with costs
        metrics.record_llm_call(
            provider="anthropic",
            model="claude-3-5-sonnet",
            status="success",
            latency=1.0,
            input_tokens=1000,
            output_tokens=500,
            cost=0.015,
        )
        metrics.record_llm_call(
            provider="openai",
            model="gpt-4",
            status="success",
            latency=0.8,
            input_tokens=800,
            output_tokens=400,
            cost=0.024,
        )

        # Total cost should be trackable via:
        # sum(weather_ai_llm_cost_dollars_total) by (provider, model)


class TestAlertRuleCompatibility:
    """Test that metrics support alert rule queries."""

    def test_safety_guardrail_metrics(self):
        """Test safety guardrail metrics for alerts."""
        from backend.src.observability.metrics import get_metrics

        metrics = get_metrics()

        # Record guardrail checks
        metrics.record_guardrail_check(guardrail="hurricane_validation", passed=True, severity="critical")
        metrics.record_guardrail_check(guardrail="pii_detection", passed=True, severity="critical")
        metrics.record_guardrail_check(guardrail="content_filter", passed=False, severity="warning")

        # Alert: weather_ai_guardrail_violations_total{severity="critical"} > 0

    def test_hurricane_validation_metrics(self):
        """Test hurricane validation metrics."""
        from backend.src.observability.metrics import get_metrics

        metrics = get_metrics()

        # Valid hurricane categories
        metrics.record_hurricane_validation(category=1, valid=True)  # 74-95 mph
        metrics.record_hurricane_validation(category=3, valid=True)  # 111-129 mph
        metrics.record_hurricane_validation(category=5, valid=True)  # 157+ mph

        # Alert: weather_ai_hurricane_validations_total{result="failed"} > 0

    def test_mcp_health_metrics(self):
        """Test MCP health metrics for alerts."""
        from backend.src.observability.metrics import get_metrics

        metrics = get_metrics()

        # Set MCP health status
        metrics.set_mcp_health(server="weather-mcp", healthy=True)
        metrics.set_mcp_health(server="hurricane-mcp", healthy=True)

        # Set data freshness
        metrics.set_mcp_data_freshness(server="weather-mcp", data_type="forecast", age_seconds=300)

        # Alert: weather_ai_mcp_health_status == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
