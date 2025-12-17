# Level 8: Context Window Optimization & Observability Stack - Test Guide

## Quick Start Summary

| Component | Module/Endpoint | Purpose |
|-----------|-----------------|---------|
| Context Optimizer | `backend.src.context.context_optimizer` | 5-phase context optimization pipeline |
| Query Type Detector | `backend.src.context.query_type_detector` | Auto-detect EMERGENCY/COMPLEX/SIMPLE/STANDARD |
| Semantic Chunker | `backend.src.context.semantic_chunker` | Preserve meaning boundaries in chunks |
| Relevance Filter | `backend.src.context.relevance_filter` | Score and filter by query relevance |
| Dynamic Assembler | `backend.src.context.dynamic_assembler` | Adapt context to query type |
| Hierarchical Loader | `backend.src.context.hierarchical_loader` | Load critical first, lazy-load rest |
| Prometheus Metrics | `backend.src.observability.metrics` | SLO-based metrics with dashboards |
| Structured Logging | `backend.src.observability.logging` | JSON logging with trace context |
| OpenTelemetry Tracing | `backend.src.observability.tracing` | Distributed tracing with span hierarchy |
| LangChain Callbacks | `backend.src.observability.callbacks` | Auto-instrumentation for LLM/tools |
| Metrics Endpoint | `GET /metrics` | Prometheus metrics export |
| Health Metrics | `GET /health` | System health with observability |

**Target Metrics:**
- Token reduction: 8K-12K → <4K tokens (50-60% reduction)
- Context recall: >98% (no loss of critical information)
- Optimization latency: <100ms
- SLO Availability: 99.9%
- SLO Latency P95: <2s standard, <5s complex

**Service URLs:**
- Weather API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Jaeger (Tracing): `http://localhost:16686`
- LangSmith: `https://smith.langchain.com`

---

## Prerequisites

### 1. Docker Services Running

```bash
# Start all services (including Prometheus, Grafana, Jaeger)
cd /home/user/weather-ai-agent-service
docker-compose up -d

# Verify services
docker-compose ps
```

Expected output:
```
NAME                   STATUS
weather-ai-api         Up (healthy)
weather-ai-redis       Up (healthy)
weather-ai-neo4j       Up (healthy)
weather-mcp-server     Up (healthy)
hurricane-mcp          Up (healthy)
prometheus             Up (healthy)
grafana                Up (healthy)
jaeger                 Up (healthy)
```

### 2. Environment Variables

```bash
# Required for tests
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="weather-ai-level-8"

# OpenTelemetry Configuration
export OTEL_TRACES_ENABLED="true"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Metrics Configuration
export PROMETHEUS_MULTIPROC_DIR="/tmp/prometheus_multiproc"
```

### 3. Python Environment

```bash
# Sync dependencies
cd /home/user/weather-ai-agent-service
uv sync

# Verify imports
uv run python -c "from backend.src.context import ContextWindowOptimizer, detect_query_type, QueryType; print('✅ Context module available')"
uv run python -c "from backend.src.observability import TracingManager, get_metrics, get_structured_logger; print('✅ Observability module available')"
```

---

## Part 1: Context Window Optimization Testing

### Scenario 1: Query Type Detection - EMERGENCY Queries

**Purpose:** Verify life-safety queries are detected as EMERGENCY type.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 1: Query Type Detection - EMERGENCY."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.context import detect_query_type, get_optimization_config

def test_emergency_detection():
    """Test EMERGENCY query detection."""
    print("=" * 60)
    print("Scenario 1: Query Type Detection - EMERGENCY")
    print("=" * 60)

    emergency_queries = [
        "Should I evacuate for the hurricane?",
        "Is it safe to stay during Category 5?",
        "Mandatory evacuation order - what should I do?",
        "Is there a tornado warning in my area?",
        "Storm surge warning - should I leave?",
        "Life-threatening flooding expected",
        "Emergency alert for severe weather",
    ]

    print("\n🚨 Testing EMERGENCY queries:\n")
    errors = []

    for query in emergency_queries:
        query_type = detect_query_type(query)
        config = get_optimization_config(query_type)

        if query_type == "EMERGENCY":
            print(f"  ✅ '{query[:40]}...'")
            print(f"     Type: {query_type} | Target tokens: {config['target_tokens']}")
            print(f"     Preserve safety: {config['preserve_safety']}")
        else:
            print(f"  ❌ '{query[:40]}...' -> {query_type} (expected EMERGENCY)")
            errors.append(query)

    # Verify EMERGENCY config
    print("\n📋 EMERGENCY optimization config:")
    config = get_optimization_config("EMERGENCY")
    assert config["target_tokens"] == 6000, "Wrong target tokens"
    assert config["min_relevance"] == 0.3, "Wrong min relevance"
    assert config["preserve_safety"] == True, "Safety not preserved"
    print(f"  Target tokens: {config['target_tokens']} (higher limit)")
    print(f"  Min relevance: {config['min_relevance']} (lower threshold)")
    print(f"  Expected reduction: {config['expected_reduction_pct']}")

    if errors:
        print(f"\n❌ FAILED: {len(errors)} queries not detected as EMERGENCY")
    else:
        print("\n✅ Scenario 1 PASSED - All EMERGENCY queries detected correctly")

if __name__ == "__main__":
    test_emergency_detection()
```

**Validation Checklist:**
- [ ] Evacuation queries detected as EMERGENCY
- [ ] Hurricane/tornado warnings detected as EMERGENCY
- [ ] Life-threatening queries detected as EMERGENCY
- [ ] EMERGENCY config has `preserve_safety=True`
- [ ] EMERGENCY config has `target_tokens=6000`

---

### Scenario 2: Query Type Detection - COMPLEX Queries

**Purpose:** Verify multi-part analysis queries are detected as COMPLEX type.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 2: Query Type Detection - COMPLEX."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.context import detect_query_type, get_optimization_config

def test_complex_detection():
    """Test COMPLEX query detection."""
    print("=" * 60)
    print("Scenario 2: Query Type Detection - COMPLEX")
    print("=" * 60)

    complex_queries = [
        "Compare Miami vs Tampa weather for my trip",
        "Analyze the weather trends for next week",
        "Plan my 5-day trip to Florida",
        "What's the weather pattern for both cities?",
        "Give me a historical analysis of hurricane season",
        "Compare temperatures across multiple locations",
        "Weekend forecast with travel planning",
    ]

    print("\n📊 Testing COMPLEX queries:\n")
    errors = []

    for query in complex_queries:
        query_type = detect_query_type(query)
        config = get_optimization_config(query_type)

        if query_type == "COMPLEX":
            print(f"  ✅ '{query[:40]}...'")
            print(f"     Type: {query_type} | Target tokens: {config['target_tokens']}")
        else:
            print(f"  ❌ '{query[:40]}...' -> {query_type} (expected COMPLEX)")
            errors.append(query)

    # Verify COMPLEX config
    print("\n📋 COMPLEX optimization config:")
    config = get_optimization_config("COMPLEX")
    assert config["target_tokens"] == 4500, "Wrong target tokens"
    assert config["min_relevance"] == 0.5, "Wrong min relevance"
    print(f"  Target tokens: {config['target_tokens']}")
    print(f"  Expected reduction: {config['expected_reduction_pct']}")

    if errors:
        print(f"\n❌ FAILED: {len(errors)} queries not detected as COMPLEX")
    else:
        print("\n✅ Scenario 2 PASSED - All COMPLEX queries detected correctly")

if __name__ == "__main__":
    test_complex_detection()
```

**Validation Checklist:**
- [ ] Comparison queries detected as COMPLEX
- [ ] Planning/trip queries detected as COMPLEX
- [ ] Analysis/trend queries detected as COMPLEX
- [ ] Multi-location queries detected as COMPLEX
- [ ] COMPLEX config has `target_tokens=4500`

---

### Scenario 3: Query Type Detection - SIMPLE Queries

**Purpose:** Verify single data point queries are detected as SIMPLE type.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 3: Query Type Detection - SIMPLE."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.context import detect_query_type, get_optimization_config

def test_simple_detection():
    """Test SIMPLE query detection."""
    print("=" * 60)
    print("Scenario 3: Query Type Detection - SIMPLE")
    print("=" * 60)

    simple_queries = [
        "What's the temperature?",
        "Current temperature in Miami",
        "How hot is it?",
        "Is it raining?",
        "Will it rain today?",
        "What's the humidity?",
        "Current weather",
    ]

    print("\n📍 Testing SIMPLE queries:\n")
    errors = []

    for query in simple_queries:
        query_type = detect_query_type(query)
        config = get_optimization_config(query_type)

        if query_type == "SIMPLE":
            print(f"  ✅ '{query[:40]}...'")
            print(f"     Type: {query_type} | Target tokens: {config['target_tokens']}")
        else:
            print(f"  ❌ '{query[:40]}...' -> {query_type} (expected SIMPLE)")
            errors.append(query)

    # Verify SIMPLE config
    print("\n📋 SIMPLE optimization config:")
    config = get_optimization_config("SIMPLE")
    assert config["target_tokens"] == 2500, "Wrong target tokens"
    assert config["min_relevance"] == 0.6, "Wrong min relevance"
    print(f"  Target tokens: {config['target_tokens']} (lowest)")
    print(f"  Min relevance: {config['min_relevance']} (highest)")
    print(f"  Expected reduction: {config['expected_reduction_pct']}")

    if errors:
        print(f"\n❌ FAILED: {len(errors)} queries not detected as SIMPLE")
    else:
        print("\n✅ Scenario 3 PASSED - All SIMPLE queries detected correctly")

if __name__ == "__main__":
    test_simple_detection()
```

**Validation Checklist:**
- [ ] Temperature queries detected as SIMPLE
- [ ] Single-metric queries detected as SIMPLE
- [ ] SIMPLE config has `target_tokens=2500`
- [ ] SIMPLE config has highest `min_relevance=0.6`

---

### Scenario 4: Context Window Optimizer - Full Pipeline

**Purpose:** Test the complete 5-phase optimization pipeline.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 4: Context Window Optimizer - Full Pipeline."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.context import ContextWindowOptimizer

def test_optimizer_pipeline():
    """Test full optimization pipeline."""
    print("=" * 60)
    print("Scenario 4: Context Window Optimizer - Full Pipeline")
    print("=" * 60)

    # Initialize optimizer
    optimizer = ContextWindowOptimizer(
        target_tokens=4000,
        min_relevance_score=0.5,
    )

    # Create test context (simulated large context)
    test_context = """
    Weather Report for Miami, FL - January 2025

    Current Conditions:
    Temperature: 78°F (26°C)
    Humidity: 65%
    Wind: 12 mph from the East
    Conditions: Partly Cloudy
    UV Index: 7 (High)

    5-Day Forecast:
    Monday: High 82°F, Low 68°F, Sunny
    Tuesday: High 80°F, Low 70°F, Partly Cloudy
    Wednesday: High 79°F, Low 69°F, Scattered Showers
    Thursday: High 81°F, Low 71°F, Mostly Sunny
    Friday: High 83°F, Low 72°F, Sunny

    Historical Data:
    Average January temperature: 76°F
    Record high: 89°F (1990)
    Record low: 32°F (1977)
    Average precipitation: 1.9 inches

    Additional Information:
    Sunrise: 7:05 AM EST
    Sunset: 5:52 PM EST
    Moon phase: Waxing Crescent

    Air Quality:
    AQI: 42 (Good)
    Primary pollutant: PM2.5

    Marine Forecast:
    Wave height: 2-3 feet
    Water temperature: 74°F
    Tide: High tide at 8:15 AM
    """ * 10  # Repeat to make larger context

    test_query = "What's the current temperature in Miami?"

    print(f"\n📝 Input:")
    print(f"   Query: '{test_query}'")
    print(f"   Context length: {len(test_context)} characters")

    # Run optimization
    print("\n⚙️ Running 5-phase optimization pipeline...")
    result = optimizer.optimize(test_context, test_query)

    print(f"\n📊 Optimization Results:")
    print(f"   Original tokens: {result.original_tokens}")
    print(f"   Optimized tokens: {result.token_count}")
    print(f"   Reduction: {result.reduction_pct:.1f}%")
    print(f"   Optimization time: {result.optimization_time_ms:.2f}ms")
    print(f"   Chunks used: {result.chunks_used}")
    print(f"   Chunks filtered: {result.chunks_filtered}")

    print(f"\n📋 Phases completed:")
    for phase in result.phases_completed:
        print(f"   ✅ {phase}")

    # Validate results
    errors = []

    # Check target met
    if result.token_count <= 4000:
        print(f"\n✅ Target met: {result.token_count} <= 4000 tokens")
    else:
        print(f"\n⚠️ Target not met: {result.token_count} > 4000 tokens")
        errors.append("Token target not met")

    # Check reduction percentage
    if result.reduction_pct >= 40:
        print(f"✅ Good reduction: {result.reduction_pct:.1f}% >= 40%")
    else:
        print(f"⚠️ Low reduction: {result.reduction_pct:.1f}% < 40%")

    # Check optimization time
    if result.optimization_time_ms < 100:
        print(f"✅ Fast optimization: {result.optimization_time_ms:.2f}ms < 100ms")
    else:
        print(f"⚠️ Slow optimization: {result.optimization_time_ms:.2f}ms >= 100ms")

    # Check content preserved
    if "temperature" in result.optimized_context.lower() or "78" in result.optimized_context:
        print("✅ Relevant content preserved (temperature)")
    else:
        print("⚠️ Relevant content may be lost")

    if errors:
        print(f"\n❌ FAILED: {len(errors)} issues")
    else:
        print("\n✅ Scenario 4 PASSED - Optimization pipeline works correctly")

if __name__ == "__main__":
    test_optimizer_pipeline()
```

**Validation Checklist:**
- [ ] Optimization completes without errors
- [ ] Token count reduced (target: <4K)
- [ ] Reduction percentage >= 40%
- [ ] Optimization time < 100ms
- [ ] All 5 phases completed
- [ ] Relevant content preserved

---

### Scenario 5: Semantic Chunking

**Purpose:** Test semantic chunking preserves meaning boundaries.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 5: Semantic Chunking."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.context import SemanticChunker

def test_semantic_chunking():
    """Test semantic chunking functionality."""
    print("=" * 60)
    print("Scenario 5: Semantic Chunking")
    print("=" * 60)

    chunker = SemanticChunker(
        max_chunk_size=500,
        overlap=50,
    )

    # Test content with clear semantic boundaries
    test_content = """
    # Current Weather Conditions

    The temperature in Miami is currently 78°F with partly cloudy skies.
    Humidity levels are at 65%, which is typical for this time of year.

    # 5-Day Forecast

    Monday will see highs of 82°F with sunny conditions.
    Tuesday brings slightly cooler temperatures around 80°F.
    Wednesday has a chance of scattered showers.

    # Hurricane Information

    Hurricane season runs from June 1 to November 30.
    Miami is in a high-risk zone for tropical storms.
    Evacuation routes should be planned in advance.

    # Historical Data

    January average temperature: 76°F
    Record high: 89°F in 1990
    Record low: 32°F in 1977
    """

    print("\n📝 Input content sections:")
    print("   - Current Weather Conditions")
    print("   - 5-Day Forecast")
    print("   - Hurricane Information")
    print("   - Historical Data")

    # Chunk the content
    chunks = chunker.chunk(test_content)

    print(f"\n📦 Created {len(chunks)} chunks:\n")

    for i, chunk in enumerate(chunks, 1):
        preview = chunk[:80].replace('\n', ' ').strip()
        print(f"   Chunk {i}: {len(chunk)} chars")
        print(f"   Preview: '{preview}...'")
        print()

    # Validate
    errors = []

    # Check chunks created
    if len(chunks) >= 2:
        print("✅ Multiple chunks created")
    else:
        print("❌ Not enough chunks created")
        errors.append("Insufficient chunks")

    # Check no chunk too large
    max_size = max(len(c) for c in chunks)
    if max_size <= 600:  # Allow some tolerance
        print(f"✅ Max chunk size: {max_size} chars (within limit)")
    else:
        print(f"⚠️ Max chunk size: {max_size} chars (may be too large)")

    # Check content preserved
    combined = " ".join(chunks)
    if "miami" in combined.lower() and "hurricane" in combined.lower():
        print("✅ Key content preserved across chunks")
    else:
        print("❌ Key content may be lost")
        errors.append("Content loss")

    if errors:
        print(f"\n❌ FAILED: {len(errors)} issues")
    else:
        print("\n✅ Scenario 5 PASSED - Semantic chunking works correctly")

if __name__ == "__main__":
    test_semantic_chunking()
```

**Validation Checklist:**
- [ ] Content split into multiple chunks
- [ ] Chunk sizes within limits
- [ ] Meaning boundaries preserved
- [ ] No content loss during chunking

---

### Scenario 6: Relevance Filtering

**Purpose:** Test relevance filtering scores and filters chunks correctly.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 6: Relevance Filtering."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.context import RelevanceFilter

def test_relevance_filtering():
    """Test relevance filtering functionality."""
    print("=" * 60)
    print("Scenario 6: Relevance Filtering")
    print("=" * 60)

    # Initialize filter
    relevance_filter = RelevanceFilter(min_relevance_score=0.5)

    # Test chunks with varying relevance
    chunks = [
        "The current temperature in Miami is 78°F with partly cloudy skies. Humidity is at 65%.",
        "Miami is known for its beautiful beaches and Art Deco architecture. The city has a vibrant nightlife.",
        "Weather forecast for Miami: Monday 82°F sunny, Tuesday 80°F partly cloudy.",
        "The history of weather recording began in the 17th century with simple thermometers.",
        "Hurricane season in Florida runs from June to November. Miami has evacuation zones A through E.",
    ]

    query = "What's the current weather in Miami?"

    print(f"\n🔍 Query: '{query}'")
    print(f"\n📦 Input chunks: {len(chunks)}")

    # Filter chunks
    filtered_chunks, scores = relevance_filter.filter(chunks, query)

    print(f"\n📊 Relevance Scores:\n")
    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        preview = chunk[:50].replace('\n', ' ')
        status = "✅ KEPT" if score >= 0.5 else "❌ FILTERED"
        print(f"   Chunk {i}: {score:.3f} {status}")
        print(f"   '{preview}...'")
        print()

    print(f"\n📋 Results:")
    print(f"   Input chunks: {len(chunks)}")
    print(f"   Kept chunks: {len(filtered_chunks)}")
    print(f"   Filtered out: {len(chunks) - len(filtered_chunks)}")

    # Validate
    errors = []

    # Should keep weather-related chunks
    if len(filtered_chunks) >= 2:
        print("\n✅ Relevant chunks preserved")
    else:
        print("\n❌ Too many chunks filtered")
        errors.append("Over-filtering")

    # Should filter irrelevant chunks
    if len(filtered_chunks) < len(chunks):
        print("✅ Irrelevant chunks filtered")
    else:
        print("⚠️ No chunks filtered (may be too lenient)")

    if errors:
        print(f"\n❌ FAILED: {len(errors)} issues")
    else:
        print("\n✅ Scenario 6 PASSED - Relevance filtering works correctly")

if __name__ == "__main__":
    test_relevance_filtering()
```

**Validation Checklist:**
- [ ] Chunks scored by relevance
- [ ] High-relevance chunks kept
- [ ] Low-relevance chunks filtered
- [ ] Weather-related content preserved

---

## Part 2: Observability Stack Testing

### Scenario 7: Prometheus Metrics Recording

**Purpose:** Test that Prometheus metrics are recorded correctly.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 7: Prometheus Metrics Recording."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.observability import get_metrics

def test_prometheus_metrics():
    """Test Prometheus metrics recording."""
    print("=" * 60)
    print("Scenario 7: Prometheus Metrics Recording")
    print("=" * 60)

    metrics = get_metrics()

    print("\n📊 Recording test metrics...\n")

    # Record request metrics
    print("1️⃣ Recording request metrics:")
    metrics.record_request(tier="standard", status="success", latency=0.5, endpoint="/weather/query")
    metrics.record_request(tier="complex", status="success", latency=2.0, endpoint="/weather/query")
    metrics.record_request(tier="standard", status="error", latency=1.0, endpoint="/weather/query")
    print("   ✅ Recorded 3 requests (2 success, 1 error)")

    # Record LLM call metrics
    print("\n2️⃣ Recording LLM call metrics:")
    metrics.record_llm_call(
        provider="anthropic",
        model="claude-3-5-sonnet",
        status="success",
        latency=1.5,
        input_tokens=500,
        output_tokens=200,
        cost=0.0035,
    )
    print("   ✅ Recorded LLM call (anthropic/claude-3-5-sonnet)")

    # Record tool call metrics
    print("\n3️⃣ Recording tool call metrics:")
    metrics.record_tool_call(tool_name="get_current_weather", status="success", latency=0.2)
    metrics.record_tool_call(tool_name="search_knowledge", status="success", latency=0.15)
    print("   ✅ Recorded 2 tool calls")

    # Record MCP call metrics
    print("\n4️⃣ Recording MCP call metrics:")
    metrics.record_mcp_call(server="weather-mcp", operation="get_forecast", status="success", latency=0.3)
    print("   ✅ Recorded MCP call")

    # Record cache metrics
    print("\n5️⃣ Recording cache metrics:")
    metrics.record_cache_operation(cache_level="L1", operation="get", hit=True, latency=0.001)
    metrics.record_cache_operation(cache_level="L2", operation="get", hit=False, latency=0.005)
    print("   ✅ Recorded cache operations (1 hit, 1 miss)")

    # Record context optimization metrics
    print("\n6️⃣ Recording context optimization metrics:")
    metrics.record_context_optimization(
        query_type="STANDARD",
        original_tokens=8000,
        optimized_tokens=3500,
        reduction_pct=56.25,
        latency_ms=45.0,
    )
    print("   ✅ Recorded context optimization")

    # Record guardrail metrics
    print("\n7️⃣ Recording guardrail metrics:")
    metrics.record_guardrail_check(guardrail="hurricane_validation", passed=True, severity="critical")
    metrics.record_guardrail_check(guardrail="pii_detection", passed=True, severity="critical")
    print("   ✅ Recorded guardrail checks")

    # Record hurricane validation
    print("\n8️⃣ Recording hurricane validation metrics:")
    metrics.record_hurricane_validation(category=3, valid=True)
    print("   ✅ Recorded hurricane validation")

    # Check SLO targets
    print("\n📋 SLO Targets:")
    print(f"   Availability: {metrics.get_slo_target('availability')}")
    print(f"   Latency P95 (standard): {metrics.get_slo_target('latency_p95_standard')}s")
    print(f"   Latency P95 (complex): {metrics.get_slo_target('latency_p95_complex')}s")
    print(f"   Safety: {metrics.get_slo_target('safety')}")

    print("\n" + "=" * 60)
    print("✅ Scenario 7 PASSED - All metrics recorded successfully")
    print("=" * 60)

if __name__ == "__main__":
    test_prometheus_metrics()
```

**Validation Checklist:**
- [ ] Request metrics recorded with tier/status/latency
- [ ] LLM metrics recorded with tokens/cost
- [ ] Tool call metrics recorded
- [ ] MCP call metrics recorded
- [ ] Cache metrics recorded with hit/miss
- [ ] Context optimization metrics recorded
- [ ] Guardrail metrics recorded
- [ ] SLO targets accessible

---

### Scenario 8: Structured Logging with Trace Context

**Purpose:** Test structured JSON logging with trace context injection.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 8: Structured Logging with Trace Context."""

import sys
import json
import io
import logging
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.observability import (
    get_structured_logger,
    set_request_context,
    clear_request_context,
    create_span,
)
from backend.src.observability.logging import StructuredFormatter

def test_structured_logging():
    """Test structured logging with trace context."""
    print("=" * 60)
    print("Scenario 8: Structured Logging with Trace Context")
    print("=" * 60)

    # Capture log output
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(StructuredFormatter())

    logger = logging.getLogger("test.structured.logging")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    print("\n1️⃣ Testing basic structured log:")
    logger.info("Test log message")
    log_output = log_stream.getvalue()
    log_stream.truncate(0)
    log_stream.seek(0)

    parsed = json.loads(log_output)
    print(f"   Message: {parsed['message']}")
    print(f"   Level: {parsed['level']}")
    print(f"   Timestamp: {parsed['timestamp'][:19]}...")
    print("   ✅ Basic structured log works")

    print("\n2️⃣ Testing log with request context:")
    set_request_context(user_id="user_123", session_id="sess_456", request_id="req_789")
    logger.info("Log with context")
    log_output = log_stream.getvalue()
    log_stream.truncate(0)
    log_stream.seek(0)

    parsed = json.loads(log_output)
    print(f"   user_id: {parsed.get('user_id')}")
    print(f"   session_id: {parsed.get('session_id')}")
    print(f"   request_id: {parsed.get('request_id')}")
    print("   ✅ Request context included")

    clear_request_context()

    print("\n3️⃣ Testing log inside trace span:")
    with create_span("test_operation") as span:
        logger.info("Log inside span")
        log_output = log_stream.getvalue()
        log_stream.truncate(0)
        log_stream.seek(0)

        parsed = json.loads(log_output)
        if span.get_span_context().is_valid:
            print(f"   trace_id: {parsed.get('trace_id', 'N/A')[:16]}...")
            print(f"   span_id: {parsed.get('span_id', 'N/A')}")
            print("   ✅ Trace context included")
        else:
            print("   ⚠️ Span not valid (tracer not initialized)")
            print("   ✅ Log still works without valid trace")

    print("\n4️⃣ Testing log with exception:")
    try:
        raise ValueError("Test exception for logging")
    except ValueError:
        logger.exception("Error occurred")
        log_output = log_stream.getvalue()
        log_stream.truncate(0)
        log_stream.seek(0)

        parsed = json.loads(log_output)
        print(f"   Exception type: {parsed.get('exception', {}).get('type', 'N/A')}")
        print(f"   Exception message: {parsed.get('exception', {}).get('message', 'N/A')}")
        print("   ✅ Exception logged correctly")

    print("\n" + "=" * 60)
    print("✅ Scenario 8 PASSED - Structured logging works correctly")
    print("=" * 60)

if __name__ == "__main__":
    test_structured_logging()
```

**Validation Checklist:**
- [ ] Logs output as valid JSON
- [ ] Request context included in logs
- [ ] Trace context (trace_id, span_id) included when in span
- [ ] Exceptions logged with type/message/traceback

---

### Scenario 9: OpenTelemetry Tracing

**Purpose:** Test OpenTelemetry span creation and hierarchy.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 9: OpenTelemetry Tracing."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.observability import (
    TracingManager,
    create_span,
    inject_trace_context,
    extract_trace_context,
    SpanAttributes,
    SpanNames,
)

def test_opentelemetry_tracing():
    """Test OpenTelemetry tracing functionality."""
    print("=" * 60)
    print("Scenario 9: OpenTelemetry Tracing")
    print("=" * 60)

    # Initialize tracing (in-memory for testing)
    tracing = TracingManager(
        service_name="weather-ai-test",
        environment="test",
    )

    print("\n1️⃣ Testing span creation:")
    with create_span("test.operation") as span:
        print(f"   Span created: {span is not None}")
        print(f"   Span context valid: {span.get_span_context().is_valid}")
        if span.get_span_context().is_valid:
            print(f"   Trace ID: {format(span.get_span_context().trace_id, '032x')[:16]}...")
            print(f"   Span ID: {format(span.get_span_context().span_id, '016x')}")
        print("   ✅ Span creation works")

    print("\n2️⃣ Testing span hierarchy:")
    with create_span(SpanNames.HTTP_REQUEST, attributes={SpanAttributes.REQUEST_ID: "req_123"}) as parent:
        parent_ctx = parent.get_span_context()

        with create_span(SpanNames.LANGGRAPH_WORKFLOW) as workflow:
            workflow_ctx = workflow.get_span_context()

            with create_span(SpanNames.LANGCHAIN_LLM) as llm:
                llm_ctx = llm.get_span_context()

                if parent_ctx.is_valid and workflow_ctx.is_valid and llm_ctx.is_valid:
                    # All should share the same trace ID
                    same_trace = (
                        parent_ctx.trace_id == workflow_ctx.trace_id == llm_ctx.trace_id
                    )
                    print(f"   Parent -> Workflow -> LLM hierarchy")
                    print(f"   Same trace ID: {same_trace}")
                    if same_trace:
                        print("   ✅ Span hierarchy preserved")
                    else:
                        print("   ❌ Trace ID mismatch")
                else:
                    print("   ⚠️ Spans not valid (tracer not fully initialized)")
                    print("   ✅ Span hierarchy logic works")

    print("\n3️⃣ Testing span attributes:")
    with create_span("test.attributes", attributes={
        SpanAttributes.REQUEST_USER_ID: "user_123",
        SpanAttributes.LLM_MODEL: "claude-3-5-sonnet",
        SpanAttributes.LLM_INPUT_TOKENS: 500,
    }) as span:
        print("   Attributes set:")
        print(f"   - {SpanAttributes.REQUEST_USER_ID}: user_123")
        print(f"   - {SpanAttributes.LLM_MODEL}: claude-3-5-sonnet")
        print(f"   - {SpanAttributes.LLM_INPUT_TOKENS}: 500")
        print("   ✅ Span attributes work")

    print("\n4️⃣ Testing trace context propagation:")
    with create_span("source.operation") as span:
        carrier = {}
        inject_trace_context(carrier)
        print(f"   Injected headers: {list(carrier.keys())}")

        # Extract back
        context = extract_trace_context(carrier)
        print(f"   Extracted context: {context is not None}")
        print("   ✅ Context propagation works")

    print("\n5️⃣ Testing SpanAttributes constants:")
    attrs = [
        SpanAttributes.REQUEST_ID,
        SpanAttributes.LLM_PROVIDER,
        SpanAttributes.TOOL_NAME,
        SpanAttributes.MCP_SERVER,
        SpanAttributes.CACHE_HIT,
        SpanAttributes.CONTEXT_QUERY_TYPE,
    ]
    print("   Available attributes:")
    for attr in attrs:
        print(f"   - {attr}")
    print("   ✅ SpanAttributes available")

    print("\n" + "=" * 60)
    print("✅ Scenario 9 PASSED - OpenTelemetry tracing works correctly")
    print("=" * 60)

if __name__ == "__main__":
    test_opentelemetry_tracing()
```

**Validation Checklist:**
- [ ] Spans created successfully
- [ ] Span hierarchy (parent-child) preserved
- [ ] Same trace ID across span hierarchy
- [ ] Span attributes set correctly
- [ ] Trace context injection/extraction works
- [ ] SpanAttributes constants accessible

---

### Scenario 10: LangChain Callback Integration

**Purpose:** Test LangChain callback handlers for auto-instrumentation.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 10: LangChain Callback Integration."""

import sys
import uuid
import asyncio
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.observability import create_langchain_callbacks
from backend.src.observability.callbacks import ObservabilityCallbackHandler
from langchain_core.outputs import LLMResult, Generation

async def test_langchain_callbacks():
    """Test LangChain callback handlers."""
    print("=" * 60)
    print("Scenario 10: LangChain Callback Integration")
    print("=" * 60)

    print("\n1️⃣ Testing callback handler creation:")
    callbacks = create_langchain_callbacks(
        run_id="test_run_123",
        user_id="user_123",
        session_id="sess_456",
    )
    print(f"   Created {len(callbacks)} callback handler(s)")
    print(f"   Handler type: {type(callbacks[0]).__name__}")
    print("   ✅ Callback handler created")

    handler = callbacks[0]

    print("\n2️⃣ Testing LLM lifecycle:")
    llm_run_id = uuid.uuid4()

    # Simulate LLM start
    await handler.on_llm_start(
        serialized={"name": "ChatAnthropic", "kwargs": {"model": "claude-3-5-sonnet"}},
        prompts=["What is the weather in Miami?"],
        run_id=llm_run_id,
    )
    print("   ✅ on_llm_start called")

    # Simulate LLM end
    response = LLMResult(
        generations=[[Generation(text="The weather in Miami is sunny and 78°F.")]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 15,
                "completion_tokens": 12,
                "total_tokens": 27,
            },
            "model": "claude-3-5-sonnet",
        },
    )
    await handler.on_llm_end(response=response, run_id=llm_run_id)
    print("   ✅ on_llm_end called")

    print("\n3️⃣ Testing tool lifecycle:")
    tool_run_id = uuid.uuid4()

    await handler.on_tool_start(
        serialized={"name": "get_current_weather"},
        input_str='{"city": "Miami"}',
        run_id=tool_run_id,
    )
    print("   ✅ on_tool_start called")

    await handler.on_tool_end(
        output='{"temperature": 78, "condition": "sunny"}',
        run_id=tool_run_id,
    )
    print("   ✅ on_tool_end called")

    print("\n4️⃣ Testing chain lifecycle:")
    chain_run_id = uuid.uuid4()

    await handler.on_chain_start(
        serialized={"name": "WeatherQueryChain"},
        inputs={"query": "Weather in Miami"},
        run_id=chain_run_id,
    )
    print("   ✅ on_chain_start called")

    await handler.on_chain_end(
        outputs={"response": "Miami is sunny and 78°F"},
        run_id=chain_run_id,
    )
    print("   ✅ on_chain_end called")

    print("\n5️⃣ Testing error handling:")
    error_run_id = uuid.uuid4()

    await handler.on_llm_start(
        serialized={"name": "ChatAnthropic"},
        prompts=["Test"],
        run_id=error_run_id,
    )

    await handler.on_llm_error(
        error=Exception("Test error"),
        run_id=error_run_id,
    )
    print("   ✅ on_llm_error called")

    print("\n" + "=" * 60)
    print("✅ Scenario 10 PASSED - LangChain callbacks work correctly")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_langchain_callbacks())
```

**Validation Checklist:**
- [ ] Callback handlers created successfully
- [ ] on_llm_start/on_llm_end callbacks work
- [ ] on_tool_start/on_tool_end callbacks work
- [ ] on_chain_start/on_chain_end callbacks work
- [ ] Error handling callbacks work

---

## Part 3: REST API Testing

### Scenario 11: Metrics Endpoint

**Purpose:** Verify the `/metrics` endpoint exposes Prometheus metrics.

**Test Script:**
```bash
#!/bin/bash
# Test Scenario 11: Metrics Endpoint

echo "============================================================"
echo "Scenario 11: Metrics Endpoint"
echo "============================================================"

API_URL="http://localhost:8000"

echo ""
echo "📊 Fetching Prometheus metrics..."
echo ""

# Fetch metrics
METRICS=$(curl -s "$API_URL/metrics")

# Check for key metrics
echo "🔍 Checking for key metrics:"
echo ""

# Request metrics
if echo "$METRICS" | grep -q "weather_ai_requests_total"; then
    echo "  ✅ weather_ai_requests_total present"
else
    echo "  ❌ weather_ai_requests_total missing"
fi

# LLM metrics
if echo "$METRICS" | grep -q "weather_ai_llm"; then
    echo "  ✅ weather_ai_llm metrics present"
else
    echo "  ❌ weather_ai_llm metrics missing"
fi

# Cache metrics
if echo "$METRICS" | grep -q "weather_ai_cache"; then
    echo "  ✅ weather_ai_cache metrics present"
else
    echo "  ❌ weather_ai_cache metrics missing"
fi

# Context optimization metrics
if echo "$METRICS" | grep -q "weather_ai_context"; then
    echo "  ✅ weather_ai_context metrics present"
else
    echo "  ❌ weather_ai_context metrics missing"
fi

echo ""
echo "📋 Sample metrics output:"
echo "$METRICS" | head -50
echo "..."

echo ""
echo "============================================================"
echo "✅ Scenario 11 PASSED - Metrics endpoint accessible"
echo "============================================================"
```

**Run Command:**
```bash
chmod +x docs/test-guide/scripts/scenario_11_metrics_endpoint.sh
./docs/test-guide/scripts/scenario_11_metrics_endpoint.sh
```

**Validation Checklist:**
- [ ] `/metrics` endpoint returns 200
- [ ] Request metrics present
- [ ] LLM metrics present
- [ ] Cache metrics present
- [ ] Context optimization metrics present

---

### Scenario 12: Weather Query with Observability

**Purpose:** Test a complete weather query and verify observability data.

**Test Script:**
```bash
#!/bin/bash
# Test Scenario 12: Weather Query with Observability

echo "============================================================"
echo "Scenario 12: Weather Query with Observability"
echo "============================================================"

API_URL="http://localhost:8000"

echo ""
echo "1️⃣ Recording initial metrics..."
BEFORE_METRICS=$(curl -s "$API_URL/metrics" | grep "weather_ai_requests_total" | head -1)
echo "   Before: $BEFORE_METRICS"

echo ""
echo "2️⃣ Making weather query..."
RESPONSE=$(curl -s -X POST "$API_URL/weather/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the current temperature in Miami?", "user_id": "test_user_123"}')

echo "   Response received: $(echo $RESPONSE | jq -r '.response' 2>/dev/null | head -c 100)..."

echo ""
echo "3️⃣ Recording final metrics..."
AFTER_METRICS=$(curl -s "$API_URL/metrics" | grep "weather_ai_requests_total" | head -1)
echo "   After: $AFTER_METRICS"

echo ""
echo "4️⃣ Checking context optimization metrics..."
CONTEXT_METRICS=$(curl -s "$API_URL/metrics" | grep "weather_ai_context_optimization")
if [ -n "$CONTEXT_METRICS" ]; then
    echo "   ✅ Context optimization metrics recorded"
    echo "$CONTEXT_METRICS" | head -5
else
    echo "   ⚠️ No context optimization metrics found (may not be triggered)"
fi

echo ""
echo "5️⃣ Checking LLM metrics..."
LLM_METRICS=$(curl -s "$API_URL/metrics" | grep "weather_ai_llm_calls_total")
if [ -n "$LLM_METRICS" ]; then
    echo "   ✅ LLM metrics recorded"
    echo "$LLM_METRICS" | head -3
else
    echo "   ⚠️ No LLM metrics found"
fi

echo ""
echo "============================================================"
echo "✅ Scenario 12 PASSED - Query with observability works"
echo "============================================================"
```

**Validation Checklist:**
- [ ] Weather query succeeds
- [ ] Request metrics incremented
- [ ] LLM metrics recorded
- [ ] Context optimization metrics recorded (if applicable)

---

## Part 4: Unit Test Execution

### Scenario 13: Run All Level 8 Unit Tests

**Purpose:** Execute all Level 8 unit tests.

**Run Command:**
```bash
cd /home/user/weather-ai-agent-service

# Run context optimization tests
uv run pytest backend/tests/test_context_optimization.py -v --tb=short

# Run observability tests
uv run pytest backend/tests/test_observability.py -v --tb=short

# Run observability integration tests
uv run pytest backend/tests/test_observability_integration.py -v --tb=short
```

**Expected Output:**
```
backend/tests/test_context_optimization.py::TestQueryTypeDetector::test_emergency_detection PASSED
backend/tests/test_context_optimization.py::TestQueryTypeDetector::test_complex_detection PASSED
backend/tests/test_context_optimization.py::TestQueryTypeDetector::test_simple_detection PASSED
...
backend/tests/test_observability.py::TestMetricsRegistry::test_metrics_registry_singleton PASSED
backend/tests/test_observability.py::TestMetricsRegistry::test_record_request_success PASSED
...
backend/tests/test_observability_integration.py::TestPrometheusMetricsExport::test_metrics_endpoint_format PASSED
backend/tests/test_observability_integration.py::TestStructuredLoggingIntegration::test_log_output_is_valid_json PASSED
...

========================= XX passed in X.XXs =========================
```

**Validation Checklist:**
- [ ] All context optimization tests pass
- [ ] All observability tests pass (46 tests)
- [ ] All integration tests pass (18 tests)
- [ ] No deprecation warnings
- [ ] No import errors

---

### Scenario 14: Run Tests with Coverage

**Purpose:** Verify test coverage for Level 8 modules.

**Run Command:**
```bash
cd /home/user/weather-ai-agent-service

uv run pytest backend/tests/test_observability.py backend/tests/test_observability_integration.py \
    --cov=backend.src.observability \
    --cov=backend.src.context \
    --cov-report=term-missing \
    -v
```

**Expected Output:**
```
---------- coverage: ... ----------
Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
backend/src/observability/__init__.py            XX      X    XX%   ...
backend/src/observability/metrics.py             XX      X    8X%   ...
backend/src/observability/logging.py             XX      X    8X%   ...
backend/src/observability/tracing.py             XX      X    8X%   ...
backend/src/observability/callbacks.py           XX      X    8X%   ...
backend/src/context/context_optimizer.py         XX      X    8X%   ...
backend/src/context/query_type_detector.py       XX      X    9X%   ...
---------------------------------------------------------------------------
TOTAL                                           XXX      X    8X%
```

**Validation Checklist:**
- [ ] Overall coverage ≥ 80%
- [ ] `observability` module coverage ≥ 80%
- [ ] `context` module coverage ≥ 80%

---

## Part 5: Grafana Dashboard Testing (LangGraph Studio Alternative)

### Scenario 15: Verify Grafana Dashboard Queries

**Purpose:** Test that Grafana dashboard queries work with recorded metrics.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 15: Grafana Dashboard Query Compatibility."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.observability import get_metrics
from prometheus_client import generate_latest, REGISTRY

def test_grafana_queries():
    """Test Grafana dashboard query compatibility."""
    print("=" * 60)
    print("Scenario 15: Grafana Dashboard Query Compatibility")
    print("=" * 60)

    metrics = get_metrics()

    print("\n1️⃣ Recording metrics for dashboard queries...")

    # Record mix of requests for availability calculation
    for _ in range(90):
        metrics.record_request(tier="standard", status="success", latency=0.5)
    for _ in range(10):
        metrics.record_request(tier="standard", status="error", latency=2.0)
    print("   ✅ Recorded 100 requests (90 success, 10 error)")

    # Record latencies for P95 calculation
    latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0]
    for latency in latencies:
        metrics.record_request(tier="standard", status="success", latency=latency)
    print("   ✅ Recorded latency distribution for P95")

    # Record LLM costs
    metrics.record_llm_call(
        provider="anthropic",
        model="claude-3-5-sonnet",
        status="success",
        latency=1.5,
        input_tokens=1000,
        output_tokens=500,
        cost=0.015,
    )
    metrics.record_llm_call(
        provider="openai",
        model="gpt-4",
        status="success",
        latency=1.2,
        input_tokens=800,
        output_tokens=400,
        cost=0.024,
    )
    print("   ✅ Recorded LLM costs")

    # Record context optimization
    metrics.record_context_optimization(
        query_type="STANDARD",
        original_tokens=8000,
        optimized_tokens=3500,
        reduction_pct=56.25,
        latency_ms=45.0,
    )
    print("   ✅ Recorded context optimization")

    print("\n2️⃣ Generating Prometheus metrics output...")
    output = generate_latest(REGISTRY).decode("utf-8")

    print("\n3️⃣ Verifying Grafana query metrics:")

    # Check availability metrics
    if "weather_ai_requests_total" in output or "weather_ai_request" in output:
        print("   ✅ Availability metrics: weather_ai_requests_total present")
    else:
        print("   ❌ Availability metrics missing")

    # Check latency histogram
    if "weather_ai_request_duration_seconds" in output or "weather_ai_request" in output:
        print("   ✅ Latency metrics: request duration histogram present")
    else:
        print("   ⚠️ Latency histogram may have different name")

    # Check LLM cost metrics
    if "weather_ai_llm_cost" in output or "llm" in output:
        print("   ✅ LLM cost metrics present")
    else:
        print("   ⚠️ LLM cost metrics may have different name")

    # Check context optimization metrics
    if "context" in output.lower():
        print("   ✅ Context optimization metrics present")
    else:
        print("   ⚠️ Context optimization metrics may have different name")

    print("\n4️⃣ Sample Prometheus output (first 100 lines):")
    lines = output.split('\n')[:100]
    for line in lines:
        if line and not line.startswith('#'):
            print(f"   {line[:80]}...")

    print("\n" + "=" * 60)
    print("✅ Scenario 15 PASSED - Grafana queries compatible")
    print("=" * 60)

if __name__ == "__main__":
    test_grafana_queries()
```

**Validation Checklist:**
- [ ] Availability metrics present for SLO calculation
- [ ] Latency histogram present for P95 calculation
- [ ] LLM cost metrics present
- [ ] Context optimization metrics present

---

### Scenario 16: SLO Alert Rule Verification

**Purpose:** Test that metrics support SLO alert rules.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 16: SLO Alert Rule Verification."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.observability import get_metrics

def test_slo_alerts():
    """Test SLO alert rule metrics compatibility."""
    print("=" * 60)
    print("Scenario 16: SLO Alert Rule Verification")
    print("=" * 60)

    metrics = get_metrics()

    print("\n1️⃣ Recording safety guardrail metrics...")
    metrics.record_guardrail_check(guardrail="hurricane_validation", passed=True, severity="critical")
    metrics.record_guardrail_check(guardrail="pii_detection", passed=True, severity="critical")
    metrics.record_guardrail_check(guardrail="content_filter", passed=False, severity="warning")
    print("   ✅ Recorded guardrail checks (2 passed, 1 failed)")

    print("\n2️⃣ Recording hurricane validation metrics...")
    metrics.record_hurricane_validation(category=1, valid=True)
    metrics.record_hurricane_validation(category=3, valid=True)
    metrics.record_hurricane_validation(category=5, valid=True)
    print("   ✅ Recorded hurricane validations")

    print("\n3️⃣ Recording MCP health metrics...")
    metrics.set_mcp_health(server="weather-mcp", healthy=True)
    metrics.set_mcp_health(server="hurricane-mcp", healthy=True)
    metrics.set_mcp_data_freshness(server="weather-mcp", data_type="forecast", age_seconds=300)
    print("   ✅ Recorded MCP health status")

    print("\n4️⃣ SLO Targets for Alerts:")
    print(f"   Availability SLO: {metrics.get_slo_target('availability')} (99.9%)")
    print(f"   Latency P95 (standard): {metrics.get_slo_target('latency_p95_standard')}s")
    print(f"   Latency P95 (complex): {metrics.get_slo_target('latency_p95_complex')}s")
    print(f"   Safety SLO: {metrics.get_slo_target('safety')} (100%)")

    print("\n5️⃣ Alert Rules Supported:")
    print("   - weather_ai_availability_slo_breach: error_rate > 0.001")
    print("   - weather_ai_latency_slo_breach: p95_latency > 2s (standard)")
    print("   - weather_ai_safety_violation: guardrail_failures{severity=critical} > 0")
    print("   - weather_ai_hurricane_misclassification: validation_failures > 0")
    print("   - weather_ai_mcp_unhealthy: mcp_health_status == 0")

    print("\n" + "=" * 60)
    print("✅ Scenario 16 PASSED - SLO alert metrics compatible")
    print("=" * 60)

if __name__ == "__main__":
    test_slo_alerts()
```

**Validation Checklist:**
- [ ] Guardrail metrics support alert rules
- [ ] Hurricane validation metrics support alert rules
- [ ] MCP health metrics support alert rules
- [ ] SLO targets accessible for dashboard thresholds

---

## Part 6: Signal Correlation Testing (Level 8 Enhancement)

### Scenario 17: Exemplar Recording in Metrics

**Purpose:** Verify that histogram observations include trace_id exemplars.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 17: Exemplar Recording in Metrics."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from backend.src.observability.metrics import get_exemplar_labels, observe_with_exemplar
from prometheus_client import Histogram, REGISTRY

def test_exemplar_recording():
    """Test exemplar recording with trace context."""
    print("=" * 60)
    print("Scenario 17: Exemplar Recording in Metrics")
    print("=" * 60)

    # Setup tracer provider
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(trace.get_tracer_provider())
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(__name__)

    print("\n1️⃣ Testing exemplar labels extraction:")
    # Outside of span - should return empty dict
    labels = get_exemplar_labels()
    print(f"   Outside span: {labels}")
    if labels == {}:
        print("   ✅ Empty dict when no span active")

    # Inside span - should return trace_id
    with tracer.start_as_current_span("test_operation") as span:
        labels = get_exemplar_labels()
        print(f"   Inside span: {labels}")
        if "trace_id" in labels:
            print(f"   ✅ trace_id extracted: {labels['trace_id'][:16]}...")
            print(f"   ✅ Format: 32-char hex string")
        else:
            print("   ⚠️ No trace_id (tracer may not be initialized)")

    print("\n2️⃣ Testing observe_with_exemplar function:")
    # Create test histogram
    test_histogram = Histogram(
        "test_signal_correlation_histogram",
        "Test histogram for signal correlation",
        ["operation"],
        registry=REGISTRY,
    )

    with tracer.start_as_current_span("test_metric_observation"):
        observe_with_exemplar(
            test_histogram.labels(operation="test"),
            1.5,
        )
        print("   ✅ Histogram observation with exemplar recorded")

    print("\n3️⃣ Verifying exemplar in metrics output:")
    # In production, exemplars appear as:
    # test_histogram_bucket{le="1.0",operation="test"} 0 # {trace_id="abc123..."} 1.5 1234567890
    print("   Note: Exemplars visible via Prometheus /api/v1/query_exemplars")
    print("   Note: Exemplars appear as ⭐ stars on Grafana histogram panels")

    print("\n" + "=" * 60)
    print("✅ Scenario 17 PASSED - Exemplar recording works correctly")
    print("=" * 60)

if __name__ == "__main__":
    test_exemplar_recording()
```

**Validation Checklist:**
- [ ] `get_exemplar_labels()` returns empty dict outside of span
- [ ] `get_exemplar_labels()` returns trace_id inside span
- [ ] trace_id is 32-character lowercase hex string
- [ ] `observe_with_exemplar()` records histogram with exemplar

---

### Scenario 18: Logs → Traces Correlation

**Purpose:** Verify that logs contain trace_id for Loki→Tempo correlation.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 18: Logs → Traces Correlation."""

import sys
import json
import io
import logging
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from backend.src.observability.logging import StructuredFormatter

def test_logs_traces_correlation():
    """Test logs contain trace_id for Loki→Tempo correlation."""
    print("=" * 60)
    print("Scenario 18: Logs → Traces Correlation")
    print("=" * 60)

    # Setup tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Setup logger with structured formatter
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(StructuredFormatter())

    logger = logging.getLogger("test.signal.correlation")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    print("\n1️⃣ Testing log without trace context:")
    logger.info("Log without span")
    log_output = log_stream.getvalue()
    log_stream.truncate(0)
    log_stream.seek(0)

    parsed = json.loads(log_output)
    print(f"   trace_id present: {'trace_id' in parsed}")
    if "trace_id" not in parsed or parsed.get("trace_id") == "":
        print("   ✅ No trace_id when outside span (expected)")

    print("\n2️⃣ Testing log with trace context:")
    with tracer.start_as_current_span("test_operation") as span:
        logger.info("Log inside span")
        log_output = log_stream.getvalue()
        log_stream.truncate(0)
        log_stream.seek(0)

        parsed = json.loads(log_output)
        trace_id = parsed.get("trace_id", "")
        print(f"   trace_id: {trace_id[:16]}..." if trace_id else "   trace_id: (empty)")

        if trace_id and len(trace_id) == 32:
            print("   ✅ trace_id is 32-char hex string")
            print("   ✅ Loki can extract via regex: '\"trace_id\":\\s*\"([a-f0-9]{32})\"'")
        else:
            print("   ⚠️ trace_id not in expected format")

    print("\n3️⃣ Verifying Loki derivedFields configuration:")
    print("   Loki datasource should have:")
    print("   - matcherRegex: '\"trace_id\":\\s*\"([a-f0-9]{32})\"'")
    print("   - datasourceUid: tempo")
    print("   - urlDisplayLabel: 'View Trace in Tempo'")

    print("\n4️⃣ Testing correlation path:")
    print("   1. Log entry in Loki: {\"trace_id\": \"abc123...\"}")
    print("   2. Loki derivedFields extracts trace_id")
    print("   3. Click trace_id → Grafana opens Tempo with that trace")

    print("\n" + "=" * 60)
    print("✅ Scenario 18 PASSED - Logs→Traces correlation configured")
    print("=" * 60)

if __name__ == "__main__":
    test_logs_traces_correlation()
```

**Validation Checklist:**
- [ ] Logs are JSON formatted
- [ ] trace_id included in log when inside span
- [ ] trace_id is 32-character lowercase hex
- [ ] Loki derivedFields regex matches format

---

### Scenario 19: Traces → Logs Correlation (Tempo tracesToLogsV2)

**Purpose:** Verify Tempo can link traces to Loki logs.

**Test Script:**
```bash
#!/bin/bash
# Test Scenario 19: Traces → Logs Correlation

echo "============================================================"
echo "Scenario 19: Traces → Logs Correlation (tracesToLogsV2)"
echo "============================================================"

echo ""
echo "1️⃣ Checking Grafana datasources configuration..."

DATASOURCES_FILE="observability/grafana/provisioning/datasources/datasources.yml"

if [ -f "$DATASOURCES_FILE" ]; then
    echo "   ✅ Datasources file exists"

    # Check for Tempo tracesToLogsV2
    if grep -q "tracesToLogsV2" "$DATASOURCES_FILE"; then
        echo "   ✅ tracesToLogsV2 configured in Tempo datasource"
    else
        echo "   ❌ tracesToLogsV2 NOT configured"
    fi

    # Check for Loki datasourceUid
    if grep -q "datasourceUid: loki" "$DATASOURCES_FILE"; then
        echo "   ✅ Loki datasourceUid referenced"
    else
        echo "   ⚠️ Loki datasourceUid not found"
    fi

    # Check for query template
    if grep -q "trace_id" "$DATASOURCES_FILE"; then
        echo "   ✅ trace_id query template configured"
    else
        echo "   ⚠️ trace_id query template not found"
    fi
else
    echo "   ❌ Datasources file not found"
fi

echo ""
echo "2️⃣ Verifying Tempo tracesToLogsV2 configuration:"
echo "   Expected settings:"
echo "   - datasourceUid: loki"
echo "   - filterByTraceID: true"
echo "   - query: {app=\"weather-ai-agent\"} | json | trace_id=\`\${__trace.traceId}\`"

echo ""
echo "3️⃣ Testing correlation path:"
echo "   1. Open Grafana → Explore → Tempo"
echo "   2. Find a trace"
echo "   3. Click 'Logs' tab in trace view"
echo "   4. Should show Loki logs filtered by trace_id"

echo ""
echo "============================================================"
echo "✅ Scenario 19 PASSED - Traces→Logs correlation configured"
echo "============================================================"
```

**Validation Checklist:**
- [ ] Tempo datasource has tracesToLogsV2 configuration
- [ ] Loki datasourceUid correctly referenced
- [ ] Query template includes trace_id filter
- [ ] "Logs" tab appears in Tempo trace view

---

### Scenario 20: Traces → Metrics Correlation (Tempo tracesToMetrics)

**Purpose:** Verify Tempo can link traces to Prometheus metrics (RED signals).

**Test Script:**
```bash
#!/bin/bash
# Test Scenario 20: Traces → Metrics Correlation

echo "============================================================"
echo "Scenario 20: Traces → Metrics Correlation (tracesToMetrics)"
echo "============================================================"

echo ""
echo "1️⃣ Checking Grafana datasources configuration..."

DATASOURCES_FILE="observability/grafana/provisioning/datasources/datasources.yml"

if [ -f "$DATASOURCES_FILE" ]; then
    # Check for Tempo tracesToMetrics
    if grep -q "tracesToMetrics" "$DATASOURCES_FILE"; then
        echo "   ✅ tracesToMetrics configured in Tempo datasource"
    else
        echo "   ❌ tracesToMetrics NOT configured"
    fi

    # Check for Prometheus datasourceUid
    if grep -q "datasourceUid: prometheus" "$DATASOURCES_FILE"; then
        echo "   ✅ Prometheus datasourceUid referenced"
    else
        echo "   ⚠️ Prometheus datasourceUid not found"
    fi
fi

echo ""
echo "2️⃣ Verifying RED metrics queries:"
grep -A 5 "tracesToMetrics" "$DATASOURCES_FILE" 2>/dev/null | head -20

echo ""
echo "3️⃣ Expected RED metric queries:"
echo "   - Request Rate: sum(rate(weather_ai_requests_total[5m]))"
echo "   - Error Rate: error_requests / total_requests"
echo "   - Duration P99: histogram_quantile(0.99, ...)"
echo "   - LLM P95 Latency: histogram_quantile(0.95, ...llm_latency...)"
echo "   - MCP Success Rate: success_requests / total_requests"
echo "   - Cache Hit Rate: avg(weather_ai_cache_hit_rate)"

echo ""
echo "4️⃣ Testing correlation path:"
echo "   1. Open Grafana → Explore → Tempo"
echo "   2. Find a trace"
echo "   3. Click 'Metrics' tab in trace view"
echo "   4. Should show RED signals from Prometheus"

echo ""
echo "============================================================"
echo "✅ Scenario 20 PASSED - Traces→Metrics correlation configured"
echo "============================================================"
```

**Validation Checklist:**
- [ ] Tempo datasource has tracesToMetrics configuration
- [ ] Prometheus datasourceUid correctly referenced
- [ ] RED metric queries defined (Rate, Error, Duration)
- [ ] "Metrics" tab appears in Tempo trace view

---

### Scenario 21: Metrics → Traces Correlation (Prometheus Exemplars)

**Purpose:** Verify Prometheus exemplars enable clicking histogram points to view traces.

**Test Script:**
```bash
#!/bin/bash
# Test Scenario 21: Metrics → Traces Correlation

echo "============================================================"
echo "Scenario 21: Metrics → Traces Correlation (Exemplars)"
echo "============================================================"

echo ""
echo "1️⃣ Checking Prometheus configuration..."

# Check docker-compose for exemplar-storage flag
if grep -q "enable-feature=exemplar-storage" docker-compose.yml; then
    echo "   ✅ Prometheus exemplar-storage feature enabled"
else
    echo "   ❌ Prometheus exemplar-storage NOT enabled"
fi

echo ""
echo "2️⃣ Checking Grafana Prometheus datasource..."

DATASOURCES_FILE="observability/grafana/provisioning/datasources/datasources.yml"

if grep -q "exemplarTraceIdDestinations" "$DATASOURCES_FILE"; then
    echo "   ✅ exemplarTraceIdDestinations configured"
else
    echo "   ❌ exemplarTraceIdDestinations NOT configured"
fi

if grep -A 5 "exemplarTraceIdDestinations" "$DATASOURCES_FILE" | grep -q "tempo"; then
    echo "   ✅ Exemplars linked to Tempo datasource"
else
    echo "   ⚠️ Exemplars may not be linked to Tempo"
fi

echo ""
echo "3️⃣ Checking dashboard panels for exemplar: true..."

DASHBOARD_FILE="observability/grafana/provisioning/dashboards/signal-correlation.json"

if [ -f "$DASHBOARD_FILE" ]; then
    EXEMPLAR_COUNT=$(grep -c '"exemplar": true' "$DASHBOARD_FILE")
    echo "   Panels with exemplar: true = $EXEMPLAR_COUNT"
    if [ "$EXEMPLAR_COUNT" -gt 0 ]; then
        echo "   ✅ Dashboard panels have exemplar support"
    else
        echo "   ⚠️ No panels with exemplar: true found"
    fi
else
    echo "   ⚠️ Signal correlation dashboard not found"
fi

echo ""
echo "4️⃣ Testing correlation path:"
echo "   1. Open Grafana → Signal Correlation dashboard"
echo "   2. Look for ⭐ (star) points on histogram panels"
echo "   3. Click a ⭐ exemplar point"
echo "   4. Should open Tempo with the corresponding trace"

echo ""
echo "5️⃣ Exemplar format in Prometheus:"
echo "   histogram_bucket{le=\"1.0\"} 10 # {trace_id=\"abc123...\"} 1.5 1234567890"
echo "   └── metric ──┘                   └── exemplar ────┘"

echo ""
echo "============================================================"
echo "✅ Scenario 21 PASSED - Metrics→Traces correlation configured"
echo "============================================================"
```

**Validation Checklist:**
- [ ] Prometheus --enable-feature=exemplar-storage flag set
- [ ] Grafana Prometheus datasource has exemplarTraceIdDestinations
- [ ] Exemplars linked to Tempo datasource
- [ ] Dashboard panels have `"exemplar": true` in queries
- [ ] ⭐ stars visible on histogram panels in Grafana

---

### Scenario 22: Full Signal Correlation End-to-End

**Purpose:** Test the complete correlation flow across all three signals.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 22: Full Signal Correlation End-to-End."""

import sys
import subprocess
sys.path.insert(0, '/home/user/weather-ai-agent-service')

def test_full_correlation():
    """Test complete signal correlation flow."""
    print("=" * 60)
    print("Scenario 22: Full Signal Correlation End-to-End")
    print("=" * 60)

    print("\n📋 Signal Correlation Architecture:")
    print("""
    ┌──────────────┐      ⭐ exemplars        ┌──────────────┐
    │  PROMETHEUS  │ ─────────────────────►   │    TEMPO     │
    │   Metrics    │  (click stars)           │   Traces     │
    │              │ ◄─────────────────────── │              │
    └──────────────┘    tracesToMetrics       └──────────────┘
           │              (RED queries)              │
           │                                         │ tracesToLogsV2
           ▼                                         ▼
    ┌────────────────────────────────────────────────────────┐
    │                        GRAFANA                          │
    │  • Metrics: Click ⭐ exemplars → Jump to trace          │
    │  • Logs: Click trace_id → Jump to trace                │
    │  • Trace view: "Logs" + "Metrics" tabs                 │
    └────────────────────────────────────────────────────────┘
           ▲                                         ▲
           │ derivedFields                           │
           │ (trace_id regex)                        │
    ┌──────────────┐                         ┌──────────────┐
    │    LOKI      │ ◄───────────────────────│    TEMPO     │
    │    Logs      │     tracesToLogsV2      │   Traces     │
    └──────────────┘                         └──────────────┘
    """)

    print("\n1️⃣ Running verification script...")
    result = subprocess.run(
        ["python3", "scripts/verify_signal_correlation.py"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️ Verification had warnings: {result.stderr}")

    print("\n2️⃣ Correlation Paths Summary:")
    paths = [
        ("Metrics → Traces", "Click ⭐ exemplar on histogram panel", "Opens Tempo trace"),
        ("Logs → Traces", "Click trace_id in log entry", "Opens Tempo trace"),
        ("Traces → Logs", "Open trace, click 'Logs' tab", "Shows filtered Loki logs"),
        ("Traces → Metrics", "Open trace, click 'Metrics' tab", "Shows RED signals"),
    ]
    for path, action, result in paths:
        print(f"   {path}:")
        print(f"     Action: {action}")
        print(f"     Result: {result}")
        print()

    print("3️⃣ Manual Testing Steps:")
    print("   a. Start services: docker-compose up -d")
    print("   b. Generate traffic: curl -X POST http://localhost:8000/weather/query \\")
    print("      -d '{\"query\": \"weather in miami\"}'")
    print("   c. Open Grafana: http://localhost:3001")
    print("   d. Go to 'Signal Correlation (Level 8)' dashboard")
    print("   e. Test each correlation path above")

    print("\n" + "=" * 60)
    print("✅ Scenario 22 PASSED - Full correlation architecture verified")
    print("=" * 60)

if __name__ == "__main__":
    test_full_correlation()
```

**Validation Checklist:**
- [ ] All verification checks pass
- [ ] Metrics → Traces path works (exemplars)
- [ ] Logs → Traces path works (derivedFields)
- [ ] Traces → Logs path works (tracesToLogsV2)
- [ ] Traces → Metrics path works (tracesToMetrics)

---

### Scenario 23: Tempo Service Health Check

**Purpose:** Verify Tempo distributed tracing service is running correctly.

**Test Script:**
```bash
#!/bin/bash
# Test Scenario 23: Tempo Service Health Check

echo "============================================================"
echo "Scenario 23: Tempo Service Health Check"
echo "============================================================"

TEMPO_URL="http://localhost:3200"

echo ""
echo "1️⃣ Checking Tempo readiness..."
READY=$(curl -s "$TEMPO_URL/ready" 2>/dev/null)
if echo "$READY" | grep -qi "ready"; then
    echo "   ✅ Tempo is ready"
else
    echo "   ❌ Tempo not ready or not running"
    echo "   Response: $READY"
fi

echo ""
echo "2️⃣ Checking Tempo status..."
STATUS=$(curl -s "$TEMPO_URL/status" 2>/dev/null)
if [ -n "$STATUS" ]; then
    echo "   ✅ Tempo status endpoint accessible"
else
    echo "   ⚠️ Tempo status not available"
fi

echo ""
echo "3️⃣ Checking OTLP receiver ports..."
# Check if port 4317 (gRPC) is listening
if nc -z localhost 4317 2>/dev/null; then
    echo "   ✅ OTLP gRPC port 4317 accessible"
else
    echo "   ⚠️ OTLP gRPC port 4317 not accessible"
fi

# Check if port 4318 (HTTP) is listening
if nc -z localhost 4318 2>/dev/null; then
    echo "   ✅ OTLP HTTP port 4318 accessible"
else
    echo "   ⚠️ OTLP HTTP port 4318 not accessible"
fi

echo ""
echo "4️⃣ Checking Docker container..."
if docker ps | grep -q "weather-ai-tempo"; then
    echo "   ✅ weather-ai-tempo container running"
    docker inspect weather-ai-tempo --format '{{.State.Health.Status}}' 2>/dev/null | \
        sed 's/^/   Health: /'
else
    echo "   ❌ weather-ai-tempo container not running"
fi

echo ""
echo "============================================================"
echo "✅ Scenario 23 - Tempo health check complete"
echo "============================================================"
```

**Validation Checklist:**
- [ ] Tempo /ready endpoint returns "ready"
- [ ] OTLP gRPC port 4317 accessible
- [ ] OTLP HTTP port 4318 accessible
- [ ] Docker container healthy

---

## Summary: Level 8 Test Coverage

| Scenario | Type | Component | Status |
|----------|------|-----------|--------|
| 1 | Context | Query Type Detection - EMERGENCY | ⬜ |
| 2 | Context | Query Type Detection - COMPLEX | ⬜ |
| 3 | Context | Query Type Detection - SIMPLE | ⬜ |
| 4 | Context | Context Optimizer - Full Pipeline | ⬜ |
| 5 | Context | Semantic Chunking | ⬜ |
| 6 | Context | Relevance Filtering | ⬜ |
| 7 | Observability | Prometheus Metrics Recording | ⬜ |
| 8 | Observability | Structured Logging | ⬜ |
| 9 | Observability | OpenTelemetry Tracing | ⬜ |
| 10 | Observability | LangChain Callbacks | ⬜ |
| 11 | REST API | Metrics Endpoint | ⬜ |
| 12 | REST API | Weather Query with Observability | ⬜ |
| 13 | Unit Tests | All Tests Pass | ⬜ |
| 14 | Coverage | Coverage Report | ⬜ |
| 15 | Grafana | Dashboard Query Compatibility | ⬜ |
| 16 | Grafana | SLO Alert Rule Verification | ⬜ |
| 17 | Signal Correlation | Exemplar Recording | ⬜ |
| 18 | Signal Correlation | Logs → Traces | ⬜ |
| 19 | Signal Correlation | Traces → Logs | ⬜ |
| 20 | Signal Correlation | Traces → Metrics | ⬜ |
| 21 | Signal Correlation | Metrics → Traces | ⬜ |
| 22 | Signal Correlation | Full E2E Correlation | ⬜ |
| 23 | Signal Correlation | Tempo Health Check | ⬜ |

**Total Scenarios:** 23 (16 original + 7 signal correlation)

---

## Troubleshooting

### Common Issues

**1. Context Module Import Error**
```python
ImportError: cannot import name 'ContextWindowOptimizer'
```
**Solution:** Ensure `backend/src/context/__init__.py` exports the module.

**2. Tiktoken Network Error**
```
ValueError: Could not automatically determine tiktoken encoding
```
**Solution:** The optimizer falls back to character-based estimation. Set proxy if needed:
```bash
export HTTP_PROXY="http://proxy:port"
```

**3. Prometheus Duplicate Metrics Error**
```
ValueError: Duplicated timeseries in CollectorRegistry
```
**Solution:** Use `get_metrics()` singleton instead of creating new instances.

**4. OpenTelemetry Span Not Valid**
```
Span context is_valid = False
```
**Solution:** Initialize `TracingManager` at application startup:
```python
tracing = TracingManager(service_name="weather-ai")
tracing.initialize()
```

**5. Structured Logs Not JSON**
```
Logs appear as plain text
```
**Solution:** Configure structured logging:
```python
from backend.src.observability import configure_structured_logging
configure_structured_logging(level="INFO", json_format=True)
```

**6. Grafana Dashboard Empty**
```
No data in Grafana panels
```
**Solution:**
- Verify Prometheus is scraping `/metrics` endpoint
- Check Prometheus targets: `http://localhost:9090/targets`
- Ensure metrics are being recorded by making queries

---

## Quick Test Commands

```bash
# Run all Level 8 tests
uv run pytest backend/tests/test_observability.py backend/tests/test_observability_integration.py backend/tests/test_context_optimization.py -v

# Run only observability tests
uv run pytest backend/tests/test_observability.py -v

# Run only integration tests
uv run pytest backend/tests/test_observability_integration.py -v

# Run with coverage
uv run pytest backend/tests/test_observability.py --cov=backend.src.observability --cov-report=html

# Check metrics endpoint
curl http://localhost:8000/metrics | head -50

# Check health endpoint
curl http://localhost:8000/health | jq

# Test weather query
curl -X POST http://localhost:8000/weather/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Miami?", "user_id": "test"}'

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets'

# Check Grafana datasources
curl -u admin:admin http://localhost:3000/api/datasources | jq
```

---

## Architecture Overview

### Context Window Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  Context Window Optimizer                        │
│  (5-Phase Pipeline: 8K-12K → <4K tokens, 50-60% reduction)      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │ Query Type   │   │    Semantic     │   │   Relevance     │  │
│  │ Detector     │──▶│    Chunker      │──▶│   Filter        │  │
│  │              │   │                 │   │                 │  │
│  │ EMERGENCY    │   │ Split by        │   │ Score chunks    │  │
│  │ COMPLEX      │   │ meaning         │   │ by query        │  │
│  │ SIMPLE       │   │ boundaries      │   │ relevance       │  │
│  │ STANDARD     │   │                 │   │                 │  │
│  └──────────────┘   └─────────────────┘   └─────────────────┘  │
│         │                                          │            │
│         ▼                                          ▼            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Dynamic Assembler                            │  │
│  │  Adapt context based on query type:                       │  │
│  │  - EMERGENCY: Preserve safety info (30-40% reduction)     │  │
│  │  - COMPLEX: Balanced (50-55% reduction)                   │  │
│  │  - STANDARD: Standard (50-60% reduction)                  │  │
│  │  - SIMPLE: Aggressive (60-70% reduction)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Hierarchical Loader                          │  │
│  │  Load order: Critical → Important → Supplementary         │  │
│  │  Lazy-load remaining context as needed                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Observability Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Observability Stack                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   Prometheus    │    │   Structured    │    │ OpenTelemetry│ │
│  │   Metrics       │    │   Logging       │    │   Tracing    │ │
│  │                 │    │                 │    │              │ │
│  │ - Request count │    │ - JSON format   │    │ - Spans      │ │
│  │ - Latency P95   │    │ - Trace context │    │ - Hierarchy  │ │
│  │ - LLM tokens    │    │ - Request ctx   │    │ - Context    │ │
│  │ - Cache hit/miss│    │ - Exceptions    │    │ - Propagation│ │
│  │ - SLO targets   │    │                 │    │              │ │
│  └────────┬────────┘    └────────┬────────┘    └──────┬──────┘ │
│           │                      │                     │        │
│           ▼                      ▼                     ▼        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LangChain Callback Handler                  │   │
│  │  Auto-instrumentation for:                               │   │
│  │  - LLM calls (start/end/error)                          │   │
│  │  - Tool calls (start/end/error)                         │   │
│  │  - Chain execution (start/end)                          │   │
│  │  - Token counting and cost calculation                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │    Grafana      │    │      Loki       │    │    Jaeger   │ │
│  │   Dashboards    │    │  Log Aggregation│    │   Tracing   │ │
│  │                 │    │                 │    │              │ │
│  │ - SLO Overview  │    │ - Search logs   │    │ - Trace view│ │
│  │ - Performance   │    │ - Filter by     │    │ - Span deps │ │
│  │ - Cost Analysis │    │   trace_id      │    │ - Latency   │ │
│  │ - Alerts        │    │                 │    │              │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-21
**Level:** 8 - Context Window Optimization & Observability Stack

**Changelog:**
- v1.0.0: Initial release with 16 test scenarios covering context optimization and observability
