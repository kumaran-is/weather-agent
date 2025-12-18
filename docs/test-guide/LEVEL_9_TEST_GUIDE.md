# Level 9 Test Guide: Semantic Caching

## Overview

Level 9 implements a **two-tier semantic caching architecture** that significantly reduces API costs and improves response latency by caching similar queries, tool results, and LLM responses.

### Cache Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Level 9 Cache Hierarchy                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query Response Cache (Q1 → Q2 → Q3):                                      │
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐                         │
│  │    Q1    │ →  │    Q2    │ →  │      Q3       │                         │
│  │ In-Memory│    │  Redis   │    │Qdrant Semantic│                         │
│  │  (<1ms)  │    │ (<10ms)  │    │   (<50ms)     │                         │
│  │  5 min   │    │  30 min  │    │    30 min     │                         │
│  │user-spec │    │  shared  │    │  similarity   │                         │
│  └──────────┘    └──────────┘    └───────────────┘                         │
│                                                                             │
│  Tool Result Cache (T1 → T2):                                              │
│  ┌──────────┐    ┌───────────────┐                                         │
│  │    T1    │ →  │      T2       │  BYPASS: hurricane alerts               │
│  │  Redis   │    │Qdrant Semantic│  (life-safety tools)                    │
│  │ (<10ms)  │    │   (<50ms)     │                                         │
│  │ per-tool │    │  per-tool     │                                         │
│  └──────────┘    └───────────────┘                                         │
│                                                                             │
│  LLM Response Cache (R1 → R2):                                             │
│  ┌──────────┐    ┌───────────────┐                                         │
│  │    R1    │ →  │      R2       │  EXCLUDE: tool-calling prompts          │
│  │  Redis   │    │Qdrant Semantic│  (agent prompts, actions)               │
│  │ (<10ms)  │    │   (<50ms)     │                                         │
│  │ per-model│    │  per-model    │                                         │
│  └──────────┘    └───────────────┘                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Infrastructure Requirements

```bash
# Start all services (includes Qdrant, Redis for semantic cache)
make docker-up-dev

# Verify services are healthy
docker ps --format "table {{.Names}}\t{{.Status}}"

# Expected output:
# NAMES                  STATUS
# weather-ai-api         Up (healthy)
# weather-ai-qdrant      Up
# weather-ai-redis       Up (healthy)
# weather-ai-neo4j       Up (healthy)
# weather-ai-postgres    Up (healthy)
# weather-ai-prometheus  Up (healthy)
# weather-ai-grafana     Up (healthy)
```

### Environment Variables

Ensure these are set in your `.env` file:

```bash
# Level 9 Semantic Cache Configuration
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.85  # Similarity threshold (0.0-1.0)
SEMANTIC_CACHE_TTL=1800        # 30 minutes
TOOL_CACHE_ENABLED=true
LLM_CACHE_ENABLED=true
CACHE_METRICS_ENABLED=true

# Qdrant (semantic vector store)
QDRANT_URL=http://localhost:6333

# Redis (exact match cache)
REDIS_URL=redis://localhost:6379/0
L2_CACHE_REDIS_URL=redis://localhost:6379/0
```

---

## Test Categories

### 1. Query Normalization Tests

Test that queries are normalized for better cache hit rates.

#### Test 1.1: Location Alias Normalization

```python
import pytest
from backend.src.cache.query_normalizer import QueryNormalizer, get_query_normalizer

def test_location_aliases():
    """Test location alias normalization."""
    normalizer = get_query_normalizer()

    # Test common aliases
    assert "san francisco" in normalizer.normalize("What's the weather in SF?").lower()
    assert "new york city" in normalizer.normalize("NYC weather forecast").lower()
    assert "los angeles" in normalizer.normalize("LA temperature").lower()
    assert "washington dc" in normalizer.normalize("DC weather").lower()

def test_term_aliases():
    """Test weather term alias expansion."""
    normalizer = get_query_normalizer()

    normalized = normalizer.normalize("What's the temp in Miami?").lower()
    assert "temperature" in normalized

    normalized = normalizer.normalize("Show me wx forecast").lower()
    assert "weather" in normalized
```

#### Test 1.2: Query Normalization Consistency

```python
def test_normalization_consistency():
    """Test that similar queries normalize to same form."""
    normalizer = get_query_normalizer()

    # These should all normalize to similar forms
    queries = [
        "Weather in San Francisco",
        "weather in san francisco",
        "WEATHER IN SAN FRANCISCO",
        "  weather  in  san francisco  ",
    ]

    normalized = [normalizer.normalize(q) for q in queries]

    # All should be identical after normalization
    assert len(set(normalized)) == 1, "All queries should normalize to same form"
```

### 2. Semantic Query Cache (Q3) Tests

Test the semantic similarity matching for query responses.

#### Test 2.1: Exact Match (Q2/Redis)

```python
import pytest
import asyncio
from backend.src.cache.semantic_query_cache import SemanticQueryCache, create_semantic_query_cache

@pytest.mark.asyncio
async def test_exact_match_cache():
    """Test exact match through Redis (T1)."""
    cache = await create_semantic_query_cache(
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
    )

    query = "What's the weather in Miami?"
    response = {"response": "Currently sunny, 85°F in Miami."}

    # Store in cache
    await cache.set_response(query, response, enable_rag=True, enable_cot=True)

    # Retrieve - should be T1 hit
    result = await cache.get_response(query, enable_rag=True, enable_cot=True)

    assert result.hit is True
    assert result.tier == "tier1"
    assert result.value["response"] == response["response"]
```

#### Test 2.2: Semantic Match (Qdrant)

```python
@pytest.mark.asyncio
async def test_semantic_match_cache():
    """Test semantic similarity matching through Qdrant (T2)."""
    cache = await create_semantic_query_cache(
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
        threshold=0.85,
    )

    # Store original query
    original = "What's the weather in San Francisco?"
    response = {"response": "Currently foggy, 62°F in San Francisco."}
    await cache.set_response(original, response, enable_rag=True, enable_cot=True)

    # Query with similar but not identical text
    similar = "Weather forecast for SF"  # SF normalizes to San Francisco

    result = await cache.get_response(similar, enable_rag=True, enable_cot=True)

    # Should get semantic match
    assert result.hit is True
    assert result.tier == "tier2"  # Semantic tier
    assert result.similarity_score >= 0.85
```

#### Test 2.3: Similarity Threshold

```python
@pytest.mark.asyncio
async def test_similarity_threshold():
    """Test that low similarity queries don't match."""
    cache = await create_semantic_query_cache(
        threshold=0.90,  # High threshold
    )

    # Store weather query
    await cache.set_response(
        "Weather in Miami",
        {"response": "Sunny in Miami"},
        enable_rag=True, enable_cot=True,
    )

    # Query something completely different
    result = await cache.get_response(
        "Best restaurants in Seattle",
        enable_rag=True, enable_cot=True,
    )

    # Should NOT match (different topic)
    assert result.hit is False
```

### 3. Tool Result Cache Tests

Test caching of MCP tool call results.

#### Test 3.1: Tool Cache Hit

```python
import pytest
from backend.src.cache.tool_cache import ToolResultCache

@pytest.mark.asyncio
async def test_tool_cache_hit():
    """Test tool result caching."""
    cache = ToolResultCache(
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
    )
    await cache.initialize()

    # Store tool result
    tool_name = "get_forecast"
    tool_args = {"location": "Miami", "days": 7}
    result = {"forecast": [{"day": 1, "temp": 85}]}

    await cache.set_tool_result(tool_name, tool_args, result)

    # Retrieve
    cached = await cache.get_tool_result(tool_name, tool_args)

    assert cached.hit is True
    assert cached.value["forecast"][0]["temp"] == 85
```

#### Test 3.2: Life-Safety Bypass

```python
@pytest.mark.asyncio
async def test_hurricane_alerts_bypass():
    """Test that hurricane alerts are NEVER cached (life-safety)."""
    cache = ToolResultCache(
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
    )
    await cache.initialize()

    tool_name = "get_hurricane_alerts"
    tool_args = {"location": "Tampa"}

    # Try to get (should always miss due to bypass)
    result = await cache.get_tool_result(tool_name, tool_args)

    assert result.hit is False
    assert result.bypassed is True
    assert "life-safety" in result.bypass_reason.lower() or tool_name in cache.BYPASS_TOOLS
```

#### Test 3.3: Per-Tool TTL

```python
from backend.src.cache.tool_cache.tool_cache_config import ToolCacheConfig

def test_per_tool_ttl_config():
    """Test that different tools have different TTLs."""
    config = ToolCacheConfig()

    # Weather forecast - longer TTL (stable data)
    forecast_config = config.get_config("get_forecast")
    assert forecast_config.ttl >= 1800  # At least 30 min

    # Current weather - shorter TTL (changes frequently)
    current_config = config.get_config("get_current_weather")
    assert current_config.ttl <= forecast_config.ttl

    # Hurricane alerts - disabled (life-safety)
    alert_config = config.get_config("get_hurricane_alerts")
    assert alert_config.enabled is False
```

### 4. Cached Tool Decorator Tests

Test the `@cached_tool` decorator for transparent caching.

#### Test 4.1: Decorator Usage

```python
import pytest
from backend.src.cache.tool_cache import cached_tool, set_tool_cache, ToolResultCache

@cached_tool("get_forecast")
async def mock_get_forecast(location: str, days: int = 7) -> dict:
    """Mock forecast function."""
    return {"location": location, "days": days, "forecast": "sunny"}

@pytest.mark.asyncio
async def test_cached_tool_decorator():
    """Test @cached_tool decorator caches results."""
    # Setup cache
    cache = ToolResultCache(
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
    )
    await cache.initialize()
    set_tool_cache(cache)

    # First call - cache miss, executes function
    result1 = await mock_get_forecast("Miami", days=5)
    assert result1["location"] == "Miami"

    # Second call - should be cache hit
    result2 = await mock_get_forecast("Miami", days=5)
    assert result2 == result1
```

#### Test 4.2: Force Refresh

```python
@pytest.mark.asyncio
async def test_cached_tool_force_refresh():
    """Test force_refresh bypasses cache."""
    # First call (populate cache)
    result1 = await mock_get_forecast("Boston", days=3)

    # Force refresh - should execute function again
    result2 = await mock_get_forecast("Boston", days=3, force_refresh=True)

    # Results should be same (same function)
    assert result1 == result2
```

### 5. LLM Response Cache Tests

Test caching of LLM responses with exclusion patterns.

#### Test 5.1: LLM Cache Hit

```python
from backend.src.cache.llm_cache import LLMResponseCache

@pytest.mark.asyncio
async def test_llm_cache_hit():
    """Test LLM response caching."""
    cache = LLMResponseCache(
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
    )
    await cache.initialize()

    prompt = "Explain weather patterns in Florida"
    response = "Florida experiences subtropical climate..."
    model = "claude-3-sonnet"

    # Store
    await cache.set_response(prompt, response, model=model)

    # Retrieve
    result = await cache.get_response(prompt, model=model)

    assert result.hit is True
    assert result.value == response
```

#### Test 5.2: Tool-Calling Prompt Exclusion

```python
@pytest.mark.asyncio
async def test_llm_cache_exclusion():
    """Test that tool-calling prompts are NOT cached."""
    cache = LLMResponseCache()
    await cache.initialize()

    # Agent tool-calling prompt (should be excluded)
    agent_prompt = """
    Action: get_forecast
    Action Input: {"location": "Miami"}
    """

    result = await cache.get_response(agent_prompt, model="gpt-4")

    # Should be excluded (not cached)
    assert result.hit is False
    assert result.excluded is True
```

### 6. Cache Orchestrator Integration Tests

Test the full cache orchestration flow.

#### Test 6.1: Full Cache Flow

```python
from backend.src.cache.orchestrator import CacheOrchestrator
from backend.src.cache.l1_memory_cache import QueryCache
from backend.src.cache.l2_redis_cache import RedisQueryCache
from backend.src.cache.semantic_query_cache import create_semantic_query_cache

@pytest.mark.asyncio
async def test_full_cache_flow():
    """Test Q1 → Q2 → Q3 cache flow."""
    # Setup
    l1 = QueryCache(max_size=100, ttl_seconds=300)
    l2 = RedisQueryCache(redis_url="redis://localhost:6379/0")
    await l2.connect()
    q3 = await create_semantic_query_cache()

    orchestrator = CacheOrchestrator(l1_cache=l1, l2_cache=l2, q3_cache=q3)

    query = "What's the weather in Chicago?"
    user_id = "test_user"

    # First request - all caches miss
    result1 = await orchestrator.get(query, user_id, enable_rag=True, enable_cot=True)
    assert result1 is None

    # Store response
    await orchestrator.set(query, user_id, True, True, "Windy and cold in Chicago.")

    # Second request - Q1 hit
    result2 = await orchestrator.get(query, user_id, enable_rag=True, enable_cot=True)
    assert result2 is not None
    assert result2.cache_tier == "Q1"

    # Clear Q1, should get Q2 hit
    l1.cache.clear()
    result3 = await orchestrator.get(query, user_id, enable_rag=True, enable_cot=True)
    assert result3 is not None
    assert result3.cache_tier == "Q2"
```

#### Test 6.2: Semantic Backfill

```python
@pytest.mark.asyncio
async def test_semantic_backfill():
    """Test Q3 hit backfills Q1 and Q2."""
    orchestrator = CacheOrchestrator(l1_cache=l1, l2_cache=l2, q3_cache=q3)

    # Store with one query
    query1 = "Weather forecast for New York City"
    await orchestrator.set(query1, "user1", True, True, "Rainy in NYC")

    # Clear Q1 and Q2
    l1.cache.clear()
    await l2.clear()

    # Query with similar text (should get Q3 semantic hit)
    query2 = "NYC weather forecast"
    result = await orchestrator.get(query2, "user1", True, True)

    assert result is not None
    assert result.cache_tier == "Q3"

    # Q1 should now have backfill
    result2 = await orchestrator.get(query2, "user1", True, True)
    assert result2.cache_tier == "Q1"  # Backfilled from Q3
```

### 7. Prometheus Metrics Tests

Test cache observability metrics.

#### Test 7.1: Metrics Export

```python
import requests

def test_prometheus_metrics_export():
    """Test that cache metrics are exported to Prometheus."""
    # Query Prometheus metrics endpoint
    response = requests.get("http://localhost:8000/metrics")
    assert response.status_code == 200

    metrics_text = response.text

    # Check for Level 9 cache metrics
    assert "weather_cache_hit_total" in metrics_text
    assert "weather_cache_miss_total" in metrics_text
    assert "weather_cache_latency_seconds" in metrics_text
    assert "weather_semantic_similarity_score" in metrics_text
    assert "weather_cache_cost_savings_usd" in metrics_text
```

#### Test 7.2: Cache Hit Rate Dashboard

```python
def test_grafana_dashboard_exists():
    """Test that semantic cache Grafana dashboard exists."""
    import json

    dashboard_path = "observability/grafana/provisioning/dashboards/semantic_cache_dashboard.json"

    with open(dashboard_path) as f:
        dashboard = json.load(f)

    assert dashboard["title"] == "Semantic Cache Performance"

    # Check for key panels
    panel_titles = [p["title"] for p in dashboard.get("panels", [])]
    assert "Cache Hit Rate by Tier" in panel_titles
    assert "Semantic Similarity Distribution" in panel_titles
    assert "Cost Savings" in panel_titles
```

---

## Performance Benchmarks

### Expected Performance

| Cache Tier | Latency | Hit Rate Target |
|------------|---------|-----------------|
| Q1 (Memory) | <1ms | 20-30% |
| Q2 (Redis) | <10ms | 30-40% |
| Q3 (Semantic) | <50ms | 15-25% |
| Combined | - | 65-85% |

### Benchmark Script

```python
import asyncio
import time
import statistics

async def benchmark_cache_performance(orchestrator, queries, iterations=100):
    """Benchmark cache performance."""
    latencies = {"Q1": [], "Q2": [], "Q3": [], "MISS": []}

    for _ in range(iterations):
        for query in queries:
            start = time.perf_counter()
            result = await orchestrator.get(query, "benchmark_user", True, True)
            latency_ms = (time.perf_counter() - start) * 1000

            tier = result.cache_tier if result else "MISS"
            latencies[tier].append(latency_ms)

    # Print results
    for tier, times in latencies.items():
        if times:
            print(f"{tier}: p50={statistics.median(times):.2f}ms, "
                  f"p95={statistics.quantiles(times, n=20)[-1]:.2f}ms, "
                  f"count={len(times)}")

# Run benchmark
queries = [
    "Weather in Miami",
    "Weather forecast for San Francisco",
    "NYC temperature",
    "What's the weather like in LA?",
]
asyncio.run(benchmark_cache_performance(orchestrator, queries))
```

---

## Troubleshooting

### Common Issues

#### 1. Low Semantic Hit Rate

**Symptoms**: Q3 hit rate below 10%

**Solutions**:
- Lower similarity threshold (e.g., 0.80 instead of 0.85)
- Check embedding model consistency
- Verify Qdrant collection exists

```bash
# Check Qdrant collections
curl http://localhost:6333/collections

# Check collection details
curl http://localhost:6333/collections/semantic_query_cache
```

#### 2. Cache Keys Not Matching

**Symptoms**: Identical queries getting cache misses

**Solutions**:
- Check query normalization
- Verify feature flags (enable_rag, enable_cot) match
- Check user_id consistency (Q1 is user-specific)

```python
# Debug normalization
normalizer = get_query_normalizer()
print(normalizer.normalize("Your query here"))
```

#### 3. Tool Cache Bypass Not Working

**Symptoms**: Hurricane alerts being cached

**Solutions**:
- Verify tool name matches exactly (`get_hurricane_alerts`)
- Check ToolCacheConfig.BYPASS_TOOLS set
- Verify cache is properly initialized

```python
from backend.src.cache.tool_cache.tool_cache_config import ToolCacheConfig
config = ToolCacheConfig()
print(config.BYPASS_TOOLS)  # Should include 'get_hurricane_alerts'
```

#### 4. Prometheus Metrics Not Appearing

**Symptoms**: No `weather_cache_*` metrics in /metrics

**Solutions**:
- Check CACHE_METRICS_ENABLED=true
- Verify Prometheus is scraping the API
- Check for import errors in cache_metrics.py

```bash
# Check metrics endpoint directly
curl http://localhost:8000/metrics | grep weather_cache
```

---

## Test Execution

### Run All Level 9 Tests

```bash
# Run all cache tests
pytest tests/cache/ -v

# Run with coverage
pytest tests/cache/ -v --cov=backend.src.cache --cov-report=html

# Run specific test categories
pytest tests/cache/test_semantic_query_cache.py -v
pytest tests/cache/test_tool_cache.py -v
pytest tests/cache/test_llm_cache.py -v
pytest tests/cache/test_orchestrator.py -v
```

### Integration Tests

```bash
# Start services
make docker-up-dev

# Run integration tests
pytest tests/integration/test_cache_integration.py -v

# Cleanup
make docker-down-dev
```

### Load Tests

```bash
# Run cache load test
locust -f tests/load/test_cache_load.py --host=http://localhost:8000

# Or run headless
locust -f tests/load/test_cache_load.py --host=http://localhost:8000 \
  --users 50 --spawn-rate 10 --run-time 5m --headless
```

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Overall Cache Hit Rate | ≥65% | `weather_cache_hit_total / (hit + miss)` |
| Q1 Latency (p95) | <5ms | `weather_cache_latency_seconds{tier="tier1"}` |
| Q2 Latency (p95) | <20ms | `weather_cache_latency_seconds{tier="tier2"}` |
| Q3 Latency (p95) | <100ms | Semantic similarity search |
| Hurricane Alert Bypass | 100% | Never cached |
| Cost Savings | >40% | `weather_cache_cost_savings_usd` |
| Test Coverage | ≥90% | pytest --cov |

---

## Related Documentation

- [Level 9 Plan](../plan/level-9-plan.md)
- [Cache Architecture](../knowledge/caching_semantic_cache.md)
- [Prometheus Metrics](../observability/prometheus/cache_alerts.yml)
- [Grafana Dashboard](../observability/grafana/dashboards/semantic_cache_dashboard.json)

---

*Last Updated: 2025-01-21*
*Level: 9 (Semantic Caching)*
*Status: Implementation Complete*
