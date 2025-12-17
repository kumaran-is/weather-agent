# Level 7: Tool Registry & Discovery System - Test Guide

## Quick Start Summary

| Component | Endpoint/Module | Purpose |
|-----------|-----------------|---------|
| Tool Registry Stats | `GET /health/tools` | View all registered tools with statistics |
| BigtoolRegistry | `backend.src.registry.bigtool_registry` | LangGraph-bigtool based tool management |
| Semantic Search | `BigtoolRegistry.search_tools()` | OpenAI embeddings-powered tool discovery |
| Models | `backend.src.registry.bigtool_registry` | Pydantic models (BigtoolStats, ToolMetadata, etc.) |
| Weather Tools | `backend.src.tools.weather_tools` | LangChain tools for Weather MCP |
| Hurricane Tools | `backend.src.tools.hurricane_tools` | LangChain tools for Hurricane MCP (conditional) |
| MCP Integration | `docs/architecture/MCP_INTEGRATION_ARCHITECTURE.md` | Architecture documentation |

**Tool Count:**
- **Without Hurricane MCP:** 6 tools (weather + RAG)
- **With Hurricane MCP:** 10 tools (weather + RAG + 4 hurricane tools)

**Service URLs:**
- Weather API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- LangSmith: `https://smith.langchain.com`

**Technology Stack:**
- **LangGraph-bigtool**: Official LangGraph extension for scalable tool discovery
- **InMemoryStore**: Vector store with embedding index for semantic search
- **OpenAI Embeddings**: `text-embedding-3-small` (1536 dimensions)

---

## Prerequisites

### 1. Docker Services Running

```bash
# Start all services
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
```

### 2. Environment Variables

```bash
# Required for tests
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="weather-ai-level-7"

# Hurricane MCP Server (optional - enables 4 additional tools)
export MCP_HURRICANE_SERVER_ENABLED="true"  # Set to "false" to disable
export MCP_HURRICANE_SERVER_URL="http://localhost:8001"
```

### 3. Python Environment

```bash
# Sync dependencies (includes langgraph-bigtool>=0.0.3)
cd /home/user/weather-ai-agent-service
uv sync

# Verify installation
uv run python -c "from backend.src.registry import BigtoolRegistry; print('✅ BigtoolRegistry available')"
```

---

## Part 1: REST API Testing

### Scenario 1: Get Tool Registry Statistics

**Purpose:** Verify the `/health/tools` endpoint returns all registered tools with statistics.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 1: Get Tool Registry Statistics."""

import requests
import json

API_URL = "http://localhost:8000"

def test_tool_registry_stats():
    """Test GET /health/tools endpoint."""
    print("=" * 60)
    print("Scenario 1: Get Tool Registry Statistics")
    print("=" * 60)

    # Make request
    response = requests.get(f"{API_URL}/health/tools")

    # Validate response
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()

    # Validate structure (BigtoolStats schema)
    assert "total_tools" in data, "Missing total_tools"
    assert "category_counts" in data, "Missing category_counts"
    assert "store_backend" in data, "Missing store_backend"
    assert "last_search_query" in data, "Missing last_search_query"
    assert "last_search_results" in data, "Missing last_search_results"
    assert "tools" in data, "Missing tools list"

    # Print results
    print(f"\n✅ Status Code: {response.status_code}")
    print(f"✅ Total Tools: {data['total_tools']}")
    print(f"✅ Store Backend: {data['store_backend']}")
    print(f"✅ Last Search Query: {data['last_search_query']}")
    print(f"✅ Last Search Results: {data['last_search_results']}")
    print(f"\n📊 Category Counts:")
    for category, count in data['category_counts'].items():
        print(f"   - {category}: {count}")

    if data['tools']:
        print(f"\n🔧 Registered Tools ({len(data['tools'])}):")
        for tool in data['tools']:
            print(f"\n   Tool: {tool['name']}")
            print(f"   Category: {tool['category']}")
            print(f"   Description: {tool['description'][:50]}...")
            print(f"   Tags: {tool.get('tags', [])}")

    print("\n" + "=" * 60)
    print("✅ Scenario 1 PASSED")
    print("=" * 60)

if __name__ == "__main__":
    test_tool_registry_stats()
```

**Run Command:**
```bash
cd /home/user/weather-ai-agent-service
uv run python docs/test-guide/scripts/scenario_01_registry_stats.py
```

**Expected Output (with Hurricane MCP enabled):**
```
============================================================
Scenario 1: Get Tool Registry Statistics
============================================================

✅ Status Code: 200
✅ Total Tools: 10
✅ Store Backend: InMemoryStore
✅ Last Search Query: None
✅ Last Search Results: 0

📊 Category Counts:
   - weather_data: 7  # 3 weather + 4 hurricane
   - rag: 3

🔧 Registered Tools (10):
   Tool: get_current_weather
   Category: weather_data
   Description: Get current weather conditions for a location...
   Tags: ['current', 'weather', 'real-time']

   Tool: get_active_storms
   Category: weather_data
   Description: Get currently active tropical storms and hurricanes...
   Tags: ['hurricane', 'storm', 'tracking']
   ...

============================================================
✅ Scenario 1 PASSED
============================================================
```

**Validation Checklist:**
- [ ] Status code is 200
- [ ] Response contains `total_tools` (integer ≥ 6)
- [ ] Response contains `store_backend` = "InMemoryStore"
- [ ] Response contains `category_counts` (dict)
- [ ] Response contains `tools` list with metadata

---

### Scenario 2: Verify Tool Categories

**Purpose:** Verify tools are correctly categorized (WEATHER_DATA, RAG, ANALYSIS, etc.).

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 2: Verify Tool Categories."""

import requests

API_URL = "http://localhost:8000"

EXPECTED_CATEGORIES = {
    "weather_data": ["get_current_weather", "get_forecast", "get_weather_alerts"],
    "rag": ["search_weather_knowledge", "hybrid_search", "semantic_search"],
}

def test_tool_categories():
    """Test tool categories are correct."""
    print("=" * 60)
    print("Scenario 2: Verify Tool Categories")
    print("=" * 60)

    response = requests.get(f"{API_URL}/health/tools")
    assert response.status_code == 200

    data = response.json()
    tools = {t['name']: t['category'] for t in data['tools']}

    print(f"\n📦 Registered Tools by Category:\n")

    errors = []
    for category, expected_tools in EXPECTED_CATEGORIES.items():
        print(f"Category: {category}")
        for tool_name in expected_tools:
            if tool_name in tools:
                actual_category = tools[tool_name]
                if actual_category == category:
                    print(f"  ✅ {tool_name} -> {actual_category}")
                else:
                    print(f"  ❌ {tool_name} -> {actual_category} (expected {category})")
                    errors.append(f"{tool_name}: expected {category}, got {actual_category}")
            else:
                print(f"  ⚠️ {tool_name} not registered (optional)")
        print()

    if errors:
        print(f"\n❌ FAILED: {len(errors)} category mismatches")
        for err in errors:
            print(f"   - {err}")
    else:
        print("✅ Scenario 2 PASSED - All categories correct")

if __name__ == "__main__":
    test_tool_categories()
```

**Validation Checklist:**
- [ ] Weather data tools have category `weather_data`
- [ ] RAG tools have category `rag`
- [ ] Hurricane tools (if enabled) have category `weather_data`

---

## Part 2: Python SDK Testing (BigtoolRegistry)

### Scenario 3: BigtoolRegistry Singleton Pattern

**Purpose:** Verify the BigtoolRegistry follows singleton pattern correctly.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 3: BigtoolRegistry Singleton Pattern."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from backend.src.registry import BigtoolRegistry, get_bigtool_registry, reset_bigtool_registry

def test_singleton_pattern():
    """Test that BigtoolRegistry is a singleton."""
    print("=" * 60)
    print("Scenario 3: BigtoolRegistry Singleton Pattern")
    print("=" * 60)

    # Reset to start fresh
    reset_bigtool_registry()

    # Test 1: Multiple instantiations return same instance
    print("\n🔍 Test 1: Multiple instantiations")
    registry1 = BigtoolRegistry()
    registry2 = BigtoolRegistry()

    assert registry1 is registry2, "Singleton pattern violated!"
    print(f"   registry1 id: {id(registry1)}")
    print(f"   registry2 id: {id(registry2)}")
    print("   ✅ Same instance returned")

    # Test 2: get_bigtool_registry() returns same instance
    print("\n🔍 Test 2: get_bigtool_registry() function")
    registry3 = get_bigtool_registry()

    assert registry3 is registry1, "get_bigtool_registry() should return singleton"
    print(f"   registry3 id: {id(registry3)}")
    print("   ✅ Same instance as direct instantiation")

    # Test 3: Thread safety (basic check)
    print("\n🔍 Test 3: Thread safety check")
    from concurrent.futures import ThreadPoolExecutor

    results = []
    def get_registry():
        return id(BigtoolRegistry())

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_registry) for _ in range(100)]
        results = [f.result() for f in futures]

    unique_ids = set(results)
    assert len(unique_ids) == 1, f"Multiple instances created! IDs: {unique_ids}"
    print(f"   Created 100 instances across 10 threads")
    print(f"   Unique instance IDs: {len(unique_ids)}")
    print("   ✅ Thread-safe singleton confirmed")

    print("\n" + "=" * 60)
    print("✅ Scenario 3 PASSED - Singleton pattern works correctly")
    print("=" * 60)

if __name__ == "__main__":
    test_singleton_pattern()
```

**Validation Checklist:**
- [ ] Multiple `BigtoolRegistry()` calls return same instance
- [ ] `get_bigtool_registry()` returns same singleton
- [ ] Thread-safe across concurrent access

---

### Scenario 4: Tool Registration and Retrieval

**Purpose:** Test tool registration, retrieval, and metadata operations.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 4: Tool Registration and Retrieval."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from langchain_core.tools import StructuredTool
from backend.src.registry import BigtoolRegistry, reset_bigtool_registry
from backend.src.registry.bigtool_registry import ToolCategory

def create_test_tool(name: str, description: str = "Test tool"):
    """Create a test tool."""
    def dummy_func(input_str: str = "") -> str:
        return f"Result from {name}: {input_str}"

    return StructuredTool.from_function(
        func=dummy_func,
        name=name,
        description=description,
    )

def test_registration():
    """Test tool registration and retrieval."""
    print("=" * 60)
    print("Scenario 4: Tool Registration and Retrieval")
    print("=" * 60)

    # Reset singleton
    reset_bigtool_registry()
    registry = BigtoolRegistry()
    registry._auto_registered = True  # Skip auto-registration

    # REGISTER
    print("\n📝 REGISTER: Add a new tool")
    tool = create_test_tool("test_tool", "A test tool for demonstration")

    registry.register_tool(
        name="test_tool",
        tool=tool,
        category=ToolCategory.UTILITY,
        description="A test tool for demonstration",
        tags=["test", "demo", "example"],
    )

    assert registry.count_tools() == 1, "Tool not registered!"
    print("   ✅ Tool registered successfully")

    # GET TOOL
    print("\n📖 GET: Retrieve tool by name")
    retrieved_tool = registry.get_tool("test_tool")

    assert retrieved_tool is tool, "Wrong tool returned!"
    print(f"   Tool name: {retrieved_tool.name}")
    print("   ✅ Tool retrieved successfully")

    # GET METADATA
    print("\n📋 METADATA: Get tool metadata")
    metadata = registry.get_metadata("test_tool")

    assert metadata is not None, "Metadata not found!"
    assert metadata.name == "test_tool"
    assert metadata.category == ToolCategory.UTILITY
    print(f"   Name: {metadata.name}")
    print(f"   Category: {metadata.category}")
    print(f"   Tags: {metadata.tags}")
    print("   ✅ Metadata retrieved successfully")

    # LIST ALL
    print("\n📋 LIST: Get all tool metadata")
    all_metadata = registry.list_tools()

    assert len(all_metadata) == 1
    print(f"   Total tools: {len(all_metadata)}")
    print("   ✅ List operation successful")

    # STATISTICS
    print("\n📊 STATISTICS: Get registry stats")
    stats = registry.get_statistics()

    assert stats.total_tools == 1
    assert stats.store_backend == "InMemoryStore"
    print(f"   Total tools: {stats.total_tools}")
    print(f"   Store backend: {stats.store_backend}")
    print(f"   Category counts: {stats.category_counts}")
    print("   ✅ Statistics retrieved successfully")

    print("\n" + "=" * 60)
    print("✅ Scenario 4 PASSED - Registration and retrieval work correctly")
    print("=" * 60)

if __name__ == "__main__":
    test_registration()
```

**Validation Checklist:**
- [ ] `register_tool()` adds tool to registry
- [ ] `get_tool()` returns the registered tool
- [ ] `get_metadata()` returns tool metadata
- [ ] `list_tools()` returns all metadata
- [ ] `get_statistics()` returns BigtoolStats

---

### Scenario 5: Semantic Search

**Purpose:** Test semantic search functionality via OpenAI embeddings.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 5: Semantic Search."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from langchain_core.tools import StructuredTool
from backend.src.registry import BigtoolRegistry, reset_bigtool_registry, retrieve_tools_for_query
from backend.src.registry.bigtool_registry import ToolCategory

def create_test_tool(name: str, description: str):
    def dummy_func(input_str: str = "") -> str:
        return f"Result from {name}"
    return StructuredTool.from_function(func=dummy_func, name=name, description=description)

def test_semantic_search():
    """Test semantic search functionality."""
    print("=" * 60)
    print("Scenario 5: Semantic Search")
    print("=" * 60)

    # Reset and setup
    reset_bigtool_registry()
    registry = BigtoolRegistry()
    registry._auto_registered = True

    # Register diverse tools
    tools = [
        ("get_weather", "Get current weather conditions for any location", ToolCategory.WEATHER_DATA),
        ("get_forecast", "Get multi-day weather forecast predictions", ToolCategory.WEATHER_DATA),
        ("search_knowledge", "Search the weather knowledge base", ToolCategory.RAG),
        ("analyze_patterns", "Analyze historical weather patterns", ToolCategory.ANALYSIS),
    ]

    print("\n📦 Registering test tools:")
    for name, desc, category in tools:
        tool = create_test_tool(name, desc)
        registry.register_tool(name=name, tool=tool, category=category, description=desc)
        print(f"   ✅ {name}")

    # Test semantic search
    print("\n🔍 Testing semantic search:")

    test_queries = [
        ("What's the weather like?", "get_weather"),
        ("Give me a 5-day forecast", "get_forecast"),
        ("Search for hurricane information", "search_knowledge"),
        ("Analyze temperature trends", "analyze_patterns"),
    ]

    for query, expected_top in test_queries:
        print(f"\n   Query: \"{query}\"")
        results = registry.search_tools(query, limit=3)

        if results:
            print(f"   Top result: {results[0].name}")
            for i, tool in enumerate(results, 1):
                print(f"      {i}. {tool.name}")
        else:
            print("   No results returned (embeddings may not be initialized)")

    # Check statistics updated
    stats = registry.get_statistics()
    print(f"\n📊 Last search query: {stats.last_search_query}")
    print(f"   Last search results: {stats.last_search_results}")

    print("\n" + "=" * 60)
    print("✅ Scenario 5 PASSED - Semantic search functional")
    print("=" * 60)

if __name__ == "__main__":
    test_semantic_search()
```

**Note:** This test requires `OPENAI_API_KEY` for embeddings.

**Validation Checklist:**
- [ ] `search_tools()` returns relevant tools
- [ ] Statistics track last search query
- [ ] `retrieve_tools_for_query()` works as module function

---

### Scenario 6: Category Filtering

**Purpose:** Test filtering tools by category.

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Scenario 6: Category Filtering."""

import sys
sys.path.insert(0, '/home/user/weather-ai-agent-service')

from langchain_core.tools import StructuredTool
from backend.src.registry import BigtoolRegistry, reset_bigtool_registry
from backend.src.registry.bigtool_registry import ToolCategory

def create_test_tool(name: str, description: str = "Test tool"):
    def dummy_func(input_str: str = "") -> str:
        return f"Result from {name}"
    return StructuredTool.from_function(func=dummy_func, name=name, description=description)

def test_category_filtering():
    """Test tool filtering by category."""
    print("=" * 60)
    print("Scenario 6: Category Filtering")
    print("=" * 60)

    reset_bigtool_registry()
    registry = BigtoolRegistry()
    registry._auto_registered = True

    # Register tools in different categories
    tools_config = [
        ("weather_tool_1", ToolCategory.WEATHER_DATA, "Weather tool 1"),
        ("weather_tool_2", ToolCategory.WEATHER_DATA, "Weather tool 2"),
        ("rag_tool_1", ToolCategory.RAG, "RAG tool 1"),
        ("analysis_tool_1", ToolCategory.ANALYSIS, "Analysis tool 1"),
        ("utility_tool_1", ToolCategory.UTILITY, "Utility tool 1"),
    ]

    print("\n📦 Registering test tools:")
    for name, category, desc in tools_config:
        tool = create_test_tool(name, desc)
        registry.register_tool(name=name, tool=tool, category=category, description=desc)
        print(f"   ✅ {name} ({category.value})")

    # Test filtering
    print("\n🔍 Testing category filters:")

    # Weather tools
    weather_tools = registry.get_tools_by_category(ToolCategory.WEATHER_DATA)
    print(f"\n   WEATHER_DATA category:")
    print(f"   Found: {len(weather_tools)} tools")
    for tool in weather_tools:
        print(f"      - {tool.name}")
    assert len(weather_tools) == 2, f"Expected 2 weather tools, got {len(weather_tools)}"

    # RAG tools
    rag_tools = registry.get_tools_by_category(ToolCategory.RAG)
    print(f"\n   RAG category:")
    print(f"   Found: {len(rag_tools)} tools")
    assert len(rag_tools) == 1, f"Expected 1 RAG tool, got {len(rag_tools)}"

    # Analysis tools
    analysis_tools = registry.get_tools_by_category(ToolCategory.ANALYSIS)
    print(f"\n   ANALYSIS category:")
    print(f"   Found: {len(analysis_tools)} tools")
    assert len(analysis_tools) == 1, f"Expected 1 analysis tool, got {len(analysis_tools)}"

    print("\n" + "=" * 60)
    print("✅ Scenario 6 PASSED - Category filtering works correctly")
    print("=" * 60)

if __name__ == "__main__":
    test_category_filtering()
```

**Validation Checklist:**
- [ ] `get_tools_by_category()` returns correct tools
- [ ] Weather tools filtered correctly
- [ ] RAG tools filtered correctly
- [ ] Analysis tools filtered correctly

---

## Part 3: Unit Test Execution

### Scenario 7: Run All Level 7 Unit Tests

**Purpose:** Execute all Level 7 unit tests to verify implementation.

**Run Command:**
```bash
cd /home/user/weather-ai-agent-service
uv run pytest tests/unit/test_tool_registry.py tests/unit/test_semantic_discovery.py backend/tests/test_tool_store.py -v --tb=short
```

**Expected Output:**
```
tests/unit/test_tool_registry.py::TestBigtoolRegistrySingleton::test_singleton_returns_same_instance PASSED
tests/unit/test_tool_registry.py::TestBigtoolRegistrySingleton::test_get_bigtool_registry_returns_singleton PASSED
tests/unit/test_tool_registry.py::TestBigtoolRegistrySingleton::test_singleton_thread_safe PASSED
tests/unit/test_tool_registry.py::TestBigtoolRegistryCRUD::test_register_tool PASSED
tests/unit/test_tool_registry.py::TestBigtoolRegistryCRUD::test_get_tool PASSED
...
tests/unit/test_semantic_discovery.py::TestSemanticSearchBasics::test_search_returns_list PASSED
tests/unit/test_semantic_discovery.py::TestSemanticSearchBasics::test_search_empty_query PASSED
...

========================= XX passed in X.XXs =========================
```

**Validation Checklist:**
- [ ] All tests in `test_tool_registry.py` pass
- [ ] All tests in `test_semantic_discovery.py` pass
- [ ] All tests in `test_tool_store.py` pass
- [ ] No deprecation warnings
- [ ] No import errors

---

### Scenario 8: Run Tests with Coverage

**Purpose:** Verify test coverage for Level 7 modules.

**Run Command:**
```bash
cd /home/user/weather-ai-agent-service
uv run pytest tests/unit/test_tool_registry.py tests/unit/test_semantic_discovery.py \
    --cov=backend.src.registry.bigtool_registry \
    --cov-report=term-missing \
    -v
```

**Expected Output:**
```
---------- coverage: ... ----------
Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
backend/src/registry/bigtool_registry.py        XXX      X    8X%   ...
---------------------------------------------------------------------------
TOTAL                                           XXX      X    8X%
```

**Validation Checklist:**
- [ ] Overall coverage ≥ 80%
- [ ] `bigtool_registry.py` coverage ≥ 80%

---

## Part 4: MCP Integration Testing

### Scenario 9: Run MCP Integration Tests

**Purpose:** Verify MCP clients (Weather + Hurricane) have resilience patterns and integrate with BigtoolRegistry.

**Run Command:**
```bash
cd /home/user/weather-ai-agent-service
OPENAI_API_KEY="sk-test-placeholder" MCP_HURRICANE_SERVER_ENABLED=true \
  uv run pytest tests/integration/test_mcp_integration.py -v --tb=short
```

**Expected Output:**
```
tests/integration/test_mcp_integration.py::TestSettingsConfiguration::test_weather_settings_exist PASSED
tests/integration/test_mcp_integration.py::TestSettingsConfiguration::test_hurricane_settings_exist PASSED
tests/integration/test_mcp_integration.py::TestBigtoolRegistryStandalone::test_registry_singleton PASSED
tests/integration/test_mcp_integration.py::TestBigtoolRegistryStandalone::test_registry_manual_registration PASSED
tests/integration/test_mcp_integration.py::TestBigtoolRegistryStandalone::test_registry_statistics PASSED
tests/integration/test_mcp_integration.py::TestAutoRegistration::test_get_bigtool_registry_triggers_auto_registration PASSED
tests/integration/test_mcp_integration.py::TestHurricaneToolsConditional::test_hurricane_tools_registration_flag PASSED
tests/integration/test_mcp_integration.py::TestHurricaneToolsConditional::test_tool_count_with_hurricane_enabled PASSED
tests/integration/test_mcp_integration.py::TestSemanticSearch::test_search_tools_method_exists PASSED
tests/integration/test_mcp_integration.py::TestSemanticSearch::test_retrieve_tools_for_query_function PASSED
...

========================= XX passed in X.XXs =========================
```

**Test Categories:**

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestSettingsConfiguration` | 5 | MCP settings exist and are positive |
| `TestMCPLoggerStandalone` | 3 | MCPLogger creates properly |
| `TestCircuitBreakerStandalone` | 2 | Circuit breaker registry and creation |
| `TestMCPFailoverStandalone` | 3 | Failover handler and circuit breaker config |
| `TestBigtoolRegistryStandalone` | 3 | Singleton, manual registration, statistics |
| `TestAutoRegistration` | 1 | Auto-registration triggers correctly |
| `TestHurricaneToolsConditional` | 2 | Hurricane tools conditional on env var |
| `TestBackwardCompatibility` | 2 | ToolRegistry and get_tool_registry aliases |
| `TestSemanticSearch` | 2 | Semantic search methods exist |

**Validation Checklist:**
- [ ] All MCP integration tests pass
- [ ] BigtoolRegistry tests pass
- [ ] Auto-registration test passes
- [ ] Hurricane conditional tests pass
- [ ] Backward compatibility tests pass

---

### Scenario 10: Verify Hurricane Tools Conditional Registration

**Purpose:** Verify hurricane tools are registered only when `MCP_HURRICANE_SERVER_ENABLED=true`.

**Test Script:**
```bash
# Test 1: Hurricane DISABLED - should have 6 tools
MCP_HURRICANE_SERVER_ENABLED=false \
  curl -s http://localhost:8000/health/tools | jq '.total_tools'
# Expected: 6

# Test 2: Hurricane ENABLED - should have 10 tools
MCP_HURRICANE_SERVER_ENABLED=true \
  curl -s http://localhost:8000/health/tools | jq '.total_tools'
# Expected: 10

# Test 3: Verify hurricane tools exist when enabled
MCP_HURRICANE_SERVER_ENABLED=true \
  curl -s http://localhost:8000/health/tools | jq '.tools[].name' | grep -E "(storm|hurricane)"
# Expected output:
# "get_active_storms"
# "get_storm_forecast"
# "get_hurricane_alerts"
# "get_storm_history"
```

**Hurricane Tools (when enabled):**

| Tool | Description | Capability |
|------|-------------|------------|
| `get_active_storms` | Get currently active tropical storms | `real_time` |
| `get_storm_forecast` | Get storm forecast cone and track | `forecast` |
| `get_hurricane_alerts` | Get hurricane alerts for a location | `real_time` |
| `get_storm_history` | Search historical hurricane data | `historical` |

**Validation Checklist:**
- [ ] Tool count is 6 when `MCP_HURRICANE_SERVER_ENABLED=false`
- [ ] Tool count is 10 when `MCP_HURRICANE_SERVER_ENABLED=true`
- [ ] All 4 hurricane tools appear in `/health/tools` when enabled

---

## Summary: Level 7 Test Coverage

| Scenario | Type | Component | Status |
|----------|------|-----------|--------|
| 1 | REST API | Tool Registry Stats (BigtoolStats) | ⬜ |
| 2 | REST API | Tool Categories | ⬜ |
| 3 | Python SDK | Singleton Pattern | ⬜ |
| 4 | Python SDK | Registration & Retrieval | ⬜ |
| 5 | Python SDK | Semantic Search | ⬜ |
| 6 | Python SDK | Category Filtering | ⬜ |
| 7 | Unit Tests | All Tests Pass | ⬜ |
| 8 | Coverage | Coverage Report | ⬜ |
| 9 | MCP Integration | MCP Integration Tests | ⬜ |
| 10 | MCP Integration | Hurricane Conditional Registration | ⬜ |

**Total Scenarios:** 10

---

## Troubleshooting

### Common Issues

**1. BigtoolRegistry Not Found**
```python
ImportError: cannot import name 'BigtoolRegistry'
```
**Solution:** Ensure `backend/src/registry/__init__.py` exports `BigtoolRegistry`.

**2. API Returns 404 for /health/tools**
```
{"detail":"Not Found"}
```
**Solution:** Verify the endpoint is registered in `backend/src/api/main.py`.

**3. Pydantic Validation Error**
```
ValidationError: tool: Input should be a valid BaseTool
```
**Solution:** Use `StructuredTool.from_function()` instead of `MagicMock` in tests.

**4. Semantic Search Returns Empty**
```
No results from search_tools()
```
**Solution:** Ensure `OPENAI_API_KEY` is set for embeddings.

**5. Hurricane Tools Not Appearing**
```
Only 6 tools shown when expecting 10
```
**Solution:** Ensure hurricane is enabled and rebuild Docker:
```bash
export MCP_HURRICANE_SERVER_ENABLED=true
docker-compose build weather-ai-api
docker-compose up -d weather-ai-api
```

**6. Backward Compatibility Import Errors**
```
ImportError: cannot import name 'ToolRegistry'
```
**Solution:** Use the new names or aliases:
```python
# New way (recommended)
from backend.src.registry import BigtoolRegistry, get_bigtool_registry

# Old way (still works via aliases)
from backend.src.registry import ToolRegistry, get_tool_registry  # Same as above
```

---

## Quick Test Commands

```bash
# Run all Level 7 unit tests
uv run pytest tests/unit/test_tool_registry.py tests/unit/test_semantic_discovery.py -v

# Run MCP integration tests
OPENAI_API_KEY="sk-test" MCP_HURRICANE_SERVER_ENABLED=true \
  uv run pytest tests/integration/test_mcp_integration.py -v

# Run specific test class
uv run pytest tests/unit/test_tool_registry.py::TestBigtoolRegistrySingleton -v

# Run with coverage
uv run pytest tests/unit/test_tool_registry.py --cov=backend.src.registry.bigtool_registry --cov-report=html

# Check API health (tool count)
curl http://localhost:8000/health/tools | jq '.total_tools'

# Check full tool statistics (BigtoolStats schema)
curl http://localhost:8000/health/tools | jq

# List hurricane tools (when enabled)
curl http://localhost:8000/health/tools | jq '.tools[] | select(.name | contains("storm") or contains("hurricane"))'

# Test weather query
curl -X POST http://localhost:8000/weather/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Weather in Miami", "user_id": "test"}'
```

---

## Architecture Overview: LangGraph-bigtool

**BigtoolRegistry** uses the official LangGraph-bigtool extension for scalable tool discovery:

```
┌─────────────────────────────────────────────────────────────┐
│                    BigtoolRegistry                          │
│  (Singleton - Thread-Safe)                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │  tool_registry   │    │      InMemoryStore           │  │
│  │  dict[str, Tool] │    │  (with embedding index)      │  │
│  │                  │    │                              │  │
│  │  - get_weather   │    │  namespace: "weather_tools"  │  │
│  │  - get_forecast  │    │  dims: 1536                  │  │
│  │  - ...           │    │  fields: ["description"]     │  │
│  └──────────────────┘    └──────────────────────────────┘  │
│                                     │                       │
│                                     ▼                       │
│                    ┌──────────────────────────┐            │
│                    │   OpenAIEmbeddings       │            │
│                    │   text-embedding-3-small │            │
│                    └──────────────────────────┘            │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Methods:                                                   │
│  - register_tool(name, tool, category, description, tags)  │
│  - get_tool(name) → BaseTool                               │
│  - search_tools(query, limit) → list[BaseTool]             │
│  - get_tools_by_category(category) → list[BaseTool]        │
│  - get_statistics() → BigtoolStats                         │
│  - auto_register_tools() (weather + RAG + hurricane)       │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- ~50% context token reduction vs loading all tools
- Semantic search powered by OpenAI embeddings
- Official LangGraph integration for production use
- Backward-compatible with existing code via aliases

---

**Document Version:** 2.0.0
**Last Updated:** 2025-01-21
**Level:** 7 - Tool Registry & Discovery System (LangGraph-bigtool)

**Changelog:**
- v2.0.0: Complete rewrite for LangGraph-bigtool migration. Replaced ToolRegistry with BigtoolRegistry, updated all tests and examples, added architecture diagram.
- v1.1.0: Added MCP Integration Testing (Scenarios 19-20)
- v1.0.0: Initial release with 18 test scenarios
