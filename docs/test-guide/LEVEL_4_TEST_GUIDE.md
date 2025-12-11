# Comprehensive Testing Guide: Auto-Routing Architecture (v0.6.0)

**Purpose**: Manual testing via Swagger UI and LangSmith Studio to validate the intelligent auto-routing system that replaces explicit `use_multi_agent` and `agent_level` flags.

**Duration**: 30-45 minutes for complete testing
**Date**: December 11, 2025
**Version**: v0.6.0
**Status**: ✅ **28/28 UNIT TESTS PASSING** (100% coverage on routing module)

---

## ⚡ Quick Start Summary

### What Changed in v0.6.0

| Before (v0.5.0) | After (v0.6.0) |
|-----------------|----------------|
| `use_multi_agent: true/false` | **REMOVED** - Auto-detected |
| `agent_level: "basic/l4a/l4b/l4c/auto"` | **REMOVED** - Auto-routed |
| User must choose routing | System classifies intent |

### Auto-Routing Tiers

| Tier | Agent Level | Query Examples | Cost |
|------|-------------|----------------|------|
| **SIMPLE** | `basic` | "Weather in London", "Temperature today" | $0.001/query |
| **STANDARD** | `l4a` (3-agent) | "Track Hurricane Milton", "Storm surge forecast" | $0.015/query |
| **COMPLEX** | `l4b` (8-agent) | "Compare hurricanes", "Historical analysis" | $0.035/query |
| **EMERGENCY** | `l4c` (15-agent) | "Should I evacuate?", "Am I safe?" | $0.10/query |

---

## 🎯 What We're Testing

### **Auto-Routing Classification**:
1. ✅ SIMPLE tier - Basic weather queries → Basic agent
2. ✅ STANDARD tier - Hurricane/storm queries → L4A (3-agent)
3. ✅ COMPLEX tier - Analysis/comparison queries → L4B (8-agent)
4. ✅ EMERGENCY tier - Safety/evacuation queries → L4C (15-agent)

### **Signal Detection**:
1. ✅ Emergency keywords (evacuate, danger, life-threatening)
2. ✅ Complex keywords (compare, analyze, historical, forecast)
3. ✅ Storm keywords (hurricane, tropical storm, NHC, landfall)
4. ✅ Multi-question detection (2+ question marks)
5. ✅ Context escalation (follow-up to emergency conversation)

### **Fallback Behavior**:
1. ✅ L4B unavailable → Falls back to L4A
2. ✅ L4A unavailable → Falls back to basic agent
3. ✅ Graceful degradation with logging

### **Response Metadata**:
1. ✅ `agent_level` reflects actual tier used
2. ✅ `query_complexity` matches routing decision
3. ✅ `agents_invoked` shows participating agents

---

## 📋 Prerequisites

### 1. Start All Services

```bash
# Start Docker services
docker-compose -f docker-compose.dev.yml up -d

# Verify all services healthy
docker ps --filter "name=weather"

# Expected services:
# - weather-ai-api-dev (FastAPI)
# - weather-ai-redis (caching)
# - weather-ai-neo4j (memory)
# - weather-ai-qdrant (RAG)
# - weather-mcp (weather data)
# - hurricane-mcp (hurricane tracking)
```

### 2. Open Swagger UI

**URL**: http://localhost:8000/docs

**What you'll see**:
- API version: **0.6.0**
- Description mentions **AUTO-ROUTING**
- No `use_multi_agent` or `agent_level` parameters in `/weather/query`

### 3. Setup LangSmith (Recommended)

```bash
# Add to .env file
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_api_key_here
export LANGCHAIN_PROJECT=weather-ai-agent-v0.6.0-routing

# Restart API container
docker-compose restart weather-ai-api
```

**Access LangSmith Studio**: https://smith.langchain.com/

---

## 🧪 Testing Scenarios

---

## **SCENARIO 1: SIMPLE Tier - Basic Weather Query**

### Purpose
Verify simple weather queries route to basic single agent.

### Steps

**1. Open Swagger UI**: http://localhost:8000/docs

**2. Find Endpoint**: POST `/weather/query`

**3. Click "Try it out"**

**4. Paste Request Body**:
```json
{
  "query": "What's the weather in London?",
  "user_id": "test_simple_001"
}
```

**Note**: No `use_multi_agent` or `agent_level` fields - they're auto-detected!

**5. Click "Execute"**

### Expected Response

**Status Code**: 200

**Response Body** (sample):
```json
{
  "response": "The current weather in London is:\n\n- Temperature: 12°C (54°F)\n- Condition: Partly cloudy\n- Humidity: 78%\n- Wind: 15 km/h from the west",
  "user_id": "test_simple_001",
  "timestamp": "2025-12-11T14:00:00Z",
  "agents_invoked": ["weather_agent"],
  "agent_level": "basic",
  "query_complexity": "simple",
  "execution_time_ms": 1523.5,
  "cache_hit": false,
  "cache_layer": "MISS"
}
```

### Validation Checklist
- [ ] `agent_level` is `"basic"`
- [ ] `query_complexity` is `"simple"`
- [ ] `agents_invoked` contains `["weather_agent"]`
- [ ] Response is concise (weather data only)
- [ ] Duration: < 5 seconds

### LangSmith Validation
- Navigate to LangSmith → Projects → weather-ai-agent-v0.6.0-routing
- Find trace for "test_simple_001"
- Check:
  - [ ] Log shows: `🎯 Auto-routing decision | tier: simple | agent_level: basic`
  - [ ] Single agent invocation
  - [ ] No multi-agent orchestration

---

## **SCENARIO 2: SIMPLE Tier - Various Simple Queries**

### Purpose
Verify multiple simple query patterns all route to SIMPLE tier.

### Test Queries (run each separately)

**Query A**: Temperature query
```json
{
  "query": "Temperature in New York today",
  "user_id": "test_simple_002a"
}
```

**Query B**: Rain query
```json
{
  "query": "Will it rain tomorrow in Seattle?",
  "user_id": "test_simple_002b"
}
```

**Query C**: Sunny query
```json
{
  "query": "Is it sunny in Miami?",
  "user_id": "test_simple_002c"
}
```

**Query D**: Weekend query
```json
{
  "query": "How warm will it be this weekend?",
  "user_id": "test_simple_002d"
}
```

### Validation Checklist (for each)
- [ ] All queries return `agent_level: "basic"`
- [ ] All queries return `query_complexity: "simple"`
- [ ] Fast response times (< 5 seconds)

---

## **SCENARIO 3: STANDARD Tier - Hurricane Query**

### Purpose
Verify hurricane/storm queries route to L4A (3-agent system).

### Steps

**1. Request Body**:
```json
{
  "query": "Is Hurricane Milton going to hit Tampa?",
  "user_id": "test_standard_001"
}
```

**2. Click "Execute"**

### Expected Response

**Status Code**: 200

**Response Body** (sample):
```json
{
  "response": "Based on the latest NHC advisory, Hurricane Milton is currently...\n\n**Current Status:**\n- Category: 4\n- Wind Speed: 145 mph\n- Location: 150 miles west of Tampa\n\n**Forecast:**\n- Expected landfall: Wednesday evening\n- Tampa Bay area is in the cone of uncertainty\n\n**Recommendations:**\n- Monitor official NHC updates\n- Prepare evacuation plans if in flood zones",
  "user_id": "test_standard_001",
  "timestamp": "2025-12-11T14:05:00Z",
  "agents_invoked": ["triage", "hurricane_specialist", "alert_manager"],
  "agent_level": "l4a",
  "query_complexity": "standard",
  "execution_time_ms": 8923.5,
  "cache_hit": false,
  "cache_layer": "MISS"
}
```

### Validation Checklist
- [ ] `agent_level` is `"l4a"`
- [ ] `query_complexity` is `"standard"`
- [ ] `agents_invoked` includes hurricane-related agents
- [ ] Response includes NHC data
- [ ] Response includes safety recommendations

### LangSmith Validation
- Check trace shows:
  - [ ] `🎯 Auto-routing decision | tier: standard | agent_level: l4a`
  - [ ] Triage agent classification
  - [ ] Hurricane specialist invocation
  - [ ] Alert manager (if high risk)

---

## **SCENARIO 4: STANDARD Tier - Various Storm Queries**

### Purpose
Verify multiple storm-related patterns route to STANDARD tier.

### Test Queries

**Query A**: Track hurricane
```json
{
  "query": "Track Hurricane Ian",
  "user_id": "test_standard_002a"
}
```

**Query B**: NHC advisory
```json
{
  "query": "What's the NHC advisory for the tropical storm?",
  "user_id": "test_standard_002b"
}
```

**Query C**: Storm surge
```json
{
  "query": "Storm surge predictions for Miami",
  "user_id": "test_standard_002c"
}
```

**Query D**: Landfall timing
```json
{
  "query": "When will the hurricane make landfall?",
  "user_id": "test_standard_002d"
}
```

**Query E**: Category question
```json
{
  "query": "What category is the storm?",
  "user_id": "test_standard_002e"
}
```

### Validation Checklist
- [ ] All queries return `agent_level: "l4a"`
- [ ] All queries return `query_complexity: "standard"`
- [ ] Multi-agent orchestration visible in traces

---

## **SCENARIO 5: COMPLEX Tier - Analysis Query**

### Purpose
Verify analysis/comparison queries route to L4B (8-agent system).

### Steps

**1. Request Body**:
```json
{
  "query": "Compare Hurricane Milton to Hurricane Ian patterns",
  "user_id": "test_complex_001"
}
```

**2. Click "Execute"**

### Expected Response

**Status Code**: 200

**Response Body** (sample):
```json
{
  "response": "## Hurricane Comparison: Milton vs Ian\n\n### Similarities:\n1. Both made landfall in Florida\n2. Both reached Category 4+ intensity\n3. Similar rapid intensification patterns\n\n### Key Differences:\n| Metric | Hurricane Milton | Hurricane Ian |\n|--------|------------------|---------------|\n| Peak Category | 5 (180 mph) | 4 (150 mph) |\n| Landfall Location | Tampa Bay | Fort Myers |\n| Storm Surge | 10-15 ft expected | 12-18 ft actual |\n| Evacuation Impact | Ongoing | 2.5M displaced |\n\n### Historical Context:\n...",
  "user_id": "test_complex_001",
  "timestamp": "2025-12-11T14:10:00Z",
  "agents_invoked": ["supervisor", "triage", "hurricane_specialist", "historical_analyst", "forecaster"],
  "agent_level": "l4b",
  "query_complexity": "complex",
  "execution_time_ms": 15234.5,
  "cache_hit": false,
  "cache_layer": "MISS"
}
```

### Validation Checklist
- [ ] `agent_level` is `"l4b"`
- [ ] `query_complexity` is `"complex"`
- [ ] `agents_invoked` includes multiple specialists
- [ ] Response includes comparative analysis
- [ ] Response includes historical data

### LangSmith Validation
- Check trace shows:
  - [ ] `🎯 Auto-routing decision | tier: complex | agent_level: l4b`
  - [ ] Supervisor orchestration
  - [ ] Parallel agent execution
  - [ ] Historical analyst invocation

---

## **SCENARIO 6: COMPLEX Tier - Various Analysis Queries**

### Purpose
Verify multiple analysis patterns route to COMPLEX tier.

### Test Queries

**Query A**: Historical patterns
```json
{
  "query": "Analyze the historical pattern of hurricanes in Florida",
  "user_id": "test_complex_002a"
}
```

**Query B**: Trend analysis
```json
{
  "query": "What's the trend for Atlantic hurricanes over 10 years?",
  "user_id": "test_complex_002b"
}
```

**Query C**: Multi-day forecast
```json
{
  "query": "Forecast next 7 days with detailed analysis",
  "user_id": "test_complex_002c"
}
```

**Query D**: Multi-question
```json
{
  "query": "What's the weather today? And will it rain tomorrow? What about the weekend?",
  "user_id": "test_complex_002d"
}
```

### Validation Checklist
- [ ] All queries return `agent_level: "l4b"` (or `"l4a"` fallback)
- [ ] All queries return `query_complexity: "complex"`
- [ ] Detailed analysis in responses

---

## **SCENARIO 7: EMERGENCY Tier - Evacuation Query**

### Purpose
Verify safety/evacuation queries route to L4C (15-agent system with HITL).

### Steps

**1. Request Body**:
```json
{
  "query": "Should I evacuate from Tampa Beach?",
  "user_id": "test_emergency_001"
}
```

**2. Click "Execute"**

### Expected Response

**Status Code**: 200

**Response Body** (sample):
```json
{
  "response": "## ⚠️ EVACUATION ASSESSMENT\n\n**Location**: Tampa Beach, FL\n**Risk Level**: HIGH\n\n### Current Threat:\n- Hurricane Milton (Category 4) approaching\n- Landfall expected: Wednesday 8 PM EDT\n- Storm surge forecast: 10-15 feet\n\n### Evacuation Recommendation:\n**YES - EVACUATE IMMEDIATELY**\n\n### Reasons:\n1. Tampa Beach is in Evacuation Zone A\n2. Storm surge >6 feet predicted for your area\n3. Category 4+ winds will be life-threatening\n\n### Action Items:\n1. Leave within the next 6-12 hours\n2. Head north or inland (I-75 north recommended)\n3. Take essential documents, medications, 3-day supplies\n\n### Emergency Contacts:\n- Hillsborough County Emergency: (813) 272-5900\n- Florida Emergency Hotline: 1-800-342-3557\n\n**Your safety is the priority. Do not delay evacuation.**",
  "user_id": "test_emergency_001",
  "timestamp": "2025-12-11T14:15:00Z",
  "agents_invoked": ["supervisor", "triage", "hurricane_specialist", "emergency_response", "alert_manager", "verification"],
  "agent_level": "l4c",
  "query_complexity": "emergency",
  "execution_time_ms": 22456.5,
  "cache_hit": false,
  "cache_layer": "MISS"
}
```

### Validation Checklist
- [ ] `agent_level` is `"l4c"`
- [ ] `query_complexity` is `"emergency"`
- [ ] `agents_invoked` includes emergency response agents
- [ ] Response includes evacuation recommendation
- [ ] Response includes specific action items
- [ ] Response includes emergency contacts

### LangSmith Validation
- Check trace shows:
  - [ ] `🎯 Auto-routing decision | tier: emergency | agent_level: l4c`
  - [ ] Full 15-agent orchestration
  - [ ] Emergency response agent invocation
  - [ ] Verification/reflection cycle

---

## **SCENARIO 8: EMERGENCY Tier - Various Safety Queries**

### Purpose
Verify multiple emergency patterns route to EMERGENCY tier.

### Test Queries

**Query A**: Am I safe
```json
{
  "query": "Am I safe in Miami Beach?",
  "user_id": "test_emergency_002a"
}
```

**Query B**: Danger assessment
```json
{
  "query": "Is it dangerous to stay during the hurricane?",
  "user_id": "test_emergency_002b"
}
```

**Query C**: Life-threatening
```json
{
  "query": "Are conditions life-threatening?",
  "user_id": "test_emergency_002c"
}
```

**Query D**: Shelter in place
```json
{
  "query": "Should I shelter in place or evacuate?",
  "user_id": "test_emergency_002d"
}
```

**Query E**: Mandatory evacuation
```json
{
  "query": "Is there a mandatory evacuation order?",
  "user_id": "test_emergency_002e"
}
```

### Validation Checklist
- [ ] All queries return `agent_level: "l4c"`
- [ ] All queries return `query_complexity: "emergency"`
- [ ] All responses include safety recommendations

---

## **SCENARIO 9: Edge Cases - Priority Override**

### Purpose
Verify emergency keywords override other classifications.

### Test A: Emergency + Complex keywords
```json
{
  "query": "Analyze whether I should evacuate based on historical hurricane patterns",
  "user_id": "test_priority_001"
}
```

**Expected**: `agent_level: "l4c"` (emergency overrides complex)

### Test B: Emergency + Standard keywords
```json
{
  "query": "Should I evacuate because of the hurricane?",
  "user_id": "test_priority_002"
}
```

**Expected**: `agent_level: "l4c"` (emergency overrides standard)

### Test C: Complex + Standard keywords
```json
{
  "query": "Compare this hurricane to historical patterns",
  "user_id": "test_priority_003"
}
```

**Expected**: `agent_level: "l4b"` (complex overrides standard)

### Validation Checklist
- [ ] Priority order: EMERGENCY > COMPLEX > STANDARD > SIMPLE
- [ ] Emergency keywords always win

---

## **SCENARIO 10: Edge Cases - Case Insensitivity**

### Purpose
Verify keyword matching is case-insensitive.

### Test Queries

**Query A**: ALL CAPS
```json
{
  "query": "HURRICANE MILTON",
  "user_id": "test_case_001a"
}
```

**Query B**: lowercase
```json
{
  "query": "hurricane milton",
  "user_id": "test_case_001b"
}
```

**Query C**: Mixed case
```json
{
  "query": "HuRrIcAnE MiLtOn",
  "user_id": "test_case_001c"
}
```

### Validation Checklist
- [ ] All queries return same `agent_level`
- [ ] Case variations don't affect classification

---

## **SCENARIO 11: Edge Cases - False Positive Prevention**

### Purpose
Verify partial keywords don't trigger false positives.

### Test Queries

**Query A**: Cat (not category)
```json
{
  "query": "I have a cat named Stormy",
  "user_id": "test_false_001a"
}
```

**Expected**: `agent_level: "basic"` (SIMPLE, not STANDARD)

**Query B**: Evacuate in different context
```json
{
  "query": "How long did it take to evacuate the stadium?",
  "user_id": "test_false_001b"
}
```

**Expected**: May trigger EMERGENCY (keyword present) - edge case to monitor

### Validation Checklist
- [ ] "cat" alone doesn't match "cat 4" or "category"
- [ ] Monitor false positives for improvement

---

## **SCENARIO 12: Context Escalation (with Memory)**

### Purpose
Verify context from previous conversation affects routing.

### Part A: Establish Emergency Context

**Request 1**:
```json
{
  "query": "Should I evacuate from Tampa?",
  "user_id": "test_context_001",
  "session_id": "session_context_001",
  "use_memory": true
}
```

**Expected**: `agent_level: "l4c"` (emergency)

### Part B: Follow-up Query (same session)

**Request 2** (run immediately after):
```json
{
  "query": "What about my pets?",
  "user_id": "test_context_001",
  "session_id": "session_context_001",
  "use_memory": true
}
```

**Expected**: May maintain elevated tier due to context escalation

### Validation Checklist
- [ ] First query: EMERGENCY tier
- [ ] Follow-up: Check if tier maintained
- [ ] Context signals visible in logs

---

## **SCENARIO 13: Cache Behavior with Auto-Routing**

### Purpose
Verify caching works correctly with auto-routing.

### Part A: First Query (Cache Miss)

```json
{
  "query": "Track Hurricane Milton",
  "user_id": "test_cache_001"
}
```

**Expected**:
- `cache_hit: false`
- `cache_layer: "MISS"`
- `agent_level: "l4a"`

### Part B: Repeat Query (Cache Hit)

Run exact same query again.

**Expected**:
- `cache_hit: true`
- `cache_layer: "L1"` or `"L2"`
- Response returned faster

### Validation Checklist
- [ ] First request: Cache miss, full agent invocation
- [ ] Second request: Cache hit, fast response
- [ ] Routing classification still logged (before cache check)

---

## **SCENARIO 14: API Response Validation**

### Purpose
Verify response metadata is complete and accurate.

### Request

```json
{
  "query": "Is Hurricane Milton going to hit Tampa?",
  "user_id": "test_metadata_001"
}
```

### Required Response Fields

| Field | Type | Expected Value |
|-------|------|----------------|
| `response` | string | Non-empty weather/hurricane data |
| `user_id` | string | `"test_metadata_001"` |
| `timestamp` | string | ISO 8601 format |
| `agents_invoked` | array | Non-empty list |
| `agent_level` | string | `"basic"`, `"l4a"`, `"l4b"`, or `"l4c"` |
| `query_complexity` | string | `"simple"`, `"standard"`, `"complex"`, or `"emergency"` |
| `execution_time_ms` | number | > 0 |
| `cache_hit` | boolean | `true` or `false` |
| `cache_layer` | string | `"L1"`, `"L2"`, or `"MISS"` |

### Validation Checklist
- [ ] All required fields present
- [ ] Types match expectations
- [ ] Values are consistent (e.g., `l4a` → `standard`)

---

## **SCENARIO 15: Error Handling**

### Purpose
Verify graceful error handling.

### Test A: Empty query
```json
{
  "query": "",
  "user_id": "test_error_001"
}
```

**Expected**: 422 Validation Error (min_length=1)

### Test B: Query too long (>500 chars)
```json
{
  "query": "What is the weather in London? [repeat 100 times]...",
  "user_id": "test_error_002"
}
```

**Expected**: 422 Validation Error (max_length=500)

### Test C: Missing user_id
```json
{
  "query": "What's the weather?"
}
```

**Expected**: 422 Validation Error (required field)

### Validation Checklist
- [ ] Validation errors return 422
- [ ] Error messages are clear
- [ ] No server crashes

---

## 📊 LangSmith Studio Validation

### How to Use LangSmith for Auto-Routing Testing

**Step 1**: Navigate to LangSmith
- URL: https://smith.langchain.com/
- Select Project: `weather-ai-agent-v0.6.0-routing`

**Step 2**: Find Traces by User ID
- Use the search/filter feature
- Filter by `user_id` (e.g., `test_simple_001`)

**Step 3**: Validate Routing Decision
For each trace, look for log entries containing:
```
🎯 Auto-routing decision | tier: <TIER> | agent_level: <LEVEL> | confidence: <SCORE> | rule: <RULE_NAME>
```

**Step 4**: Check Agent Invocations
- Expand the trace tree
- Verify expected agents were called
- Check tool calls (MCP weather, hurricane tools)

### LangSmith Validation Checklist

| Scenario | Expected Tier | Expected Agents |
|----------|--------------|-----------------|
| Simple weather | `simple` | `weather_agent` |
| Hurricane track | `standard` | `triage`, `hurricane_specialist` |
| Compare hurricanes | `complex` | `supervisor`, `historical_analyst`, `forecaster` |
| Evacuate query | `emergency` | `emergency_response`, `verification` |

---

## 📈 Performance Expectations

### Response Time by Tier

| Tier | Expected Time | Max Acceptable |
|------|---------------|----------------|
| SIMPLE | 1-3 seconds | 5 seconds |
| STANDARD | 5-10 seconds | 15 seconds |
| COMPLEX | 10-20 seconds | 30 seconds |
| EMERGENCY | 15-30 seconds | 45 seconds |

### Cost by Tier

| Tier | Estimated Cost | Notes |
|------|---------------|-------|
| SIMPLE | $0.001/query | Single agent |
| STANDARD | $0.015/query | 3 agents |
| COMPLEX | $0.035/query | 8 agents |
| EMERGENCY | $0.10/query | 15 agents + HITL potential |

---

## ✅ Testing Summary Checklist

### Auto-Routing Classification
- [ ] SIMPLE tier works for basic weather queries
- [ ] STANDARD tier works for hurricane queries
- [ ] COMPLEX tier works for analysis queries
- [ ] EMERGENCY tier works for safety queries

### Priority Override
- [ ] Emergency overrides complex
- [ ] Emergency overrides standard
- [ ] Complex overrides standard

### Edge Cases
- [ ] Case-insensitive matching works
- [ ] Partial keywords don't false-positive
- [ ] Context escalation works with memory

### Response Metadata
- [ ] `agent_level` accurate
- [ ] `query_complexity` accurate
- [ ] `agents_invoked` complete
- [ ] `execution_time_ms` present

### Error Handling
- [ ] Validation errors handled
- [ ] Graceful degradation when workflows unavailable

---

## 🐛 Troubleshooting

### Issue: Query Not Routing to Expected Tier

**Check**:
1. Look for routing decision in logs: `🎯 Auto-routing decision`
2. Check matched patterns in the log
3. Verify keyword is in the expected category

**Common Causes**:
- Keyword spelled differently
- Priority rule overriding
- Context escalation from previous query

### Issue: Agents Not Invoked

**Check**:
1. Verify workflows initialized: Check startup logs for `✅ Level 4a workflow compiled`
2. Check fallback behavior in logs: `⚠️ L4B workflow not available, using L4A`

### Issue: Cache Interfering with Testing

**Solution**:
```bash
# Clear all caches
curl -X POST http://localhost:8000/cache/clear
```

### Issue: LangSmith Not Showing Traces

**Check**:
1. Verify environment variables set
2. Restart API container
3. Check LANGCHAIN_TRACING_V2=true

---

## 📚 Related Documentation

- **Routing Module**: `backend/src/routing/`
- **Unit Tests**: `tests/test_routing.py` (28 tests)
- **API Main**: `backend/src/api/main.py`
- **Models**: `backend/src/models/weather.py`

---

**Last Updated**: December 11, 2025
**Version**: v0.6.0
**Test Coverage**: 28/28 unit tests passing
