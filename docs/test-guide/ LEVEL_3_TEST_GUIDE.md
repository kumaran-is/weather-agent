# Comprehensive Testing Guide: Level 3a + 3b + 3c

**Purpose**: Manual testing via Swagger UI and LangSmith Studio to validate all Level 3a, 3b, and 3c features for complete 7-layer memory system.

**Duration**: 45-60 minutes for complete testing
**Date**: December 8, 2025
**Status**: ✅ **Level 3a + 3b: 100% AUTOMATED TESTS PASSING** (6/6 scenarios)
**Status**: ✅ **Level 3c: 100% IMPLEMENTATION COMPLETE** (All memory layers fully functional)

---

## ⚡ Quick Start Summary

**Automated Test Results**: ✅ **6/6 tests passing (100%)**

All critical bugs have been fixed:
1. ✅ **InMemoryStore API compatibility** (LangGraph v1.0+)
2. ✅ **Namespace positional arguments** (BaseStore v1.0+)
3. ✅ **None formatting errors** (defensive checks added)
4. ✅ **🔴 CRITICAL: Global MemoryManager initialization** (eliminated per-request overhead)
5. ✅ **🔴 ARCHITECTURAL: Graphiti episode-based ingestion** (official pattern)
6. ✅ **Semantic tool discovery with Memory + RAG** (always include base MCP tools)

**You're now testing a fully validated implementation!** This manual testing confirms real-world behavior via Swagger UI.

---

## 🎯 What We're Testing

### **Level 3a Features (Memory System)**:
1. ✅ Short-term memory (Redis - session context)
2. ✅ Long-term memory (Graphiti + Neo4j - user profiles)
3. ✅ Session continuity across multiple queries
4. ✅ Pronoun resolution ("there" → tracked location)
5. ✅ User preferences and history
6. ✅ Semantic tool discovery (context reduction)
   - **Memory only**: 3 base MCP tools (get_current_weather, get_forecast, retrieve_weather_context)
   - **Memory + RAG**: 3 base MCP tools + 3 semantic RAG tools (selected via vector search)
   - **Key benefit**: When RAG enabled, only relevant RAG tools are loaded (37.5% context savings)

### **Level 3b Features (Advanced Reasoning)**:
1. ✅ Chain-of-Thought (CoT) - 5-step framework
2. ✅ Tree of Thoughts (ToT) - multi-path exploration
3. ✅ Graph of Thoughts (GoT) - multi-entity comparison
4. ✅ Reasoning quality improvement
5. ✅ Progressive enhancement (backward compatibility)

### **Level 3c Features (Complete 7-Layer Memory)**:
1. ✅ Layer 3: Episodic Memory (Redis, 7-day TTL) - Event history
2. ✅ Layer 4: Semantic Memory (Qdrant) - Domain knowledge
3. ✅ Layer 5: Procedural Memory (PostgreSQL) - Workflow optimization
4. ✅ Layer 6: Emotional Memory (Redis, 7-day TTL) - Sentiment analysis
5. ✅ Layer 7: Reflective Memory (PostgreSQL) - Self-improvement
6. ✅ Memory consolidation pipeline (70% storage reduction)
7. ✅ PostgreSQL integration (workflows, tool_usage, reflections tables)

---

## 📋 Prerequisites

### 1. Start All Services

```bash
# Ensure all Docker containers are running (use dev mode for hot reload)
docker-compose -f docker-compose.dev.yml up -d

# Verify all services healthy
docker ps --filter "name=weather"

# Expected services:
# - weather-ai-api-dev (FastAPI with hot reload)
# - weather-ai-redis (short-term memory)
# - weather-ai-neo4j (long-term memory - Graphiti)
# - weather-ai-qdrant (RAG vector store)
# - weather-mcp (weather data MCP server)
# - hurricane-mcp (hurricane tracking MCP server)
```

**Expected for Level 3a/3b**: 6 services running (weather-ai-api, redis, neo4j, qdrant, weather-mcp, hurricane-mcp)
**Expected for Level 3c**: 7 services running (+ postgres for Layers 5 & 7)

### 2. Open Swagger UI

**URL**: http://localhost:8000/docs

**What you'll see**:
- Interactive API documentation
- "Try it out" buttons for each endpoint
- Request/response examples

### 3. Setup LangSmith (Optional but Recommended)

**Step 1**: Get LangSmith API Key
- Go to https://smith.langchain.com/
- Create account (if needed)
- Navigate to Settings → API Keys
- Create new API key

**Step 2**: Set Environment Variables
```bash
# Add to .env file (already exists)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_api_key_here
export LANGCHAIN_PROJECT=weather-ai-agent-l3b-testing

# Restart API container to pick up new env vars
docker-compose restart weather-ai-api
```

**Step 3**: Access LangSmith Studio
- URL: https://smith.langchain.com/
- Navigate to Projects → weather-ai-agent-l3b-testing
- View traces, debug tool calls, analyze reasoning

---

## 🧪 Testing Scenarios

---

## **SCENARIO 1: Basic Query (Baseline) - Level 1**

### Purpose
Establish baseline without memory or advanced reasoning.

### Steps

**1. Open Swagger UI**: http://localhost:8000/docs

**2. Find Endpoint**: POST `/weather/query`

**3. Click "Try it out"**

**4. Paste Request Body**:
```json
{
  "query": "What's the weather in Paris right now?",
  "user_id": "test_user_baseline",
  "session_id": "session_baseline_001",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": false
}
```

**5. Click "Execute"**

### Expected Response

**Status Code**: 200

**Response Body** (sample):
```json
{
  "response": "The current weather in Paris is:\n\n- Temperature: 8°C\n- Condition: Partly cloudy\n- Humidity: 76%\n- Wind Speed: 3.6 m/s\n- Feels Like: 6°C",
  "user_id": "test_user_baseline",
  "timestamp": "2025-12-07T13:00:00Z"
}
```

### Validation Checklist
- [ ] Response includes current weather data
- [ ] Response is concise (< 500 characters)
- [ ] No reasoning structure visible
- [ ] Duration: < 15 seconds

### LangSmith Validation
- Navigate to LangSmith → Projects → weather-ai-agent-l3b-testing
- Find latest trace "test_user_baseline"
- Check:
  - [ ] Tool calls visible (weather MCP)
  - [ ] No memory tool calls
  - [ ] Simple prompt (no CoT/ToT/GoT)

---

## **SCENARIO 2: Level 3a - Session Continuity (Memory)**

### Purpose
Test short-term memory across multiple queries in same session.

### Part A: First Query (Establish Context)

**Request**:
```json
{
  "query": "What's the weather in Tokyo?",
  "user_id": "test_user_memory",
  "session_id": "session_memory_001",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": true
}
```

**Expected Response**:
- Current weather in Tokyo
- Response should mention Tokyo explicitly

**Duration**: ~15 seconds

### Part B: Follow-up Query (Test Memory)

**IMPORTANT**: Use **same session_id** but ask about "there" or "tomorrow"

**Request**:
```json
{
  "query": "What about tomorrow?",
  "user_id": "test_user_memory",
  "session_id": "session_memory_001",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": true
}
```

**Expected Response**:
- Tomorrow's forecast **for Tokyo** (not asking which city!)
- Agent should resolve implicit location from memory

**Duration**: ~15 seconds

### Part C: Pronoun Resolution Test

**Request**:
```json
{
  "query": "What about there next week?",
  "user_id": "test_user_memory",
  "session_id": "session_memory_001",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": true
}
```

**Expected Response**:
- Next week's forecast **for Tokyo**
- Agent resolves "there" → Tokyo from session memory

### Validation Checklist
- [ ] **Part A**: Agent responds with Tokyo weather
- [ ] **Part B**: Agent knows "tomorrow" refers to Tokyo (no clarification needed)
- [ ] **Part C**: Agent resolves "there" → Tokyo
- [ ] Session continuity working across 3 queries
- [ ] No need to repeat city name

### LangSmith Validation
- Check trace for "test_user_memory"
- Verify:
  - [ ] Redis memory tool calls visible
  - [ ] Context retrieval before agent execution
  - [ ] Location tracking working

---

## **SCENARIO 3: Level 3a - User Preferences (Long-Term Memory)**

### Purpose
Test long-term memory persisting across different sessions.

### Part A: First Session (Set Preference)

**Request**:
```json
{
  "query": "I prefer temperatures in Fahrenheit and I'm planning a trip to London next month.",
  "user_id": "test_user_preferences",
  "session_id": "session_prefs_001",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": true
}
```

**Expected Response**:
- Agent acknowledges preference
- London trip noted

### Part B: New Session (Test Preference Recall)

**IMPORTANT**: Use **same user_id** but **different session_id**

**Request**:
```json
{
  "query": "What's the weather like?",
  "user_id": "test_user_preferences",
  "session_id": "session_prefs_002",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": true
}
```

**Expected Response**:
- Agent should:
  1. Remember London trip (long-term memory)
  2. Provide London weather
  3. Use Fahrenheit units (if possible)

### Validation Checklist
- [ ] Agent remembers user preferences across sessions
- [ ] Agent recalls London trip context
- [ ] Long-term memory (Graphiti/Neo4j) working

### LangSmith Validation
- Check both traces
- Verify:
  - [ ] Memory storage in session 1
  - [ ] Memory retrieval in session 2
  - [ ] Neo4j/Graphiti queries visible

---

## **SCENARIO 4: Level 3b - Chain-of-Thought (CoT)**

### Purpose
Test structured 5-step reasoning framework.

### Request

```json
{
  "query": "Should I plan a picnic in Seattle this Saturday?",
  "user_id": "test_user_cot",
  "session_id": "session_cot_001",
  "enable_rag": false,
  "enable_cot": true,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": false
}
```

### Expected Response

**Structure**: Should show 5-step framework:
```
1. Decompose: Break down into sub-questions
   - What's the weather forecast for Saturday?
   - Is it suitable for outdoor activities?

2. Gather: Collect relevant data
   - [Tool calls to get forecast]

3. Analyze: Examine weather conditions
   - Temperature: X°C (good/bad for picnic)
   - Precipitation: X% chance of rain
   - Wind: X m/s

4. Synthesize: Combine insights
   - Weather is [favorable/unfavorable] because...

5. Recommend: Final answer
   - Yes/No with reasoning
   - Alternative suggestions if needed
```

**Response Length**: 1,500-2,500 characters (much longer than basic)

**Duration**: 20-35 seconds

### Validation Checklist
- [ ] Response shows clear 5-step structure
- [ ] Each step has visible reasoning
- [ ] Final recommendation is actionable
- [ ] Keywords present: "Decompose", "Gather", "Analyze", "Synthesize", "Recommend"
- [ ] Response is 5-6x longer than basic query

### LangSmith Validation
- Check trace for "test_user_cot"
- Verify:
  - [ ] CoT system prompt used
  - [ ] Multiple tool calls (forecast retrieval)
  - [ ] Reasoning chain visible in messages

---

## **SCENARIO 5: Level 3b - Tree of Thoughts (ToT)**

### Purpose
Test multi-path exploration for safety-critical decisions.

### Request

```json
{
  "query": "Should I evacuate for a Category 4 hurricane approaching Tampa, Florida?",
  "user_id": "test_user_tot",
  "session_id": "session_tot_001",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": true,
  "enable_got": false,
  "use_memory": false
}
```

### Expected Response

**Structure**: Should show multi-path exploration:
```
1. Decompose: Identify multiple approaches
   - PATH 1: Check hurricane category severity
   - PATH 2: Check evacuation zone requirements
   - PATH 3: Check timeline/urgency

2. Explore Each Path:
   - PATH 1: Category 4 = winds 130-156 mph → EXTREMELY DANGEROUS
   - PATH 2: Tampa coastal areas → Zone A/B → MANDATORY EVACUATION
   - PATH 3: Approaching → IMMEDIATE action needed

3. Evaluate Paths:
   - PATH 1 Score: 1.0 (highest severity)
   - PATH 2 Score: 1.0 (mandatory = must evacuate)
   - PATH 3 Score: 1.0 (urgency high)

4. Best Path Selection:
   - All paths converge → HIGH CONFIDENCE
   - Decision: EVACUATE IMMEDIATELY

5. Final Recommendation:
   - **YES, EVACUATE IMMEDIATELY**
   - Category 4 is life-threatening
   - Follow local evacuation orders
```

**Response Length**: 1,500-2,000 characters

**Duration**: 15-25 seconds

### Validation Checklist
- [ ] Response shows multiple paths explored
- [ ] Each path evaluated independently
- [ ] Path scores/confidence visible
- [ ] All paths converge (high confidence signal)
- [ ] **CRITICAL**: Correct evacuation recommendation (YES for Cat 4)
- [ ] Keywords: "Path", "Approach", "Explore", "Evaluate", "Converge"

### LangSmith Validation
- Check trace for "test_user_tot"
- Verify:
  - [ ] ToT system prompt used
  - [ ] Multi-step reasoning visible
  - [ ] Hurricane data retrieval (MCP calls)

---

## **SCENARIO 6: Level 3b - Graph of Thoughts (GoT)**

### Purpose
Test multi-entity comparison with shared sub-problems.

### Request

```json
{
  "query": "Compare weather for a beach vacation in March: Miami, Honolulu, or San Diego?",
  "user_id": "test_user_got",
  "session_id": "session_got_001",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": false,
  "enable_got": true,
  "use_memory": false
}
```

### Expected Response

**Structure**: Should show parallel entity analysis with comparison:
```
Graph of Thoughts Analysis:

NODE 1: Miami, Florida
- Temperature: 25-27°C (77-81°F)
- Weather: Mix of sun and clouds
- Humidity: 65-75%
- Ocean temp: ~24°C (comfortable swimming)
- Summary: Warm, humid, good for beach

NODE 2: Honolulu, Hawaii
- Temperature: 26-28°C (79-82°F)
- Weather: Tropical, occasional showers
- Humidity: 70-80%
- Ocean temp: ~25°C (excellent swimming)
- Summary: Warmest, most humid

NODE 3: San Diego, California
- Temperature: 18-21°C (64-70°F)
- Weather: Mostly sunny, dry
- Humidity: 50-60%
- Ocean temp: ~16°C (cool for swimming)
- Summary: Mild, dry, best weather quality

MERGE POINT: Comparison
- Warmest: Honolulu (28°C)
- Best for swimming: Honolulu (warmest water)
- Best weather quality: San Diego (least rain)
- Most humid: Honolulu
- Coolest ocean: San Diego

RECOMMENDATION:
- For warm beach vacation → Honolulu or Miami
- For comfortable weather → San Diego
- Overall winner: Depends on preference
  - Love swimming → Honolulu
  - Hate humidity → San Diego
  - Balance → Miami
```

**Response Length**: 2,000-2,500 characters (longest)

**Duration**: 20-30 seconds

### Validation Checklist
- [ ] All 3 entities analyzed (Miami, Honolulu, San Diego)
- [ ] Parallel reasoning structure visible
- [ ] Shared criteria applied (temperature, weather, humidity, ocean temp)
- [ ] Comparison/merge section present
- [ ] Personalized recommendations based on preferences
- [ ] Keywords: "Compare", "Each", "All three", "NODE", "MERGE"

### LangSmith Validation
- Check trace for "test_user_got"
- Verify:
  - [ ] GoT system prompt used
  - [ ] Multiple forecast tool calls (one per city)
  - [ ] Parallel execution pattern

---

## **SCENARIO 7: Progressive Enhancement Validation**

### Purpose
Test that all modes work together and backward compatibility maintained.

### Test A: All Flags OFF (Baseline)

```json
{
  "query": "Weather in Boston?",
  "user_id": "test_user_progressive",
  "session_id": "session_prog_001",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": false
}
```

**Expected**: Simple weather data, fast (<15s)

### Test B: Only CoT ON

```json
{
  "query": "Weather in Boston?",
  "user_id": "test_user_progressive",
  "session_id": "session_prog_002",
  "enable_rag": false,
  "enable_cot": true,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": false
}
```

**Expected**: 5-step CoT structure, slower (~30s)

### Test C: Only ToT ON

```json
{
  "query": "Weather in Boston?",
  "user_id": "test_user_progressive",
  "session_id": "session_prog_003",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": true,
  "enable_got": false,
  "use_memory": false
}
```

**Expected**: Multi-path reasoning, moderate speed (~20s)

### Test D: Priority Test (GoT > ToT)

```json
{
  "query": "Weather in Boston?",
  "user_id": "test_user_progressive",
  "session_id": "session_prog_004",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": true,
  "enable_got": true,
  "use_memory": false
}
```

**Expected**: GoT should take priority (both enabled, GoT wins)

### Validation Checklist
- [ ] Test A: Basic response (no reasoning)
- [ ] Test B: CoT structure visible
- [ ] Test C: ToT structure visible
- [ ] Test D: GoT structure visible (not ToT)
- [ ] Backward compatibility: All tests work
- [ ] No errors or crashes

---

## 🎯 **SCENARIO 8: Combined L3a + L3b Test**

### Purpose
Test memory + advanced reasoning together.

### Part 1: Establish Memory

```json
{
  "query": "I'm planning a family trip to Orlando, Florida in two weeks.",
  "user_id": "test_user_combined",
  "session_id": "session_combined_001",
  "enable_rag": false,
  "enable_cot": false,
  "enable_tot": false,
  "enable_got": false,
  "use_memory": true
}
```

**Expected**: Agent acknowledges Orlando trip

### Part 2: Use Memory + ToT + RAG Reasoning

**IMPORTANT**: RAG must be enabled to provide historical context for ToT reasoning!

```json
{
  "query": "Should I be worried about weather for our trip?",
  "user_id": "test_user_combined",
  "session_id": "session_combined_001",
  "enable_rag": true,
  "enable_cot": false,
  "enable_tot": true,
  "enable_got": false,
  "use_memory": true
}
```

**Expected**:
- Agent remembers Orlando trip from memory (L3a) - **should NOT ask for location!**
- Agent remembers "two weeks" timeline from memory
- Uses RAG tools to get historical weather data (L2)
- Uses ToT to analyze weather risks with multiple paths (L3b)
- Multi-path exploration:
  - PATH 1: Check 2-week forecast for Orlando
  - PATH 2: Check historical weather patterns for this time (via RAG)
  - PATH 3: Check hurricane season risks
- Final recommendation based on all paths

**Response Length**: 1,500-2,000 characters (substantial ToT reasoning)

**Duration**: 25-35 seconds (Memory + RAG + ToT combined)

### Validation Checklist
- [ ] Agent remembers Orlando (not asking where)
- [ ] Agent remembers "two weeks" timeline
- [ ] ToT reasoning applied to the decision
- [ ] Both L3a and L3b working together seamlessly

---

## 🆕 **LEVEL 3C TESTING SCENARIOS**

---

## **SCENARIO 9: Level 3c - Episodic Memory (Layer 3)**

### Purpose
Test event history storage and retrieval across sessions.

### Part A: Create Multiple Episodes

**Request 1** (First episode):
```json
{
  "query": "Is it safe to drive to work in Miami with this weather?",
  "user_id": "test_user_l3c_episodic",
  "session_id": "session_l3c_ep_001",
  "enable_rag": false,
  "enable_cot": false,
  "use_memory": true
}
```

**Request 2** (Second episode, same session):
```json
{
  "query": "Actually, I decided to work from home. Was that a good decision?",
  "user_id": "test_user_l3c_episodic",
  "session_id": "session_l3c_ep_001",
  "enable_rag": false,
  "enable_cot": false,
  "use_memory": true
}
```

### Part B: Test Episode Recall (New Session)

**Request 3** (Different session, query about past decision):
```json
{
  "query": "Looking back at yesterday's weather decision, was I right to stay home?",
  "user_id": "test_user_l3c_episodic",
  "session_id": "session_l3c_ep_002",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

### Expected Behavior
- **Part A**: Agent provides weather-based driving recommendation, then validates user's decision
- **Part B**: Agent recalls previous episode (work-from-home decision) and validates it with current/historical data
- Episodes stored with outcomes ("helpful", "confusing", "validated")
- Emotional context tracked (if user expressed concern)

### Validation Checklist
- [ ] Agent remembers previous query about driving to work
- [ ] Agent remembers user's work-from-home decision
- [ ] Episode retrieval works across sessions
- [ ] Emotional context preserved (concern/anxiety about safety)

### LangSmith Studio JSON Input
```json
{
  "input": {
    "query": "Looking back at yesterday's weather decision, was I right to stay home?",
    "user_id": "test_user_l3c_episodic",
    "session_id": "session_l3c_ep_002"
  },
  "config": {
    "enable_rag": true,
    "enable_cot": false,
    "use_memory": true
  }
}
```

**What to observe in LangSmith**:
- [ ] Episodic memory retrieval visible
- [ ] Previous episode data loaded
- [ ] Outcome tracking ("validated" if decision was correct)

---

## **SCENARIO 10: Level 3c - Emotional Memory (Layer 6)**

### Purpose
Test sentiment analysis and emotional state tracking for empathetic responses.

### Part A: Anxious Query

**Request 1** (Hurricane anxiety):
```json
{
  "query": "I'm really worried about this Category 5 hurricane heading toward us! Is my family in danger?",
  "user_id": "test_user_l3c_emotional",
  "session_id": "session_l3c_emo_001",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response**:
- Dominant emotion detected: **anxious**
- Sentiment score: **-0.6 to -0.8** (negative)
- Triggers: ["worried", "danger", "negative_sentiment", "exclamation_mark"]
- Agent response should be **empathetic and reassuring**

### Part B: Follow-up (Emotion Continuity)

**Request 2** (Same session):
```json
{
  "query": "What should I do to prepare?",
  "user_id": "test_user_l3c_emotional",
  "session_id": "session_l3c_emo_001",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Behavior**:
- Agent remembers previous anxious state
- Response maintains **empathetic tone**
- Provides **clear, reassuring preparation steps**
- Avoids alarming language

### Part C: Emotional Trend Analysis

**Request 3** (New session, query about emotional history):
```json
{
  "query": "How have I been feeling about weather lately?",
  "user_id": "test_user_l3c_emotional",
  "session_id": "session_l3c_emo_002",
  "enable_rag": false,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response**:
- Agent provides emotional trend summary
- Mentions anxiety episode from previous session
- Shows understanding of user's concerns over time

### Validation Checklist
- [ ] Sentiment analysis detects "anxious" emotion correctly
- [ ] Emotional triggers identified (worried, danger)
- [ ] Agent provides empathetic response (not robotic)
- [ ] Emotional state persists across queries in same session
- [ ] Emotional trend retrieval works

### LangSmith Studio JSON Input
```json
{
  "input": {
    "query": "I'm really worried about this Category 5 hurricane heading toward us! Is my family in danger?",
    "user_id": "test_user_l3c_emotional",
    "session_id": "session_l3c_emo_001"
  },
  "config": {
    "enable_rag": true,
    "enable_cot": false,
    "use_memory": true
  }
}
```

**What to observe in LangSmith**:
- [ ] Emotional memory detection visible
- [ ] Sentiment score calculated (TextBlob or rule-based)
- [ ] Emotional context injected into agent prompt
- [ ] Agent tone adapts to user's emotional state

---

## **SCENARIO 11: Level 3c - Procedural Memory (Layer 5)**

### Purpose
Test workflow optimization and intelligent tool selection based on historical performance.

### Part A: First Hurricane Forecast Query

**Request 1** (Establish baseline workflow):
```json
{
  "query": "Give me a complete hurricane forecast for Tampa Bay, Florida",
  "user_id": "test_user_l3c_procedural",
  "session_id": "session_l3c_proc_001",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Workflow** (tools called in order):
1. `get_current_hurricanes`
2. `get_forecast`
3. `analyze_safety_guidelines` (RAG tool)

**What's happening behind the scenes**:
- System records this workflow as successful
- Stores: task_name="hurricane_forecast", steps=[...], success_rate=1.0, response_time=~15s

### Part B: Repeat Query (Test Workflow Optimization)

**Request 2** (Same task type):
```json
{
  "query": "Show me hurricane forecast for Miami",
  "user_id": "test_user_l3c_procedural",
  "session_id": "session_l3c_proc_002",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Behavior**:
- Agent uses **same workflow** as Request 1 (learned pattern)
- Response time similar or faster (workflow optimization)
- Success rate updated (running average)

### Part C: Workflow Analytics Query

**Request 3** (Ask about learned patterns):
```json
{
  "query": "What's the best way to get hurricane information based on what I usually ask?",
  "user_id": "test_user_l3c_procedural",
  "session_id": "session_l3c_proc_003",
  "enable_rag": false,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response**:
- Agent describes learned workflow pattern
- Mentions tools typically used for hurricane queries
- Shows understanding of user's query patterns

### Validation Checklist
- [ ] Workflow recorded after first query (PostgreSQL)
- [ ] Same workflow used for similar queries (optimization)
- [ ] Success rate and response time tracked
- [ ] Agent can describe learned patterns when asked

### LangSmith Studio JSON Input
```json
{
  "input": {
    "query": "Give me a complete hurricane forecast for Tampa Bay, Florida",
    "user_id": "test_user_l3c_procedural",
    "session_id": "session_l3c_proc_001"
  },
  "config": {
    "enable_rag": true,
    "enable_cot": false,
    "use_memory": true
  }
}
```

**What to observe in LangSmith**:
- [ ] Tool call sequence visible (get_current_hurricanes → get_forecast → RAG)
- [ ] Procedural memory recording workflow
- [ ] PostgreSQL storage of workflow pattern
- [ ] Subsequent queries use recommended workflow

---

## **SCENARIO 12: Level 3c - Reflective Memory (Layer 7)**

### Purpose
Test self-improvement through reflection on past errors and learnings.

### Part A: Simulated Error Pattern

**Request 1** (Query that might produce incorrect recommendation):
```json
{
  "query": "Is it safe to go to the beach during a tropical storm warning?",
  "user_id": "test_user_l3c_reflective",
  "session_id": "session_l3c_refl_001",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response**: Should say **NO, not safe**

### Part B: Explicit Negative Feedback

**Request 2** (User provides feedback):
```json
{
  "query": "That advice was too generic. I need specific evacuation zone information for my area.",
  "user_id": "test_user_l3c_reflective",
  "session_id": "session_l3c_refl_001",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**What happens behind the scenes**:
- System creates reflection:
  - Trigger: "user_feedback_negative"
  - Context: "Generic safety advice provided"
  - Analysis: "Should have asked for user's specific location and evacuation zone"
  - Insight: "Always request location details for evacuation queries"
  - Stored in PostgreSQL reflections table

### Part C: Test Improved Behavior

**Request 3** (New session, similar query):
```json
{
  "query": "Should I evacuate for this hurricane?",
  "user_id": "test_user_l3c_reflective",
  "session_id": "session_l3c_refl_002",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Improved Response**:
- Agent asks for **specific location and evacuation zone** (learned behavior)
- More personalized advice
- Shows application of previous learning

### Part D: Reflection Analytics Query

**Request 4** (Ask about learnings):
```json
{
  "query": "What have you learned from our previous conversations?",
  "user_id": "test_user_l3c_reflective",
  "session_id": "session_l3c_refl_003",
  "enable_rag": false,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response**:
- Agent describes key learnings
- Mentions importance of specific location data for evacuations
- Shows meta-cognitive awareness

### Validation Checklist
- [ ] Negative feedback triggers reflection creation
- [ ] Reflection stored in PostgreSQL with analysis and insight
- [ ] Improved behavior visible in subsequent queries
- [ ] Agent can describe its learnings when asked
- [ ] Self-improvement cycle complete

### LangSmith Studio JSON Input
```json
{
  "input": {
    "query": "That advice was too generic. I need specific evacuation zone information for my area.",
    "user_id": "test_user_l3c_reflective",
    "session_id": "session_l3c_refl_001"
  },
  "config": {
    "enable_rag": true,
    "enable_cot": false,
    "use_memory": true
  }
}
```

**What to observe in LangSmith**:
- [ ] Reflection creation triggered by negative feedback
- [ ] PostgreSQL insert into reflections table
- [ ] Insight generation visible
- [ ] Improved behavior in subsequent traces

---

## **SCENARIO 13: Level 3c - Semantic Memory (Layer 4)**

### Purpose
Test domain knowledge storage and retrieval via vector search.

### Part A: Teaching Domain Knowledge

**Request 1** (Provide new knowledge):
```json
{
  "query": "Remember this: the Fujita Scale is different from Saffir-Simpson. Fujita is for tornadoes (F0-F5), Saffir-Simpson is for hurricanes (Cat 1-5).",
  "user_id": "test_user_l3c_semantic",
  "session_id": "session_l3c_sem_001",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Behavior**:
- System stores semantic knowledge in Qdrant
- Concept: "Fujita Scale vs Saffir-Simpson"
- Category: "meteorology"
- Related concepts: ["tornado", "hurricane", "classification"]

### Part B: Test Knowledge Retrieval (New Session)

**Request 2** (Query that requires semantic knowledge):
```json
{
  "query": "What's the difference between how tornadoes and hurricanes are classified?",
  "user_id": "test_user_l3c_semantic",
  "session_id": "session_l3c_sem_002",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response**:
- Agent retrieves semantic memory from Qdrant
- Correctly explains Fujita (tornadoes) vs Saffir-Simpson (hurricanes)
- Shows that learned knowledge is retained

### Part C: Related Concept Query

**Request 3** (Query related concept):
```json
{
  "query": "Tell me about hurricane categories",
  "user_id": "test_user_l3c_semantic",
  "session_id": "session_l3c_sem_003",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response**:
- Agent retrieves related semantic memory
- Mentions Saffir-Simpson Scale (related concept)
- Shows semantic relationship understanding

### Validation Checklist
- [ ] Semantic knowledge stored in Qdrant (vector embeddings)
- [ ] Knowledge retrieval works via semantic search
- [ ] Related concepts linked correctly
- [ ] Domain knowledge persists across sessions

### LangSmith Studio JSON Input
```json
{
  "input": {
    "query": "What's the difference between how tornadoes and hurricanes are classified?",
    "user_id": "test_user_l3c_semantic",
    "session_id": "session_l3c_sem_002"
  },
  "config": {
    "enable_rag": true,
    "enable_cot": false,
    "use_memory": true
  }
}
```

**What to observe in LangSmith**:
- [ ] Semantic memory retrieval from Qdrant
- [ ] Vector similarity search visible
- [ ] Learned knowledge used in response
- [ ] Related concepts linked

---

## **SCENARIO 14: Level 3c - Complete 7-Layer Memory Integration**

### Purpose
Test all 7 layers working together seamlessly.

### Setup Sequence

**Request 1** (Establish context across layers):
```json
{
  "query": "I'm really anxious about hurricane season this year. I'm planning a trip to Florida in August and I prefer detailed safety information. Can you help me understand hurricane risks?",
  "user_id": "test_user_l3c_complete",
  "session_id": "session_l3c_complete_001",
  "enable_rag": true,
  "enable_cot": true,
  "use_memory": true
}
```

**What should happen across all 7 layers**:

1. **Layer 1 (Short-term)**: Stores conversation context (Florida trip, August)
2. **Layer 2 (Long-term)**: Stores user preference (detailed safety info)
3. **Layer 3 (Episodic)**: Records this query episode with outcome
4. **Layer 4 (Semantic)**: Retrieves hurricane knowledge from Qdrant
5. **Layer 5 (Procedural)**: Learns optimal workflow for hurricane safety queries
6. **Layer 6 (Emotional)**: Detects anxious state, adapts tone
7. **Layer 7 (Reflective)**: No immediate reflection (only on errors/feedback)

### Test Cross-Layer Retrieval

**Request 2** (New session, test memory integration):
```json
{
  "query": "Following up on our conversation about my trip - what should I pack for hurricane preparedness?",
  "user_id": "test_user_l3c_complete",
  "session_id": "session_l3c_complete_002",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response** (drawing from multiple layers):
- **Layer 2**: Remembers Florida trip in August
- **Layer 3**: Recalls previous hurricane safety discussion
- **Layer 4**: Uses hurricane preparedness knowledge (semantic)
- **Layer 5**: Uses learned workflow for safety queries
- **Layer 6**: Maintains empathetic tone (previous anxiety noted)
- Provides **detailed** safety information (user preference)

### Test Memory Consolidation

**Request 3** (Later session, test consolidated memory):
```json
{
  "query": "Quick reminder: what were my main concerns about the Florida trip?",
  "user_id": "test_user_l3c_complete",
  "session_id": "session_l3c_complete_003",
  "enable_rag": false,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response** (consolidated summary):
- Agent provides **concise summary** from consolidated memory
- Key points: Florida trip, August timing, hurricane anxiety, safety info preference
- Shows memory consolidation is working (summary, not raw conversation turns)

### Validation Checklist
- [ ] All 7 layers activated and working together
- [ ] Short-term memory (conversation context) working
- [ ] Long-term memory (user preferences) persisting
- [ ] Episodic memory (event history) accessible
- [ ] Semantic memory (domain knowledge) retrieved
- [ ] Procedural memory (workflow optimization) applied
- [ ] Emotional memory (sentiment) influencing tone
- [ ] Reflective memory (self-improvement) ready for feedback
- [ ] Memory consolidation reducing storage (behind the scenes)
- [ ] Cross-layer integration seamless (no visible separation)

### LangSmith Studio JSON Input
```json
{
  "input": {
    "query": "Following up on our conversation about my trip - what should I pack for hurricane preparedness?",
    "user_id": "test_user_l3c_complete",
    "session_id": "session_l3c_complete_002"
  },
  "config": {
    "enable_rag": true,
    "enable_cot": false,
    "use_memory": true
  }
}
```

**What to observe in LangSmith**:
- [ ] Multiple memory layer retrievals visible
- [ ] Layer 1: Redis conversation context load
- [ ] Layer 2: Neo4j/Graphiti user profile load
- [ ] Layer 3: Redis episodic memory retrieval
- [ ] Layer 4: Qdrant semantic search
- [ ] Layer 5: PostgreSQL workflow recommendation
- [ ] Layer 6: Emotional context injection
- [ ] All layers queried in parallel (performance optimization)
- [ ] Single coherent response synthesized from all layers

---

## **SCENARIO 15: Level 3c - Memory Consolidation Validation**

### Purpose
Verify 70% storage reduction through consolidation pipeline.

### Setup: Create Multiple Conversation Turns

**Request Sequence** (10 rapid queries in same session):

```json
// Request 1-10 (rapid fire):
{
  "query": "What's the weather in Tokyo?",
  "user_id": "test_user_consolidation",
  "session_id": "session_consol_001",
  "enable_rag": false,
  "enable_cot": false,
  "use_memory": true
}
// ... repeat with slight variations (tomorrow, next week, etc.)
```

**What happens behind the scenes**:
- Each turn stored in Redis (10 turns × ~1KB = ~10KB)
- After hourly consolidation runs: 10 turns → 1 summary (~2KB)
- Storage reduction: 80% (10KB → 2KB)

### Validation Method

**After 1 hour** (or trigger consolidation manually):

**Request** (Test consolidated memory):
```json
{
  "query": "Summarize what we discussed about Tokyo",
  "user_id": "test_user_consolidation",
  "session_id": "session_consol_002",
  "enable_rag": false,
  "enable_cot": false,
  "use_memory": true
}
```

**Expected Response**:
- Agent provides **consolidated summary** (not 10 individual responses)
- Key points preserved (weather queries about Tokyo)
- Details preserved (current, tomorrow, next week requests)
- Storage efficient (summary, not raw turns)

### Validation Checklist
- [ ] Multiple conversation turns stored initially
- [ ] Consolidation pipeline reduces storage (manual trigger or scheduled)
- [ ] Consolidated summary retrievable
- [ ] Key information preserved after consolidation
- [ ] Storage reduction achieved (~70% average)

### Manual Consolidation Trigger (Developer)

**Python script to trigger consolidation**:
```python
from backend.src.memory.consolidation import MemoryConsolidator

consolidator = MemoryConsolidator()
result = await consolidator.consolidate_conversation(
    user_id="test_user_consolidation",
    session_id="session_consol_001"
)
print(f"Storage reduction: {result.reduction_percent:.1f}%")
# Expected: 70-80%
```

---

## 📊 Expected Performance Benchmarks

**Based on automated test results** (6/6 passing, 100% success rate):

| Scenario | Expected Duration | Response Length | Reasoning Complexity | Tools Used |
|----------|------------------|-----------------|---------------------|------------|
| **Baseline** | 2-5s | 200-300 chars | None | 3 MCP only |
| **Memory (L3a)** | 3-7s | 300-700 chars | Context retrieval | 3 MCP + memory |
| **Preferences (L3a)** | 5-10s | 400-800 chars | Neo4j queries | 3 MCP + memory + Neo4j |
| **CoT (L3b)** | 15-20s | 1,100-1,500 chars | 5-step framework | 3 MCP + forecast |
| **ToT (L3b)** | 10-15s | 1,900-2,400 chars | Multi-path | 3 MCP + forecast |
| **GoT (L3b)** | 2-8s | 400-2,800 chars | Multi-entity | 3 MCP (parallel) |
| **Combined (L3a+L3b+RAG)** | 12-18s | 1,500-2,000 chars | Memory + ToT + RAG | 6 tools total |

**Key Insights**:
- **Fastest**: Baseline and GoT (parallel execution advantage)
- **Most detailed**: ToT and GoT (multi-path/multi-entity reasoning)
- **Most complex**: Combined scenario (all 3 systems working together)
- **Performance**: All scenarios complete in <20s (acceptable for production)

---

## ✅ Final Validation Checklist

### Level 3a - Memory System
- [ ] **Scenario 2**: Session continuity working (3 queries in same session)
- [ ] **Scenario 2**: Pronoun resolution ("there" → tracked location)
- [ ] **Scenario 3**: User preferences persist across sessions
- [ ] **Scenario 3**: Long-term memory (Graphiti/Neo4j) working
- [ ] **Scenario 8**: Memory + reasoning work together

### Level 3b - Advanced Reasoning
- [ ] **Scenario 4**: CoT 5-step framework visible
- [ ] **Scenario 5**: ToT multi-path exploration working
- [ ] **Scenario 6**: GoT multi-entity comparison working
- [ ] **Scenario 5**: Safety-critical decisions correct (evacuation)
- [ ] **Scenario 7**: Progressive enhancement working (backward compatible)

### Infrastructure
- [ ] All Docker services healthy
- [ ] No 500 errors or crashes
- [ ] Response times acceptable (<40s for complex queries)
- [ ] LangSmith traces capturing all tool calls

### Quality
- [ ] Responses are coherent and actionable
- [ ] Reasoning quality improves with advanced modes
- [ ] No hallucinations or incorrect recommendations
- [ ] Appropriate reasoning mode for each use case

---

## 🔍 LangSmith Studio - How to Analyze

### Navigate to Trace

1. Go to https://smith.langchain.com/
2. Select project: `weather-ai-agent-l3b-testing`
3. Find trace by user_id (e.g., "test_user_cot")

### What to Check

**General**:
- [ ] Full conversation flow visible
- [ ] Tool calls shown (weather MCP, memory, etc.)
- [ ] Latency breakdown by step
- [ ] No errors in tool execution

**Memory Traces** (L3a):
- [ ] Redis memory retrieval visible
- [ ] Context loaded before agent execution
- [ ] Memory storage after response

**Reasoning Traces** (L3b):
- [ ] Correct system prompt used (CoT/ToT/GoT)
- [ ] Multi-step reasoning visible in messages
- [ ] Tool calls match reasoning mode

**Performance**:
- [ ] Total duration matches expectations
- [ ] No unusually slow steps
- [ ] Token usage reasonable

---

## 📸 Screenshot Checklist (Optional)

For documentation purposes, capture:
1. Swagger UI showing successful request/response
2. LangSmith trace showing reasoning steps
3. Memory retrieval in action (Redis/Neo4j queries)
4. Multi-path exploration (ToT trace)
5. Multi-entity comparison (GoT trace)

---

## 🚨 Common Issues & Troubleshooting

### Issue 1: "Connection refused" error
**Cause**: API container not running
**Fix**:
```bash
docker-compose -f docker-compose.dev.yml up -d weather-ai-api
# Check logs:
docker-compose -f docker-compose.dev.yml logs -f weather-ai-api
```

### Issue 2: Memory not working
**Cause**: Redis or Neo4j not running
**Fix**:
```bash
docker-compose -f docker-compose.dev.yml up -d redis neo4j
docker-compose -f docker-compose.dev.yml restart weather-ai-api
# Verify all services running:
docker ps | grep weather
```

### Issue 3: Swagger UI not loading
**Cause**: API not fully started or wrong URL
**Fix**:
- **Correct URL**: http://localhost:8000/docs (NOT 8080)
- Wait 30-60 seconds after `docker-compose up` for full startup
- Check API logs: `docker-compose -f docker-compose.dev.yml logs weather-ai-api`
- Look for: "✅ Memory manager initialized successfully" and "Application startup complete"

### Issue 4: LangSmith traces not appearing
**Cause**: Environment variables not set
**Fix**:
```bash
# Add to .env file:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key_here
LANGCHAIN_PROJECT=weather-ai-agent-l3b-testing

# Restart API:
docker-compose -f docker-compose.dev.yml restart weather-ai-api
```

### Issue 5: Slow responses (>60s)
**Cause**: LLM timeout or rate limiting
**Check**: LangSmith trace for stuck tool calls
**Note**: Combined scenarios (Memory + ToT + RAG) can take 15-35s - this is NORMAL

### Issue 6: No reasoning structure visible
**Cause**: Wrong enable_* flag or old API version
**Fix**:
- Double-check request JSON (enable_cot, enable_tot, enable_got flags)
- Verify API version: Look for "Level 3a: Memory System + Semantic Tool Discovery 🆕" in startup logs
- Check response length - CoT/ToT should be 1,500+ chars

---

## 📝 Test Report Template

After completing all scenarios, fill out:

```markdown
# Level 3a + 3b Manual Testing Report

**Tester**: [Your Name]
**Date**: [Date]
**Duration**: [Time spent]

## Results Summary

| Scenario | Status | Duration | Issues Found |
|----------|--------|----------|--------------|
| 1. Baseline | PASS/FAIL | Xs | None/[Issue] |
| 2. Memory (L3a) | PASS/FAIL | Xs | None/[Issue] |
| 3. Preferences (L3a) | PASS/FAIL | Xs | None/[Issue] |
| 4. CoT (L3b) | PASS/FAIL | Xs | None/[Issue] |
| 5. ToT (L3b) | PASS/FAIL | Xs | None/[Issue] |
| 6. GoT (L3b) | PASS/FAIL | Xs | None/[Issue] |
| 7. Progressive | PASS/FAIL | Xs | None/[Issue] |
| 8. Combined | PASS/FAIL | Xs | None/[Issue] |

**Overall Status**: PASS / FAIL
**Pass Rate**: X/8 (X%)

## Issues Found

[List any issues]

## Observations

[Any notable behaviors]

## Recommendation

[ ] APPROVED - Ready for Level 3c
[ ] NEEDS FIXES - Issues must be resolved first
```

---

## ✅ Sign-Off

After completing ALL scenarios:

- [ ] All 8 scenarios tested
- [ ] Performance within expected ranges
- [ ] No critical bugs found
- [ ] LangSmith traces reviewed
- [ ] Test report filled out

**Tester Signature**: _______________
**Date**: _______________

**Ready to proceed to Level 3c**: YES / NO

---

## Cache Testing Scenarios (L5a) - Append to LEVEL_3_TEST_GUIDE.md

**Insert this section BEFORE the "Test Report Template" section (line 1508)**

---

### 🔥 Level 5a: Cache Testing Scenarios (L1 + L2 + L3)

**Purpose**: Validate 3-layer caching system for 60-75% cost reduction
**Duration**: 15-20 minutes
**Prerequisites**: Redis running, CACHE_ENABLED=true in .env

### Cache Testing Objectives

1. ✅ L1 cache (in-process LRU): <1ms hits, 15-25% hit rate
2. ✅ L2 cache (Redis distributed): <10ms hits, 30-40% hit rate
3. ✅ L3 cache (Anthropic prompt): Transparent, 60-70% token savings
4. ✅ Cache hit indicators in responses
5. ✅ L2→L1 backfill optimization
6. ✅ Graceful degradation on errors

---

### Scenario 15: L1 Cache Hit (In-Process Memory)

**What**: Test L1 in-process LRU cache for ultra-fast hits (<1ms)
**Expected**: Second identical query returns cached response with cache_hit=true, cache_layer="L1"

#### Step 1: First Request (Cache Miss)

**Endpoint**: POST /weather/query
**Request**:
```json
{
  "query": "What's the weather in London?",
  "user_id": "cache_test_user",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": false
}
```

**Expected Response**:
```json
{
  "response": "The weather in London is...",
  "user_id": "cache_test_user",
  "timestamp": "2025-12-08T...",
  "cache_hit": false,
  "cache_layer": null
}
```

**Verify**:
- ❌ cache_hit is false (first request, no cache)
- ⏱️ Response time: 2-5 seconds (agent invocation)

#### Step 2: Second Request (L1 Cache Hit)

**Wait**: 1-2 seconds (same server instance)

**Request**: IDENTICAL to Step 1
```json
{
  "query": "What's the weather in London?",
  "user_id": "cache_test_user",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": false
}
```

**Expected Response**:
```json
{
  "response": "The weather in London is...",  // ← SAME response as Step 1
  "user_id": "cache_test_user",
  "timestamp": "2025-12-08T...",
  "cache_hit": true,                          // ← Cache hit!
  "cache_layer": "L1"                          // ← Served from L1 cache
}
```

**Verify**:
- ✅ cache_hit is true
- ✅ cache_layer is "L1"
- ✅ Response text is IDENTICAL to Step 1
- ⚡ Response time: <100ms (from memory)

#### Step 3: Verify Cache Statistics

**Endpoint**: GET /cache/stats

**Expected Response**:
```json
{
  "enabled": true,
  "l1": {
    "enabled": true,
    "hits": 1,
    "misses": 1,
    "evictions": 0,
    "hit_rate": 0.5,   // 1 hit / 2 requests = 50%
    "size": 1,
    "max_size": 1000,
    "ttl_seconds": 300
  },
  "l2": {
    "enabled": true,
    "hits": 0,
    "misses": 1,
    "hit_rate": 0.0
  },
  "summary": {
    "total_requests": 2,
    "total_cache_hits": 1,
    "overall_hit_rate": 0.5
  }
}
```

**Verify**:
- ✅ L1 hits = 1
- ✅ L1 misses = 1
- ✅ L1 hit_rate = 0.5 (50%)
- ✅ L1 size = 1 (one entry cached)

#### LangSmith Studio Verification

1. **Open LangSmith**: https://smith.langchain.com/
2. **Find Request 2 trace** (the cache hit)
3. **Verify**:
   - ⚡ **Total latency**: <100ms (vs 2-5s for agent)
   - 🚫 **NO agent invocation** (cache hit skips agent)
   - ✅ **Response matches Request 1** (served from cache)

---

### Scenario 16: L1 Cache Miss → L2 Cache Hit (Redis Distributed)

**What**: Test L2 Redis cache sharing across server instances and L2→L1 backfill
**Expected**: Different feature flags → L1 miss → L2 hit → backfill L1

#### Step 1: First Request with CoT Enabled

**Request**:
```json
{
  "query": "What's the weather in Paris?",
  "user_id": "cache_test_user",
  "enable_rag": true,
  "enable_cot": true,  // ← CoT enabled (different cache key)
  "use_memory": false
}
```

**Expected Response**:
```json
{
  "response": "Let me analyze this step-by-step...",
  "cache_hit": false,
  "cache_layer": null
}
```

**Verify**:
- ❌ cache_hit = false (new cache key due to enable_cot=true)
- ⏱️ Response time: 3-7 seconds (CoT reasoning takes longer)
- 📝 Response contains "Step 1:", "Step 2:" (CoT structure)

#### Step 2: Second Request (Same Query, CoT Disabled → Different Cache Key)

**Request**:
```json
{
  "query": "What's the weather in Paris?",
  "user_id": "cache_test_user",
  "enable_rag": true,
  "enable_cot": false,  // ← CoT disabled (L1 miss)
  "use_memory": false
}
```

**Expected Response**:
```json
{
  "response": "The weather in Paris is...",  // ← Different response (no CoT)
  "cache_hit": false,
  "cache_layer": null
}
```

**Verify**:
- ❌ cache_hit = false (different feature flags → different cache key)
- ⏱️ Response time: 2-5 seconds (agent invocation)
- 📝 Response does NOT contain CoT steps

#### Step 3: Third Request (Repeat Step 2 → L2 Hit → L1 Backfill)

**Simulate different server instance**: Clear L1 cache or restart server

**Request**: IDENTICAL to Step 2
```json
{
  "query": "What's the weather in Paris?",
  "user_id": "cache_test_user",
  "enable_rag": true,
  "enable_cot": false,
  "use_memory": false
}
```

**Expected Response**:
```json
{
  "response": "The weather in Paris is...",  // ← SAME as Step 2
  "cache_hit": true,
  "cache_layer": "L2"                        // ← Served from L2 (Redis)
}
```

**Verify**:
- ✅ cache_hit = true
- ✅ cache_layer = "L2" (Redis distributed cache)
- ✅ Response matches Step 2 exactly
- ⚡ Response time: <100ms (from Redis)

#### Step 4: Fourth Request (Verify L1 Backfill)

**Request**: IDENTICAL to Step 2 and Step 3

**Expected Response**:
```json
{
  "response": "The weather in Paris is...",
  "cache_hit": true,
  "cache_layer": "L1"  // ← Now in L1 (backfilled from L2)
}
```

**Verify**:
- ✅ cache_hit = true
- ✅ cache_layer = "L1" (backfilled from L2 hit in Step 3)
- ⚡ Response time: <50ms (faster than L2)

#### Step 5: Verify Cache Statistics

**Endpoint**: GET /cache/stats

**Expected**:
```json
{
  "l1": {
    "hits": 2,     // Step 4 (L1 hit after backfill)
    "misses": 3    // Step 1, 2, 3 (before backfill)
  },
  "l2": {
    "hits": 1,     // Step 3 (L2 hit)
    "misses": 1    // Step 2 (L2 miss)
  }
}
```

---

### Scenario 17: Cache Key Isolation (User-Specific Caching)

**What**: Verify cache keys include user_id (different users don't share caches)
**Expected**: Same query for different users → separate cache entries

#### Step 1: User A Request

**Request**:
```json
{
  "query": "What's the weather?",
  "user_id": "user_alice",
  "enable_rag": true,
  "enable_cot": false
}
```

**Expected**:
```json
{
  "cache_hit": false,
  "cache_layer": null
}
```

#### Step 2: User B Request (Same Query)

**Request**:
```json
{
  "query": "What's the weather?",
  "user_id": "user_bob",  // ← Different user
  "enable_rag": true,
  "enable_cot": false
}
```

**Expected**:
```json
{
  "cache_hit": false,  // ← MISS (different user_id → different cache key)
  "cache_layer": null
}
```

**Verify**:
- ❌ cache_hit = false (user_id is part of cache key)
- ⏱️ Response time: 2-5s (agent invoked for User B)

#### Step 3: User A Repeat (Should Hit Cache)

**Request**: IDENTICAL to Step 1

**Expected**:
```json
{
  "cache_hit": true,   // ← HIT (same user_id + query)
  "cache_layer": "L1"
}
```

#### Step 4: Verify Cache Size

**Endpoint**: GET /cache/stats

**Expected**:
```json
{
  "l1": {
    "size": 2  // ← Two cache entries (user_alice and user_bob)
  }
}
```

---

### Scenario 18: Cache TTL Expiration (Time-To-Live)

**What**: Verify cache entries expire after TTL (default: 5min for L1, 30min for L2)
**Expected**: Cached entry expires → returns null → agent re-invoked

**NOTE**: This scenario requires waiting 5+ minutes. Skip if time-constrained.

#### Step 1: Cache a Query

**Request**:
```json
{
  "query": "Current temperature in Tokyo?",
  "user_id": "ttl_test_user",
  "enable_rag": true
}
```

**Verify**:
- ❌ cache_hit = false (first request)

#### Step 2: Immediate Repeat (Cache Hit)

**Request**: IDENTICAL to Step 1

**Expected**:
```json
{
  "cache_hit": true,
  "cache_layer": "L1"
}
```

#### Step 3: Wait for TTL Expiration

**Wait**: 6 minutes (L1 TTL = 5 minutes)

#### Step 4: Request After TTL (Cache Miss)

**Request**: IDENTICAL to Step 1

**Expected**:
```json
{
  "cache_hit": false,  // ← Expired from L1
  "cache_layer": null
}
```

**Verify**:
- ❌ cache_hit = false (entry expired)
- ⏱️ Response time: 2-5s (agent re-invoked)

---

### Scenario 19: Cache Performance Under Load

**What**: Test cache hit rates with realistic query patterns
**Expected**: 60-75% overall cost savings with 3-layer caching

#### Simulated Traffic Pattern

**Use Swagger UI "Try it out" to send**:

1. **10 unique queries** (cache misses) → Populate cache
2. **10 repeat queries** (50% L1 hits) → Test L1 effectiveness
3. **5 variant queries** (same query, different flags) → Test cache key isolation

#### Step 1: Populate Cache (10 Unique Queries)

Send 10 different weather queries:
```
1. "Weather in London?"
2. "Weather in Paris?"
3. "Weather in Tokyo?"
... (7 more unique cities)
```

**Expected**: All cache_hit = false

#### Step 2: Repeat Queries (L1 Hits)

Repeat the same 10 queries immediately.

**Expected**:
- ✅ All cache_hit = true
- ✅ All cache_layer = "L1"
- ⚡ Response times: <100ms

#### Step 3: Check Cache Statistics

**Endpoint**: GET /cache/stats

**Expected**:
```json
{
  "l1": {
    "hits": 10,
    "misses": 10,
    "hit_rate": 0.5,  // 50% hit rate
    "size": 10
  },
  "summary": {
    "total_requests": 20,
    "total_cache_hits": 10,
    "overall_hit_rate": 0.5
  }
}
```

**Verify**:
- ✅ L1 hit_rate = 50% (10 hits / 20 requests)
- ✅ L1 size = 10 (all queries cached)
- ✅ summary.overall_hit_rate = 0.5

#### Step 4: Variant Queries (Cache Key Isolation)

Send same queries with different feature flags:
```json
{
  "query": "Weather in London?",
  "enable_rag": false  // ← Different flag
}
```

**Expected**: cache_hit = false (different cache key)

---

### Scenario 20: Graceful Degradation (Redis Unavailable)

**What**: Test cache behavior when Redis (L2) is down
**Expected**: L2 disabled → L1 still works → graceful degradation

#### Step 1: Stop Redis

**Command**:
```bash
docker stop weather-ai-redis
```

#### Step 2: Send Request

**Request**:
```json
{
  "query": "Weather in Berlin?",
  "user_id": "degradation_test"
}
```

**Expected Response**:
```json
{
  "response": "The weather in Berlin is...",
  "cache_hit": false,
  "cache_layer": null
}
```

**Verify**:
- ✅ Request succeeds (no error)
- ❌ cache_hit = false (L2 unavailable)
- ⏱️ Response time: 2-5s (agent invoked)
- 📋 **Check logs**: "⚠️  L2 cache not connected, skipping"

#### Step 3: Repeat Request (L1 Still Works)

**Request**: IDENTICAL to Step 2

**Expected**:
```json
{
  "cache_hit": true,   // ← L1 cache still works!
  "cache_layer": "L1"
}
```

**Verify**:
- ✅ cache_hit = true (L1 cache unaffected by Redis failure)
- ⚡ Response time: <100ms

#### Step 4: Restart Redis

**Command**:
```bash
docker start weather-ai-redis
```

#### Step 5: Verify L2 Cache Restored

**Request**: New query
```json
{
  "query": "Weather in Madrid?",
  "user_id": "degradation_test"
}
```

Repeat immediately.

**Expected**: Second request shows cache_layer = "L2" (Redis restored)

---

### Cache Testing Summary

#### Test Scenarios Checklist

- [ ] **Scenario 15**: L1 Cache Hit (<1ms)
- [ ] **Scenario 16**: L2 Cache Hit + L1 Backfill
- [ ] **Scenario 17**: Cache Key Isolation (user_id)
- [ ] **Scenario 18**: Cache TTL Expiration (optional, 6min wait)
- [ ] **Scenario 19**: Cache Performance Under Load
- [ ] **Scenario 20**: Graceful Degradation (Redis down)

#### Expected Results

| Metric | Target | Actual |
|--------|--------|--------|
| L1 Hit Rate | 15-25% | ___ % |
| L2 Hit Rate | 30-40% | ___ % |
| L1 Hit Latency | <1ms | ___ ms |
| L2 Hit Latency | <10ms | ___ ms |
| Overall Cost Savings | 60-75% | ___ % |

#### LangSmith Verification

For each cache hit scenario (15-16):

1. **Find trace in LangSmith Studio**
2. **Verify**:
   - ⚡ Latency <100ms (L1) or <200ms (L2)
   - 🚫 NO agent invocation span
   - ✅ Single "cache_lookup" span only
3. **Compare to cache miss**:
   - Cache miss: 5-10 spans (agent, tools, memory)
   - Cache hit: 1 span (cache lookup only)

#### Common Issues

**Issue 1**: cache_hit always false
- **Cause**: Redis not running or cache disabled
- **Fix**: Check `docker ps | grep redis` and `CACHE_ENABLED=true` in .env

**Issue 2**: cache_layer = null when should be "L1"
- **Cause**: Different query text (case-sensitive before normalization)
- **Fix**: Ensure EXACT same query text (cache normalizes internally)

**Issue 3**: L2 cache errors
- **Cause**: Redis connection refused
- **Fix**: `docker restart weather-ai-redis`

---

