# Level 2 Test Guide

Quick guide to test RAG + CoT + Hybrid Search using Swagger UI and LangGraph Studio.

## Table of Contents

- [Level 2 Test Guide](#level-2-test-guide)
  - [Table of Contents](#table-of-contents)
  - [What's Being Tested](#whats-being-tested)
  - [Prerequisites](#prerequisites)
    - [1. Start services](#1-start-services)
    - [2. Verify services](#2-verify-services)
    - [3. Access UIs](#3-access-uis)
  - [Swagger UI Tests](#swagger-ui-tests)
    - [Test 1: Basic Weather Query (No RAG/CoT)](#test-1-basic-weather-query-no-ragcot)
    - [Test 2: RAG-Enhanced Query](#test-2-rag-enhanced-query)
    - [Test 3: CoT Reasoning Query](#test-3-cot-reasoning-query)
    - [Test 4: RAG + CoT Combined](#test-4-rag--cot-combined)
    - [Test 5: Hybrid Search (Semantic + Keyword)](#test-5-hybrid-search-semantic--keyword)
    - [Test 6: Pattern Analysis (RAG Tool)](#test-6-pattern-analysis-rag-tool)
  - [LangSmith Studio Tests](#langsmith-studio-tests)
    - [Prerequisites](#prerequisites-1)
    - [1. Start LangSmith Studio Server](#1-start-langsmith-studio-server)
    - [Configuration 1: Basic Agent (No RAG/CoT)](#configuration-1-basic-agent-no-ragcot)
    - [Configuration 2: RAG Agent](#configuration-2-rag-agent)
    - [Configuration 3: CoT Agent](#configuration-3-cot-agent)
    - [Configuration 4: Multi-Turn Conversation](#configuration-4-multi-turn-conversation)
    - [Configuration 5: Complex Multi-Tool Query](#configuration-5-complex-multi-tool-query)
    - [Configuration 6: Edge Case - Ambiguous Query](#configuration-6-edge-case---ambiguous-query)
  - [Quick Validation Checklist](#quick-validation-checklist)
    - [Swagger UI](#swagger-ui)
    - [LangGraph Studio](#langgraph-studio)
  - [Common Issues](#common-issues)
  - [Success Criteria](#success-criteria)
  - [Quick Test Script](#quick-test-script)

---

## What's Being Tested

Level 2 features:
- **RAG**: 4 enhanced tools with 603 weather documents
- **CoT**: 5-step reasoning framework
- **Hybrid Search**: 70% semantic + 30% keyword (BM25)

---

## Prerequisites

### 1. Start services
```bash
make docker-up-dev
```
or

```bash
docker-compose up -d
```

### 2. Verify services
- Check the Health amd makesure API ready
```bash
curl http://localhost:8000/health        
```
- Check vector database Qdrant ready
```bash
curl http://localhost:6333/collections   
```

### 3. Access UIs
**Swagger:** http://localhost:8000/docs
**LangGraph Studio:** http://localhost:56789


---

## Swagger UI Tests

### Test 1: Basic Weather Query (No RAG/CoT)

**Endpoint**: `POST /weather/query`

```json
{
  "query": "What's the weather in Miami?",
  "use_rag": false,
  "use_cot": false
}
```

**Expected**:
- Uses basic MCP weather tools only
- Fast response (<2s)
- Direct weather data from MCP server

**Verify**: Response contains current temperature, conditions

---

### Test 2: RAG-Enhanced Query

**Endpoint**: `POST /weather/query`

```json
{
  "query": "What are the characteristics of a Category 5 hurricane?",
  "use_rag": true,
  "use_cot": false
}
```

**Expected**:
- Uses RAG tool: `retrieve_weather_knowledge`
- Retrieves from 603 document knowledge base
- Response includes: wind speed (157+ mph), Saffir-Simpson scale details

**Verify**: Response cites knowledge base information, mentions specific wind speeds

---

### Test 3: CoT Reasoning Query

**Endpoint**: `POST /weather/query`

```json
{
  "query": "Should I evacuate for a hurricane with 140 mph winds?",
  "use_rag": false,
  "use_cot": true
}
```

**Expected**:
- 5-step CoT framework activated
- Response shows reasoning: Understand → Plan → Execute → Verify → Respond
- Includes evacuation recommendation with reasoning

**Verify**: Response contains reasoning steps, category classification (Cat 4), evacuation advice

---

### Test 4: RAG + CoT Combined

**Endpoint**: `POST /weather/query`

```json
{
  "query": "Compare the storm surge potential of Category 3 vs Category 5 hurricanes",
  "use_rag": true,
  "use_cot": true
}
```

**Expected**:
- Uses both RAG (knowledge retrieval) and CoT (reasoning)
- Retrieves storm surge data from knowledge base
- Applies reasoning to compare categories
- Response >200 tokens with detailed comparison

**Verify**: Mentions specific surge heights (9-12 ft for Cat 3, 18+ ft for Cat 5), cites sources

---

### Test 5: Pattern Analysis (RAG Tool)

**Endpoint**: `POST /weather/query`

```json
{
  "query": "What patterns indicate an intensifying tropical storm?",
  "use_rag": true,
  "use_cot": false
}
```

**Expected**:
- Uses RAG tool: `identify_patterns`
- Retrieves pattern data from knowledge base
- Lists indicators: pressure drop, wind increase, eye formation

**Verify**: Response lists 3+ specific patterns with meteorological details

---

## LangSmith Studio Tests

### Prerequisites
Refer [LangSmith Studio Setup](./../setup/langsmith-studio-setup.md)

### 1. Start LangSmith Studio Server

```bash
langgraph dev
```
**Access**: <https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024>

### Configuration 1: Basic Agent (No RAG/CoT)

**Select Graph**: `weather_agent`

**Test Case**: Simple Weather Query

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What's the temperature in Boston?"
    }
  ]
}
```

**Expected**:
- Agent uses `get_weather` MCP tool
- Single tool call
- Response in <3 seconds

**Verify**: Temperature value returned, no RAG retrieval shown

---

### Configuration 2: RAG Agent

**Select Graph**: `weather_agent`

**Test Case**: Knowledge-Based Query

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the Saffir-Simpson scale?"
    }
  ]
}
```

**Enable**: RAG mode in Studio

**Expected**:
- Agent uses `retrieve_weather_knowledge` tool
- Shows retrieval from vector store
- Response includes 5 hurricane categories

**Verify**: Trace shows vector store retrieval, response cites category definitions

---

### Configuration 3: CoT Agent

**Select Graph**: `weather_agent`

**Test Case**: Reasoning Required

```json
{
  "messages": [
    {
      "role": "user",
      "content": "A hurricane has sustained winds of 125 mph. What category is it and should coastal residents evacuate?"
    }
  ]
}
```

**Enable**: CoT mode in Studio

**Expected**:
- Agent shows 5-step reasoning process
- Classifies as Category 3 (111-129 mph)
- Provides evacuation recommendation

**Verify**: Trace shows CoT steps, reasoning visible in response

---

### Configuration 4: Multi-Turn Conversation

**Test Case**: Follow-up Questions

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Tell me about Hurricane categories"
    }
  ]
}
```

Then add follow-up:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Tell me about Hurricane categories"
    },
    {
      "role": "assistant",
      "content": "<previous response>"
    },
    {
      "role": "user",
      "content": "What's the wind speed for Category 5?"
    }
  ]
}
```

**Expected**:
- Agent maintains context
- Second response references first query
- Provides specific Cat 5 wind speed (157+ mph)

**Verify**: Coherent conversation flow, context awareness

---

### Configuration 5: Complex Multi-Tool Query

**Test Case**: Requires Multiple Tools

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Get current weather for Miami, then compare it to historical hurricane patterns in Florida"
    }
  ]
}
```

**Enable**: RAG + CoT

**Expected**:
- Agent calls `get_weather` (current data)
- Agent calls `analyze_trends` (historical patterns)
- Agent synthesizes both results
- Response >300 tokens

**Verify**: Trace shows 2+ tool calls, final response combines both data sources

---

### Configuration 6: Edge Case - Ambiguous Query

**Test Case**: Vague Question

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Is it dangerous?"
    }
  ]
}
```

**Expected**:
- Agent asks for clarification
- Or provides general safety information
- Handles gracefully without errors

**Verify**: No crashes, reasonable response

---

## Quick Validation Checklist

### Swagger UI
- [ ] Basic query works (Test 1)
- [ ] RAG retrieves from knowledge base (Test 2)
- [ ] CoT shows reasoning steps (Test 3)
- [ ] RAG + CoT combines both (Test 4)
- [ ] Hybrid search returns ranked results (Test 5)
- [ ] RAG tools work individually (Test 6)

### LangGraph Studio
- [ ] Basic agent responds (Config 1)
- [ ] RAG agent retrieves documents (Config 2)
- [ ] CoT agent shows reasoning (Config 3)
- [ ] Multi-turn conversation works (Config 4)
- [ ] Multi-tool queries execute (Config 5)
- [ ] Edge cases handled gracefully (Config 6)

---

## Common Issues

**Issue**: RAG returns no results
- **Fix**: Check Qdrant has documents: `curl http://localhost:6333/collections/weather_knowledge/points/count`
- **Expected**: Count > 0

**Issue**: Hybrid search fails
- **Fix**: Rebuild BM25 index by restarting API service

**Issue**: CoT not showing reasoning
- **Fix**: Ensure `use_cot: true` in request

**Issue**: LangGraph Studio connection failed
- **Fix**: Restart Studio: `langgraph dev` in terminal

---

## Success Criteria

✅ **All 6 Swagger tests pass** (responses contain expected data)
✅ **All 6 LangGraph configs work** (no errors in trace)
✅ **Response times**:
  - Basic queries: <2s
  - RAG queries: <5s
  - CoT queries: <8s
  - RAG + CoT: <10s

✅ **Knowledge base functioning**: Responses cite specific data from 603 documents
✅ **Hybrid search accuracy**: Top results relevant to query (semantic + keyword match)

---

## Quick Test Script

```bash
# Test all endpoints quickly
curl -X POST "http://localhost:8000/weather/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Weather in NYC?","use_rag":false,"use_cot":false}'

curl -X POST "http://localhost:8000/weather/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a Category 5 hurricane?","use_rag":true,"use_cot":false}'

# Check health
curl http://localhost:8000/health

# Check vector store
curl http://localhost:6333/collections/weather_knowledge
```

---

