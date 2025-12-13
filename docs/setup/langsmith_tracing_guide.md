# LangSmith Tracing Guide for Weather AI Agent

**Project**: `weather-ai-agent-service`

This guide covers how to navigate, explore, and debug your Weather AI Agent using LangSmith's observability platform.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Dashboard Overview](#2-dashboard-overview)
3. [Navigating Traces](#3-navigating-traces)
4. [Understanding Runs](#4-understanding-runs)
5. [Analyzing LLM Calls](#5-analyzing-llm-calls)
6. [Debugging Techniques](#6-debugging-techniques)
7. [Performance Optimization](#7-performance-optimization)
8. [Filtering and Search](#8-filtering-and-search)
9. [Threads View](#9-threads-view)
10. [Evaluators](#10-evaluators)
11. [Alerts and Monitoring](#11-alerts-and-monitoring)
12. [Best Practices](#12-best-practices)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Getting Started

### Accessing LangSmith

1. **URL**: https://smith.langchain.com
2. **Login**: Use your LangSmith account (same as your API key account)
3. **Project**: Navigate to `weather-ai-agent-service`

### Configuration (Already Set Up)

Your `.env` file should have:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your-api-key-here
LANGCHAIN_PROJECT=weather-ai-agent-service
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Verifying Tracing is Active

1. Make a test query to your API:
   ```bash
   curl -X POST http://localhost:8000/weather/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the weather in Miami?", "user_id": "test123"}'
   ```

2. Check LangSmith - a new trace should appear within seconds

---

## 2. Dashboard Overview

### Main Navigation (Left Sidebar)

| Section | Purpose |
|---------|---------|
| **Home** | Overview and recent activity |
| **Tracing Projects** | All your traced projects |
| **Monitoring** | Real-time dashboards and alerts |
| **Datasets & Experiments** | Test datasets and evaluation runs |
| **Annotation Queues** | Human review workflows |
| **Prompts** | Prompt management and versioning |
| **Playground** | Interactive LLM testing |
| **Deployments** | Deployed chains/agents |

### Project Dashboard Stats (Right Panel)

When viewing a project, you'll see:

| Metric | What It Means | Target for Weather AI |
|--------|---------------|----------------------|
| **Run Count** | Total traces in time period | Higher = more testing |
| **Total Tokens** | LLM tokens consumed | Monitor for cost |
| **Total Cost** | API costs | Target: <$0.05/query |
| **Median Tokens** | Typical tokens per run | ~1,500-2,500 |
| **Error Rate** | % of failed runs | Target: <5% |
| **P50 Latency** | Median response time | Target: <10s |
| **P99 Latency** | Worst-case response time | Target: <45s |

---

## 3. Navigating Traces

### Trace List View

The main trace list shows all runs with these columns:

| Column | Description |
|--------|-------------|
| **Name** | Run identifier (often the workflow/agent name) |
| **Input** | First part of the input query |
| **Output** | First part of the response |
| **Error** | Error message if failed |
| **Start Time** | When the run started |
| **Latency** | Total execution time |
| **Dataset** | If linked to a dataset |
| **Tokens** | Total tokens used |

### Clicking on a Trace

Opens the **Trace Detail View** with:

1. **Timeline** - Visual execution flow
2. **Tree View** - Hierarchical call structure
3. **Input/Output** - Full request/response data
4. **Metadata** - Tags, user_id, session_id
5. **Feedback** - Human annotations

### Understanding the Trace Tree

```
LangGraph (root)
├── supervisor_node (8.2s)
│   └── ChatOpenAI (3.1s)
├── triage_agent_node (2.4s)
│   └── ChatOpenAI (1.8s)
├── hurricane_specialist_node (4.1s)
│   ├── ChatOpenAI (2.2s)
│   └── MCP Tool Call (1.5s)
└── alert_manager_node (3.5s)
    └── ChatOpenAI (2.8s)
```

- **Indentation** shows parent-child relationships
- **Time** shows duration of each step
- **Color coding**: Green = success, Red = error, Yellow = warning

---

## 4. Understanding Runs

### Run Types in Weather AI Agent

| Run Type | Description | What to Look For |
|----------|-------------|------------------|
| **LangGraph** | Multi-agent workflow | Total orchestration time |
| **ChatOpenAI** | LLM API call | Token count, latency |
| **Tool** | MCP server call | Response data, errors |
| **Retriever** | RAG retrieval | Documents retrieved |
| **Chain** | Sequential operations | Step-by-step flow |

### Run Statuses

| Status | Icon | Meaning |
|--------|------|---------|
| **Success** | ✅ Green | Completed without errors |
| **Error** | ❌ Red | Failed with exception |
| **Pending** | 🟡 Yellow | Still running |
| **Cancelled** | ⚪ Gray | Manually stopped |

### Key Metadata Fields

In each trace, check the **Metadata** tab for:

```json
{
  "user_id": "user123",
  "session_id": "session_abc",
  "query_complexity": "standard",
  "agent_level": "l4a",
  "agents_invoked": ["triage", "hurricane_specialist"],
  "cache_hit": false,
  "cache_layer": "MISS"
}
```

---

## 5. Analyzing LLM Calls

### Viewing LLM Call Details

Click on any `ChatOpenAI` or `ChatAnthropic` node to see:

1. **Input Messages** - The full prompt sent to the LLM
2. **Output** - The model's response
3. **Token Usage**:
   - `prompt_tokens` - Input tokens
   - `completion_tokens` - Output tokens
   - `total_tokens` - Sum of both
4. **Model Info** - Model name, temperature, etc.

### Token Analysis

```
Prompt Tokens:    1,200 (system + user + context)
Completion Tokens:  517 (model response)
Total Tokens:     1,717
Estimated Cost:   $0.002
```

### Checking Prompt Quality

In the **Input** section, verify:

1. **System Prompt** - Is it complete and correct?
2. **User Message** - Is the query properly formatted?
3. **Context** - Is memory/RAG context included?
4. **Tools** - Are the right tools available?

---

## 6. Debugging Techniques

### Finding Errors

**Method 1: Filter by Errors**
1. Click "Errors" filter button in trace list
2. Shows only failed runs

**Method 2: Search for Error Types**
```
error_type == "TimeoutError"
error_type == "ValidationError"
error_type == "HTTPStatusError"
```

### Common Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| `TimeoutError` | LLM/MCP call too slow | Check MCP server health |
| `ValidationError` | Invalid response format | Check Pydantic models |
| `RateLimitError` | API rate limit hit | Add retry logic |
| `ConnectionError` | Network/service down | Check Docker containers |
| `JSONDecodeError` | Malformed response | Check tool outputs |

### Debugging a Slow Trace

1. **Open the trace** and look at the timeline
2. **Identify the slowest node** (longest bar)
3. **Click on it** to see details
4. **Common culprits**:
   - LLM calls (check model, prompt length)
   - MCP tool calls (check server health)
   - Memory retrieval (check Neo4j/Redis)

### Debugging Incorrect Responses

1. **Check the Input** - Was the query understood correctly?
2. **Check Tool Calls** - Did tools return correct data?
3. **Check LLM Response** - Did the model reason correctly?
4. **Check Final Synthesis** - Was the output formatted properly?

### Using the Playground for Debugging

1. Click "Open in Playground" on any LLM call
2. Modify the prompt and test variations
3. Compare outputs
4. Save improved prompts

---

## 7. Performance Optimization

### Latency Breakdown

Typical Weather AI query latency:

```
Total: 6-10s
├── Memory retrieval:     0.5-2s
├── Query classification: 0.3-0.5s
├── Agent orchestration:  3-5s
│   ├── Triage:          1-2s
│   ├── Specialist:      2-3s
│   └── Synthesis:       1-2s
└── Cache write:         0.1-0.2s
```

### Identifying Bottlenecks

1. **Sort by Latency** (click column header)
2. **Look for P99 outliers** (>45s indicates problems)
3. **Check specific slow traces**:
   - Memory timeouts (>5s per layer)
   - MCP server delays (>2s)
   - LLM API latency (>5s per call)

### Optimization Opportunities

| Issue | Metric | Solution |
|-------|--------|----------|
| High token count | >5,000 tokens/query | Reduce context, use compression |
| Slow memory | >5s memory retrieval | Check Neo4j indexes |
| Many LLM calls | >5 calls/query | Use parallel execution |
| Cache misses | <30% hit rate | Check cache key design |
| Slow MCP | >2s per tool | Check MCP server health |

### Monitoring Cache Performance

Look for these metadata fields:
```json
{
  "cache_hit": true,
  "cache_layer": "L1",  // or "L2", "L3", "MISS"
  "cache_latency_ms": 5.2
}
```

Target cache hit rates:
- **L1 (Memory)**: 15-25%
- **L2 (Redis)**: 30-40%
- **Overall**: 40-60%

---

## 8. Filtering and Search

### Quick Filters (Top Bar)

| Filter | Purpose |
|--------|---------|
| **Last 7 days** | Time range selector |
| **Traces** | Show trace roots only |
| **LLM Calls** | Show only LLM invocations |
| **Errors** | Show only failed runs |

### Advanced Search Syntax

**By Input Content:**
```
input CONTAINS "hurricane"
input CONTAINS "Tampa"
```

**By Output Content:**
```
output CONTAINS "Category 5"
output CONTAINS "evacuate"
```

**By Metadata:**
```
metadata.user_id == "user123"
metadata.agent_level == "l4b"
metadata.cache_hit == true
```

**By Performance:**
```
latency > 10s
total_tokens > 5000
error_type IS NOT NULL
```

**Combined Filters:**
```
input CONTAINS "hurricane" AND latency > 5s
metadata.agent_level == "l4b" AND error_type IS NOT NULL
```

### Saving Filter Presets

1. Apply your filters
2. Click "Save as View"
3. Name it (e.g., "Slow Hurricane Queries")
4. Access from "Views" dropdown

### Useful Saved Views for Weather AI

| View Name | Filter | Purpose |
|-----------|--------|---------|
| Slow Queries | `latency > 15s` | Performance issues |
| Cache Misses | `metadata.cache_hit == false` | Cache optimization |
| Emergency Tier | `metadata.agent_level == "l4c"` | Critical queries |
| Errors Only | `error_type IS NOT NULL` | Bug investigation |
| Hurricane Queries | `input CONTAINS "hurricane"` | Domain-specific |

---

## 9. Threads View

### What is Threads View?

Groups traces by `session_id` to show conversation history.

### Accessing Threads

1. Click **Threads** tab (next to Runs)
2. See all sessions grouped together
3. Click a thread to see full conversation

### Thread Details Show:

- All messages in the session
- Chronological order
- Memory context evolution
- User interaction patterns

### Using Threads for Debugging

1. Find a problematic trace
2. Note the `session_id`
3. Go to Threads view
4. Find the session
5. See full context of what led to the issue

---

## 10. Evaluators

### What are Evaluators?

Automated quality checks that run on your traces.

### Built-in Evaluators

| Evaluator | What It Checks |
|-----------|----------------|
| **Correctness** | Does output match expected? |
| **Relevance** | Is response relevant to query? |
| **Coherence** | Is response well-structured? |
| **Harmfulness** | Contains harmful content? |
| **Groundedness** | Based on retrieved docs? |

### Setting Up Evaluators

1. Go to **Evaluators** tab
2. Click **+ New Evaluator**
3. Choose evaluator type
4. Configure criteria
5. Apply to project

### Custom Evaluators for Weather AI

**Hurricane Safety Evaluator:**
```python
def hurricane_safety_eval(run, example):
    """Check hurricane responses for safety compliance."""
    output = run.outputs.get("response", "")

    # Check for category validation
    if "Category 5" in output and "157 mph" not in output:
        return {"score": 0, "comment": "Missing wind speed validation"}

    # Check for evacuation guidance
    if "evacuate" in output.lower() and "zone" not in output.lower():
        return {"score": 0.5, "comment": "Missing evacuation zone"}

    return {"score": 1, "comment": "Passed safety checks"}
```

---

## 11. Alerts and Monitoring

### Setting Up Alerts

1. Go to **Monitoring** → **Alerts**
2. Click **+ Alert**
3. Configure:
   - **Metric**: Error rate, latency, etc.
   - **Threshold**: When to trigger
   - **Window**: Time period
   - **Notification**: Email, Slack, etc.

### Recommended Alerts for Weather AI

| Alert | Condition | Priority |
|-------|-----------|----------|
| High Error Rate | error_rate > 5% over 1h | Critical |
| Slow P99 | p99_latency > 45s over 1h | High |
| Cost Spike | daily_cost > $10 | Medium |
| Memory Timeout | metadata.memory_timeout > 10 over 1h | High |
| MCP Failures | error CONTAINS "MCP" > 5 over 1h | Critical |

### Prebuilt Dashboards

Access via **Monitoring** → **Prebuilt Dashboards**:

- **Latency Dashboard**: P50, P95, P99 trends
- **Error Dashboard**: Error types and frequencies
- **Cost Dashboard**: Token usage and costs
- **Volume Dashboard**: Request patterns

### Custom Dashboards

1. Go to **Custom Dashboards**
2. Click **+ New Dashboard**
3. Add widgets:
   - Time series charts
   - Tables
   - Stat cards
   - Heatmaps

---

## 12. Best Practices

### Daily Monitoring Routine

1. **Check Error Rate** - Should be <5%
2. **Check P99 Latency** - Should be <45s
3. **Review New Errors** - Investigate any new error types
4. **Check Cache Hit Rate** - Should be >40%
5. **Monitor Costs** - Stay within budget

### Weekly Review

1. **Analyze Slow Queries** - Find patterns in P99 outliers
2. **Review Error Trends** - Are errors increasing/decreasing?
3. **Check Token Usage** - Optimize high-token queries
4. **Update Filters** - Refine saved views
5. **Review Evaluator Results** - Fix failing criteria

### Debugging Workflow

```
1. User reports issue
       ↓
2. Search by user_id/session_id
       ↓
3. Find the problematic trace
       ↓
4. Check trace timeline for slowest step
       ↓
5. Examine inputs/outputs at each step
       ↓
6. Identify root cause
       ↓
7. Test fix in Playground
       ↓
8. Deploy fix
       ↓
9. Verify in new traces
```

### Tagging Conventions

Add tags to important traces:

| Tag | Purpose |
|-----|---------|
| `bug:confirmed` | Known bug investigation |
| `perf:slow` | Performance issue |
| `quality:poor` | Quality issue |
| `test:regression` | Regression test |
| `review:needed` | Needs human review |

---

## 13. Troubleshooting

### Traces Not Appearing

**Check 1: Environment Variables**
```bash
# In container
docker exec weather-ai-api env | grep LANGCHAIN
```

Should show:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=weather-ai-agent-service
```

**Check 2: Network Connectivity**
```bash
# Test LangSmith API
curl -I https://api.smith.langchain.com/health
```

**Check 3: API Key Valid**
- Go to https://smith.langchain.com/settings
- Verify API key is active

### Traces Delayed

- LangSmith has ~5-30 second ingestion delay
- Check "Most Recent Run" timestamp
- Wait and refresh

### Missing LLM Calls in Trace

- Ensure `ChatOpenAI` or `ChatAnthropic` is wrapped with tracing
- Check if `LANGCHAIN_TRACING_V2=true`
- Verify callbacks are propagating through chains

### High Token Counts

1. Check system prompt length
2. Check memory context size
3. Check RAG document count
4. Use token compression (Level 3a)

### Inconsistent Latency

1. Check MCP server health
2. Check database connection pools
3. Check for memory leaks
4. Review async/await patterns

---

## Quick Reference Card

### Keyboard Shortcuts (in LangSmith UI)

| Shortcut | Action |
|----------|--------|
| `j/k` | Navigate up/down in list |
| `Enter` | Open selected trace |
| `Esc` | Close detail view |
| `f` | Focus search |
| `?` | Show help |

### Useful URLs

| Page | URL |
|------|-----|
| Project Dashboard | `https://smith.langchain.com/o/{org}/projects/weather-ai-agent-service` |
| Traces | `https://smith.langchain.com/o/{org}/projects/weather-ai-agent-service/runs` |
| Threads | `https://smith.langchain.com/o/{org}/projects/weather-ai-agent-service/threads` |
| Monitoring | `https://smith.langchain.com/o/{org}/dashboards` |
| Settings | `https://smith.langchain.com/settings` |

### API Quick Commands

```bash
# List recent runs
curl -X GET "https://api.smith.langchain.com/runs" \
  -H "x-api-key: $LANGCHAIN_API_KEY" \
  -d '{"project_name": "weather-ai-agent-service", "limit": 10}'

# Get specific run
curl -X GET "https://api.smith.langchain.com/runs/{run_id}" \
  -H "x-api-key: $LANGCHAIN_API_KEY"
```

---

## Support Resources

- **LangSmith Docs**: https://docs.smith.langchain.com
- **LangChain Discord**: https://discord.gg/langchain
- **GitHub Issues**: https://github.com/langchain-ai/langsmith-sdk/issues

---
