# LangSmith Studio Setup Guide

## Overview

LangSmith Studio is the **official free visual debugger** from LangChain for developing and testing LangChain/LangGraph agents locally. It provides real-time visualization of agent execution, including prompts, tool calls, and results.

**Official Documentation**: https://docs.langchain.com/oss/python/langgraph/studio
**Official Youtube Video**: https://www.youtube.com/watch?v=Mi1gSlHwZLM

---

## What is LangSmith Studio?

LangSmith Studio is a specialized IDE for:
- ✅ **Visual debugging** - See every step your agent takes
- ✅ **Real-time interaction** - Test inputs and inspect states
- ✅ **Hot-reloading** - Changes to code reflect immediately
- ✅ **Exception capture** - Debug with full execution context
- ✅ **Metrics tracking** - Token usage, latency, cost per run
- ✅ **Thread management** - Re-run from any step

**Key Benefits**:
- No additional code changes required
- Works with all  LangChain/LangGraph patterns (ReAct, StateGraph, Command, interrupts)
- Free forever (no subscription)
- Official LangChain tooling

---

## Prerequisites

- ✅ Python 3.13+ (we have 3.13.5)
- ✅ LangChain v1.1.0+ (we have 1.1.0)
- ✅ LangGraph v1.0.4+ (we have 1.0.4)
- ✅ LangSmith API key configured in `.env`
- ✅ Weather AI Agent Level 1 complete

---

## Installation

### 1. Install LangGraph CLI with InMem Extra

```bash
# Using uv (our package manager)
uv pip install --upgrade "langgraph-cli[inmem]"

# Or using pip directly
pip install --upgrade "langgraph-cli[inmem]"
```

**Why `[inmem]` extra?**
- Enables in-memory state persistence for local development
- No Docker or PostgreSQL required
- Perfect for testing and debugging

### 2. Verify Installation

```bash
langgraph --version
# Expected output: langgraph-cli, version 0.4.7+
```

---

## Configuration

### 1. Create `langgraph.json` (Project Root)

This file tells the CLI where to find your agent graphs:

```json
{
  "dependencies": ["."],
  "graphs": {
    "weather_agent": "./backend/src/agents/weather_agent.py:create_weather_agent",
    "weather_hitl_workflow": "./backend/src/workflows/weather_graph.py:get_weather_hitl_workflow"
  },
  "env": ".env"
}
```

**Configuration Explained**:
- `dependencies`: Python packages to load (`.` = current directory)
- `graphs`: Agent/workflow entry points to expose
- `env`: Environment file for API keys

### 2. Verify Environment Variables

Ensure these are set in `.env`:

```bash
# LangSmith (required)
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=weather-ai-agent

# OpenAI (required for agent)
OPENAI_API_KEY=sk-proj-...

# MCP Servers (required for tools)
MCP_WEATHER_SERVER_URL=http://localhost:8080
MCP_HURRICANE_SERVER_URL=http://localhost:8081
```

---

## Usage

### 1. Start LangSmith Studio Server

```bash
# Start development server (default port 2024)
langgraph dev

# Options:
# --port 3000          Change port (default: 2024)
# --no-browser         Don't auto-open browser
# --tunnel             Use Cloudflare tunnel (for Safari compatibility)
# --allow-blocking     Don't error on sync I/O
```

**Expected Output**:
```
> Ready!
>
> - API: http://localhost:2024
> - Docs: http://localhost:2024/docs
> - LangSmith Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 2. Access Studio UI

**Option 1: Auto-opens in browser**
- Browser automatically opens Studio UI
- Connected to `http://127.0.0.1:2024`

**Option 2: Manual navigation**
- Visit: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

**Option 3: Via LangSmith Deployments**
- Go to https://smith.langchain.com/deployments
- Click "Studio" button
- Enter `http://127.0.0.1:2024` and click "Connect"

**Safari Users**: Use `langgraph dev --tunnel` (Safari blocks localhost connections)

---

## Testing Weather AI Agent in Studio

### 1. Select Agent Graph

In Studio UI:
1. Select graph from dropdown: `weather_agent` or `weather_hitl_workflow`
2. View graph structure visualization

### 2. Run Test Queries

**Example 1: Weather Query**
```
Input: "What's the weather in Seattle?"

Expected Flow:
1. Agent receives query
2. Calls get_current_weather tool
3. MCP client fetches data from weather-mcp server
4. Agent formats response
5. Returns weather data
```

**Example 2: Hurricane HITL Workflow**
```
Input: Category 4 hurricane alert

Expected Flow:
1. Workflow detects Category 4
2. Interrupt triggered (requires approval)
3. Studio shows "Pending Approval" state
4. Human approves via Studio UI
5. Workflow resumes and sends alert
```

### 3. Inspect Execution

Studio shows:
- ✅ **Prompts sent** to LLM (system + user messages)
- ✅ **Tool calls** with arguments
- ✅ **Tool results** from MCP servers
- ✅ **Intermediate states** at each node
- ✅ **Final output** to user
- ✅ **Metrics**: Tokens used, latency, cost

### 4. Debug Issues

When errors occur:
1. Studio captures exception with full context
2. See which node failed
3. Inspect input state at failure point
4. View error message and stack trace
5. Fix code and hot-reload automatically

---

## Common Use Cases

### Use Case 1: Debug MCP Integration
**Problem**: Agent can't fetch weather data
**Solution**:
1. Run query in Studio
2. Inspect `get_current_weather` tool call
3. See MCP request/response in trace
4. Verify session ID is being sent

### Use Case 2: Validate HITL Workflow
**Problem**: Hurricane alerts not interrupting correctly
**Solution**:
1. Send Category 4 alert in Studio
2. Verify workflow shows `__interrupt__` state
3. Test approval/rejection flows
4. Confirm correct branching

### Use Case 3: Optimize Prompts
**Problem**: Agent responses are verbose
**Solution**:
1. Run test query in Studio
2. View exact prompt sent to LLM
3. Edit system prompt in code
4. Hot-reload picks up changes
5. Re-run to see improvement

### Use Case 4: Monitor Token Usage
**Problem**: Queries are expensive
**Solution**:
1. Run multiple test queries
2. View token metrics per run
3. Identify which tools use most tokens
4. Optimize prompts or tool calls

---

## Hot-Reloading

**How It Works**:
- Studio watches for file changes in `backend/src/`
- Automatically restarts server on code changes
- No need to stop/restart `langgraph dev`

**Example Workflow**:
```bash
# 1. Start Studio
langgraph dev

# 2. Edit backend/src/agents/weather_agent.py
vim backend/src/agents/weather_agent.py
# Change system prompt

# 3. Save file
# → Studio auto-reloads (see logs: "Reloading...")

# 4. Re-run test in Studio UI
# → New prompt is active immediately
```

---

## Troubleshooting

### Issue 1: Port 2024 Already in Use
**Error**: `Address already in use`
**Solution**:
```bash
# Find process using port 2024
lsof -i :2024

# Kill process
kill -9 <PID>

# Or use different port
langgraph dev --port 3000
```

### Issue 2: Safari Blocks Localhost
**Error**: "Cannot connect to http://127.0.0.1:2024"
**Solution**:
```bash
# Use tunnel flag
langgraph dev --tunnel

# Access via public URL (shown in logs)
```

### Issue 3: Agent Not Found
**Error**: "Graph 'weather_agent' not found"
**Solution**:
1. Verify `langgraph.json` paths are correct
2. Check function is exported from module
3. Restart server: `Ctrl+C` then `langgraph dev`

### Issue 4: MCP Servers Not Reachable
**Error**: "Failed to connect to MCP server"
**Solution**:
```bash
# Verify MCP servers are running
docker ps | grep -E "weather-mcp|hurricane-mcp"

# Check .env URLs match Docker ports
grep MCP_ .env
```

### Issue 5: Missing Environment Variables
**Error**: "OPENAI_API_KEY not set"
**Solution**:
```bash
# Verify .env file exists and has keys
cat .env | grep -E "OPENAI_API_KEY|LANGCHAIN_API_KEY"

# Restart server after adding keys
```

---

## Integration with Other Tools

### With Pytest (Unit Tests)
```python
# tests/test_with_studio.py
import pytest
from backend.src.agents.weather_agent import create_weather_agent

async def test_weather_query_traced():
    """Test traced in Studio AND LangSmith."""
    agent = create_weather_agent()

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Weather in Seattle?"}]
    })

    # Test passes/fails
    assert "seattle" in result["messages"][-1].content.lower()

    # → Execution traced to both:
    # 1. LangSmith web UI (all test runs)
    # 2. Studio (if langgraph dev is running)
```

### With FastAPI (API Endpoints)
```python
# Run Studio + FastAPI simultaneously
# Terminal 1: FastAPI server
uvicorn backend.src.api.main:app --reload --port 8000

# Terminal 2: Studio server
langgraph dev --port 2024

# Test via Swagger, see traces in Studio
curl http://localhost:8000/weather/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Weather in London?", "user_id": "test"}'
```

---

## Best Practices

### 1. Use Studio During Active Development
- ✅ **DO**: Keep Studio running during coding sessions
- ✅ **DO**: Use for debugging new features
- ✅ **DO**: Test edge cases interactively
- ❌ **DON'T**: Use for automated testing (use Pytest)
- ❌ **DON'T**: Use in production (Studio is dev-only)

### 2. Leverage Hot-Reloading
- ✅ **DO**: Make small incremental changes
- ✅ **DO**: Test immediately after each change
- ❌ **DON'T**: Make massive refactors without testing

### 3. Document Findings
- ✅ **DO**: Screenshot interesting traces
- ✅ **DO**: Note token usage for optimization
- ✅ **DO**: Share traces with team (LangSmith URLs)

### 4. Combine with LangSmith Web UI
- Studio: Real-time debugging during development
- LangSmith: Historical analysis and team collaboration

---

## Performance Considerations

### Resource Usage
- **CPU**: Low (single Python process)
- **Memory**: ~200-500 MB (in-memory state)
- **Network**: Minimal (local HTTP on port 2024)
- **Disk**: Logs and state stored in `./.langgraph`

### State Persistence
- In-memory by default (lost on restart)
- For persistent state, use PostgreSQL checkpointer (Level 3+)

---

## Next Steps

### Level 1 (Current)
- ✅ Setup Studio (this guide)
- ✅ Debug ReAct agent tool calls
- ✅ Validate HITL workflow interrupts
- ✅ Test MCP integration

### Level 2 (CoT + RAG)
- Debug RAG retrieval (see which docs retrieved)
- Validate CoT reasoning chains
- Optimize prompt templates

### Level 3-6 (Advanced)
- Debug multi-agent handoffs (Level 4)
- Inspect memory updates (Level 3)
- Monitor production traces (Level 5)

---

## Additional Resources

- **Official Docs**: https://docs.langchain.com/oss/python/langgraph/studio
- **LangGraph CLI Reference**: https://docs.langchain.com/langsmith/cli
- **LangSmith Observability**: https://docs.langchain.com/oss/python/langchain/observability
- **Studio Tutorial**: https://docs.langchain.com/langsmith/quick-start-studio

---

**Last Updated**: 2025-12-04
**Version**: Level 1 Compatible
**Tested With**: LangChain 1.1.0, LangGraph 1.0.4, Python 3.13.5
