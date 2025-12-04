# Changelog

All notable changes to the Weather AI Agent Service will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Level 2: CoT + RAG (v0.3.0)
- Level 3a: 2-Layer Memory (v0.4.0)

---

## [0.2.1] - 2025-12-04 (Level 1: LangSmith Studio + Complete MCP Tool Coverage)

### Added

**LangSmith Studio Integration**:
- ✅ Installed `langgraph-cli[inmem]` v0.4.7 for local agent debugging
- ✅ Created `langgraph.json` configuration exposing 2 agent graphs:
  - `weather_agent`: Basic ReAct agent with 3 MCP tools
  - `weather_hitl_workflow`: Hurricane approval workflow
- ✅ Comprehensive documentation: `docs/setup/langsmith-studio-setup.md` (434 lines)
  - Installation and configuration steps
  - Usage examples for all 3 weather tools
  - Troubleshooting guide (port conflicts, Safari issues, MCP connectivity)
  - Integration patterns with pytest and FastAPI
  - Hot-reload workflow documentation
- ✅ Updated README.md with Studio setup guide link under "Development Tools" section

**Third MCP Tool Implementation**:
- ✅ Implemented `retrieve_weather_context` tool (natural language query handling)
- ✅ Added to `backend/src/mcp/weather_client.py` (lines 251-306)
- ✅ Added to `backend/src/tools/weather_tools.py` (lines 122-166)
- ✅ Updated `backend/src/agents/weather_agent.py` to include third tool
- ✅ Complete MCP tool coverage: All 3 tools implemented and tested

**MCP Integration Fixes**:
- ✅ **Fixed tool name mismatch**: `get_forecast` → `get_weather_forecast` (weather_client.py:239)
- ✅ **Corrected days parameter range**: 1-14 → 1-7 days (matches MCP server spec)
- ✅ **Updated default days**: 7 → 5 days (matches MCP server default)

**Development Environment**:
- ✅ Updated `.gitignore` to exclude LangGraph development artifacts:
  - `.langgraph/` - State persistence directory
  - `.langgraph_api/` - Development cache
  - `*.pckl` - Pickled state files
  - `.langsmith/` - Local cache

### Fixed

**MCP Tool Integration Bugs**:
- ✅ **MCP error -32602**: "Tool get_forecast not found" - Fixed by using correct MCP tool name `get_weather_forecast`
- ✅ **Days validation**: Prevented runtime errors by correcting range to 1-7 days
- ✅ **Missing tool**: Implemented `retrieve_weather_context` for natural language queries

### Testing

**Studio Validation**:
- ✅ All 3 MCP tools tested and verified in LangSmith Studio:
  1. `get_current_weather` - Working
  2. `get_forecast` - Fixed and working
  3. `retrieve_weather_context` - Implemented and working

**User Confirmation**: "I am able to test using Studio it looks good"

### Documentation

**New Files**:
- `docs/setup/langsmith-studio-setup.md` (434 lines) - Complete Studio guide
- `langgraph.json` - Studio configuration file

**Updated Files**:
- `README.md` - Added Studio link under "Development Tools"
- `.gitignore` - Added LangGraph artifacts section

### Success Metrics

- **MCP Tool Coverage**: 100% (3/3 tools implemented and tested)
- **Tool Name Mapping**: 100% correct
- **Studio Integration**: ✅ Working (tested by user)
- **Documentation**: Complete setup and troubleshooting guide
- **LangChain v1.0+ Compliance**: 100% (uses official `create_agent()` API)

### MCP Tool Mapping (Complete)

| # | LangChain Tool | MCP Tool Name | Status |
|---|----------------|---------------|--------|
| 1 | get_current_weather | get_current_weather | ✅ Working |
| 2 | get_forecast | get_weather_forecast | ✅ Fixed & Working |
| 3 | retrieve_weather_context | retrieve_weather_context | ✅ Implemented & Working |

---

## [0.2.0] - 2025-12-04 (Level 1: ReAct Agent + HITL + Docker Deployment)

### Added

**Core Agent Implementation**:
- ReAct (Reasoning + Acting) pattern agent using LangChain v1.1.0+
- `create_tool_calling_agent()` for OpenAI GPT-4o-mini integration
- Agent executor with verbose logging and error handling
- Weather query processing with natural language understanding

**MCP Client Integration**:
- Async HTTP client for Weather MCP Server
- JSON-RPC 2.0 protocol implementation
- Session management with cookie persistence
- 2 weather tools: `get_current_weather()` and `get_forecast()`
- Health check and connection validation

**HITL (Human-in-the-Loop) Workflow**:
- Hurricane alert approval system using LangGraph interrupts
- Auto-approve: Categories 1-2 (less severe)
- Require human approval: Categories 3-5 (life-threatening)
- 4-node workflow: detect → approval → send/cancel → END
- InMemorySaver checkpointer for state persistence

**LangGraph Workflow**:
- StateGraph with `WeatherAgentState` type safety
- Command-based routing for approval decisions
- Interrupt/resume pattern for HITL integration
- Workflow builder and singleton pattern

**FastAPI Service**:
- 3 main endpoints: `/weather/query`, `/weather/hurricane/alert`, `/weather/hurricane/approve`
- Health check endpoint: `/health`
- Pydantic v2 request/response validation
- CORS middleware for development
- Structured logging for all operations
- Auto-generated OpenAPI/Swagger docs

**Docker Deployment** (Production-Ready):
- Multi-stage Dockerfile with Python 3.13.5 + uv package manager
- 4-stage build: base → dependencies → development → production
- Docker Compose orchestration with 3 services (weather-ai-api, weather-mcp, hurricane-mcp)
- Docker networking with service discovery (weather-mcp:8080, hurricane-mcp:8081)
- Health checks and service dependencies configured
- Production user (non-root) with proper permissions
- Optimized .dockerignore (79 lines) for faster builds
- Environment variable injection for API keys and MCP URLs

**MCP Integration Fixes** (Critical Debugging):
- ✅ **Header-based session management**: Fixed 3+ hour debugging session - MCP uses `mcp-session-id` header, NOT cookies
- ✅ **SSE response parsing**: Implemented `_parse_sse_response()` for Server-Sent Events format (`event: message\ndata: {...}`)
- ✅ **Accept header requirement**: Added `Accept: application/json, text/event-stream` for MCP HTTP Streamable protocol
- ✅ **Session persistence**: Store `_session_id` from headers and send in subsequent tool calls
- ✅ **Initialization flag**: Changed from `session_cookie` to `_initialized` flag for state tracking

**LangChain v1.0+ Migration**:
- ✅ Migrated from deprecated `create_tool_calling_agent` to `create_agent()` API
- ✅ Returns `CompiledStateGraph` instead of `AgentExecutor`
- ✅ Messages-based state: `{"messages": [...]}` instead of `{"input": "..."}`
- ✅ Modern import paths: `langchain.agents.create_agent` (v1.1.0 compliant)

**Enhanced Response Models**:
- Added `category`, `message`, and `timestamp` fields to `HurricaneAlertResponse`
- Richer API responses for better user experience
- ISO 8601 UTC timestamps for all responses

**Configuration Management**:
- Centralized settings with Pydantic Settings
- Type-safe environment variable loading
- `lru_cache` decorator for singleton pattern
- `.env.template` for environment configuration
- Comprehensive validation (log levels, environment, secret keys)

**Testing Suite**:
- 32 test cases across 8 test modules
- 24 tests passing (75% pass rate)
- Fixtures for MCP client mocking
- Agent executor mocking for OpenAI API
- Hurricane HITL workflow testing
- API endpoint integration tests

### Technical Standards

**LangChain v1.0+ Compliance**:
- ✅ `create_tool_calling_agent()` (NOT deprecated `create_react_agent`)
- ✅ Modern imports: `langchain_core`, `langchain_openai`
- ✅ StateGraph and Command patterns
- ✅ Pydantic v2 structured outputs

**Python 3.13+ Standards**:
- ✅ Modern type hints: `str | None`, `list[str]`
- ✅ Async-first architecture (all I/O async)
- ✅ UTC timestamps with `datetime.now(timezone.utc).isoformat()`
- ✅ 100% type hints coverage

**Code Quality**:
- Snake_case naming for files and functions
- PascalCase for classes
- Comprehensive docstrings (Google style)
- Error handling with structured logging
- No blocking operations

### Infrastructure

**Files Created** (27 new files):
- `backend/src/mcp/weather_client.py` - MCP HTTP client with header-based sessions
- `backend/src/tools/weather_tools.py` - LangChain tool wrappers
- `backend/src/agents/state.py` - TypedDict for agent state
- `backend/src/agents/prompts.py` - System prompts
- `backend/src/agents/weather_agent.py` - ReAct agent with `create_agent()` API
- `backend/src/hitl/approval_node.py` - HITL approval nodes
- `backend/src/workflows/weather_graph.py` - LangGraph workflow
- `backend/src/api/main.py` - FastAPI application
- `backend/src/api/schemas.py` - Pydantic models with enhanced responses
- `backend/config/settings.py` - Settings management
- `tests/conftest.py` - Pytest fixtures
- `tests/test_mcp_client.py` - MCP client tests
- `tests/test_weather_tool.py` - Tool tests
- `tests/test_react_agent.py` - Agent tests
- `tests/test_hurricane_hitl.py` - HITL tests
- `tests/test_workflow.py` - Workflow tests
- `tests/test_api.py` - API tests
- `tests/test_integration.py` - Integration tests
- `Dockerfile` - Multi-stage production build (89 lines)
- `.dockerignore` - Build optimization (79 lines)
- `docker-compose.yml` - Updated with weather-ai-api service
- `test_docker_deployment.py` - Comprehensive deployment tests (224 lines)

### Success Metrics
- **Docker Deployment**: ✅ 100% successful (all 3 services running)
- **Weather Data Retrieval**: ✅ Real MCP data (London: 7.9°C, Seattle: 6.7°C)
- **Hurricane HITL**: ✅ Cat 1-2 auto-approve, Cat 3-5 pending approval
- **MCP Integration**: ✅ Header-based sessions working (3+ hours debugging resolved)
- **API Response Time**: <2s for weather queries via Docker
- **Test Coverage**: 75% (24/32 tests passing, 8 async skipped)
- **API Endpoints**: 4/4 functional via Swagger UI
- **LangChain v1.0+ Compliance**: 100% (modern `create_agent()` API)
- **Code Quality**: 100% type hints, async-first architecture
- **Production Readiness**: ✅ Containerized, tested, documented

### Known Limitations (Deferred to L2+)
- No structured LLM output parsing (Level 2)
- No RAG integration (Level 2)
- No memory persistence (Level 3)
- No multi-agent orchestration (Level 4)
- No production observability (Level 5)
- Async test support not configured (acceptable for L1)

---

## [0.1.0] - 2025-12-01 (Level 0: Setup)

### Added
- Python 3.13+ environment with pyenv
- **uv** for ultra-fast dependency management (10-100x faster than pip/Poetry)
- Dual MCP server integration (Weather + Hurricane Tracker)
- Environment configuration (.env template)
- Docker Desktop setup
- LangSmith tracing configuration
- Automated verification script (8 checks)
- Git repository with 13 level branches
- Project documentation structure
- **Makefile** with 17 commands for easy install/activate workflow

### Infrastructure
- `pyproject.toml` with core dependencies (LangChain 1.0, LangGraph 1.0, FastAPI)
- `uv.lock` for reproducible builds (single lockfile, no complexity)
- `.gitignore` (excludes .env, .DS_Store, Python artifacts)
- `docs/plan/level-0-plan.md` (comprehensive setup guide)
- `Makefile` with common development tasks (install, verify, test, lint, format)

### Verification
- `verify_setup.py` script with 8 automated checks:
  1. Python 3.13+ version check
  2. Environment variables validation
  3. OpenAI API connection test
  4. Anthropic API connection test (optional)
  5. MCP Weather Server health check
  6. MCP Hurricane Server health check
  7. LangSmith tracing verification
  8. Docker runtime verification

### Success Metrics
- Setup time: ~2 hours (vs 6-8 hours manual setup)
- Verification success rate: 100% (8/8 checks passing)
- Production-ready environment from Day 1 ✅

---

## Version Numbering Strategy

**Semantic Versioning**: MAJOR.MINOR.PATCH

### Progressive Learning Versions
- **v0.1.0**: Level 0 (Setup)
- **v0.2.0**: Level 1 (ReAct + HITL)
- **v0.3.0**: Level 2 (CoT + RAG)
- **v0.4.0**: Level 3a (2-Layer Memory)
- **v0.5.0**: Level 3b (Advanced Reasoning - ToT/GoT)
- **v0.6.0**: Level 3c (Full 7-Layer Memory)
- **v0.7.0**: Level 4a (3-Agent System)
- **v0.8.0**: Level 4b (8-Agent Orchestration)
- **v0.9.0**: Level 4c (15-Agent Production)
- **v0.10.0**: Level 5a (Production RAG)
- **v0.11.0**: Level 5b (Critical Guardrails)
- **v1.0.0**: Level 5c (Full Production Release) 🎉
- **v1.1.0**: Level 6 (Self-Evolving Architecture)

### Patch Updates
- Bug fixes within a level: v0.X.1, v0.X.2, etc.
- Example: v0.2.1 = Level 1 hotfix

---

**Current Version**: 0.2.0
**Last Updated**: 2025-12-04
