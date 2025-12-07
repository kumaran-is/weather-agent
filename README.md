# Weather AI Agent Service

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kumaran-is/weather-agent)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.0+](https://img.shields.io/badge/langchain-1.0+-green.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph 1.0+](https://img.shields.io/badge/langgraph-1.0+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![MCP Protocol](https://img.shields.io/badge/MCP%20Protocol-Dual%20Servers-orange)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](./CHANGELOG.md)

**Production-grade AI agent for weather forecast intelligence:** Built with LangChain 1.0, LangGraph 1.0, FastAPI and OpenAI. Features real-time weather and hurricane data via MCP integration, RAG-enhanced tools with hybrid search (semantic + keyword), Chain-of-Thought reasoning, streaming responses, type-safe validation with Pydantic, structured logging, and enterprise-grade observability.

**From zero to production** Progressive implementation showcasing enterprise AI patterns, multi-agent orchestration, 7-layer memory systems, and production deployment strategies.

**How It Works:** This project follows a progressive complexity model—each module builds on the previous one, making it accessible to beginners while scaling toward advanced concepts.

**Current Stage**: ✅ **Level 2 Implementation Complete** — Unified Agent with RAG + CoT + Hybrid Search

[Read the Medium Blog Post Series](https://medium.com/@yourusername)

---

## Table of Contents

- [Weather AI Agent Service](#weather-ai-agent-service)
  - [Table of Contents](#table-of-contents)
  - [Technology Stack](#technology-stack)
  - [Testing Tools \& Interfaces](#testing-tools--interfaces)
  - [Quick Start](#quick-start)
    - [System Requirements](#system-requirements)
    - [Prerequisites](#prerequisites)
    - [MCP Servers Required](#mcp-servers-required)
    - [Setup (15 minutes)](#setup-15-minutes)
      - [Option 1: Quick Start with Makefile (Recommended)](#option-1-quick-start-with-makefile-recommended)
    - [Required API Keys](#required-api-keys)
    - [MCP Server Configuration](#mcp-server-configuration)
    - [RAG \& Vector Store Configuration (Level 2)](#rag--vector-store-configuration-level-2)
    - [Agent Feature Flags (Level 2)](#agent-feature-flags-level-2)
    - [LangSmith Observability (Optional)](#langsmith-observability-optional)
    - [Optional/Advanced Configuration](#optionaladvanced-configuration)
      - [Option 2: Manual Setup (Without Makefile)](#option-2-manual-setup-without-makefile)
    - [Common Makefile Commands](#common-makefile-commands)
    - [Project Structure](#project-structure)
    - [Progressive Learning Path](#progressive-learning-path)
  - [API Endpoints Reference](#api-endpoints-reference)
  - [Testing Level 2](#testing-level-2)
  - [Contributing](#contributing)
  - [License](#license)
  - [Support](#support)

---

## Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| [**Python**](https://www.python.org/) | `3.13.5` | Modern Python runtime with performance improvements |
| [**LangChain**](https://github.com/langchain-ai/langchain) | `1.1.0` | **AI framework** for building LLM applications |
| [**LangGraph**](https://github.com/langchain-ai/langgraph) | `1.0.4` | **Agent runtime** with StateGraph, checkpointing, HITL |
| [**LangGraph CLI**](https://docs.langchain.com/langgraph/cli) | `0.4.7` | **Development server** for LangSmith Studio (local agent debugging) |
| [**LangSmith**](https://smith.langchain.com/) | `0.4.49` | **Observability & tracing** for debugging LLM apps |
| [**LangSmith Studio**](https://docs.langchain.com/langgraph/studio) | Web | **Visual debugger** for real-time agent execution (prompts, tools, states) |
| [**FastAPI**](https://fastapi.tiangolo.com/) | `0.123.0` | High-performance async web framework |
| [**Pydantic**](https://docs.pydantic.dev/) | `2.12.5` | **Runtime validation** & type-safe data models |
| [**OpenAI GPT-4**](https://platform.openai.com/) | `2.8.1` | Primary LLM provider (GPT-4o, GPT-4o-mini) |
| [**Anthropic Claude**](https://www.anthropic.com/claude) | `0.42.0` | Secondary LLM provider (Claude 3.5 Sonnet) |
| [**Qdrant**](https://qdrant.tech/) | `1.16.1` | **Vector database** for RAG & semantic search |
| [**PostgreSQL**](https://www.postgresql.org/) | `3.3.0` | Production checkpointer for agent state persistence |
| [**SQLite**](https://www.sqlite.org/) | `2.0.10` | Development checkpointer for local testing |
| [**Redis**](https://redis.io/) | `7.1.0` | In-memory cache for session & conversation memory |
| [**Docker**](https://www.docker.com/) | `27.5.1` | Containerization for deployment |
| [**uv**](https://github.com/astral-sh/uv) | `0.9.9` | **Ultra-fast package manager** (10-100x faster than pip) |
| [**Structlog**](https://www.structlog.org/) | `25.5.0` | Structured logging for production observability |
| [**MCP Protocol**](https://modelcontextprotocol.io/) | `1.0` | **Model Context Protocol** for tool integration |
| [**Weather MCP Server**](https://github.com/kumaran-is/mcp-weather-server) | Custom | Real-time weather data via NWS API |
| [**Hurricane Tracker MCP**](https://github.com/kumaran-is/hurricane-tracker-mcp) | Custom | Real-time hurricane data via NOAA/NHC APIs |

**Data Sources**:

*Real-Time Data (via MCP Servers)*:
- [**NOAA/NHC**](https://www.nhc.noaa.gov/) - National Hurricane Center real-time data
- [**NWS API**](https://www.weather.gov/documentation/services-web-api) - Weather alerts & forecasts
- [**IBTrACS**](https://www.ncei.noaa.gov/products/international-best-track-archive) - Historical hurricane tracks

*Knowledge Base (Level 2 RAG)*:
- **Mock Curated Data** (3 files) - Authoritative weather safety information
  - Saffir-Simpson hurricane scale
  - Evacuation zones (A-E)
  - Heat index guidelines
- [**Kaggle: Daily Temperature (Major Cities)**](https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities) - 2.9M daily temps (500 sampled)
- [**Kaggle: City Temperature (1980-2020)**](https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities) - 1,000 cities × 14,897 time periods (100 city profiles)

*Total Knowledge Base*: 603 documents, 632 chunks, embedded in Qdrant vector store

---

## Testing Tools & Interfaces

Quick reference for testing and debugging the Weather AI Agent:

| Tool | URL | Description | Use Case |
|------|-----|-------------|----------|
| **Swagger UI** | http://localhost:8000/docs | Interactive API documentation with live testing | Test REST endpoints, view request/response schemas, execute queries via browser |
| **ReDoc** | http://localhost:8000/redoc | Alternative API documentation (read-only) | Clean, searchable API reference for documentation review |
| **LangSmith Studio** | https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024 | Visual agent debugger with graph execution | Debug agent workflows, inspect tool calls, trace multi-step reasoning, test HITL approval |

**Test Guides by Level**:

| Level | Guide | Features Tested | Duration | Status |
|-------|-------|-----------------|----------|--------|
| **Level 0** | [Setup Verification](./verify_setup.py) | Python 3.13+, uv, Docker, API keys, MCP servers | ~5 min | ✅ Complete |
| **Level 1** | TBD | ReAct agent, HITL approval, MCP integration | ~10 min | ✅ Complete (no guide) |
| **Level 2** | [RAG + CoT + Hybrid Search](./docs/test-guide/LEVEL_2_TEST_GUIDE.md) | 603 docs, hybrid search (70/30), CoT reasoning, 8 tools | ~15 min | ✅ Complete |
| **Level 3a** | TBD | 2-layer memory (Conversation + Session), Redis | ~20 min | 🔜 Next |
| **Level 3b** | TBD | Advanced reasoning (ToT, GoT) | ~20 min | 📋 Planned |
| **Level 3c** | TBD | Full 7-layer memory, personalization | ~25 min | 📋 Planned |
| **Level 4a** | TBD | 3-agent system (Triage + Specialists) | ~20 min | 📋 Planned |
| **Level 4b** | TBD | 8-agent orchestration (parallel execution) | ~25 min | 📋 Planned |
| **Level 4c** | TBD | 15-agent production (reflection + critique) | ~30 min | 📋 Planned |
| **Level 5a** | TBD | Production RAG (<200ms retrieval, caching) | ~25 min | 📋 Planned |
| **Level 5b** | TBD | 6 critical guardrails (zero PII leaks) | ~20 min | 📋 Planned |
| **Level 5c** | TBD | Full production (99.9% uptime, observability) | ~30 min | 📋 Planned |
| **Level 6** | TBD | Self-evolving architecture (dual-loop) | ~35 min | 📋 Planned |

---

## Quick Start

### System Requirements

**Install these first** (one-time setup):

**1. Python 3.13.5+**
```bash
# Using pyenv (recommended)
curl https://pyenv.run | bash
pyenv install 3.13.5
pyenv global 3.13.5
```

**2. uv (ultra-fast package manager)**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

**3. Docker Desktop**

Download from https://www.docker.com/products/docker-desktop

### Prerequisites

**Note:** The following are required BEFORE running `make install-dev`:

- Python 3.13+ (installed above)
- **uv** (installed above)
- Docker Desktop (installed above)
- API Keys: OpenAI, LangSmith

### MCP Servers Required

This project uses **2 MCP servers** (Docker + HTTP Streamable transport) for weather data:

| MCP Server | Port | Transport | Repository |
|------------|------|-----------|------------|
| **Weather MCP Server** | 8080 | HTTP | [kumaran-is/mcp-weather-server](https://github.com/kumaran-is/mcp-weather-server/tree/develop) |
| **Hurricane Tracker MCP** | 8081 | HTTP | [kumaran-is/hurricane-tracker-mcp](https://github.com/kumaran-is/hurricane-tracker-mcp) |

**[MCP Servers Setup Guide](docs/setup/mcp-servers-setup.md)** - Step-by-step instructions to clone, build, and run both servers as Docker containers.

**Development Tools:**
- **[LangSmith Studio Setup Guide](docs/setup/langsmith-studio-setup.md)** - Official LangChain visual debugger for testing agents locally with real-time visualization of prompts, tool calls, and execution flow. Perfect for debugging all 3 MCP weather tools.
- **[Docker Usage Guide](docs/setup/docker-usage-guide.md)** - Complete guide for managing Docker containers with production and development modes, hot reload setup, and troubleshooting.

### Setup (15 minutes)

#### Option 1: Quick Start with Makefile (Recommended)

**1. Clone repository**
```bash
git clone https://github.com/kumaran-is/weather-agent.git
cd weather-agent
```

**2. Configure environment**
```bash
cp .env.template .env
```

Edit `.env` with your configuration. Below are all available settings organized by category:

### Required API Keys

| Variable | Required | Get From | Description |
|----------|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ **YES** | [OpenAI Platform](https://platform.openai.com/api-keys) | Used for GPT-4o-mini LLM and text-embedding-3-small |
| `LANGCHAIN_API_KEY` | ⚠️ **Recommended** | [LangSmith](https://smith.langchain.com/) | Enable tracing and debugging (optional for local dev) |

### MCP Server Configuration

**For Local Development** (default):

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `MCP_WEATHER_SERVER_URL` | `http://localhost:8080` | Weather MCP server endpoint |
| `MCP_WEATHER_SERVER_ENABLED` | `true` | Enable weather data retrieval |
| `MCP_HURRICANE_SERVER_URL` | `http://localhost:8081` | Hurricane Tracker MCP endpoint |
| `MCP_HURRICANE_SERVER_ENABLED` | `true` | Enable hurricane tracking data |

**For Docker Deployment**: URLs auto-configured via Docker Compose (use `http://weather-mcp:8080` and `http://hurricane-mcp:8081` - no manual .env changes needed)

**Setup Guide**: See [MCP Servers Setup](docs/setup/mcp-servers-setup.md) for complete installation instructions

### RAG & Vector Store Configuration (Level 2)

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector database endpoint (Level 2+) |
| `QDRANT_API_KEY` | `None` | Qdrant API key (optional, only for Qdrant Cloud in Level 5a+) |
| `HYBRID_SEARCH_VECTOR_WEIGHT` | `0.7` | Semantic search weight (70% semantic + 30% BM25) |
| `HYBRID_SEARCH_BM25_WEIGHT` | `0.3` | Keyword search weight (BM25 algorithm) |

### Agent Feature Flags (Level 2)

| Variable | Default | Description | Can Override at Runtime? |
|----------|---------|-------------|--------------------------|
| `ENABLE_RAG` | `true` | Enable RAG retrieval (8 tools: 3 MCP + 5 RAG) | ✅ Yes (via `?enable_rag=true/false`) |
| `ENABLE_COT` | `true` | Enable Chain-of-Thought reasoning (5-step framework) | ✅ Yes (via `?enable_cot=true/false`) |

**Example**: `/weather/query?enable_rag=true&enable_cot=true` (full Level 2 capabilities)

### LangSmith Observability (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing (set to `true` to activate) |
| `LANGCHAIN_PROJECT` | `weather-ai-agent-service` | LangSmith project name for organizing traces |
| `LANGCHAIN_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith API endpoint |

**Access LangSmith Studio**: http://localhost:8123 (after running `make docker-up-dev`)

### Optional/Advanced Configuration

| Variable | Default | Description | When to Use |
|----------|---------|-------------|-------------|
| `ANTHROPIC_API_KEY` | `None` | Anthropic API key for Claude models | Future levels (Level 3+) |
| `LOG_LEVEL` | `INFO` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `PRIMARY_MODEL` | `gpt-4o-mini` | Default LLM model | Override with `gpt-4o`, `claude-3-5-sonnet-20241022` |
| `MODEL_TEMPERATURE` | `0.7` | LLM creativity (0.0-2.0) | Lower = deterministic, Higher = creative |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis cache endpoint | Level 3a+ (memory systems) |

**Configuration Priority**: Runtime params > Environment variables (.env) > Code defaults

**3. Install all dependencies**

This creates `.venv/` and installs 121 packages (including dev tools):
```bash
make install-dev
```
Or alternatively:
```bash
make sync
```

**4. Run verification**

Run 9 automated checks:
```bash
make verify
```
Expected output: `ALL 9 CHECKS PASSED!`

**5. (Optional) Activate virtual environment manually**

Note: `uv run` and `make` commands work without activation.
```bash
source .venv/bin/activate
```

**6. Check installed versions**
```bash
make version
```

#### Option 2: Manual Setup (Without Makefile)

**1. Clone repository**
```bash
git clone https://github.com/kumaran-is/weather-agent.git
cd weather-agent
```

**2. Configure environment**
```bash
cp .env.template .env
```
Edit `.env` and add your API keys and MCP server paths (see Option 1 for details).

**3. Install all dependencies**

Creates `.venv/` automatically:
```bash
uv sync
```

**4. Run verification**

Run **8** automated checks:
```bash
uv run python verify_setup.py
```
Expected output: `ALL 9 CHECKS PASSED!`

**5. (Optional) Activate venv manually**
```bash
source .venv/bin/activate
```

**6. Check versions**
```bash
uv pip list | grep -E "langchain|langgraph|fastapi"
```

**Note**:
- `uv` automatically creates and manages `.venv/` virtual environment
- Commands like `uv run` and `make` work without manual venv activation
- Manual activation (`source .venv/bin/activate`) is only needed for direct Python commands

### Common Makefile Commands

**Development & Testing**

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install-dev` | Install all dependencies (including dev tools) - 121 packages |
| `make sync` | Sync dependencies from pyproject.toml |
| `make verify` | Run Level 0 verification (8 automated checks) |
| `make version` | Show installed package versions |
| `make clean` | Clean up cache and temporary files |
| `make test` | Run tests with pytest |
| `make lint` | Run ruff linter |
| `make format` | Format code with black and ruff |
| `make type-check` | Run mypy type checker |
| `make all-checks` | Run all quality checks (lint + type-check + test) |
| `make update` | Update all dependencies to latest compatible versions |
| `make lock` | Generate/update uv.lock file |

**Docker Commands (4 Services: weather-mcp:8080, hurricane-mcp:8081, qdrant:6333, weather-ai-api:8000)**

**Production Mode** (recommended for testing production builds):

| Command | Description |
|---------|-------------|
| `make docker-up` | Start all containers in PRODUCTION mode |
| `make docker-down` | Stop and remove all PRODUCTION containers |
| `make docker-restart` | Restart all PRODUCTION containers |
| `make docker-ps` | Show status of PRODUCTION containers |
| `make docker-logs` | Follow logs from PRODUCTION containers (Ctrl+C to exit) |
| `make docker-health` | Check health status of all services |
| `make docker-clean` | Stop PRODUCTION containers and remove volumes (⚠️ deletes all data) |

**Development Mode** (recommended for coding with hot reload):

| Command | Description |
|---------|-------------|
| `make docker-up-dev` | Start all containers in DEVELOPMENT mode (hot reload enabled) ⭐ |
| `make docker-down-dev` | Stop and remove all DEVELOPMENT containers |
| `make docker-restart-dev` | Restart all DEVELOPMENT containers |
| `make docker-ps-dev` | Show status of DEVELOPMENT containers |
| `make docker-logs-dev` | Follow logs from DEVELOPMENT containers (Ctrl+C to exit) |
| `make docker-clean-dev` | Stop DEVELOPMENT containers and remove volumes (⚠️ deletes all data) |

**Production vs Development:**

| Feature | Production (`make docker-up`) | Development (`make docker-up-dev`) |
|---------|-------------------------------|-------------------------------------|
| **Hot Reload** | ❌ No (rebuild required) | ✅ Yes (code changes reflect immediately) |
| **Source Mounting** | ❌ No (baked into image) | ✅ Yes (`./backend` mounted as volume) |
| **Log Level** | `info` | `debug` (verbose) |
| **LangSmith Tracing** | Optional | ✅ Enabled by default |
| **Best For** | Testing production builds | Active development |

**Typical Workflows:**

**Development Workflow (Recommended):**
```bash
# Start in dev mode with hot reload
make docker-up-dev

# Edit code in ./backend (changes reflect immediately)

# Check status
make docker-ps-dev

# View logs
make docker-logs-dev

# Stop when done
make docker-down-dev
```

**Production Workflow:**
```bash
# Start all services
make docker-up

# Check status
make docker-ps
make docker-health

# Debug issues
make docker-logs

# Stop all services
make docker-down
```

**RAG Commands (Level 2: Qdrant Vector Database)**

| Command | Description |
|---------|-------------|
| `make rag-validate` | Validate Kaggle datasets before loading |
| `make rag-load` | Load mock + Kaggle data into Qdrant (632 chunks) |
| `make rag-test` | Test retrieval from Qdrant (mock + Kaggle data) |
| `make rag-load-mock-only` | Load only mock data (skip Kaggle datasets) |

**Datasets Guide**: See [RAG Datasets Guide](./docs/setup/rag-datasets-guide.md) for complete details on mock data sources, Kaggle datasets (2.9M+ records), download instructions, and dataset statistics.

**Typical RAG Workflow:**
```bash
# 1. Validate datasets
make rag-validate

# 2. Load knowledge base (one-time setup)
make rag-load

# 3. Test retrieval (verify 603 documents loaded)
make rag-test
```

**⚠️ Important: Qdrant Data Persistence**

Your embedded documents (603 chunks) are stored in a **Docker named volume** and persist across container restarts:

**Data is PRESERVED when:**
- ✅ `make docker-restart` / `make docker-restart-dev` - Safe to restart
- ✅ `make docker-down` / `make docker-down-dev` - Safe to stop containers
- ✅ `make docker-up` / `make docker-up-dev` - Data reloads automatically

**Data is DELETED when:**
- ❌ `make docker-clean` / `make docker-clean-dev` - Removes volumes (5-second warning)
- ❌ `docker-compose down -v` - The `-v` flag deletes volumes

**Verify data persists after restart:**
```bash
# Check document count
make rag-test

# Restart Qdrant
make docker-restart

# Check again - same 603 documents!
make rag-test
```

**When to run `make rag-load` again:**
1. After `make docker-clean` (volumes deleted)
2. When adding new documents to `backend/data/raw/`
3. When switching between production and dev (different volumes)

**Note:** Production and development use separate volumes (`weather-ai-qdrant-data` vs `weather-ai-qdrant-data-dev`), so you may need to load data separately for each environment.

### Project Structure
```
weather-agent/
├── backend/                     # Backend application (Level 1+2 Complete)
│   ├── config/                  # Configuration - SINGLE LOCATION ⚠️ (ALL config files here ONLY)
│   │   ├── __init__.py          # Package initialization
│   │   ├── settings.py          # Centralized app settings (Qdrant, OpenAI, MCP, env vars)
│   │   └── llm_config.py        # LLM configuration (use cases: emergency/forecast/conversational)
│   ├── data/                    # RAG data (Level 2)
│   │   └── raw/
│   │       ├── mock/            # Curated weather docs (3 files)
│   │       │   ├── hurricanes/  # Saffir-Simpson, evacuation zones
│   │       │   └── weather_terminology/ # Heat index
│   │       └── kaggle/          # Kaggle datasets (600 docs)
│   │           ├── daily_temperature_major_cities.csv (500 sampled)
│   │           └── city_temperature_1980_2020.csv (100 city profiles)
│   └── src/                     # Source code
│       ├── agents/              # LangChain agents (Level 1+2)
│       │   ├── state.py         # Agent state TypedDict
│       │   ├── prompts.py       # System prompts (simple + CoT)
│       │   └── weather_agent.py # Unified ReAct agent (enable_rag + enable_cot)
│       ├── api/                 # FastAPI service (Level 1+2)
│       │   └── main.py          # FastAPI app + 4 endpoints
│       ├── models/              # Pydantic models - SINGLE SOURCE OF TRUTH
│       │   ├── __init__.py      # Central exports + strict architectural rules
│       │   ├── weather.py       # Weather domain (WeatherQuery, WeatherResponse)
│       │   ├── hurricane.py     # Hurricane domain (4 HITL models)
│       │   └── health.py        # Health check domain (HealthCheckResponse)
│       │   # Future: memory.py (Level 3), agents.py (Level 4), observability.py (Level 5)
│       ├── hitl/                # Human-in-the-Loop (Level 1)
│       │   └── approval_node.py # HITL approval nodes (Category 3+ hurricanes)
│       ├── mcp/                 # MCP client integration (Level 1)
│       │   └── weather_client.py # Async HTTP MCP client (header-based sessions)
│       ├── rag/                 # RAG system (Level 2 Complete)
│       │   ├── embeddings.py    # OpenAI embeddings (text-embedding-3-small)
│       │   ├── vector_store.py  # QdrantVectorStore setup (Cosine distance)
│       │   ├── retriever.py     # LangChain retriever interface
│       │   ├── hybrid_search.py # RRF algorithm (70% semantic + 30% BM25)
│       │   ├── build_knowledge_base.py # Knowledge base builder
│       │   └── loaders/         # Document loaders
│       │       ├── mock_loader.py         # Mock weather docs loader
│       │       ├── kaggle_loader.py       # Kaggle CSV loader
│       │       ├── csv_to_narrative.py    # CSV-to-text converter
│       │       └── validate_kaggle_datasets.py # 7-check validation
│       ├── tools/               # LangChain tools (Level 1+2)
│       │   ├── weather_tools.py # 3 MCP tool wrappers (@tool decorator)
│       │   └── rag_tools.py     # 4 RAG-enhanced tools (analyze_trends, identify_patterns, compare_conditions, retrieve_weather_knowledge_tool)
│       └── workflows/           # LangGraph workflows (Level 1)
│           └── weather_graph.py # HITL StateGraph workflow
├── docs/                        # Documentation
│   ├── setup/                   # Setup guides
│   │   ├── mcp-servers-setup.md        # MCP servers deployment
│   │   ├── langsmith-studio-setup.md   # Studio integration (434 lines)
│   │   ├── docker-usage-guide.md       # Docker orchestration (649 lines)
│   │   └── rag-datasets-guide.md       # RAG datasets guide (211 lines)
│   ├── test-guide/              # Testing guides
│   │   └── LEVEL_2_TEST_GUIDE.md # ✅ RAG + CoT testing (~15 minutes)
├── tests/                       # Test suite (Level 1+2)
│   ├── conftest.py              # Pytest fixtures
│   ├── test_mcp_client.py       # MCP client tests
│   ├── test_weather_tool.py     # Weather tool tests
│   ├── test_rag_tools.py        # RAG tools tests (Level 2)
│   ├── test_react_agent.py      # ReAct agent tests
│   ├── test_hurricane_hitl.py   # HITL approval tests
│   ├── test_workflow.py         # LangGraph workflow tests
│   ├── test_api.py              # FastAPI endpoint tests
│   └── test_integration.py      # End-to-end integration tests
├── Dockerfile                   # Multi-stage production + development build
├── docker-compose.yml           # Production orchestration (4 services: API, 2 MCP, Qdrant)
├── docker-compose.dev.yml       # Development orchestration (hot reload enabled)
├── langgraph.json               # LangSmith Studio configuration (2 graphs)
├── LANGSMITH_STUDIO_TESTING.md  # Studio testing guide (agent configurations)
├── pyproject.toml               # Project config (v0.3.0, source of truth)
├── uv.lock                      # uv lockfile (reproducible builds)
├── Makefile                     # Development commands (22 tasks: setup, RAG, Docker)
├── verify_setup.py              # Level 0 verification (8 checks)
├── CHANGELOG.md                 # Version history (v0.3.0)
└── README.md                    # This file
```

**Note:** We use both `pyproject.toml` (defines what dependencies you want) and `uv.lock` (locks exact versions) to ensure reproducible builds across all environments.

### Progressive Learning Path
- **Level 0** (v0.1.0): Setup ✅ COMPLETE
- **Level 1** (v0.2.0-v0.2.1): ReAct + HITL + LangSmith Studio ✅ COMPLETE
- **Level 2** (v0.3.0): ReAct +CoT + RAG + hybrid search (Semantic +Key Word) ✅ COMPLETE
- **Level 3a** (v0.4.0): 2-Layer Memory (Conversation + Session) 🔜 NEXT
- **Level 3b** (v0.5.0): Advanced Reasoning (Tree-of-Thoughts, Graph-of-Thoughts)
- **Level 3c** (v0.6.0): Full 7-Layer Memory + Personalization
- **Level 4a-c** (v0.7.0-0.9.0): Multi-Agent (3 → 15 agents)
- **Level 5a-c** (v0.10.0-1.0.0): Production (RAG optimization, guardrails, observability)
- **Level 6** (v1.1.0): Self-Evolving (dual-loop architecture)

## API Endpoints Reference

All REST endpoints with descriptions and use cases:

| Method | Endpoint | Description | Features |
|--------|----------|-------------|----------|
| **POST** | `/weather/query` | General weather queries | Supports `enable_rag` and `enable_cot` query parameters for runtime configuration |
| **POST** | `/weather/hurricane/alert` | Create hurricane alert | HITL approval workflow: Cat 1-2 auto-approved, Cat 3-5 require human approval |
| **POST** | `/weather/hurricane/approve/{thread_id}` | Approve or reject pending alert | Human review endpoint for Category 3+ hurricanes, accepts `approved: true/false` |
| **GET** | `/health` | Service health check | Returns status, implementation level, and timestamp |

**Access Points**:
- **Swagger UI**: http://localhost:8000/docs (interactive testing)
- **ReDoc**: http://localhost:8000/redoc (documentation)
- **Base URL**: http://localhost:8000

**Example Usage**:
```bash
# Basic weather query (Level 1 - 3 MCP tools)
curl -X POST "http://localhost:8000/weather/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Miami?"}'

# Query with RAG enabled (Level 2 - 8 tools with knowledge base)
curl -X POST "http://localhost:8000/weather/query?enable_rag=true" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a Category 5 hurricane?"}'

# Query with CoT reasoning (5-step framework)
curl -X POST "http://localhost:8000/weather/query?enable_rag=true&enable_cot=true" \
  -H "Content-Type: application/json" \
  -d '{"query": "Should I evacuate for a Category 4 hurricane in Zone B?"}'

# Health check
curl http://localhost:8000/health
```

**LangSmith Studio Graphs** (visual debugging):
- `weather_agent`: Unified agent with configurable RAG + CoT (see [LANGSMITH_STUDIO_TESTING.md](./LANGSMITH_STUDIO_TESTING.md))
- `weather_hitl_workflow`: Hurricane alert HITL approval workflow

---

## Testing Level 2

Comprehensive test guide for validating all Level 2 features (RAG + CoT + Hybrid Search).

**[Level 2 Test Guide](./docs/test-guide/LEVEL_2_TEST_GUIDE.md)**

**What you'll test:**
- ✅ 6 Swagger UI test cases (basic queries, RAG, CoT, hybrid search)
- ✅ 6 LangGraph Studio configurations (agent workflows, multi-turn conversations)
- ✅ Quick validation checklist and troubleshooting

**Prerequisites:** Docker services running

```bash
# Quick start
make docker-up-dev
# Then follow: docs/test-guide/LEVEL_2_TEST_GUIDE.md
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/kumaran-is/weather-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kumaran-is/weather-agent/discussions)
- **Documentation**: See `docs/` directory

**Built for teams exploring agentic AI systems**
