# Weather AI Agent Service

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kumaran-is/weather-agent)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.0+](https://img.shields.io/badge/langchain-1.0+-green.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph 1.0+](https://img.shields.io/badge/langgraph-1.0+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![MCP Protocol](https://img.shields.io/badge/MCP%20Protocol-Dual%20Servers-orange)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.3.1-blue.svg)](./CHANGELOG.md)

**Production-grade AI agent for weather forecast intelligence:** Built with LangChain 1.0, LangGraph 1.0, FastAPI and OpenAI. Features 15-agent multi-agent orchestration, auto-routing (intent-based query classification), 7-layer memory architecture, advanced reasoning (Tree/Graph-of-Thought), emotional intelligence, multi-layer caching (L1+L2+L3), real-time weather data via dual MCP servers, RAG-enhanced knowledge base, Chain-of-Thought reasoning, Human-in-the-Loop (HITL) approval workflows, comprehensive golden dataset evaluation with ZERO safety violations, guardrails and observability.

**From zero to production:** Progressive implementation showcasing enterprise AI patterns including multi-agent orchestration (3→8→15 agents), auto-routing architecture, 7-layer memory systems (99.7% storage reduction), context window optimization (50-60% token reduction for <4K tokens/query), multi-layer caching (L1+L2+L3 for 60-75% cost reduction), full observability stack (Prometheus, Grafana, Loki), adversarial testing, LLM-as-Judge, Advanced Evaluation(Ragas, DeepEval, Context Optimization, Property/Snapshot Testing, Promptfoo, OpenAI Evals) and  Self-Improvement(TruLens, LangChain Benchmark, Auto-Prompt), constitutional AI (12-layer safety guardrails), and **[planned]** self-evolving agentic dual-loop architecture for continuous improvement.

**How It Works:** Progressive implementation showcasing enterprise AI patterns including multi-agent orchestration (3→8→15 agents), auto-routing architecture, 7-layer memory systems (99.7% storage reduction), multi-layer caching (L1+L2+L3 for 60-75% cost reduction), Comprehensive golden dataset evaluation and full observability stack (Prometheus, Grafana, Loki

**Current Stage**: ✅ **Level 6 Complete**  — Advanced Evaluation(Ragas, DeepEval, Context Optimization, Property/Snapshot Testing, Promptfoo, OpenAI Evals) and  Self-Improvement(TruLens, LangChain Benchmark, Auto-Prompt)

[Read the Medium Blog Post Series](https://medium.com/@yourusername)

---

## Table of Contents

- [Weather AI Agent Service](#weather-ai-agent-service)
  - [Table of Contents](#table-of-contents)
  - [Technology Stack](#technology-stack)
  - [Testing Tools \& Interfaces](#testing-tools--interfaces)
  - [| **Level 10** | Context Engineering \& Optimization|  | 📋 Planned |](#-level-10--context-engineering--optimization----planned-)
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
  - [Testing (Progressive Build Approach)](#testing-progressive-build-approach)
  - [Contributing](#contributing)
  - [License](#license)
  - [Support](#support)

---

## Technology Stack

**Core Framework**:

| Technology | Version | Purpose |
|------------|---------|---------|
| [**Python**](https://www.python.org/) | `3.13+` | Modern Python runtime |
| [**LangChain**](https://github.com/langchain-ai/langchain) | `1.0+` | AI framework for LLM applications |
| [**LangGraph**](https://github.com/langchain-ai/langgraph) | `1.0+` | Agent runtime with StateGraph, checkpointing, HITL |
| [**FastAPI**](https://fastapi.tiangolo.com/) | `0.115+` | High-performance async web framework |
| [**Pydantic**](https://docs.pydantic.dev/) | `2.0+` | Runtime validation & type-safe data models |

**AI/ML & LLM**:

| Technology | Purpose |
|------------|---------|
| [**OpenAI GPT-4**](https://platform.openai.com/) | Primary LLM (GPT-4o, GPT-4o-mini) |
| [**Anthropic Claude**](https://www.anthropic.com/claude) | Secondary LLM (Claude 3.5 Sonnet) + L3 prompt caching |
| [**MCP Protocol**](https://modelcontextprotocol.io/) | Model Context Protocol for tool integration |

**Databases & Storage**:

| Technology | Purpose |
|------------|---------|
| [**Qdrant**](https://qdrant.tech/) | Vector database for RAG & semantic search |
| [**Redis**](https://redis.io/) | L2 distributed cache + session memory |
| [**Neo4j**](https://neo4j.com/) | Graph database for episodic/semantic memory |
| [**PostgreSQL**](https://www.postgresql.org/) | Production checkpointer + procedural memory |

**Observability**:

| Technology | Purpose |
|------------|---------|
| [**Prometheus**](https://prometheus.io/) | Metrics collection |
| [**Grafana**](https://grafana.com/) | Dashboards & visualization |
| [**Loki**](https://grafana.com/oss/loki/) | Log aggregation |
| [**LangSmith**](https://smith.langchain.com/) | LLM tracing & debugging |

**Data Sources** (via MCP Servers):
- [**NOAA/NHC**](https://www.nhc.noaa.gov/) - Hurricane tracking data
- [**NWS API**](https://www.weather.gov/documentation/services-web-api) - Weather alerts & forecasts

**Knowledge Base (Level 2 RAG)**:
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

| Tool | URL | Description |
|------|-----|-------------|
| **Swagger UI** | http://localhost:8000/docs | Interactive API testing with example data dropdowns |
| **ReDoc** | http://localhost:8000/redoc | Read-only API documentation |
| **LangSmith Dashboard** | https://smith.langchain.com/ | LLM tracing, agent debugging, cost analysis |
| **LangSmith Tracing** | https://smith.langchain.com/o/{org_id}/projects/p/{project_id} | Detailed agent trace inspection & debugging |
| **LangSmith Evaluation** | https://smith.langchain.com/o/{org_id}/datasets | Golden dataset management & batch evaluation (105 test cases) |
| **LangGraph Studio** | https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024 | Visual agent debugger (4 graphs: weather_agent, weather_hitl_workflow, multi_agent_workflow, level4b_workflow) |
| **Grafana Dashboards** | http://localhost:3001 | Observability dashboards (login: admin/weatherai2025) |
| **Prometheus UI** | http://localhost:9090 | Metrics explorer and alerting |

**Service Access URLs**

| Service | URL | Description |
|---------|-----|-------------|
| **Weather AI API** | http://localhost:8000 | Main API service |
| **API Health** | http://localhost:8000/health | Service health check |
| **Prometheus Metrics** | http://localhost:8000/metrics | Prometheus-format metrics endpoint |
| **Weather MCP** | http://localhost:8080/health | Weather data MCP server |
| **Hurricane MCP** | http://localhost:8081/health | Hurricane tracking MCP server |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Vector database UI (603 documents) |
| **Neo4j Browser** | http://localhost:7474 | Graph database UI (neo4j/weatherai2025) |

**Observability Stack (Level 5c)**

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3001 | admin / weatherai2025 |
| **Prometheus** | http://localhost:9090 | None |
| **Loki** | http://localhost:3100 | API only |

**Test Guides by Level**:

| Level | Guide | Key Features | Status |
|-------|-------|--------------|--------|
| **Level 0** | [Setup Verification](./verify_setup.py) | Python 3.13+, uv, Docker, API keys | ✅ Complete |
| **Level 1** | [Test Suite](./tests/test_react_agent.py) | ReAct agent, HITL approval, MCP integration | ✅ Complete |
| **Level 2** | [RAG + CoT Guide](./docs/test-guide/LEVEL_2_TEST_GUIDE.md) | 603 docs, hybrid search, CoT reasoning | ✅ Complete |
| **Level 3** | [Memory + Reasoning Guide](./docs/test-guide/LEVEL_3_TEST_GUIDE.md) | 7-layer memory, ToT/GoT, emotional intelligence | ✅ Complete |
| **Level 4** | [Multi-Agent Guide](./docs/test-guide/LEVEL_4_TEST_GUIDE.md) | 15-agent orchestration, auto-routing | ✅ Complete |
| **Level 5** | [Evaluation + Guardrail Guide](./docs/test-guide/LEVEL_5_TEST_GUIDE.md) | Caching (L1/L2/L3), Prometheus, Swagger UI | ✅ Complete |
| **Level 6** | [Advanced Evaluation + Self-Improvement](./docs/test-guide/LEVEL_6_TEST_GUIDE.md) |  | ✅ Complete |
| **Level 7** | MCP Tool Dynamic Registry = Context Optimization |  | 📋 Planned |
| **Level 8** | Self Evolving Agenitc Dual-Loop Architecture |  | 📋 Planned |
| **Level 9** | Advanced HITL |  | 📋 Planned |
| **Level 10** | Context Engineering & Optimization|  | 📋 Planned |
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
| `OPENAI_API_KEY` | **YES** | [OpenAI Platform](https://platform.openai.com/api-keys) | Used for GPT-4o-mini LLM and text-embedding-3-small |
| `ANTHROPIC_API_KEY` | **Nice-to-have** | [Anthropic Platform](https://console.anthropic.com/settings/keys) | Used for claude-3-5-sonnet LLM as Judge |
| `LANGCHAIN_API_KEY` | **YES** | [LangSmith](https://smith.langchain.com/) | Enable tracing and debugging (optional for local dev) |

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
| `ENABLE_RAG` | `true` | Enable RAG retrieval (8 tools: 3 MCP + 5 RAG) | Yes (via `?enable_rag=true/false`) |
| `ENABLE_COT` | `true` | Enable Chain-of-Thought reasoning (5-step framework) | Yes (via `?enable_cot=true/false`) |

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
| `POSTGRES_URL` | `postgresql://weather_ai:weatherai2025@localhost:5432/weather_ai` | PostgreSQL database endpoint | Level 3c (Procedural & Reflective Memory) |
| `POSTGRES_MAX_CONNECTIONS` | `20` | PostgreSQL connection pool size | Adjust based on load (5-100) |

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

**Development & Testing** (Progressive Build Approach)

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install-dev` | Install all dependencies (including dev tools) - 121 packages |
| `make sync` | Sync dependencies from pyproject.toml |
| `make verify` | Run Level 0 verification (8 automated checks) |
| `make version` | Show installed package versions |
| `make clean` | Clean up cache and temporary files |
| `make test` | Run all tests with pytest (quick, no restart) |
| `make test-dev` | **Unified testing** - Restart dev environment + run complete test suite + compliance check (progressive: currently tests Level 1+2+3 complete) |
| `make lint` | Run ruff linter |
| `make format` | Format code with black and ruff |
| `make type-check` | Run mypy type checker |
| `make compliance-check` | Verify LangChain v1.x compliance (100% required) |
| `make all-checks` | Run all quality checks (lint + type-check + test) |
| `make update` | Update all dependencies to latest compatible versions |
| `make lock` | Generate/update uv.lock file |

**Progressive Build Philosophy**: `make test-dev` automatically tests all implemented features for the current level without requiring Makefile updates as you progress through levels.

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

**Development Workflow (Recommended - Progressive Build):**
```bash
# 1. Start in dev mode with hot reload
make docker-up-dev

# 2. Load RAG knowledge base (one-time setup)
make rag-load

# 3. Run unified test suite (tests ALL current features: Level 1+2+3)
make test-dev

# 4. Edit code in ./backend (changes reflect immediately)

# 5. View logs
make docker-logs-dev

# 6. Stop when done
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

**Memory Commands (Level 3a+: Redis + Neo4j)**

| Command | Description |
|---------|-------------|
| `make docker-ps-dev` | View all running DEVELOPMENT containers |
| `make docker-health` | Check health of all services (auto-detects prod/dev) |
| `make memory-test` | Test memory system (Redis + Neo4j connectivity, auto-detects dev containers) |
| `make memory-redis-cli` | Open Redis CLI (interactive shell, auto-detects dev container) |
| `make memory-neo4j-browser` | Open Neo4j Browser in default browser |
| `make docker-down-dev` | Stop all DEVELOPMENT containers |

**Typical Memory Workflow:**
```bash
# 1. Start development containers
make docker-up-dev

# 2. Check all services are running
make docker-ps-dev

# 3. Verify health
make docker-health

# 4. Test memory connectivity
make memory-test

# 5. Explore Redis data (interactive CLI)
make memory-redis-cli

# 6. Open Neo4j Browser (graph visualization)
make memory-neo4j-browser

# 7. Stop when done
make docker-down-dev
```

**Observability Commands (Level 5c: Prometheus + Grafana + Loki)**

| Command | Description |
|---------|-------------|
| `make observability-status` | Check Prometheus/Grafana/Loki status |
| `make grafana-open` | Open Grafana dashboard (http://localhost:3001) |
| `make prometheus-open` | Open Prometheus UI (http://localhost:9090) |
| `make prometheus-reload` | Reload Prometheus configuration (hot reload) |
| `make loki-logs` | Query recent logs from Loki |
| `make observability-logs` | Follow logs from observability stack |

**Typical Observability Workflow:**
```bash
# 1. Start dev containers (includes observability stack)
make docker-up-dev

# 2. Check observability status
make observability-status

# 3. Open Grafana dashboards
make grafana-open
# Login: admin / weatherai2025

# 4. View Prometheus metrics
make prometheus-open

# 5. Query logs via Loki (in Grafana)
make loki-logs
```

**Evaluation Commands (Level 5b: Golden Dataset Testing & Quality Gates)**

All 8 evaluation commands validated and working correctly:

| # | Command | Description | Duration | Output File | Status |
|---|---------|-------------|----------|-------------|--------|
| 1 | `make eval-upload-dataset` | Upload golden dataset to LangSmith (105 test cases) | ~5 seconds | N/A (uploads to LangSmith) | ✅ Validated |
| 2 | `make eval-quick` | Quick smoke test (10 random cases) | ~1-2 minutes | `evaluation_quick.json` | ✅ Validated |
| 3 | `make eval-category CATEGORY=simple` | Test simple weather queries (40 cases) | ~5-6 minutes | `evaluation_simple.json` | ✅ Validated |
| 4 | `make eval-category CATEGORY=hurricane` | Test hurricane safety-critical queries (20 cases) | ~4-5 minutes | `evaluation_hurricane.json` | ✅ Validated |
| 5 | `make eval-category CATEGORY=complex` | Test complex multi-location queries (30 cases) | ~6-7 minutes | `evaluation_complex.json` | ✅ Validated |
| 6 | `make eval-category CATEGORY=edge` | Test edge cases and error handling (15 cases) | ~2-3 minutes | `evaluation_edge.json` | ✅ Validated |
| 7 | `make eval-check-gates` | Check quality gates on evaluation results | ~2 seconds | N/A (reads `evaluation_results.json`) | ✅ Validated |
| 8 | `make eval-full` | **Full pipeline**: Upload → Run all 105 cases → Check gates | ~17-20 minutes | `evaluation_results.json` | ✅ Validated |

**Quality Gates** (5 gates, all must pass for deployment):

| Gate | Threshold | Purpose |
|------|-----------|---------|
| **Pass Rate** | ≥85% | Overall test success rate |
| **Effectiveness** | ≥85% | Answer quality and correctness |
| **Efficiency** | ≥80% | Tool usage optimization |
| **Robustness** | ≥80% | Error handling and edge cases |
| **Safety** | 0 violations | Life-safety critical errors (ZERO tolerance) |

**Typical Evaluation Workflow:**
```bash
# Quick smoke test (recommended first)
make eval-quick

# Test specific category
make eval-category CATEGORY=hurricane

# Full evaluation pipeline (all 105 cases)
make eval-full

# View results in LangSmith
# https://smith.langchain.com/datasets/14c92fff-0c08-49a3-976c-9084544327cb
```

**Quality Gates Guide**: See [docs/knowledgebase/quality-gates-guide.md](docs/knowledgebase/quality-gates-guide.md) for comprehensive details on what quality gates are, how they work, and what to do when they fail.

### Project Structure

```
weather-agent/
├── backend/                     # Backend application (Level 5c Complete)
│   ├── config/                  # Configuration (SINGLE LOCATION)
│   │   ├── settings.py          # Centralized app settings (50+ env vars)
│   │   ├── llm_config.py        # LLM use case configuration
│   │   ├── memory_config.py     # Memory system configuration
│   │   └── cache_config.py      # Cache system configuration
│   ├── data/raw/                # RAG knowledge base (603 documents)
│   ├── migrations/              # Database migration scripts
│   ├── src/                     # Source code
│   │   ├── agents/              # 15 specialized agents (weather, triage, hurricane, etc.)
│   │   ├── api/main.py          # FastAPI app (9 endpoints + Prometheus metrics)
│   │   ├── cache/               # Multi-layer caching (L1/L2/L3)
│   │   ├── evaluation/          # Evaluation framework
│   │   ├── guardrails/          # 12-layer safety guardrails
│   │   ├── hitl/                # Human-in-the-Loop approval nodes
│   │   ├── mcp/                 # MCP client integration
│   │   ├── memory/              # 7-layer memory system
│   │   ├── models/              # Pydantic models (weather, hurricane, health, etc.)
│   │   ├── orchestration/       # Multi-agent workflow orchestration
│   │   ├── rag/                 # RAG pipeline (embeddings, hybrid search)
│   │   ├── reasoning/           # ToT/GoT advanced reasoning
│   │   ├── routing/             # Auto-routing system
│   │   ├── services/            # Shared business logic services
│   │   ├── tools/               # LangChain tools (MCP + RAG)
│   │   ├── utils/               # Utility functions and helpers
│   │   └── workflows/           # LangGraph workflows
│   └── tests/                   # Backend-specific tests
├── docs/                        # Documentation
│   ├── setup/                   # Setup and configuration guides
│   ├── test-guide/              # Test guides by level
├── observability/               # Observability stack configuration
│   ├── grafana/                 # Grafana dashboards and datasources
│   ├── loki/                    # Loki log aggregation config
│   └── prometheus/              # Prometheus metrics config
├── scripts/                     # Evaluation and automation scripts
│   ├── check_quality_gates.py   # CI/CD quality gate validation
│   ├── run_batch_evaluation.py  # LangSmith batch evaluation runner
│   └── upload_golden_dataset.py # Golden dataset upload to LangSmith
├── tests/                       # Test suite (22 test files)
├── .env.template                # Environment variables template
├── CHANGELOG.md                 # Version history and release notes
├── Dockerfile                   # Multi-stage production container
├── docker-compose.yml           # Production orchestration (9 services)
├── docker-compose.dev.yml       # Development orchestration (hot reload)
├── langgraph.json               # LangGraph Studio (4 graphs)
├── LICENSE                      # MIT License
├── Makefile                     # Development and deployment commands
├── pyproject.toml               # Project config (v1.3.1)
└── README.md                    # This file
```

**Key Directories**:
- **backend/src/agents/**: 15 specialized agents for weather intelligence (supervisor, triage, hurricane, emergency, forecaster, climate, historical, research, personalization, meta-prompt, critique, debate, reflection, self-healing, alert manager)
  - **routing/**: Auto-routing system v0.6.0 (~500 lines, <1ms classification)
  - **agents/**:  15 specialized agents for weather intelligence (supervisor, triage, hurricane, emergency, forecaster, climate, historical, research, personalization, meta-prompt, critique, debate, reflection, self-healing, alert manager)
  - **memory/**: 7-layer memory (conversation, session, episodic, semantic, procedural, emotional, reflective)
  - **reasoning/**: Tree/Graph-of-Thought (27 parallel paths, network reasoning)
  - **orchestration/**: Multi-agent workflows (supervisor, parallel execution, auto-routing)
  - **evaluation/**: 4-pillar evaluation framework (effectiveness, efficiency, robustness, safety)
  - **cache/**: Multi-layer caching (L1+L2+L3 utilities, 60-75% cost reduction potential)
  - **rag/**: Complete RAG pipeline (embeddings, vector store, hybrid search)
  - **tools/**: 7 tools total (3 MCP + 4 RAG-enhanced)
- **backend/config/**: 3 config classes (Settings, MemoryConfig, CacheConfig) = 80+ env vars
- **backend/data/**: 603 documents, 632 chunks embedded in Qdrant (3 mock + 600 Kaggle)
- **docs/setup/**:7 comprehensive guides (Dependency management, MCP, Studio, Docker, RAG datasets, Neo4j Desktop, Redis Insight)
- **tests/**: 20 test modules covering MCP, RAG, HITL, API, memory, reasoning, routing, cache, Level 4c agents
- **langgraph.json**: 2 graphs configured (weather_agent, weather_hitl_workflow)
- **Makefile**: Commands (setup, RAG, Docker, Evaluating, testing)
  
**Note:** We use both `pyproject.toml` (defines what dependencies you want) and `uv.lock` (locks exact versions) to ensure reproducible builds across all environments.

### Progressive Learning Path
- **Level 0** (v0.1.0): Setup ✅ COMPLETE
- **Level 1** (v0.2.0-v0.2.1): ReAct + HITL + LangSmith Studio ✅ COMPLETE
- **Level 2** (v0.3.0): ReAct + CoT + RAG + Hybrid Search (Semantic + Keyword) ✅ COMPLETE
- **Level 3a** (v0.4.0): 2-Layer Memory (Conversation + Session) ✅ COMPLETE
- **Level 3b** (v0.5.0): Advanced Reasoning (Tree-of-Thoughts + Graph-of-Thoughts) ✅ COMPLETE
- **Level 3c** (v0.6.0): Full 7-Layer Memory + Emotional Intelligence + Personalization ✅ COMPLETE
- **Level 4a-c** (v0.7.0): Multi-Agent Orchestration (3 → 15 agents) ✅ COMPLETE
- **Level 5a-c** (v0.10.0-1.0.0): Production (RAG optimization, guardrails, observability) ✅ COMPLETE
- **Level 6** (v1.3.0): Self-Evolving AI Platform ✅ COMPLETE
  - Level 6a: Context Optimization + Ragas/DeepEval Evaluation ✅ COMPLETE
  - Level 6b: TruLens + AgentBench + Auto-Prompt Engineering ✅ COMPLETE
  - Level 6c: Constitutional AI + Production Testing Infrastructure ✅ COMPLETE
- **Level 7** : MCP Tool Dynamic Registry + Context Optimization 🔜 NEXT
- **Level 8** : Self Evolving Agenitc Architecture 🔜 NEXT
- **Level 9** : Advanced HITL 🔜 NEXT

**Current Status**: Level 6 - Advanced Evaluation(Ragas, DeepEval, Context Optimization, Property/Snapshot Testing, Promptfoo, OpenAI Evals) and  Self-Improvement(TruLens, LangChain Benchmark, Auto-Prompt) COMPLETE ✅ | Ready for Level 7 MCP Tool Dynamic Registry | Test with `make test-dev`

## API Endpoints Reference

All REST endpoints (9 total):

| Method | Endpoint | Description |
|--------|----------|-------------|
| **POST** | `/weather/query` | Weather queries with AUTO-ROUTING (v0.6.0), multi-agent orchestration, RAG, CoT, memory, 3-layer caching |
| **POST** | `/weather/hurricane/alert` | Create hurricane alert (Cat 1-2 auto-approved, Cat 3+ requires HITL approval) |
| **POST** | `/weather/hurricane/approve/{thread_id}` | Approve or reject pending hurricane alerts (Cat 3+ only) |
| **GET** | `/health` | Service health check with 9 monitored services (returns level: L4+L5c) |
| **GET** | `/cache/stats` | Cache statistics for all 3 layers (L1 in-process, L2 Redis, L3 Anthropic) |
| **POST** | `/cache/clear` | Clear L1 (in-process) and L2 (Redis) cache layers |
| **POST** | `/cache/invalidate` | Invalidate specific cache entry at L1 and L2 |
| **GET** | `/cache/config` | View current cache configuration (L1, L2, L3 settings) |
| **GET** | `/metrics` | Prometheus metrics endpoint (text format for scraping) |

**Access Points**:
- **Swagger UI**: http://localhost:8000/docs (interactive testing)
- **ReDoc**: http://localhost:8000/redoc (documentation)
- **Base URL**: http://localhost:8000

**Example Usage**:
```bash
# Simple weather query
curl -X POST "http://localhost:8000/weather/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Miami?", "user_id": "test_001"}'

# Hurricane query (triggers multi-agent routing)
curl -X POST "http://localhost:8000/weather/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the status of Hurricane Milton?", "user_id": "test_002"}'

# Emergency evacuation query (triggers l4b orchestration)
curl -X POST "http://localhost:8000/weather/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Should I evacuate from Tampa Beach?", "user_id": "test_003"}'

# Create Category 4 hurricane alert (requires HITL approval)
curl -X POST "http://localhost:8000/weather/hurricane/alert" \
  -H "Content-Type: application/json" \
  -d '{"category": 4, "message": "Cat 4 Hurricane approaching Tampa", "thread_id": "alert-001"}'

# Approve pending alert
curl -X POST "http://localhost:8000/weather/hurricane/approve/alert-001" \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "reviewer_notes": "Verified with NHC"}'

# Health check
curl http://localhost:8000/health

# Cache statistics
curl http://localhost:8000/cache/stats

# Prometheus metrics
curl http://localhost:8000/metrics
```
**LangGraph Studio Graphs** (4 graphs for visual debugging):
- `weather_agent`: Unified agent with RAG + CoT
- `weather_hitl_workflow`: Hurricane alert HITL approval workflow
- `multi_agent_workflow`: Level 4a 3-agent orchestration
- `level4b_workflow`: Level 4b 8-agent parallel execution

---

## Testing (Progressive Build Approach)

**Unified Testing Command** (Recommended):
```bash
# Test ALL implemented features for current level (Level 1+2+3)
make test-dev
```

This single command:
1. ✅ Restarts dev environment (clean state)
2. ✅ Runs complete test suite (MCP, RAG, HITL, Memory, Reasoning, Cache, Multi-Agent Orchestration, Auto-Routing intelligence)
3. ✅ Verifies LangChain v1.x compliance (100% required)

**Progressive**: As you build more levels, `make test-dev` automatically includes new tests without Makefile changes!

**Level-Specific Test Guides**:
- **[Level 2 Test Guide](./docs/test-guide/LEVEL_2_TEST_GUIDE.md)** - RAG + CoT + Hybrid Search (6 Swagger tests + 6 Studio configs)
- **[Level 3 Test Guide](./docs/test-guide/LEVEL_3_TEST_GUIDE.md)** - Memory + ToT/GoT
- **[Level 4 Test Guide](./docs/test-guide/LEVEL_4_TEST_GUIDE.md)** -  Multi-Agent Orchestration + Auto-Routing intelligence
- **[Level 5 Test Guide](./docs/test-guide/LEVEL_5_TEST_GUIDE.md)** - Eval + Guardrail + Observability
- 
**Prerequisites:** Docker services running
```bash
make docker-up-dev
make rag-load
make test-dev
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

**Version**: 1.3.1 | **Last Updated**: 2025-12-14