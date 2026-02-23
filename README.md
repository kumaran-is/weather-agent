# Weather AI Agent Service

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kumaran-is/weather-agent)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.0+](https://img.shields.io/badge/langchain-1.0+-green.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph 1.0+](https://img.shields.io/badge/langgraph-1.0+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![MCP Protocol](https://img.shields.io/badge/MCP%20Protocol-Dual%20Servers-orange)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.7.0-blue.svg)](./CHANGELOG.md)

[![Blog: Multi-Agent Architecture](https://img.shields.io/badge/Medium-3--Layer%20Multi--Agent%20Architecture-000000?style=flat&logo=medium&logoColor=white)](https://medium.com/@kumaran.isk/from-zero-to-production-ready-ai-agent-the-3-layer-multi-agent-architecture-4f22206bde7d)
[![Blog: Multi-Agent Orchestration](https://img.shields.io/badge/Medium-Multi--Agent%20Orchestration%20with%20Self--Healing-000000?style=flat&logo=medium&logoColor=white)](https://medium.com/@kumaran.isk/15-agents-zero-downtime-multi-agent-orchestration-with-self-healing-52a91394fc57)
[![Blog: Memory Architecture](https://img.shields.io/badge/Medium-7--Layer%20Memory%20Architecture-000000?style=flat&logo=medium&logoColor=white)](https://medium.com/@kumaran.isk/teaching-ai-to-remember-7-layer-memory-architecture-and-token-compression-c52f9c5db6ea)
[![Blog: AI Guardrails](https://img.shields.io/badge/Medium-12--Layer%20AI%20Guardrails%20%26%20Evaluation-000000?style=flat&logo=medium&logoColor=white)](https://medium.com/@kumaran.isk/building-safe-ai-systems-12-layer-ai-guardrails-and-evaluation-736c62bded4c)
[![Blog: Semantic Caching](https://img.shields.io/badge/Medium-Semantic%20Caching%20%26%20LLM%20Cost%20Optimization-000000?style=flat&logo=medium&logoColor=white)](https://medium.com/@kumaran.isk/ai-safety-first-semantic-caching-llm-cost-optimization-f637d847cbc8)
[![Blog: Design Patterns](https://img.shields.io/badge/Medium-20%20AI%20Agent%20Design%20Patterns-000000?style=flat&logo=medium&logoColor=white)](https://medium.com/@kumaran.isk/20-ai-agent-design-patterns-organized-as-a-stack-not-a-menu-8e0df0ee99c3)
[![Blog: Production Patterns](https://img.shields.io/badge/Medium-11%20Operational%20Patterns%20for%20Production-000000?style=flat&logo=medium&logoColor=white)](https://medium.com/@kumaran.isk/the-11-operational-patterns-that-keep-ai-agents-running-in-production-889b054f24a4)

**Production-grade AI agent for weather forecast intelligence:** Built with LangChain 1.0, LangGraph 1.0, FastAPI and OpenAI. Features 15-agent multi-agent orchestration, auto-routing (intent-based query classification), 7-layer memory architecture, advanced reasoning (Tree/Graph-of-Thought), emotional intelligence, **two-tier semantic caching** (40-60% cost reduction), multi-layer caching (L1+L2+L3+Semantic), Prometheus metrics and observability, real-time weather data via dual MCP servers, RAG-enhanced knowledge base, Chain-of-Thought reasoning, Human-in-the-Loop (HITL) approval workflows, self-evolving AI with constitutional guardrails, comprehensive golden dataset evaluation (185 test cases, ZERO safety violations) and Two-Tier Semantic Caching -LLM resposne and Tool caching.

**From zero to production:** Progressive implementation showcasing enterprise AI patterns including multi-agent orchestration (3→8→15 agents), auto-routing architecture, 7-layer memory systems (99.7% storage reduction), context window optimization (50-60% token reduction for <4K tokens/query), **semantic caching** (65-85% hit rate with similarity matching), full observability stack (Prometheus, Grafana, OpenTelemetry), adversarial testing (RAGAS/DeepEval), constitutional AI (12-layer safety guardrails), LangGraph-bigtool Dynamic Tool Registry & Semantic Discovery, Context Window Optimization, Two-Tier Semantic Caching and **[planned]** self-evolving agentic dual-loop architecture that continuously learns, adapts, and improves based on real-world performance.

**How It Works:** This project follows a progressive complexity model—each module builds on the previous one, making it accessible to beginners while scaling toward advanced concepts.

**Current Stage**: ✅ **Level 9 Complete** — Two-Tier Semantic Caching: Query/Tool/LLM Response Caching with Similarity Matching, Life-Safety Bypass, Prometheus Metrics (100% LangChain v1.x/LangGraph v1.x Compliant)

[Read the Medium Blog Post Series](https://medium.com/@kumaran.isk)

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
| [**Langgraph Bigtool**](https://github.com/langchain-ai/langgraph-bigtool) | `0.0.3` | Tool Registry & Semantic Discovery |

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
| [**Prometheus**](https://prometheus.io/) | Metrics collection (with exemplar storage) |
| [**Grafana**](https://grafana.com/) | Dashboards & visualization (signal correlation) |
| [**Tempo**](https://grafana.com/oss/tempo/) | Distributed tracing (Level 8) |
| [**Loki**](https://grafana.com/oss/loki/) | Log aggregation |
| [**LangSmith**](https://smith.langchain.com/) | LLM tracing & debugging |

**Data Sources** (via MCP Servers):
- [**NOAA/NHC**](https://www.nhc.noaa.gov/) - Hurricane tracking data
- [**NWS API**](https://www.weather.gov/documentation/services-web-api) - Weather alerts & forecasts

---

## Testing Tools & Interfaces

Quick reference for testing and debugging the Weather AI Agent:

| Tool | URL | Description |
|------|-----|-------------|
| **Swagger UI** | http://localhost:8000/docs | Interactive API testing with example data dropdowns |
| **ReDoc** | http://localhost:8000/redoc | Read-only API documentation |
| **LangSmith Dashboard** | https://smith.langchain.com/ | LLM tracing, agent debugging, cost analysis |
| **LangSmith Tracing** | https://smith.langchain.com/o/{org_id}/projects/p/{project_id} | Detailed agent trace inspection & debugging |
| **LangSmith Evaluation** | https://smith.langchain.com/o/{org_id}/datasets | Golden dataset management & batch evaluation (185 test cases) |
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

**Observability Stack (Level 8 - Signal Correlation)**

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3001 | admin / weatherai2025 |
| **Prometheus** | http://localhost:9090 | None (exemplar storage enabled) |
| **Tempo** | http://localhost:3200 | None (OTLP: 4317/4318) |
| **Loki** | http://localhost:3100 | API only |

**Signal Correlation (Level 8)**: Click exemplars on metrics → Jump to traces → View related logs

**Test Guides by Level**:

| Level | Guide | Key Features | Status |
|-------|-------|--------------|--------|
| **Level 0** | [Setup Verification](./verify_setup.py) | Python 3.13+, uv, Docker, API keys | ✅ Complete |
| **Level 1** | [Test Suite](./tests/test_react_agent.py) | ReAct agent, HITL approval, MCP integration | ✅ Complete |
| **Level 2** | [RAG + CoT Guide](./docs/test-guide/LEVEL_2_TEST_GUIDE.md) | 603 docs, hybrid search, CoT reasoning | ✅ Complete |
| **Level 3** | [Memory + Reasoning Guide](./docs/test-guide/LEVEL_3_TEST_GUIDE.md) | 7-layer memory, ToT/GoT, emotional intelligence | ✅ Complete |
| **Level 4** | [Multi-Agent Guide](./docs/test-guide/LEVEL_4_TEST_GUIDE.md) | 15-agent orchestration, auto-routing | ✅ Complete |
| **Level 5** | [Production Guide](./docs/test-guide/LEVEL_5_TEST_GUIDE.md) | Caching (L1/L2/L3), Prometheus, Swagger UI (37 scenarios) | ✅ Complete |
| **Level 6** | [Self-Evolving Guide](./docs/test-guide/LEVEL_6_TEST_GUIDE.md) | Context optimization, RAGAS, Constitutional AI, 185 test cases | ✅ Complete |
| **Level 7** | [MCP Tool Dynamic Registry](./docs/test-guide/LEVEL_7_TEST_GUIDE.md)| LangGraph-bigtool Dynamic Tool Registry & Semantic Discovery | ✅ Complete |
| **Level 8** | [Context Engineering & Optimization](./docs/test-guide/LEVEL_8_TEST_GUIDE.md)| Context Window Optimization & Full Observability Stack(Tracing, Structured Logging, Metrics and Alerts)  | ✅ Complete  |
| **Level 9** | [Semantic Caching for AI Agents](./docs/test-guide/LEVEL_9_TEST_GUIDE.md)| Two-Tier Semantic Caching: Query/Tool/LLM Response Caching with Similarity Matching | ✅ Complete |
| **Level 10** | Self Evolving Agenitc Dual-Loop Architecture | Self-evolving agentic systems that continuously learns, adapts, and improves based on real-world performance. | 📋 Planned |
| **Level 11** | Advanced HITL |  | 📋 Planned |

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
| `LANGCHAIN_PROJECT` | `weather-agent` | LangSmith project name for organizing traces |
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
| `make test-dev` | 🌟 **Unified testing** - Restart dev environment + run complete test suite + compliance check (progressive: currently tests Level 1+2+3 complete) |
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

**Docker Troubleshooting**

| Issue | Symptom | Solution |
|-------|---------|----------|
| **BuildKit Cache Corruption** | `failed to prepare extraction snapshot... parent snapshot does not exist` | Run `docker builder prune -f` then rebuild |
| **Stale Images** | Old code running after changes | Run `docker-compose build --no-cache` |
| **Port Conflicts** | `port is already allocated` | Check `docker ps -a` and stop conflicting containers |
| **Volume Permission Issues** | Permission denied errors | Run `docker-compose down -v` and restart |

**Common Docker Build Errors:**

```bash
# Error: "parent snapshot does not exist: not found"
# Cause: Docker BuildKit cache corruption (common after Docker restarts/updates)
# Fix:
docker builder prune -f
make docker-rebuild-dev

# If still failing, full cleanup:
docker system prune -a --volumes -f
make docker-rebuild-dev
```

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

**Observability Commands (Level 8: Prometheus + Grafana + Loki + Tempo)**

| Command | Description |
|---------|-------------|
| `make observability-status` | Check Prometheus/Grafana/Loki/Tempo status |
| `make grafana-open` | Open Grafana dashboard (http://localhost:3001) |
| `make prometheus-open` | Open Prometheus UI (http://localhost:9090) |
| `make prometheus-reload` | Reload Prometheus configuration (hot reload) |
| `make loki-logs` | Query recent logs from Loki |
| `make observability-logs` | Follow logs from observability stack |
| `make verify-signal-correlation` | Verify signal correlation components (Level 8) |

**Typical Observability Workflow:**
```bash
# 1. Start dev containers (includes observability stack + Tempo)
make docker-up-dev

# 2. Check observability status (includes Tempo)
make observability-status

# 3. Verify signal correlation (Level 8)
make verify-signal-correlation

# 4. Open Grafana dashboards (Signal Correlation dashboard)
make grafana-open
# Login: admin / weatherai2025

# 5. View Prometheus metrics (with exemplars)
make prometheus-open

# 6. Query logs via Loki (in Grafana) - click trace_id to jump to Tempo
make loki-logs
```

**Evaluation Commands (Level 5b + Level 6: Golden Dataset Testing & Quality Gates)**

All 8 evaluation commands validated and working correctly:

| # | Command | Description | Duration | Output File | Status |
|---|---------|-------------|----------|-------------|--------|
| 1 | `make eval-upload-dataset` | Upload golden dataset to LangSmith (185 test cases) | ~5 seconds | N/A (uploads to LangSmith) | ✅ Validated |
| 2 | `make eval-quick` | Quick smoke test (10 random cases) | ~1-2 minutes | `evaluation_quick.json` | ✅ Validated |
| 3 | `make eval-category CATEGORY=simple` | Test simple weather queries (70 cases) | ~8-10 minutes | `evaluation_simple.json` | ✅ Validated |
| 4 | `make eval-category CATEGORY=hurricane` | Test hurricane safety-critical queries (35 cases) | ~7-9 minutes | `evaluation_hurricane.json` | ✅ Validated |
| 5 | `make eval-category CATEGORY=complex` | Test complex multi-location queries (50 cases) | ~10-12 minutes | `evaluation_complex.json` | ✅ Validated |
| 6 | `make eval-category CATEGORY=edge` | Test edge cases and error handling (30 cases) | ~5-6 minutes | `evaluation_edge.json` | ✅ Validated |
| 7 | `make eval-check-gates` | Check quality gates on evaluation results | ~2 seconds | N/A (reads `evaluation_results.json`) | ✅ Validated |
| 8 | `make eval-full` | **Full pipeline**: Upload → Run all 185 cases → Check gates | ~30-35 minutes | `evaluation_results.json` | ✅ Validated |

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

# Full evaluation pipeline (all 185 cases)
make eval-full

# View results in LangSmith
# https://smith.langchain.com/datasets
```

**Quality Gates Guide**: See [docs/knowledgebase/quality-gates-guide.md](docs/knowledgebase/quality-gates-guide.md) for comprehensive details on what quality gates are, how they work, and what to do when they fail.

### Project Structure

```
weather-agent/
├── backend/                     # Backend application (Level 8 Complete)
│   ├── config/                  # Configuration (SINGLE LOCATION)
│   │   ├── settings.py          # Centralized app settings (50+ env vars)
│   │   ├── llm_config.py        # LLM use case configuration
│   │   ├── memory_config.py     # Memory system configuration
│   │   ├── cache_config.py      # Cache system configuration
│   │   └── tool_registry_config.py  # Tool Registry configuration (Level 7)
│   ├── data/raw/                # RAG knowledge base (603 documents)
│   ├── migrations/              # Database migration scripts
│   ├── src/                     # Source code
│   │   ├── agents/              # 15 specialized agents (weather, triage, hurricane, etc.)
│   │   ├── api/main.py          # FastAPI app (13 endpoints + Prometheus metrics)
│   │   ├── cache/               # Multi-layer + Semantic Caching (Level 9)
│   │   │   ├── l1_memory_cache.py        # L1 in-process LRU (<1ms, life-safety bypass)
│   │   │   ├── l2_redis_cache.py         # L2 Redis distributed (<10ms, life-safety bypass)
│   │   │   ├── l3_anthropic_cache.py     # L3 Anthropic prompt cache (transparent)
│   │   │   ├── query_normalizer.py       # Query normalization (aliases, case, whitespace)
│   │   │   ├── semantic_matcher.py       # Semantic similarity matching (Qdrant)
│   │   │   ├── two_tier_cache.py         # Two-tier architecture (exact + semantic)
│   │   │   ├── tool_cache.py             # Tool result caching with bypass
│   │   │   └── llm_cache.py              # LLM response caching
│   │   ├── context/             # Context Window Optimization (Level 8)
│   │   │   ├── context_optimizer.py      # 5-phase optimization pipeline (50-60% reduction)
│   │   │   ├── query_type_detector.py    # Auto-classification (EMERGENCY/COMPLEX/SIMPLE)
│   │   │   ├── semantic_chunker.py       # Semantic text chunking
│   │   │   ├── relevance_filter.py       # Relevance-based filtering
│   │   │   ├── dynamic_assembler.py      # Dynamic context assembly
│   │   │   └── hierarchical_loader.py    # Hierarchical context loading
│   │   ├── evaluation/          # Evaluation framework
│   │   ├── guardrails/          # 12-layer safety guardrails
│   │   ├── hitl/                # Human-in-the-Loop approval nodes
│   │   ├── mcp/                 # MCP client integration
│   │   ├── memory/              # 7-layer memory system
│   │   ├── models/              # Pydantic models (weather, hurricane, health, tool_registry)
│   │   ├── observability/       # Full Observability Stack (Level 8 + Level 9)
│   │   │   ├── callbacks.py             # LangSmith tracing callbacks
│   │   │   ├── logging.py               # Structured JSON logging with trace context
│   │   │   ├── metrics.py               # Prometheus metrics (context optimization + LLM)
│   │   │   ├── cache_metrics.py         # Cache metrics (Level 9: hit/miss/latency/similarity/cost)
│   │   │   └── tracing.py               # OpenTelemetry tracing integration
│   │   ├── orchestration/       # Multi-agent workflow orchestration
│   │   ├── rag/                 # RAG pipeline (embeddings, hybrid search)
│   │   ├── reasoning/           # ToT/GoT advanced reasoning
│   │   ├── registry/            # Tool Registry & Semantic Discovery (Level 7)
│   │   │   ├── tool_registry.py         # Thread-safe singleton registry
│   │   │   └── semantic_discovery.py    # AI-powered tool discovery
│   │   ├── routing/             # Auto-routing system
│   │   ├── services/            # Shared business logic services
│   │   ├── tools/               # LangChain tools (MCP + RAG + Hurricane)
│   │   │   ├── weather_tools.py         # Weather MCP tool wrappers
│   │   │   ├── rag_tools.py             # RAG retrieval tools
│   │   │   └── hurricane_tools.py       # Hurricane MCP tool wrappers (Level 7)
│   │   ├── utils/               # Utility functions and helpers
│   │   └── workflows/           # LangGraph workflows
│   └── tests/                   # Backend-specific tests
├── docs/                        # Documentation
│   ├── setup/                   # Setup and configuration guides
│   ├── test-guide/              # Test guides by level (9 guides)
├── observability/               # Observability stack configuration (Level 8)
│   ├── grafana/                 # Grafana dashboards and datasources (signal correlation)
│   ├── loki/                    # Loki log aggregation config
│   ├── prometheus/              # Prometheus metrics config (exemplar storage)
│   └── tempo/                   # Tempo distributed tracing config (Level 8)
├── scripts/                     # Evaluation and automation scripts
├── tests/                       # Test suite (27 test files)
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── .env.template                # Environment variables template
├── CHANGELOG.md                 # Version history and release notes
├── Dockerfile                   # Multi-stage production container
├── docker-compose.yml           # Production orchestration (9 services)
├── docker-compose.dev.yml       # Development orchestration (hot reload)
├── langgraph.json               # LangGraph Studio (4 graphs)
├── LICENSE                      # MIT License
├── Makefile                     # Development and deployment commands
├── pyproject.toml               # Project config (v1.7.0)
└── README.md                    # This file
```

**Key Directories**:
- **backend/src/agents/**: 15 specialized agents for weather intelligence (supervisor, triage, hurricane, emergency, forecaster, climate, historical, research, personalization, meta-prompt, critique, debate, reflection, self-healing, alert manager)
- **backend/src/cache/**: L1 (in-process), L2 (Redis), L3 (Anthropic) caching
- **backend/src/context/**: Context Window Optimization (Level 8) - 5-phase pipeline achieving 50-60% token reduction
- **backend/src/memory/**: 7-layer memory (conversation, session, episodic, semantic, procedural, emotional, reflective)
- **backend/src/observability/**: Full Observability Stack (Level 8) - Tracing, Structured JSON Logging, Metrics, Alerts
- **backend/src/registry/**: Tool Registry & Semantic Discovery (Level 7) - Thread-safe singleton, AI-powered search
- **backend/src/orchestration/**: Multi-agent workflows (supervisor, parallel execution, auto-routing)
- **backend/src/evaluation/**: 4-pillar evaluation framework (effectiveness, efficiency, robustness, safety)
- **backend/src/tools/**: LangChain tool wrappers (weather MCP, hurricane MCP, RAG retrieval)
- **docs/test-guide/**: 9 test guides (L0-L8, 18 scenarios for Level 8)
- **observability/**: Prometheus, Grafana, Loki, Tempo configuration (Level 8 signal correlation)
- **scripts/**: Evaluation automation (quality gates, batch evaluation, dataset upload)
- **tests/**: 27 test files (unit + integration, including 2 new Level 8 test files)

### Progressive Learning Path
- **Level 0** (v0.1.0): Setup ✅ COMPLETE
- **Level 1** (v0.2.0-v0.2.1): ReAct + HITL + LangSmith Studio ✅ COMPLETE
- **Level 2** (v0.3.0): ReAct + CoT + RAG + Hybrid Search (Semantic + Keyword) ✅ COMPLETE
- **Level 3a** (v0.4.0): 2-Layer Memory (Conversation + Session) ✅ COMPLETE
- **Level 3b** (v0.5.0): Advanced Reasoning (Tree-of-Thoughts + Graph-of-Thoughts) ✅ COMPLETE
- **Level 3c** (v0.6.0): Full 7-Layer Memory + Emotional Intelligence + Personalization ✅ COMPLETE
- **Level 4** (v0.7.0): Multi-Agent Orchestration + Auto-Routing ✅ COMPLETE
  - Auto-Routing v0.6.0: Intent-based query classification (<1ms) ✅ COMPLETE
  - Level 4a: 3-Agent Foundation (Triage + Specialist + Alert) ✅ COMPLETE
  - Level 4b: 8-Agent Orchestration (+ Supervisor + parallel) ✅ COMPLETE
  - Level 4c: 15-Agent Production (+ Meta-Prompt + Debate + Self-Healing + Circuit Breakers) ✅ COMPLETE
- **Level 5a** (v0.8.0): Multi-Layer Caching (L1+L2+L3) ✅ COMPLETE
- **Level 5b** (v0.9.0): Evaluation Framework + 12-Layer Guardrails ✅ COMPLETE
- **Level 5c** (v0.10.0): Full Production + Observability Stack ✅ COMPLETE
- **Level 6** (v1.3.0): Self-Evolving AI Platform ✅ COMPLETE
  - Level 6a: Context Optimization + Ragas/DeepEval Evaluation ✅ COMPLETE
  - Level 6b: TruLens + AgentBench + Auto-Prompt Engineering ✅ COMPLETE
  - Level 6c: Constitutional AI + Production Testing Infrastructure ✅ COMPLETE
- **Level 7** (v1.5.0): LangGraph-bigtool Tool Registry & Semantic Discovery ✅ COMPLETE
- **Level 8** (v1.6.0): Context Window Optimization & Full Observability Stack ✅ COMPLETE
  - Level 8a: Query Type Detection + 5-Phase Optimization Pipeline ✅ COMPLETE
  - Level 8b: OpenTelemetry Tracing + Structured JSON Logging ✅ COMPLETE
  - Level 8c: Prometheus Metrics + Grafana Dashboards + Alert Rules ✅ COMPLETE
  - Test Coverage: 114/114 tests passing (18 scenarios) ✅ COMPLETE
- **Level 9** (v1.7.0): Two-Tier Semantic Caching Architecture ✅ COMPLETE
  - Level 9a: TwoTierCache Base Class + QueryNormalizer + SemanticMatcher ✅ COMPLETE
  - Level 9b: Tool Result Cache with Life-Safety Bypass ✅ COMPLETE
  - Level 9c: LLM Response Cache + Prometheus Metrics ✅ COMPLETE
- **Level 10**: Self Evolving Agentic Architecture: Self-evolving agentic systems that continuously learns, adapts, and improves based on real-world performance. 🔜 NEXT
- **Level 11**: Advanced HITL 🔜 FUTURE

**Current Status**: Level 9 (v1.7.0) - Two-Tier Semantic Caching Architecture (100% LangChain v1.x/LangGraph v1.x Compliant) | Test with `make test-dev`

## API Endpoints Reference

All REST endpoints (13 total):

| Method | Endpoint | Description |
|--------|----------|-------------|
| **POST** | `/weather/query` | Weather queries with AUTO-ROUTING (v0.6.0), multi-agent orchestration, RAG, CoT, memory, 3-layer caching, context optimization (Level 8) |
| **POST** | `/weather/hurricane/alert` | Create hurricane alert (Cat 1-2 auto-approved, Cat 3+ requires HITL approval) |
| **POST** | `/weather/hurricane/approve/{thread_id}` | Approve or reject pending hurricane alerts (Cat 3+ only) |
| **GET** | `/health` | Service health check with 10 monitored services (returns level: L8) |
| **GET** | `/health/context` | Context optimization health metrics (Level 8) - total optimizations, avg reduction %, target achievement rate |
| **GET** | `/health/tools` | Tool Registry health check (Level 7) - registered tools, categories, health status |
| **GET** | `/mcp/health` | MCP server health check (weather + hurricane dual servers) - connectivity, latency, status |
| **GET** | `/cache/stats` | Cache statistics for all 3 layers (L1 in-process, L2 Redis, L3 Anthropic) |
| **POST** | `/cache/clear` | Clear L1 (in-process) and L2 (Redis) cache layers |
| **POST** | `/cache/invalidate` | Invalidate specific cache entry at L1 and L2 |
| **GET** | `/cache/config` | View current cache configuration (L1, L2, L3 settings) |
| **GET** | `/tools/list` | List all registered tools in Tool Registry (Level 7) - with categories and metadata |
| **GET** | `/metrics` | Prometheus metrics endpoint (text format for scraping) - includes context optimization metrics (Level 8) |

**Access Points**:
- **Swagger UI**: http://localhost:8000/docs (interactive testing with 37 example scenarios)
- **ReDoc**: http://localhost:8000/redoc (comprehensive API documentation)
- **Prometheus Metrics**: http://localhost:8000/metrics (plain text format)

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

# Context optimization health metrics (Level 8)
curl http://localhost:8000/health/context

# Tool Registry health check (Level 7)
curl http://localhost:8000/health/tools

# MCP server health check
curl http://localhost:8000/mcp/health

# Cache statistics
curl http://localhost:8000/cache/stats

# List all registered tools (Level 7)
curl http://localhost:8000/tools/list

# Prometheus metrics (includes Level 8 context optimization metrics)
curl http://localhost:8000/metrics
```

**LangGraph Studio Graphs** (4 graphs for visual debugging):
- `weather_agent`: Unified agent with RAG + CoT
- `weather_hitl_workflow`: Hurricane alert HITL approval workflow
- `multi_agent_workflow`: Level 4a 3-agent orchestration
- `level4b_workflow`: Level 4b 8-agent parallel execution

---

## Testing (Progressive Build Approach)

**Current Test Coverage**: 49 test files + 185 golden dataset test cases + Level 9 semantic caching integration tests covering ALL levels (L1-L9)

**Unified Testing Command** (Recommended):
```bash
# Test ALL implemented features (Level 1-9: 49 unit tests + 185 golden dataset cases + semantic caching tests)
make test-dev
```

This single command:
1. ✅ Restarts dev environment (10 Docker services, clean state)
2. ✅ Runs complete unit test suite (49 tests across all levels)
   - **Level 1** (7 tests): ReAct Agent, HITL, MCP integration
   - **Level 2** (4 tests): RAG, CoT, Hybrid Search
   - **Level 3** (15 tests): 7-Layer Memory, ToT/GoT reasoning, 3-Layer Caching (L1+L2+L3)
   - **Level 4** (20 tests): 15-Agent Multi-Agent Orchestration, Supervisor, Parallel Execution
   - **Level 5** (3 tests): Production API, Guardrails, Evaluation Framework
   - **Level 9** (5 tests): Semantic Caching (Query Normalization, Semantic Matching, Life-Safety Bypass, Metrics Export)
3. ✅ Verifies LangChain v1.x compliance (100% required, zero deprecated imports)

**Golden Dataset Evaluation** (Level 6):
```bash
# Run comprehensive evaluation with 185 test cases
make eval-full  # ~30-35 minutes

# Or test specific categories
make eval-category CATEGORY=simple      # 70 simple queries
make eval-category CATEGORY=hurricane   # 35 safety-critical queries
make eval-category CATEGORY=complex     # 50 multi-location queries
make eval-category CATEGORY=edge        # 30 edge cases
```

**Progressive**: As you build more levels, `make test-dev` automatically includes new tests without Makefile changes!

**Level-Specific Test Guides**:
- **[Level 2 Test Guide](./docs/test-guide/LEVEL_2_TEST_GUIDE.md)** - RAG + CoT + Hybrid Search (6 Swagger tests + 6 Studio configs)
- **[Level 3 Test Guide](./docs/test-guide/LEVEL_3_TEST_GUIDE.md)** - Memory + ToT/GoT + Multi-Layer Caching
- **[Level 4 Test Guide](./docs/test-guide/LEVEL_4_TEST_GUIDE.md)** - Multi-Agent Orchestration (15 specialized agents)
- **[Level 5 Test Guide](./docs/test-guide/LEVEL_5_TEST_GUIDE.md)** - Production Deployment + Observability
- **[Level 6 Test Guide](./docs/test-guide/LEVEL_6_TEST_GUIDE.md)** - Advanced Evals + 185 Golden Dataset Cases
- **[Level 7 Test Guide](./docs/test-guide/LEVEL_7_TEST_GUIDE.md)** - LangGraph-bigtool Tool Registry & Semantic Discovery
- **[Level 8 Test Guide](./docs/test-guide/LEVEL_8_TEST_GUIDE.md)** - Context Window Optimization & Full Observability Stack (Tracing, Structured Logging, Metrics and Alerts)
- **[Level 9 Test Guide](./docs/test-guide/LEVEL_9_TEST_GUIDE.md)** - Two-Tier Semantic Caching (Query/Tool/LLM Response Caching, Life-Safety Bypass, Prometheus Metrics) 
  
**Prerequisites:** Docker services running (10 services in dev mode)
```bash
# Start all services (weather-ai-api, MCP servers, databases, observability)
make docker-up-dev

# Load RAG knowledge base (603 documents)
make rag-load

# Run complete unit test suite (49 tests)
make test-dev

# Run comprehensive golden dataset evaluation (185 cases)
make eval-full
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

**Version**: 1.7.0 | **Last Updated**: 2025-12-17