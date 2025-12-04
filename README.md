# weather-agent

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kumaran-is/weather-agent)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.0+](https://img.shields.io/badge/langchain-1.0+-green.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph 1.0+](https://img.shields.io/badge/langgraph-1.0+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![MCP Protocol](https://img.shields.io/badge/MCP%20Protocol-Dual%20Servers-orange)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](./CHANGELOG.md)

**Production-grade AI agent for weather forecast intelligence:** Built with LangChain 1.0, LangGraph 1.0, a, FastAPI and OpenAI. Features real-time weather and hurricane  data via MCP integration, RAG, streaming responses, type-safe validation with Pydantic, structured logging, and enterprise-grade observability.

**From zero to production** Progressive implementation showcasing enterprise AI patterns, multi-agent orchestration, 7-layer memory systems, and production deployment strategies.

**How It Works:** This project follows a progressive complexity model—each module builds on the previous one, making it accessible to beginners while scaling toward advanced concepts.

**Current Stage**: Level 0 (Setup Complete) — Production environment

**Current Stage**: ✅ **Level 1 Complete (v0.2.1)** — Production-ready weather agent with Docker containerization, complete MCP integration (all 3 tools: `get_current_weather`, `get_forecast`, `retrieve_weather_context`), ReAct pattern reasoning, Human-in-the-Loop (HITL) approval for safety-critical alerts, LangSmith Studio integration for visual real-time debugging, LangSmith Studio for real-time debugging, HITL workflow for hurricane alerts, and comprehensive testing suite

[Read the Medium Blog Post Series](https://medium.com/@yourusername)

---

## Table of Contents

- [weather-agent](#weather-agent)
  - [Table of Contents](#table-of-contents)
  - [Technology Stack](#technology-stack)
  - [Quick Start](#quick-start)
    - [System Requirements](#system-requirements)
    - [Prerequisites](#prerequisites)
    - [MCP Servers Required](#mcp-servers-required)
    - [Setup (15 minutes)](#setup-15-minutes)
      - [Option 1: Quick Start with Makefile (Recommended)](#option-1-quick-start-with-makefile-recommended)
      - [Option 2: Manual Setup (Without Makefile)](#option-2-manual-setup-without-makefile)
    - [Common Makefile Commands](#common-makefile-commands)
    - [Project Structure](#project-structure)
    - [Progressive Learning Path](#progressive-learning-path)
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
- [**NOAA/NHC**](https://www.nhc.noaa.gov/) - National Hurricane Center real-time data
- [**NWS API**](https://www.weather.gov/documentation/services-web-api) - Weather alerts & forecasts
- [**IBTrACS**](https://www.ncei.noaa.gov/products/international-best-track-archive) - Historical hurricane tracks

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
Edit `.env` with your editor of choice and add:

**API Keys:**
- `OPENAI_API_KEY` (Get from https://platform.openai.com/api-keys)
- `LANGCHAIN_API_KEY` (Get from https://smith.langchain.com/)

**MCP Server URLs** (after running [setup guide](docs/setup/mcp-servers-setup.md)):

*For local development (default)*:
- `MCP_WEATHER_SERVER_URL=http://localhost:8080`
- `MCP_WEATHER_SERVER_ENABLED=true`
- `MCP_HURRICANE_SERVER_URL=http://localhost:8081`
- `MCP_HURRICANE_SERVER_ENABLED=true`

*For Docker deployment*: URLs are auto-configured via Docker Compose (no .env changes needed)

**Optional Configuration:**
- `ANTHROPIC_API_KEY` (for future levels using Claude models)
- `LOG_LEVEL=info` (default: info, options: debug, info, warning, error)
- `LANGCHAIN_TRACING_V2=true` (enable LangSmith tracing, default: true)
- `LANGCHAIN_ENDPOINT=https://api.smith.langchain.com` (LangSmith API endpoint)
- `LANGCHAIN_PROJECT=weather-ai-agent-service` (LangSmith project name for traces)
- 
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

Run **9** automated checks:
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
Expected output: `ALL 8 CHECKS PASSED!`

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


**Docker Commands (3 Services: weather-mcp:8080, hurricane-mcp:8081, weather-ai-api:8000)**

| Command | Description |
|---------|-------------|
| `make docker-up` | Start all Docker containers in detached mode |
| `make docker-down` | Stop and remove all Docker containers |
| `make docker-restart` | Restart all Docker containers |
| `make docker-ps` | Show status of all Docker containers |
| `make docker-logs` | Follow logs from all containers (Ctrl+C to exit) |
| `make docker-health` | Check health status of all 3 services |
| `make docker-clean` | Stop containers and remove volumes (⚠️ deletes all data) |

**Typical Docker Workflow:**
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

### Project Structure
```
weather-agent/
├── backend/                     # Backend application (Level 1+)
│   ├── config/                  # Settings and configuration
│   │   └── settings.py          # Centralized app settings
│   └── src/                     # Source code
│       ├── agents/              # LangChain agents
│       │   ├── state.py         # Agent state TypedDict
│       │   ├── prompts.py       # System prompts
│       │   └── weather_agent.py # ReAct agent (create_agent)
│       ├── api/                 # FastAPI service
│       │   ├── main.py          # FastAPI app + endpoints
│       │   └── schemas.py       # Pydantic request/response models
│       ├── hitl/                # Human-in-the-Loop
│       │   └── approval_node.py # HITL approval nodes
│       ├── mcp/                 # MCP client integration
│       │   └── weather_client.py # Async HTTP MCP client
│       ├── tools/               # LangChain tools
│       │   └── weather_tools.py # 3 MCP tool wrappers (@tool)
│       └── workflows/           # LangGraph workflows
│           └── weather_graph.py # HITL StateGraph workflow
├── docs/                        # Documentation
│   ├── setup/                   # Setup guides
│   │   ├── mcp-servers-setup.md        # MCP servers deployment
│   │   ├── langsmith-studio-setup.md   # Studio integration (434 lines)
│   │   └── docker-usage-guide.md       # Docker orchestration (production + development modes)
├── tests/                       # Test suite (Level 1+)
│   ├── conftest.py              # Pytest fixtures
│   ├── test_mcp_client.py       # MCP client tests
│   ├── test_weather_tool.py     # Tool tests
│   ├── test_react_agent.py      # Agent tests
│   ├── test_hurricane_hitl.py   # HITL tests
│   ├── test_workflow.py         # Workflow tests
│   ├── test_api.py              # API endpoint tests
│   └── test_integration.py      # Integration tests
├── Dockerfile                   # Multi-stage production build
├── docker-compose.yml           # Service orchestration (MCP servers)
├── langgraph.json               # LangSmith Studio configuration
├── pyproject.toml               # Project config (source of truth)
├── uv.lock                      # uv lockfile (reproducible builds)
├── Makefile                     # Development commands (17 tasks)
├── verify_setup.py              # Level 0 verification (8 checks)
├── CHANGELOG.md                 # Version history (v0.2.1)
└── README.md                    # This file
```

**Key Highlights**:
- **backend/src/**: Complete Level 1 implementation (agents, API, MCP, HITL, workflows)
- **docs/setup/**: Comprehensive setup guides (MCP servers, Studio, Docker)
- **tests/**: 8 test modules with 32 test cases (75% pass rate)
- **langgraph.json**: Studio configuration for visual debugging

**Note:** We use both `pyproject.toml` (defines what dependencies you want) and `uv.lock` (locks exact versions) to ensure reproducible builds across all environments.

📖 **Learn more**: See [Dependency Management Guide](./docs/setup/dependency-management.md) for detailed workflows on adding, updating, and removing dependencies.

### Progressive Learning Path
- **Level 0** (v0.1.0): Setup ✅
- **Level 1** (v0.2.0-v0.2.1): ReAct + HITL + LangSmith Studio ✅
- **Level 2** (v0.3.0): CoT + RAG
- **Level 3a-c** (v0.4.0-0.6.0): Memory (2-layer → 7-layer)
- **Level 4a-c** (v0.7.0-0.9.0): Multi-Agent (3 → 15 agents)
- **Level 5a-c** (v0.10.0-1.0.0): Production
- **Level 6** (v1.1.0): Self-Evolving

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

