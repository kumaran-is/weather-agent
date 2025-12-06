.PHONY: help install install-dev sync verify clean test lint format type-check security run-verify run-agent all-checks quickstart dev update lock version rag-validate rag-load rag-load-mock-only rag-load-skip-validation docker-up docker-up-dev docker-down docker-down-dev docker-restart docker-restart-dev docker-logs docker-logs-dev docker-ps docker-ps-dev docker-health docker-clean docker-clean-dev

# Default Python version
PYTHON_VERSION := 3.13

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:  ## Show this help message
	@echo "$(BLUE)Weather AI Agent Service - Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Available Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install:  ## Install production dependencies using uv
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	uv sync --no-dev
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

install-dev:  ## Install all dependencies including dev tools
	@echo "$(BLUE)Installing all dependencies (including dev tools)...$(NC)"
	uv sync
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

sync:  ## Sync dependencies from pyproject.toml and update uv.lock
	@echo "$(BLUE)Syncing dependencies...$(NC)"
	uv sync
	@echo "$(GREEN)✓ Dependencies synced$(NC)"

verify:  ## Run Level 0 setup verification (8 automated checks)
	@echo "$(BLUE)Running Level 0 setup verification...$(NC)"
	uv run python verify_setup.py
	@echo "$(GREEN)✓ Verification complete$(NC)"

clean:  ## Clean up cache and temporary files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

test:  ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	uv run pytest
	@echo "$(GREEN)✓ Tests passed$(NC)"

lint:  ## Run ruff linter
	@echo "$(BLUE)Running ruff linter...$(NC)"
	uv run ruff check .
	@echo "$(GREEN)✓ Linting complete$(NC)"

format:  ## Format code with ruff and black
	@echo "$(BLUE)Formatting code...$(NC)"
	uv run ruff check --fix .
	uv run black .
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check:  ## Run mypy type checker
	@echo "$(BLUE)Running type checker...$(NC)"
	uv run mypy . --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete$(NC)"

security:  ## Run security checks with bandit
	@echo "$(BLUE)Running security checks...$(NC)"
	uv run bandit -r . -x ./.venv,./tests
	@echo "$(GREEN)✓ Security checks passed$(NC)"

all-checks: lint type-check test  ## Run all quality checks (lint + type-check + test)
	@echo "$(GREEN)✓ All quality checks passed!$(NC)"

# Run commands
run-verify:  ## Run Level 0 verification script (alias for 'verify')
	@echo "$(BLUE)Running Level 0 setup verification...$(NC)"
	uv run python verify_setup.py

run-agent:  ## Run Weather AI Agent Service (when implemented in Level 1+)
	@echo "$(BLUE)Starting Weather AI Agent Service...$(NC)"
	@echo "$(YELLOW)⚠️  Agent implementation starts in Level 1$(NC)"
	@echo "$(YELLOW)   Current: Level 0 (Setup Complete)$(NC)"

# Version info
version:  ## Show installed versions
	@echo "$(BLUE)Installed Versions:$(NC)"
	@uv run python --version
	@echo ""
	@uv run python -c "import langchain; print(f'langchain: {langchain.__version__}')" 2>/dev/null || echo "langchain: not installed"
	@uv run python -c "import langchain_core; print(f'langchain-core: {langchain_core.__version__}')" 2>/dev/null || echo "langchain-core: not installed"
	@uv run python -c "import langgraph; print(f'langgraph: {langgraph.__version__}')" 2>/dev/null || echo "langgraph: not installed"
	@uv run python -c "import langchain_openai; print(f'langchain-openai: {langchain_openai.__version__}')" 2>/dev/null || echo "langchain-openai: not installed"
	@uv run python -c "import langchain_anthropic; print(f'langchain-anthropic: {langchain_anthropic.__version__}')" 2>/dev/null || echo "langchain-anthropic: not installed"

# Quick start
quickstart: install-dev version verify  ## Quick start - install everything, show versions, and verify setup
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. If verification failed, copy .env.template to .env and add your API keys"
	@echo "  2. Run 'make verify' again to confirm setup"
	@echo "  3. Run 'make help' to see all available commands"

# Development workflow
dev: install-dev  ## Setup development environment
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Copy .env.template to .env and add your API keys"
	@echo "  2. Run 'make verify' to validate setup (8 checks)"
	@echo "  3. Run 'make all-checks' before committing code"
	@echo "  4. Proceed to Level 1: git checkout level-1-react-agent-hitl"

# Update dependencies
update:  ## Update all dependencies to latest compatible versions
	@echo "$(BLUE)Updating dependencies...$(NC)"
	uv lock --upgrade
	uv sync
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

# Lock file
lock:  ## Generate/update uv.lock file
	@echo "$(BLUE)Generating uv.lock...$(NC)"
	uv lock
	@echo "$(GREEN)✓ uv.lock generated$(NC)"

# ============================================================================
# RAG Knowledge Base Commands (Level 2)
# ============================================================================

rag-validate:  ## Validate Kaggle datasets before loading
	@echo "$(BLUE)Validating Kaggle datasets...$(NC)"
	@uv run python -m backend.src.rag.loaders.validate_kaggle_datasets
	@echo "$(GREEN)✓ Validation complete$(NC)"
	@echo ""
	@echo "$(YELLOW)Next step: make rag-load$(NC)"

rag-load:  ## Load weather knowledge base into Qdrant (mock + Kaggle data)
	@echo "$(BLUE)Building and loading RAG knowledge base...$(NC)"
	@echo "$(YELLOW)This will:$(NC)"
	@echo "  1. Validate Kaggle datasets"
	@echo "  2. Load mock weather documents (~20-30 files)"
	@echo "  3. Load Kaggle datasets (~500-600 documents)"
	@echo "  4. Generate embeddings (OpenAI text-embedding-3-small)"
	@echo "  5. Load into Qdrant vector store"
	@echo ""
	@uv run python -m backend.src.rag.build_knowledge_base
	@echo ""
	@echo "$(GREEN)✓ Knowledge base loaded!$(NC)"
	@echo ""
	@echo "$(YELLOW)Test similarity search:$(NC)"
	@echo "  uv run python -c \"from backend.src.rag import get_vector_store; vs = get_vector_store(); print(vs.similarity_search('Category 5 hurricane'))\""

rag-load-mock-only:  ## Load only mock data (skip Kaggle datasets)
	@echo "$(BLUE)Loading mock data only...$(NC)"
	@uv run python -m backend.src.rag.build_knowledge_base --mock-only
	@echo "$(GREEN)✓ Mock data loaded$(NC)"

rag-load-skip-validation:  ## Load knowledge base without validating datasets first
	@echo "$(BLUE)Loading knowledge base (skipping validation)...$(NC)"
	@uv run python -m backend.src.rag.build_knowledge_base --skip-validation
	@echo "$(GREEN)✓ Knowledge base loaded$(NC)"

rag-test:  ## Test RAG retrieval (verify both mock and Kaggle data)
	@echo "$(BLUE)Testing RAG knowledge base retrieval...$(NC)"
	@echo ""
	@echo "$(YELLOW)[Test 1/3] Testing Mock Data - Hurricane Information$(NC)"
	@uv run python -c "from backend.src.rag import get_vector_store; vs = get_vector_store(); results = vs.similarity_search('What is a Category 5 hurricane?', k=2); print('\n'.join([f'  ✓ {r.page_content[:100]}...' for r in results]))"
	@echo ""
	@echo "$(YELLOW)[Test 2/3] Testing Kaggle Data - City Climate$(NC)"
	@uv run python -c "from backend.src.rag import get_vector_store; vs = get_vector_store(); results = vs.similarity_search('climate profile of Tokyo', k=2); print('\n'.join([f'  ✓ {r.page_content[:100]}...' for r in results]))"
	@echo ""
	@echo "$(YELLOW)[Test 3/3] Testing Kaggle Data - Daily Temperature$(NC)"
	@uv run python -c "from backend.src.rag import get_vector_store; vs = get_vector_store(); results = vs.similarity_search('temperature in summer', k=2); print('\n'.join([f'  ✓ {r.page_content[:100]}...' for r in results]))"
	@echo ""
	@echo "$(GREEN)✓ All RAG tests passed!$(NC)"
	@echo ""
	@echo "$(YELLOW)Collection Stats:$(NC)"
	@uv run python -c "from backend.src.rag.vector_store import get_qdrant_client; client = get_qdrant_client(); info = client.get_collection('weather_knowledge'); print(f'  Total documents: {info.points_count}'); print(f'  Vector size: {info.config.params.vectors.size}'); print(f'  Distance: {info.config.params.vectors.distance}')"

# ============================================================================
# Docker Compose Commands (Level 2: 4 Services)
# Services: weather-mcp:8080, hurricane-mcp:8081, qdrant:6333, weather-ai-api:8000
# ============================================================================

docker-up:  ## Start all Docker containers in PRODUCTION mode (Level 2: 4 services)
	@echo "$(BLUE)Starting all Docker containers (PRODUCTION MODE)...$(NC)"
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  - weather-mcp:      http://localhost:8080"
	@echo "  - hurricane-mcp:    http://localhost:8081"
	@echo "  - qdrant:           http://localhost:6333  (NEW in Level 2)"
	@echo "  - weather-ai-api:   http://localhost:8000"
	docker-compose up -d
	@echo "$(GREEN)✓ All containers started$(NC)"
	@echo ""
	@echo "$(YELLOW)Check status: make docker-ps$(NC)"
	@echo "$(YELLOW)View logs:    make docker-logs$(NC)"
	@echo "$(YELLOW)Load RAG:     make rag-load$(NC)"

docker-up-dev:  ## Start all Docker containers in DEVELOPMENT mode (hot reload enabled)
	@echo "$(BLUE)Starting all Docker containers (DEVELOPMENT MODE)...$(NC)"
	@echo "$(YELLOW)Dev Features:$(NC)"
	@echo "  ✅ Hot reload enabled (code changes reflect immediately)"
	@echo "  ✅ Source code mounted as volume (no rebuild needed)"
	@echo "  ✅ Debug logging enabled"
	@echo "  ✅ LangSmith tracing enabled"
	@echo ""
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  - weather-mcp:      http://localhost:8080"
	@echo "  - hurricane-mcp:    http://localhost:8081"
	@echo "  - qdrant:           http://localhost:6333"
	@echo "  - weather-ai-api:   http://localhost:8000 (DEV MODE)"
	docker-compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)✓ All containers started in DEV mode$(NC)"
	@echo ""
	@echo "$(YELLOW)Check status: make docker-ps-dev$(NC)"
	@echo "$(YELLOW)View logs:    make docker-logs-dev$(NC)"
	@echo "$(YELLOW)Stop:         make docker-down-dev$(NC)"

docker-down:  ## Stop and remove all PRODUCTION Docker containers
	@echo "$(BLUE)Stopping all Docker containers (PRODUCTION)...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ All containers stopped and removed$(NC)"

docker-down-dev:  ## Stop and remove all DEVELOPMENT Docker containers
	@echo "$(BLUE)Stopping all Docker containers (DEVELOPMENT)...$(NC)"
	docker-compose -f docker-compose.dev.yml down
	@echo "$(GREEN)✓ All dev containers stopped and removed$(NC)"

docker-restart:  ## Restart all PRODUCTION Docker containers
	@echo "$(BLUE)Restarting all Docker containers (PRODUCTION)...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✓ All containers restarted$(NC)"

docker-restart-dev:  ## Restart all DEVELOPMENT Docker containers
	@echo "$(BLUE)Restarting all Docker containers (DEVELOPMENT)...$(NC)"
	docker-compose -f docker-compose.dev.yml restart
	@echo "$(GREEN)✓ All dev containers restarted$(NC)"

docker-logs:  ## Follow logs from all PRODUCTION Docker containers (Ctrl+C to exit)
	@echo "$(BLUE)Following logs from all containers (Ctrl+C to exit)...$(NC)"
	docker-compose logs -f

docker-logs-dev:  ## Follow logs from all DEVELOPMENT Docker containers (Ctrl+C to exit)
	@echo "$(BLUE)Following logs from all dev containers (Ctrl+C to exit)...$(NC)"
	docker-compose -f docker-compose.dev.yml logs -f

docker-ps:  ## Show status of all PRODUCTION Docker containers
	@echo "$(BLUE)Docker container status (PRODUCTION):$(NC)"
	@docker-compose ps

docker-ps-dev:  ## Show status of all DEVELOPMENT Docker containers
	@echo "$(BLUE)Docker container status (DEVELOPMENT):$(NC)"
	@docker-compose -f docker-compose.dev.yml ps

docker-health:  ## Check health status of all Docker containers
	@echo "$(BLUE)Checking health status of all containers...$(NC)"
	@echo ""
	@echo "$(YELLOW)Weather MCP Server (8080):$(NC)"
	@docker inspect --format='{{.State.Health.Status}}' weather-mcp-server 2>/dev/null || echo "$(RED)Not running$(NC)"
	@echo ""
	@echo "$(YELLOW)Hurricane Tracker MCP (8081):$(NC)"
	@docker inspect --format='{{.State.Health.Status}}' hurricane-tracker-mcp 2>/dev/null || echo "$(RED)Not running$(NC)"
	@echo ""
	@echo "$(YELLOW)Weather AI API (8000):$(NC)"
	@docker inspect --format='{{.State.Health.Status}}' weather-ai-api 2>/dev/null || echo "$(RED)Not running$(NC)"
	@echo ""
	@echo "$(GREEN)✓ Health check complete$(NC)"

docker-clean:  ## Stop PRODUCTION containers and remove volumes (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will remove all PRODUCTION Docker volumes and delete all data$(NC)"
	@echo "$(YELLOW)Press Ctrl+C within 5 seconds to cancel...$(NC)"
	@sleep 5
	docker-compose down -v
	@echo "$(GREEN)✓ All production containers and volumes removed$(NC)"

docker-clean-dev:  ## Stop DEVELOPMENT containers and remove volumes (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will remove all DEVELOPMENT Docker volumes and delete all data$(NC)"
	@echo "$(YELLOW)Press Ctrl+C within 5 seconds to cancel...$(NC)"
	@sleep 5
	docker-compose -f docker-compose.dev.yml down -v
	@echo "$(GREEN)✓ All dev containers and volumes removed$(NC)"
