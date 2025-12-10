.PHONY: help install install-dev sync verify clean test test-dev lint format type-check security compliance-check run-verify run-agent all-checks quickstart dev update lock version rag-validate rag-load rag-load-mock-only rag-load-skip-validation rag-test memory-test memory-redis-cli memory-neo4j-browser memory-clear memory-stats docker-up docker-up-dev docker-down docker-down-dev docker-restart docker-restart-dev docker-logs docker-logs-dev docker-ps docker-ps-dev docker-health docker-clean docker-clean-dev

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
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

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

test:  ## Run all tests with pytest
	@echo "$(BLUE)Running all tests...$(NC)"
	uv run pytest
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-dev:  ## Restart dev environment and run complete test suite (progressive: currently Level 3 complete)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Weather AI Agent - Development Test Suite$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Current Level: Level 3 COMPLETE (v0.6.0)$(NC)"
	@echo "  ✅ Level 1: ReAct Agent + HITL"
	@echo "  ✅ Level 2: RAG + Chain-of-Thought"
	@echo "  ✅ Level 3: 7-Layer Memory + Advanced Reasoning + Multi-Layer Caching"
	@echo ""
	@echo "$(BLUE)[1/3] Restarting development environment...$(NC)"
	@docker-compose -f docker-compose.dev.yml restart
	@echo "$(GREEN)✓ Development environment restarted$(NC)"
	@echo ""
	@echo "$(BLUE)[2/3] Running complete test suite...$(NC)"
	@echo "$(YELLOW)Testing: MCP, RAG, HITL, Memory (L3a+L3c), Reasoning (ToT+GoT), Cache (L1+L2+L3)$(NC)"
	@uv run pytest -v
	@echo "$(GREEN)✓ All tests passed$(NC)"
	@echo ""
	@echo "$(BLUE)[3/3] Verifying LangChain v1.x compliance...$(NC)"
	@$(MAKE) compliance-check
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Development test suite COMPLETE$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Next: View logs with 'make docker-logs-dev'$(NC)"

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

compliance-check:  ## Check LangChain v1.x compliance (uses compliance report template)
	@echo "$(BLUE)Running LangChain v1.x compliance check...$(NC)"
	@echo "$(YELLOW)Checking for:$(NC)"
	@echo "  - Deprecated imports (langchain.llms, langchain.agents.AgentExecutor)"
	@echo "  - Legacy type hints (Optional[] instead of | None)"
	@echo "  - Modern agent patterns (create_agent vs create_react_agent)"
	@echo ""
	@ERRORS=0; \
	echo "$(YELLOW)[1/4] Checking for deprecated imports...$(NC)"; \
	if grep -r "from langchain.llms import" backend/src/ 2>/dev/null | grep -v __pycache__; then \
		echo "$(RED)❌ Found deprecated langchain.llms imports$(NC)"; \
		ERRORS=$$((ERRORS + 1)); \
	else \
		echo "$(GREEN)✓ No deprecated langchain.llms imports$(NC)"; \
	fi; \
	echo ""; \
	echo "$(YELLOW)[2/4] Checking for AgentExecutor usage...$(NC)"; \
	if grep -r "AgentExecutor" backend/src/ 2>/dev/null | grep -v __pycache__; then \
		echo "$(RED)❌ Found AgentExecutor usage (deprecated in v1.x)$(NC)"; \
		ERRORS=$$((ERRORS + 1)); \
	else \
		echo "$(GREEN)✓ No AgentExecutor usage$(NC)"; \
	fi; \
	echo ""; \
	echo "$(YELLOW)[3/4] Checking for legacy type hints...$(NC)"; \
	if grep -r "from typing import.*Optional" backend/src/ 2>/dev/null | grep -v __pycache__; then \
		echo "$(RED)❌ Found Optional imports (use | None syntax)$(NC)"; \
		ERRORS=$$((ERRORS + 1)); \
	else \
		echo "$(GREEN)✓ No legacy Optional imports$(NC)"; \
	fi; \
	echo ""; \
	echo "$(YELLOW)[4/4] Checking for modern agent patterns...$(NC)"; \
	if grep -r "create_react_agent" backend/src/ 2>/dev/null | grep -v __pycache__ | grep "from langgraph.prebuilt"; then \
		echo "$(RED)❌ Found create_react_agent from langgraph.prebuilt (deprecated)$(NC)"; \
		ERRORS=$$((ERRORS + 1)); \
	else \
		echo "$(GREEN)✓ Using modern agent patterns$(NC)"; \
	fi; \
	echo ""; \
	if [ $$ERRORS -eq 0 ]; then \
		echo "$(GREEN)✓ LangChain v1.x compliance: PASS (100%)$(NC)"; \
		echo "$(GREEN)  Full report: LANGCHAIN_V1_COMPLIANCE_REPORT.md$(NC)"; \
	else \
		echo "$(RED)❌ LangChain v1.x compliance: FAIL ($$ERRORS issues)$(NC)"; \
		echo "$(YELLOW)  Review: LANGCHAIN_V1_COMPLIANCE_REPORT.md$(NC)"; \
		exit 1; \
	fi

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
version:  ## Show installed versions (Level 1+2+3 dependencies)
	@echo "$(BLUE)Installed Versions:$(NC)"
	@uv run python --version
	@echo ""
	@echo "$(YELLOW)LangChain Ecosystem (Level 1+2):$(NC)"
	@uv run python -c "import langchain; print(f'  langchain:           {langchain.__version__}')" 2>/dev/null || echo "  langchain: not installed"
	@uv run python -c "import langchain_core; print(f'  langchain-core:      {langchain_core.__version__}')" 2>/dev/null || echo "  langchain-core: not installed"
	@uv run python -c "import langgraph; print(f'  langgraph:           {langgraph.__version__}')" 2>/dev/null || echo "  langgraph: not installed"
	@uv run python -c "import langchain_openai; print(f'  langchain-openai:    {langchain_openai.__version__}')" 2>/dev/null || echo "  langchain-openai: not installed"
	@uv run python -c "import langchain_anthropic; print(f'  langchain-anthropic: {langchain_anthropic.__version__}')" 2>/dev/null || echo "  langchain-anthropic: not installed"
	@echo ""
	@echo "$(YELLOW)Memory & Cache (Level 3a+3c):$(NC)"
	@uv run python -c "import redis; print(f'  redis:               {redis.__version__}')" 2>/dev/null || echo "  redis: not installed"
	@uv run python -c "import graphiti_core; print(f'  graphiti-core:       {graphiti_core.__version__}')" 2>/dev/null || echo "  graphiti-core: not installed"
	@uv run python -c "import neo4j; print(f'  neo4j:               {neo4j.__version__}')" 2>/dev/null || echo "  neo4j: not installed"
	@uv run python -c "import textblob; print(f'  textblob:            {textblob.__version__}')" 2>/dev/null || echo "  textblob: not installed (Level 3c - emotional memory)"
	@echo ""
	@echo "$(YELLOW)Vector Store (Level 2):$(NC)"
	@uv run python -c "import qdrant_client; print(f'  qdrant-client:       {qdrant_client.__version__}')" 2>/dev/null || echo "  qdrant-client: not installed"

# Quick start
quickstart: install-dev version verify  ## Quick start - install everything, show versions, and verify setup
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. If verification failed, copy .env.template to .env and add your API keys"
	@echo "  2. Run 'make verify' again to confirm setup"
	@echo "  3. Run 'make docker-up-dev' to start all services in dev mode"
	@echo "  4. Run 'make test-dev' to test all features (Level 1+2+3)"
	@echo "  5. Run 'make help' to see all available commands"

# Development workflow
dev: install-dev  ## Setup development environment (progressive: currently Level 3 complete)
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo ""
	@echo "$(YELLOW)Current Level: Level 3 COMPLETE (v0.6.0)$(NC)"
	@echo "  ✅ Level 1: ReAct Agent + HITL"
	@echo "  ✅ Level 2: RAG + Chain-of-Thought"
	@echo "  ✅ Level 3: 7-Layer Memory + Advanced Reasoning + Multi-Layer Caching"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Copy .env.template to .env and add your API keys"
	@echo "  2. Run 'make docker-up-dev' to start all services in dev mode"
	@echo "  3. Run 'make test-dev' to run complete test suite"
	@echo "  4. Run 'make all-checks' before committing code"

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
# Memory System Commands (Level 3a)
# ============================================================================

memory-test:  ## Test memory system (Redis + Neo4j connectivity) - auto-detects prod/dev
	@echo "$(BLUE)Testing memory system...$(NC)"
	@REDIS_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-redis' | head -1); \
	NEO4J_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-neo4j' | head -1); \
	if [ -z "$$REDIS_CONTAINER" ] || [ -z "$$NEO4J_CONTAINER" ]; then \
		echo "$(RED)Error: Redis or Neo4j container not running$(NC)"; \
		echo "$(YELLOW)Start containers with: make docker-up (prod) or make docker-up-dev (dev)$(NC)"; \
		exit 1; \
	fi; \
	echo "$(YELLOW)Using containers: $$REDIS_CONTAINER, $$NEO4J_CONTAINER$(NC)"; \
	echo ""; \
	echo "$(YELLOW)[Test 1/3] Redis connectivity$(NC)"; \
	docker exec -it $$REDIS_CONTAINER redis-cli ping || echo "$(RED)Redis connection failed$(NC)"; \
	echo ""; \
	echo "$(YELLOW)[Test 2/3] Neo4j connectivity$(NC)"; \
	docker exec -it $$NEO4J_CONTAINER cypher-shell -u neo4j -p weatherai2025 "RETURN 1" || echo "$(RED)Neo4j connection failed$(NC)"; \
	echo ""; \
	echo "$(YELLOW)[Test 3/4] Running Level 3a memory unit tests$(NC)"; \
	uv run pytest backend/tests/test_memory_l3a.py -v; \
	echo ""; \
	echo "$(YELLOW)[Test 4/4] Running Level 3c memory unit tests$(NC)"; \
	uv run pytest backend/tests/test_memory_l3c.py -v; \
	echo ""; \
	echo "$(GREEN)✓ Memory system tests complete (L3a + L3c)$(NC)"


memory-redis-cli:  ## Open Redis CLI (interactive shell) - auto-detects prod/dev
	@echo "$(BLUE)Opening Redis CLI...$(NC)"
	@REDIS_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-redis' | head -1); \
	if [ -z "$$REDIS_CONTAINER" ]; then \
		echo "$(RED)Error: Redis container not running$(NC)"; \
		echo "$(YELLOW)Start containers with: make docker-up (prod) or make docker-up-dev (dev)$(NC)"; \
		exit 1; \
	fi; \
	echo "$(YELLOW)Using container: $$REDIS_CONTAINER$(NC)"; \
	echo "$(YELLOW)Commands:$(NC)"; \
	echo "  - KEYS *              (list all keys)"; \
	echo "  - GET stm:user_123:*  (get session context)"; \
	echo "  - TTL <key>           (check time-to-live)"; \
	echo "  - EXIT                (quit)"; \
	echo ""; \
	docker exec -it $$REDIS_CONTAINER redis-cli

memory-neo4j-browser:  ## Open Neo4j Browser in default browser
	@echo "$(BLUE)Opening Neo4j Browser...$(NC)"
	@echo "$(YELLOW)URL:$(NC) http://localhost:7474"
	@echo "$(YELLOW)Username:$(NC) neo4j"
	@echo "$(YELLOW)Password:$(NC) weatherai2025"
	@echo ""
	@echo "$(YELLOW)Useful Cypher queries:$(NC)"
	@echo "  MATCH (n) RETURN n LIMIT 25;        // View all nodes"
	@echo "  MATCH (u:User) RETURN u;            // View all users"
	@echo "  MATCH (f:Fact) RETURN f;            // View all facts"
	@echo "  MATCH (e:WeatherEvent) RETURN e;    // View weather events"
	@open http://localhost:7474 2>/dev/null || xdg-open http://localhost:7474 2>/dev/null || echo "$(YELLOW)Please open http://localhost:7474 in your browser$(NC)"

memory-clear:  ## Clear all memory data (Redis + Neo4j) - WARNING: destructive, auto-detects prod/dev
	@echo "$(RED)WARNING: This will delete ALL memory data (Redis + Neo4j)$(NC)"
	@echo "$(YELLOW)Press Ctrl+C within 5 seconds to cancel...$(NC)"
	@sleep 5
	@REDIS_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-redis' | head -1); \
	NEO4J_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-neo4j' | head -1); \
	if [ -z "$$REDIS_CONTAINER" ] || [ -z "$$NEO4J_CONTAINER" ]; then \
		echo "$(RED)Error: Redis or Neo4j container not running$(NC)"; \
		exit 1; \
	fi; \
	echo ""; \
	echo "$(YELLOW)Using containers: $$REDIS_CONTAINER, $$NEO4J_CONTAINER$(NC)"; \
	echo "$(BLUE)Clearing Redis data...$(NC)"; \
	docker exec -it $$REDIS_CONTAINER redis-cli FLUSHALL || echo "$(RED)Redis clear failed$(NC)"; \
	echo ""; \
	echo "$(BLUE)Clearing Neo4j data...$(NC)"; \
	docker exec -it $$NEO4J_CONTAINER cypher-shell -u neo4j -p weatherai2025 "MATCH (n) DETACH DELETE n" || echo "$(RED)Neo4j clear failed$(NC)"; \
	echo ""; \
	echo "$(GREEN)✓ All memory data cleared$(NC)"

memory-stats:  ## Show memory system statistics - auto-detects prod/dev
	@echo "$(BLUE)Memory System Statistics$(NC)"
	@REDIS_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-redis' | head -1); \
	NEO4J_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-neo4j' | head -1); \
	if [ -z "$$REDIS_CONTAINER" ] || [ -z "$$NEO4J_CONTAINER" ]; then \
		echo "$(RED)Error: Redis or Neo4j container not running$(NC)"; \
		exit 1; \
	fi; \
	echo ""; \
	echo "$(YELLOW)Using containers: $$REDIS_CONTAINER, $$NEO4J_CONTAINER$(NC)"; \
	echo ""; \
	echo "$(YELLOW)Redis Stats:$(NC)"; \
	docker exec -it $$REDIS_CONTAINER redis-cli INFO stats | grep -E "total_commands_processed|keyspace_hits|keyspace_misses" || echo "$(RED)Redis stats failed$(NC)"; \
	echo ""; \
	echo "$(YELLOW)Redis Memory Usage:$(NC)"; \
	docker exec -it $$REDIS_CONTAINER redis-cli INFO memory | grep -E "used_memory_human|used_memory_peak_human" || echo "$(RED)Redis memory info failed$(NC)"; \
	echo ""; \
	echo "$(YELLOW)Neo4j Node Count:$(NC)"; \
	docker exec -it $$NEO4J_CONTAINER cypher-shell -u neo4j -p weatherai2025 "MATCH (n) RETURN count(n) as total_nodes" || echo "$(RED)Neo4j node count failed$(NC)"; \
	echo ""; \
	echo "$(YELLOW)Neo4j Relationship Count:$(NC)"; \
	docker exec -it $$NEO4J_CONTAINER cypher-shell -u neo4j -p weatherai2025 "MATCH ()-[r]->() RETURN count(r) as total_relationships" || echo "$(RED)Neo4j relationship count failed$(NC)"

# ============================================================================
# Docker Compose Commands (Level 3a: 6 Services)
# Services: weather-mcp:8080, hurricane-mcp:8081, qdrant:6333,
#           redis:6379, neo4j:7474/7687, weather-ai-api:8000
# ============================================================================

docker-up:  ## Start all Docker containers in PRODUCTION mode (Level 3a: 6 services)
	@echo "$(BLUE)Starting all Docker containers (PRODUCTION MODE)...$(NC)"
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  - weather-mcp:      http://localhost:8080"
	@echo "  - hurricane-mcp:    http://localhost:8081"
	@echo "  - qdrant:           http://localhost:6333  (Level 2 - RAG)"
	@echo "  - redis:            redis://localhost:6379 (Level 3a - Short-term memory)"
	@echo "  - neo4j:            http://localhost:7474  (Level 3a - Long-term memory)"
	@echo "  - weather-ai-api:   http://localhost:8000"
	docker-compose up -d
	@echo "$(GREEN)✓ All containers started$(NC)"
	@echo ""
	@echo "$(YELLOW)Check status:  make docker-ps$(NC)"
	@echo "$(YELLOW)View logs:     make docker-logs$(NC)"
	@echo "$(YELLOW)Load RAG:      make rag-load$(NC)"
	@echo "$(YELLOW)Run tests:     make test-dev$(NC)"

docker-up-dev:  ## Start all Docker containers in DEVELOPMENT mode (hot reload enabled)
	@echo "$(BLUE)Starting all Docker containers (DEVELOPMENT MODE)...$(NC)"
	@echo "$(YELLOW)Dev Features:$(NC)"
	@echo "  ✅ Hot reload enabled (code changes reflect immediately)"
	@echo "  ✅ Source code mounted as volume (no rebuild needed)"
	@echo "  ✅ Debug logging enabled"
	@echo "  ✅ LangSmith tracing enabled"
	@echo "  ✅ Memory system enabled (Redis + Neo4j)"
	@echo ""
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  - weather-mcp:      http://localhost:8080"
	@echo "  - hurricane-mcp:    http://localhost:8081"
	@echo "  - qdrant:           http://localhost:6333"
	@echo "  - redis:            redis://localhost:6379 (Level 3a)"
	@echo "  - neo4j:            http://localhost:7474  (Level 3a)"
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

docker-health:  ## Check health status of all Docker containers - auto-detects prod/dev
	@echo "$(BLUE)Checking health status of all containers...$(NC)"
	@echo ""
	@QDRANT_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-qdrant' | head -1); \
	REDIS_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-redis' | head -1); \
	NEO4J_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-neo4j' | head -1); \
	API_CONTAINER=$$(docker ps --format '{{.Names}}' | grep 'weather-ai-api' | head -1); \
	echo "$(YELLOW)Weather MCP Server (8080):$(NC)"; \
	docker inspect --format='{{.State.Health.Status}}' weather-mcp-server 2>/dev/null || echo "$(RED)Not running$(NC)"; \
	echo ""; \
	echo "$(YELLOW)Hurricane Tracker MCP (8081):$(NC)"; \
	docker inspect --format='{{.State.Health.Status}}' hurricane-tracker-mcp 2>/dev/null || echo "$(RED)Not running$(NC)"; \
	echo ""; \
	echo "$(YELLOW)Qdrant Vector DB (6333):$(NC)"; \
	if [ -n "$$QDRANT_CONTAINER" ]; then \
		docker inspect --format='{{.State.Status}}' $$QDRANT_CONTAINER 2>/dev/null || echo "$(RED)Not running$(NC)"; \
	else \
		echo "$(RED)Not running$(NC)"; \
	fi; \
	echo ""; \
	echo "$(YELLOW)Redis Memory Store (6379):$(NC)"; \
	if [ -n "$$REDIS_CONTAINER" ]; then \
		docker inspect --format='{{.State.Health.Status}}' $$REDIS_CONTAINER 2>/dev/null || echo "$(RED)Not running$(NC)"; \
	else \
		echo "$(RED)Not running$(NC)"; \
	fi; \
	echo ""; \
	echo "$(YELLOW)Neo4j Graph DB (7474/7687):$(NC)"; \
	if [ -n "$$NEO4J_CONTAINER" ]; then \
		docker inspect --format='{{.State.Health.Status}}' $$NEO4J_CONTAINER 2>/dev/null || echo "$(RED)Not running$(NC)"; \
	else \
		echo "$(RED)Not running$(NC)"; \
	fi; \
	echo ""; \
	echo "$(YELLOW)Weather AI API (8000):$(NC)"; \
	if [ -n "$$API_CONTAINER" ]; then \
		docker inspect --format='{{.State.Health.Status}}' $$API_CONTAINER 2>/dev/null || echo "$(RED)Not running$(NC)"; \
	else \
		echo "$(RED)Not running$(NC)"; \
	fi; \
	echo ""; \
	echo "$(GREEN)✓ Health check complete$(NC)"

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
