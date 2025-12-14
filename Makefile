.PHONY: help install install-dev sync verify clean test test-dev lint format type-check security compliance-check run-verify run-agent all-checks quickstart dev update lock version rag-validate rag-load rag-load-curated-only rag-load-skip-validation rag-test memory-test memory-redis-cli memory-neo4j-browser memory-clear memory-stats eval-upload-dataset eval-run-batch eval-check-gates eval-quick eval-category eval-full eval-upload-level6 eval-level6 eval-bleu-rouge eval-snapshot eval-retrieval eval-ragas eval-agentbench docker-up docker-up-dev docker-down docker-down-dev docker-restart docker-restart-dev docker-logs docker-logs-dev docker-ps docker-ps-dev docker-health docker-clean docker-clean-dev observability-status observability-logs grafana-open prometheus-open prometheus-reload loki-logs

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

test-dev:  ## Restart dev environment and run complete test suite (progressive: currently Level 5c complete)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Weather AI Agent - Development Test Suite$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Current Level: Level 5c COMPLETE (v0.10.0)$(NC)"
	@echo "  ✅ Level 1: ReAct Agent + HITL (7 tests)"
	@echo "  ✅ Level 2: RAG + Chain-of-Thought (4 tests)"
	@echo "  ✅ Level 3: 7-Layer Memory + Advanced Reasoning + 3-Layer Caching (15 tests)"
	@echo "  ✅ Level 4: 15-Agent Multi-Agent Orchestration (20 tests)"
	@echo "  ✅ Level 5: Production + Guardrails + Evaluation + Observability (3 tests)"
	@echo "  $(GREEN)Total: 49 test files$(NC)"
	@echo ""
	@echo "$(BLUE)[1/3] Restarting development environment (10 services)...$(NC)"
	@docker-compose -f docker-compose.dev.yml restart
	@echo "$(GREEN)✓ Development environment restarted$(NC)"
	@echo ""
	@echo "$(BLUE)[2/3] Running complete test suite (49 tests)...$(NC)"
	@echo "$(YELLOW)Testing: MCP, RAG, HITL, Memory (L3a+L3c), Reasoning (ToT+GoT),$(NC)"
	@echo "$(YELLOW)         Cache (L1+L2+L3), Multi-Agent, Guardrails, Evaluation$(NC)"
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
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  - View logs:        make docker-logs-dev"
	@echo "  - Observability:    make observability-status"
	@echo "  - Open Grafana:     make grafana-open"

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

run-agent:  ## Run Weather AI Agent Service (use docker-up-dev instead)
	@echo "$(BLUE)Starting Weather AI Agent Service...$(NC)"
	@echo "$(YELLOW)⚠️  Use 'make docker-up-dev' to run the service$(NC)"
	@echo "$(YELLOW)   Current: Level 5c COMPLETE (v0.10.0)$(NC)"
	@echo ""
	@echo "$(GREEN)Quick start:$(NC)"
	@echo "  1. make docker-up-dev   # Start all 10 services"
	@echo "  2. make test-dev        # Run 49 tests"
	@echo "  3. Visit http://localhost:8000/docs"

# Version info
version:  ## Show installed versions (Level 5c: all dependencies)
	@echo "$(BLUE)Installed Versions (Level 5c v0.10.0):$(NC)"
	@uv run python --version
	@echo ""
	@echo "$(YELLOW)LangChain Ecosystem (Level 1+2+3+4):$(NC)"
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
	@echo ""
	@echo "$(YELLOW)Observability (Level 5c):$(NC)"
	@uv run python -c "import prometheus_client; print(f'  prometheus-client:   {prometheus_client.__version__}')" 2>/dev/null || echo "  prometheus-client: not installed"

# Quick start
quickstart: install-dev version verify  ## Quick start - install everything, show versions, and verify setup
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. If verification failed, copy .env.template to .env and add your API keys"
	@echo "  2. Run 'make verify' again to confirm setup"
	@echo "  3. Run 'make docker-up-dev' to start all 10 services in dev mode"
	@echo "  4. Run 'make test-dev' to test all features (Level 1-5c, 49 tests)"
	@echo "  5. Run 'make help' to see all available commands"

# Development workflow
dev: install-dev  ## Setup development environment (progressive: currently Level 5c complete)
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo ""
	@echo "$(YELLOW)Current Level: Level 5c COMPLETE (v0.10.0)$(NC)"
	@echo "  ✅ Level 1: ReAct Agent + HITL"
	@echo "  ✅ Level 2: RAG + Chain-of-Thought"
	@echo "  ✅ Level 3: 7-Layer Memory + Advanced Reasoning + 3-Layer Caching"
	@echo "  ✅ Level 4: 15-Agent Multi-Agent Orchestration"
	@echo "  ✅ Level 5: Production + Guardrails + Evaluation + Observability"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Copy .env.template to .env and add your API keys"
	@echo "  2. Run 'make docker-up-dev' to start all 10 services in dev mode"
	@echo "  3. Run 'make test-dev' to run complete test suite (49 tests)"
	@echo "  4. Run 'make all-checks' before committing code"
	@echo "  5. Run 'make observability-status' to check monitoring stack"

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

rag-load:  ## Load weather knowledge base into Qdrant (curated + Kaggle data)
	@echo "$(BLUE)Building and loading RAG knowledge base...$(NC)"
	@echo "$(YELLOW)This will:$(NC)"
	@echo "  1. Validate Kaggle datasets"
	@echo "  2. Load curated knowledge documents (~20-30 files)"
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

rag-load-curated-only:  ## Load only curated knowledge (skip Kaggle datasets)
	@echo "$(BLUE)Loading curated knowledge only...$(NC)"
	@uv run python -m backend.src.rag.build_knowledge_base --curated-only
	@echo "$(GREEN)✓ Curated knowledge loaded$(NC)"

rag-load-skip-validation:  ## Load knowledge base without validating datasets first
	@echo "$(BLUE)Loading knowledge base (skipping validation)...$(NC)"
	@uv run python -m backend.src.rag.build_knowledge_base --skip-validation
	@echo "$(GREEN)✓ Knowledge base loaded$(NC)"

rag-test:  ## Test RAG retrieval (verify both curated and Kaggle data)
	@echo "$(BLUE)Testing RAG knowledge base retrieval...$(NC)"
	@echo ""
	@echo "$(YELLOW)[Test 1/3] Testing Curated Knowledge - Hurricane Information$(NC)"
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
# Evaluation & Testing Commands (Level 5b: LangSmith + Golden Dataset)
# ============================================================================

eval-upload-dataset:  ## Upload golden dataset to LangSmith (185 test cases: Level 5 + Level 6)
	@echo "$(BLUE)Uploading golden dataset to LangSmith...$(NC)"
	@echo "$(YELLOW)Dataset:$(NC) tests/evaluation/golden_dataset.yaml"
	@echo "$(YELLOW)Categories:$(NC)"
	@echo "  - Simple:    40 test cases (basic weather queries)"
	@echo "  - Complex:   30 test cases (multi-location comparisons)"
	@echo "  - Hurricane: 20 test cases (safety-critical)"
	@echo "  - Edge:      15 test cases (error handling)"
	@echo ""
	uv run python scripts/upload_golden_dataset.py
	@echo ""
	@echo "$(GREEN)✓ Dataset uploaded to LangSmith$(NC)"
	@echo "$(YELLOW)View at: https://smith.langchain.com/datasets$(NC)"

eval-run-batch:  ## Run batch evaluation on full golden dataset (185 cases: Level 5 + Level 6)
	@echo "$(BLUE)Running batch evaluation (185 test cases)...$(NC)"
	@echo "$(YELLOW)This will:$(NC)"
	@echo "  1. Run all 185 test cases through the agent (Level 5: 105, Level 6: 80)"
	@echo "  2. Evaluate using 4-pillar framework + Level 6 metrics"
	@echo "  3. Generate evaluation report"
	@echo "  4. Check quality gates (pass_rate ≥85%, safety=0)"
	@echo ""
	@PYTHONPATH=$(PWD) uv run python scripts/run_batch_evaluation.py
	@echo ""
	@echo "$(GREEN)✓ Batch evaluation complete$(NC)"
	@echo "$(YELLOW)Results saved to: evaluation_results.json$(NC)"

eval-check-gates:  ## Check quality gates from evaluation results
	@echo "$(BLUE)Checking quality gates...$(NC)"
	@echo "$(YELLOW)Thresholds:$(NC)"
	@echo "  - Pass Rate:        ≥85%"
	@echo "  - Effectiveness:    ≥85%"
	@echo "  - Efficiency:       ≥80%"
	@echo "  - Robustness:       ≥80%"
	@echo "  - Safety Violations: 0 (zero tolerance)"
	@echo ""
	@uv run python scripts/check_quality_gates.py
	@echo ""

eval-quick:  ## Quick smoke test (10 random test cases)
	@echo "$(BLUE)Running quick smoke test (10 cases)...$(NC)"
	@PYTHONPATH=$(PWD) uv run python scripts/run_batch_evaluation.py --max-cases=10 --output=evaluation_quick.json
	@echo ""
	@echo "$(GREEN)✓ Quick test complete$(NC)"
	@echo "$(YELLOW)Results: evaluation_quick.json$(NC)"

eval-category:  ## Run evaluation for specific category (use: make eval-category CATEGORY=hurricane)
	@if [ -z "$(CATEGORY)" ]; then \
		echo "$(RED)Error: CATEGORY not specified$(NC)"; \
		echo "$(YELLOW)Level 5: make eval-category CATEGORY=<simple|complex|hurricane|edge>$(NC)"; \
		echo "$(YELLOW)Level 6: make eval-category CATEGORY=<bleu_rouge|snapshot|retrieval|ragas_recall|agentbench>$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running evaluation for category: $(CATEGORY)$(NC)"
	@PYTHONPATH=$(PWD) uv run python scripts/run_batch_evaluation.py --category=$(CATEGORY) --output=evaluation_$(CATEGORY).json
	@echo ""
	@echo "$(GREEN)✓ Category evaluation complete$(NC)"
	@echo "$(YELLOW)Results: evaluation_$(CATEGORY).json$(NC)"

eval-full:  ## Full evaluation pipeline (upload → run → check gates)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Full Evaluation Pipeline (Level 5b + Level 6)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)[1/3] Uploading golden dataset to LangSmith...$(NC)"
	@$(MAKE) eval-upload-dataset
	@echo ""
	@echo "$(YELLOW)[2/3] Running batch evaluation (185 cases)...$(NC)"
	@$(MAKE) eval-run-batch
	@echo ""
	@echo "$(YELLOW)[3/3] Checking quality gates...$(NC)"
	@$(MAKE) eval-check-gates
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Full evaluation pipeline COMPLETE$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  - View results: cat evaluation_results.json"
	@echo "  - View traces:  https://smith.langchain.com"
	@echo "  - Run category: make eval-category CATEGORY=hurricane"
	@echo "  - Run Level 6:  make eval-level6"

# ============================================================================
# Level 6 Evaluation Commands (80 test cases)
# BLEU/ROUGE, Snapshot, Retrieval, RAGAS, AgentBench
# ============================================================================

eval-upload-level6:  ## Upload Level 6 golden dataset to LangSmith (80 test cases)
	@echo "$(BLUE)Uploading Level 6 Golden Dataset to LangSmith...$(NC)"
	@echo "Categories: bleu_rouge (20), snapshot (15), retrieval (20), ragas_recall (10), agentbench (15)"
	@PYTHONPATH=$(PWD) uv run python scripts/upload_golden_dataset.py --level6-only
	@echo "$(GREEN)✓ Level 6 dataset uploaded$(NC)"

eval-level6:  ## Run all Level 6 evaluations (80 test cases)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Level 6 Evaluation (80 test cases)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Categories: BLEU/ROUGE, Snapshot, Retrieval, RAGAS, AgentBench"
	@PYTHONPATH=$(PWD) uv run python scripts/run_level6_evaluation.py
	@echo ""
	@echo "$(GREEN)✓ Level 6 evaluation complete$(NC)"

eval-bleu-rouge:  ## Run BLEU/ROUGE evaluation only (20 test cases)
	@echo "$(BLUE)Running BLEU/ROUGE evaluation...$(NC)"
	@PYTHONPATH=$(PWD) uv run python scripts/run_level6_evaluation.py --eval-type bleu_rouge
	@echo "$(GREEN)✓ BLEU/ROUGE evaluation complete$(NC)"

eval-snapshot:  ## Run Snapshot evaluation only (15 test cases)
	@echo "$(BLUE)Running Snapshot evaluation...$(NC)"
	@PYTHONPATH=$(PWD) uv run python scripts/run_level6_evaluation.py --eval-type snapshot
	@echo "$(GREEN)✓ Snapshot evaluation complete$(NC)"

eval-retrieval:  ## Run Retrieval metrics evaluation only (20 test cases)
	@echo "$(BLUE)Running Retrieval evaluation...$(NC)"
	@PYTHONPATH=$(PWD) uv run python scripts/run_level6_evaluation.py --eval-type retrieval
	@echo "$(GREEN)✓ Retrieval evaluation complete$(NC)"

eval-ragas:  ## Run RAGAS Context Recall evaluation only (10 test cases)
	@echo "$(BLUE)Running RAGAS evaluation...$(NC)"
	@PYTHONPATH=$(PWD) uv run python scripts/run_level6_evaluation.py --eval-type ragas_recall
	@echo "$(GREEN)✓ RAGAS evaluation complete$(NC)"

eval-agentbench:  ## Run AgentBench evaluation only (15 test cases)
	@echo "$(BLUE)Running AgentBench evaluation...$(NC)"
	@PYTHONPATH=$(PWD) uv run python scripts/run_level6_evaluation.py --eval-type agentbench
	@echo "$(GREEN)✓ AgentBench evaluation complete$(NC)"

# ============================================================================
# Docker Compose Commands (Level 5c: 10 Services)
# Core Services: weather-mcp:8080, hurricane-mcp:8081, weather-ai-api:8000
# Databases: qdrant:6333, redis:6379, neo4j:7474/7687, postgres:5432
# Observability: prometheus:9090, grafana:3001, loki:3100
# ============================================================================

docker-up:  ## Start all Docker containers in PRODUCTION mode (Level 5c: 10 services)
	@echo "$(BLUE)Starting all Docker containers (PRODUCTION MODE - Level 5c)...$(NC)"
	@echo "$(YELLOW)Core Services:$(NC)"
	@echo "  - weather-mcp:      http://localhost:8080"
	@echo "  - hurricane-mcp:    http://localhost:8081"
	@echo "  - weather-ai-api:   http://localhost:8000"
	@echo "$(YELLOW)Databases:$(NC)"
	@echo "  - qdrant:           http://localhost:6333  (Level 2 - RAG)"
	@echo "  - redis:            redis://localhost:6379 (Level 3a - Memory + Cache)"
	@echo "  - neo4j:            http://localhost:7474  (Level 3a - Long-term memory)"
	@echo "  - postgres:         localhost:5432         (Level 3c - Procedural memory)"
	@echo "$(YELLOW)Observability (Level 5c):$(NC)"
	@echo "  - prometheus:       http://localhost:9090  (Metrics)"
	@echo "  - grafana:          http://localhost:3001  (Dashboards)"
	@echo "  - loki:             http://localhost:3100  (Logs)"
	docker-compose up -d
	@echo "$(GREEN)✓ All containers started$(NC)"
	@echo ""
	@echo "$(YELLOW)Check status:        make docker-ps$(NC)"
	@echo "$(YELLOW)View logs:           make docker-logs$(NC)"
	@echo "$(YELLOW)Observability:       make observability-status$(NC)"
	@echo "$(YELLOW)Open Grafana:        make grafana-open$(NC)"

docker-up-dev:  ## Start all Docker containers in DEVELOPMENT mode (Level 5c: 10 services + hot reload)
	@echo "$(BLUE)Starting all Docker containers (DEVELOPMENT MODE - Level 5c)...$(NC)"
	@echo "$(YELLOW)Dev Features:$(NC)"
	@echo "  ✅ Hot reload enabled (code changes reflect immediately)"
	@echo "  ✅ Source code mounted as volume (no rebuild needed)"
	@echo "  ✅ Debug logging enabled"
	@echo "  ✅ LangSmith tracing enabled"
	@echo "  ✅ Memory system enabled (Redis + Neo4j)"
	@echo "  ✅ Observability stack (Prometheus + Grafana + Loki)"
	@echo ""
	@echo "$(YELLOW)Core Services:$(NC)"
	@echo "  - weather-mcp:      http://localhost:8080"
	@echo "  - hurricane-mcp:    http://localhost:8081"
	@echo "  - weather-ai-api:   http://localhost:8000 (DEV MODE)"
	@echo "$(YELLOW)Databases:$(NC)"
	@echo "  - qdrant:           http://localhost:6333  (Level 2 - RAG)"
	@echo "  - redis:            redis://localhost:6379 (Level 3a - Memory + Cache)"
	@echo "  - neo4j:            http://localhost:7474  (Level 3a - Long-term memory)"
	@echo "  - postgres:         localhost:5432         (Level 3c - Procedural memory)"
	@echo "$(YELLOW)Observability (Level 5c):$(NC)"
	@echo "  - prometheus:       http://localhost:9090  (Metrics)"
	@echo "  - grafana:          http://localhost:3001  (Dashboards - admin/weatherai2025)"
	@echo "  - loki:             http://localhost:3100  (Logs)"
	docker-compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)✓ All containers started in DEV mode$(NC)"
	@echo ""
	@echo "$(YELLOW)Check status:        make docker-ps-dev$(NC)"
	@echo "$(YELLOW)View logs:           make docker-logs-dev$(NC)"
	@echo "$(YELLOW)Observability:       make observability-status$(NC)"
	@echo "$(YELLOW)Open Grafana:        make grafana-open$(NC)"
	@echo "$(YELLOW)Stop:                make docker-down-dev$(NC)"

docker-down:  ## Stop and remove all PRODUCTION Docker containers
	@echo "$(BLUE)Stopping all Docker containers (PRODUCTION)...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ All containers stopped and removed$(NC)"

docker-down-dev:  ## Stop and remove all DEVELOPMENT Docker containers
	@echo "$(BLUE)Stopping all Docker containers (DEVELOPMENT)...$(NC)"
	docker-compose -f docker-compose.dev.yml down
	@echo "$(GREEN)✓ All dev containers stopped and removed$(NC)"

docker-rebuild-dev:  ## Rebuild and restart DEVELOPMENT containers (use after Dockerfile/docker-compose changes)
	@echo "$(BLUE)Rebuilding all Docker containers (DEVELOPMENT)...$(NC)"
	@echo "$(YELLOW)This will rebuild images and restart all containers$(NC)"
	docker-compose -f docker-compose.dev.yml up -d --build
	@echo "$(GREEN)✓ All dev containers rebuilt and started$(NC)"
	@echo ""
	@echo "$(YELLOW)Check status:        make docker-ps-dev$(NC)"
	@echo "$(YELLOW)View logs:           make docker-logs-dev$(NC)"

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

docker-health:  ## Check health status of all Docker containers (Level 5c: 10 services)
	@echo "$(BLUE)Checking health status of all containers (Level 5c)...$(NC)"
	@echo ""
	@echo "$(YELLOW)=== Core Services ===$(NC)"
	@echo "Weather MCP Server (8080):"; \
	docker inspect --format='  {{.State.Health.Status}}' weather-mcp-server 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo "Hurricane Tracker MCP (8081):"; \
	docker inspect --format='  {{.State.Health.Status}}' hurricane-tracker-mcp 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo "Weather AI API (8000):"; \
	docker inspect --format='  {{.State.Health.Status}}' weather-ai-api 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo ""
	@echo "$(YELLOW)=== Databases ===$(NC)"
	@echo "Qdrant Vector DB (6333):"; \
	docker inspect --format='  {{.State.Status}}' weather-ai-qdrant 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo "Redis Memory Store (6379):"; \
	docker inspect --format='  {{.State.Health.Status}}' weather-ai-redis 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo "Neo4j Graph DB (7474/7687):"; \
	docker inspect --format='  {{.State.Health.Status}}' weather-ai-neo4j 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo "PostgreSQL (5432):"; \
	docker inspect --format='  {{.State.Health.Status}}' weather-ai-postgres 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo ""
	@echo "$(YELLOW)=== Observability (Level 5c) ===$(NC)"
	@echo "Prometheus (9090):"; \
	docker inspect --format='  {{.State.Health.Status}}' weather-ai-prometheus 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo "Grafana (3001):"; \
	docker inspect --format='  {{.State.Health.Status}}' weather-ai-grafana 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo "Loki (3100):"; \
	docker inspect --format='  {{.State.Health.Status}}' weather-ai-loki 2>/dev/null || echo "  $(RED)Not running$(NC)"
	@echo ""
	@echo "$(GREEN)✓ Health check complete$(NC)"
	@echo ""
	@echo "$(YELLOW)Quick Links:$(NC)"
	@echo "  - API Docs:   http://localhost:8000/docs"
	@echo "  - Grafana:    http://localhost:3001"
	@echo "  - Prometheus: http://localhost:9090"

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

# ============================================================================
# Observability Commands (Level 5c: Prometheus + Grafana + Loki)
# ============================================================================

observability-status:  ## Show status of observability stack (Prometheus, Grafana, Loki)
	@echo "$(BLUE)Observability Stack Status (Level 5c)$(NC)"
	@echo ""
	@echo "$(YELLOW)Prometheus (Metrics):$(NC)"
	@PROMETHEUS_STATUS=$$(docker inspect --format='{{.State.Health.Status}}' weather-ai-prometheus 2>/dev/null || echo "not running"); \
	if [ "$$PROMETHEUS_STATUS" = "healthy" ]; then \
		echo "  $(GREEN)✓ Status: $$PROMETHEUS_STATUS$(NC)"; \
		echo "  $(GREEN)✓ URL: http://localhost:9090$(NC)"; \
		curl -s http://localhost:9090/api/v1/targets 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'  ✓ Targets: {len(d.get(\"data\", {}).get(\"activeTargets\", []))} active')" 2>/dev/null || echo "  ⚠️ Cannot fetch targets"; \
	else \
		echo "  $(RED)✗ Status: $$PROMETHEUS_STATUS$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Grafana (Dashboards):$(NC)"
	@GRAFANA_STATUS=$$(docker inspect --format='{{.State.Health.Status}}' weather-ai-grafana 2>/dev/null || echo "not running"); \
	if [ "$$GRAFANA_STATUS" = "healthy" ]; then \
		echo "  $(GREEN)✓ Status: $$GRAFANA_STATUS$(NC)"; \
		echo "  $(GREEN)✓ URL: http://localhost:3001$(NC)"; \
		echo "  $(GREEN)✓ Login: admin / weatherai2025$(NC)"; \
	else \
		echo "  $(RED)✗ Status: $$GRAFANA_STATUS$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Loki (Logs):$(NC)"
	@LOKI_STATUS=$$(docker inspect --format='{{.State.Health.Status}}' weather-ai-loki 2>/dev/null || echo "not running"); \
	if [ "$$LOKI_STATUS" = "healthy" ]; then \
		echo "  $(GREEN)✓ Status: $$LOKI_STATUS$(NC)"; \
		echo "  $(GREEN)✓ URL: http://localhost:3100 (via Grafana)$(NC)"; \
	else \
		echo "  $(RED)✗ Status: $$LOKI_STATUS$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Quick Links:$(NC)"
	@echo "  - Grafana:    http://localhost:3001"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - LangSmith:  https://smith.langchain.com"

observability-logs:  ## View logs from observability stack
	@echo "$(BLUE)Observability Stack Logs$(NC)"
	@docker-compose logs -f prometheus grafana loki 2>/dev/null || \
	docker-compose -f docker-compose.dev.yml logs -f prometheus grafana loki

grafana-open:  ## Open Grafana in browser
	@echo "$(BLUE)Opening Grafana...$(NC)"
	@echo "$(YELLOW)URL: http://localhost:3001$(NC)"
	@echo "$(YELLOW)Login: admin / weatherai2025$(NC)"
	@echo ""
	@echo "$(YELLOW)Dashboards:$(NC)"
	@echo "  - MCP Health:        http://localhost:3001/d/mcp-health"
	@echo "  - Agent Performance: http://localhost:3001/d/agent-performance"
	@echo "  - Cache Metrics:     http://localhost:3001/d/cache-metrics"
	@open http://localhost:3001 2>/dev/null || xdg-open http://localhost:3001 2>/dev/null || echo "$(YELLOW)Please open http://localhost:3001 in your browser$(NC)"

prometheus-open:  ## Open Prometheus in browser
	@echo "$(BLUE)Opening Prometheus...$(NC)"
	@echo "$(YELLOW)URL: http://localhost:9090$(NC)"
	@echo ""
	@echo "$(YELLOW)Useful queries:$(NC)"
	@echo "  - MCP latency:    histogram_quantile(0.95, sum(rate(mcp_request_latency_seconds_bucket[5m])) by (le))"
	@echo "  - Cache hit rate: sum(rate(cache_hits_total[5m])) / sum(rate(cache_requests_total[5m]))"
	@echo "  - Agent cost:     sum(rate(agent_query_cost_dollars[1h]))"
	@open http://localhost:9090 2>/dev/null || xdg-open http://localhost:9090 2>/dev/null || echo "$(YELLOW)Please open http://localhost:9090 in your browser$(NC)"

prometheus-reload:  ## Reload Prometheus configuration (hot reload)
	@echo "$(BLUE)Reloading Prometheus configuration...$(NC)"
	@curl -X POST http://localhost:9090/-/reload 2>/dev/null && \
		echo "$(GREEN)✓ Prometheus configuration reloaded$(NC)" || \
		echo "$(RED)✗ Failed to reload Prometheus (is it running?)$(NC)"

loki-logs:  ## Query recent logs from Loki
	@echo "$(BLUE)Querying recent logs from Loki...$(NC)"
	@echo "$(YELLOW)Last 100 log entries from weather-ai-api:$(NC)"
	@curl -s "http://localhost:3100/loki/api/v1/query_range" \
		--data-urlencode 'query={container_name="weather-ai-api"}' \
		--data-urlencode 'limit=100' 2>/dev/null | \
		python3 -c "import json,sys; d=json.load(sys.stdin); [print(r['values'][0][1]) for r in d.get('data', {}).get('result', []) if r.get('values')]" 2>/dev/null || \
		echo "$(YELLOW)Note: Loki log query requires promtail or docker logging driver.$(NC)"
	@echo ""
	@echo "$(YELLOW)View logs in Grafana: http://localhost:3001/explore?datasource=Loki$(NC)"
