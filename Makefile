.PHONY: help install install-dev sync verify clean test lint format type-check security run-verify run-agent all-checks quickstart dev update lock version

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
