# Changelog

All notable changes to the Weather AI Agent Service will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Level 1: ReAct pattern + HITL approval (v0.2.0)

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

**Current Version**: 0.1.0
**Last Updated**: 2025-12-01
