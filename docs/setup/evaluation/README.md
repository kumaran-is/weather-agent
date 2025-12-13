# Weather AI Agent - Evaluation & Testing Documentation

Welcome to the comprehensive evaluation and testing documentation for the Weather AI Agent Service. This guide covers everything from initial setup to continuous monitoring.

---

## Quick Start

| Document | Time Required | Purpose |
|----------|---------------|---------|
| [01-LANGSMITH_EVALUATION_SETUP](./01-LANGSMITH_EVALUATION_SETUP.md) | 15-20 min | Configure LangSmith, upload golden dataset |
| [02-GOLDEN_DATASET_TESTING](./02-GOLDEN_DATASET_TESTING.md) | 30-45 min | Run tests, understand 4-pillar scoring |
| [03-GUARDRAILS_TESTING](./03-GUARDRAILS_TESTING.md) | 30-45 min | Test security, safety, compliance |
| [04-MONITORING_RESULTS](./04-MONITORING_RESULTS.md) | 45-60 min | Set up dashboards, alerts, tracking |

**Total Time**: ~2-3 hours for complete setup

---

## Progressive Testing Path

Follow these documents in order for a complete testing setup:

```
Step 1: LangSmith Setup
   ↓
Step 2: Golden Dataset Testing
   ↓
Step 3: Guardrails Testing
   ↓
Step 4: Monitoring & Results
```

---

## What's Included

### Golden Dataset (105 Test Cases)

Located at: `tests/evaluation/golden_dataset.yaml`

| Category | Test Cases | Focus |
|----------|------------|-------|
| **Simple** | 40 | Basic weather queries |
| **Complex** | 30 | Multi-location, analysis |
| **Hurricane** | 20 | Safety-critical queries |
| **Edge** | 15 | Error handling, invalid inputs |

### 4-Pillar Evaluation Framework

| Pillar | Weight | Threshold | Focus |
|--------|--------|-----------|-------|
| **Effectiveness** | 40% | ≥0.80 | Answer correctness |
| **Efficiency** | 20% | ≥0.70 | Tool usage, latency |
| **Robustness** | 20% | ≥0.70 | Edge case handling |
| **Safety** | 20% | =1.00 | Zero-tolerance |

### 12-Layer Guardrails System

1. Input Validation
2. Authentication/Authorization
3. PII/PHI Protection
4. Prompt Injection Prevention
5. Content Filtering
6. Hallucination Detection
7. Bias Mitigation
8. Output Validation
9. Audit Logging
10. Rate Limiting
11. Encryption
12. Compliance Reporting

---

## Quality Gates

All gates must pass before deployment:

```
Pass Rate:      ≥85%
Effectiveness:  ≥0.85
Efficiency:     ≥0.80
Robustness:     ≥0.80
Safety:         100% (zero violations)
```

---

## ⚡ Quick Commands (NEW - Automated)

### Recommended: Makefile Automation

```bash
# 🚀 Full pipeline (upload + run + check)
make eval-full

# Upload golden dataset to LangSmith
make eval-upload-dataset

# Run batch evaluation (105 cases)
make eval-run-batch

# Check quality gates
make eval-check-gates

# Quick smoke test (10 cases, ~2-3 minutes)
make eval-quick

# Test specific category
make eval-category CATEGORY=hurricane
```

### Alternative: Python Scripts

```bash
# Upload dataset
uv run python scripts/upload_golden_dataset.py

# Run batch evaluation
uv run python scripts/run_batch_evaluation.py
uv run python scripts/run_batch_evaluation.py --max-cases=10 --category=hurricane

# Check quality gates
uv run python scripts/check_quality_gates.py
uv run python scripts/check_quality_gates.py --pass-rate=0.90

# Test guardrails
uv run pytest tests/guardrails/ -v
```

---

## CI/CD Integration (NEW - Automated)

✅ **GitHub Actions workflow is pre-configured** at `.github/workflows/evaluation.yml`

**What it does**:
1. ✅ Runs automatically on every PR and push to main/develop
2. ✅ Uploads golden dataset to LangSmith
3. ✅ Runs batch evaluation (105 test cases)
4. ✅ Checks all quality gates
5. ✅ **Blocks merge if gates fail** 🛑
6. ✅ Comments on PR with detailed results
7. ✅ Uploads evaluation results as artifacts

**How to enable**:
1. Add GitHub secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LANGCHAIN_API_KEY`
2. Push code to a branch
3. Create PR → workflow runs automatically
4. View results in PR comment

**Key Features**:
- ⏱️ Timeout: 60 minutes
- 🐳 Docker services: Starts Redis, Neo4j, PostgreSQL, Qdrant
- 📊 Quality gates: Pass rate ≥85%, Safety = 0
- 💬 PR comments: Detailed evaluation report
- 📦 Artifacts: Results saved for 30 days

---

## Support

- **LangSmith Issues**: Check [01-LANGSMITH_EVALUATION_SETUP.md](./01-LANGSMITH_EVALUATION_SETUP.md#troubleshooting)
- **Test Failures**: See [02-GOLDEN_DATASET_TESTING.md](./02-GOLDEN_DATASET_TESTING.md#understanding-results)
- **Guardrail Alerts**: Review [03-GUARDRAILS_TESTING.md](./03-GUARDRAILS_TESTING.md#troubleshooting)
- **Dashboard Issues**: Check [04-MONITORING_RESULTS.md](./04-MONITORING_RESULTS.md#interpreting-results)

---


**What's New in v2.0.0**:
- ✅ Automated evaluation with Makefile commands (`make eval-full`)
- ✅ GitHub Actions CI/CD workflow pre-configured
- ✅ Batch evaluation scripts with quality gate enforcement
- ✅ PR comments with detailed evaluation reports
- ✅ One-command full pipeline: `make eval-full`
