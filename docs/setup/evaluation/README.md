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

### Golden Dataset (185 Test Cases)

Located at: `tests/evaluation/golden_dataset.yaml`

**Level 5 Categories (105 cases)**:

| Category | Test Cases | Focus |
|----------|------------|-------|
| **Simple** | 40 | Basic weather queries |
| **Complex** | 30 | Multi-location, analysis |
| **Hurricane** | 20 | Safety-critical queries |
| **Edge** | 15 | Error handling, invalid inputs |

**Level 6 Categories (80 cases)**:

| Category | Test Cases | Focus |
|----------|------------|-------|
| **BLEU/ROUGE** | 20 | Text generation quality |
| **Snapshot** | 15 | Regression detection |
| **Retrieval** | 20 | MRR, NDCG, MAP, Precision@k |
| **RAGAS Recall** | 10 | Context recall validation |
| **AgentBench** | 15 | Task-specific accuracy |

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

## ⚡ Quick Commands (Updated for Level 6)

### Recommended: Makefile Automation

```bash
# 🚀 Full pipeline (upload + run + check) - 185 cases
make eval-full

# Upload golden dataset to LangSmith (185 test cases)
make eval-upload-dataset

# Run batch evaluation (185 cases: Level 5 + Level 6)
make eval-run-batch

# Check quality gates
make eval-check-gates

# Quick smoke test (10 cases, ~2-3 minutes)
make eval-quick

# Test specific category (Level 5)
make eval-category CATEGORY=hurricane
make eval-category CATEGORY=simple
make eval-category CATEGORY=complex
make eval-category CATEGORY=edge

# Test specific category (Level 6)
make eval-category CATEGORY=bleu_rouge
make eval-category CATEGORY=snapshot
make eval-category CATEGORY=retrieval
make eval-category CATEGORY=ragas_recall
make eval-category CATEGORY=agentbench
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
2. ✅ Uploads golden dataset to LangSmith (185 test cases)
3. ✅ Runs batch evaluation (185 test cases: Level 5 + Level 6)
4. ✅ Checks all quality gates (including Level 6 thresholds)
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

**Documentation Version**: 3.0.0 ✨ (Level 6: Golden Dataset Expansion + Advanced Metrics)
**Last Updated**: 2025-12-13

**What's New in v3.0.0**:
- ✅ Golden dataset expanded from 105 to 185 test cases
- ✅ Level 6 categories: BLEU/ROUGE, Snapshot, Retrieval, RAGAS, AgentBench
- ✅ Level 6 quality thresholds (BLEU ≥0.30, MRR ≥0.70, Context Recall ≥0.85)
- ✅ `eval-category` supports all Level 5 + Level 6 categories
- ✅ Full pipeline runs all 185 cases with Level 6 metrics

**What's in v2.0.0**:
- ✅ Automated evaluation with Makefile commands (`make eval-full`)
- ✅ GitHub Actions CI/CD workflow pre-configured
- ✅ Batch evaluation scripts with quality gate enforcement
- ✅ PR comments with detailed evaluation reports
- ✅ One-command full pipeline: `make eval-full`
