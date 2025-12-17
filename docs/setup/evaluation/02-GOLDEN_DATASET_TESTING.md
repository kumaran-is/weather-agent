# Step 2: Golden Dataset Testing Guide

**Document**: 2 of 4 (Progressive Testing Series)
**Purpose**: Run and evaluate golden dataset tests with 4-pillar scoring + Level 6 metrics
**Prerequisites**: Complete [01-LANGSMITH_EVALUATION_SETUP.md](./01-LANGSMITH_EVALUATION_SETUP.md)
**Time Required**: 30-45 minutes
**Version**: 2.0.0 (Updated for Level 6)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Test Categories](#test-categories)
3. [Level 6 Evaluation Types](#level-6-evaluation-types) (NEW)
4. [Running Tests Locally](#running-tests-locally)
5. [Running Tests with LangSmith](#running-tests-with-langsmith)
6. [Level 6 Specific Commands](#level-6-specific-commands) (NEW)
7. [Understanding Results](#understanding-results)
8. [CI/CD Integration](#cicd-integration)
9. [Progressive Testing Strategy](#progressive-testing-strategy)

---

## Overview

### What is Golden Dataset Testing?

Golden dataset testing validates your AI agent against a curated set of **185 test cases** across different scenarios:

- **Known Paths**: Expected queries the agent should handle well
- **Unknown Paths**: Edge cases, ambiguous inputs, errors
- **Safety-Critical**: Hurricane/evacuation queries requiring accuracy
- **Guardrail Tests**: Security, PII, injection attempts

### 4-Pillar Evaluation Framework

Each test case is scored on 4 pillars:

| Pillar | Weight | Focus | Pass Threshold |
|--------|--------|-------|----------------|
| **Effectiveness** | 40% | Answer correctness (LLM-as-Judge) | ≥0.80 |
| **Efficiency** | 20% | Optimal tool usage, latency, tokens | ≥0.70 |
| **Robustness** | 20% | Edge case handling, error recovery | ≥0.70 |
| **Safety** | 20% | Zero-tolerance for violations | =1.00 |

**Overall Pass**: `effectiveness≥0.80 AND efficiency≥0.70 AND robustness≥0.70 AND safety=1.0`

---

## Test Categories

### Category 1: Simple Queries (40 cases)

Basic weather lookups with single location.

```yaml
# Example: weather_simple_001
query: "What's the weather in Miami?"
expected_tools: ["get_current_weather"]
expected_answer_contains: ["Miami", "temperature", "°F"]
success_criteria:
  effectiveness: ">0.9"
  efficiency: "1.0"
```

**Test Focus**:
- Single tool call
- Correct location parsing
- Complete weather information

### Category 2: Complex Queries (30 cases)

Multi-location, comparison, and analysis queries.

```yaml
# Example: complex_compare_001
query: "Compare weather in Miami and Tampa today"
expected_tools: ["get_current_weather", "get_current_weather"]
expected_answer_contains: ["Miami", "Tampa"]
success_criteria:
  effectiveness: ">0.85"
  efficiency: ">0.8"
```

**Test Focus**:
- Multiple tool calls
- Comparison reasoning
- Context handling

### Category 3: Hurricane Queries (20 cases) - SAFETY CRITICAL

Hurricane tracking, evacuation guidance, category validation.

```yaml
# Example: hurricane_category_001
query: "What category is a hurricane with 145 mph winds?"
expected_answer: "Category 4"
expected_answer_contains: ["Category 4"]
is_safety_critical: true
success_criteria:
  effectiveness: ">0.95"
  safety: "1.0"
```

**Test Focus**:
- Saffir-Simpson accuracy
- Evacuation guidance safety
- Time specificity (EDT/UTC)

### Category 4: Edge Cases (15 cases)

Invalid inputs, ambiguous queries, error handling.

```yaml
# Example: edge_invalid_location_001
query: "Weather in asdfghjkl?"
expected_answer_contains: ["couldn't find", "valid"]
is_edge_case: true
success_criteria:
  robustness: "1.0"
```

**Test Focus**:
- Graceful error handling
- Clarification requests
- No hallucination on invalid input

---

## Level 6 Evaluation Types

Level 6 adds 80 advanced evaluation test cases for comprehensive AI quality measurement.

### Category 5: BLEU/ROUGE (20 cases) - Text Generation Quality

Reference-based text generation quality metrics.

```yaml
# Example: bleu_rouge_001
query: "What is the current weather in Miami?"
category: "bleu_rouge"
reference_answer: "Miami currently has sunny weather with temperatures around 85°F..."
eval_type: "generation"
success_criteria:
  bleu_score: ">0.4"
  rouge_1: ">0.5"
  rouge_l: ">0.45"
```

**Test Focus**:
- Text generation quality
- Answer similarity to reference
- Key information inclusion

### Category 6: Snapshot (15 cases) - Regression Detection

Baseline comparison for regression detection.

```yaml
# Example: snapshot_001
query: "Current weather in Miami"
category: "snapshot"
snapshot_baseline: "Miami is currently experiencing sunny conditions with temperatures..."
eval_type: "snapshot"
success_criteria:
  snapshot_similarity: ">0.85"
  regression_threshold: "0.10"
```

**Test Focus**:
- Consistency across versions
- Regression detection
- Format stability

### Category 7: Retrieval (20 cases) - Information Retrieval Quality

MRR, NDCG, MAP, Precision@k, Recall@k metrics.

```yaml
# Example: retrieval_001
query: "What is the Saffir-Simpson scale?"
category: "retrieval"
eval_type: "retrieval"
relevant_docs:
  - doc_id: "hurricane_scale_001"
    title: "Saffir-Simpson Hurricane Wind Scale"
    relevance: 3  # Highly relevant
  - doc_id: "hurricane_prep_001"
    title: "Hurricane Preparedness Guide"
    relevance: 2  # Moderately relevant
success_criteria:
  mrr: ">0.8"
  ndcg_at_5: ">0.75"
  precision_at_3: ">0.7"
  recall_at_5: ">0.8"
```

**Test Focus**:
- Retrieval accuracy (MRR, NDCG)
- Precision/Recall at k
- Document relevance ranking

### Category 8: RAGAS Context Recall (10 cases)

Ground truth context validation using RAGAS framework.

```yaml
# Example: ragas_recall_001
query: "What are the hurricane categories and their wind speeds?"
category: "ragas_recall"
eval_type: "ragas_context_recall"
ground_truth_info:
  - "Category 1: 74-95 mph"
  - "Category 2: 96-110 mph"
  - "Category 3: 111-129 mph"
  - "Category 4: 130-156 mph"
  - "Category 5: 157+ mph"
success_criteria:
  context_recall: ">0.9"
```

**Test Focus**:
- Ground truth coverage
- Context recall accuracy
- Factual completeness

### Category 9: AgentBench (15 cases) - Task-Specific Accuracy

Structured output and task execution validation.

```yaml
# Example: agentbench_001
query: "What is the temperature in Miami in Celsius?"
category: "agentbench"
eval_type: "agentbench"
task_spec:
  task_type: "unit_conversion"
  expected_format: "temperature in Celsius"
expected_result:
  must_contain: ["°C", "Celsius"]
success_criteria:
  task_accuracy: ">0.95"
```

**Test Focus**:
- Task execution accuracy
- Structured output format
- Tool usage correctness

### Level 6 Quality Thresholds

| Eval Type | Metric | Threshold |
|-----------|--------|-----------|
| **BLEU/ROUGE** | min_bleu | 0.30 |
| | min_rouge_1 | 0.40 |
| | min_rouge_l | 0.35 |
| **Snapshot** | min_similarity | 0.75 |
| | regression_threshold | 0.10 |
| **Retrieval** | min_mrr | 0.70 |
| | min_ndcg_at_5 | 0.65 |
| | min_precision_at_3 | 0.60 |
| | min_recall_at_5 | 0.70 |
| | min_map | 0.65 |
| **RAGAS** | min_context_recall | 0.85 |
| **AgentBench** | min_task_accuracy | 0.85 |
| | min_tool_accuracy | 0.90 |

---

## Running Tests Locally

### Step 2.1: Quick Test (5 cases)

Run a quick validation with a subset of tests:

```bash
cd /home/user/weather-ai-agent-service

# Set environment variables
export OPENAI_API_KEY=sk-your-key
export MCP_WEATHER_SERVER_URL=http://localhost:8080
export MCP_HURRICANE_SERVER_URL=http://localhost:8081

# Run quick test (5 cases)
uv run pytest tests/evaluation/ -v -k "test_golden" --max-cases=5

# Alternative: Run with the golden dataset runner directly
uv run python -c "
import asyncio
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner

async def quick_test():
    runner = GoldenDatasetRunner()
    results = await runner.run_all(max_cases=5)
    print(runner.generate_report(results))

asyncio.run(quick_test())
"
```

### Step 2.2: Category-Specific Tests

Run tests for a specific category:

```bash
# Simple queries only
uv run python -c "
import asyncio
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner

async def test_simple():
    runner = GoldenDatasetRunner()
    results = await runner.run_category('simple')
    print(runner.generate_report(results))
    return results.pass_rate >= 0.85

asyncio.run(test_simple())
"

# Hurricane queries only (safety-critical)
uv run python -c "
import asyncio
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner

async def test_hurricane():
    runner = GoldenDatasetRunner()
    results = await runner.run_category('hurricane')
    print(runner.generate_report(results))
    # Zero safety violations required
    assert results.safety_violations == 0, 'Safety violations detected!'

asyncio.run(test_hurricane())
"

# Edge cases only
uv run python -c "
import asyncio
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner

async def test_edge():
    runner = GoldenDatasetRunner()
    results = await runner.run_category('edge')
    print(runner.generate_report(results))

asyncio.run(test_edge())
"
```

### Step 2.3: Full Test Suite

Run all 105 test cases:

```bash
# Full evaluation (takes 10-15 minutes)
uv run python -c "
import asyncio
import json
from datetime import datetime
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner

async def full_evaluation():
    runner = GoldenDatasetRunner()

    print('Starting full golden dataset evaluation...')
    print(f'Timestamp: {datetime.now().isoformat()}')
    print()

    results = await runner.run_all()

    # Print report
    print(runner.generate_report(results))

    # Check quality gates
    gates = runner.check_quality_gates(results)
    print()
    print('Quality Gate Result:', gates['recommendation'])

    # Save results to file
    with open('evaluation_results.json', 'w') as f:
        json.dump(results.model_dump(), f, indent=2, default=str)
    print('Results saved to: evaluation_results.json')

    return gates['all_passed']

asyncio.run(full_evaluation())
"
```

---

## Running Tests with LangSmith

### ⚡ Quick Start: Automated Evaluation (NEW - Recommended)

**RECOMMENDED**: Use the new automation commands for simplified workflow:

```bash
# 1️⃣ Upload golden dataset to LangSmith (one-time setup)
make eval-upload-dataset

# 2️⃣ Run batch evaluation (105 test cases)
make eval-run-batch

# 3️⃣ Check quality gates
make eval-check-gates

# 🚀 ALL-IN-ONE: Full pipeline (upload + run + check)
make eval-full
```

**Quick Commands**:
```bash
# Quick smoke test (10 random cases, ~2-3 minutes)
make eval-quick

# Test specific category
make eval-category CATEGORY=hurricane
make eval-category CATEGORY=simple
make eval-category CATEGORY=complex
make eval-category CATEGORY=edge
```

**Output Files**:
- `evaluation_results.json` - Full batch evaluation results
- `evaluation_quick.json` - Quick test results
- `evaluation_<category>.json` - Category-specific results

### Step 3.1: Upload Golden Dataset to LangSmith

Before running batch evaluation, upload the golden dataset to LangSmith:

```bash
# Upload 105 test cases to LangSmith
make eval-upload-dataset
```

**What this does**:
1. Loads `tests/evaluation/golden_dataset.yaml`
2. Creates/updates `weather-ai-golden-dataset` in LangSmith
3. Uploads all 105 test cases with:
   - Test inputs (query, category)
   - Expected outputs (expected_tools, expected_answer)
   - Success criteria (effectiveness, efficiency, robustness, safety thresholds)

**LangSmith Dataset URL**: https://smith.langchain.com/datasets

### Step 3.1b: Upload Level 6 Dataset (NEW)

Upload only Level 6 evaluation test cases:

```bash
# Upload only Level 6 categories
make eval-upload-level6

# Or use Python script directly
uv run python scripts/upload_golden_dataset.py --level6-only

# Upload specific Level 6 category
uv run python scripts/upload_golden_dataset.py --category bleu_rouge
uv run python scripts/upload_golden_dataset.py --category retrieval
uv run python scripts/upload_golden_dataset.py --category agentbench
```

**What this uploads**:
- BLEU/ROUGE test cases (20) with reference_answer
- Snapshot test cases (15) with snapshot_baseline
- Retrieval test cases (20) with relevant_docs
- RAGAS Recall test cases (10) with ground_truth_info
- AgentBench test cases (15) with task_spec and expected_result

### Step 3.2: Run Batch Evaluation

Run batch evaluation on the uploaded dataset:

```bash
# Full evaluation (105 cases, ~15-20 minutes)
make eval-run-batch

# Or use Python script directly
uv run python scripts/run_batch_evaluation.py

# With options
uv run python scripts/run_batch_evaluation.py \
  --max-cases=10 \
  --category=hurricane \
  --output=custom_results.json
```

**What this does**:
1. Runs all 105 test cases through the Weather AI Agent
2. Evaluates using 4-pillar framework:
   - Effectiveness (LLM-as-Judge for answer correctness)
   - Efficiency (tool usage, latency, tokens)
   - Robustness (edge case handling)
   - Safety (zero-tolerance for violations)
3. Generates evaluation report
4. Saves results to `evaluation_results.json`
5. Checks quality gates (pass_rate ≥85%, safety=0)
6. Exits with code 0 (pass) or 1 (fail) for CI/CD

**Output**:
```json
{
  "timestamp": "2025-12-12T...",
  "total_cases": 105,
  "passed_cases": 92,
  "failed_cases": 13,
  "pass_rate": 0.876,
  "pillar_averages": {
    "effectiveness": 0.89,
    "efficiency": 0.85,
    "robustness": 0.82
  },
  "safety_violations": 0,
  "execution_time_ms": 45000.0,
  "quality_gates": { ... },
  "all_gates_passed": true
}
```

### Step 3.3: Check Quality Gates

After evaluation, check if all quality gates pass:

```bash
# Check quality gates from evaluation_results.json
make eval-check-gates

# Or check custom results file
uv run python scripts/check_quality_gates.py --results-file=evaluation_quick.json

# With custom thresholds
uv run python scripts/check_quality_gates.py \
  --pass-rate=0.90 \
  --effectiveness=0.90 \
  --efficiency=0.85 \
  --robustness=0.85 \
  --safety=0
```

**Quality Gate Thresholds** (defaults):
| Gate | Threshold | Description |
|------|-----------|-------------|
| **Pass Rate** | ≥85% | Overall test pass percentage |
| **Effectiveness** | ≥85% | Answer correctness (LLM-as-Judge) |
| **Efficiency** | ≥80% | Tool usage, latency, tokens |
| **Robustness** | ≥80% | Edge case handling |
| **Safety** | 0 | Zero tolerance for violations |

**Exit Codes**:
- `0` - All quality gates passed (safe to deploy)
- `1` - One or more quality gates failed (do not deploy)

### Step 3.4: View Results in LangSmith UI

After running, view results at:

1. **Projects View**: `https://smith.langchain.com/projects`
   - Find project: `weather-ai-agent`
   - See all evaluation runs

2. **Trace View**: Click any run to see:
   - Input/output pairs
   - Tool calls made
   - Latency breakdown
   - Token usage

3. **Dataset View**: `https://smith.langchain.com/datasets`
   - See example success/failure rates
   - Filter by category
   - Compare across experiments

---

## Understanding Results

### Step 4.1: Reading the Evaluation Report

```
============================================================
WEATHER AI AGENT - GOLDEN DATASET EVALUATION REPORT
============================================================

Timestamp: 2025-12-11T12:00:00
Dataset: golden_dataset.yaml
Evaluation Time: 45000.0ms

----------------------------------------
SUMMARY
----------------------------------------
Total Cases: 105
Passed: 92
Failed: 13
Pass Rate: 87.6%

----------------------------------------
PILLAR AVERAGES
----------------------------------------
Effectiveness: 0.89
Efficiency: 0.85
Robustness: 0.82
Safety Pass Rate: 100.0%
Overall Average: 0.87

----------------------------------------
FAILED TESTS
----------------------------------------
  - edge_invalid_location_001
  - edge_ambiguous_002
  - complex_context_002
  ... and 10 more

----------------------------------------
QUALITY GATES
----------------------------------------
  pass_rate: PASS (actual: 0.88, threshold: 0.85)
  effectiveness: PASS (actual: 0.89, threshold: 0.85)
  efficiency: PASS (actual: 0.85, threshold: 0.80)
  robustness: PASS (actual: 0.82, threshold: 0.80)
  safety: PASS (actual: 0, threshold: 0)

----------------------------------------
RECOMMENDATION
----------------------------------------
DEPLOY: All quality gates passed

============================================================
```

### Step 4.2: Interpreting Pillar Scores

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| **0.95-1.00** | Excellent | Production-ready |
| **0.85-0.94** | Good | Minor improvements possible |
| **0.70-0.84** | Acceptable | Review failing cases |
| **0.50-0.69** | Needs Work | Investigate root causes |
| **<0.50** | Critical | Do not deploy |

### Step 4.3: Analyzing Failed Tests

For each failed test, examine:

1. **Which pillar failed?**
   ```python
   # Example failed result
   {
     "test_id": "hurricane_category_001",
     "effectiveness": 0.7,  # Failed (<0.95)
     "efficiency": 1.0,
     "robustness": 1.0,
     "safety": 1.0,
     "passed": False,
     "effectiveness_details": {
       "reasoning": "Answer mentioned Category 3 but should be Category 4"
     }
   }
   ```

2. **Is it a model issue or test issue?**
   - Model issue: Improve prompts/tools
   - Test issue: Update expected_answer

3. **Is it reproducible?**
   - Run single test multiple times
   - Check for non-determinism

---

## CI/CD Integration

### ⚡ Automated CI/CD Workflow (NEW - Pre-configured)

**Good News**: GitHub Actions workflow is already configured at `.github/workflows/evaluation.yml`

**What it does**:
1. ✅ Runs on every pull request and push to main/develop
2. ✅ Uploads golden dataset to LangSmith
3. ✅ Runs batch evaluation (105 test cases)
4. ✅ Checks all quality gates
5. ✅ Blocks merge if gates fail
6. ✅ Comments on PR with detailed results
7. ✅ Uploads evaluation results as artifacts

**Required GitHub Secrets**:
```bash
# Go to: Settings → Secrets and variables → Actions → New repository secret
OPENAI_API_KEY         # Your OpenAI API key
ANTHROPIC_API_KEY      # Your Anthropic API key
LANGCHAIN_API_KEY      # Your LangSmith API key (format: lsv2_pt_...)
```

**How to enable**:
1. Add the 3 secrets above to your GitHub repository
2. Push code to a branch
3. Create pull request → workflow runs automatically
4. View results in PR comment

**Workflow File**: `.github/workflows/evaluation.yml`

**Key Features**:
- ⏱️ Timeout: 60 minutes (full evaluation takes ~30-45 minutes)
- 🐳 Docker services: Starts Redis, Neo4j, PostgreSQL, Qdrant
- 📊 Quality gates: Blocks merge if any gate fails
- 💬 PR comments: Posts detailed evaluation report
- 📦 Artifacts: Uploads results for 30 days

### Step 5.1: View CI/CD Results

**In Pull Request**:
1. Go to your PR
2. Scroll to "Checks" section
3. Click "Evaluation & Quality Gates" workflow
4. View full logs and evaluation report

**In Workflow Run**:
1. Go to Actions tab
2. Click latest "Evaluation & Quality Gates" run
3. Download `evaluation-results` artifact
4. View `evaluation_results.json` and `evaluation_report.md`

**Example PR Comment** (auto-posted):
```markdown
## 📊 Evaluation Results

**Status:** ✅ PASS - All quality gates met

### Test Summary
- **Total Test Cases:** 105
- **Passed:** 92
- **Failed:** 13
- **Pass Rate:** 87.6%

### 4-Pillar Evaluation
| Pillar | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Effectiveness | 89.0% | ≥85% | ✅ |
| Efficiency | 85.0% | ≥80% | ✅ |
| Robustness | 82.0% | ≥80% | ✅ |
| Safety Violations | 0 | 0 | ✅ |

### Quality Gates
- **PASS_RATE:** 0.876 (threshold: 0.85) - ✅ PASS
- **EFFECTIVENESS:** 0.89 (threshold: 0.85) - ✅ PASS
- **EFFICIENCY:** 0.85 (threshold: 0.80) - ✅ PASS
- **ROBUSTNESS:** 0.82 (threshold: 0.80) - ✅ PASS
- **SAFETY:** 0 (threshold: 0) - ✅ PASS

### Next Steps
✅ All quality gates passed. Ready to merge!

View detailed traces: [LangSmith Dashboard](https://smith.langchain.com/projects)
```

### Step 5.2: Manual Workflow Trigger

You can also trigger the workflow manually:

1. Go to Actions tab
2. Click "Evaluation & Quality Gates" workflow
3. Click "Run workflow" button
4. Select branch
5. Click "Run workflow"

**Use cases**:
- Test evaluation before creating PR
- Re-run after fixing issues
- Validate specific branch

### Step 5.3: Pre-Commit Hook (Optional)

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running quick golden dataset validation..."

# Run quick test (5 cases) before commit
uv run python -c "
import asyncio
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner

async def quick_check():
    runner = GoldenDatasetRunner()
    results = await runner.run_all(max_cases=5)
    if results.safety_violations > 0:
        print('❌ Safety violations detected!')
        return False
    if results.pass_rate < 0.80:
        print('❌ Pass rate below threshold')
        return False
    print('✅ Quick validation passed')
    return True

success = asyncio.run(quick_check())
exit(0 if success else 1)
"
```

---

## Progressive Testing Strategy

### Phase 1: Smoke Test (Before Commit)
- **Cases**: 5 random
- **Time**: <30 seconds
- **Focus**: No crashes, basic functionality

```bash
uv run pytest tests/evaluation/ -v --max-cases=5
```

### Phase 2: Category Tests (During PR)
- **Cases**: 1 category at a time
- **Time**: 2-5 minutes each
- **Focus**: Category-specific validation

```bash
# Run each category
for category in simple complex hurricane edge; do
    echo "Testing $category..."
    uv run python -c "
import asyncio
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner

async def test():
    runner = GoldenDatasetRunner()
    results = await runner.run_category('$category')
    print(f'$category: {results.pass_rate:.1%} pass rate')

asyncio.run(test())
"
done
```

### Phase 3: Full Evaluation (Before Merge)
- **Cases**: All 105
- **Time**: 10-15 minutes
- **Focus**: Complete quality validation

```bash
uv run python scripts/run_langsmith_evaluation.py
```

### Phase 4: Regression Monitoring (Daily)
- **Cases**: All 105
- **Time**: Scheduled overnight
- **Focus**: Detect regressions over time

---

## Quick Reference Commands

### ⚡ Recommended: Makefile Commands (Updated for Level 6)

```bash
# 🚀 Full pipeline (upload + run + check) - 185 cases
make eval-full

# Upload golden dataset to LangSmith
make eval-upload-dataset

# Run batch evaluation (185 cases)
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

# 🆕 Level 6 Commands
make eval-level6              # Run all Level 6 evaluations (80 cases)
make eval-upload-level6       # Upload only Level 6 categories
make eval-bleu-rouge          # BLEU/ROUGE evaluation only
make eval-snapshot            # Snapshot testing only
make eval-retrieval           # Retrieval metrics only
make eval-ragas               # RAGAS evaluation only
make eval-agentbench          # AgentBench evaluation only
```

### 🆕 Level 6 Python Scripts (NEW)

```bash
# Run all Level 6 evaluations
uv run python scripts/run_level6_evaluation.py

# Run specific Level 6 eval type
uv run python scripts/run_level6_evaluation.py --eval-type bleu_rouge
uv run python scripts/run_level6_evaluation.py --eval-type retrieval
uv run python scripts/run_level6_evaluation.py --eval-type snapshot
uv run python scripts/run_level6_evaluation.py --eval-type ragas_recall
uv run python scripts/run_level6_evaluation.py --eval-type agentbench

# Limit cases
uv run python scripts/run_level6_evaluation.py --max-cases 5

# Custom output file
uv run python scripts/run_level6_evaluation.py --output level6_results.json
```

### Alternative: Python Scripts

```bash
# Upload dataset
uv run python scripts/upload_golden_dataset.py

# Run batch evaluation
uv run python scripts/run_batch_evaluation.py
uv run python scripts/run_batch_evaluation.py --max-cases=10
uv run python scripts/run_batch_evaluation.py --category=hurricane
uv run python scripts/run_batch_evaluation.py --output=custom.json

# Check quality gates
uv run python scripts/check_quality_gates.py
uv run python scripts/check_quality_gates.py --results-file=custom.json
uv run python scripts/check_quality_gates.py --pass-rate=0.90 --safety=0
```

### Legacy: Direct Python Runner

```bash
# Quick test (5 cases)
uv run pytest tests/evaluation/ -v --max-cases=5

# Test specific category
uv run python -c "
import asyncio
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner
runner = GoldenDatasetRunner()
results = asyncio.run(runner.run_category('hurricane'))
print(f'Pass rate: {results.pass_rate:.1%}')
"

# Full evaluation
uv run python -c "
import asyncio
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner
runner = GoldenDatasetRunner()
results = asyncio.run(runner.run_all())
print(runner.generate_report(results))
"

# Check safety-critical tests only
uv run python -c "
import asyncio
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner
runner = GoldenDatasetRunner()
results = asyncio.run(runner.run_safety_critical())
print(f'Safety violations: {results.safety_violations}')
"
```

---

**Document Version**: 2.0.0
**Last Updated**: 2025-12-13
**Previous Document**: [01-LANGSMITH_EVALUATION_SETUP.md](./01-LANGSMITH_EVALUATION_SETUP.md)
**Next Document**: [03-GUARDRAILS_TESTING.md](./03-GUARDRAILS_TESTING.md)

## Changelog

### v2.0.0 (2025-12-13)
- Added Level 6 evaluation types (BLEU/ROUGE, Snapshot, Retrieval, RAGAS, AgentBench)
- Updated total test cases from 105 to 185
- Added Level 6 specific commands section
- Added `run_level6_evaluation.py` script documentation
- Updated Makefile commands for Level 6 support
