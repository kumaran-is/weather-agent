# Step 4: Monitoring & Interpreting Results

**Purpose**: Set up dashboards, alerts, and interpret evaluation metrics over time
**Prerequisites**: Complete all previous documents (01-03)
**Time Required**: 45-60 minutes

---

## Table of Contents

1. [Overview](#overview)
2. [LangSmith Dashboard Setup](#langsmith-dashboard-setup)
3. [Prometheus + Grafana Monitoring](#prometheus--grafana-monitoring)
4. [Alert Configuration](#alert-configuration)
5. [Interpreting Results](#interpreting-results)
6. [Regression Detection](#regression-detection)
7. [Continuous Improvement Workflow](#continuous-improvement-workflow)
8. [Reporting & Documentation](#reporting--documentation)

---

## Overview

### Why Monitoring Matters

Effective monitoring enables:
- **Early Detection**: Catch regressions before they reach production
- **Trend Analysis**: Understand quality changes over time
- **Cost Optimization**: Track token usage and API costs
- **Safety Assurance**: Zero-tolerance for safety violations
- **Stakeholder Communication**: Clear metrics for decision-making

### Monitoring Stack

| Component | Purpose | Data Source |
|-----------|---------|-------------|
| **LangSmith** | Trace analysis, evaluation tracking | LangChain traces |
| **Prometheus** | Metrics collection | Application metrics |
| **Grafana** | Visualization, dashboards | Prometheus, LangSmith |
| **Alertmanager** | Alert routing | Prometheus rules |

---

## LangSmith Dashboard Setup

### Step 1.1: Access LangSmith Projects

1. Navigate to [smith.langchain.com](https://smith.langchain.com)
2. Select your project: `weather-ai-agent`
3. View default dashboards:
   - **Traces**: All agent invocations
   - **Runs**: Execution details
   - **Feedback**: Human evaluations
   - **Datasets**: Golden dataset examples

### Step 1.2: Create Custom Views

```python
# scripts/langsmith_custom_views.py
"""Create custom LangSmith views for monitoring."""

from langsmith import Client
from datetime import datetime, timedelta

client = Client()

def get_recent_runs(project_name: str, hours: int = 24) -> dict:
    """Get summary of recent runs."""
    start_time = datetime.now() - timedelta(hours=hours)

    runs = list(client.list_runs(
        project_name=project_name,
        start_time=start_time,
        run_type="chain",
    ))

    # Categorize by status
    summary = {
        "total": len(runs),
        "success": sum(1 for r in runs if r.status == "success"),
        "error": sum(1 for r in runs if r.status == "error"),
        "avg_latency_ms": 0,
        "total_tokens": 0,
    }

    latencies = []
    for run in runs:
        if run.end_time and run.start_time:
            latency = (run.end_time - run.start_time).total_seconds() * 1000
            latencies.append(latency)
        if run.total_tokens:
            summary["total_tokens"] += run.total_tokens

    if latencies:
        summary["avg_latency_ms"] = sum(latencies) / len(latencies)

    return summary

def get_evaluation_trends(dataset_name: str, days: int = 7) -> list:
    """Get evaluation score trends over time."""
    # Get dataset
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if not datasets:
        return []

    dataset = datasets[0]

    # Get experiments for this dataset
    experiments = list(client.list_experiments(dataset_id=dataset.id))

    trends = []
    for exp in experiments[-days:]:  # Last N experiments
        # Get aggregate results
        results = list(client.list_experiment_results(experiment_id=exp.id))

        if results:
            avg_score = sum(r.score for r in results if r.score) / len(results)
            trends.append({
                "date": exp.created_at.isoformat(),
                "experiment": exp.name,
                "avg_score": avg_score,
                "total_cases": len(results),
            })

    return trends

if __name__ == "__main__":
    # Get recent run summary
    summary = get_recent_runs("weather-ai-agent")
    print("Recent Runs (24h):")
    print(f"  Total: {summary['total']}")
    print(f"  Success: {summary['success']}")
    print(f"  Error: {summary['error']}")
    print(f"  Avg Latency: {summary['avg_latency_ms']:.0f}ms")
    print(f"  Total Tokens: {summary['total_tokens']}")

    # Get evaluation trends
    trends = get_evaluation_trends("weather-ai-golden-dataset")
    print("\nEvaluation Trends:")
    for t in trends:
        print(f"  {t['date']}: {t['avg_score']:.2f} ({t['total_cases']} cases)")
```

### Step 1.3: LangSmith Filters for Analysis

**Filter by Category**:
```
metadata.category = "hurricane"
```

**Filter by Safety-Critical**:
```
metadata.is_safety_critical = true
```

**Filter by Failed Tests**:
```
status = "error" OR feedback.score < 0.7
```

**Filter by High Latency**:
```
latency_ms > 2000
```

---

## Prometheus + Grafana Monitoring

### Step 2.1: Configure Prometheus Metrics

The Weather AI Agent exposes metrics at `/metrics`. Key metrics:

```yaml
# Prometheus scrape config (prometheus.yml)
scrape_configs:
  - job_name: 'weather-ai-agent'
    static_configs:
      - targets: ['weather-ai-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Step 2.2: Key Metrics to Monitor

**Cache Metrics**:
```promql
# L1 Cache Hit Rate
weather_cache_l1_hit_rate

# L2 Cache Hit Rate
weather_cache_l2_hit_rate

# Overall Cache Hit Rate
weather_cache_overall_hit_rate

# Cache Latency (p95)
histogram_quantile(0.95, weather_cache_latency_seconds_bucket)
```

**Agent Metrics**:
```promql
# Request Rate
rate(weather_agent_requests_total[5m])

# Error Rate
rate(weather_agent_errors_total[5m]) / rate(weather_agent_requests_total[5m])

# Latency P95
histogram_quantile(0.95, weather_agent_latency_seconds_bucket)

# Token Usage
sum(rate(weather_agent_tokens_total[5m])) by (model)
```

**Guardrail Metrics**:
```promql
# Guardrail Violations by Type
sum(weather_guardrail_violations_total) by (violation_type)

# PII Detection Rate
rate(weather_guardrail_pii_detected_total[5m])

# Blocked Requests
rate(weather_guardrail_blocked_total[5m])
```

### Step 2.3: Import Grafana Dashboard

```bash
# Import pre-built dashboard
curl -X POST \
  -H "Content-Type: application/json" \
  -d @grafana/dashboards/weather-ai-monitoring.json \
  http://localhost:3000/api/dashboards/db
```

**Dashboard Panels**:

1. **Overview Row**:
   - Request Rate (requests/sec)
   - Error Rate (%)
   - P95 Latency (ms)
   - Cache Hit Rate (%)

2. **Cache Performance Row**:
   - L1 vs L2 vs L3 Hit Rates
   - Cache Latency Distribution
   - Cost Savings from Cache

3. **Quality Row**:
   - Evaluation Pass Rate
   - Pillar Scores (Effectiveness, Efficiency, Robustness, Safety)
   - Failed Test Count

4. **Safety Row**:
   - Guardrail Violations
   - PII Detection Events
   - Blocked Requests

5. **Cost Row**:
   - Token Usage by Model
   - Estimated Cost ($)
   - Cost per Query

### Step 2.4: Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "Weather AI Agent - Production Monitoring",
    "uid": "weather-ai-prod",
    "panels": [
      {
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(weather_agent_requests_total[5m]))",
            "legendFormat": "req/s"
          }
        ],
        "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4}
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(weather_agent_errors_total[5m])) / sum(rate(weather_agent_requests_total[5m])) * 100",
            "legendFormat": "%"
          }
        ],
        "thresholds": {
          "steps": [
            {"color": "green", "value": null},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 5}
          ]
        },
        "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4}
      },
      {
        "title": "P95 Latency",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(weather_agent_latency_seconds_bucket[5m])) by (le)) * 1000",
            "legendFormat": "ms"
          }
        ],
        "thresholds": {
          "steps": [
            {"color": "green", "value": null},
            {"color": "yellow", "value": 500},
            {"color": "red", "value": 1000}
          ]
        },
        "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4}
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "weather_cache_overall_hit_rate * 100",
            "legendFormat": "%"
          }
        ],
        "thresholds": {
          "steps": [
            {"color": "red", "value": null},
            {"color": "yellow", "value": 30},
            {"color": "green", "value": 50}
          ]
        },
        "gridPos": {"x": 18, "y": 0, "w": 6, "h": 4}
      },
      {
        "title": "Evaluation Pass Rate Trend",
        "type": "timeseries",
        "targets": [
          {
            "expr": "weather_evaluation_pass_rate * 100",
            "legendFormat": "Pass Rate %"
          }
        ],
        "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8}
      },
      {
        "title": "Guardrail Violations",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(weather_guardrail_violations_total[5m])) by (violation_type)",
            "legendFormat": "{{violation_type}}"
          }
        ],
        "gridPos": {"x": 12, "y": 4, "w": 12, "h": 8}
      }
    ]
  }
}
```

---

## Alert Configuration

### Step 3.1: Prometheus Alert Rules

```yaml
# prometheus/rules/weather-ai-alerts.yml
groups:
  - name: weather-ai-quality
    rules:
      # Safety violation alert (CRITICAL)
      - alert: SafetyViolationDetected
        expr: increase(weather_guardrail_safety_violations_total[5m]) > 0
        for: 0s
        labels:
          severity: critical
        annotations:
          summary: "Safety violation detected"
          description: "A safety-critical guardrail was triggered. Immediate review required."

      # Evaluation pass rate drop
      - alert: EvaluationPassRateLow
        expr: weather_evaluation_pass_rate < 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Evaluation pass rate below threshold"
          description: "Pass rate is {{ $value | humanizePercentage }}, threshold is 85%"

      # High error rate
      - alert: HighErrorRate
        expr: |
          sum(rate(weather_agent_errors_total[5m]))
          / sum(rate(weather_agent_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, sum(rate(weather_agent_latency_seconds_bucket[5m])) by (le)) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency exceeds 1 second"
          description: "P95 latency is {{ $value | humanizeDuration }}"

      # Cache hit rate drop
      - alert: LowCacheHitRate
        expr: weather_cache_overall_hit_rate < 0.30
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Cache hit rate below 30%"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"

  - name: weather-ai-guardrails
    rules:
      # PII detection spike
      - alert: PIIDetectionSpike
        expr: increase(weather_guardrail_pii_detected_total[1h]) > 10
        for: 0s
        labels:
          severity: warning
        annotations:
          summary: "Spike in PII detection"
          description: "{{ $value }} PII instances detected in last hour"

      # Prompt injection attempts
      - alert: PromptInjectionAttempts
        expr: increase(weather_guardrail_injection_blocked_total[1h]) > 5
        for: 0s
        labels:
          severity: warning
        annotations:
          summary: "Multiple prompt injection attempts"
          description: "{{ $value }} injection attempts blocked in last hour"
```

### Step 3.2: Alertmanager Configuration

```yaml
# alertmanager/alertmanager.yml
global:
  slack_api_url: '$SLACK_WEBHOOK_URL'

route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    # Critical alerts go to PagerDuty
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    # Safety violations go to dedicated channel
    - match:
        alertname: SafetyViolationDetected
      receiver: 'safety-team'

    # All warnings go to Slack
    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#weather-ai-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '$PAGERDUTY_SERVICE_KEY'
        severity: critical

  - name: 'safety-team'
    slack_configs:
      - channel: '#weather-ai-safety'
        title: 'SAFETY ALERT: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        color: 'danger'

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#weather-ai-warnings'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        color: 'warning'
```

---

## Interpreting Results

### Step 4.1: Understanding Evaluation Reports

**Sample Report Analysis**:

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
QUALITY GATES
----------------------------------------
  pass_rate: PASS (actual: 0.88, threshold: 0.85)
  effectiveness: PASS (actual: 0.89, threshold: 0.85)
  efficiency: PASS (actual: 0.85, threshold: 0.80)
  robustness: PASS (actual: 0.82, threshold: 0.80)
  safety: PASS (actual: 0, threshold: 0)
```

**Interpretation Guide**:

| Metric | Good | Warning | Critical | Action |
|--------|------|---------|----------|--------|
| Pass Rate | ≥85% | 70-84% | <70% | Review failed cases |
| Effectiveness | ≥0.85 | 0.70-0.84 | <0.70 | Improve prompts |
| Efficiency | ≥0.80 | 0.60-0.79 | <0.60 | Optimize tool usage |
| Robustness | ≥0.80 | 0.60-0.79 | <0.60 | Add edge case handling |
| Safety | 100% | N/A | <100% | IMMEDIATE FIX |

### Step 4.2: Analyzing Failed Tests

```python
# scripts/analyze_failures.py
"""Analyze failed test cases for patterns."""

import json
from collections import defaultdict
from pathlib import Path

def analyze_failures(results_file: str) -> dict:
    """Analyze patterns in failed tests."""
    with open(results_file) as f:
        results = json.load(f)

    failures = [r for r in results["results"] if not r["passed"]]

    analysis = {
        "total_failures": len(failures),
        "by_category": defaultdict(int),
        "by_pillar": defaultdict(int),
        "common_patterns": [],
    }

    for failure in failures:
        # Categorize by test category
        category = failure.get("category", "unknown")
        analysis["by_category"][category] += 1

        # Find which pillar(s) failed
        if failure.get("effectiveness", 1.0) < 0.8:
            analysis["by_pillar"]["effectiveness"] += 1
        if failure.get("efficiency", 1.0) < 0.7:
            analysis["by_pillar"]["efficiency"] += 1
        if failure.get("robustness", 1.0) < 0.7:
            analysis["by_pillar"]["robustness"] += 1
        if failure.get("safety", 1.0) < 1.0:
            analysis["by_pillar"]["safety"] += 1

    # Identify patterns
    if analysis["by_category"]["edge"] > analysis["total_failures"] * 0.5:
        analysis["common_patterns"].append("Most failures in edge cases - improve error handling")

    if analysis["by_pillar"]["effectiveness"] > analysis["total_failures"] * 0.5:
        analysis["common_patterns"].append("Effectiveness issues - review prompt templates")

    if analysis["by_pillar"]["efficiency"] > analysis["total_failures"] * 0.3:
        analysis["common_patterns"].append("Efficiency issues - optimize tool selection")

    return analysis

def print_analysis(analysis: dict):
    """Print failure analysis report."""
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nTotal Failures: {analysis['total_failures']}")

    print("\nBy Category:")
    for cat, count in sorted(analysis["by_category"].items()):
        print(f"  {cat}: {count}")

    print("\nBy Failed Pillar:")
    for pillar, count in sorted(analysis["by_pillar"].items()):
        print(f"  {pillar}: {count}")

    if analysis["common_patterns"]:
        print("\nIdentified Patterns:")
        for pattern in analysis["common_patterns"]:
            print(f"  - {pattern}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    analysis = analyze_failures("evaluation_results.json")
    print_analysis(analysis)
```

### Step 4.3: Cost Analysis

```python
# scripts/cost_analysis.py
"""Analyze API costs from evaluation runs."""

from langsmith import Client
from datetime import datetime, timedelta

# Token pricing (per 1K tokens)
PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
}

def calculate_costs(project_name: str, hours: int = 24) -> dict:
    """Calculate API costs from LangSmith traces."""
    client = Client()
    start_time = datetime.now() - timedelta(hours=hours)

    runs = list(client.list_runs(
        project_name=project_name,
        start_time=start_time,
        run_type="llm",
    ))

    costs = {
        "total_cost": 0.0,
        "by_model": {},
        "total_tokens": {"input": 0, "output": 0},
        "runs_analyzed": len(runs),
    }

    for run in runs:
        model = run.extra.get("model_name", "unknown")
        input_tokens = run.prompt_tokens or 0
        output_tokens = run.completion_tokens or 0

        # Calculate cost
        if model in PRICING:
            input_cost = (input_tokens / 1000) * PRICING[model]["input"]
            output_cost = (output_tokens / 1000) * PRICING[model]["output"]
            run_cost = input_cost + output_cost
        else:
            run_cost = 0

        costs["total_cost"] += run_cost
        costs["total_tokens"]["input"] += input_tokens
        costs["total_tokens"]["output"] += output_tokens

        if model not in costs["by_model"]:
            costs["by_model"][model] = {"cost": 0, "calls": 0, "tokens": 0}
        costs["by_model"][model]["cost"] += run_cost
        costs["by_model"][model]["calls"] += 1
        costs["by_model"][model]["tokens"] += input_tokens + output_tokens

    return costs

if __name__ == "__main__":
    costs = calculate_costs("weather-ai-agent", hours=24)

    print("\n" + "=" * 60)
    print("COST ANALYSIS (Last 24 Hours)")
    print("=" * 60)
    print(f"\nTotal Cost: ${costs['total_cost']:.4f}")
    print(f"Total Runs: {costs['runs_analyzed']}")
    print(f"Total Tokens: {costs['total_tokens']['input'] + costs['total_tokens']['output']:,}")

    print("\nBy Model:")
    for model, data in sorted(costs["by_model"].items(), key=lambda x: -x[1]["cost"]):
        print(f"  {model}:")
        print(f"    Cost: ${data['cost']:.4f}")
        print(f"    Calls: {data['calls']}")
        print(f"    Tokens: {data['tokens']:,}")
```

---

## Regression Detection

### Step 5.1: Automated Regression Detection

```python
# scripts/detect_regressions.py
"""Detect quality regressions in evaluation results."""

from langsmith import Client
from datetime import datetime, timedelta
from typing import Optional

def detect_regressions(
    dataset_name: str,
    baseline_days: int = 7,
    threshold: float = 0.05,
) -> dict:
    """
    Detect regressions by comparing recent results to baseline.

    Args:
        dataset_name: Name of the golden dataset
        baseline_days: Days to consider for baseline
        threshold: Minimum score drop to flag as regression

    Returns:
        Dict with regression analysis
    """
    client = Client()

    # Get dataset
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if not datasets:
        return {"error": "Dataset not found"}

    dataset = datasets[0]

    # Get experiments
    experiments = list(client.list_experiments(dataset_id=dataset.id))
    if len(experiments) < 2:
        return {"error": "Not enough experiments for comparison"}

    # Sort by date
    experiments.sort(key=lambda x: x.created_at)

    # Calculate baseline (older experiments)
    baseline_cutoff = datetime.now() - timedelta(days=baseline_days)
    baseline_exps = [e for e in experiments if e.created_at < baseline_cutoff]
    recent_exps = [e for e in experiments if e.created_at >= baseline_cutoff]

    if not baseline_exps or not recent_exps:
        # Use last 2 experiments if date filtering doesn't work
        baseline_exps = experiments[:-1]
        recent_exps = experiments[-1:]

    # Calculate scores
    def get_avg_score(exps):
        scores = []
        for exp in exps:
            results = list(client.list_experiment_results(experiment_id=exp.id))
            if results:
                scores.extend([r.score for r in results if r.score is not None])
        return sum(scores) / len(scores) if scores else 0

    baseline_score = get_avg_score(baseline_exps)
    recent_score = get_avg_score(recent_exps)

    delta = recent_score - baseline_score
    is_regression = delta < -threshold

    return {
        "baseline_score": baseline_score,
        "recent_score": recent_score,
        "delta": delta,
        "is_regression": is_regression,
        "baseline_experiments": len(baseline_exps),
        "recent_experiments": len(recent_exps),
        "recommendation": (
            "REGRESSION DETECTED: Review recent changes"
            if is_regression else
            "No regression detected"
        ),
    }

if __name__ == "__main__":
    result = detect_regressions("weather-ai-golden-dataset")

    print("\n" + "=" * 60)
    print("REGRESSION DETECTION REPORT")
    print("=" * 60)
    print(f"\nBaseline Score: {result.get('baseline_score', 0):.3f}")
    print(f"Recent Score: {result.get('recent_score', 0):.3f}")
    print(f"Delta: {result.get('delta', 0):+.3f}")
    print(f"\nRegression Detected: {'YES' if result.get('is_regression') else 'NO'}")
    print(f"Recommendation: {result.get('recommendation')}")
```

### Step 5.2: CI/CD Quality Gate

```yaml
# .github/workflows/quality-gate.yml
name: Quality Gate

on:
  pull_request:
    branches: [main, develop]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Run Evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
          LANGCHAIN_TRACING_V2: true
          LANGCHAIN_PROJECT: weather-ai-agent-pr-${{ github.event.pull_request.number }}
        run: |
          uv run python scripts/run_langsmith_evaluation.py

      - name: Check Quality Gates
        run: |
          uv run python -c "
          import json
          with open('evaluation_results.json') as f:
              results = json.load(f)

          gates = {
              'pass_rate': (results['pass_rate'], 0.85),
              'effectiveness': (results['pillar_averages']['effectiveness'], 0.85),
              'efficiency': (results['pillar_averages']['efficiency'], 0.80),
              'robustness': (results['pillar_averages']['robustness'], 0.80),
              'safety': (results['safety_violations'], 0),
          }

          failed = []
          for gate, (actual, threshold) in gates.items():
              if gate == 'safety':
                  if actual > threshold:
                      failed.append(f'{gate}: {actual} violations (max: {threshold})')
              else:
                  if actual < threshold:
                      failed.append(f'{gate}: {actual:.2f} < {threshold:.2f}')

          if failed:
              print('Quality gates failed:')
              for f in failed:
                  print(f'  - {f}')
              exit(1)

          print('All quality gates passed!')
          "

      - name: Check for Regressions
        run: |
          uv run python scripts/detect_regressions.py
```

---

## Continuous Improvement Workflow

### Step 6.1: Weekly Review Process

```markdown
## Weekly Quality Review Checklist

### 1. Metrics Review (15 min)
- [ ] Check Grafana dashboard for anomalies
- [ ] Review pass rate trend (should be ≥85%)
- [ ] Check safety violations (should be 0)
- [ ] Review cost trends

### 2. Failed Test Analysis (30 min)
- [ ] Run failure analysis script
- [ ] Categorize failures by type
- [ ] Identify patterns in failures
- [ ] Create tickets for fixes

### 3. Regression Check (15 min)
- [ ] Run regression detection
- [ ] Compare to baseline
- [ ] Document any regressions
- [ ] Identify root causes

### 4. Action Items (ongoing)
- [ ] Prioritize fixes based on severity
- [ ] Update golden dataset if needed
- [ ] Adjust thresholds if justified
- [ ] Schedule remediation work
```

### Step 6.2: Feedback Loop

```python
# scripts/add_feedback.py
"""Add human feedback to LangSmith runs for improvement."""

from langsmith import Client

client = Client()

def add_feedback(run_id: str, score: float, comment: str):
    """Add human feedback to a specific run."""
    client.create_feedback(
        run_id=run_id,
        key="human_review",
        score=score,
        comment=comment,
    )
    print(f"Feedback added to run {run_id}")

def flag_for_review(run_id: str, reason: str):
    """Flag a run for human review."""
    client.create_feedback(
        run_id=run_id,
        key="needs_review",
        score=0,
        comment=reason,
    )
    print(f"Run {run_id} flagged for review: {reason}")

if __name__ == "__main__":
    # Example: Add feedback to a run
    # add_feedback("run-uuid-here", score=0.8, comment="Good response but missing time specificity")

    # Example: Flag for review
    # flag_for_review("run-uuid-here", "Potential incorrect hurricane category")
    pass
```

---

## Reporting & Documentation

### Step 7.1: Generate Weekly Report

```python
# scripts/generate_weekly_report.py
"""Generate weekly quality report."""

import json
from datetime import datetime, timedelta
from pathlib import Path

def generate_weekly_report() -> str:
    """Generate a markdown weekly report."""
    # Load latest results
    with open("evaluation_results.json") as f:
        results = json.load(f)

    report = f"""# Weather AI Agent - Weekly Quality Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Period**: Last 7 days

---

## Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Pass Rate | {results['pass_rate']:.1%} | {'PASS' if results['pass_rate'] >= 0.85 else 'FAIL'} |
| Effectiveness | {results['pillar_averages']['effectiveness']:.2f} | {'PASS' if results['pillar_averages']['effectiveness'] >= 0.85 else 'FAIL'} |
| Efficiency | {results['pillar_averages']['efficiency']:.2f} | {'PASS' if results['pillar_averages']['efficiency'] >= 0.80 else 'FAIL'} |
| Robustness | {results['pillar_averages']['robustness']:.2f} | {'PASS' if results['pillar_averages']['robustness'] >= 0.80 else 'FAIL'} |
| Safety Violations | {results['safety_violations']} | {'PASS' if results['safety_violations'] == 0 else 'CRITICAL'} |

---

## Test Results

- **Total Tests**: {results['total_cases']}
- **Passed**: {results['passed']}
- **Failed**: {results['failed']}

### Failed Tests

| Test ID | Category | Failed Pillar |
|---------|----------|---------------|
"""

    for failure in results.get("failed_tests", [])[:10]:
        report += f"| {failure['id']} | {failure.get('category', 'N/A')} | {failure.get('failed_pillar', 'N/A')} |\n"

    report += """
---

## Recommendations

1. **High Priority**: Address any safety violations immediately
2. **Medium Priority**: Improve failed test cases
3. **Low Priority**: Optimize efficiency metrics

---

## Next Steps

- [ ] Review failed tests in detail
- [ ] Update prompts for effectiveness improvements
- [ ] Add edge case handling for robustness
- [ ] Schedule follow-up evaluation

---

*Report generated automatically by Weather AI Agent quality system.*
"""

    return report

if __name__ == "__main__":
    report = generate_weekly_report()

    # Save to file
    output_path = Path("reports") / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.md"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(report)

    print(f"Report saved to: {output_path}")
    print("\nReport Preview:")
    print(report[:500] + "...")
```

### Step 7.2: Archive Results

```bash
# scripts/archive_results.sh
#!/bin/bash

# Archive evaluation results
DATE=$(date +%Y%m%d)
ARCHIVE_DIR="evaluation_archives/${DATE}"

mkdir -p "$ARCHIVE_DIR"

# Copy results
cp evaluation_results.json "$ARCHIVE_DIR/"
cp -r reports/*.md "$ARCHIVE_DIR/" 2>/dev/null || true

# Generate summary
echo "Archived evaluation results to: $ARCHIVE_DIR"
ls -la "$ARCHIVE_DIR"
```

---

## Quick Reference Commands

```bash
# View LangSmith dashboard
open https://smith.langchain.com/projects

# Access Grafana (local)
open http://localhost:3000

# Run quick metrics check
uv run python scripts/langsmith_custom_views.py

# Analyze failures
uv run python scripts/analyze_failures.py

# Calculate costs
uv run python scripts/cost_analysis.py

# Detect regressions
uv run python scripts/detect_regressions.py

# Generate weekly report
uv run python scripts/generate_weekly_report.py

# Check alerts (Alertmanager)
curl http://localhost:9093/api/v1/alerts
```

---

## Conclusion

You have now completed the 4-part progressive testing series:

1. **LangSmith Setup** - API configuration, dataset upload
2. **Golden Dataset Testing** - Running evaluations, understanding results
3. **Guardrails Testing** - Security validation, safety checks
4. **Monitoring & Results** - Dashboards, alerts, continuous improvement

### Next Steps

1. Set up scheduled evaluations (daily/weekly)
2. Configure alerts for your team
3. Establish weekly review process
4. Build regression detection into CI/CD

---

**Previous Document**: [03-GUARDRAILS_TESTING.md](./03-GUARDRAILS_TESTING.md)
**Series Complete**: 4 of 4
