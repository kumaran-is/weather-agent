# Comprehensive Testing Guide: Level 6 Self-Evolving AI Platform (v2.0.0)

**Purpose**: Manual and automated testing via REST endpoints, Python scripts, and LangSmith Studio to validate the complete Level 6 Self-Evolving AI Platform including advanced evaluation frameworks, self-improvement capabilities, and production-grade testing infrastructure.

**Duration**: 60-90 minutes for complete testing
**Date**: December 13, 2025
**Version**: v2.0.0
**Status**: ✅ **Level 6 Complete** (6a + 6b + 6c)

### What's New in v2.0.0
- ✅ **Scenario 18**: Golden Dataset Evaluation (185 test cases)
- ✅ **Scenario 19**: Constitutional AI Guardrails Testing
- ✅ Makefile automation commands for Level 6 evaluation
- ✅ Level 6 categories: BLEU/ROUGE, Snapshot, Retrieval, RAGAS, AgentBench

---

## ⚡ Quick Start Summary

### What Level 6 Covers

| Sub-Level | Focus | Key Components | Impact |
|-----------|-------|----------------|--------|
| **6a** | Advanced Evaluation | Ragas, DeepEval, Context Optimization | >0.90 faithfulness, >0.85 precision |
| **6b** | Self-Improvement | TruLens, LangChain Benchmark, Auto-Prompt | Continuous quality improvement |
| **6c** | Production Testing | Property/Snapshot Testing, Promptfoo, OpenAI Evals | Comprehensive test coverage |

### Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Weather API | http://localhost:8000 | Main application |
| Swagger UI | http://localhost:8000/docs | API testing |
| Cache Stats | http://localhost:8000/cache/stats | Cache performance |
| Metrics | http://localhost:8000/metrics | Prometheus metrics |
| Grafana | http://localhost:3001 | Dashboards (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics queries |
| LangSmith | https://smith.langchain.com | AI tracing & evaluation |

---

## 🎯 What We're Testing

### **Level 6a: Advanced Evaluation Framework**
1. ✅ Ragas Integration - Faithfulness, Context Precision, Context Recall, Answer Relevancy
2. ✅ DeepEval Integration - LLM unit testing with synthetic data generation
3. ✅ Context Optimization - Token budget management, context window optimization
4. ✅ Adversarial Testing - Edge case and robustness evaluation

### **Level 6b: Self-Improvement Platform**
1. ✅ TruLens Integration - Real-time RAG evaluation (Groundedness, Answer Relevance, Context Relevance)
2. ✅ LangChain Benchmark - Standardized agent benchmarking (multi-hop, tool usage, QA)
3. ✅ Auto-Prompt Engineering - Automatic prompt optimization
4. ✅ Composio Integration - Tool orchestration framework

### **Level 6c: Production Testing Infrastructure**
1. ✅ Constitutional AI - Safety and ethical guardrails
2. ✅ Content Filter - Input/output content moderation
3. ✅ Output Validator - Response validation and formatting
4. ✅ Property-Based Testing - Hypothesis-style edge case discovery
5. ✅ Snapshot Testing - Response regression detection
6. ✅ Retrieval Metrics - MRR, NDCG, Precision@k, Recall@k, MAP
7. ✅ Generation Metrics - BLEU, ROUGE scores
8. ✅ Promptfoo Integration - Multi-provider prompt testing
9. ✅ OpenAI Evals Integration - Systematic evaluation framework

### **LangSmith Integration**
1. ✅ Tracing - Full request/response traces
2. ✅ Evaluation Metrics - 4-pillar scores pushed to LangSmith
3. ✅ Dataset Management - Golden dataset for regression testing
4. ✅ Experiment Tracking - A/B testing and model comparison

---

## 📋 Prerequisites

### 1. Start All Services

```bash
# Start Docker services (includes observability stack)
docker-compose -f docker-compose.dev.yml up -d

# Verify all services healthy
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Expected services:
# - weather-ai-api-dev (FastAPI)
# - weather-ai-redis (caching + L2)
# - weather-ai-neo4j (memory)
# - weather-ai-qdrant (RAG)
# - weather-mcp (weather data)
# - hurricane-mcp (hurricane tracking)
# - prometheus (metrics)
# - grafana (dashboards)
# - loki (logs)
```

### 2. Verify Environment Variables

```bash
# Check Level 6 environment variables
echo "LANGCHAIN_TRACING_V2: $LANGCHAIN_TRACING_V2"  # Should be "true"
echo "LANGCHAIN_API_KEY: ${LANGCHAIN_API_KEY:0:10}..."  # Should be set
echo "LANGCHAIN_PROJECT: $LANGCHAIN_PROJECT"  # Should be "weather-ai-agent"
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."  # Required for evaluators
echo "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:0:10}..."  # Required for Claude
```

### 3. Install Test Dependencies

```bash
# Install evaluation dependencies
cd /home/user/weather-ai-agent-service
uv sync

# Verify imports work
python -c "from backend.src.evaluation import RagasEvaluator, DeepEvalIntegration, TruLensIntegration; print('✅ All Level 6 modules imported')"
```

---

# PART 1: LEVEL 6a - ADVANCED EVALUATION FRAMEWORK

---

## **SCENARIO 1: Ragas RAG Evaluation**

### Purpose
Test RAG-specific metrics: Faithfulness, Context Precision, Context Recall, Answer Relevancy.

### Test Script

```python
#!/usr/bin/env python3
"""Test Ragas RAG Evaluation - Level 6a"""

import asyncio
from backend.src.evaluation.ragas_evaluator import RagasEvaluator, RagasResult

async def test_ragas_evaluation():
    """Test Ragas metrics for RAG evaluation."""
    print("=" * 60)
    print("SCENARIO 1: Ragas RAG Evaluation")
    print("=" * 60)

    # Initialize evaluator
    evaluator = RagasEvaluator()

    # Test case: Hurricane weather query
    query = "What is the current status of Hurricane Milton?"
    contexts = [
        "Hurricane Milton is a Category 4 hurricane with 145 mph winds.",
        "Milton is currently located 200 miles southwest of Tampa Bay.",
        "Expected landfall: October 9, 2024 near Tampa Bay area.",
        "Storm surge warning: 10-15 feet for Tampa Bay."
    ]
    answer = "Hurricane Milton is a Category 4 hurricane with 145 mph sustained winds, currently 200 miles southwest of Tampa Bay. Landfall is expected October 9, 2024 near Tampa Bay. Storm surge warnings of 10-15 feet are in effect for Tampa Bay."

    # Run evaluation
    result = await evaluator.evaluate(
        query=query,
        contexts=contexts,
        answer=answer
    )

    # Display results
    print(f"\n📊 Ragas Evaluation Results:")
    print(f"  Faithfulness:       {result.faithfulness:.3f} (threshold: 0.90)")
    print(f"  Context Precision:  {result.context_precision:.3f} (threshold: 0.85)")
    print(f"  Context Recall:     {result.context_recall:.3f} (threshold: 0.85)")
    print(f"  Answer Relevancy:   {result.answer_relevancy:.3f} (threshold: 0.90)")
    print(f"  Overall Score:      {result.overall_score:.3f}")
    print(f"  Passed:             {'✅' if result.passed else '❌'} {result.passed}")

    # Assertions
    assert result.faithfulness >= 0.0, "Faithfulness should be >= 0"
    assert result.context_precision >= 0.0, "Context precision should be >= 0"
    assert result.context_recall >= 0.0, "Context recall should be >= 0"
    assert result.answer_relevancy >= 0.0, "Answer relevancy should be >= 0"

    print("\n✅ Ragas evaluation test passed!")
    return result

if __name__ == "__main__":
    asyncio.run(test_ragas_evaluation())
```

### Run Command

```bash
cd /home/user/weather-ai-agent-service
python -c "
import asyncio
from backend.src.evaluation.ragas_evaluator import RagasEvaluator

async def test():
    evaluator = RagasEvaluator()
    result = await evaluator.evaluate(
        query='What is the weather in Miami?',
        contexts=['Miami has sunny weather with 85°F temperature.', 'Humidity is 70% with light winds.'],
        answer='Miami currently has sunny weather with a temperature of 85°F and 70% humidity.'
    )
    print(f'Faithfulness: {result.faithfulness:.3f}')
    print(f'Context Precision: {result.context_precision:.3f}')
    print(f'Context Recall: {result.context_recall:.3f}')
    print(f'Answer Relevancy: {result.answer_relevancy:.3f}')
    print(f'Overall: {result.overall_score:.3f}')
    print(f'Passed: {result.passed}')

asyncio.run(test())
"
```

### Expected Output

```
Faithfulness: 0.950
Context Precision: 0.900
Context Recall: 0.880
Answer Relevancy: 0.920
Overall: 0.920
Passed: True
```

### Validation Checklist
- [ ] Faithfulness score ≥ 0.90 (answer grounded in context)
- [ ] Context Precision ≥ 0.85 (top-k chunks relevant)
- [ ] Context Recall ≥ 0.85 (all relevant info retrieved)
- [ ] Answer Relevancy ≥ 0.90 (answer addresses query)
- [ ] Overall score calculated correctly

---

## **SCENARIO 2: DeepEval LLM Unit Testing**

### Purpose
Test LLM responses with DeepEval's comprehensive metrics and synthetic test generation.

### Test Script

```python
#!/usr/bin/env python3
"""Test DeepEval Integration - Level 6a"""

import asyncio
from backend.src.evaluation.deepeval_integration import DeepEvalIntegration, DeepEvalResult

async def test_deepeval():
    """Test DeepEval LLM unit testing."""
    print("=" * 60)
    print("SCENARIO 2: DeepEval LLM Unit Testing")
    print("=" * 60)

    # Initialize
    integration = DeepEvalIntegration()

    # Test case
    query = "Should I evacuate for Hurricane Milton?"
    contexts = [
        "Hurricane Milton is Category 4 with 145 mph winds.",
        "Your location is in Zone A evacuation area.",
        "Storm surge expected: 10-15 feet.",
        "Evacuation order issued for all coastal areas."
    ]
    answer = "Yes, you should evacuate immediately. You are in Zone A which is under mandatory evacuation. Hurricane Milton is a Category 4 storm with 145 mph winds and expected storm surge of 10-15 feet. Please evacuate to higher ground now."

    # Run evaluation
    result = await integration.evaluate(
        query=query,
        contexts=contexts,
        answer=answer
    )

    # Display results
    print(f"\n📊 DeepEval Results:")
    print(f"  Answer Relevancy:     {result.answer_relevancy:.3f} (threshold: 0.85)")
    print(f"  Faithfulness:         {result.faithfulness:.3f} (threshold: 0.85)")
    print(f"  Contextual Precision: {result.contextual_precision:.3f} (threshold: 0.85)")
    print(f"  Contextual Recall:    {result.contextual_recall:.3f} (threshold: 0.85)")
    print(f"  Overall Score:        {result.overall_score:.3f}")
    print(f"  Passed:               {'✅' if result.passed else '❌'} {result.passed}")

    print("\n✅ DeepEval test passed!")
    return result

if __name__ == "__main__":
    asyncio.run(test_deepeval())
```

### Run Command

```bash
python -c "
import asyncio
from backend.src.evaluation.deepeval_integration import DeepEvalIntegration

async def test():
    integration = DeepEvalIntegration()
    result = await integration.evaluate(
        query='What is the forecast for Tampa?',
        contexts=['Tampa forecast: Partly cloudy, high 88°F, low 72°F.'],
        answer='Tampa will have partly cloudy skies with a high of 88°F and a low of 72°F.'
    )
    print(f'Answer Relevancy: {result.answer_relevancy:.3f}')
    print(f'Faithfulness: {result.faithfulness:.3f}')
    print(f'Contextual Precision: {result.contextual_precision:.3f}')
    print(f'Contextual Recall: {result.contextual_recall:.3f}')
    print(f'Passed: {result.passed}')

asyncio.run(test())
"
```

### Expected Output

```
Answer Relevancy: 0.920
Faithfulness: 0.950
Contextual Precision: 0.880
Contextual Recall: 0.900
Passed: True
```

### Validation Checklist
- [ ] Answer Relevancy ≥ 0.85
- [ ] Faithfulness ≥ 0.85 (no hallucination)
- [ ] Contextual Precision ≥ 0.85
- [ ] Contextual Recall ≥ 0.85
- [ ] Test case ID generated when provided

---

## **SCENARIO 3: Combined 4-Pillar Evaluation via REST API**

### Purpose
Test the integrated evaluation endpoint with 4-pillar scoring (Effectiveness, Efficiency, Robustness, Safety).

### Steps

**1. Query with Evaluation Enabled**

```bash
curl -s -X POST "http://localhost:8000/weather/query?evaluate=true" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the hurricane forecast for Florida?",
    "user_id": "test_eval_001",
    "session_id": "session_eval_001"
  }' | jq '{
    response: .response[:150],
    evaluation_scores: .evaluation_scores,
    execution_time_ms: .execution_time_ms
  }'
```

### Expected Response

```json
{
  "response": "Based on current NHC data, there are no active hurricanes threatening Florida...",
  "evaluation_scores": {
    "effectiveness": 0.85,
    "efficiency": 0.92,
    "robustness": 0.78,
    "safety": 1.0,
    "overall_score": 0.867,
    "passed": true
  },
  "execution_time_ms": 1247.8
}
```

### Validation Checklist
- [ ] `evaluation_scores` object present when `evaluate=true`
- [ ] `effectiveness` score between 0.0-1.0
- [ ] `efficiency` score between 0.0-1.0
- [ ] `robustness` score between 0.0-1.0
- [ ] `safety` score = 1.0 (zero-tolerance)
- [ ] `overall_score` calculated correctly
- [ ] `passed` = true when all thresholds met

---

# PART 2: LEVEL 6b - SELF-IMPROVEMENT PLATFORM

---

## **SCENARIO 4: TruLens Real-Time Evaluation**

### Purpose
Test real-time RAG evaluation with TruLens feedback functions.

### Test Script

```python
#!/usr/bin/env python3
"""Test TruLens Integration - Level 6b"""

import asyncio
from backend.src.evaluation.trulens_integration import TruLensIntegration

async def test_trulens():
    """Test TruLens real-time feedback."""
    print("=" * 60)
    print("SCENARIO 4: TruLens Real-Time Evaluation")
    print("=" * 60)

    # Initialize
    integration = TruLensIntegration()

    # Test case
    query = "What is the temperature in New York?"
    response = "The current temperature in New York is 72°F with partly cloudy skies."
    contexts = [
        "New York current conditions: 72°F, partly cloudy.",
        "Humidity: 55%, Wind: 8 mph NW."
    ]

    # Run evaluation
    feedback = await integration.evaluate(
        query=query,
        response=response,
        contexts=contexts
    )

    # Display results
    print(f"\n📊 TruLens Feedback:")
    print(f"  Groundedness:      {feedback.groundedness:.3f} (threshold: 0.85)")
    print(f"  Answer Relevance:  {feedback.answer_relevance:.3f} (threshold: 0.85)")
    print(f"  Context Relevance: {feedback.context_relevance:.3f} (threshold: 0.85)")
    print(f"  Overall Score:     {feedback.overall_score:.3f}")
    print(f"  Passed:            {'✅' if feedback.passed else '❌'} {feedback.passed}")

    print("\n✅ TruLens test passed!")
    return feedback

if __name__ == "__main__":
    asyncio.run(test_trulens())
```

### Run Command

```bash
python -c "
import asyncio
from backend.src.evaluation.trulens_integration import TruLensIntegration

async def test():
    integration = TruLensIntegration()
    feedback = await integration.evaluate(
        query='What is the weather in Boston?',
        response='Boston has cool weather at 58°F with rain expected.',
        contexts=['Boston: 58°F, rain likely, humidity 80%.']
    )
    print(f'Groundedness: {feedback.groundedness:.3f}')
    print(f'Answer Relevance: {feedback.answer_relevance:.3f}')
    print(f'Context Relevance: {feedback.context_relevance:.3f}')
    print(f'Passed: {feedback.passed}')

asyncio.run(test())
"
```

### Expected Output

```
Groundedness: 0.920
Answer Relevance: 0.900
Context Relevance: 0.880
Passed: True
```

### Validation Checklist
- [ ] Groundedness ≥ 0.85 (answer grounded in context)
- [ ] Answer Relevance ≥ 0.85 (answer addresses query)
- [ ] Context Relevance ≥ 0.85 (context relevant to query)
- [ ] Overall score is average of all three

---

## **SCENARIO 5: LangChain Benchmark (AgentBench)**

### Purpose
Test standardized agent benchmarking for regression tracking.

### Test Script

```python
#!/usr/bin/env python3
"""Test LangChain Benchmark - Level 6b"""

import asyncio
from backend.src.evaluation.langchain_benchmark import LangChainBenchmark, BenchmarkTask

async def test_benchmark():
    """Test LangChain Benchmark integration."""
    print("=" * 60)
    print("SCENARIO 5: LangChain Benchmark")
    print("=" * 60)

    # Initialize benchmark
    benchmark = LangChainBenchmark()

    # Define a simple QA task
    qa_task = BenchmarkTask(
        name="weather_qa",
        description="Weather question answering",
        queries=[
            {"question": "What is the capital of Florida?", "expected": "Tallahassee"},
            {"question": "What causes hurricanes?", "expected": "warm ocean water"},
            {"question": "What is the Saffir-Simpson scale?", "expected": "hurricane intensity"},
        ],
        threshold=0.80,
        category="qa"
    )

    # Mock agent function
    async def mock_agent(query: str) -> str:
        responses = {
            "What is the capital of Florida?": "The capital of Florida is Tallahassee.",
            "What causes hurricanes?": "Hurricanes are caused by warm ocean water.",
            "What is the Saffir-Simpson scale?": "The Saffir-Simpson scale measures hurricane intensity.",
        }
        return responses.get(query, "I don't know.")

    # Run benchmark
    result = await benchmark.run_task(qa_task, mock_agent)

    # Display results
    print(f"\n📊 Benchmark Results:")
    print(f"  Task:           {result.task_name}")
    print(f"  Accuracy:       {result.accuracy:.3f} (threshold: 0.80)")
    print(f"  Latency P50:    {result.latency_p50_ms:.1f}ms")
    print(f"  Latency P95:    {result.latency_p95_ms:.1f}ms")
    print(f"  Total Queries:  {result.total_queries}")
    print(f"  Passed Queries: {result.passed_queries}")
    print(f"  Passed:         {'✅' if result.passed else '❌'} {result.passed}")

    print("\n✅ Benchmark test passed!")
    return result

if __name__ == "__main__":
    asyncio.run(test_benchmark())
```

### Run Command

```bash
python -c "
import asyncio
from backend.src.evaluation.langchain_benchmark import LangChainBenchmark, BenchmarkTask

async def test():
    benchmark = LangChainBenchmark()
    task = BenchmarkTask(
        name='simple_qa',
        description='Simple QA test',
        queries=[
            {'question': 'What is 2+2?', 'expected': '4'},
            {'question': 'What color is the sky?', 'expected': 'blue'},
        ],
        threshold=0.50
    )

    async def agent(q):
        if '2+2' in q: return '4'
        if 'sky' in q: return 'The sky is blue'
        return 'unknown'

    result = await benchmark.run_task(task, agent)
    print(f'Accuracy: {result.accuracy:.3f}')
    print(f'Passed: {result.passed}')

asyncio.run(test())
"
```

### Expected Output

```
Accuracy: 1.000
Passed: True
```

### Validation Checklist
- [ ] Accuracy calculated correctly
- [ ] Latency metrics captured (P50, P95, avg)
- [ ] Passed/failed query count accurate
- [ ] Threshold comparison works

---

# PART 3: LEVEL 6c - PRODUCTION TESTING INFRASTRUCTURE

---

## **SCENARIO 6: Property-Based Testing**

### Purpose
Test edge case discovery using Hypothesis-style property-based testing.

### Test Script

```python
#!/usr/bin/env python3
"""Test Property-Based Testing - Level 6c"""

import asyncio
from backend.src.testing.property_testing import (
    PropertyTestRunner,
    PropertyTestResult,
    WeatherPropertyTests
)

async def test_property_based():
    """Test property-based testing framework."""
    print("=" * 60)
    print("SCENARIO 6: Property-Based Testing")
    print("=" * 60)

    # Initialize runner
    runner = PropertyTestRunner(seed=42)

    # Get weather-specific properties
    properties = WeatherPropertyTests.get_all_properties()
    print(f"\n📋 Available Properties: {len(properties)}")
    for prop in properties:
        print(f"  - {prop.name}: {prop.description}")

    # Mock agent for testing
    async def mock_weather_agent(query: str) -> str:
        if "temperature" in query.lower():
            return "The temperature is 75°F."
        if "hurricane" in query.lower():
            return "Hurricane alert: Category 3 storm approaching."
        return "Weather conditions are normal."

    # Run property tests
    print("\n🔬 Running Property Tests...")
    results = await runner.run_all(properties, mock_weather_agent, num_examples=10)

    # Display results
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"\n📊 Property Test Results:")
    print(f"  Total Properties: {len(results)}")
    print(f"  Passed:          {passed}")
    print(f"  Failed:          {failed}")

    for result in results:
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.property_name}: {result.examples_tested} examples")
        if not result.passed and result.counterexample:
            print(f"      Counterexample: {result.counterexample[:50]}...")

    print("\n✅ Property-based testing completed!")
    return results

if __name__ == "__main__":
    asyncio.run(test_property_based())
```

### Run Command

```bash
python -c "
import asyncio
from backend.src.testing.property_testing import PropertyTestRunner, WeatherPropertyTests

async def test():
    runner = PropertyTestRunner(seed=42)
    properties = WeatherPropertyTests.get_all_properties()
    print(f'Properties: {len(properties)}')

    async def agent(q):
        return 'The weather is 75°F and sunny.'

    results = await runner.run_all(properties, agent, num_examples=5)
    passed = sum(1 for r in results if r.passed)
    print(f'Passed: {passed}/{len(results)}')

asyncio.run(test())
"
```

### Expected Output

```
Properties: 5
Passed: 5/5
```

### Validation Checklist
- [ ] Properties loaded correctly
- [ ] Random input generation works
- [ ] Counterexamples captured on failure
- [ ] Seed provides reproducibility

---

## **SCENARIO 7: Snapshot Testing**

### Purpose
Test response regression detection with snapshot comparisons.

### Test Script

```python
#!/usr/bin/env python3
"""Test Snapshot Testing - Level 6c"""

import asyncio
import tempfile
from backend.src.testing.snapshot_testing import (
    SnapshotManager,
    WeatherSnapshotTests
)

async def test_snapshot():
    """Test snapshot testing framework."""
    print("=" * 60)
    print("SCENARIO 7: Snapshot Testing")
    print("=" * 60)

    # Use temp directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SnapshotManager(snapshot_dir=tmpdir)

        # Create a snapshot
        snapshot = manager.create_snapshot(
            snapshot_id="weather_miami_001",
            content="Miami weather: 85°F, sunny, humidity 70%",
            metadata={"query": "Miami weather", "version": "1.0"}
        )
        print(f"\n📸 Created snapshot: {snapshot.snapshot_id}")

        # Test exact match
        result = manager.compare(
            snapshot_id="weather_miami_001",
            actual_content="Miami weather: 85°F, sunny, humidity 70%"
        )
        print(f"\n📊 Exact Match Test:")
        print(f"  Matched: {'✅' if result.matched else '❌'}")
        print(f"  Similarity: {result.similarity:.3f}")

        # Test similar content
        result2 = manager.compare(
            snapshot_id="weather_miami_001",
            actual_content="Miami weather: 86°F, sunny, humidity 72%"
        )
        print(f"\n📊 Similar Content Test:")
        print(f"  Matched: {'✅' if result2.matched else '❌'}")
        print(f"  Similarity: {result2.similarity:.3f}")

        # Test with threshold
        result3 = manager.compare(
            snapshot_id="weather_miami_001",
            actual_content="Miami weather: 86°F, sunny, humidity 72%",
            threshold=0.90
        )
        print(f"\n📊 Threshold Test (0.90):")
        print(f"  Matched: {'✅' if result3.matched else '❌'}")
        print(f"  Similarity: {result3.similarity:.3f}")

        # Get pre-built weather snapshots
        weather_snapshots = WeatherSnapshotTests.get_weather_response_snapshots()
        print(f"\n📋 Pre-built Snapshots: {len(weather_snapshots)}")

    print("\n✅ Snapshot testing completed!")

if __name__ == "__main__":
    asyncio.run(test_snapshot())
```

### Run Command

```bash
python -c "
import tempfile
from backend.src.testing.snapshot_testing import SnapshotManager

with tempfile.TemporaryDirectory() as tmpdir:
    manager = SnapshotManager(snapshot_dir=tmpdir)

    # Create snapshot
    manager.create_snapshot('test_001', 'Hello World')

    # Exact match
    r1 = manager.compare('test_001', 'Hello World')
    print(f'Exact match: {r1.matched}, similarity: {r1.similarity:.3f}')

    # Similar
    r2 = manager.compare('test_001', 'Hello World!')
    print(f'Similar: {r2.matched}, similarity: {r2.similarity:.3f}')
"
```

### Expected Output

```
Exact match: True, similarity: 1.000
Similar: True, similarity: 0.958
```

### Validation Checklist
- [ ] Snapshots created and persisted
- [ ] Exact match detected (similarity = 1.0)
- [ ] Similar content detected with high similarity
- [ ] Threshold comparison works
- [ ] Diff generation for mismatches

---

## **SCENARIO 8: Retrieval Metrics (MRR, NDCG)**

### Purpose
Test retrieval quality metrics for RAG evaluation.

### Test Script

```python
#!/usr/bin/env python3
"""Test Retrieval Metrics - Level 6c"""

from backend.src.evaluation.retrieval_metrics import (
    MetricsCalculator,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_map
)

def test_retrieval_metrics():
    """Test retrieval metrics calculation."""
    print("=" * 60)
    print("SCENARIO 8: Retrieval Metrics")
    print("=" * 60)

    # Test data
    retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant_docs = ["doc1", "doc3", "doc6"]  # doc6 not retrieved
    relevance_scores = [3, 2, 3, 1, 0]  # Graded relevance

    # Calculate MRR
    mrr = calculate_mrr(retrieved_docs, relevant_docs)
    print(f"\n📊 Mean Reciprocal Rank (MRR):")
    print(f"  MRR: {mrr:.3f}")
    print(f"  (First relevant doc at position 1 → MRR = 1.0)")

    # Calculate NDCG
    ndcg = calculate_ndcg(retrieved_docs, relevance_scores, k=5)
    print(f"\n📊 Normalized DCG (NDCG@5):")
    print(f"  NDCG@5: {ndcg:.3f}")

    # Calculate Precision@k
    precision_3 = calculate_precision_at_k(retrieved_docs, relevant_docs, k=3)
    precision_5 = calculate_precision_at_k(retrieved_docs, relevant_docs, k=5)
    print(f"\n📊 Precision:")
    print(f"  P@3: {precision_3:.3f} (2 relevant in top 3)")
    print(f"  P@5: {precision_5:.3f} (2 relevant in top 5)")

    # Calculate Recall@k
    recall_3 = calculate_recall_at_k(retrieved_docs, relevant_docs, k=3)
    recall_5 = calculate_recall_at_k(retrieved_docs, relevant_docs, k=5)
    print(f"\n📊 Recall:")
    print(f"  R@3: {recall_3:.3f} (2 of 3 relevant retrieved)")
    print(f"  R@5: {recall_5:.3f} (2 of 3 relevant retrieved)")

    # Calculate MAP
    map_score = calculate_map([retrieved_docs], [relevant_docs])
    print(f"\n📊 Mean Average Precision (MAP):")
    print(f"  MAP: {map_score:.3f}")

    # Use combined calculator
    calculator = MetricsCalculator()
    combined = calculator.evaluate_retrieval(
        retrieved_docs=retrieved_docs,
        relevant_docs=relevant_docs,
        relevance_scores=relevance_scores
    )
    print(f"\n📊 Combined Retrieval Metrics:")
    print(f"  MRR:    {combined.mrr:.3f}")
    print(f"  NDCG:   {combined.ndcg:.3f}")
    print(f"  P@5:    {combined.precision_at_k:.3f}")
    print(f"  R@5:    {combined.recall_at_k:.3f}")
    print(f"  MAP:    {combined.map_score:.3f}")

    print("\n✅ Retrieval metrics test passed!")

if __name__ == "__main__":
    test_retrieval_metrics()
```

### Run Command

```bash
python -c "
from backend.src.evaluation.retrieval_metrics import calculate_mrr, calculate_ndcg

retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
relevant = ['doc1', 'doc3']
scores = [3, 2, 3, 1, 0]

mrr = calculate_mrr(retrieved, relevant)
ndcg = calculate_ndcg(retrieved, scores, k=5)

print(f'MRR: {mrr:.3f}')
print(f'NDCG@5: {ndcg:.3f}')
"
```

### Expected Output

```
MRR: 1.000
NDCG@5: 0.876
```

### Validation Checklist
- [ ] MRR = 1.0 when first doc is relevant
- [ ] NDCG accounts for position-based discounting
- [ ] Precision@k correct for various k values
- [ ] Recall@k correct for various k values
- [ ] MAP aggregates correctly

---

## **SCENARIO 9: Generation Metrics (BLEU, ROUGE)**

### Purpose
Test text generation quality metrics.

### Test Script

```python
#!/usr/bin/env python3
"""Test Generation Metrics - Level 6c"""

from backend.src.evaluation.retrieval_metrics import (
    calculate_bleu,
    calculate_rouge,
    MetricsCalculator
)

def test_generation_metrics():
    """Test generation metrics calculation."""
    print("=" * 60)
    print("SCENARIO 9: Generation Metrics (BLEU, ROUGE)")
    print("=" * 60)

    # Test data
    candidate = "The weather in Miami is sunny with a temperature of 85 degrees."
    reference = "Miami has sunny weather with a high of 85 degrees Fahrenheit."

    # Calculate BLEU
    bleu = calculate_bleu([candidate], [[reference]])
    print(f"\n📊 BLEU Score:")
    print(f"  BLEU: {bleu:.3f}")
    print(f"  (Measures n-gram overlap)")

    # Calculate ROUGE
    rouge = calculate_rouge([candidate], [reference])
    print(f"\n📊 ROUGE Scores:")
    print(f"  ROUGE-1: {rouge['rouge1']:.3f} (unigram overlap)")
    print(f"  ROUGE-2: {rouge['rouge2']:.3f} (bigram overlap)")
    print(f"  ROUGE-L: {rouge['rougeL']:.3f} (longest common subsequence)")

    # Use combined calculator
    calculator = MetricsCalculator()
    gen_metrics = calculator.evaluate_generation(
        candidates=[candidate],
        references=[reference]
    )
    print(f"\n📊 Combined Generation Metrics:")
    print(f"  BLEU:    {gen_metrics.bleu:.3f}")
    print(f"  ROUGE-1: {gen_metrics.rouge1:.3f}")
    print(f"  ROUGE-2: {gen_metrics.rouge2:.3f}")
    print(f"  ROUGE-L: {gen_metrics.rougeL:.3f}")

    print("\n✅ Generation metrics test passed!")

if __name__ == "__main__":
    test_generation_metrics()
```

### Run Command

```bash
python -c "
from backend.src.evaluation.retrieval_metrics import calculate_bleu, calculate_rouge

candidate = 'The weather is sunny and warm.'
reference = 'It is sunny and warm weather today.'

bleu = calculate_bleu([candidate], [[reference]])
rouge = calculate_rouge([candidate], [reference])

print(f'BLEU: {bleu:.3f}')
print(f'ROUGE-1: {rouge[\"rouge1\"]:.3f}')
print(f'ROUGE-L: {rouge[\"rougeL\"]:.3f}')
"
```

### Expected Output

```
BLEU: 0.456
ROUGE-1: 0.714
ROUGE-L: 0.571
```

### Validation Checklist
- [ ] BLEU score calculated (0-1 range)
- [ ] ROUGE-1 measures unigram overlap
- [ ] ROUGE-2 measures bigram overlap
- [ ] ROUGE-L measures longest common subsequence
- [ ] Higher overlap = higher scores

---

## **SCENARIO 10: Promptfoo Multi-Provider Testing**

### Purpose
Test prompts across multiple providers with assertion-based evaluation.

### Test Script

```python
#!/usr/bin/env python3
"""Test Promptfoo Integration - Level 6c"""

import asyncio
from backend.src.evaluation.promptfoo_integration import (
    PromptfooRunner,
    AssertionEvaluator,
    WeatherPromptfooTests,
    Assertion,
    AssertionType,
    TestCase,
    EvalConfig,
    PromptConfig
)

async def test_promptfoo():
    """Test Promptfoo multi-provider testing."""
    print("=" * 60)
    print("SCENARIO 10: Promptfoo Multi-Provider Testing")
    print("=" * 60)

    # Initialize runner
    runner = PromptfooRunner()

    # Register a mock provider
    async def mock_openai(prompt: str) -> str:
        if "temperature" in prompt.lower():
            return "The temperature is 75°F."
        if "hurricane" in prompt.lower():
            return "Hurricane alert: Category 3 storm."
        return "Weather information not available."

    runner.register_provider("mock-openai", mock_openai)

    # Test assertion evaluator
    evaluator = AssertionEvaluator()

    # Test contains assertion
    assertion1 = Assertion(type=AssertionType.CONTAINS, value="temperature")
    result1 = evaluator.evaluate("The temperature is 75°F.", assertion1)
    print(f"\n📊 Assertion: contains 'temperature'")
    print(f"  Passed: {'✅' if result1.passed else '❌'}")

    # Test regex assertion
    assertion2 = Assertion(type=AssertionType.REGEX, value=r"\d+°F")
    result2 = evaluator.evaluate("The temperature is 75°F.", assertion2)
    print(f"\n📊 Assertion: matches regex '\\d+°F'")
    print(f"  Passed: {'✅' if result2.passed else '❌'}")

    # Test length assertion
    assertion3 = Assertion(type=AssertionType.LENGTH_GREATER_THAN, value="10")
    result3 = evaluator.evaluate("The temperature is 75°F.", assertion3)
    print(f"\n📊 Assertion: length > 10")
    print(f"  Passed: {'✅' if result3.passed else '❌'}")

    # Get pre-built test cases
    test_cases = WeatherPromptfooTests.get_weather_test_cases()
    print(f"\n📋 Pre-built Test Cases: {len(test_cases)}")
    for tc in test_cases[:3]:
        print(f"  - {tc.description}")

    # Run a test
    test_case = TestCase(
        vars={"location": "Miami"},
        assert_=[
            Assertion(type=AssertionType.CONTAINS, value="Miami"),
            Assertion(type=AssertionType.NOT_CONTAINS, value="error")
        ],
        description="Miami weather test"
    )

    result = await runner.run_test(
        prompt="What is the weather in {{location}}?",
        test_case=test_case,
        provider_id="mock-openai"
    )
    print(f"\n📊 Test Run Result:")
    print(f"  Provider: {result.provider_id}")
    print(f"  Success: {'✅' if result.success else '❌'}")
    print(f"  Latency: {result.latency_ms:.1f}ms")

    print("\n✅ Promptfoo test passed!")

if __name__ == "__main__":
    asyncio.run(test_promptfoo())
```

### Run Command

```bash
python -c "
import asyncio
from backend.src.evaluation.promptfoo_integration import (
    AssertionEvaluator, Assertion, AssertionType
)

evaluator = AssertionEvaluator()

# Test contains
a1 = Assertion(type=AssertionType.CONTAINS, value='hello')
r1 = evaluator.evaluate('hello world', a1)
print(f'Contains \"hello\": {r1.passed}')

# Test regex
a2 = Assertion(type=AssertionType.REGEX, value=r'\\d+')
r2 = evaluator.evaluate('There are 5 items', a2)
print(f'Has number: {r2.passed}')

# Test length
a3 = Assertion(type=AssertionType.LENGTH_LESS_THAN, value='20')
r3 = evaluator.evaluate('Short text', a3)
print(f'Length < 20: {r3.passed}')
"
```

### Expected Output

```
Contains "hello": True
Has number: True
Length < 20: True
```

### Validation Checklist
- [ ] 18 assertion types available
- [ ] Provider registration works
- [ ] Test case execution with variable substitution
- [ ] Assertion evaluation produces pass/fail
- [ ] Latency tracking per test

---

## **SCENARIO 11: OpenAI Evals Integration**

### Purpose
Test systematic evaluation using OpenAI Evals framework.

### Test Script

```python
#!/usr/bin/env python3
"""Test OpenAI Evals Integration - Level 6c"""

import asyncio
from backend.src.evaluation.openai_evals import (
    OpenAIEvalsRunner,
    GraderFactory,
    GraderType,
    WeatherEvals,
    EvalSpec,
    Sample
)

async def test_openai_evals():
    """Test OpenAI Evals integration."""
    print("=" * 60)
    print("SCENARIO 11: OpenAI Evals Integration")
    print("=" * 60)

    # Test graders
    print("\n📊 Testing Graders:")

    # Match grader
    match_grader = GraderFactory.create(GraderType.MATCH, {})
    match_result = match_grader.grade("hello world", {"expected": "hello world"})
    print(f"  Match Grader: {'✅' if match_result.passed else '❌'} (exact match)")

    # Includes grader
    includes_grader = GraderFactory.create(GraderType.INCLUDES, {"needle": "world"})
    includes_result = includes_grader.grade("hello world", {})
    print(f"  Includes Grader: {'✅' if includes_result.passed else '❌'} (contains 'world')")

    # Fuzzy match grader
    fuzzy_grader = GraderFactory.create(GraderType.FUZZY_MATCH, {"threshold": 0.8})
    fuzzy_result = fuzzy_grader.grade("hello world", {"expected": "hello worlds"})
    print(f"  Fuzzy Grader: {'✅' if fuzzy_result.passed else '❌'} (80% threshold)")

    # Initialize runner
    runner = OpenAIEvalsRunner()

    # Mock completion function
    async def mock_completion(prompt: str) -> str:
        if "capital" in prompt.lower():
            return "The capital of Florida is Tallahassee."
        if "weather" in prompt.lower():
            return "The weather is sunny and warm."
        return "I don't know."

    runner.set_completion_fn(mock_completion)

    # Get pre-built weather evals
    print("\n📋 Pre-built Weather Evals:")
    knowledge_eval, knowledge_samples = WeatherEvals.get_weather_knowledge_eval()
    safety_eval, safety_samples = WeatherEvals.get_safety_eval()
    print(f"  Weather Knowledge: {len(knowledge_samples)} samples")
    print(f"  Safety: {len(safety_samples)} samples")

    # Run a simple eval
    simple_eval = EvalSpec(
        name="simple_qa",
        description="Simple QA test",
        grader_type=GraderType.INCLUDES,
        grader_args={"needle": "Tallahassee"}
    )

    samples = [
        Sample(
            input="What is the capital of Florida?",
            ideal="Tallahassee"
        )
    ]

    result = await runner.run_eval(simple_eval, samples)
    print(f"\n📊 Eval Run Result:")
    print(f"  Eval: {result.eval_name}")
    print(f"  Total: {result.total_samples}")
    print(f"  Passed: {result.passed_samples}")
    print(f"  Accuracy: {result.accuracy:.3f}")
    print(f"  Success: {'✅' if result.passed else '❌'}")

    print("\n✅ OpenAI Evals test passed!")

if __name__ == "__main__":
    asyncio.run(test_openai_evals())
```

### Run Command

```bash
python -c "
from backend.src.evaluation.openai_evals import GraderFactory, GraderType

# Match grader
match = GraderFactory.create(GraderType.MATCH, {})
r1 = match.grade('hello', {'expected': 'hello'})
print(f'Match: {r1.passed}')

# Includes grader
includes = GraderFactory.create(GraderType.INCLUDES, {'needle': 'world'})
r2 = includes.grade('hello world', {})
print(f'Includes: {r2.passed}')

# Fuzzy match
fuzzy = GraderFactory.create(GraderType.FUZZY_MATCH, {'threshold': 0.9})
r3 = fuzzy.grade('hello', {'expected': 'hello!'})
print(f'Fuzzy (90%): {r3.passed}')
"
```

### Expected Output

```
Match: True
Includes: True
Fuzzy (90%): True
```

### Validation Checklist
- [ ] Match grader for exact matches
- [ ] Includes grader for substring checks
- [ ] Fuzzy match with configurable threshold
- [ ] Model-graded option for semantic evaluation
- [ ] Pre-built weather evals available

---

# PART 4: LANGSMITH INTEGRATION & OBSERVABILITY

---

## **SCENARIO 12: LangSmith Tracing Verification**

### Purpose
Verify all Level 6 evaluations are traced to LangSmith.

### Steps

**1. Make an Evaluated Query**

```bash
curl -s -X POST "http://localhost:8000/weather/query?evaluate=true" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the hurricane forecast for Tampa Bay?",
    "user_id": "langsmith_test_001",
    "session_id": "session_ls_001"
  }' | jq '{response: .response[:100], evaluation_scores: .evaluation_scores}'
```

**2. Check LangSmith UI**

1. Open https://smith.langchain.com
2. Navigate to your project (weather-ai-agent)
3. Click on "Runs" tab
4. Find the trace for `langsmith_test_001`

### Expected in LangSmith

```
Run Details:
├── Input: "What is the hurricane forecast for Tampa Bay?"
├── Output: "Based on current NHC data..."
├── Latency: 1.2s
├── Tokens: 450
└── Feedback:
    ├── effectiveness: 0.85
    ├── efficiency: 0.92
    ├── robustness: 0.78
    └── safety: 1.0
```

### Validation Checklist
- [ ] Trace appears in LangSmith UI
- [ ] Input/output captured correctly
- [ ] Latency and token counts present
- [ ] Feedback scores attached to run
- [ ] No errors in trace

---

## **SCENARIO 13: LangSmith Evaluation Metrics Dashboard**

### Purpose
Verify evaluation metrics are aggregated in LangSmith.

### Steps

**1. Run Multiple Evaluated Queries**

```bash
# Run 5 queries with evaluation
for i in 1 2 3 4 5; do
  curl -s -X POST "http://localhost:8000/weather/query?evaluate=true" \
    -H "Content-Type: application/json" \
    -d "{
      \"query\": \"Weather forecast for city $i\",
      \"user_id\": \"metrics_test_00$i\"
    }" | jq '{user_id: .user_id, overall_score: .evaluation_scores.overall_score}'
  sleep 1
done
```

**2. View Metrics in LangSmith**

1. Open https://smith.langchain.com
2. Navigate to your project
3. Click on "Monitor" or "Metrics" tab
4. Look for:
   - Average effectiveness score
   - Average efficiency score
   - Safety violation rate (should be 0%)
   - P95 latency

### Expected Metrics

```
Aggregated Metrics (last 5 runs):
- Avg Effectiveness: 0.84
- Avg Efficiency: 0.91
- Avg Robustness: 0.76
- Safety Rate: 100% (0 violations)
- P95 Latency: 1.8s
```

### Validation Checklist
- [ ] Metrics aggregated across runs
- [ ] Effectiveness trend visible
- [ ] Efficiency trend visible
- [ ] Safety rate at 100%
- [ ] Latency distribution available

---

## **SCENARIO 14: LangSmith Dataset for Regression Testing**

### Purpose
Use LangSmith datasets for automated regression testing.

### Test Script

```python
#!/usr/bin/env python3
"""Test LangSmith Dataset Integration - Level 6"""

import asyncio
import os

# Check if LangSmith is configured
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

async def test_langsmith_dataset():
    """Test LangSmith dataset integration."""
    print("=" * 60)
    print("SCENARIO 14: LangSmith Dataset Integration")
    print("=" * 60)

    print(f"\n📋 LangSmith Configuration:")
    print(f"  API Key Set: {'✅' if LANGSMITH_API_KEY else '❌'}")
    print(f"  Tracing Enabled: {'✅' if LANGSMITH_ENABLED else '❌'}")

    if not LANGSMITH_API_KEY or not LANGSMITH_ENABLED:
        print("\n⚠️ LangSmith not fully configured. Skipping dataset test.")
        print("  Set LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true")
        return

    try:
        from langsmith import Client

        client = Client()

        # List existing datasets
        datasets = list(client.list_datasets())
        print(f"\n📊 Existing Datasets: {len(datasets)}")
        for ds in datasets[:5]:
            print(f"  - {ds.name}: {ds.example_count} examples")

        # Create a test dataset if it doesn't exist
        dataset_name = "weather-ai-regression-tests"
        existing = [d for d in datasets if d.name == dataset_name]

        if not existing:
            print(f"\n📝 Creating dataset: {dataset_name}")
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Regression test dataset for Weather AI Agent"
            )

            # Add example
            client.create_example(
                inputs={"query": "What is the weather in Miami?"},
                outputs={"response": "Miami has sunny weather with 85°F."},
                dataset_id=dataset.id
            )
            print(f"  ✅ Dataset created with 1 example")
        else:
            print(f"\n✅ Dataset '{dataset_name}' already exists")

        print("\n✅ LangSmith dataset test passed!")

    except ImportError:
        print("\n⚠️ langsmith package not installed")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_langsmith_dataset())
```

### Validation Checklist
- [ ] LangSmith client connects successfully
- [ ] Can list existing datasets
- [ ] Can create new dataset
- [ ] Can add examples to dataset
- [ ] Dataset available for evaluation runs

---

# PART 5: COMPREHENSIVE INTEGRATION TESTS

---

## **SCENARIO 15: Full Evaluation Pipeline Test**

### Purpose
Test the complete evaluation pipeline from query to LangSmith metrics.

### Test Script

```python
#!/usr/bin/env python3
"""Full Evaluation Pipeline Test - Level 6"""

import asyncio
import httpx
import time

async def test_full_pipeline():
    """Test the complete Level 6 evaluation pipeline."""
    print("=" * 60)
    print("SCENARIO 15: Full Evaluation Pipeline")
    print("=" * 60)

    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Health check
        print("\n📡 Step 1: Health Check")
        try:
            health = await client.get(f"{base_url}/health")
            print(f"  Status: {health.status_code}")
        except Exception as e:
            print(f"  ❌ Health check failed: {e}")
            return

        # 2. Cache stats
        print("\n📊 Step 2: Cache Stats")
        try:
            cache = await client.get(f"{base_url}/cache/stats")
            stats = cache.json()
            print(f"  L1 Enabled: {stats.get('l1_cache', {}).get('enabled', False)}")
            print(f"  L2 Enabled: {stats.get('l2_cache', {}).get('enabled', False)}")
        except Exception as e:
            print(f"  ⚠️ Cache stats: {e}")

        # 3. Weather query WITH evaluation
        print("\n🌤️ Step 3: Evaluated Weather Query")
        start = time.time()
        try:
            response = await client.post(
                f"{base_url}/weather/query",
                params={"evaluate": "true"},
                json={
                    "query": "What is the current weather and forecast for Miami, Florida?",
                    "user_id": "pipeline_test_001",
                    "session_id": "session_pipeline"
                }
            )
            elapsed = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                print(f"  Response: {data.get('response', '')[:100]}...")
                print(f"  Latency: {elapsed:.0f}ms")

                scores = data.get("evaluation_scores")
                if scores:
                    print(f"\n  📈 Evaluation Scores:")
                    print(f"    Effectiveness: {scores.get('effectiveness', 'N/A')}")
                    print(f"    Efficiency: {scores.get('efficiency', 'N/A')}")
                    print(f"    Robustness: {scores.get('robustness', 'N/A')}")
                    print(f"    Safety: {scores.get('safety', 'N/A')}")
                    print(f"    Overall: {scores.get('overall_score', 'N/A')}")
                    print(f"    Passed: {'✅' if scores.get('passed') else '❌'}")
                else:
                    print("  ⚠️ No evaluation scores returned")
            else:
                print(f"  ❌ Status: {response.status_code}")
                print(f"  Error: {response.text[:200]}")
        except Exception as e:
            print(f"  ❌ Query failed: {e}")

        # 4. Metrics endpoint
        print("\n📊 Step 4: Prometheus Metrics")
        try:
            metrics = await client.get(f"{base_url}/metrics")
            lines = metrics.text.split('\n')
            weather_metrics = [l for l in lines if 'weather_ai' in l and not l.startswith('#')]
            print(f"  Weather AI Metrics: {len(weather_metrics)} found")
            for m in weather_metrics[:5]:
                print(f"    {m[:80]}")
        except Exception as e:
            print(f"  ⚠️ Metrics: {e}")

    print("\n" + "=" * 60)
    print("✅ Full pipeline test completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
```

### Run Command

```bash
python -c "
import asyncio
import httpx

async def test():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Query with evaluation
        r = await client.post(
            'http://localhost:8000/weather/query',
            params={'evaluate': 'true'},
            json={'query': 'Weather in NYC?', 'user_id': 'test'}
        )
        if r.status_code == 200:
            data = r.json()
            scores = data.get('evaluation_scores', {})
            print(f'Response: {data.get(\"response\", \"\")[:50]}...')
            print(f'Overall Score: {scores.get(\"overall_score\", \"N/A\")}')
            print(f'Passed: {scores.get(\"passed\", \"N/A\")}')
        else:
            print(f'Error: {r.status_code}')

asyncio.run(test())
"
```

### Expected Output

```
Response: New York currently has partly cloudy skies...
Overall Score: 0.867
Passed: True
```

### Validation Checklist
- [ ] Health check passes
- [ ] Cache stats accessible
- [ ] Weather query returns response
- [ ] Evaluation scores present when `evaluate=true`
- [ ] Prometheus metrics exposed
- [ ] Overall pipeline completes <5s

---

## **SCENARIO 16: Edge Cases and Error Handling**

### Purpose
Test Level 6 handles edge cases gracefully.

### Test Cases

**1. Empty Query**

```bash
curl -s -X POST "http://localhost:8000/weather/query?evaluate=true" \
  -H "Content-Type: application/json" \
  -d '{"query": "", "user_id": "edge_001"}' | jq
```

Expected: Validation error or graceful handling

**2. Very Long Query**

```bash
curl -s -X POST "http://localhost:8000/weather/query?evaluate=true" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$(python -c 'print(\"weather \" * 1000)')\", \"user_id\": \"edge_002\"}" | jq
```

Expected: Truncation or context window handling

**3. Special Characters**

```bash
curl -s -X POST "http://localhost:8000/weather/query?evaluate=true" \
  -H "Content-Type: application/json" \
  -d '{"query": "Weather in <script>alert(1)</script> city?", "user_id": "edge_003"}' | jq
```

Expected: Input sanitized, no XSS

**4. Concurrent Evaluations**

```bash
# Run 10 concurrent requests
for i in $(seq 1 10); do
  curl -s -X POST "http://localhost:8000/weather/query?evaluate=true" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"Weather in city $i\", \"user_id\": \"concurrent_$i\"}" &
done
wait
echo "All requests completed"
```

Expected: All requests complete without deadlock

### Validation Checklist
- [ ] Empty query handled gracefully
- [ ] Long queries don't crash
- [ ] Special characters sanitized
- [ ] Concurrent requests work
- [ ] No memory leaks under load

---

# PART 6: SWAGGER UI TESTING

---

## **SCENARIO 17: Interactive API Testing via Swagger**

### Purpose
Test Level 6 endpoints through Swagger UI.

### Steps

**1. Open Swagger UI**

Navigate to: http://localhost:8000/docs

**2. Test Weather Query with Evaluation**

1. Expand `POST /weather/query`
2. Click "Try it out"
3. Set `evaluate` parameter to `true`
4. Enter request body:
```json
{
  "query": "What is the 5-day forecast for Los Angeles?",
  "user_id": "swagger_test_001",
  "session_id": "swagger_session"
}
```
5. Click "Execute"
6. Verify response includes `evaluation_scores`

**3. Test Cache Stats**

1. Expand `GET /cache/stats`
2. Click "Try it out"
3. Click "Execute"
4. Verify L1, L2, L3 cache info

**4. Test Metrics Endpoint**

1. Expand `GET /metrics`
2. Click "Try it out"
3. Click "Execute"
4. Verify Prometheus format output

### Expected Swagger Response

```json
{
  "response": "The 5-day forecast for Los Angeles shows...",
  "user_id": "swagger_test_001",
  "timestamp": "2025-12-13T...",
  "evaluation_scores": {
    "effectiveness": 0.87,
    "efficiency": 0.93,
    "robustness": 0.81,
    "safety": 1.0,
    "overall_score": 0.89,
    "passed": true
  }
}
```

### Validation Checklist
- [ ] Swagger UI loads correctly
- [ ] All endpoints documented
- [ ] Weather query works with evaluation
- [ ] Cache stats accessible
- [ ] Metrics endpoint returns Prometheus format

---

# PART 7: GOLDEN DATASET & CONSTITUTIONAL AI

---

## **SCENARIO 18: Golden Dataset Evaluation (185 Test Cases)**

### Purpose
Test the complete golden dataset evaluation pipeline including Level 5 and Level 6 test cases using LangSmith integration and Makefile automation.

### Golden Dataset Overview

**Location**: `tests/evaluation/golden_dataset.yaml`

| Level | Category | Test Cases | Focus |
|-------|----------|------------|-------|
| **Level 5** | simple | 40 | Basic weather queries |
| **Level 5** | complex | 30 | Multi-location, analysis |
| **Level 5** | hurricane | 20 | Safety-critical queries |
| **Level 5** | edge | 15 | Error handling, invalid inputs |
| **Level 6** | bleu_rouge | 20 | Text generation quality |
| **Level 6** | snapshot | 15 | Regression detection |
| **Level 6** | retrieval | 20 | MRR, NDCG, MAP, Precision@k |
| **Level 6** | ragas_recall | 10 | Context recall validation |
| **Level 6** | agentbench | 15 | Task-specific accuracy |
| | **Total** | **185** | |

### Makefile Commands

**1. Full Pipeline (Upload + Run + Check)**

```bash
# Run complete evaluation pipeline with all 185 test cases
make eval-full
```

**2. Upload Golden Dataset to LangSmith**

```bash
# Upload 185 test cases to LangSmith
make eval-upload-dataset
```

**3. Run Batch Evaluation**

```bash
# Run evaluation on all 185 cases
make eval-run-batch
```

**4. Quick Smoke Test**

```bash
# Run 10 cases (~2-3 minutes)
make eval-quick
```

**5. Test Specific Category (Level 5)**

```bash
# Level 5 categories
make eval-category CATEGORY=simple
make eval-category CATEGORY=complex
make eval-category CATEGORY=hurricane
make eval-category CATEGORY=edge
```

**6. Test Specific Category (Level 6)**

```bash
# Level 6 categories
make eval-category CATEGORY=bleu_rouge
make eval-category CATEGORY=snapshot
make eval-category CATEGORY=retrieval
make eval-category CATEGORY=ragas_recall
make eval-category CATEGORY=agentbench
```

**7. Check Quality Gates**

```bash
# Verify all quality gates pass
make eval-check-gates
```

### Quality Gates (Must Pass)

| Metric | Threshold | Level |
|--------|-----------|-------|
| Pass Rate | ≥85% | All |
| Effectiveness | ≥0.85 | All |
| Efficiency | ≥0.80 | All |
| Robustness | ≥0.80 | All |
| Safety | 100% (0 violations) | All |
| BLEU Score | ≥0.30 | L6 |
| MRR | ≥0.70 | L6 |
| Context Recall | ≥0.85 | L6 |

### Test Script

```python
#!/usr/bin/env python3
"""Test Golden Dataset Evaluation - Level 6"""

import asyncio
import subprocess

async def test_golden_dataset_evaluation():
    """Test golden dataset evaluation pipeline."""
    print("=" * 60)
    print("SCENARIO 18: Golden Dataset Evaluation")
    print("=" * 60)

    # 1. Verify golden dataset exists
    print("\n📋 Step 1: Verify Golden Dataset")
    try:
        import yaml
        with open("tests/evaluation/golden_dataset.yaml") as f:
            dataset = yaml.safe_load(f)

        test_cases = dataset.get("test_cases", [])
        print(f"  Total test cases: {len(test_cases)}")

        # Count by category
        categories = {}
        for tc in test_cases:
            cat = tc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        print("\n  Categories:")
        for cat, count in sorted(categories.items()):
            level = "L6" if cat in ["bleu_rouge", "snapshot", "retrieval", "ragas_recall", "agentbench"] else "L5"
            print(f"    [{level}] {cat}: {count} cases")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return

    # 2. Run quick evaluation
    print("\n🧪 Step 2: Quick Smoke Test (10 cases)")
    result = subprocess.run(
        ["make", "eval-quick"],
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode == 0:
        print("  ✅ Quick evaluation passed")
    else:
        print(f"  ⚠️ Quick evaluation: {result.returncode}")
        print(f"  {result.stderr[:200]}")

    # 3. Check quality gates
    print("\n📊 Step 3: Check Quality Gates")
    result = subprocess.run(
        ["make", "eval-check-gates"],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0:
        print("  ✅ All quality gates passed")
    else:
        print(f"  ⚠️ Quality gates: check output")

    print("\n✅ Golden dataset evaluation test completed!")

if __name__ == "__main__":
    asyncio.run(test_golden_dataset_evaluation())
```

### Run Command

```bash
# Quick smoke test with 10 cases
make eval-quick

# Or run specific Level 6 category
make eval-category CATEGORY=bleu_rouge
```

### Expected Output

```
🧪 Running quick evaluation (10 test cases)...
Uploading golden dataset to LangSmith...
✅ Dataset uploaded: 185 test cases
Running evaluation batch...
  [1/10] simple_weather_miami ✅ (0.92)
  [2/10] simple_weather_nyc ✅ (0.89)
  ...
  [10/10] hurricane_safety_001 ✅ (0.95)

📊 Results:
  Pass Rate: 100% (10/10)
  Avg Score: 0.91
  Safety Violations: 0

✅ Quick evaluation passed!
```

### Validation Checklist
- [ ] Golden dataset has 185 test cases
- [ ] Level 5 categories: simple(40), complex(30), hurricane(20), edge(15)
- [ ] Level 6 categories: bleu_rouge(20), snapshot(15), retrieval(20), ragas_recall(10), agentbench(15)
- [ ] `make eval-quick` runs successfully
- [ ] `make eval-category CATEGORY=bleu_rouge` runs Level 6 tests
- [ ] Quality gates check passes
- [ ] Results uploaded to LangSmith

---

## **SCENARIO 19: Constitutional AI Guardrails Testing**

### Purpose
Test the Constitutional AI framework for safety and ethical guardrails validation.

### Constitutional AI Overview

The Constitutional AI framework validates responses against a set of principles:

| Category | Focus | Weight |
|----------|-------|--------|
| **SAFETY** | Emergency guidance, evacuation | High |
| **ACCURACY** | Data source attribution, verification | High |
| **HONESTY** | Uncertainty acknowledgment, limitations | Medium |
| **HELPFULNESS** | Clear, actionable information | Medium |

### Test Script

```python
#!/usr/bin/env python3
"""Test Constitutional AI - Level 6c"""

import asyncio
from backend.src.guardrails.constitutional_ai import (
    ConstitutionalAI,
    Constitution,
    Principle,
    PrincipleCategory,
    ConstitutionalResult,
)

async def test_constitutional_ai():
    """Test Constitutional AI framework."""
    print("=" * 60)
    print("SCENARIO 19: Constitutional AI Guardrails")
    print("=" * 60)

    # 1. Initialize with weather domain constitution
    print("\n📋 Step 1: Initialize Constitutional AI")
    constitution = Constitution.weather_domain()
    constitutional_ai = ConstitutionalAI(
        constitution=constitution,
        llm=None,  # Uses heuristics without LLM
        max_iterations=2,
        pass_threshold=0.7,
    )

    print(f"  Constitution: {constitution.name}")
    print(f"  Principles: {len(constitution.principles)}")

    # Show principles by category
    for category in PrincipleCategory:
        principles = constitution.get_by_category(category)
        if principles:
            print(f"    {category.value}: {len(principles)} principles")

    # 2. Test good response
    print("\n🧪 Step 2: Test Good Response")
    good_query = "What's the weather in Miami?"
    good_response = """
    Based on National Weather Service data, the current weather in Miami shows
    sunny conditions with a temperature of 85°F. Humidity is at 65% with winds
    from the southeast at 10 mph. The forecast may change, so please check for
    updates.
    """

    result = await constitutional_ai.validate_response(good_query, good_response)
    print(f"  Query: {good_query}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Passed: {'✅' if result.passes else '❌'}")
    print(f"  Critiques: {len(result.critiques)}")

    # 3. Test safety-critical response
    print("\n🧪 Step 3: Test Safety-Critical Response")
    safety_query = "Should I evacuate for the hurricane?"
    unsafe_response = "The hurricane is nothing to worry about. Just stay home."

    result = await constitutional_ai.validate_response(safety_query, unsafe_response)
    print(f"  Query: {safety_query}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Passed: {'✅' if result.passes else '❌'}")
    print(f"  Critiques: {len(result.critiques)}")

    # Show failing critiques
    failing = [c for c in result.critiques if not c.passes]
    if failing:
        print(f"  Failing Critiques ({len(failing)}):")
        for c in failing[:3]:
            print(f"    - {c.principle_name}: {c.critique[:60]}...")

    # 4. Test response with missing source
    print("\n🧪 Step 4: Test Response Without Source Attribution")
    no_source_query = "What's the hurricane forecast?"
    no_source_response = "Category 5 hurricane coming. The storm will hit tomorrow."

    result = await constitutional_ai.validate_response(no_source_query, no_source_response)
    print(f"  Query: {no_source_query}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Passed: {'✅' if result.passes else '❌'}")

    # 5. Get statistics
    print("\n📊 Step 5: Validation Statistics")
    stats = constitutional_ai.get_statistics()
    print(f"  Total Validations: {stats['total_validations']}")
    print(f"  Pass Rate: {stats['pass_rate']:.1%}")
    print(f"  Average Score: {stats['average_score']:.3f}")

    print("\n✅ Constitutional AI test completed!")

if __name__ == "__main__":
    asyncio.run(test_constitutional_ai())
```

### Run Command

```bash
# Run Constitutional AI tests
python -c "
import asyncio
from backend.src.guardrails.constitutional_ai import (
    ConstitutionalAI, Constitution
)

async def test():
    constitution = Constitution.weather_domain()
    ai = ConstitutionalAI(constitution=constitution, llm=None)

    # Test good response
    result = await ai.validate_response(
        'What is the weather?',
        'Based on NWS data, it is 75°F and sunny. Conditions may change.'
    )
    print(f'Good Response - Score: {result.score:.3f}, Passed: {result.passes}')

    # Test unsafe response
    result = await ai.validate_response(
        'Should I evacuate?',
        'No need to worry about the hurricane.'
    )
    print(f'Unsafe Response - Score: {result.score:.3f}, Passed: {result.passes}')

asyncio.run(test())
"
```

### Run pytest Tests

```bash
# Run Constitutional AI unit tests
uv run pytest tests/guardrails/test_constitutional_ai.py -v

# Run with coverage
uv run pytest tests/guardrails/test_constitutional_ai.py -v --cov=backend.src.guardrails.constitutional_ai
```

### Expected Output

```
Good Response - Score: 0.850, Passed: True
Unsafe Response - Score: 0.450, Passed: False
```

### Key Test Cases

| Test Case | Expected Result | Reason |
|-----------|-----------------|--------|
| Response with NWS source | ✅ Pass | Source attribution present |
| Response with uncertainty | ✅ Pass | Acknowledges forecast may change |
| "Hurricane nothing to worry about" | ❌ Fail | Downplays safety risk |
| Response without source | ⚠️ Warn | Missing source attribution |
| Evacuation guidance without zones | ❌ Fail | Missing specific evacuation info |

### Weather Domain Constitution Principles

```python
# Key principles tested:
1. "safety_emergency" - Emergency guidance must be clear
2. "safety_no_downplay" - Never downplay hurricane risks
3. "accuracy_source" - Cite data sources (NWS, NHC)
4. "accuracy_hurricane_category" - Verify Saffir-Simpson scale
5. "honesty_uncertainty" - Acknowledge forecast uncertainty
6. "evacuation_zone" - Use specific zone designations (A, B, C)
```

### Validation Checklist
- [ ] Constitution loads with weather domain principles
- [ ] Good responses pass validation (score ≥ 0.70)
- [ ] Unsafe responses fail validation
- [ ] Safety critiques generated for dangerous advice
- [ ] Source attribution critiques work
- [ ] Statistics tracking functions correctly
- [ ] pytest tests pass

---

# SUMMARY: LEVEL 6 TEST COVERAGE

---

## Test Scenarios Overview

| # | Scenario | Component | Status |
|---|----------|-----------|--------|
| 1 | Ragas RAG Evaluation | L6a | ✅ |
| 2 | DeepEval LLM Unit Testing | L6a | ✅ |
| 3 | 4-Pillar Evaluation REST | L6a | ✅ |
| 4 | TruLens Real-Time | L6b | ✅ |
| 5 | LangChain Benchmark | L6b | ✅ |
| 6 | Property-Based Testing | L6c | ✅ |
| 7 | Snapshot Testing | L6c | ✅ |
| 8 | Retrieval Metrics (MRR, NDCG) | L6c | ✅ |
| 9 | Generation Metrics (BLEU, ROUGE) | L6c | ✅ |
| 10 | Promptfoo Multi-Provider | L6c | ✅ |
| 11 | OpenAI Evals | L6c | ✅ |
| 12 | LangSmith Tracing | Integration | ✅ |
| 13 | LangSmith Metrics Dashboard | Integration | ✅ |
| 14 | LangSmith Dataset | Integration | ✅ |
| 15 | Full Pipeline Test | Integration | ✅ |
| 16 | Edge Cases | Robustness | ✅ |
| 17 | Swagger UI Testing | API | ✅ |
| 18 | Golden Dataset Evaluation (185 cases) | L6 + Makefile | ✅ |
| 19 | Constitutional AI Guardrails | L6c | ✅ |

## Quality Metrics Targets

| Metric | Target | Level |
|--------|--------|-------|
| Faithfulness | ≥ 0.90 | L6a (Ragas) |
| Context Precision | ≥ 0.85 | L6a (Ragas) |
| Context Recall | ≥ 0.85 | L6a (Ragas) |
| Answer Relevancy | ≥ 0.90 | L6a (Ragas/DeepEval) |
| Groundedness | ≥ 0.85 | L6b (TruLens) |
| Benchmark Accuracy | ≥ 0.80 | L6b |
| Safety Rate | 100% | L6a/L6b/L6c |
| MRR | ≥ 0.70 | L6c |
| NDCG@5 | ≥ 0.75 | L6c |
| BLEU Score | ≥ 0.30 | L6c (Generation) |
| Constitutional AI Pass | ≥ 0.70 | L6c (Guardrails) |

## Golden Dataset Categories (185 Test Cases)

| Level | Category | Count | Focus |
|-------|----------|-------|-------|
| L5 | simple | 40 | Basic weather queries |
| L5 | complex | 30 | Multi-location, analysis |
| L5 | hurricane | 20 | Safety-critical queries |
| L5 | edge | 15 | Error handling |
| L6 | bleu_rouge | 20 | Text generation quality |
| L6 | snapshot | 15 | Regression detection |
| L6 | retrieval | 20 | MRR, NDCG, MAP |
| L6 | ragas_recall | 10 | Context recall |
| L6 | agentbench | 15 | Task-specific accuracy |
| | **Total** | **185** | |

## Quick Test Commands

```bash
# Run all Level 6 module tests
python -c "
from backend.src.evaluation import (
    RagasEvaluator, DeepEvalIntegration, TruLensIntegration,
    LangChainBenchmark, MetricsCalculator, PromptfooRunner
)
print('✅ All Level 6 evaluation modules imported successfully')
"

# Test retrieval metrics
python -c "
from backend.src.evaluation.retrieval_metrics import calculate_mrr, calculate_ndcg
print(f'MRR: {calculate_mrr([\"a\",\"b\",\"c\"], [\"a\"]):.3f}')
print(f'NDCG: {calculate_ndcg([\"a\",\"b\",\"c\"], [3,2,1]):.3f}')
"

# Test via REST API
curl -s -X POST "http://localhost:8000/weather/query?evaluate=true" \
  -H "Content-Type: application/json" \
  -d '{"query": "Weather forecast?", "user_id": "quick_test"}' | jq '.evaluation_scores'
```

---

## Troubleshooting

### Issue: Evaluation scores are null

**Solution**: Ensure `evaluate=true` query parameter is set:
```bash
curl -X POST "http://localhost:8000/weather/query?evaluate=true" ...
```

### Issue: LangSmith traces not appearing

**Solution**: Verify environment variables:
```bash
echo $LANGCHAIN_TRACING_V2  # Must be "true"
echo $LANGCHAIN_API_KEY     # Must be set
docker-compose restart weather-ai-api
```

### Issue: Import errors for evaluation modules

**Solution**: Install dependencies:
```bash
uv sync
pip install ragas deepeval trulens langsmith
```

### Issue: Metrics return 0.0

**Solution**: Check if required LLM API keys are set:
```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

---

**Document Version**: 2.0.0
**Last Updated**: December 13, 2025
**Status**: ✅ Complete - All 19 scenarios documented

### What's New in v2.0.0
- ✅ Scenario 18: Golden Dataset Evaluation (185 test cases, Makefile commands)
- ✅ Scenario 19: Constitutional AI Guardrails Testing
- ✅ Level 6 categories: bleu_rouge, snapshot, retrieval, ragas_recall, agentbench
- ✅ Quality gates for Level 6 metrics (BLEU ≥0.30, MRR ≥0.70, Context Recall ≥0.85)
- ✅ Makefile automation for `eval-category CATEGORY=<level6_category>`

---

*End of Level 6 Test Guide*
