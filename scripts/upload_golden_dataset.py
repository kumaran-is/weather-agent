#!/usr/bin/env python3
"""Upload golden dataset to LangSmith for evaluation tracking.

This script uploads the golden dataset (tests/evaluation/golden_dataset.yaml)
to LangSmith as a named dataset for batch evaluation and regression testing.

Supports Level 5 and Level 6 evaluation types:
- Level 5: 4-pillar evaluation (effectiveness, efficiency, robustness, safety)
- Level 6: BLEU/ROUGE, Snapshot, Retrieval metrics, RAGAS, AgentBench

Usage:
    uv run python scripts/upload_golden_dataset.py

    # Or via Makefile:
    make eval-upload-dataset

    # Upload only specific categories:
    uv run python scripts/upload_golden_dataset.py --category bleu_rouge
    uv run python scripts/upload_golden_dataset.py --category retrieval
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from .env file
load_dotenv()

# Configuration
DATASET_PATH = Path("tests/evaluation/golden_dataset.yaml")
DATASET_NAME = "weather-ai-golden-dataset"
DATASET_DESCRIPTION = """
Weather AI Agent Golden Dataset (Level 5b + Level 6)
- 185 test cases for regression testing
- Level 5 Categories: simple (40), complex (30), hurricane (20), edge (15)
- Level 6 Categories: bleu_rouge (20), snapshot (15), retrieval (20), ragas_recall (10), agentbench (15)
- Used for multi-pillar evaluation (effectiveness, efficiency, robustness, safety)
- Level 6 adds: BLEU/ROUGE, Snapshot testing, Retrieval metrics, RAGAS, AgentBench
- Pass threshold: 85% overall, 100% safety

Level 5 Test Categories:
- Simple: Basic weather queries (single location)
- Complex: Multi-location comparisons, analysis
- Hurricane: Safety-critical (Saffir-Simpson validation)
- Edge: Error handling, invalid inputs

Level 6 Test Categories:
- BLEU/ROUGE: Text generation quality with reference answers
- Snapshot: Regression detection with baseline comparisons
- Retrieval: MRR, NDCG, MAP, Precision@k, Recall@k metrics
- RAGAS Recall: Context recall with ground truth validation
- AgentBench: Task-specific accuracy with structured outputs

Created: {datetime.now().strftime("%Y-%m-%d")}
Version: 2.0.0
"""

# Level 6 evaluation types
LEVEL6_EVAL_TYPES = ["bleu_rouge", "snapshot", "retrieval", "ragas_context_recall", "agentbench"]
LEVEL6_CATEGORIES = ["bleu_rouge", "snapshot", "retrieval", "ragas_recall", "agentbench"]


def load_golden_dataset(path: Path) -> list[dict]:
    """Load test cases from YAML file."""
    if not path.exists():
        print(f"❌ Dataset file not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = yaml.safe_load(f)

    test_cases = data.get("golden_dataset", [])
    if not test_cases:
        print("❌ No test cases found in dataset")
        sys.exit(1)

    return test_cases


def upload_to_langsmith(test_cases: list[dict], category_filter: str | None = None) -> str:
    """Upload dataset to LangSmith.

    Args:
        test_cases: List of test cases from YAML
        category_filter: Optional category to filter (e.g., "bleu_rouge", "retrieval")

    Returns:
        Dataset ID
    """
    client = Client()

    # Filter test cases by category if specified
    if category_filter:
        test_cases = [c for c in test_cases if c.get("category") == category_filter]
        if not test_cases:
            print(f"❌ No test cases found for category: {category_filter}")
            sys.exit(1)
        print(f"📋 Filtered to {len(test_cases)} cases for category: {category_filter}")

    # Determine dataset name (append category if filtering)
    dataset_name = DATASET_NAME
    if category_filter:
        dataset_name = f"{DATASET_NAME}-{category_filter}"

    # Check if dataset exists
    existing = list(client.list_datasets(dataset_name=dataset_name))

    if existing:
        print(f"⚠️  Dataset '{dataset_name}' already exists. Updating...")
        dataset = existing[0]
        # Delete existing examples for clean upload
        examples = list(client.list_examples(dataset_id=dataset.id))
        for ex in examples:
            client.delete_example(ex.id)
        print(f"   Deleted {len(examples)} existing examples")
    else:
        print(f"📦 Creating new dataset: {dataset_name}")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=DATASET_DESCRIPTION,
        )

    # Upload each test case as an example
    examples_created = 0
    level6_count = 0

    for case in test_cases:
        category = case.get("category", "unknown")
        eval_type = case.get("eval_type", "standard")

        # Create input/output structure
        inputs = {
            "query": case["query"],
            "test_id": case["id"],
            "category": category,
            "eval_type": eval_type,
        }

        # Base outputs for all test types
        outputs = {
            "expected_tools": case.get("expected_tools", []),
            "expected_answer_contains": case.get("expected_answer_contains", []),
            "expected_answer": case.get("expected_answer"),
        }

        # Level 6 specific outputs based on eval_type
        if eval_type == "generation" or category == "bleu_rouge":
            # BLEU/ROUGE evaluation - needs reference answer
            outputs["reference_answer"] = case.get("reference_answer")
            outputs["bleu_threshold"] = case.get("success_criteria", {}).get("bleu_score", ">0.3")
            outputs["rouge_1_threshold"] = case.get("success_criteria", {}).get("rouge_1", ">0.4")
            outputs["rouge_l_threshold"] = case.get("success_criteria", {}).get("rouge_l", ">0.35")
            level6_count += 1

        elif eval_type == "snapshot" or category == "snapshot":
            # Snapshot testing - needs baseline
            outputs["snapshot_baseline"] = case.get("snapshot_baseline")
            outputs["snapshot_similarity_threshold"] = case.get("success_criteria", {}).get("snapshot_similarity", ">0.75")
            level6_count += 1

        elif eval_type == "retrieval" or category == "retrieval":
            # Retrieval metrics - needs relevant docs with scores
            outputs["relevant_docs"] = case.get("relevant_docs", [])
            outputs["mrr_threshold"] = case.get("success_criteria", {}).get("mrr", ">0.7")
            outputs["ndcg_at_5_threshold"] = case.get("success_criteria", {}).get("ndcg_at_5", ">0.65")
            outputs["precision_at_3_threshold"] = case.get("success_criteria", {}).get("precision_at_3", ">0.6")
            outputs["recall_at_5_threshold"] = case.get("success_criteria", {}).get("recall_at_5", ">0.7")
            outputs["map_threshold"] = case.get("success_criteria", {}).get("map", ">0.65")
            level6_count += 1

        elif eval_type == "ragas_context_recall" or category == "ragas_recall":
            # RAGAS Context Recall - needs ground truth info
            outputs["ground_truth_info"] = case.get("ground_truth_info", [])
            outputs["context_recall_threshold"] = case.get("success_criteria", {}).get("context_recall", ">0.85")
            level6_count += 1

        elif eval_type == "agentbench" or category == "agentbench":
            # AgentBench - needs task spec and expected result
            outputs["task_spec"] = case.get("task_spec", {})
            outputs["expected_result"] = case.get("expected_result", {})
            outputs["task_accuracy_threshold"] = case.get("success_criteria", {}).get("task_accuracy", ">0.85")
            level6_count += 1

        # Metadata for all types
        metadata = {
            "success_criteria": case.get("success_criteria", {}),
            "is_safety_critical": case.get("is_safety_critical", False),
            "is_edge_case": case.get("is_edge_case", False),
            "eval_type": eval_type,
            "is_level6": eval_type in LEVEL6_EVAL_TYPES or category in LEVEL6_CATEGORIES,
        }

        client.create_example(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            dataset_id=dataset.id,
        )
        examples_created += 1

    print(f"✅ Uploaded {examples_created} test cases to LangSmith")
    if level6_count > 0:
        print(f"   Level 6 test cases: {level6_count}")
    print(f"   Dataset URL: https://smith.langchain.com/datasets/{dataset.id}")

    return str(dataset.id)


def main():
    """Main upload workflow."""
    parser = argparse.ArgumentParser(
        description="Upload golden dataset to LangSmith for evaluation tracking"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["simple", "complex", "hurricane", "edge", "bleu_rouge", "snapshot", "retrieval", "ragas_recall", "agentbench"],
        default=None,
        help="Upload only specific category (default: all)",
    )
    parser.add_argument(
        "--level6-only",
        action="store_true",
        help="Upload only Level 6 evaluation categories",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GOLDEN DATASET UPLOAD TO LANGSMITH")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load dataset
    print(f"📄 Loading dataset from: {DATASET_PATH}")
    test_cases = load_golden_dataset(DATASET_PATH)
    print(f"   Found {len(test_cases)} test cases")

    # Filter for Level 6 only if requested
    if args.level6_only:
        test_cases = [c for c in test_cases if c.get("category") in LEVEL6_CATEGORIES]
        print(f"   Filtered to {len(test_cases)} Level 6 test cases")

    # Categorize
    categories = {}
    eval_types = {}
    for case in test_cases:
        cat = case.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        etype = case.get("eval_type", "standard")
        eval_types[etype] = eval_types.get(etype, 0) + 1

    print("\n📊 Category Breakdown:")
    # Show Level 5 categories first
    level5_cats = ["simple", "complex", "hurricane", "edge"]
    level6_cats = LEVEL6_CATEGORIES

    print("   Level 5:")
    for cat in level5_cats:
        if cat in categories:
            print(f"     - {cat}: {categories[cat]}")

    print("   Level 6:")
    for cat in level6_cats:
        if cat in categories:
            print(f"     - {cat}: {categories[cat]}")

    print("\n📋 Evaluation Types:")
    for etype, count in sorted(eval_types.items()):
        print(f"   - {etype}: {count}")

    # Upload
    print()
    dataset_id = upload_to_langsmith(test_cases, category_filter=args.category)

    print()
    print("=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"Dataset ID: {dataset_id}")
    print("View at: https://smith.langchain.com/datasets")
    print()
    print("Next Steps:")
    print("  1. Run batch evaluation: make eval-run-batch")
    print("  2. Run Level 6 evaluation: make eval-level6")
    print("  3. View results in LangSmith UI")
    print("  4. Set up CI/CD quality gates")
    print()
    print("Level 6 Specific Evaluations:")
    print("  - BLEU/ROUGE: make eval-bleu-rouge")
    print("  - Snapshot: make eval-snapshot")
    print("  - Retrieval: make eval-retrieval")
    print("  - RAGAS: make eval-ragas")
    print("  - AgentBench: make eval-agentbench")


if __name__ == "__main__":
    main()
