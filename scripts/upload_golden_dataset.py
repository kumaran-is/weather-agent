#!/usr/bin/env python3
"""Upload golden dataset to LangSmith for evaluation tracking.

This script uploads the golden dataset (tests/evaluation/golden_dataset.yaml)
to LangSmith as a named dataset for batch evaluation and regression testing.

Usage:
    uv run python scripts/upload_golden_dataset.py

    # Or via Makefile:
    make eval-upload-dataset
"""

import yaml
from pathlib import Path
from langsmith import Client
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
DATASET_PATH = Path("tests/evaluation/golden_dataset.yaml")
DATASET_NAME = "weather-ai-golden-dataset"
DATASET_DESCRIPTION = """
Weather AI Agent Golden Dataset (Level 5b)
- 105 test cases for regression testing
- Categories: simple (40), complex (30), hurricane (20), edge (15)
- Used for 4-pillar evaluation (effectiveness, efficiency, robustness, safety)
- Pass threshold: 85% overall, 100% safety

Test Categories:
- Simple: Basic weather queries (single location)
- Complex: Multi-location comparisons, analysis
- Hurricane: Safety-critical (Saffir-Simpson validation)
- Edge: Error handling, invalid inputs

Created: {datetime.now().strftime("%Y-%m-%d")}
Version: 1.0.0
"""


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


def upload_to_langsmith(test_cases: list[dict]) -> str:
    """Upload dataset to LangSmith.

    Returns:
        Dataset ID
    """
    client = Client()

    # Check if dataset exists
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))

    if existing:
        print(f"⚠️  Dataset '{DATASET_NAME}' already exists. Updating...")
        dataset = existing[0]
        # Delete existing examples for clean upload
        examples = list(client.list_examples(dataset_id=dataset.id))
        for ex in examples:
            client.delete_example(ex.id)
        print(f"   Deleted {len(examples)} existing examples")
    else:
        print(f"📦 Creating new dataset: {DATASET_NAME}")
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION,
        )

    # Upload each test case as an example
    examples_created = 0
    for case in test_cases:
        # Create input/output structure
        inputs = {
            "query": case["query"],
            "test_id": case["id"],
            "category": case.get("category", "unknown"),
        }

        outputs = {
            "expected_tools": case.get("expected_tools", []),
            "expected_answer_contains": case.get("expected_answer_contains", []),
            "expected_answer": case.get("expected_answer"),
        }

        metadata = {
            "success_criteria": case.get("success_criteria", {}),
            "is_safety_critical": case.get("is_safety_critical", False),
            "is_edge_case": case.get("is_edge_case", False),
        }

        client.create_example(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            dataset_id=dataset.id,
        )
        examples_created += 1

    print(f"✅ Uploaded {examples_created} test cases to LangSmith")
    print(f"   Dataset URL: https://smith.langchain.com/datasets/{dataset.id}")

    return str(dataset.id)


def main():
    """Main upload workflow."""
    print("=" * 60)
    print("GOLDEN DATASET UPLOAD TO LANGSMITH")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load dataset
    print(f"📄 Loading dataset from: {DATASET_PATH}")
    test_cases = load_golden_dataset(DATASET_PATH)
    print(f"   Found {len(test_cases)} test cases")

    # Categorize
    categories = {}
    for case in test_cases:
        cat = case.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\n📊 Category Breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"   - {cat}: {count}")

    # Upload
    print()
    dataset_id = upload_to_langsmith(test_cases)

    print()
    print("=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"Dataset ID: {dataset_id}")
    print(f"View at: https://smith.langchain.com/datasets")
    print()
    print("Next Steps:")
    print("  1. Run batch evaluation: make eval-run-batch")
    print("  2. View results in LangSmith UI")
    print("  3. Set up CI/CD quality gates")


if __name__ == "__main__":
    main()
