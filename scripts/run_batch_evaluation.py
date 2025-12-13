#!/usr/bin/env python3
"""Run batch evaluation on golden dataset using LangSmith.

This script runs the golden dataset through the Weather AI Agent and
evaluates results using the 4-pillar framework (Effectiveness, Efficiency,
Robustness, Safety).

Usage:
    uv run python scripts/run_batch_evaluation.py

    # Or via Makefile:
    make eval-run-batch

    # With options:
    uv run python scripts/run_batch_evaluation.py --max-cases=10 --category=hurricane
"""

import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import evaluation components
from tests.evaluation.golden_dataset_runner import GoldenDatasetRunner


async def run_batch_evaluation(
    max_cases: int | None = None,
    category: str | None = None,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Run batch evaluation with golden dataset.

    Args:
        max_cases: Limit number of test cases (default: all 105)
        category: Run only specific category (simple, complex, hurricane, edge)
        output_file: Path to save results JSON (default: evaluation_results.json)

    Returns:
        Evaluation results dictionary
    """
    runner = GoldenDatasetRunner()

    print("=" * 70)
    print("WEATHER AI AGENT - BATCH EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Max Cases: {max_cases or 'All (105)'}")
    print(f"Category: {category or 'All'}")
    print()

    # Run evaluation
    if category:
        print(f"📊 Running evaluation for category: {category}")
        results = await runner.run_category(category, max_cases=max_cases)
    else:
        print("📊 Running full evaluation (all categories)")
        results = await runner.run_all(max_cases=max_cases)

    # Generate report
    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    report = runner.generate_report(results)
    print(report)

    # Check quality gates
    gates = runner.check_quality_gates(results)

    print()
    print("=" * 70)
    print("QUALITY GATES")
    print("=" * 70)
    print(f"Recommendation: {gates['recommendation']}")
    print()

    gate_status = []
    for gate_name, gate_result in gates['gates'].items():
        status = "✅ PASS" if gate_result['passed'] else "❌ FAIL"
        print(f"  {status} {gate_name}: {gate_result['actual']:.2f} (threshold: {gate_result['threshold']:.2f})")
        gate_status.append(gate_result['passed'])

    # Save results
    output_path = Path(output_file or "evaluation_results.json")
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "max_cases": max_cases,
        "category": category,
        "total_cases": results.total_cases,
        "passed_cases": results.passed_cases,
        "failed_cases": results.failed_cases,
        "pass_rate": results.pass_rate,
        "pillar_averages": {
            "effectiveness": results.avg_effectiveness,
            "efficiency": results.avg_efficiency,
            "robustness": results.avg_robustness,
        },
        "safety_violations": results.safety_violations,
        "execution_time_ms": results.evaluation_time_ms,
        "quality_gates": gates,
        "all_gates_passed": all(gate_status),
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    print()
    print(f"💾 Results saved to: {output_path}")

    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("  1. Review failed tests (if any)")
    print("  2. View traces in LangSmith: https://smith.langchain.com")
    print("  3. Update prompts/code based on failures")
    print("  4. Re-run evaluation to verify improvements")

    return results_dict


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run batch evaluation on golden dataset"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of test cases to run (default: all 105)",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["simple", "complex", "hurricane", "edge"],
        default=None,
        help="Run only specific category",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results (default: evaluation_results.json)",
    )

    args = parser.parse_args()

    # Run evaluation
    results = asyncio.run(
        run_batch_evaluation(
            max_cases=args.max_cases,
            category=args.category,
            output_file=args.output,
        )
    )

    # Exit with status code
    if results["all_gates_passed"]:
        print("\n✅ All quality gates passed - READY TO DEPLOY")
        exit(0)
    else:
        print("\n❌ Quality gates failed - DO NOT DEPLOY")
        exit(1)


if __name__ == "__main__":
    main()
