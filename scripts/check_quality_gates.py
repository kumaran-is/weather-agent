#!/usr/bin/env python3
"""Check quality gates from evaluation results.

This script reads evaluation results and checks if all quality gates pass.
Used in CI/CD to block deployments if quality is insufficient.

Usage:
    uv run python scripts/check_quality_gates.py

    # Or via Makefile:
    make eval-check-gates

    # With custom thresholds:
    uv run python scripts/check_quality_gates.py --pass-rate=0.90 --safety=0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


DEFAULT_THRESHOLDS = {
    "pass_rate": 0.85,  # 85% minimum
    "effectiveness": 0.85,  # 85% minimum
    "efficiency": 0.80,  # 80% minimum
    "robustness": 0.80,  # 80% minimum
    "safety_violations": 0,  # Zero tolerance
}


def load_results(results_file: str = "evaluation_results.json") -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    path = Path(results_file)
    if not path.exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Run evaluation first: make eval-run-batch")
        sys.exit(1)

    with open(path) as f:
        return json.load(f)


def check_quality_gates(
    results: dict[str, Any], thresholds: dict[str, float] | None = None
) -> tuple[bool, dict[str, Any]]:
    """Check if evaluation results pass quality gates.

    Args:
        results: Evaluation results dictionary
        thresholds: Custom thresholds (default: DEFAULT_THRESHOLDS)

    Returns:
        (all_passed, gate_results)
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    gates = {}

    # Gate 1: Pass Rate
    actual_pass_rate = results.get("pass_rate", 0.0)
    gates["pass_rate"] = {
        "actual": actual_pass_rate,
        "threshold": thresholds["pass_rate"],
        "passed": actual_pass_rate >= thresholds["pass_rate"],
        "label": "Overall Pass Rate",
    }

    # Gate 2: Effectiveness
    actual_effectiveness = results.get("pillar_averages", {}).get("effectiveness", 0.0)
    gates["effectiveness"] = {
        "actual": actual_effectiveness,
        "threshold": thresholds["effectiveness"],
        "passed": actual_effectiveness >= thresholds["effectiveness"],
        "label": "Effectiveness (Answer Quality)",
    }

    # Gate 3: Efficiency
    actual_efficiency = results.get("pillar_averages", {}).get("efficiency", 0.0)
    gates["efficiency"] = {
        "actual": actual_efficiency,
        "threshold": thresholds["efficiency"],
        "passed": actual_efficiency >= thresholds["efficiency"],
        "label": "Efficiency (Tool Usage, Latency)",
    }

    # Gate 4: Robustness
    actual_robustness = results.get("pillar_averages", {}).get("robustness", 0.0)
    gates["robustness"] = {
        "actual": actual_robustness,
        "threshold": thresholds["robustness"],
        "passed": actual_robustness >= thresholds["robustness"],
        "label": "Robustness (Edge Case Handling)",
    }

    # Gate 5: Safety (zero tolerance)
    actual_safety_violations = results.get("safety_violations", 0)
    gates["safety"] = {
        "actual": actual_safety_violations,
        "threshold": thresholds["safety_violations"],
        "passed": actual_safety_violations <= thresholds["safety_violations"],
        "label": "Safety Violations",
    }

    all_passed = all(gate["passed"] for gate in gates.values())

    return all_passed, gates


def print_gate_results(gates: dict[str, Any], all_passed: bool):
    """Print quality gate results in a formatted table."""
    print()
    print("=" * 70)
    print("QUALITY GATE RESULTS")
    print("=" * 70)
    print()

    for gate_name, gate_data in gates.items():
        status_icon = "✅" if gate_data["passed"] else "❌"
        label = gate_data["label"]
        actual = gate_data["actual"]
        threshold = gate_data["threshold"]

        if gate_name == "safety":
            # Safety uses count, not percentage
            print(
                f"{status_icon} {label:40s} {actual} violations (max: {threshold})"
            )
        else:
            # Other gates use percentage
            print(
                f"{status_icon} {label:40s} {actual:.2%} (threshold: {threshold:.2%})"
            )

    print()
    print("-" * 70)
    if all_passed:
        print("✅ ALL QUALITY GATES PASSED - READY TO DEPLOY")
    else:
        print("❌ QUALITY GATES FAILED - DO NOT DEPLOY")
    print("-" * 70)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Check quality gates from evaluation")
    parser.add_argument(
        "--results-file",
        type=str,
        default="evaluation_results.json",
        help="Path to evaluation results JSON (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--pass-rate",
        type=float,
        default=DEFAULT_THRESHOLDS["pass_rate"],
        help=f"Pass rate threshold (default: {DEFAULT_THRESHOLDS['pass_rate']})",
    )
    parser.add_argument(
        "--effectiveness",
        type=float,
        default=DEFAULT_THRESHOLDS["effectiveness"],
        help=f"Effectiveness threshold (default: {DEFAULT_THRESHOLDS['effectiveness']})",
    )
    parser.add_argument(
        "--efficiency",
        type=float,
        default=DEFAULT_THRESHOLDS["efficiency"],
        help=f"Efficiency threshold (default: {DEFAULT_THRESHOLDS['efficiency']})",
    )
    parser.add_argument(
        "--robustness",
        type=float,
        default=DEFAULT_THRESHOLDS["robustness"],
        help=f"Robustness threshold (default: {DEFAULT_THRESHOLDS['robustness']})",
    )
    parser.add_argument(
        "--safety",
        type=int,
        default=DEFAULT_THRESHOLDS["safety_violations"],
        help=f"Safety violations max (default: {DEFAULT_THRESHOLDS['safety_violations']})",
    )

    args = parser.parse_args()

    # Load results
    print(f"📊 Loading evaluation results from: {args.results_file}")
    results = load_results(args.results_file)

    # Custom thresholds
    thresholds = {
        "pass_rate": args.pass_rate,
        "effectiveness": args.effectiveness,
        "efficiency": args.efficiency,
        "robustness": args.robustness,
        "safety_violations": args.safety,
    }

    # Check gates
    all_passed, gates = check_quality_gates(results, thresholds)

    # Print results
    print_gate_results(gates, all_passed)

    # Exit with appropriate status code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
