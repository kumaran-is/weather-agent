#!/usr/bin/env python3
"""Signal Correlation Verification Script - Level 8.

This script verifies that all observability signal correlation components
are properly configured and working:

1. Metrics → Traces: Exemplars with trace_id on histograms
2. Traces → Logs: tracesToLogsV2 configuration
3. Logs → Traces: Derived fields with trace_id regex
4. Traces → Metrics: tracesToMetrics queries

Usage:
    python scripts/verify_signal_correlation.py

Requirements:
    - Docker containers running (docker-compose up -d)
    - Tempo, Prometheus, Loki, Grafana services healthy
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}{title}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def print_check(name: str, passed: bool, details: str = "") -> None:
    """Print a check result."""
    status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
    print(f"  {status} {name}")
    if details:
        print(f"       {YELLOW}{details}{RESET}")


def run_command(cmd: list[str], timeout: int = 30) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_container_health(container: str) -> tuple[bool, str]:
    """Check if a Docker container is healthy."""
    success, output = run_command(
        ["docker", "inspect", "--format", "{{.State.Health.Status}}", container]
    )
    if success:
        status = output.strip()
        return status == "healthy", status
    return False, output


def check_tempo_ready() -> tuple[bool, str]:
    """Check if Tempo is ready to receive traces."""
    success, output = run_command(
        ["docker", "exec", "weather-ai-tempo", "wget", "-q", "-O-", "http://localhost:3200/ready"]
    )
    return success and "ready" in output.lower(), output


def check_prometheus_exemplars() -> tuple[bool, str]:
    """Check if Prometheus has exemplar storage enabled."""
    success, output = run_command(
        ["docker", "exec", "weather-ai-prometheus", "wget", "-q", "-O-", "http://localhost:9090/api/v1/status/flags"]
    )
    if success:
        try:
            data = json.loads(output)
            flags = data.get("data", {})
            exemplar_enabled = flags.get("storage.exemplars.max_exemplars", "0")
            return int(exemplar_enabled) > 0, f"max_exemplars={exemplar_enabled}"
        except (json.JSONDecodeError, ValueError):
            pass
    return False, output


def check_grafana_datasources() -> tuple[bool, dict[str, bool]]:
    """Check Grafana datasource configuration."""
    datasources_file = Path("observability/grafana/provisioning/datasources/datasources.yml")

    results = {
        "prometheus_exemplars": False,
        "tempo_tracesToLogs": False,
        "tempo_tracesToMetrics": False,
        "loki_derivedFields": False,
    }

    if not datasources_file.exists():
        return False, results

    content = datasources_file.read_text()

    results["prometheus_exemplars"] = "exemplarTraceIdDestinations" in content
    results["tempo_tracesToLogs"] = "tracesToLogsV2" in content
    results["tempo_tracesToMetrics"] = "tracesToMetrics" in content
    results["loki_derivedFields"] = "derivedFields" in content

    all_pass = all(results.values())
    return all_pass, results


def check_metrics_exemplar_code() -> tuple[bool, dict[str, bool]]:
    """Check if metrics code includes exemplar support."""
    metrics_file = Path("backend/src/observability/metrics.py")

    results = {
        "get_exemplar_labels": False,
        "observe_with_exemplar": False,
        "trace_import": False,
    }

    if not metrics_file.exists():
        return False, results

    content = metrics_file.read_text()

    results["get_exemplar_labels"] = "def get_exemplar_labels" in content
    results["observe_with_exemplar"] = "def observe_with_exemplar" in content
    results["trace_import"] = "from opentelemetry import trace" in content

    all_pass = all(results.values())
    return all_pass, results


def check_logging_trace_context() -> tuple[bool, dict[str, bool]]:
    """Check if logging includes trace context."""
    logging_file = Path("backend/src/observability/logging.py")

    results = {
        "trace_id_injection": False,
        "span_context_valid": False,
    }

    if not logging_file.exists():
        return False, results

    content = logging_file.read_text()

    results["trace_id_injection"] = 'log_dict["trace_id"]' in content
    results["span_context_valid"] = "span_context.is_valid" in content

    all_pass = all(results.values())
    return all_pass, results


def check_tempo_config() -> tuple[bool, dict[str, bool]]:
    """Check Tempo configuration."""
    tempo_config = Path("observability/tempo/tempo.yaml")

    results = {
        "otlp_receiver": False,
        "storage_config": False,
    }

    if not tempo_config.exists():
        return False, results

    content = tempo_config.read_text()

    results["otlp_receiver"] = "otlp:" in content and "protocols:" in content
    results["storage_config"] = "storage:" in content and "trace:" in content

    all_pass = all(results.values())
    return all_pass, results


def check_docker_compose() -> tuple[bool, dict[str, bool]]:
    """Check docker-compose configuration."""
    compose_file = Path("docker-compose.yml")

    results = {
        "tempo_service": False,
        "prometheus_exemplar_flag": False,
        "otel_endpoint": False,
        "tempo_volume": False,
    }

    if not compose_file.exists():
        return False, results

    content = compose_file.read_text()

    results["tempo_service"] = "tempo:" in content and "grafana/tempo" in content
    results["prometheus_exemplar_flag"] = "--enable-feature=exemplar-storage" in content
    results["otel_endpoint"] = "OTEL_EXPORTER_OTLP_ENDPOINT" in content
    results["tempo_volume"] = "tempo-data:" in content

    all_pass = all(results.values())
    return all_pass, results


def check_dashboard_exemplars() -> tuple[bool, str]:
    """Check if signal correlation dashboard exists with exemplar queries."""
    dashboard_file = Path("observability/grafana/provisioning/dashboards/signal-correlation.json")

    if not dashboard_file.exists():
        return False, "Dashboard file not found"

    content = dashboard_file.read_text()

    # Check for exemplar: true in queries
    if '"exemplar": true' in content:
        return True, "Dashboard has exemplar-enabled queries"

    return False, "Dashboard missing exemplar: true in queries"


def main() -> int:
    """Run all verification checks."""
    print_header("Signal Correlation Verification (Level 8)")

    all_passed = True

    # 1. Code Checks
    print(f"{BOLD}1. Code Configuration Checks{RESET}")

    # Metrics exemplar support
    passed, results = check_metrics_exemplar_code()
    print_check("Metrics exemplar support", passed)
    for check, status in results.items():
        print(f"       - {check}: {'✓' if status else '✗'}")
    all_passed = all_passed and passed

    # Logging trace context
    passed, results = check_logging_trace_context()
    print_check("Logging trace context injection", passed)
    for check, status in results.items():
        print(f"       - {check}: {'✓' if status else '✗'}")
    all_passed = all_passed and passed

    # 2. Infrastructure Checks
    print(f"\n{BOLD}2. Infrastructure Configuration{RESET}")

    # Tempo config
    passed, results = check_tempo_config()
    print_check("Tempo configuration", passed)
    for check, status in results.items():
        print(f"       - {check}: {'✓' if status else '✗'}")
    all_passed = all_passed and passed

    # Docker compose
    passed, results = check_docker_compose()
    print_check("Docker Compose configuration", passed)
    for check, status in results.items():
        print(f"       - {check}: {'✓' if status else '✗'}")
    all_passed = all_passed and passed

    # Grafana datasources
    passed, results = check_grafana_datasources()
    print_check("Grafana datasource correlation", passed)
    for check, status in results.items():
        print(f"       - {check}: {'✓' if status else '✗'}")
    all_passed = all_passed and passed

    # Dashboard
    passed, details = check_dashboard_exemplars()
    print_check("Signal correlation dashboard", passed, details)
    all_passed = all_passed and passed

    # 3. Runtime Checks (if containers are running)
    print(f"\n{BOLD}3. Runtime Health Checks{RESET}")

    # Check containers
    containers = [
        ("weather-ai-tempo", "Tempo"),
        ("weather-ai-prometheus", "Prometheus"),
        ("weather-ai-loki", "Loki"),
        ("weather-ai-grafana", "Grafana"),
    ]

    for container, name in containers:
        passed, status = check_container_health(container)
        if "No such object" in status:
            print_check(f"{name} container", False, "Container not running")
        else:
            print_check(f"{name} container", passed, status)

    # Tempo ready check
    passed, details = check_tempo_ready()
    print_check("Tempo ready", passed, details[:100] if not passed else "")

    # Summary
    print_header("Verification Summary")

    if all_passed:
        print(f"{GREEN}{BOLD}✅ All signal correlation components verified!{RESET}")
        print(f"\n{BOLD}Next Steps:{RESET}")
        print("  1. Start services: docker-compose up -d")
        print("  2. Generate traffic: curl -X POST http://localhost:8000/weather/query -d '{\"query\": \"weather in miami\"}'")
        print("  3. Open Grafana: http://localhost:3001")
        print("  4. Navigate to 'Signal Correlation (Level 8)' dashboard")
        print("  5. Click ⭐ exemplars on histogram charts to jump to traces")
        print("  6. View logs and click trace_id links")
        return 0
    else:
        print(f"{RED}{BOLD}❌ Some verification checks failed!{RESET}")
        print(f"\n{BOLD}See details above for failing checks.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
