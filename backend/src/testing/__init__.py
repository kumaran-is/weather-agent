"""Level 6c: Advanced Testing Framework.

This module provides advanced testing capabilities:
1. Property-Based Testing - Hypothesis-driven test generation
2. Snapshot Testing - Regression detection
3. Fuzz Testing - Edge case discovery
4. Coverage Analysis - Test completeness tracking

Usage:
    >>> from backend.src.testing import PropertyTestRunner, SnapshotManager
    >>> runner = PropertyTestRunner()
    >>> results = runner.run_weather_properties()

    >>> snapshots = SnapshotManager(snapshot_dir="./snapshots")
    >>> snapshots.capture("response_v1", response)
    >>> snapshots.compare("response_v1", new_response)
"""

from backend.src.testing.property_testing import (
    PropertyTestRunner,
    PropertyTestResult,
    WeatherPropertyTests,
)
from backend.src.testing.snapshot_testing import (
    SnapshotManager,
    SnapshotResult,
    SnapshotComparison,
)

__all__ = [
    # Property Testing
    "PropertyTestRunner",
    "PropertyTestResult",
    "WeatherPropertyTests",
    # Snapshot Testing
    "SnapshotManager",
    "SnapshotResult",
    "SnapshotComparison",
]
