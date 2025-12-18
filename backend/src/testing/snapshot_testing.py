"""
Snapshot Testing Module.

Level 6c: Self-Evolving Platform

Provides snapshot testing for:
1. Response regression detection
2. Format consistency verification
3. Golden response comparison
4. Semantic drift detection

Target: Catch response regressions before production
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SnapshotStatus(str, Enum):
    """Status of a snapshot comparison."""

    MATCH = "match"
    MISMATCH = "mismatch"
    NEW = "new"
    MISSING = "missing"


class SnapshotResult(BaseModel):
    """Result of a single snapshot test."""

    snapshot_id: str
    status: SnapshotStatus
    expected_hash: str = ""
    actual_hash: str = ""
    diff_summary: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class SnapshotComparison(BaseModel):
    """Detailed comparison between expected and actual content."""

    snapshot_id: str
    expected_content: str
    actual_content: str
    is_match: bool
    similarity_score: float = 0.0  # 0.0 to 1.0
    differences: list[dict[str, Any]] = Field(default_factory=list)
    semantic_match: bool = False


class Snapshot(BaseModel):
    """A stored snapshot."""

    id: str
    name: str
    content: str
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SnapshotManager:
    """
    Manage snapshots for regression testing.

    Provides storage, comparison, and update functionality for snapshots.
    """

    def __init__(
        self,
        snapshot_dir: str | Path | None = None,
        update_mode: bool = False,
    ):
        """
        Initialize snapshot manager.

        Args:
            snapshot_dir: Directory to store snapshots (None for in-memory)
            update_mode: If True, update snapshots instead of failing on mismatch
        """
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.update_mode = update_mode

        # In-memory storage
        self.snapshots: dict[str, Snapshot] = {}

        # Results tracking
        self.results: list[SnapshotResult] = []

        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._load_snapshots()

        logger.info(
            f"SnapshotManager initialized | dir={snapshot_dir} | update_mode={update_mode}"
        )

    def _load_snapshots(self) -> None:
        """Load snapshots from disk."""
        if not self.snapshot_dir:
            return

        snapshot_file = self.snapshot_dir / "snapshots.json"
        if snapshot_file.exists():
            with open(snapshot_file) as f:
                data = json.load(f)
                for snapshot_data in data.get("snapshots", []):
                    snapshot = Snapshot(**snapshot_data)
                    self.snapshots[snapshot.id] = snapshot

            logger.info(f"Loaded {len(self.snapshots)} snapshots from disk")

    def _save_snapshots(self) -> None:
        """Save snapshots to disk."""
        if not self.snapshot_dir:
            return

        snapshot_file = self.snapshot_dir / "snapshots.json"
        data = {
            "snapshots": [s.model_dump(mode="json") for s in self.snapshots.values()],
            "updated_at": datetime.now().isoformat(),
        }

        with open(snapshot_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"Saved {len(self.snapshots)} snapshots to disk")

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def create_snapshot(
        self,
        snapshot_id: str,
        content: str,
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Snapshot:
        """
        Create a new snapshot.

        Args:
            snapshot_id: Unique identifier for the snapshot
            content: Content to snapshot
            name: Human-readable name
            metadata: Optional metadata

        Returns:
            Created Snapshot
        """
        content_hash = self._compute_hash(content)

        snapshot = Snapshot(
            id=snapshot_id,
            name=name or snapshot_id,
            content=content,
            content_hash=content_hash,
            metadata=metadata or {},
        )

        self.snapshots[snapshot_id] = snapshot
        self._save_snapshots()

        logger.info(f"Created snapshot '{snapshot_id}' | hash={content_hash}")

        return snapshot

    def get_snapshot(self, snapshot_id: str) -> Snapshot | None:
        """Get a snapshot by ID."""
        return self.snapshots.get(snapshot_id)

    def update_snapshot(
        self,
        snapshot_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Snapshot | None:
        """
        Update an existing snapshot.

        Args:
            snapshot_id: ID of snapshot to update
            content: New content
            metadata: Optional new metadata

        Returns:
            Updated Snapshot or None if not found
        """
        if snapshot_id not in self.snapshots:
            return None

        existing = self.snapshots[snapshot_id]
        content_hash = self._compute_hash(content)

        updated = Snapshot(
            id=snapshot_id,
            name=existing.name,
            content=content,
            content_hash=content_hash,
            metadata=metadata or existing.metadata,
            created_at=existing.created_at,
            updated_at=datetime.now(),
        )

        self.snapshots[snapshot_id] = updated
        self._save_snapshots()

        logger.info(f"Updated snapshot '{snapshot_id}' | old_hash={existing.content_hash} | new_hash={content_hash}")

        return updated

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        if snapshot_id in self.snapshots:
            del self.snapshots[snapshot_id]
            self._save_snapshots()
            logger.info(f"Deleted snapshot '{snapshot_id}'")
            return True
        return False

    def compare(
        self,
        snapshot_id: str,
        actual_content: str,
    ) -> SnapshotResult:
        """
        Compare actual content against a stored snapshot.

        Args:
            snapshot_id: ID of snapshot to compare against
            actual_content: Actual content to compare

        Returns:
            SnapshotResult with comparison outcome
        """
        actual_hash = self._compute_hash(actual_content)

        # Check if snapshot exists
        snapshot = self.snapshots.get(snapshot_id)

        if snapshot is None:
            # New snapshot - create it if in update mode
            if self.update_mode:
                self.create_snapshot(snapshot_id, actual_content)
                result = SnapshotResult(
                    snapshot_id=snapshot_id,
                    status=SnapshotStatus.NEW,
                    actual_hash=actual_hash,
                    diff_summary="New snapshot created",
                )
            else:
                result = SnapshotResult(
                    snapshot_id=snapshot_id,
                    status=SnapshotStatus.MISSING,
                    actual_hash=actual_hash,
                    diff_summary="Snapshot not found - run with update_mode=True to create",
                )
        elif snapshot.content_hash == actual_hash:
            # Match
            result = SnapshotResult(
                snapshot_id=snapshot_id,
                status=SnapshotStatus.MATCH,
                expected_hash=snapshot.content_hash,
                actual_hash=actual_hash,
                diff_summary="Content matches snapshot",
            )
        else:
            # Mismatch
            if self.update_mode:
                self.update_snapshot(snapshot_id, actual_content)
                result = SnapshotResult(
                    snapshot_id=snapshot_id,
                    status=SnapshotStatus.MISMATCH,
                    expected_hash=snapshot.content_hash,
                    actual_hash=actual_hash,
                    diff_summary="Snapshot updated with new content",
                )
            else:
                diff = self._compute_diff(snapshot.content, actual_content)
                result = SnapshotResult(
                    snapshot_id=snapshot_id,
                    status=SnapshotStatus.MISMATCH,
                    expected_hash=snapshot.content_hash,
                    actual_hash=actual_hash,
                    diff_summary=diff,
                )

        self.results.append(result)

        logger.info(
            f"Snapshot comparison '{snapshot_id}' | status={result.status} | "
            f"expected={result.expected_hash} | actual={result.actual_hash}"
        )

        return result

    def compare_detailed(
        self,
        snapshot_id: str,
        actual_content: str,
    ) -> SnapshotComparison:
        """
        Perform detailed comparison with diff analysis.

        Args:
            snapshot_id: ID of snapshot to compare against
            actual_content: Actual content to compare

        Returns:
            SnapshotComparison with detailed analysis
        """
        snapshot = self.snapshots.get(snapshot_id)

        if snapshot is None:
            return SnapshotComparison(
                snapshot_id=snapshot_id,
                expected_content="",
                actual_content=actual_content,
                is_match=False,
                similarity_score=0.0,
                differences=[{"type": "missing", "message": "Snapshot not found"}],
            )

        expected = snapshot.content
        is_match = expected == actual_content
        similarity = self._compute_similarity(expected, actual_content)
        differences = self._compute_detailed_diff(expected, actual_content)
        semantic_match = similarity >= 0.95  # 95% similarity threshold

        return SnapshotComparison(
            snapshot_id=snapshot_id,
            expected_content=expected,
            actual_content=actual_content,
            is_match=is_match,
            similarity_score=similarity,
            differences=differences,
            semantic_match=semantic_match,
        )

    def _compute_diff(self, expected: str, actual: str) -> str:
        """Compute a simple diff summary."""
        expected_lines = expected.split("\n")
        actual_lines = actual.split("\n")

        if len(expected_lines) != len(actual_lines):
            return f"Line count differs: expected {len(expected_lines)}, got {len(actual_lines)}"

        diff_count = 0
        for i, (e, a) in enumerate(zip(expected_lines, actual_lines)):
            if e != a:
                diff_count += 1

        if diff_count > 0:
            return f"{diff_count} line(s) differ out of {len(expected_lines)}"

        return "Content differs but same line count"

    def _compute_similarity(self, expected: str, actual: str) -> float:
        """Compute similarity score between two strings."""
        if expected == actual:
            return 1.0

        if not expected or not actual:
            return 0.0

        # Simple character-based similarity
        expected_chars = set(expected)
        actual_chars = set(actual)

        intersection = len(expected_chars & actual_chars)
        union = len(expected_chars | actual_chars)

        char_similarity = intersection / union if union > 0 else 0.0

        # Word-based similarity
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())

        word_intersection = len(expected_words & actual_words)
        word_union = len(expected_words | actual_words)

        word_similarity = word_intersection / word_union if word_union > 0 else 0.0

        # Combined similarity (weighted average)
        return round(0.3 * char_similarity + 0.7 * word_similarity, 4)

    def _compute_detailed_diff(
        self,
        expected: str,
        actual: str,
    ) -> list[dict[str, Any]]:
        """Compute detailed differences."""
        differences: list[dict[str, Any]] = []

        expected_lines = expected.split("\n")
        actual_lines = actual.split("\n")

        # Check line count
        if len(expected_lines) != len(actual_lines):
            differences.append({
                "type": "line_count",
                "expected": len(expected_lines),
                "actual": len(actual_lines),
            })

        # Check individual lines (up to a limit)
        max_lines = min(len(expected_lines), len(actual_lines), 50)
        for i in range(max_lines):
            if expected_lines[i] != actual_lines[i]:
                differences.append({
                    "type": "line_diff",
                    "line": i + 1,
                    "expected": expected_lines[i][:100],  # Truncate
                    "actual": actual_lines[i][:100],
                })

                if len(differences) >= 10:  # Limit diff output
                    differences.append({
                        "type": "truncated",
                        "message": "Diff truncated after 10 differences",
                    })
                    break

        return differences

    def assert_match(
        self,
        snapshot_id: str,
        actual_content: str,
    ) -> None:
        """
        Assert that content matches snapshot (raises on mismatch).

        Args:
            snapshot_id: ID of snapshot to compare against
            actual_content: Actual content to compare

        Raises:
            AssertionError: If content doesn't match and not in update mode
        """
        result = self.compare(snapshot_id, actual_content)

        if result.status == SnapshotStatus.MISMATCH and not self.update_mode:
            raise AssertionError(
                f"Snapshot mismatch for '{snapshot_id}': {result.diff_summary}\n"
                f"Expected hash: {result.expected_hash}\n"
                f"Actual hash: {result.actual_hash}"
            )

        if result.status == SnapshotStatus.MISSING:
            raise AssertionError(
                f"Snapshot '{snapshot_id}' not found. "
                "Run with update_mode=True to create."
            )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all snapshot comparisons."""
        total = len(self.results)
        matches = sum(1 for r in self.results if r.status == SnapshotStatus.MATCH)
        mismatches = sum(1 for r in self.results if r.status == SnapshotStatus.MISMATCH)
        new = sum(1 for r in self.results if r.status == SnapshotStatus.NEW)
        missing = sum(1 for r in self.results if r.status == SnapshotStatus.MISSING)

        return {
            "total_comparisons": total,
            "matches": matches,
            "mismatches": mismatches,
            "new_snapshots": new,
            "missing_snapshots": missing,
            "pass_rate": round(matches / total * 100, 2) if total > 0 else 0,
            "total_snapshots": len(self.snapshots),
        }

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all snapshots."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "content_hash": s.content_hash,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "metadata": s.metadata,
            }
            for s in self.snapshots.values()
        ]

    def clear_results(self) -> None:
        """Clear comparison results."""
        self.results = []


class WeatherSnapshotTests:
    """Pre-built snapshot tests for weather responses."""

    @staticmethod
    def create_weather_snapshots(manager: SnapshotManager) -> None:
        """Create standard weather response snapshots."""
        # Basic weather response format
        manager.create_snapshot(
            "weather_basic_response",
            content=(
                "Current weather in Miami:\n"
                "Temperature: 85°F (29°C)\n"
                "Humidity: 65%\n"
                "Wind: 10 mph from SE\n"
                "Conditions: Partly cloudy"
            ),
            name="Basic Weather Response",
            metadata={"type": "weather", "format": "basic"},
        )

        # Hurricane warning format
        manager.create_snapshot(
            "hurricane_warning_response",
            content=(
                "⚠️ HURRICANE WARNING\n"
                "Hurricane Milton - Category 4\n"
                "Current winds: 130 mph\n"
                "Location: 26.5°N, 82.0°W\n"
                "Movement: NNE at 12 mph\n\n"
                "SAFETY INSTRUCTIONS:\n"
                "- Evacuate Zone A immediately\n"
                "- Secure loose outdoor items\n"
                "- Fill prescriptions and gas tank\n"
                "- Follow local emergency management guidance"
            ),
            name="Hurricane Warning Response",
            metadata={"type": "hurricane", "format": "warning"},
        )

        # Evacuation guidance format
        manager.create_snapshot(
            "evacuation_guidance_response",
            content=(
                "EVACUATION GUIDANCE\n"
                "Zone: A (Mandatory Evacuation)\n"
                "Deadline: October 8, 2024 at 6:00 PM EDT\n\n"
                "Recommended Routes:\n"
                "1. I-75 North to I-10 West\n"
                "2. US-19 North (alternate)\n\n"
                "Nearest Shelter:\n"
                "Hillsborough County Convention Center\n"
                "333 S Franklin St, Tampa, FL 33602\n\n"
                "Contact: 1-800-FL-HELP (354-3577)"
            ),
            name="Evacuation Guidance Response",
            metadata={"type": "evacuation", "format": "guidance"},
        )

    @staticmethod
    def get_standard_tests() -> list[tuple[str, str]]:
        """
        Get standard snapshot test cases.

        Returns:
            List of (snapshot_id, description) tuples
        """
        return [
            ("weather_basic_response", "Basic weather response format"),
            ("hurricane_warning_response", "Hurricane warning format"),
            ("evacuation_guidance_response", "Evacuation guidance format"),
        ]


def run_weather_snapshot_tests(
    snapshot_dir: str | Path | None = None,
    update_mode: bool = False,
) -> dict[str, Any]:
    """
    Convenience function to run all weather snapshot tests.

    Args:
        snapshot_dir: Directory for snapshots
        update_mode: Whether to update snapshots

    Returns:
        Test summary dict
    """
    manager = SnapshotManager(snapshot_dir=snapshot_dir, update_mode=update_mode)

    # Create standard snapshots if in update mode
    if update_mode:
        WeatherSnapshotTests.create_weather_snapshots(manager)

    return manager.get_summary()
