"""
Tests for Snapshot Testing Module.

Level 6c: Self-Evolving Platform

Tests:
1. Snapshot creation and storage
2. Snapshot comparison
3. Update mode behavior
4. Detailed comparison
5. Weather-specific snapshots
"""

import tempfile
from pathlib import Path

import pytest

from backend.src.testing.snapshot_testing import (
    Snapshot,
    SnapshotComparison,
    SnapshotManager,
    SnapshotResult,
    SnapshotStatus,
    WeatherSnapshotTests,
    run_weather_snapshot_tests,
)


class TestSnapshotManager:
    """Test SnapshotManager class."""

    @pytest.fixture
    def manager(self):
        """Create in-memory snapshot manager."""
        return SnapshotManager()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_initialization_in_memory(self, manager):
        """Test in-memory initialization."""
        assert manager.snapshot_dir is None
        assert manager.update_mode is False
        assert manager.snapshots == {}
        assert manager.results == []

    def test_initialization_with_dir(self, temp_dir):
        """Test initialization with directory."""
        manager = SnapshotManager(snapshot_dir=temp_dir)
        assert manager.snapshot_dir == Path(temp_dir)

    def test_create_snapshot(self, manager):
        """Test creating a snapshot."""
        snapshot = manager.create_snapshot(
            snapshot_id="test_001",
            content="Test content",
            name="Test Snapshot",
            metadata={"key": "value"},
        )

        assert isinstance(snapshot, Snapshot)
        assert snapshot.id == "test_001"
        assert snapshot.name == "Test Snapshot"
        assert snapshot.content == "Test content"
        assert snapshot.metadata == {"key": "value"}
        assert snapshot.content_hash != ""

    def test_create_snapshot_with_default_name(self, manager):
        """Test creating snapshot with default name."""
        snapshot = manager.create_snapshot("my_id", "content")
        assert snapshot.name == "my_id"

    def test_get_snapshot(self, manager):
        """Test getting a snapshot."""
        manager.create_snapshot("get_test", "content")

        snapshot = manager.get_snapshot("get_test")
        assert snapshot is not None
        assert snapshot.content == "content"

    def test_get_snapshot_not_found(self, manager):
        """Test getting non-existent snapshot."""
        result = manager.get_snapshot("nonexistent")
        assert result is None

    def test_update_snapshot(self, manager):
        """Test updating a snapshot."""
        original = manager.create_snapshot("update_test", "original")
        original_hash = original.content_hash

        updated = manager.update_snapshot("update_test", "updated")

        assert updated is not None
        assert updated.content == "updated"
        assert updated.content_hash != original_hash
        assert updated.created_at == original.created_at
        assert updated.updated_at > original.updated_at

    def test_update_snapshot_not_found(self, manager):
        """Test updating non-existent snapshot."""
        result = manager.update_snapshot("nonexistent", "new content")
        assert result is None

    def test_delete_snapshot(self, manager):
        """Test deleting a snapshot."""
        manager.create_snapshot("delete_test", "content")
        assert "delete_test" in manager.snapshots

        result = manager.delete_snapshot("delete_test")

        assert result is True
        assert "delete_test" not in manager.snapshots

    def test_delete_snapshot_not_found(self, manager):
        """Test deleting non-existent snapshot."""
        result = manager.delete_snapshot("nonexistent")
        assert result is False

    def test_list_snapshots(self, manager):
        """Test listing snapshots."""
        manager.create_snapshot("s1", "content1")
        manager.create_snapshot("s2", "content2")

        snapshots = manager.list_snapshots()

        assert len(snapshots) == 2
        ids = {s["id"] for s in snapshots}
        assert "s1" in ids
        assert "s2" in ids


class TestSnapshotComparison:
    """Test snapshot comparison functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager with a test snapshot."""
        m = SnapshotManager()
        m.create_snapshot("test_snap", "Expected content")
        return m

    def test_compare_match(self, manager):
        """Test comparison with matching content."""
        result = manager.compare("test_snap", "Expected content")

        assert isinstance(result, SnapshotResult)
        assert result.status == SnapshotStatus.MATCH
        assert result.expected_hash == result.actual_hash

    def test_compare_mismatch(self, manager):
        """Test comparison with mismatching content."""
        result = manager.compare("test_snap", "Different content")

        assert result.status == SnapshotStatus.MISMATCH
        assert result.expected_hash != result.actual_hash
        assert result.diff_summary != ""

    def test_compare_missing_snapshot(self, manager):
        """Test comparison with missing snapshot."""
        result = manager.compare("nonexistent", "Some content")

        assert result.status == SnapshotStatus.MISSING
        assert "not found" in result.diff_summary.lower()

    def test_compare_update_mode_new(self):
        """Test update mode creates new snapshots."""
        manager = SnapshotManager(update_mode=True)

        result = manager.compare("new_snap", "New content")

        assert result.status == SnapshotStatus.NEW
        assert "new_snap" in manager.snapshots

    def test_compare_update_mode_mismatch(self):
        """Test update mode updates mismatched snapshots."""
        manager = SnapshotManager(update_mode=True)
        manager.create_snapshot("update_snap", "Original")

        result = manager.compare("update_snap", "Updated")

        assert result.status == SnapshotStatus.MISMATCH
        assert manager.snapshots["update_snap"].content == "Updated"

    def test_compare_detailed(self, manager):
        """Test detailed comparison."""
        comparison = manager.compare_detailed("test_snap", "Expected content")

        assert isinstance(comparison, SnapshotComparison)
        assert comparison.is_match is True
        assert comparison.similarity_score == 1.0
        assert comparison.semantic_match is True

    def test_compare_detailed_mismatch(self, manager):
        """Test detailed comparison with mismatch."""
        comparison = manager.compare_detailed("test_snap", "Completely different text here")

        assert comparison.is_match is False
        assert comparison.similarity_score < 1.0
        assert len(comparison.differences) > 0

    def test_compare_detailed_missing(self, manager):
        """Test detailed comparison with missing snapshot."""
        comparison = manager.compare_detailed("nonexistent", "Content")

        assert comparison.is_match is False
        assert comparison.expected_content == ""
        assert any(d["type"] == "missing" for d in comparison.differences)


class TestAssertMatch:
    """Test assert_match functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager with test snapshot."""
        m = SnapshotManager()
        m.create_snapshot("assert_test", "Expected")
        return m

    def test_assert_match_passes(self, manager):
        """Test assert_match with matching content."""
        # Should not raise
        manager.assert_match("assert_test", "Expected")

    def test_assert_match_fails_on_mismatch(self, manager):
        """Test assert_match raises on mismatch."""
        with pytest.raises(AssertionError, match="mismatch"):
            manager.assert_match("assert_test", "Different")

    def test_assert_match_fails_on_missing(self, manager):
        """Test assert_match raises on missing."""
        with pytest.raises(AssertionError, match="not found"):
            manager.assert_match("nonexistent", "Content")

    def test_assert_match_update_mode(self):
        """Test assert_match in update mode doesn't raise."""
        manager = SnapshotManager(update_mode=True)
        manager.create_snapshot("update_assert", "Original")

        # Should not raise in update mode
        manager.assert_match("update_assert", "Different")


class TestSnapshotPersistence:
    """Test snapshot persistence to disk."""

    def test_save_and_load(self):
        """Test saving and loading snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            manager1 = SnapshotManager(snapshot_dir=tmpdir)
            manager1.create_snapshot("persist1", "Content 1")
            manager1.create_snapshot("persist2", "Content 2")

            # Load in new manager
            manager2 = SnapshotManager(snapshot_dir=tmpdir)

            assert len(manager2.snapshots) == 2
            assert manager2.snapshots["persist1"].content == "Content 1"
            assert manager2.snapshots["persist2"].content == "Content 2"

    def test_update_persists(self):
        """Test updates are persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = SnapshotManager(snapshot_dir=tmpdir)
            manager1.create_snapshot("update_persist", "Original")
            manager1.update_snapshot("update_persist", "Updated")

            manager2 = SnapshotManager(snapshot_dir=tmpdir)
            assert manager2.snapshots["update_persist"].content == "Updated"


class TestSnapshotSummary:
    """Test summary functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager with snapshots."""
        m = SnapshotManager()
        m.create_snapshot("s1", "Content 1")
        m.create_snapshot("s2", "Content 2")
        return m

    def test_get_summary(self, manager):
        """Test getting summary."""
        manager.compare("s1", "Content 1")  # Match
        manager.compare("s2", "Different")   # Mismatch
        manager.compare("s3", "New")         # Missing

        summary = manager.get_summary()

        assert summary["total_comparisons"] == 3
        assert summary["matches"] == 1
        assert summary["mismatches"] == 1
        assert summary["missing_snapshots"] == 1
        assert summary["pass_rate"] == pytest.approx(33.33, rel=0.1)
        assert summary["total_snapshots"] == 2

    def test_clear_results(self, manager):
        """Test clearing results."""
        manager.compare("s1", "Content 1")
        assert len(manager.results) == 1

        manager.clear_results()
        assert len(manager.results) == 0


class TestSimilarityCalculation:
    """Test similarity score calculation."""

    @pytest.fixture
    def manager(self):
        return SnapshotManager()

    def test_identical_strings(self, manager):
        """Test identical strings have similarity 1.0."""
        similarity = manager._compute_similarity("hello world", "hello world")
        assert similarity == 1.0

    def test_completely_different(self, manager):
        """Test completely different strings."""
        similarity = manager._compute_similarity("abc", "xyz")
        assert similarity < 0.5

    def test_similar_strings(self, manager):
        """Test similar strings have high similarity."""
        similarity = manager._compute_similarity(
            "The weather is sunny today",
            "The weather is cloudy today"
        )
        assert similarity > 0.7

    def test_empty_strings(self, manager):
        """Test empty strings."""
        assert manager._compute_similarity("", "") == 1.0
        assert manager._compute_similarity("hello", "") == 0.0
        assert manager._compute_similarity("", "world") == 0.0


class TestDetailedDiff:
    """Test detailed diff computation."""

    @pytest.fixture
    def manager(self):
        return SnapshotManager()

    def test_line_count_difference(self, manager):
        """Test detection of line count differences."""
        diff = manager._compute_detailed_diff("line1\nline2", "line1")

        assert any(d["type"] == "line_count" for d in diff)

    def test_line_content_difference(self, manager):
        """Test detection of line content differences."""
        diff = manager._compute_detailed_diff("line1\noriginal", "line1\nmodified")

        assert any(d["type"] == "line_diff" for d in diff)

    def test_diff_truncation(self, manager):
        """Test diff is truncated after 10 differences."""
        many_lines_a = "\n".join([f"line{i}" for i in range(20)])
        many_lines_b = "\n".join([f"different{i}" for i in range(20)])

        diff = manager._compute_detailed_diff(many_lines_a, many_lines_b)

        # Should be truncated
        assert any(d.get("type") == "truncated" for d in diff)


class TestWeatherSnapshotTests:
    """Test WeatherSnapshotTests class."""

    def test_create_weather_snapshots(self):
        """Test creating weather snapshots."""
        manager = SnapshotManager()
        WeatherSnapshotTests.create_weather_snapshots(manager)

        assert "weather_basic_response" in manager.snapshots
        assert "hurricane_warning_response" in manager.snapshots
        assert "evacuation_guidance_response" in manager.snapshots

    def test_weather_basic_response_content(self):
        """Test basic weather response snapshot content."""
        manager = SnapshotManager()
        WeatherSnapshotTests.create_weather_snapshots(manager)

        snap = manager.snapshots["weather_basic_response"]
        assert "Temperature" in snap.content
        assert "°F" in snap.content
        assert "Humidity" in snap.content

    def test_hurricane_warning_content(self):
        """Test hurricane warning snapshot content."""
        manager = SnapshotManager()
        WeatherSnapshotTests.create_weather_snapshots(manager)

        snap = manager.snapshots["hurricane_warning_response"]
        assert "HURRICANE WARNING" in snap.content
        assert "Category" in snap.content
        assert "SAFETY" in snap.content

    def test_evacuation_guidance_content(self):
        """Test evacuation guidance snapshot content."""
        manager = SnapshotManager()
        WeatherSnapshotTests.create_weather_snapshots(manager)

        snap = manager.snapshots["evacuation_guidance_response"]
        assert "EVACUATION" in snap.content
        assert "Zone" in snap.content
        assert "Shelter" in snap.content

    def test_get_standard_tests(self):
        """Test getting standard test list."""
        tests = WeatherSnapshotTests.get_standard_tests()

        assert len(tests) == 3
        ids = [t[0] for t in tests]
        assert "weather_basic_response" in ids
        assert "hurricane_warning_response" in ids
        assert "evacuation_guidance_response" in ids


class TestRunWeatherSnapshotTests:
    """Test convenience function."""

    def test_run_weather_snapshot_tests(self):
        """Test running weather snapshot tests."""
        summary = run_weather_snapshot_tests(update_mode=True)

        assert "total_comparisons" in summary
        assert "total_snapshots" in summary
        assert summary["total_snapshots"] == 3

    def test_run_with_temp_dir(self):
        """Test running with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_weather_snapshot_tests(
                snapshot_dir=tmpdir,
                update_mode=True
            )

            assert summary["total_snapshots"] == 3


class TestSnapshotModels:
    """Test Pydantic models."""

    def test_snapshot_result_model(self):
        """Test SnapshotResult model."""
        result = SnapshotResult(
            snapshot_id="test",
            status=SnapshotStatus.MATCH,
            expected_hash="abc123",
            actual_hash="abc123",
        )

        assert result.snapshot_id == "test"
        assert result.status == SnapshotStatus.MATCH
        assert result.expected_hash == result.actual_hash

    def test_snapshot_comparison_model(self):
        """Test SnapshotComparison model."""
        comparison = SnapshotComparison(
            snapshot_id="test",
            expected_content="expected",
            actual_content="actual",
            is_match=False,
            similarity_score=0.75,
        )

        assert comparison.is_match is False
        assert comparison.similarity_score == 0.75

    def test_snapshot_model(self):
        """Test Snapshot model."""
        snapshot = Snapshot(
            id="test",
            name="Test",
            content="Content",
            content_hash="abc123",
        )

        assert snapshot.id == "test"
        assert snapshot.metadata == {}

    def test_snapshot_status_enum(self):
        """Test SnapshotStatus enum values."""
        assert SnapshotStatus.MATCH == "match"
        assert SnapshotStatus.MISMATCH == "mismatch"
        assert SnapshotStatus.NEW == "new"
        assert SnapshotStatus.MISSING == "missing"
