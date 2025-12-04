"""Tests for HITL approval workflow.

Tests the Human-in-the-Loop approval nodes for hurricane alerts.
"""

import pytest
from backend.src.hitl.approval_node import (
    detect_hurricane_node,
    send_alert_node,
    cancel_alert_node
)


def test_detect_hurricane_node(sample_hurricane_cat2):
    """Test hurricane detection node."""
    result = detect_hurricane_node(sample_hurricane_cat2)

    assert result is not None
    assert "hurricane_category" in result
    assert result["hurricane_category"] == 2
    assert "alert_message" in result


def test_send_alert_node(sample_hurricane_cat2):
    """Test send alert node."""
    result = send_alert_node(sample_hurricane_cat2)

    assert result is not None
    assert "status" in result
    assert result["status"] == "sent"
    assert "current_step" in result
    assert result["current_step"] == "response"


def test_cancel_alert_node(sample_hurricane_cat4):
    """Test cancel alert node."""
    result = cancel_alert_node(sample_hurricane_cat4)

    assert result is not None
    assert "status" in result
    assert result["status"] == "cancelled"
    assert "current_step" in result
    assert result["current_step"] == "response"


def test_category_extraction(sample_hurricane_cat2, sample_hurricane_cat4):
    """Test that category is correctly extracted from state."""
    result_cat2 = detect_hurricane_node(sample_hurricane_cat2)
    result_cat4 = detect_hurricane_node(sample_hurricane_cat4)

    assert result_cat2["hurricane_category"] == 2
    assert result_cat4["hurricane_category"] == 4
