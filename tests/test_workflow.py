"""Tests for LangGraph workflow.

Tests the Level 1 workflow orchestration with HITL approval.
"""

from backend.src.workflows.weather_graph import build_weather_hitl_workflow


def test_build_weather_hitl_workflow():
    """Test that Weather HITL workflow can be built."""
    graph = build_weather_hitl_workflow()

    assert graph is not None
    # Workflow should be compiled and ready to use


def test_workflow_cat2_auto_approve(sample_hurricane_cat2):
    """Test Category 2 hurricane auto-approves."""
    graph = build_weather_hitl_workflow()
    config = {"configurable": {"thread_id": "test-cat2-auto"}}

    result = graph.invoke(sample_hurricane_cat2, config)

    # Category 2 should auto-approve (no interrupt)
    assert result is not None
    assert "__interrupt__" not in result
    # Should have completed successfully
    assert result.get("approved") is True


def test_workflow_cat4_requires_approval(sample_hurricane_cat4):
    """Test Category 4 hurricane requires approval."""
    graph = build_weather_hitl_workflow()
    config = {"configurable": {"thread_id": "test-cat4-approval"}}

    result = graph.invoke(sample_hurricane_cat4, config)

    # Category 4 should require approval (has interrupt)
    assert result is not None
    assert "__interrupt__" in result
    # Interrupt should have approval request details
    interrupt_value = result["__interrupt__"]
    assert interrupt_value is not None


def test_workflow_has_checkpointer():
    """Test that workflow has a checkpointer configured."""
    graph = build_weather_hitl_workflow()

    # Workflow should have checkpointer for HITL
    assert graph.checkpointer is not None


def test_workflow_different_categories():
    """Test workflow behavior with different hurricane categories."""
    graph = build_weather_hitl_workflow()

    # Test Cat 1 (should auto-approve)
    config_cat1 = {"configurable": {"thread_id": "test-cat1"}}
    result_cat1 = graph.invoke({
        "user_id": "test",
        "session_id": "test",
        "current_query": "Cat 1 hurricane",
        "current_step": "input",
        "approved": False,
        "hurricane_category": 1,
        "alert_message": "Category 1 hurricane detected"
    }, config_cat1)
    assert "__interrupt__" not in result_cat1

    # Test Cat 5 (should require approval)
    config_cat5 = {"configurable": {"thread_id": "test-cat5"}}
    result_cat5 = graph.invoke({
        "user_id": "test",
        "session_id": "test",
        "current_query": "Cat 5 hurricane",
        "current_step": "input",
        "approved": False,
        "hurricane_category": 5,
        "alert_message": "Category 5 hurricane detected"
    }, config_cat5)
    assert "__interrupt__" in result_cat5
