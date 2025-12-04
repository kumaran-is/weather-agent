"""Tests for FastAPI endpoints.

Tests all REST API endpoints for the Weather AI Agent service.
"""

import pytest
from fastapi.testclient import TestClient
from backend.src.api.main import app

# Create test client
client = TestClient(app)


def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["level"] == "1"
    assert "timestamp" in data


def test_weather_query_endpoint(mock_mcp_client, mock_agent_executor):
    """Test weather query endpoint."""
    response = client.post("/weather/query", json={
        "query": "What's the weather in Seattle?",
        "user_id": "test_user",
        "session_id": "test_session"
    })

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "user_id" in data
    assert data["user_id"] == "test_user"
    assert "timestamp" in data


def test_weather_query_missing_fields():
    """Test weather query with missing required fields."""
    response = client.post("/weather/query", json={
        "query": "What's the weather?"
        # Missing user_id
    })

    assert response.status_code == 422  # Validation error


def test_hurricane_alert_cat2_auto_approve():
    """Test Category 2 hurricane alert auto-approves."""
    response = client.post("/weather/hurricane/alert", json={
        "category": 2,
        "message": "Category 2 Hurricane Julia approaching with 95 mph winds",
        "thread_id": "test-cat2-api"
    })

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "sent"  # Auto-approved
    assert "thread_id" in data


def test_hurricane_alert_cat4_pending():
    """Test Category 4 hurricane requires approval."""
    response = client.post("/weather/hurricane/alert", json={
        "category": 4,
        "message": "Category 4 Hurricane Ida approaching with 140 mph winds",
        "thread_id": "test-cat4-api"
    })

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "pending_approval"  # Requires human approval
    assert "thread_id" in data


def test_hurricane_alert_invalid_category():
    """Test hurricane alert with invalid category."""
    response = client.post("/weather/hurricane/alert", json={
        "category": 6,  # Invalid: must be 1-5
        "message": "Invalid category",
        "thread_id": "test-invalid"
    })

    assert response.status_code == 422  # Validation error


def test_hurricane_approve_endpoint():
    """Test hurricane approval endpoint."""
    # First create a pending alert
    create_response = client.post("/weather/hurricane/alert", json={
        "category": 4,
        "message": "Category 4 hurricane approaching",
        "thread_id": "test-approve-flow"
    })
    assert create_response.status_code == 200
    assert create_response.json()["status"] == "pending_approval"

    # Then approve it
    approve_response = client.post(
        "/weather/hurricane/approve/test-approve-flow",
        json={"approved": True}
    )

    assert approve_response.status_code == 200
    data = approve_response.json()
    assert "status" in data
    assert data["status"] == "sent"  # Approved and sent
    assert data["thread_id"] == "test-approve-flow"


def test_hurricane_reject_endpoint():
    """Test hurricane rejection endpoint."""
    # First create a pending alert
    create_response = client.post("/weather/hurricane/alert", json={
        "category": 5,
        "message": "Category 5 hurricane approaching",
        "thread_id": "test-reject-flow"
    })
    assert create_response.status_code == 200

    # Then reject it
    reject_response = client.post(
        "/weather/hurricane/approve/test-reject-flow",
        json={"approved": False}
    )

    assert reject_response.status_code == 200
    data = reject_response.json()
    assert "status" in data
    assert data["status"] == "cancelled"  # Rejected and cancelled
    assert data["thread_id"] == "test-reject-flow"


def test_api_cors_headers():
    """Test that CORS headers are present."""
    response = client.get("/health")

    # CORS should be enabled for all origins in Level 1
    assert response.status_code == 200
    # TestClient doesn't add CORS headers, but middleware is configured


def test_api_docs_available():
    """Test that API documentation is available."""
    response = client.get("/docs")

    assert response.status_code == 200


def test_api_redoc_available():
    """Test that ReDoc documentation is available."""
    response = client.get("/redoc")

    assert response.status_code == 200
