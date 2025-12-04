"""FastAPI application for Weather AI Agent - Level 1.

This module provides the main FastAPI application with REST endpoints for
weather queries and hurricane alert management with HITL approval.

Level 1 Implementation:
- 3 main endpoints: /weather/query, /weather/hurricane/alert, /weather/hurricane/approve
- 1 health check endpoint: /health
- Basic error handling
- Pydantic request/response validation
- NO authentication (deferred to L5c)
- NO rate limiting (deferred to L5c)
- NO metrics tracking (deferred to L5c)

API Endpoints:
- POST /weather/query - Query weather via ReAct agent
- POST /weather/hurricane/alert - Create hurricane alert (may require approval)
- POST /weather/hurricane/approve/{thread_id} - Approve or reject pending alert
- GET /health - Health check
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from langgraph.types import Command

from backend.src.api.schemas import (
    WeatherQuery,
    WeatherResponse,
    HurricaneAlertRequest,
    HurricaneAlertResponse,
    HurricaneApprovalRequest,
    HurricaneApprovalResponse,
    HealthCheckResponse
)
from backend.src.agents.weather_agent import query_weather
from backend.src.workflows.weather_graph import get_weather_hitl_workflow
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Weather AI Agent API",
    description="Level 1: Basic ReAct Agent with HITL approval for hurricane alerts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (for Level 1, allow all origins)
# In production (L5c), configure specific allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will restrict in L5c
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow
workflow = get_weather_hitl_workflow()


@app.post(
    "/weather/query",
    response_model=WeatherResponse,
    status_code=status.HTTP_200_OK,
    summary="Query weather conditions",
    description="Ask the weather agent about current conditions or forecasts for any location",
    tags=["Weather"]
)
async def weather_query_endpoint(query: WeatherQuery):
    """Query weather via ReAct agent.

    This endpoint accepts natural language weather questions and returns
    responses from the ReAct agent using MCP weather tools.

    Example Request:
        POST /weather/query
        {
            "query": "What's the weather in London?",
            "user_id": "user123",
            "session_id": "session456"
        }

    Example Response:
        {
            "response": "The weather in London is currently 15°C and rainy with 85% humidity...",
            "user_id": "user123",
            "timestamp": "2025-12-03T16:20:00Z"
        }

    Args:
        query: WeatherQuery with user question, user_id, and optional session_id

    Returns:
        WeatherResponse with agent's answer and metadata

    Raises:
        HTTPException: 500 if agent query fails
    """
    try:
        logger.info(
            f"Weather query received | "
            f"user_id: {query.user_id} | "
            f"query: {query.query}"
        )

        # Query the weather agent
        result = await query_weather(query.query)

        logger.info(
            f"Weather query successful | "
            f"user_id: {query.user_id} | "
            f"response_length: {len(result)}"
        )

        return WeatherResponse(
            response=result,
            user_id=query.user_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error(
            f"Weather query failed | "
            f"user_id: {query.user_id} | "
            f"error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process weather query: {str(e)}"
        )


@app.post(
    "/weather/hurricane/alert",
    response_model=HurricaneAlertResponse,
    status_code=status.HTTP_200_OK,
    summary="Create hurricane alert",
    description="Create a hurricane alert that may require human approval (Cat 3+) before being sent",
    tags=["Hurricane Alerts"]
)
async def create_hurricane_alert(alert: HurricaneAlertRequest):
    """Create hurricane alert (may require human approval).

    This endpoint creates a hurricane alert that follows the HITL approval workflow:
    - Category 1-2: Auto-approved and sent immediately
    - Category 3-5: Requires human approval before sending

    Example Request (Cat 2 - auto-approved):
        POST /weather/hurricane/alert
        {
            "category": 2,
            "message": "Category 2 Hurricane Julia approaching with 95 mph winds",
            "thread_id": "alert-uuid"
        }

    Example Response (auto-approved):
        {
            "status": "sent",
            "thread_id": "alert-uuid"
        }

    Example Request (Cat 4 - requires approval):
        POST /weather/hurricane/alert
        {
            "category": 4,
            "message": "Category 4 Hurricane Ida approaching with 140 mph winds",
            "thread_id": "alert-uuid"
        }

    Example Response (pending approval):
        {
            "status": "pending_approval",
            "thread_id": "alert-uuid"
        }

    Args:
        alert: HurricaneAlertRequest with category, message, and thread_id

    Returns:
        HurricaneAlertResponse with status and thread_id

    Raises:
        HTTPException: 500 if workflow execution fails
    """
    config = {"configurable": {"thread_id": alert.thread_id}}

    try:
        logger.info(
            f"Hurricane alert received | "
            f"category: {alert.category} | "
            f"thread_id: {alert.thread_id}"
        )

        # Invoke workflow with hurricane alert data
        result = workflow.invoke({
            "user_id": "system",  # System-generated alert
            "session_id": alert.thread_id,
            "current_query": f"Category {alert.category} hurricane",
            "current_step": "input",
            "approved": False,
            "hurricane_category": alert.category,
            "alert_message": alert.message
        }, config)

        # Check if workflow is interrupted (pending human approval)
        if "__interrupt__" in result:
            logger.warning(
                f"Hurricane alert pending approval | "
                f"category: {alert.category} | "
                f"thread_id: {alert.thread_id}"
            )
            return HurricaneAlertResponse(
                status="pending_approval",
                thread_id=alert.thread_id,
                category=alert.category,
                message=alert.message
            )

        # Alert was auto-approved or completed
        final_status = "sent" if result.get("approved") else "cancelled"

        logger.info(
            f"Hurricane alert completed | "
            f"status: {final_status} | "
            f"category: {alert.category} | "
            f"thread_id: {alert.thread_id}"
        )

        return HurricaneAlertResponse(
            status=final_status,
            thread_id=alert.thread_id,
            category=alert.category,
            message=alert.message
        )

    except Exception as e:
        logger.error(
            f"Hurricane alert failed | "
            f"category: {alert.category} | "
            f"thread_id: {alert.thread_id} | "
            f"error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create hurricane alert: {str(e)}"
        )


@app.post(
    "/weather/hurricane/approve/{thread_id}",
    response_model=HurricaneApprovalResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve or reject hurricane alert",
    description="Approve or reject a pending hurricane alert (Cat 3+) after human review",
    tags=["Hurricane Alerts"]
)
async def approve_hurricane_alert(
    thread_id: str,
    approval: HurricaneApprovalRequest
):
    """Approve or reject pending hurricane alert.

    This endpoint is called by human reviewers to approve or reject
    Category 3+ hurricane alerts that are pending in the workflow.

    Example Request (approve):
        POST /weather/hurricane/approve/alert-uuid
        {
            "approved": true
        }

    Example Response:
        {
            "status": "sent",
            "thread_id": "alert-uuid"
        }

    Example Request (reject):
        POST /weather/hurricane/approve/alert-uuid
        {
            "approved": false
        }

    Example Response:
        {
            "status": "cancelled",
            "thread_id": "alert-uuid"
        }

    Args:
        thread_id: Thread ID of the pending alert workflow
        approval: HurricaneApprovalRequest with approval decision

    Returns:
        HurricaneApprovalResponse with final status

    Raises:
        HTTPException: 404 if thread_id not found
        HTTPException: 500 if workflow resume fails
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info(
            f"Hurricane approval received | "
            f"thread_id: {thread_id} | "
            f"approved: {approval.approved}"
        )

        # Resume workflow with approval decision
        result = workflow.invoke(
            Command(resume={"approved": approval.approved}),
            config
        )

        # Determine final status
        final_status = "sent" if result.get("approved") else "cancelled"

        logger.info(
            f"Hurricane approval completed | "
            f"status: {final_status} | "
            f"thread_id: {thread_id}"
        )

        return HurricaneApprovalResponse(
            status=final_status,
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(
            f"Hurricane approval failed | "
            f"thread_id: {thread_id} | "
            f"error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process hurricane approval: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the Weather AI Agent service is healthy and operational",
    tags=["System"]
)
async def health_check():
    """Health check endpoint.

    Returns service health status and current implementation level.

    Example Response:
        {
            "status": "healthy",
            "level": "1",
            "timestamp": "2025-12-03T16:20:00Z"
        }

    Returns:
        HealthCheckResponse with status, level, and timestamp
    """
    logger.debug("Health check requested")

    return HealthCheckResponse(
        status="healthy",
        level="1",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# Optional: Add startup and shutdown events for Level 1
@app.on_event("startup")
async def startup_event():
    """Log startup message."""
    logger.info("Weather AI Agent API starting up...")
    logger.info("Level 1: Basic ReAct Agent + HITL")
    logger.info("API documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown message."""
    logger.info("Weather AI Agent API shutting down...")


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # For development
        log_level="info"
    )
