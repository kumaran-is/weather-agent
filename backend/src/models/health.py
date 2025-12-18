"""Health Check Models

This module defines Pydantic models for service health check endpoint.

Level 1 Implementation:
- HealthCheckResponse: Health status with current level and timestamp

Level 5c Enhancement:
- Comprehensive service health checks for all containers
- Detailed status for: Redis, Neo4j, Qdrant, PostgreSQL, MCP servers, Observability stack
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ServiceHealth(BaseModel):
    """Health status for an individual service."""

    status: Literal["healthy", "unhealthy", "disabled", "unknown"] = Field(
        description="Service health status"
    )
    latency_ms: float | None = Field(
        default=None,
        description="Response latency in milliseconds"
    )
    message: str | None = Field(
        default=None,
        description="Additional status message or error details"
    )
    version: str | None = Field(
        default=None,
        description="Service version if available"
    )


class ServicesHealth(BaseModel):
    """Health status for all services."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "redis": {"status": "healthy", "latency_ms": 1.2, "message": "Connected"},
                    "neo4j": {"status": "healthy", "latency_ms": 15.3, "message": "Connected"},
                    "qdrant": {"status": "healthy", "latency_ms": 8.5, "message": "Connected"},
                    "postgres": {"status": "disabled", "message": "Not configured"},
                    "weather_mcp": {"status": "healthy", "latency_ms": 45.2, "message": "OK"},
                    "hurricane_mcp": {"status": "healthy", "latency_ms": 38.7, "message": "OK"},
                    "prometheus": {"status": "healthy", "latency_ms": 5.1, "message": "OK"},
                    "grafana": {"status": "healthy", "latency_ms": 12.4, "message": "OK"},
                    "loki": {"status": "healthy", "latency_ms": 6.8, "message": "OK"}
                }
            ]
        }
    }

    redis: ServiceHealth = Field(description="Redis cache/memory service")
    neo4j: ServiceHealth = Field(description="Neo4j graph database (Graphiti)")
    qdrant: ServiceHealth = Field(description="Qdrant vector database")
    postgres: ServiceHealth = Field(description="PostgreSQL database")
    weather_mcp: ServiceHealth = Field(description="Weather MCP Server")
    hurricane_mcp: ServiceHealth = Field(description="Hurricane Tracker MCP")
    prometheus: ServiceHealth = Field(description="Prometheus metrics server")
    grafana: ServiceHealth = Field(description="Grafana dashboard")
    loki: ServiceHealth = Field(description="Loki log aggregation")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint.

    Returns service health status and current implementation level.
    Level 5c includes comprehensive health checks for all services.

    Test Scenarios (from test guide):
    - Scenario: Basic health check verification
    - Expected: status="healthy", level="L4+L5a"
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "level": "L4+L5a",
                    "timestamp": "2025-12-11T20:00:00Z",
                    "healthy_services": 7,
                    "total_services": 9,
                    "services": {
                        "redis": {"status": "healthy", "latency_ms": 1.2, "message": "Connected"},
                        "neo4j": {"status": "healthy", "latency_ms": 15.3, "message": "Connected"},
                        "qdrant": {"status": "healthy", "latency_ms": 8.5, "message": "Connected"},
                        "postgres": {"status": "disabled", "message": "Not configured"},
                        "weather_mcp": {"status": "healthy", "latency_ms": 45.2, "message": "OK"},
                        "hurricane_mcp": {"status": "healthy", "latency_ms": 38.7, "message": "OK"},
                        "prometheus": {"status": "healthy", "latency_ms": 5.1, "message": "OK"},
                        "grafana": {"status": "disabled", "message": "Not configured"},
                        "loki": {"status": "disabled", "message": "Not configured"}
                    }
                }
            ]
        }
    }

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        default="healthy",
        description="Overall health status (healthy=all core services up, degraded=some services down, unhealthy=critical services down)"
    )
    level: str = Field(
        default="L4+L5a",
        description="Current implementation level (L4 Multi-Agent + L5a Caching)"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 timestamp in UTC"
    )
    healthy_services: int = Field(
        default=0,
        description="Number of healthy services"
    )
    total_services: int = Field(
        default=9,
        description="Total number of monitored services"
    )
    services: ServicesHealth | None = Field(
        default=None,
        description="Detailed health status for each service"
    )
