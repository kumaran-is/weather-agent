"""Health Check Utilities for All Services

This module provides async health check functions for all services in the Weather AI Agent stack.
Level 5c implementation for comprehensive observability.

Services monitored:
- Redis: Short-term memory and L2 cache
- Neo4j: Long-term memory via Graphiti
- Qdrant: Vector database for RAG
- PostgreSQL: Procedural memory (if configured)
- Weather MCP: Weather data service
- Hurricane MCP: Hurricane tracking service
- Prometheus: Metrics collection
- Grafana: Dashboards (optional)
- Loki: Log aggregation (optional)
"""

import asyncio
import logging
import time
from typing import Literal

import httpx

from backend.src.models.health import ServiceHealth

logger = logging.getLogger(__name__)

# Health check timeout (in seconds)
HEALTH_CHECK_TIMEOUT = 5.0


async def check_redis_health(redis_url: str) -> ServiceHealth:
    """Check Redis health by sending PING command.

    Args:
        redis_url: Redis connection URL (e.g., redis://localhost:6379/0)

    Returns:
        ServiceHealth with status and latency
    """
    try:
        import redis.asyncio as aioredis

        start = time.time()
        client = aioredis.from_url(redis_url, socket_timeout=HEALTH_CHECK_TIMEOUT)

        # PING command - fastest health check
        pong = await client.ping()
        latency_ms = (time.time() - start) * 1000

        # Get server info for version
        info = await client.info("server")
        version = info.get("redis_version", "unknown")

        await client.close()

        if pong:
            return ServiceHealth(
                status="healthy",
                latency_ms=round(latency_ms, 2),
                message="Connected",
                version=version,
            )
        else:
            return ServiceHealth(
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                message="PING failed",
            )

    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return ServiceHealth(
            status="unhealthy",
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_neo4j_health(
    bolt_url: str = "bolt://neo4j:7687",
    username: str = "neo4j",
    password: str = "password",
) -> ServiceHealth:
    """Check Neo4j health by running a simple query.

    Args:
        bolt_url: Neo4j Bolt URL (e.g., bolt://localhost:7687)
        username: Neo4j username
        password: Neo4j password

    Returns:
        ServiceHealth with status and latency
    """
    try:
        from neo4j import AsyncGraphDatabase

        start = time.time()
        driver = AsyncGraphDatabase.driver(
            bolt_url,
            auth=(username, password),
        )

        # Simple query to verify connectivity
        async with driver.session() as session:
            result = await session.run("RETURN 1 as n")
            record = await result.single()
            latency_ms = (time.time() - start) * 1000

        # Get server version
        async with driver.session() as session:
            result = await session.run("CALL dbms.components() YIELD versions RETURN versions[0] as version")
            version_record = await result.single()
            version = version_record["version"] if version_record else "unknown"

        await driver.close()

        if record and record["n"] == 1:
            return ServiceHealth(
                status="healthy",
                latency_ms=round(latency_ms, 2),
                message="Connected",
                version=version,
            )
        else:
            return ServiceHealth(
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                message="Query validation failed",
            )

    except Exception as e:
        logger.warning(f"Neo4j health check failed: {e}")
        return ServiceHealth(
            status="unhealthy",
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_qdrant_health(qdrant_url: str) -> ServiceHealth:
    """Check Qdrant health via HTTP API.

    Args:
        qdrant_url: Qdrant HTTP URL (e.g., http://localhost:6333)

    Returns:
        ServiceHealth with status and latency
    """
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            # Qdrant health endpoint
            response = await client.get(f"{qdrant_url}/")
            latency_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            version = data.get("version", "unknown")
            return ServiceHealth(
                status="healthy",
                latency_ms=round(latency_ms, 2),
                message="Connected",
                version=version,
            )
        else:
            return ServiceHealth(
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                message=f"HTTP {response.status_code}",
            )

    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        return ServiceHealth(
            status="unhealthy",
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_postgres_health(postgres_url: str | None) -> ServiceHealth:
    """Check PostgreSQL health via connection test.

    Args:
        postgres_url: PostgreSQL connection URL or None if not configured

    Returns:
        ServiceHealth with status and latency
    """
    if not postgres_url:
        return ServiceHealth(
            status="disabled",
            message="Not configured",
        )

    try:
        import psycopg

        start = time.time()

        # psycopg3 async connection
        async with await psycopg.AsyncConnection.connect(
            postgres_url,
            connect_timeout=int(HEALTH_CHECK_TIMEOUT),
        ) as conn:
            async with conn.cursor() as cur:
                # Simple query
                await cur.execute("SELECT version()")
                row = await cur.fetchone()
                version = row[0] if row else "unknown"

        latency_ms = (time.time() - start) * 1000

        # Extract version number
        version_short = version.split(",")[0] if version else "unknown"

        return ServiceHealth(
            status="healthy",
            latency_ms=round(latency_ms, 2),
            message="Connected",
            version=version_short,
        )

    except Exception as e:
        logger.warning(f"PostgreSQL health check failed: {e}")
        return ServiceHealth(
            status="unhealthy",
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_mcp_server_health(mcp_url: str, service_name: str) -> ServiceHealth:
    """Check MCP server health via /health endpoint.

    Args:
        mcp_url: MCP server URL (e.g., http://localhost:8080)
        service_name: Service name for logging

    Returns:
        ServiceHealth with status and latency
    """
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{mcp_url}/health")
            latency_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            return ServiceHealth(
                status="healthy",
                latency_ms=round(latency_ms, 2),
                message=data.get("status", "OK"),
                version=data.get("version"),
            )
        else:
            return ServiceHealth(
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                message=f"HTTP {response.status_code}",
            )

    except Exception as e:
        logger.warning(f"{service_name} health check failed: {e}")
        return ServiceHealth(
            status="unhealthy",
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_prometheus_health(prometheus_url: str = "http://prometheus:9090") -> ServiceHealth:
    """Check Prometheus health via /-/healthy endpoint.

    Args:
        prometheus_url: Prometheus server URL

    Returns:
        ServiceHealth with status and latency
    """
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{prometheus_url}/-/healthy")
            latency_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            # Get version from build info endpoint
            try:
                build_response = await client.get(f"{prometheus_url}/api/v1/status/buildinfo")
                if build_response.status_code == 200:
                    build_data = build_response.json()
                    version = build_data.get("data", {}).get("version", "unknown")
                else:
                    version = None
            except Exception:
                version = None

            return ServiceHealth(
                status="healthy",
                latency_ms=round(latency_ms, 2),
                message="OK",
                version=version,
            )
        else:
            return ServiceHealth(
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                message=f"HTTP {response.status_code}",
            )

    except Exception as e:
        logger.warning(f"Prometheus health check failed: {e}")
        return ServiceHealth(
            status="unhealthy",
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_grafana_health(grafana_url: str = "http://grafana:3000") -> ServiceHealth:
    """Check Grafana health via /api/health endpoint.

    Args:
        grafana_url: Grafana server URL

    Returns:
        ServiceHealth with status and latency
    """
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{grafana_url}/api/health")
            latency_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            return ServiceHealth(
                status="healthy",
                latency_ms=round(latency_ms, 2),
                message=data.get("database", "OK"),
                version=data.get("version"),
            )
        else:
            return ServiceHealth(
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                message=f"HTTP {response.status_code}",
            )

    except Exception as e:
        logger.warning(f"Grafana health check failed: {e}")
        return ServiceHealth(
            status="unhealthy",
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_loki_health(loki_url: str = "http://loki:3100") -> ServiceHealth:
    """Check Loki health via /ready endpoint.

    Args:
        loki_url: Loki server URL

    Returns:
        ServiceHealth with status and latency
    """
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{loki_url}/ready")
            latency_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            return ServiceHealth(
                status="healthy",
                latency_ms=round(latency_ms, 2),
                message="Ready",
            )
        else:
            return ServiceHealth(
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                message=f"HTTP {response.status_code}",
            )

    except Exception as e:
        logger.warning(f"Loki health check failed: {e}")
        return ServiceHealth(
            status="unhealthy",
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_all_services(
    redis_url: str,
    neo4j_bolt_url: str,
    qdrant_url: str,
    postgres_url: str | None,
    weather_mcp_url: str,
    hurricane_mcp_url: str,
    prometheus_url: str = "http://prometheus:9090",
    grafana_url: str = "http://grafana:3000",
    loki_url: str = "http://loki:3100",
    neo4j_username: str = "neo4j",
    neo4j_password: str = "password",
) -> dict[str, ServiceHealth]:
    """Check health of all services concurrently.

    Args:
        redis_url: Redis connection URL
        neo4j_bolt_url: Neo4j Bolt URL
        qdrant_url: Qdrant HTTP URL
        postgres_url: PostgreSQL URL (or None)
        weather_mcp_url: Weather MCP server URL
        hurricane_mcp_url: Hurricane MCP server URL
        prometheus_url: Prometheus URL
        grafana_url: Grafana URL
        loki_url: Loki URL
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        Dictionary with service name -> ServiceHealth
    """
    # Run all health checks concurrently
    results = await asyncio.gather(
        check_redis_health(redis_url),
        check_neo4j_health(neo4j_bolt_url, neo4j_username, neo4j_password),
        check_qdrant_health(qdrant_url),
        check_postgres_health(postgres_url),
        check_mcp_server_health(weather_mcp_url, "Weather MCP"),
        check_mcp_server_health(hurricane_mcp_url, "Hurricane MCP"),
        check_prometheus_health(prometheus_url),
        check_grafana_health(grafana_url),
        check_loki_health(loki_url),
        return_exceptions=True,
    )

    # Map results to service names
    service_names = [
        "redis",
        "neo4j",
        "qdrant",
        "postgres",
        "weather_mcp",
        "hurricane_mcp",
        "prometheus",
        "grafana",
        "loki",
    ]

    health_results = {}
    for name, result in zip(service_names, results):
        if isinstance(result, Exception):
            logger.error(f"Health check for {name} raised exception: {result}")
            health_results[name] = ServiceHealth(
                status="unknown",
                message=f"Check failed: {str(result)[:100]}",
            )
        else:
            health_results[name] = result

    return health_results


def calculate_overall_status(
    services: dict[str, ServiceHealth],
    critical_services: list[str] | None = None,
) -> Literal["healthy", "degraded", "unhealthy"]:
    """Calculate overall system health status.

    Args:
        services: Dictionary of service name -> ServiceHealth
        critical_services: List of service names that are critical for "healthy" status.
                          Defaults to ["redis", "neo4j", "weather_mcp"]

    Returns:
        Overall status: healthy, degraded, or unhealthy
    """
    if critical_services is None:
        critical_services = ["redis", "neo4j", "weather_mcp"]

    healthy_count = 0
    critical_unhealthy = False

    for name, health in services.items():
        if health.status == "healthy":
            healthy_count += 1
        elif health.status == "unhealthy" and name in critical_services:
            critical_unhealthy = True

    # If any critical service is unhealthy, overall is unhealthy
    if critical_unhealthy:
        return "unhealthy"

    # If all services are healthy (excluding disabled), overall is healthy
    enabled_services = [h for h in services.values() if h.status != "disabled"]
    if all(h.status == "healthy" for h in enabled_services):
        return "healthy"

    # Otherwise, degraded
    return "degraded"
