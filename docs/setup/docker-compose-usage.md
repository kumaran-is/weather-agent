# Docker Compose Usage for MCP Servers

**Quick reference for using the unified docker-compose setup.**

---

## Overview

The `docker-compose.yml` file in the project root orchestrates both MCP servers required for the Weather AI Agent:

- **Weather MCP Server** (port 8080)
- **Hurricane Tracker MCP** (port 8081)

---

## Prerequisites

1. **MCP Server Repositories Cloned**:
   ```
   ~/mydrive/personal/
   ├── weather-ai-agent-service/    # This project
   ├── mcp-weather-server/          # Weather MCP
   └── hurricane-tracker-mcp/       # Hurricane MCP
   ```

2. **Docker Desktop Running**:
   ```bash
   docker info
   ```

---

## Basic Commands

### Start Both Servers

```bash
docker-compose up -d
```

Builds images (if needed) and starts both containers in detached mode.

### Stop Both Servers

```bash
docker-compose down
```

Stops and removes containers (images and volumes remain).

### View Status

```bash
docker-compose ps
```

Shows running containers with their status and ports.

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f weather-mcp
docker-compose logs -f hurricane-mcp

# Last 50 lines
docker-compose logs --tail 50
```

### Rebuild After Code Changes

```bash
docker-compose up -d --build
```

Forces rebuild of Docker images and restarts containers.

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart weather-mcp
docker-compose restart hurricane-mcp
```

---

## Advanced Usage

### Build Without Starting

```bash
docker-compose build
```

### Start Specific Service

```bash
# Start only Weather MCP
docker-compose up -d weather-mcp

# Start only Hurricane MCP
docker-compose up -d hurricane-mcp
```

### Stop Without Removing

```bash
docker-compose stop
```

### Remove Containers and Volumes

```bash
docker-compose down -v
```

**Warning**: This removes log volumes.

### Scale (Not Recommended for MCP)

```bash
# Don't use - MCP servers should run as singletons
# docker-compose up -d --scale weather-mcp=2  # ❌ DON'T DO THIS
```

---

## Environment Variables

Override default values using environment variables:

```bash
# Example: Change log level
export WEATHER_LOG_LEVEL=debug
export HURRICANE_LOG_LEVEL=debug
docker-compose up -d
```

Available variables (see `docker-compose.yml`):
- `NODE_ENV` (default: production)
- `WEATHER_LOG_LEVEL` (default: info)
- `HURRICANE_LOG_LEVEL` (default: info)
- `CACHE_MAX_SIZE` (default: 1000)
- `REQUEST_TIMEOUT_MS` (default: 30000)
- And more...

---

## Troubleshooting

### Port Already in Use

If you have individual MCP containers running, stop them first:

```bash
# Stop individual Weather MCP
cd ../mcp-weather-server
docker-compose down

# Stop individual Hurricane MCP
cd ../hurricane-tracker-mcp
docker-compose down

# Return and start unified
cd ../weather-ai-agent-service
docker-compose up -d
```

### Check Container Logs

```bash
# All logs
docker-compose logs

# Specific service errors
docker-compose logs weather-mcp | grep -i error
docker-compose logs hurricane-mcp | grep -i error
```

### Verify Network

```bash
docker network ls | grep mcp
docker network inspect weather-ai-mcp-network
```

### Health Check Status

```bash
# Weather MCP (may show unhealthy due to missing curl)
docker inspect weather-mcp-server --format='{{.State.Health.Status}}'

# Hurricane MCP (should show healthy)
docker inspect hurricane-tracker-mcp --format='{{.State.Health.Status}}'
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker-compose up -d` | Start both servers |
| `docker-compose down` | Stop both servers |
| `docker-compose ps` | View status |
| `docker-compose logs -f` | View logs (live) |
| `docker-compose up -d --build` | Rebuild and restart |
| `docker-compose restart` | Restart services |

---

## See Also

- [MCP Servers Setup Guide](mcp-servers-setup.md) - Full setup instructions
- [Docker Compose Documentation](https://docs.docker.com/compose/)
