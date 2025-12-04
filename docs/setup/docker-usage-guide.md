# Docker Usage Guide - Weather AI Agent Service

**Complete guide for managing Docker containers with and without hot reload**

---

## Table of Contents

- [Quick Start](#quick-start)
- [Production vs Development Mode](#production-vs-development-mode)
- [Essential Commands](#essential-commands)
- [Hot Reload Setup](#hot-reload-setup)
- [Health Checks](#health-checks)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Quick Start

### Current Setup (3 Containers)

| Container | Port | Service | Hot Reload |
|-----------|------|---------|------------|
| `weather-mcp-server` | 8080 | Weather MCP Server | ❌ No |
| `hurricane-tracker-mcp` | 8081 | Hurricane Tracker MCP | ❌ No |
| `weather-ai-api` | 8000 | Weather AI FastAPI | ❌ No (production)<br>✅ Yes (development) |

### One-Line Commands

**Start all containers** (production mode):
```bash
docker-compose up -d
```

**Start all containers** (development mode with hot reload):
```bash
docker-compose -f docker-compose.dev.yml up -d
```

**Stop all containers**:
```bash
docker-compose down
# OR for development:
docker-compose -f docker-compose.dev.yml down
```

---

## Production vs Development Mode

### Production Mode (`docker-compose.yml`)

**Features**:
- ✅ Optimized for deployment
- ✅ Multi-worker uvicorn (4 workers)
- ✅ Minimal image size
- ✅ Non-root user (security)
- ❌ **NO hot reload** - must rebuild after code changes

**When to Use**:
- Deploying to staging/production
- Testing production build
- Performance benchmarking
- Final validation before release

**Commands**:
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Development Mode (`docker-compose.dev.yml`)

**Features**:
- ✅ **Hot reload enabled** - code changes reflect immediately
- ✅ Dev dependencies included (pytest, mypy, ruff)
- ✅ Debug logging enabled
- ✅ Source code mounted as volumes
- ✅ LangSmith tracing enabled by default

**When to Use**:
- Local development
- Testing new features
- Debugging issues
- Iterative development

**Commands**:
```bash
# Start
docker-compose -f docker-compose.dev.yml up -d

# Stop
docker-compose -f docker-compose.dev.yml down

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Rebuild (only needed after dependency changes)
docker-compose -f docker-compose.dev.yml up -d --build
```

---

## Essential Commands

### Start/Stop Operations

**Start all services** (detached mode):
```bash
# Production
docker-compose up -d

# Development (hot reload)
docker-compose -f docker-compose.dev.yml up -d
```

**Start with logs visible** (foreground):
```bash
# Production
docker-compose up

# Development
docker-compose -f docker-compose.dev.yml up
```

**Stop all services**:
```bash
# Production
docker-compose down

# Development
docker-compose -f docker-compose.dev.yml down
```

**Stop and remove volumes**:
```bash
# Production
docker-compose down -v

# Development
docker-compose -f docker-compose.dev.yml down -v
```

### Rebuild Operations

**Rebuild specific service**:
```bash
# Production
docker-compose build weather-ai-api

# Development
docker-compose -f docker-compose.dev.yml build weather-ai-api
```

**Rebuild and restart all services**:
```bash
# Production
docker-compose up -d --build

# Development
docker-compose -f docker-compose.dev.yml up -d --build
```

**Force rebuild** (no cache):
```bash
# Production
docker-compose build --no-cache

# Development
docker-compose -f docker-compose.dev.yml build --no-cache
```

### Restart Operations

**Restart specific service**:
```bash
# Production
docker-compose restart weather-ai-api

# Development
docker-compose -f docker-compose.dev.yml restart weather-ai-api
```

**Restart all services**:
```bash
# Production
docker-compose restart

# Development
docker-compose -f docker-compose.dev.yml restart
```

### Logs and Monitoring

**View logs (all services, follow mode)**:
```bash
# Production
docker-compose logs -f

# Development
docker-compose -f docker-compose.dev.yml logs -f
```

**View logs (specific service)**:
```bash
# Production
docker-compose logs -f weather-ai-api
docker-compose logs -f weather-mcp
docker-compose logs -f hurricane-mcp

# Development
docker-compose -f docker-compose.dev.yml logs -f weather-ai-api
```

**View last N lines**:
```bash
# Production
docker-compose logs --tail=100 weather-ai-api

# Development
docker-compose -f docker-compose.dev.yml logs --tail=100 weather-ai-api
```

**View logs with timestamps**:
```bash
# Production
docker-compose logs -f --timestamps weather-ai-api

# Development
docker-compose -f docker-compose.dev.yml logs -f --timestamps weather-ai-api
```

### Status and Inspection

**Check service status**:
```bash
# Production
docker-compose ps

# Development
docker-compose -f docker-compose.dev.yml ps
```

**View resource usage**:
```bash
docker stats
```

**Inspect container details**:
```bash
docker inspect weather-ai-api
```

**Enter running container** (interactive shell):
```bash
docker exec -it weather-ai-api bash

# Or for development:
docker exec -it weather-ai-api-dev bash
```

---

## Hot Reload Setup

### Understanding Hot Reload

**What is Hot Reload?**
- Code changes automatically detected
- Application restarts with new code
- No need to rebuild Docker image
- Faster development iteration

**Current Status**:
- ❌ **Production mode**: Hot reload **DISABLED**
- ✅ **Development mode**: Hot reload **ENABLED**

### How Hot Reload Works

**Development mode configuration**:

1. **Dockerfile** uses `development` target (line 63):
```dockerfile
CMD ["uv", "run", "uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
#                                                                                             ^^^^^^^^ Hot reload enabled
```

2. **docker-compose.dev.yml** mounts source code as volumes:
```yaml
volumes:
  - ./backend:/app/backend:ro       # Mount backend code
  - ./pyproject.toml:/app/pyproject.toml:ro
  - ./uv.lock:/app/uv.lock:ro
```

3. **Uvicorn** watches for file changes and automatically restarts

### Testing Hot Reload

**Step 1: Start development containers**:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

**Step 2: Watch logs**:
```bash
docker-compose -f docker-compose.dev.yml logs -f weather-ai-api
```

**Step 3: Make a code change**:
Edit `backend/src/api/main.py`:
```python
@app.get("/")
async def root():
    return {"message": "Hello World - UPDATED"}  # Change this line
```

**Step 4: Observe auto-restart**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [1] using WatchFiles
INFO:     Started server process [8]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
WARNING:  WatchFiles detected changes in 'backend/src/api/main.py'. Reloading...
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [8]
INFO:     Started server process [9]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Step 5: Verify changes**:
```bash
curl http://localhost:8000/
# Should show: {"message":"Hello World - UPDATED"}
```

### When to Rebuild (Even in Development Mode)

You **DO need to rebuild** if you change:
- ✅ `pyproject.toml` (new dependencies)
- ✅ `uv.lock` (dependency versions)
- ✅ `Dockerfile` (build steps)
- ✅ System dependencies (apt packages)

You **DON'T need to rebuild** if you change:
- ❌ Python code in `backend/`
- ❌ Configuration in `.env`
- ❌ Documentation files

---

## Health Checks

### Automated Health Checks

All containers have automated health checks that run every 30 seconds:

**Weather MCP Server**:
```bash
curl http://localhost:8080/health
```

**Hurricane Tracker MCP**:
```bash
curl http://localhost:8081/health
```

**Weather AI API**:
```bash
curl http://localhost:8000/health
```

### Manual Health Verification

**Check all services at once**:
```bash
# Production
docker-compose ps

# Development
docker-compose -f docker-compose.dev.yml ps
```

**Expected output**:
```
NAME                    STATUS              PORTS
weather-ai-api          Up 2 minutes        0.0.0.0:8000->8000/tcp
weather-mcp-server      Up 2 minutes        0.0.0.0:8080->8080/tcp
hurricane-tracker-mcp   Up 2 minutes        0.0.0.0:8081->8080/tcp
```

### Health Check Troubleshooting

**Container unhealthy**:
```bash
# View logs
docker-compose logs weather-ai-api

# Check health check command
docker inspect weather-ai-api | grep -A 10 Healthcheck

# Restart container
docker-compose restart weather-ai-api
```

---

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

**Symptoms**: Container exits immediately or keeps restarting

**Solution**:
```bash
# View logs to identify error
docker-compose logs weather-ai-api

# Common causes:
# - Missing API keys in .env
# - Port already in use
# - Dependency missing
```

#### 2. Port Already in Use

**Symptoms**: `Error: bind: address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use port 8001 on host instead
```

#### 3. Code Changes Not Reflecting

**Symptoms**: Code changes don't appear after editing files

**Check**:
```bash
# 1. Verify using development mode
docker-compose -f docker-compose.dev.yml ps

# 2. Verify volumes mounted
docker inspect weather-ai-api-dev | grep -A 10 Mounts

# 3. Check uvicorn logs for reload messages
docker-compose -f docker-compose.dev.yml logs -f weather-ai-api
```

**Solution**:
```bash
# If using production mode, switch to development:
docker-compose down
docker-compose -f docker-compose.dev.yml up -d

# Or rebuild production image:
docker-compose up -d --build
```

#### 4. MCP Servers Unreachable

**Symptoms**: `Connection refused` errors when calling MCP tools

**Solution**:
```bash
# 1. Check MCP containers are running
docker-compose ps

# 2. Test health endpoints directly
curl http://localhost:8080/health
curl http://localhost:8081/health

# 3. Verify network connectivity
docker exec -it weather-ai-api curl http://weather-mcp:8080/health

# 4. Check environment variables
docker exec -it weather-ai-api env | grep MCP_
```

#### 5. Out of Disk Space

**Symptoms**: `no space left on device`

**Solution**:
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused (CAUTION)
docker system prune -a --volumes
```

---

## Best Practices

### Development Workflow

**Recommended daily workflow**:

1. **Start containers** (development mode):
```bash
docker-compose -f docker-compose.dev.yml up -d
```

2. **Watch logs** (in separate terminal):
```bash
docker-compose -f docker-compose.dev.yml logs -f weather-ai-api
```

3. **Edit code** in your IDE - changes auto-reload

4. **Test endpoints**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Miami?"}'
```

5. **Run tests** inside container:
```bash
docker exec -it weather-ai-api-dev pytest tests/
```

6. **Stop containers** at end of day:
```bash
docker-compose -f docker-compose.dev.yml down
```

### Production Deployment

**Before deploying to production**:

1. **Test production build locally**:
```bash
docker-compose up -d --build
```

2. **Run integration tests**:
```bash
docker exec -it weather-ai-api pytest tests/test_integration.py
```

3. **Check resource usage**:
```bash
docker stats
```

4. **Verify health checks**:
```bash
curl http://localhost:8000/health
curl http://localhost:8080/health
curl http://localhost:8081/health
```

5. **Review logs for errors**:
```bash
docker-compose logs --tail=100
```

### Environment Variables

**Development** (`.env.dev` - example):
```bash
# LLM API Keys
OPENAI_API_KEY=sk-proj-...
LANGCHAIN_API_KEY=lsv2_pt_...

# Application
ENVIRONMENT=development
LOG_LEVEL=debug

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=weather-ai-agent-dev
```

**Production** (`.env.prod` - example):
```bash
# LLM API Keys
OPENAI_API_KEY=sk-proj-...
LANGCHAIN_API_KEY=lsv2_pt_...

# Application
ENVIRONMENT=production
LOG_LEVEL=info

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=weather-ai-agent-prod
```

---

## Summary

### Quick Reference Table

| Task | Production | Development |
|------|-----------|-------------|
| **Start** | `docker-compose up -d` | `docker-compose -f docker-compose.dev.yml up -d` |
| **Stop** | `docker-compose down` | `docker-compose -f docker-compose.dev.yml down` |
| **Logs** | `docker-compose logs -f` | `docker-compose -f docker-compose.dev.yml logs -f` |
| **Rebuild** | `docker-compose up -d --build` | `docker-compose -f docker-compose.dev.yml up -d --build` |
| **Hot Reload** | ❌ No | ✅ Yes |
| **When to Use** | Deployment, final testing | Daily development |

### Key Takeaways

1. **Use development mode** (`docker-compose.dev.yml`) for daily work with hot reload
2. **Use production mode** (`docker-compose.yml`) for deployment and final testing
3. **Hot reload only works** in development mode with mounted volumes
4. **Rebuild is required** only when changing dependencies or Dockerfile
5. **Health checks** verify all services are running correctly
6. **View logs** to debug issues (`docker-compose logs -f`)

---

*Last Updated*: December 4, 2025
*Version*: 1.0.0
*For*: Weather AI Agent Service Level 1
