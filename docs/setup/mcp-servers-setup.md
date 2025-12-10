# MCP Servers Setup Guide

**Quick setup guide for running Weather MCP Server and Hurricane Tracker MCP as Docker containers.**

---

## Prerequisites

- **Docker Desktop**: Running and accessible
- **Git**: For cloning repositories

Verify Docker is running:
```bash
docker --version
docker info
```

---

## Setup Options

Choose one of the following setup methods:

- **Option 1 (Recommended)**: Unified docker-compose - Start both servers with one command
- **Option 2**: Individual setup - Set up each server separately

---

## Option 1: Unified Docker Compose (Recommended) ⭐

**Easiest method** - Start both MCP servers with a single command.

### 1. Clone Both Repositories

```bash
# Navigate to your projects directory
cd ~/mydrive/personal  # Or your preferred directory

# Clone Weather MCP Server
git clone https://github.com/kumaran-is/mcp-weather-server.git
cd mcp-weather-server
git checkout develop
cd ..

# Clone Hurricane Tracker MCP
git clone https://github.com/kumaran-is/hurricane-tracker-mcp.git
cd ..

# Return to weather-agent
cd weather-agent
```

**Directory structure**:
```
~/mydrive/personal/
├── weather-agent/       # This project
│   └── docker-compose.yml          # Orchestrates both MCP servers
├── mcp-weather-server/             # Weather MCP (port 8080)
└── hurricane-tracker-mcp/          # Hurricane MCP (port 8081)
```

### 2. Start Both Servers

From the `weather-agent` directory:

```bash
docker-compose up -d --build
```

This will:
- Build both MCP server Docker images
- Start Weather MCP on port 8080
- Start Hurricane Tracker MCP on port 8081
- Run both in detached mode (background)

### 3. Verify Both Servers

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Check specific service logs
docker-compose logs weather-mcp
docker-compose logs hurricane-mcp
```

**Expected output**:
```
NAME                    STATUS                    PORTS
weather-mcp-server      Up X minutes              0.0.0.0:8080->8080/tcp
hurricane-tracker-mcp   Up X minutes (healthy)    0.0.0.0:8081->8080/tcp, 0.0.0.0:9090->9090/tcp
```

### 4. Stop/Restart Servers

```bash
# Stop both servers
docker-compose down

# Restart both servers
docker-compose up -d

# Rebuild and restart (after code changes)
docker-compose up -d --build

# View logs in real-time
docker-compose logs -f
```

### 5. Update Project .env

In your `weather-agent/.env` file:

```bash
# Weather MCP Server
MCP_WEATHER_SERVER_URL=http://localhost:8080
MCP_WEATHER_SERVER_ENABLED=true

# Hurricane Tracker MCP
MCP_HURRICANE_SERVER_URL=http://localhost:8081
MCP_HURRICANE_SERVER_ENABLED=true
```

**Done!** ✅ Both MCP servers are now running. Skip to [Testing & Validation](#testing--validation).

---

## Option 2: Individual Setup (Alternative)

Set up each server separately if you prefer granular control.

### Weather MCP Server (Port 8080)

#### Clone Repository
```bash
cd ~/mydrive/personal
git clone https://github.com/kumaran-is/mcp-weather-server.git
cd mcp-weather-server
git checkout develop
```

#### Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Set transport to http
sed -i '' 's/MCP_TRANSPORT=stdio/MCP_TRANSPORT=http/' .env
```

#### Build and Run
```bash
docker-compose up --build -d
```

#### Verify
```bash
docker ps --filter "name=mcp-weather"
docker-compose logs --tail 50
```

### Hurricane Tracker MCP (Port 8081)

#### Clone Repository
```bash
cd ~/mydrive/personal
git clone https://github.com/kumaran-is/hurricane-tracker-mcp.git
cd hurricane-tracker-mcp
```

#### Generate package-lock.json
```bash
npm install
```

#### Configure Port Mapping
Edit `docker-compose.yml` to use port 8081:

```bash
sed -i '' 's/"${HTTP_PORT:-8080}:${HTTP_PORT:-8080}"/"8081:8080"/' docker-compose.yml
```

Or manually change line ~11:
```yaml
ports:
  - "8081:8080"  # Host port 8081 → Container port 8080
```

#### Build and Run
```bash
docker-compose up --build -d
```

#### Verify
```bash
docker ps --filter "name=hurricane"
docker-compose logs --tail 50
```

---

## Testing & Validation

### Quick Status Check

```bash
# From weather-agent directory (if using Option 1)
docker-compose ps

# Or check all containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Test MCP Protocol (Optional)

**Weather MCP Server** (Port 8080):
```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
```

**Hurricane Tracker MCP** (Port 8081):
```bash
curl -X POST http://localhost:8081/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
```

**Expected Response** (Server-Sent Events format):
```
event: message
data: {"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":"mcp-weather-server","version":"3.1.0"}},"jsonrpc":"2.0","id":"1"}
```

**Note**: The `Accept: application/json, text/event-stream` header is required for HTTP Streamable transport. Without it, you'll get:
```
{"jsonrpc":"2.0","error":{"code":-32000,"message":"Not Acceptable: Client must accept both application/json and text/event-stream"},"id":null}
```

### Test Weather Query (Optional)

For HTTP Streamable transport, sessions are managed via **cookies**. Use curl's cookie jar:

**Step 1: Initialize and save cookies**
```bash
curl -X POST http://localhost:8080/mcp \
  -c cookies.txt \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
```

**Step 2: List available tools (using saved cookies)**
```bash
curl -X POST http://localhost:8080/mcp \
  -b cookies.txt \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{"jsonrpc":"2.0","id":"2","method":"tools/list"}'
```

**Step 3: Call weather tool (using saved cookies)**
```bash
curl -X POST http://localhost:8080/mcp \
  -b cookies.txt \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{"jsonrpc":"2.0","id":"3","method":"tools/call","params":{"name":"get_current_weather","arguments":{"city":"London"}}}'
```

**Expected**: Weather data for London in Server-Sent Events format

**Note**:
- `-c cookies.txt` saves cookies from the response
- `-b cookies.txt` sends cookies with the request
- The MCP-Session header is **not needed** for HTTP transport (sessions use cookies)

---

## Troubleshooting

### Weather MCP Shows "unhealthy"
- **Cause**: Docker health check uses `curl` which isn't in alpine container
- **Impact**: None - server is functional
- **Solution**: Ignore health check status, verify by checking logs

### Port Already in Use
```bash
# Find process using port
lsof -i :8080
lsof -i :8081

# Stop conflicting process or change port in docker-compose.yml
```

### Container Fails to Start
```bash
# Check logs
docker-compose logs <service-name>

# Common fixes:
# 1. Ensure parent directories exist (mcp-weather-server, hurricane-tracker-mcp)
# 2. Ensure repositories are cloned
# 3. Verify Docker has enough resources (Settings → Resources)
```

### Rebuild After Changes
```bash
# Option 1 (unified):
docker-compose down
docker-compose up -d --build

# Option 2 (individual):
cd <mcp-server-directory>
docker-compose down
docker-compose up --build -d
```

### View Real-time Logs
```bash
# Both servers (Option 1):
docker-compose logs -f

# Specific service (Option 1):
docker-compose logs -f weather-mcp
docker-compose logs -f hurricane-mcp

# Individual (Option 2):
cd <mcp-server-directory>
docker-compose logs -f
```

---

## Quick Reference

| Server | Port | Health Check | Repository |
|--------|------|--------------|------------|
| Weather MCP | 8080 | `http://localhost:8080/health` | [mcp-weather-server](https://github.com/kumaran-is/mcp-weather-server/tree/develop) |
| Hurricane Tracker | 8081 | `http://localhost:8081/health` | [hurricane-tracker-mcp](https://github.com/kumaran-is/hurricane-tracker-mcp) |
| Metrics (Optional) | 9090 | `http://localhost:9090/metrics` | Disabled by default - Uncomment in docker-compose.yml |

### Docker Compose Commands (Option 1)

```bash
# Start both servers
docker-compose up -d

# Stop both servers
docker-compose down

# View status
docker-compose ps

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up -d --build
```

---

## Next Steps

After both servers are running:
1. ✅ Verify both containers are up: `docker-compose ps`
2. ✅ Check project `.env` has correct URLs
3. ✅ Run Level 0 verification script
4. ✅ Verify all 8 checks pass
5. ✅ Proceed to Level 1 implementation
