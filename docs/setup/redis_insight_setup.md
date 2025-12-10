# RedisInsight Desktop Client Setup

Quick guide to connect RedisInsight to the Weather AI Redis container.

**← Back to [Main README](../../README.md)**

---

## Table of Contents

1. [Install RedisInsight](#install-redisinsight)
2. [Connect to Weather AI Redis](#connect-to-weather-ai-redis)
3. [Verify Connection](#verify-connection)

---

## Install RedisInsight

### Download

| Platform | Download Link |
|----------|--------------|
| **macOS** | https://redis.io/insight/#insight-form |
| **Windows** | https://redis.io/insight/#insight-form |
| **Linux** | https://redis.io/insight/#insight-form |

### Install
1. Download RedisInsight from the link above
2. Open the installer
3. Follow installation prompts
4. Launch RedisInsight

---

## Connect to Weather AI Redis

### Prerequisites
- Weather AI services running: `docker-compose up -d`
- Redis container healthy: `docker ps | grep redis`

### Connection Steps

1. **Open RedisInsight** → Click "Add Redis Database"

2. **Enter Connection Details**:
   ```
   Host:      localhost
   Port:      6379
   Name:      Weather AI Redis
   Username:  (leave empty)
   Password:  (leave empty)
   ```

3. **Click "Add Redis Database"**

4. **Connection Details** (reference):
   | Field | Value | Source |
   |-------|-------|--------|
   | Host | `localhost` | Exposed from Docker |
   | Port | `6379` | docker-compose.yml:197 |
   | Image | `redis:8.4.0-alpine` | docker-compose.yml:195 |
   | Container | `weather-ai-redis` | docker-compose.yml:194 |

---

## Verify Connection

### Check Keys

1. Click on "Weather AI Redis" database
2. Navigate to **Browser** tab
3. You should see:
   - `context:*` - Session contexts (if memory tests ran)
   - `entities:*` - Entity tracking keys
   - `tool_store:*` - Semantic tool discovery cache

### Test Commands

Click **Workbench** → Run:

```redis
# Check Redis version
INFO server

# List all keys
KEYS *

# Check a session context (if exists)
GET context:test_user_memory:session_memory_001

# Check memory stats
INFO memory
```

### Expected Output

If memory tests ran, you should see:
```json
{
  "user_id": "test_user_memory",
  "session_id": "session_memory_001",
  "conversation_history": [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "content": "..."}
  ],
  "current_entities": {
    "location": "Tokyo"
  },
  "expires_at": "..."
}
```

---

## Troubleshooting

### "Could not connect to Redis"

**Check Docker container**:
```bash
docker ps | grep redis
```

**Should show**:
```
weather-ai-redis   redis:8.4.0-alpine   Up X minutes (healthy)   0.0.0.0:6379->6379/tcp
```

**Test connection**:
```bash
docker exec weather-ai-redis redis-cli ping
# Expected: PONG
```

**Restart services**:
```bash
docker-compose restart redis
```

### Port Already in Use

If port 6379 is taken by another Redis:
1. Stop other Redis instances
2. OR change port in `docker-compose.yml`:
   ```yaml
   ports:
     - "6380:6379"  # Use 6380 instead
   ```
3. Connect to `localhost:6380` in RedisInsight

---
