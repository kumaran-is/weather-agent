# Observability Stack Setup Guide

Prometheus + Grafana + Loki Integration

**Estimated Time**: 15 minutes

---

## Quick Start

### 1. Start Observability Stack

```bash
# Production mode (with observability)
make docker-up

# Development mode (with observability)
make docker-up-dev

# Check status
make observability-status
```

### 2. Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3001 | admin / weatherai2025 |
| **Prometheus** | http://localhost:9090 | None |
| **Loki** | http://localhost:3100 | None (API only) |
| **LangSmith** | https://smith.langchain.com | Your API key |

**Note**:
- Grafana credentials can be customized via `GRAFANA_ADMIN_PASSWORD` environment variable
- Internal container URLs (from Docker network): `http://grafana:3000`, `http://prometheus:9090`, `http://loki:3100`
- External URLs (from host machine): Use `localhost` with mapped ports above

### 3. Verify Stack Health

```bash
# Check all services
make docker-health

# View observability logs
make observability-logs
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY STACK                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌────────────────┐    ┌────────────────┐                  │
│   │   Grafana      │◄───│   Prometheus   │                  │
│   │   :3001        │    │   :9090        │                  │
│   │   (Dashboards) │    │   (Metrics)    │                  │
│   └───────┬────────┘    └───────┬────────┘                  │
│           │                     │                            │
│           │    ┌────────────────┘                            │
│           │    │                                             │
│           ▼    ▼                                             │
│   ┌────────────────┐                                         │
│   │     Loki       │                                         │
│   │    :3100       │                                         │
│   │    (Logs)      │                                         │
│   └────────────────┘                                         │
│                                                              │
│   + LangSmith (Cloud) for AI/LLM tracing                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Setup

### Prometheus Configuration

**Location**: `observability/prometheus/prometheus.yml`

**Scrape Targets**:
- Weather AI API (`:8000/metrics`)
- Prometheus self (`:9090`)
- Grafana (`:3000`)
- Loki (`:3100`)

**Retention**: 15 days (production), 7 days (development)

### Grafana Configuration

**Location**: `observability/grafana/provisioning/`

**Pre-configured Dashboards**:
1. **MCP Health** - MCP server latency, errors, data freshness
2. **Agent Performance** - Query volume, latency, token usage
3. **Cache Metrics** - L1/L2/L3 hit rates, cost savings

**Data Sources** (auto-provisioned):
- Prometheus (default)
- Loki

### Loki Configuration

**Location**: `observability/loki/loki-config.yml`

**Storage**: Local filesystem (Docker volume)
**Retention**: 7 days

---

## Common Operations

### View Metrics in Prometheus

1. Open http://localhost:9090
2. Enter query in expression box:
   ```promql
   # Query latency P95
   histogram_quantile(0.95, rate(weather_query_latency_seconds_bucket[5m]))

   # Cache hit rate
   sum(cache_hits_total) / sum(cache_requests_total)

   # MCP error rate
   rate(mcp_requests_total{status="error"}[5m])
   ```
3. Click "Execute"

### Query Logs in Loki (via Grafana)

1. Open http://localhost:3001
2. Navigate to Explore (compass icon)
3. Select "Loki" as data source
4. Enter LogQL query:
   ```logql
   # All weather API logs
   {job="weather-ai-api"}

   # Errors only
   {job="weather-ai-api"} |= "error"

   # Hurricane queries
   {job="weather-ai-api"} |= "hurricane"

   # Slow queries (>5s)
   {job="weather-ai-api"} | json | latency_ms > 5000
   ```

### View Dashboards in Grafana

1. Open http://localhost:3001
2. Login: admin / admin
3. Navigate to Dashboards > Weather AI
4. Select dashboard:
   - MCP Health
   - Agent Performance
   - Cache Metrics

---

## Alert Configuration

### Current Alert Rules

| Alert | Severity | Trigger |
|-------|----------|---------|
| APIDown | critical | API health check fails for 1m |
| MCPUnhealthy | critical | MCP server down for 1m |
| HighLatency | warning | P95 > 2s for 5m |
| HighErrorRate | warning | Error rate > 5% for 5m |
| PIILeakDetected | critical | Any PII guardrail violation |
| HurricaneValidationFailed | critical | Saffir-Simpson validation error |
| CachePerformanceDegraded | warning | Hit rate < 50% for 10m |

### Configure Alert Notifications

1. Open Grafana > Alerting > Contact points
2. Add notification channel:
   - **Slack**: Add webhook URL
   - **PagerDuty**: Add integration key
   - **Email**: Configure SMTP

---

## Troubleshooting

### Prometheus Not Scraping

**Symptoms**: Empty metrics, "target down"

**Check**:
```bash
# Verify targets
curl http://localhost:9090/api/v1/targets

# Check API metrics endpoint
curl http://localhost:8000/metrics
```

**Fix**: Ensure `/metrics` endpoint is exposed

### Grafana Dashboard Empty

**Symptoms**: "No data" in panels

**Check**:
```bash
# Verify Prometheus connection
curl http://localhost:9090/api/v1/query?query=up

# Verify data source in Grafana
# Settings > Data Sources > Prometheus > Test
```

**Fix**: Check data source URL (`http://prometheus:9090` inside Docker)

### Loki Not Receiving Logs

**Symptoms**: Empty log explorer

**Check**:
```bash
# Verify Loki is running
curl http://localhost:3100/ready

# Check Docker log driver
docker inspect weather-ai-api | grep -A5 LogConfig
```

**Fix**: Ensure Docker log driver is configured (or use default)

### High Memory Usage

**Symptoms**: Prometheus/Loki consuming too much RAM

**Fix**: Adjust retention settings
```yaml
# prometheus.yml
storage.tsdb.retention.time: 7d  # Reduce from 15d
storage.tsdb.retention.size: 2GB

# loki-config.yml
limits_config:
  retention_period: 72h  # Reduce from 168h
```

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make observability-status` | Show status of observability stack |
| `make grafana-open` | Open Grafana in browser |
| `make prometheus-open` | Open Prometheus in browser |
| `make prometheus-reload` | Reload Prometheus configuration |
| `make observability-logs` | View logs from observability stack |
| `make loki-logs` | Query recent logs from Loki |

---

