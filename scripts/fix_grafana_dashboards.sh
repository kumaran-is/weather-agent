#!/bin/bash
# Fix Grafana Dashboard Metric Names
# This script updates dashboard JSON files to use the metrics that are actually exported

set -e

echo "=== Grafana Dashboard Metric Fix Script ==="
echo ""

DASHBOARD_DIR="observability/grafana/provisioning/dashboards"
BACKUP_DIR="observability/grafana/provisioning/dashboards/backup_$(date +%Y%m%d_%H%M%S)"

# Create backup
echo "📦 Creating backup at: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp "$DASHBOARD_DIR"/*.json "$BACKUP_DIR/"
echo "✅ Backup complete"
echo ""

# Fix 1: Signal Correlation Dashboard
echo "🔧 Fixing Signal Correlation Dashboard..."
SIGNAL_DASH="$DASHBOARD_DIR/signal-correlation.json"

# Replace request_latency with request_duration
sed -i.tmp 's/weather_ai_request_latency_seconds_bucket/weather_ai_request_duration_seconds_bucket/g' "$SIGNAL_DASH"

# Comment out missing metrics (LLM, MCP, Cache latency)
# For now, we'll leave them as-is and they'll show "No data" until implemented
echo "  ✅ Fixed: weather_ai_request_latency → weather_ai_request_duration"
echo "  ⚠️  Note: LLM, MCP, Cache latency panels will show 'No data' until metrics implemented"
echo ""

# Fix 2: Comprehensive Observability Dashboard
echo "🔧 Fixing Comprehensive Observability Dashboard..."
COMP_DASH="$DASHBOARD_DIR/observability-comprehensive.json"

# Replace request_latency with request_duration
sed -i.tmp 's/weather_ai_request_latency_seconds_bucket/weather_ai_request_duration_seconds_bucket/g' "$COMP_DASH"
echo "  ✅ Fixed: weather_ai_request_latency → weather_ai_request_duration"
echo "  ⚠️  Note: Missing metrics (LLM cost, tokens, guardrails) will show 'No data'"
echo ""

# Fix 3: Cache Metrics Dashboard
echo "🔧 Fixing Cache Metrics Dashboard..."
CACHE_DASH="$DASHBOARD_DIR/cache-metrics.json"

# Replace cache_hits_total with weather_ai_cache_hits_total
sed -i.tmp 's/\bcache_hits_total\b/weather_ai_cache_hits_total/g' "$CACHE_DASH"
sed -i.tmp 's/\bcache_misses_total\b/weather_ai_cache_misses_total/g' "$CACHE_DASH"

# Note: cache_requests_total doesn't exist - need to calculate from hits + misses
# Replace cache_requests_total with (cache_hits_total + cache_misses_total)
sed -i.tmp 's/cache_requests_total/(weather_ai_cache_hits_total + weather_ai_cache_misses_total)/g' "$CACHE_DASH"
echo "  ✅ Fixed: cache_hits_total → weather_ai_cache_hits_total"
echo "  ✅ Fixed: cache_misses_total → weather_ai_cache_misses_total"
echo "  ✅ Fixed: cache_requests_total → (hits + misses)"
echo "  ⚠️  Note: L3 Anthropic cache metrics not yet implemented"
echo ""

# Fix 4: Agent Performance Dashboard
echo "🔧 Fixing Agent Performance Dashboard..."
AGENT_DASH="$DASHBOARD_DIR/agent-performance.json"

# Replace agent_queries_total with weather_ai_requests_total
sed -i.tmp 's/agent_queries_total/weather_ai_requests_total/g' "$AGENT_DASH"
sed -i.tmp 's/agent_query_duration_seconds_bucket/weather_ai_request_duration_seconds_bucket/g' "$AGENT_DASH"
echo "  ✅ Fixed: agent_queries_total → weather_ai_requests_total"
echo "  ✅ Fixed: agent_query_duration → weather_ai_request_duration"
echo "  ⚠️  Note: Missing metrics (cost, success rate, tool calls, guardrails) will show 'No data'"
echo ""

# Clean up tmp files
rm -f "$DASHBOARD_DIR"/*.tmp

echo "=== Summary ==="
echo "✅ Fixed Signal Correlation: request_latency → request_duration"
echo "✅ Fixed Comprehensive Observability: request_latency → request_duration"
echo "✅ Fixed Cache Metrics: cache_* → weather_ai_cache_*"
echo "✅ Fixed Agent Performance: agent_* → weather_ai_*"
echo ""
echo "⚠️  Panels requiring unimplemented metrics will show 'No data':"
echo "   - LLM latency, tokens, cost"
echo "   - MCP latency"
echo "   - Cache latency"
echo "   - L3 Anthropic cache"
echo "   - Guardrail violations"
echo "   - Tool/agent invocation tracking"
echo ""
echo "📦 Backup saved to: $BACKUP_DIR"
echo ""
echo "🔄 Next: Restart Grafana to reload dashboards"
echo "   docker-compose restart grafana"
echo ""
