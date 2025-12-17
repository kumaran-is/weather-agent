#!/bin/bash
# Generate Test Data for Grafana Dashboards
# This script generates diverse queries to populate metrics, traces, and logs

set -e

echo "=== Grafana Dashboard Test Data Generator ==="
echo ""
echo "Generating 25 diverse queries to populate:"
echo "  - Metrics (Prometheus)"
echo "  - Traces (Tempo)"
echo "  - Logs (Loki)"
echo ""

API_URL="http://localhost:8000/weather/query"
TOTAL_QUERIES=25
SUCCESS_COUNT=0
FAILURE_COUNT=0

# Query Categories
SIMPLE_QUERIES=(
  "What is the current weather in Miami?"
  "Weather forecast for Tampa"
  "Temperature in Orlando today"
  "Is it raining in Jacksonville?"
  "Weather conditions in Fort Lauderdale"
)

COMPLEX_QUERIES=(
  "Compare weather between Miami and Tampa for next 3 days"
  "What's the best time to travel to Orlando this week considering weather?"
  "Analyze precipitation patterns in South Florida"
  "7-day forecast for Jacksonville with temperature trends"
  "Weather-based activity recommendations for Fort Myers"
)

HURRICANE_QUERIES=(
  "Any active hurricane warnings in Florida?"
  "Hurricane alerts for Gulf Coast"
  "Storm tracking updates"
  "Severe weather warnings"
  "Tropical storm status"
)

CACHED_QUERIES=(
  "What is the current weather in Miami?"
  "Weather forecast for Tampa"
  "Any active hurricane warnings in Florida?"
)

# Function to send query
send_query() {
  local query="$1"
  local user_id="$2"
  local session_id="$3"
  local category="$4"

  echo "[$category] Query $SUCCESS_COUNT: $query"

  response=$(curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\", \"user_id\": \"$user_id\", \"session_id\": \"$session_id\"}" \
    2>&1)

  if [ $? -eq 0 ]; then
    echo "  ✅ Success"
    ((SUCCESS_COUNT++))
  else
    echo "  ❌ Failed: $response"
    ((FAILURE_COUNT++))
  fi

  sleep 2  # Rate limiting
}

# Generate simple queries (10 queries)
echo ""
echo "=== Phase 1: Simple Queries (10) ==="
for i in {0..4}; do
  query="${SIMPLE_QUERIES[$i]}"
  send_query "$query" "simple_user_$i" "simple_session_$i" "SIMPLE"
done

# Repeat 5 simple queries for variety
for i in {0..4}; do
  query="${SIMPLE_QUERIES[$i]}"
  send_query "$query" "simple_user_${i}_repeat" "simple_session_${i}_repeat" "SIMPLE"
done

# Generate complex queries (5 queries)
echo ""
echo "=== Phase 2: Complex Queries (5) ==="
for i in {0..4}; do
  query="${COMPLEX_QUERIES[$i]}"
  send_query "$query" "complex_user_$i" "complex_session_$i" "COMPLEX"
done

# Generate hurricane queries (5 queries)
echo ""
echo "=== Phase 3: Hurricane Queries (5) ==="
for i in {0..4}; do
  query="${HURRICANE_QUERIES[$i]}"
  send_query "$query" "hurricane_user_$i" "hurricane_session_$i" "HURRICANE"
done

# Generate cached queries (5 queries - repeats to trigger cache hits)
echo ""
echo "=== Phase 4: Cached Queries (5) ==="
for i in {0..2}; do
  query="${CACHED_QUERIES[$i]}"
  send_query "$query" "cached_user_$i" "cached_session_$i" "CACHED"
done

# Repeat first 2 cached queries again
for i in {0..1}; do
  query="${CACHED_QUERIES[$i]}"
  send_query "$query" "cached_user_${i}_final" "cached_session_${i}_final" "CACHED"
done

# Summary
echo ""
echo "=== Summary ==="
echo "Total Queries Sent: $((SUCCESS_COUNT + FAILURE_COUNT))"
echo "✅ Successful: $SUCCESS_COUNT"
echo "❌ Failed: $FAILURE_COUNT"
echo ""

# Verify data in Prometheus
echo "=== Verifying Metrics in Prometheus ==="
echo ""

echo "1. Total requests in last 5 minutes:"
docker exec weather-ai-prometheus wget -q -O- \
  'http://localhost:9090/api/v1/query?query=sum(rate(weather_ai_requests_total[5m]))' \
  2>/dev/null | jq -r '.data.result[0].value[1] // "No data"'

echo ""
echo "2. Requests by tier:"
docker exec weather-ai-prometheus wget -q -O- \
  'http://localhost:9090/api/v1/query?query=sum(rate(weather_ai_requests_total[5m]))+by+(tier)' \
  2>/dev/null | jq -r '.data.result[] | "  - \(.metric.tier): \(.value[1])"'

echo ""
echo "3. Cache hit rate:"
docker exec weather-ai-prometheus wget -q -O- \
  'http://localhost:9090/api/v1/query?query=sum(rate(weather_ai_cache_hits_total[5m]))+/+sum(rate((weather_ai_cache_hits_total+%2B+weather_ai_cache_misses_total)[5m]))' \
  2>/dev/null | jq -r '.data.result[0].value[1] // "No data"'

echo ""
echo "4. Context optimizations in last 5 minutes:"
docker exec weather-ai-prometheus wget -q -O- \
  'http://localhost:9090/api/v1/query?query=sum(increase(weather_agent_context_optimizations_total[5m]))' \
  2>/dev/null | jq -r '.data.result[0].value[1] // "No data"'

echo ""
echo "=== Next Steps ==="
echo ""
echo "1. Open Grafana: http://localhost:3001"
echo "2. Navigate to dashboards:"
echo "   - Signal Correlation"
echo "   - Agent Performance"
echo "   - Cache Metrics"
echo "   - Context Optimization"
echo "3. Refresh dashboards (Ctrl+R / Cmd+R)"
echo "4. Look for ⭐ exemplar stars on histogram panels"
echo "5. Test signal correlation paths:"
echo "   - Click ⭐ → Traces (Metrics → Traces)"
echo "   - Click 'Logs' tab in trace view (Traces → Logs)"
echo "   - Click 'Metrics' tab in trace view (Traces → Metrics)"
echo "   - Click trace_id links in logs (Logs → Traces)"
echo ""
echo "✅ Test data generation complete!"
