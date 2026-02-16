#!/usr/bin/env bash
# Benchmark regression gate — fails if TTFB exceeds tier target.
#
# Usage: ./scripts/benchmark_regression.sh configs/thor.yaml 700
#   $1 = config path
#   $2 = max TTFB in ms (700 for thor, 1000 for orin, 1500 for nano)

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> <max_ttfb_ms>}"
MAX_TTFB="${2:?Usage: $0 <config.yaml> <max_ttfb_ms>}"

echo "Running pipeline benchmark: $CONFIG (target: <${MAX_TTFB}ms TTFB)"

RESULT=$(python scripts/benchmark_pipeline.py --config "$CONFIG" --runs 5 --json)
TTFB=$(echo "$RESULT" | python3 -c "import sys,json; print(int(json.load(sys.stdin)['ttfb_ms']))")

echo "Measured TTFB: ${TTFB}ms (target: <${MAX_TTFB}ms)"

if [ "$TTFB" -gt "$MAX_TTFB" ]; then
    echo "FAIL: TTFB ${TTFB}ms exceeds target ${MAX_TTFB}ms"
    exit 1
fi

echo "PASS"
