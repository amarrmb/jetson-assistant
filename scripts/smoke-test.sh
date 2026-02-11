#!/usr/bin/env bash
# Smoke test — verifies jetson-assistant installation on any device.
# No GPU containers, no mic/speaker required.
#
# Run after: git clone && pip install -e ".[dev]"
# Usage:    bash scripts/smoke-test.sh

set -uo pipefail

PASS=0
FAIL=0

check() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "  [PASS] $label"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $label"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== jetson-assistant smoke test ==="
echo

# 1. Core import
echo "Imports:"
check "AssistantConfig import" \
    python -c "from jetson_assistant.assistant.core import AssistantConfig; AssistantConfig()"

# 2. CLI entry points
echo "CLI entry points:"
check "jetson-assistant --help" \
    jetson-assistant --help
check "jetson-assistant assistant --help" \
    jetson-assistant assistant --help

# 3. Config presets load correctly
echo "Config presets:"
for preset in configs/*.yaml; do
    name=$(basename "$preset")
    check "Load $name" \
        python -c "
from jetson_assistant.assistant.core import AssistantConfig
data = AssistantConfig.from_yaml('$preset')
AssistantConfig(**data)
"
done

# 4. docker compose config (skip if docker not available)
echo "Docker:"
if command -v docker >/dev/null 2>&1; then
    check "docker compose config" \
        docker compose config -q
else
    echo "  [SKIP] docker not installed"
fi

# Summary
echo
echo "=== Results: $PASS passed, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
