#!/bin/bash
#
# Quick setup script for Jetson Assistant
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Jetson Assistant Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Install
echo "Installing jetson-assistant..."
pip install -e .

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Install backends:"
echo "  pip install -e '.[qwen]'    # TTS with Qwen"
echo "  pip install -e '.[whisper]' # STT with Whisper"
echo "  pip install -e '.[all]'     # Everything"
echo ""
echo "Quick test:"
echo "  jetson-assistant info"
