#!/bin/bash
#
# SOTA Voice Assistant Demo — "Visual Alexa" on Jetson Thor
#
# Only 2 terminals needed:
#   Terminal 1: vLLM VLM container (see below)
#   Terminal 2: This script (Kokoro TTS + Nemotron STT run in-process)
#
# Prerequisites:
#   1. Start vLLM VLM container on port 8001:
#      sudo sysctl -w vm.drop_caches=3
#      docker run -d --rm --runtime=nvidia --network host --ipc=host \
#          --ulimit memlock=-1 --ulimit stack=67108864 \
#          -v ~/.cache/huggingface:/root/.cache/huggingface \
#          --name vllm ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
#          vllm serve nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 \
#          --host 0.0.0.0 --port 8001 \
#          --max-model-len 4096 \
#          --gpu-memory-utilization 0.3
#
#   2. Install espeak-ng: apt-get install espeak-ng
#
#   3. Install backends: pip install -e ".[kokoro,nemotron,assistant,vision]"
#
# Architecture:
#   Mic → Nemotron STT (in-process, ~24ms)
#       → vLLM Qwen2.5-VL (container, ~643ms)
#       → Kokoro TTS (in-process, <300ms)
#       → Speaker
#
#   Total: ~700ms end-to-end (vs ~890ms with previous stack)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Ollama ships CUDA 12 libs that flash-attn needs
export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH}"

# Verify vLLM is running
echo "Checking vLLM VLM container on port 8001..."
if ! curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "ERROR: vLLM not running on port 8001."
    echo "Start it first:"
    echo "  sudo sysctl -w vm.drop_caches=3"
    echo "  docker run -d --rm --runtime=nvidia --network host --ipc=host \\"
    echo "      --ulimit memlock=-1 --ulimit stack=67108864 \\"
    echo "      -v ~/.cache/huggingface:/root/.cache/huggingface \\"
    echo "      --name vllm ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \\"
    echo "      vllm serve nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 \\"
    echo "      --host 0.0.0.0 --port 8001 \\"
    echo "      --max-model-len 4096 --gpu-memory-utilization 0.3"
    exit 1
fi
echo "vLLM OK"

# Run assistant with SOTA stack via config preset
# - TTS: Kokoro (in-process, near-human quality)
# - STT: Nemotron (in-process, ~24ms)
# - VLM: vLLM Qwen2.5-VL-7B-NVFP4 (container)
cd "$PROJECT_DIR"

exec jetson-assistant assistant \
    --config configs/thor-sota.yaml \
    "$@"
