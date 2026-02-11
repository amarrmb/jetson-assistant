#!/bin/bash
# Jetson Assistant Server startup script
# Sets up environment for flash-attn with CUDA 12 compatibility

# Add Ollama's CUDA 12 libs for flash-attn compatibility
export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH}"

# Suppress verbose warnings
export ORT_DISABLE_ALL=1  # Disable onnxruntime verbose output
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Run the server with all arguments passed through
exec jetson-assistant serve "$@"
