# Config Presets

Pre-built configurations for different hardware tiers and pipeline variants.

## Tier Configs (auto-detected)

These are selected automatically by `--auto` based on detected VRAM:

| Config | Hardware | VRAM | Pipeline | Target E2E |
|--------|----------|------|----------|------------|
| `thor.yaml` | Jetson Thor | 128GB | NVFP4 7B VLM + Kokoro + Nemotron | <700ms |
| `orin.yaml` | AGX Orin | 32-64GB | 3B VLM + Kokoro + Nemotron | <1s |
| `nano.yaml` | Orin Nano | 8GB | Ollama 1.5B + Piper + Whisper tiny | <1.5s |

## Variant Configs (manual selection)

| Config | Hardware | Pipeline | Latency | Notes |
|--------|----------|----------|---------|-------|
| `thor-sota.yaml` | Jetson Thor | NVFP4 7B + Kokoro + Nemotron | ~700ms | Same as `thor.yaml` |
| `thor-gpu.yaml` | Jetson Thor | NVFP4 7B + Piper + vLLM Whisper | ~890ms | All GPU, no in-process models |
| `orin-agx.yaml` | AGX Orin 64GB | BF16 7B + Kokoro + Nemotron | ~2-3s | 7B model (needs 64GB) |
| `desktop.yaml` | x86 + NVIDIA GPU | Ollama + Piper + Whisper CPU | varies | Development/testing |

## Usage

```bash
# Auto-detect hardware and use the right config
jetson-assistant assistant --auto

# Use a specific preset
jetson-assistant assistant --config configs/thor.yaml

# Override specific values from the preset
jetson-assistant assistant --config configs/thor.yaml --voice am_adam

# Multiple overrides
jetson-assistant assistant --config configs/thor.yaml --voice am_adam --no-stream
```

## How It Works

1. If `--config` is provided, the YAML file is loaded first
2. CLI arguments override any values from the YAML
3. Values not set in either use the built-in defaults

The YAML keys map directly to `AssistantConfig` field names (e.g., `tts_backend`, `llm_model`, `vision_enabled`).

## Creating Custom Configs

Copy any preset and modify it. Only include fields you want to change from defaults:

```yaml
# my-config.yaml
tts_backend: kokoro
tts_voice: am_adam
llm_backend: vllm
llm_host: "http://localhost:8001/v1"
llm_model: "nvidia/Qwen2.5-VL-7B-Instruct-NVFP4"
verbose: true
```

See `AssistantConfig` in `jetson_assistant/assistant/core.py` for all available fields.
