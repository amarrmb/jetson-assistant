# Jetson Assistant

On-device voice + vision AI for NVIDIA Jetson. No cloud, no API keys, sub-second response.

Kokoro TTS + Nemotron STT + Qwen2.5-VL VLM — all running locally on Jetson Thor.
Ask it questions, point a camera, get spoken answers in ~700ms.

<!-- TODO: replace with actual demo -->
<!-- ![Demo](docs/demo.gif) -->

## Try It (Jetson Thor)

```bash
sudo sysctl -w vm.drop_caches=3        # reclaim GPU memory

# Download the compose file and start
curl -fLO https://raw.githubusercontent.com/amarrmb/jetson-assistant/main/docker-compose.yml
docker compose up -d                    # pulls ~12GB, vLLM loads model (~5 min)
docker compose logs -f assistant        # watch it come up
```

Plug in a mic and speaker. Say "What time is it?" or "What do you see?" (with a USB camera).

To stop: `docker compose down`

## Make It Yours

### Options

The assistant is configured via YAML files in `configs/`. Key settings:

| Setting | Flag / YAML key | Default | Description |
|---------|----------------|---------|-------------|
| TTS backend | `--tts` / `tts_backend` | `kokoro` | `kokoro`, `qwen`, `piper` |
| STT backend | `--stt` / `stt_backend` | `nemotron` | `nemotron`, `whisper`, `sensevoice` |
| LLM backend | `--llm` / `llm_backend` | `vllm` | `vllm`, `ollama`, `openai` |
| Vision | `--vision` | off | Enable USB camera |
| Wake word | `--no-wake` | on | Disable wake word (always listen) |
| Camera config | `--camera-config` | none | Multi-camera JSON file |

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `JETSON_ASSISTANT_HOST` | `0.0.0.0` | Server bind address |
| `JETSON_ASSISTANT_PORT` | `8080` | Server port |
| `JETSON_ASSISTANT_MODEL_CACHE` | `~/.cache/jetson-assistant` | Model cache directory |

### Build Locally

```bash
git clone https://github.com/amarrmb/jetson-assistant.git
cd jetson-assistant

# Pick what you need
pip install -e ".[kokoro]"              # Kokoro TTS
pip install -e ".[nemotron]"            # Nemotron STT
pip install -e ".[assistant]"           # Voice assistant core
pip install -e ".[vision]"             # Camera support
pip install -e ".[all]"                # Everything

# Start vLLM separately, then run
docker compose up -d vllm
./scripts/run-sota-demo.sh
```

Or on Jetson with the setup script: `./scripts/install_jetson.sh`

### Build Docker

```bash
# On Jetson Thor (aarch64 only)
docker build -t jetson-assistant:thor .
```

Adapting for other hardware:
1. Change the PyTorch index URL in `Dockerfile` for your CUDA version
2. Rebuild flash-attn for your GPU arch (see `wheels/README.md`)
3. Adjust `--gpu-memory-utilization` in `docker-compose.yml`

### Extend It

**Custom tools** — the assistant supports tool calling. Add tools via the plugin system:

```python
# my_tools.py
from typing import Annotated

def register_tools(registry, context=None):
    @registry.register("Get the current weather")
    def weather(city: Annotated[str, "City name"]) -> str:
        return f"72F and sunny in {city}"
```

Add to your config YAML:
```yaml
external_tools:
  - my_tools
```

**Python API:**

```python
from jetson_assistant import Engine

engine = Engine()
engine.load_tts_backend("kokoro")
result = engine.synthesize("Hello world", voice="af_heart")
result.save("output.wav")
```

**REST + WebSocket API** — run `jetson-assistant serve --port 8080` for HTTP integration.

**CLI:**

```bash
jetson-assistant tts "Hello world"
jetson-assistant stt audio.wav
jetson-assistant benchmark tts --backends kokoro,piper
```

See `docs/` for full reference: [assistant](docs/assistant.md), [backends](docs/backends.md), [API](docs/api.md), [hardware](docs/jetson.md).

## Development

```bash
pip install -e ".[dev]"
make test       # unit tests
make lint       # ruff
make smoke      # smoke test
make ci         # full pre-push check
```

## License

Apache 2.0 — See [LICENSE](LICENSE)

## Acknowledgments

- [Kokoro](https://github.com/hexgrad/kokoro) — near-human TTS
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — Nemotron STT
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — optimized Whisper
- [Qwen3-TTS](https://github.com/QwenLM/Qwen-TTS) — neural TTS
- [Piper](https://github.com/rhasspy/piper) — lightweight TTS

Built by [DeviceNexus.ai](https://devicenexus.ai).
