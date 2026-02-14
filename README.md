# Jetson Assistant

On-device voice + vision AI for NVIDIA Jetson. No cloud, no API keys, sub-second response.

<!-- TODO: replace with actual demo GIF recorded on Thor -->
<!-- ![Demo](docs/demo.gif) -->

Kokoro TTS + Nemotron STT + Qwen2.5-VL VLM — all running locally on Jetson Thor.
Ask it questions, point a camera, get spoken answers in ~700ms.

## Try It (Jetson Thor)

```bash
# SSH into your Jetson Thor
git clone https://github.com/amarrmb/jetson-assistant.git
cd jetson-assistant

# Reclaim GPU memory (recommended before first run)
sudo sysctl -w vm.drop_caches=3

# Start everything — vLLM loads the model (~5 min), then the assistant starts
docker compose up -d

# Watch it come up
docker compose logs -f assistant
```

Plug in a mic and speaker, and start talking. Say "What time is it?" or
"What do you see?" (if a USB camera is connected).

To stop: `docker compose down`

## Like It? Make It Yours

```bash
git clone https://github.com/amarrmb/jetson-assistant.git
cd jetson-assistant
pip install -e ".[kokoro,nemotron,assistant,vision]"

# Start vLLM separately
docker compose up -d vllm

# Run the assistant directly
./scripts/run-sota-demo.sh
```

From here you can swap backends, add custom tools, change voices, or wire it
into your own application. See [docs/](docs/) for the full reference.

## What's Inside

```
Mic → Nemotron STT (24ms) → Qwen2.5-VL-7B vLLM → Kokoro TTS (<300ms) → Speaker
                                    ↓ tool calls
                           built-in + your custom tools
```

- **Voice assistant** — wake word or always-listening, tool calling, streaming TTS
- **Multi-camera vision** — USB, RTSP, or phone camera via Aether WebRTC
- **Knowledge base** — personal info lookup via RAG (ChromaDB)
- **REST + WebSocket API** — integrate into your own apps
- **Benchmarking** — compare backends on latency, quality, memory

## Platforms

| Platform | Docker Tag | Status |
|----------|-----------|--------|
| Jetson Thor (JetPack 7+) | `:thor` | Tested |
| AGX Orin 64GB | — | Community welcome |
| Desktop GPU (RTX 3090+) | — | Community welcome |

### Adapting for Your Hardware

1. Change the PyTorch index URL in `Dockerfile` for your CUDA version
2. Rebuild flash-attn for your GPU arch (see [`wheels/README.md`](wheels/README.md))
3. Adjust `--gpu-memory-utilization` in `docker-compose.yml`
4. For smaller GPUs, swap to a smaller model in the compose command
5. Open a PR — we'll add your platform to this table

---

## Supported Backends

### TTS (Text-to-Speech)
| Backend | Description | GPU Required | Quality |
|---------|-------------|--------------|---------|
| **Kokoro** | Near-human quality, 82M params, 54+ voices, 24kHz | No (CPU ok, GPU faster) | Best |
| Qwen3-TTS | High-quality neural TTS with custom voices | Yes | Great |
| Piper | Lightweight, fast TTS | No (CPU-friendly) | Good |

### STT (Speech-to-Text)
| Backend | Description | GPU Required | Latency |
|---------|-------------|--------------|---------|
| **Nemotron** | NVIDIA NeMo 0.6B, fast English transcription | Yes | ~24ms |
| vLLM Whisper | GPU-accelerated Whisper via vLLM container | Yes | ~126ms |
| Whisper | OpenAI's speech recognition (via faster-whisper) | Optional | 700-900ms |
| SenseVoice | Alibaba's multilingual STT | Yes | — |

## Installation (Without Docker)

```bash
git clone https://github.com/amarrmb/jetson-assistant.git
cd jetson-assistant

# Install with specific backends
pip install -e ".[kokoro]"          # Kokoro TTS (best quality)
pip install -e ".[nemotron]"        # Nemotron STT (fastest)
pip install -e ".[whisper]"         # Whisper STT
pip install -e ".[assistant]"       # Voice assistant core
pip install -e ".[vision]"         # Camera support
pip install -e ".[all]"             # Everything
```

### Jetson Setup Script

```bash
./scripts/install_jetson.sh
```

## Usage

### Voice Assistant

```bash
# SOTA demo (Kokoro + Nemotron + vLLM)
./scripts/run-sota-demo.sh

# Custom config
jetson-assistant assistant --config configs/thor-sota.yaml

# Lightweight mode (Orin Nano 4GB)
jetson-assistant assistant --tts piper --stt-model tiny --llm-model phi3:mini
```

### CLI

```bash
jetson-assistant tts "Hello world"
jetson-assistant stt audio.wav
jetson-assistant serve --port 8080
jetson-assistant benchmark tts --backends kokoro,piper
```

### Python API

```python
from jetson_assistant import Engine

engine = Engine()
engine.load_tts_backend("kokoro")
result = engine.synthesize("Hello world", voice="af_heart")
result.save("output.wav")
```

### Multi-Camera Monitoring

Register cameras by name, query or monitor them by voice:

```bash
jetson-assistant assistant \
    --tts kokoro --stt nemotron \
    --llm vllm --llm-host http://localhost:8001/v1 \
    --vision --no-wake \
    --camera-config ~/.assistant_cameras.json
```

| Voice Command | What Happens |
|---|---|
| *"What cameras do I have?"* | Lists all cameras |
| *"What's happening at the front door?"* | VLM describes the scene |
| *"Watch the garage for the door opening"* | Background VLM polling |

## Configuration

See [`configs/`](configs/) for preset YAML files and [`docs/`](docs/) for full reference:

- [Voice Assistant](docs/assistant.md)
- [Backend Details](docs/backends.md)
- [API Reference](docs/api.md)
- [Jetson Hardware](docs/jetson.md)

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `JETSON_ASSISTANT_HOST` | Server host | `0.0.0.0` |
| `JETSON_ASSISTANT_PORT` | Server port | `8080` |
| `JETSON_ASSISTANT_TTS_BACKEND` | Default TTS backend | `qwen` |
| `JETSON_ASSISTANT_STT_BACKEND` | Default STT backend | `whisper` |
| `JETSON_ASSISTANT_MODEL_CACHE` | Model cache directory | `~/.cache/jetson-assistant` |

## Project Structure

```
jetson-assistant/
├── jetson_assistant/
│   ├── core/           # Text extraction, audio processing
│   ├── tts/            # TTS backends (kokoro, qwen, piper)
│   ├── stt/            # STT backends (nemotron, whisper, vllm)
│   ├── server/         # FastAPI server
│   ├── assistant/      # Voice assistant
│   │   ├── core.py         # Main loop, state machine, tool registration
│   │   ├── cameras.py      # CameraPool (RTSP + USB multi-camera)
│   │   ├── multi_watch.py  # MultiWatchMonitor (concurrent VLM watches)
│   │   ├── vision.py       # Camera, VisionMonitor, MJPEG preview
│   │   ├── llm.py          # LLM backends (ollama, vllm, openai, anthropic)
│   │   └── tools.py        # ToolRegistry (auto-schema from type hints)
│   ├── rag/            # RAG pipeline (ChromaDB + sentence-transformers)
│   └── benchmark/      # Benchmarking tools
├── webui/              # Gradio web interface
├── examples/           # Example scripts
├── configs/            # YAML presets
├── wheels/             # Pre-built Python wheels for Jetson
├── scripts/            # Setup and launcher scripts
└── tests/              # Test suite
```

## Development

```bash
pip install -e ".[dev]"
make test          # unit tests (no GPU needed)
make lint          # ruff linter
make smoke         # smoke test
make ci            # full pre-push check
```

## License

Apache 2.0 — See [LICENSE](LICENSE)

## Acknowledgments

- [Kokoro](https://github.com/hexgrad/kokoro) — near-human TTS
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — Nemotron STT
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — optimized Whisper
- [Qwen3-TTS](https://github.com/QwenLM/Qwen-TTS) — neural TTS
- [Piper](https://github.com/rhasspy/piper) — lightweight TTS
