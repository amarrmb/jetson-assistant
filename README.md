# Jetson Assistant

Modular TTS (Text-to-Speech) + STT (Speech-to-Text) server for Jetson and edge devices.

## Features

- **Voice Assistant** - Build your own Alexa/Google Home with wake word, STT, LLM, and TTS
- **Multi-Camera Monitoring** - Register RTSP/USB cameras by name, ask about any camera by voice, get continuous VLM-powered alerts
- **Knowledge Base** - Personal info lookup via RAG (birthdays, contacts, preferences) — the LLM decides when to search
- **Aether Hub Integration** - Push camera alerts to mobile/web clients via local WebSocket hub
- **Multi-backend support** - Swap between Qwen, Piper, Whisper, and more
- **Streaming API** - WebSocket streaming for real-time playback
- **REST API** - FastAPI-based HTTP endpoints
- **Benchmarking** - Compare backends on latency, quality, memory
- **Jetson-optimized** - Power mode detection, memory management
- **Open source** - Apache 2.0 license

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
| SenseVoice | Alibaba's multilingual STT | Yes | - |

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/amarrmb/jetson-assistant.git
cd jetson-assistant

# Install with pip (basic)
pip install -e .

# Install with specific backends
pip install -e ".[kokoro]"          # Kokoro TTS (best quality)
pip install -e ".[qwen]"           # Qwen TTS
pip install -e ".[whisper]"        # Whisper STT
pip install -e ".[nemotron]"       # Nemotron STT (fastest)
pip install -e ".[vision]"         # Camera support (OpenCV)
pip install -e ".[aether]"         # Aether Hub integration
pip install -e ".[qwen,whisper]"   # Multiple backends
pip install -e ".[all]"            # Everything
```

### Jetson Setup

```bash
# Run the Jetson-specific installer
./scripts/install_jetson.sh
```

## Usage

### CLI

```bash
# Text-to-Speech
jetson-assistant tts "Hello world"
jetson-assistant tts "Hello" --backend qwen --voice serena
jetson-assistant tts -f document.pdf -o output.wav

# Speech-to-Text
jetson-assistant stt audio.wav
jetson-assistant stt audio.wav --backend whisper --model base

# Start the server
jetson-assistant serve --port 8080

# Run benchmarks
jetson-assistant benchmark tts --text "Hello world" --backends qwen,piper
```

### Voice Assistant

Build your own Alexa-like voice assistant:

```bash
# Install assistant dependencies
pip install -e ".[assistant,qwen,whisper]"

# Install Ollama (local LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# Run the assistant
jetson-assistant assistant --wake-word hey_jarvis
```

Options:
```bash
# Custom wake word and voice
jetson-assistant assistant --wake-word alexa --voice ryan

# Use OpenAI instead of local LLM
export OPENAI_API_KEY=sk-...
jetson-assistant assistant --llm openai --llm-model gpt-4o-mini

# Lightweight mode for Orin Nano 4GB
jetson-assistant assistant --tts piper --stt-model tiny --llm-model phi3:mini
```

#### SOTA Demo (Kokoro + Nemotron + vLLM VLM)

The fastest pipeline: ~700ms end-to-end with near-human voice quality. Only 2 terminals needed.

```bash
# Prerequisites
apt-get install espeak-ng
pip install -e ".[kokoro,nemotron,assistant,vision]"

# Terminal 1: Start vLLM VLM container
docker run -d --rm --runtime=nvidia --network host --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name vllm ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
    vllm serve nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 \
    --host 0.0.0.0 --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.3

# Terminal 2: Run SOTA assistant
./scripts/run-sota-demo.sh
```

See [docs/assistant.md](docs/assistant.md) for full documentation.

#### Multi-Camera Monitoring

Register RTSP security cameras and USB cameras by name, then query or monitor them by voice.

**1. Configure cameras** in `~/.assistant_cameras.json` (via web UI, mobile app, or manual edit):

```json
[
  {"name": "garage", "url": "rtsp://192.0.2.50:554/stream1", "location": "Garage ceiling"},
  {"name": "front_door", "url": "rtsp://192.0.2.51:554/stream1", "location": "Front porch"},
  {"name": "baby_room", "url": "rtsp://192.0.2.52:554/stream1", "location": "Nursery"},
  {"name": "desk", "url": "usb:2", "location": "Office USB cam"}
]
```

**2. Launch with vision + cameras:**

```bash
pip install -e ".[kokoro,nemotron,assistant,vision]"

jetson-assistant assistant \
    --tts kokoro --stt nemotron \
    --llm vllm --llm-host http://localhost:8001/v1 \
    --vision --no-wake \
    --camera-config ~/.assistant_cameras.json
```

**3. Talk to it:**

| Voice Command | What Happens |
|---|---|
| *"What cameras do I have?"* | Lists all cameras with locations |
| *"What's happening at the front door?"* | Captures RTSP frame, VLM describes the scene |
| *"Is the baby sleeping?"* | Captures baby_room frame, VLM answers the question |
| *"Watch the garage for the door opening"* | Starts background VLM polling every 10s |
| *"What are you watching?"* | Lists all active monitors |
| *"Stop watching the garage"* | Stops that monitor |
| *"Add a camera called patio at rtsp://..."* | Registers a new camera by voice |

Watches are continuous with cooldown — after detection, the assistant alerts you, waits 60s, then resumes watching. Confidence voting (2/3 positive VLM checks) prevents false alerts.

**Camera monitoring flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--camera-config` | `~/.assistant_cameras.json` | Camera config file path |
| `--watch-interval` | `5.0` | Seconds between VLM polls |
| `--watch-cooldown` | `60.0` | Seconds between repeated alerts |

#### Knowledge Base (Personal Info Lookup)

Store personal or domain-specific information in a local ChromaDB collection. The LLM automatically searches it when relevant.

```bash
# Install RAG dependencies
pip install chromadb sentence-transformers

# Launch with knowledge base
jetson-assistant assistant \
    --tts kokoro --stt nemotron \
    --llm vllm --llm-host http://localhost:8001/v1 \
    --knowledge personal --no-wake
```

Then ask naturally: *"What's mom's birthday?"*, *"What's the wifi password?"*, *"When is the dentist appointment?"*

The LLM decides when to call `lookup_info` vs. answering from its own knowledge.

#### Aether Hub Integration

Push camera alerts to all connected clients (future mobile app, web console) via the local Aether Hub.

```bash
# Start Aether Hub (separate process)
./aether-hub -port 8000

# Launch assistant with Hub connection
jetson-assistant assistant \
    --tts kokoro --stt nemotron \
    --llm vllm --llm-host http://localhost:8001/v1 \
    --vision --no-wake \
    --aether-hub localhost:8000 --aether-pin 123456
```

When a watch detects something, the assistant both speaks the alert and publishes a `CAMERA_ALERT` message to the Hub. Requires `pip install -e ".[aether]"`.

### API Server

Start the server:
```bash
jetson-assistant serve --port 8080
```

#### TTS Endpoints

```bash
# List voices
curl http://localhost:8080/tts/voices

# Synthesize speech
curl -X POST http://localhost:8080/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "ryan"}' \
  --output hello.wav
```

#### STT Endpoints

```bash
# Transcribe audio
curl -X POST http://localhost:8080/stt/transcribe \
  -F "audio=@recording.wav"
```

#### WebSocket Streaming

```javascript
// TTS streaming
const ws = new WebSocket('ws://localhost:8080/tts/stream');
ws.send(JSON.stringify({text: "Hello world", voice: "ryan"}));
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'audio') {
    // Play base64 audio chunk
  }
};
```

### Python API

```python
from jetson_assistant import Engine
from jetson_assistant.tts import QwenBackend

# Initialize engine
engine = Engine()
engine.load_tts_backend(QwenBackend())

# Generate speech
result = engine.synthesize("Hello world", voice="serena")
result.save("output.wav")

# Transcribe audio
text = engine.transcribe("recording.wav")
print(text)
```

## Configuration

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
│   │   ├── aether_bridge.py # Aether Hub alert publishing
│   │   ├── vision.py       # Camera, VisionMonitor, MJPEG preview
│   │   ├── llm.py          # LLM backends (ollama, vllm, openai, anthropic)
│   │   └── tools.py        # ToolRegistry (auto-schema from type hints)
│   ├── rag/            # RAG pipeline (ChromaDB + sentence-transformers)
│   └── benchmark/      # Benchmarking tools
├── webui/              # Gradio web interface
├── examples/           # Example scripts
├── scripts/            # Setup scripts
├── tests/              # Test suite
└── docs/               # Documentation
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run unit tests (no GPU, no hardware needed)
make test

# Run linter
make lint

# Validate docker-compose.yml (needs docker)
make validate

# Smoke test — verifies imports, CLI, config presets, compose
make smoke

# Full pre-push check (lint + test + validate)
make ci
```

Individual commands (if you prefer not to use `make`):

```bash
python -m pytest tests/ -v          # unit tests
ruff check jetson_assistant/ tests/    # lint
docker compose config -q            # compose validation
bash scripts/smoke-test.sh          # smoke test
```

## Benchmarking

```bash
# Compare TTS backends
jetson-assistant benchmark tts \
  --text "The quick brown fox jumps over the lazy dog." \
  --backends qwen,piper \
  --iterations 5

# Compare STT backends
jetson-assistant benchmark stt \
  --audio test.wav \
  --backends whisper \
  --models tiny,base,small
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Kokoro](https://github.com/hexgrad/kokoro) - Near-human quality TTS (82M params, 54+ voices)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen-TTS) - High-quality neural TTS
- [Piper](https://github.com/rhasspy/piper) - Fast local TTS
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - Nemotron Speech STT
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Optimized Whisper inference
