# Voice Assistant Guide

Build your own Alexa-like voice assistant using Jetson Assistant.

## Overview

The voice assistant combines:
- **Wake Word Detection** - "Hey Jarvis" (or custom)
- **Speech-to-Text** - Whisper for transcription
- **LLM** - Local (Ollama) or cloud (OpenAI, Claude)
- **Text-to-Speech** - Qwen or Piper for responses

## Quick Start

### 1. Install Dependencies

```bash
pip install jetson-assistant[assistant,qwen,whisper]
```

### 2. Install Ollama (Local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2:3b
```

### 3. Run the Assistant

```bash
jetson-assistant assistant --wake-word hey_jarvis --llm-model llama3.2:3b
```

## CLI Options

```bash
jetson-assistant assistant [OPTIONS]

Options:
  -w, --wake-word TEXT     Wake word (default: hey_jarvis)
  --tts TEXT              TTS backend (default: piper)
  --stt TEXT              STT backend (default: whisper)
  --stt-model TEXT        STT model size (default: base)
  -l, --llm TEXT          LLM backend: ollama, openai, anthropic, simple
  -m, --llm-model TEXT    LLM model name (default: llama3.2:3b)
  -v, --voice TEXT        TTS voice (default: en_US-amy-medium)
  --verbose               Show timing info
  --no-wake               Skip wake word (always listening)
```

## Examples

### Basic Usage

```bash
# Default settings
jetson-assistant assistant

# Custom wake word and voice
jetson-assistant assistant --wake-word alexa --voice ryan

# Without wake word (push-to-talk style)
jetson-assistant assistant --no-wake
```

### With Cloud LLM

```bash
# OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
jetson-assistant assistant --llm openai --llm-model gpt-4o-mini

# Anthropic (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
jetson-assistant assistant --llm anthropic --llm-model claude-haiku-4-5-20251001
```

### Testing Without LLM

```bash
# Simple rule-based responses (no LLM needed)
jetson-assistant assistant --llm simple
```

## Python API

```python
from jetson_assistant import Engine
from jetson_assistant.assistant import VoiceAssistant, AssistantConfig

# Initialize engine
engine = Engine()
engine.load_tts_backend("qwen")
engine.load_stt_backend("whisper")

# Configure assistant
config = AssistantConfig(
    wake_word="hey_jarvis",
    llm_backend="ollama",
    llm_model="llama3.2:3b",
    tts_voice="en_US-amy-medium",
    verbose=True,
)

# Create and run
assistant = VoiceAssistant(engine, config)
assistant.run()  # Blocking
```

### Background Mode

```python
# Run in background thread
thread = assistant.run_async()

# Do other things...
time.sleep(60)

# Stop
assistant.stop()
thread.join()
```

### Interactive Methods

```python
# Make the assistant say something
assistant.say("Hello! I'm your voice assistant.")

# Listen for speech (without wake word)
text = assistant.listen_once(timeout=10.0)
print(f"You said: {text}")

# Ask a question and get response
answer = assistant.ask("What's your name?")
print(f"Answer: {answer}")
```

### Custom Callbacks

```python
config = AssistantConfig(
    on_wake=lambda: print("Wake word detected!"),
    on_listen_start=lambda: print("Listening..."),
    on_listen_end=lambda text: print(f"You said: {text}"),
    on_response=lambda text: print(f"Response: {text}"),
    on_error=lambda e: print(f"Error: {e}"),
)
```

## Wake Word Options

### OpenWakeWord (Default, Open Source)

```python
from jetson_assistant.assistant.wakeword import OpenWakeWordDetector

# Built-in wake words
detector = OpenWakeWordDetector(wake_word="hey_jarvis")  # or "alexa", "hey_mycroft"

# Custom model
detector = OpenWakeWordDetector(model_path="/path/to/custom.onnx")
```

### Porcupine (Commercial, More Accurate)

Requires free API key from [Picovoice Console](https://console.picovoice.ai/).

```bash
export PICOVOICE_ACCESS_KEY=...
pip install pvporcupine
```

```python
from jetson_assistant.assistant.wakeword import PorcupineDetector

detector = PorcupineDetector(wake_word="jarvis")
```

### Energy-Based (No Wake Word)

```python
from jetson_assistant.assistant.wakeword import SimpleEnergyDetector

# Triggers on any speech
detector = SimpleEnergyDetector(threshold=500.0)
```

## LLM Options

### Ollama (Local, Recommended)

```bash
ollama pull llama3.2:3b   # Fast, good quality
ollama pull phi3:mini     # Smaller, faster
ollama pull mistral       # Good alternative
```

### OpenAI

```python
from jetson_assistant.assistant.llm import OpenAILLM

llm = OpenAILLM(
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant. Keep responses brief."
)
```

### Anthropic (Claude)

```python
from jetson_assistant.assistant.llm import AnthropicLLM

llm = AnthropicLLM(
    model="claude-haiku-4-5-20251001",
    system_prompt="You are a helpful assistant."
)
```

### Custom System Prompt

```python
config = AssistantConfig(
    system_prompt="""You are JARVIS, Tony Stark's AI assistant.
    You are witty, intelligent, and slightly sarcastic.
    Keep responses concise (1-2 sentences)."""
)
```

## Hardware Recommendations

| Device | LLM Model | TTS | STT | Experience |
|--------|-----------|-----|-----|------------|
| Orin Nano 4GB | phi3:mini | Piper | Whisper tiny | Basic |
| Orin Nano 8GB | llama3.2:3b | Qwen 0.6B | Whisper base | Good |
| Orin NX | llama3.2:3b | Qwen 0.6B | Whisper small | Great |
| AGX Orin | llama3.1:8b | Qwen 1.7B | Whisper medium | Excellent |

## Troubleshooting

### "No audio input device"

```bash
# List audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test microphone
arecord -d 3 test.wav
aplay test.wav
```

### "Ollama connection refused"

```bash
# Start Ollama server
ollama serve

# Check if running
curl http://localhost:11434/api/version
```

### Wake word not detecting

- Speak clearly and at normal volume
- Try lowering the threshold:
  ```python
  config = AssistantConfig(wake_word_threshold=0.3)
  ```
- Try the energy detector for testing:
  ```bash
  jetson-assistant assistant --no-wake
  ```

### Slow responses

- Use smaller models:
  - LLM: `phi3:mini` instead of `llama3.2:3b`
  - STT: `whisper tiny` instead of `base`
  - TTS: `piper` instead of `qwen`

### High latency

Enable verbose mode to identify bottleneck:
```bash
jetson-assistant assistant --verbose
```

Output shows timing for each stage:
```
You: What's the weather like?
  [STT: 450ms]
Assistant: I'm sorry, I don't have access to weather data.
  [LLM: 320ms, 45 tokens]
  [TTS: 680ms]
```
