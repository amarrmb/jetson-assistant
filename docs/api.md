# API Documentation

## REST API

### General Endpoints

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "is_jetson": false,
  "power_mode": null,
  "tts_backend": "qwen",
  "stt_backend": null
}
```

#### GET /info

Get detailed server information.

**Response:**
```json
{
  "is_jetson": false,
  "power_mode": null,
  "tts": {
    "name": "qwen",
    "loaded": true,
    "supports_streaming": true
  },
  "stt": {
    "loaded": false
  }
}
```

---

### TTS Endpoints

#### GET /tts/backends

List available TTS backends.

**Response:**
```json
{
  "backends": [
    {
      "name": "qwen",
      "loaded": true,
      "supports_streaming": true,
      "supports_voice_cloning": true
    },
    {
      "name": "piper",
      "loaded": false,
      "supports_streaming": true,
      "supports_voice_cloning": false
    }
  ]
}
```

#### POST /tts/backends/{name}/load

Load a TTS backend.

**Request:**
```json
{
  "model_size": "0.6B",
  "device": "cuda"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Loaded TTS backend: qwen",
  "backend": {
    "name": "qwen",
    "loaded": true
  }
}
```

#### GET /tts/voices

Get available voices for current backend.

**Response:**
```json
{
  "voices": [
    {
      "id": "ryan",
      "name": "Ryan",
      "language": "multilingual",
      "gender": "male",
      "description": "Neutral (default)"
    },
    {
      "id": "serena",
      "name": "Serena",
      "language": "multilingual",
      "gender": "female",
      "description": "Warm"
    }
  ],
  "languages": ["English", "Chinese", "Japanese"]
}
```

#### POST /tts/synthesize

Synthesize speech from text. Returns WAV audio.

**Request:**
```json
{
  "text": "Hello world",
  "voice": "ryan",
  "language": "English",
  "temperature": 1.0
}
```

**Response Headers:**
```
Content-Type: audio/wav
X-Duration: 1.5
X-Sample-Rate: 24000
X-Voice: ryan
```

**Response Body:** WAV audio binary data

#### POST /tts/synthesize/json

Synthesize speech and return base64 audio.

**Request:**
```json
{
  "text": "Hello world",
  "voice": "ryan"
}
```

**Response:**
```json
{
  "success": true,
  "duration": 1.5,
  "sample_rate": 24000,
  "voice": "ryan",
  "audio_base64": "UklGRi..."
}
```

---

### STT Endpoints

#### GET /stt/backends

List available STT backends.

**Response:**
```json
{
  "backends": [
    {
      "name": "whisper",
      "loaded": false,
      "supports_streaming": false
    }
  ]
}
```

#### POST /stt/backends/{name}/load

Load an STT backend.

**Request:**
```json
{
  "model_size": "base",
  "device": "cuda"
}
```

#### GET /stt/languages

Get supported languages.

**Response:**
```json
{
  "languages": ["en", "zh", "ja", "ko", "de", "fr", "es"]
}
```

#### POST /stt/transcribe

Transcribe audio file.

**Request:** `multipart/form-data`
- `audio`: Audio file (WAV, MP3, etc.)
- `language`: Language code (optional)

**Response:**
```json
{
  "success": true,
  "text": "Hello world",
  "language": "en",
  "duration": 1.5,
  "segments": [
    {
      "text": "Hello world",
      "start": 0.0,
      "end": 1.5,
      "confidence": 0.95
    }
  ]
}
```

---

## WebSocket API

### TTS Streaming

**Endpoint:** `ws://localhost:8080/tts/stream`

**Client → Server:**
```json
{
  "text": "Hello. This is streaming.",
  "voice": "ryan",
  "language": "English"
}
```

**Server → Client (start):**
```json
{
  "type": "start",
  "chunks": 2
}
```

**Server → Client (audio):**
```json
{
  "type": "audio",
  "chunk": 1,
  "data": "UklGRi...",
  "duration": 0.8
}
```

**Server → Client (done):**
```json
{
  "type": "done",
  "total_time": 2.5
}
```

**Server → Client (error):**
```json
{
  "type": "error",
  "error": "Model not loaded"
}
```

### STT Streaming

**Endpoint:** `ws://localhost:8080/stt/stream`

**Client → Server (start):**
```json
{
  "type": "start",
  "language": "en",
  "sample_rate": 16000
}
```

**Server → Client (ready):**
```json
{
  "type": "ready"
}
```

**Client → Server:** Binary audio chunks (16-bit PCM)

**Server → Client (segment):**
```json
{
  "type": "segment",
  "text": "Hello",
  "is_final": false
}
```

**Client → Server (stop):**
```json
{
  "type": "stop"
}
```

**Server → Client (done):**
```json
{
  "type": "done",
  "text": "Hello world"
}
```

---

## Python API

### Basic Usage

```python
from jetson_assistant import Engine

# Create engine
engine = Engine()

# Load TTS backend
engine.load_tts_backend("qwen", model_size="0.6B")

# Synthesize
result = engine.synthesize("Hello world", voice="serena")
result.save("output.wav")

# Or play directly
engine.say("Hello world")
```

### STT Usage

```python
# Load STT backend
engine.load_stt_backend("whisper", model_size="base")

# Transcribe file
result = engine.transcribe("audio.wav")
print(result.text)

# Transcribe numpy array
import numpy as np
audio = np.zeros(16000, dtype=np.int16)
result = engine.transcribe(audio, sample_rate=16000)
```

### Streaming TTS

```python
# Stream synthesis
for chunk in engine.synthesize_stream("Long text here"):
    # Process each chunk
    print(f"Chunk: {chunk.duration}s")

# Stream and play
engine.say("Long text", stream=True)
```

### File Processing

```python
# Convert document to audio
result = engine.synthesize_file(
    "document.pdf",
    output="document.wav",
    voice="ryan"
)
```
