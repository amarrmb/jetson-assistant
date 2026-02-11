# Adding Custom Backends

This guide explains how to add new TTS or STT backends to Jetson Speech.

## TTS Backend

### Step 1: Create Backend File

Create a new file in `jetson_speech/tts/` (e.g., `mybackend.py`):

```python
from jetson_speech.tts.base import TTSBackend, SynthesisResult, Voice
from jetson_speech.tts.registry import register_tts_backend

@register_tts_backend("mybackend")
class MyBackend(TTSBackend):
    """My custom TTS backend."""

    name = "mybackend"
    supports_streaming = False
    supports_voice_cloning = False

    def __init__(self):
        super().__init__()
        # Initialize your state

    def load(self, **kwargs) -> None:
        """Load the model."""
        # Load your model here
        # Example:
        # self._model = load_my_model(kwargs.get("model_path"))
        self._loaded = True

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        **kwargs,
    ) -> SynthesisResult:
        """Generate speech from text."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Generate audio using your model
        # audio should be numpy array of int16 PCM
        import numpy as np
        audio = np.zeros(24000, dtype=np.int16)  # placeholder
        sample_rate = 24000

        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            voice=voice,
            text=text,
        )

    def get_voices(self) -> list[Voice]:
        """Return available voices."""
        return [
            Voice(id="default", name="Default", language="en"),
        ]

    def is_loaded(self) -> bool:
        return self._loaded
```

### Step 2: Register in Discovery

Add your backend to `jetson_speech/tts/registry.py` in `_discover_backends()`:

```python
def _discover_backends() -> None:
    # ... existing imports ...

    # My custom backend
    try:
        from jetson_speech.tts import mybackend  # noqa: F401
    except ImportError:
        pass
```

### Step 3: Add Dependencies (Optional)

If your backend has specific dependencies, add them to `pyproject.toml`:

```toml
[project.optional-dependencies]
mybackend = ["my-model-library>=1.0"]
```

## STT Backend

The process is similar for STT backends. Create a file in `jetson_speech/stt/`:

```python
from jetson_speech.stt.base import STTBackend, TranscriptionResult, TranscriptionSegment
from jetson_speech.stt.registry import register_stt_backend

@register_stt_backend("mybackend")
class MySTTBackend(STTBackend):
    """My custom STT backend."""

    name = "mybackend"
    supports_streaming = False

    def load(self, model_size: str = "base", **kwargs) -> None:
        """Load the model."""
        self._loaded = True

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        # Perform transcription
        text = "transcribed text"

        return TranscriptionResult(
            text=text,
            segments=[
                TranscriptionSegment(
                    text=text,
                    start=0.0,
                    end=len(audio) / sample_rate,
                    confidence=1.0,
                )
            ],
            language=language or "en",
            duration=len(audio) / sample_rate,
        )

    def get_languages(self) -> list[str]:
        return ["en"]

    def is_loaded(self) -> bool:
        return self._loaded
```

## Streaming Support

To add streaming support, implement the `stream()` method:

### TTS Streaming

```python
def stream(
    self,
    text: str,
    voice: str = "default",
    **kwargs,
) -> Iterator[SynthesisResult]:
    """Stream audio chunks."""
    from jetson_speech.core.text import split_into_chunks

    chunks = split_into_chunks(text, by_sentence=True)

    for i, chunk in enumerate(chunks):
        result = self.synthesize(chunk, voice, **kwargs)
        result.metadata["chunk_index"] = i
        result.metadata["total_chunks"] = len(chunks)
        yield result
```

### STT Streaming

```python
def stream(
    self,
    audio_stream: Iterator[np.ndarray],
    sample_rate: int,
    **kwargs,
) -> Iterator[TranscriptionSegment]:
    """Stream transcription."""
    for audio_chunk in audio_stream:
        result = self.transcribe(audio_chunk, sample_rate)
        for segment in result.segments:
            yield segment
```

## Testing Your Backend

Create tests in `tests/test_mybackend.py`:

```python
import pytest
from jetson_speech.tts.registry import get_tts_backend

def test_backend_loads():
    backend = get_tts_backend("mybackend")
    backend.load()
    assert backend.is_loaded()

def test_synthesize():
    backend = get_tts_backend("mybackend")
    backend.load()
    result = backend.synthesize("Hello world")
    assert len(result.audio) > 0
```

Run tests:

```bash
pytest tests/test_mybackend.py -v
```
