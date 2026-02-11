"""
Wake word detection for voice assistant.

Supports multiple backends:
- OpenWakeWord (default, open source)
- Porcupine (commercial, more accurate)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np


class WakeWordDetector(ABC):
    """Abstract base class for wake word detectors."""

    @abstractmethod
    def detect(self, audio: np.ndarray) -> bool:
        """
        Check if wake word is present in audio chunk.

        Args:
            audio: Audio samples (int16, 16kHz mono)

        Returns:
            True if wake word detected
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass


class OpenWakeWordDetector(WakeWordDetector):
    """
    Wake word detection using OpenWakeWord.

    Supports built-in models:
    - "hey_jarvis"
    - "alexa"
    - "hey_mycroft"
    - "ok_google" (approximation)

    Or custom .onnx models.
    """

    # Built-in wake word models
    BUILTIN_MODELS = {
        "hey_jarvis": "hey_jarvis_v0.1",
        "alexa": "alexa_v0.1",
        "hey_mycroft": "hey_mycroft_v0.1",
    }

    def __init__(
        self,
        wake_word: str = "hey_jarvis",
        threshold: float = 0.5,
        model_path: Optional[str] = None,
    ):
        """
        Initialize OpenWakeWord detector.

        Args:
            wake_word: Wake word to detect (built-in name or custom)
            threshold: Detection threshold (0.0-1.0)
            model_path: Path to custom .onnx model file
        """
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as e:
            raise ImportError(
                "OpenWakeWord not installed. "
                "Install with: pip install openwakeword"
            ) from e

        self.wake_word = wake_word
        self.threshold = threshold

        # Download models if needed (API changed in newer versions)
        try:
            if hasattr(openwakeword.utils, 'download_models'):
                openwakeword.utils.download_models()
        except Exception:
            pass  # Models may already be downloaded or bundled

        # Load model
        if model_path:
            self.model = Model(wakeword_models=[model_path])
            self._model_name = Path(model_path).stem
        elif wake_word in self.BUILTIN_MODELS:
            self._model_name = self.BUILTIN_MODELS[wake_word]
            self.model = Model(wakeword_models=[self._model_name])
        else:
            # Try as direct model name
            self._model_name = wake_word
            self.model = Model(wakeword_models=[wake_word])

        logger.info("Wake word detector ready: '%s'", wake_word)

    def detect(self, audio: np.ndarray) -> bool:
        """Check if wake word is in audio chunk."""
        # OpenWakeWord expects int16 audio
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)

        # Get predictions
        predictions = self.model.predict(audio)

        # Check all model outputs (handles model name variations)
        for model_name, score in predictions.items():
            if score > self.threshold:
                return True

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self.model.reset()


class PorcupineDetector(WakeWordDetector):
    """
    Wake word detection using Picovoice Porcupine.

    Requires access key from https://console.picovoice.ai/
    More accurate but requires registration.
    """

    def __init__(
        self,
        wake_word: str = "jarvis",
        access_key: Optional[str] = None,
        sensitivity: float = 0.5,
    ):
        """
        Initialize Porcupine detector.

        Args:
            wake_word: Built-in wake word or path to .ppn file
            access_key: Picovoice access key (or set PICOVOICE_ACCESS_KEY env)
            sensitivity: Detection sensitivity (0.0-1.0)
        """
        try:
            import pvporcupine
        except ImportError as e:
            raise ImportError(
                "Porcupine not installed. "
                "Install with: pip install pvporcupine"
            ) from e

        import os

        self.wake_word = wake_word

        # Get access key
        access_key = access_key or os.environ.get("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise ValueError(
                "Porcupine requires an access key. "
                "Get one at https://console.picovoice.ai/ "
                "and set PICOVOICE_ACCESS_KEY environment variable."
            )

        # Check if custom model or built-in
        if Path(wake_word).exists():
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[wake_word],
                sensitivities=[sensitivity],
            )
        else:
            # Built-in keyword
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[wake_word],
                sensitivities=[sensitivity],
            )

        self.frame_length = self.porcupine.frame_length
        self._buffer = np.array([], dtype=np.int16)

        logger.info("Porcupine wake word detector ready: '%s'", wake_word)

    def detect(self, audio: np.ndarray) -> bool:
        """Check if wake word is in audio chunk."""
        # Porcupine expects int16 audio
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)

        # Buffer audio and process in frame_length chunks
        self._buffer = np.concatenate([self._buffer, audio.flatten()])

        while len(self._buffer) >= self.frame_length:
            frame = self._buffer[: self.frame_length]
            self._buffer = self._buffer[self.frame_length :]

            result = self.porcupine.process(frame)
            if result >= 0:
                return True

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self._buffer = np.array([], dtype=np.int16)

    def __del__(self):
        if hasattr(self, "porcupine"):
            self.porcupine.delete()


class SimpleEnergyDetector(WakeWordDetector):
    """
    Simple energy-based "wake word" detector.

    Not a real wake word detector - just detects when someone starts speaking.
    Useful for testing or push-to-talk style interfaces.
    """

    def __init__(self, threshold: float = 500.0, min_frames: int = 3):
        """
        Initialize energy detector.

        Args:
            threshold: RMS energy threshold for speech detection
            min_frames: Minimum consecutive frames above threshold
        """
        self.threshold = threshold
        self.min_frames = min_frames
        self._above_count = 0
        logger.info("Using simple energy detector (no wake word)")

    def detect(self, audio: np.ndarray) -> bool:
        """Check if speech energy is above threshold."""
        # Calculate RMS energy
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        rms = np.sqrt(np.mean(audio**2)) * 32768

        if rms > self.threshold:
            self._above_count += 1
            if self._above_count >= self.min_frames:
                self._above_count = 0
                return True
        else:
            self._above_count = 0

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self._above_count = 0


def create_wakeword_detector(
    backend: str = "openwakeword",
    wake_word: str = "hey_jarvis",
    **kwargs,
) -> WakeWordDetector:
    """
    Factory function to create wake word detector.

    Args:
        backend: "openwakeword", "porcupine", or "energy"
        wake_word: Wake word to detect
        **kwargs: Backend-specific options

    Returns:
        WakeWordDetector instance
    """
    if backend == "openwakeword":
        return OpenWakeWordDetector(wake_word=wake_word, **kwargs)
    elif backend == "porcupine":
        return PorcupineDetector(wake_word=wake_word, **kwargs)
    elif backend == "energy":
        return SimpleEnergyDetector(**kwargs)
    else:
        raise ValueError(f"Unknown wake word backend: {backend}")
