"""
Audio I/O handling for voice assistant.

Provides:
- Continuous microphone input with callbacks
- Speaker output with queue management
- Voice Activity Detection (VAD)
- Audio chimes/feedback sounds
"""

import logging
import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

import numpy as np


def _find_usb_audio_device(kind: str = "input") -> Optional[int]:
    """Auto-detect a USB audio device, preferring devices with both input and output.

    On Jetson, the system default is often an HDMI or APE virtual device with no
    real microphone. This scans for USB audio devices and picks the best one.

    Args:
        kind: "input" or "output"

    Returns:
        Device index, or None if no USB device found.
    """
    try:
        import sounddevice as sd
    except ImportError:
        return None

    # Virtual/internal devices to skip
    _SKIP = {"HDMI", "HDA", "APE", "DisplayPort"}
    need_input = kind == "input"
    need_output = kind == "output"

    best = None
    best_score = -1

    for i, dev in enumerate(sd.query_devices()):
        name = dev["name"]
        has_in = dev["max_input_channels"] > 0
        has_out = dev["max_output_channels"] > 0

        # Skip virtual/internal devices
        if any(skip in name for skip in _SKIP):
            continue

        # Must have the channels we need
        if need_input and not has_in:
            continue
        if need_output and not has_out:
            continue

        # Score: prefer devices with both in+out (speakerphones), then USB keyword
        score = 0
        if has_in and has_out:
            score += 2  # bidirectional (speakerphone) preferred
        if "USB" in name:
            score += 1

        if score > best_score:
            best = i
            best_score = score

    return best


@dataclass
class AudioConfig:
    """Audio configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100  # 100ms chunks
    dtype: np.dtype = np.int16

    @property
    def chunk_size(self) -> int:
        """Samples per chunk."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)


class VoiceActivityDetector:
    """
    Detect voice activity in audio stream.

    Uses WebRTC VAD or simple energy-based detection.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mode: int = 3,
        use_webrtc: bool = True,
    ):
        """
        Initialize VAD.

        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            mode: Aggressiveness (0-3, 3 is most aggressive)
            use_webrtc: Use WebRTC VAD if available
        """
        self.sample_rate = sample_rate
        self._webrtc_vad = None

        if use_webrtc:
            try:
                import webrtcvad

                self._webrtc_vad = webrtcvad.Vad(mode)
            except ImportError:
                pass

        if self._webrtc_vad:
            logger.info("Using WebRTC VAD")
        else:
            logger.info("Using energy-based VAD")

        # Energy-based fallback parameters
        self.energy_threshold = 500.0
        self._speech_frames = 0
        self._silence_frames = 0

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.

        Args:
            audio: Audio samples (int16)

        Returns:
            True if speech detected
        """
        if self._webrtc_vad:
            return self._webrtc_is_speech(audio)
        else:
            return self._energy_is_speech(audio)

    def _webrtc_is_speech(self, audio: np.ndarray) -> bool:
        """WebRTC VAD detection."""
        # WebRTC VAD requires specific frame sizes: 10, 20, or 30ms
        # At 16kHz: 160, 320, or 480 samples
        frame_size = 480  # 30ms at 16kHz

        if len(audio) < frame_size:
            return False

        # Check multiple frames and vote
        speech_count = 0
        total_frames = 0

        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i : i + frame_size]
            if frame.dtype != np.int16:
                frame = (frame * 32767).astype(np.int16)

            try:
                if self._webrtc_vad.is_speech(frame.tobytes(), self.sample_rate):
                    speech_count += 1
                total_frames += 1
            except Exception:
                pass

        return speech_count > total_frames / 2

    def _energy_is_speech(self, audio: np.ndarray) -> bool:
        """Simple energy-based detection."""
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio

        rms = np.sqrt(np.mean(audio_float**2)) * 32768
        return rms > self.energy_threshold

    def detect_end_of_speech(
        self,
        audio: np.ndarray,
        silence_threshold_ms: int = 1000,
        chunk_duration_ms: int = 100,
    ) -> bool:
        """
        Detect end of speech (silence after speech).

        Args:
            audio: Audio chunk
            silence_threshold_ms: How long silence = end of speech
            chunk_duration_ms: Duration of this chunk

        Returns:
            True if end of speech detected
        """
        frames_for_silence = silence_threshold_ms // chunk_duration_ms

        if self.is_speech(audio):
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            if self._speech_frames > 0:  # Only count silence after speech
                self._silence_frames += 1

        # End of speech = had speech, then enough silence
        if self._speech_frames > 2 and self._silence_frames >= frames_for_silence:
            self._speech_frames = 0
            self._silence_frames = 0
            return True

        return False

    def reset(self) -> None:
        """Reset VAD state."""
        self._speech_frames = 0
        self._silence_frames = 0


class AudioInput:
    """
    Continuous microphone input with callback.

    Runs in background thread, calls callback with audio chunks.
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        device: Optional[int] = None,
    ):
        """
        Initialize audio input.

        Args:
            config: Audio configuration
            device: Input device index (None for default)
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "sounddevice not installed. "
                "Install with: pip install sounddevice"
            ) from e

        self.config = config or AudioConfig()
        self._sd = sd

        self._stream = None
        self._callback = None
        self._running = False

        # Auto-detect USB audio device if none specified
        if device is None:
            device = _find_usb_audio_device(kind="input")
            if device is not None:
                logger.info("Auto-detected USB input device: [%d] %s",
                            device, sd.query_devices(device)["name"])
        self.device = device

        # Get device info for logging
        if self.device is None:
            device_info = sd.query_devices(kind="input")
        else:
            device_info = sd.query_devices(self.device)

        logger.info("Audio input: %s", device_info['name'])

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Start audio capture.

        Args:
            callback: Function called with each audio chunk
        """
        if self._running:
            return

        self._callback = callback
        self._running = True

        # Use larger buffer to prevent overflow
        # blocksize=0 lets PortAudio choose optimal size
        # extra_settings for even larger ALSA buffer
        self._stream = self._sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.chunk_size * 2,  # Double chunk size
            device=self.device,
            callback=self._audio_callback,
            latency=0.5,  # 500ms latency for large buffer
        )
        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback from sounddevice."""
        if status:
            logger.warning("Audio input status: %s", status)

        if self._callback and self._running:
            # Convert to 1D array
            audio = indata[:, 0] if indata.ndim > 1 else indata
            self._callback(audio.copy())

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()


class AudioOutput:
    """
    Audio output with queue for non-blocking playback.

    Plays audio chunks in sequence without blocking main thread.
    Uses aplay (ALSA) for best quality, falls back to sounddevice.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        device: Optional[int] = None,
        use_aplay: bool = True,
    ):
        """
        Initialize audio output.

        Args:
            sample_rate: Default sample rate for playback
            device: Output device index (None for default)
            use_aplay: Use aplay for playback (best quality on Linux)
        """
        self.sample_rate = sample_rate
        self._use_aplay = use_aplay

        self._queue = queue.Queue()
        self._playing = False
        self._thread = None

        # Auto-detect USB audio device if none specified
        if device is None:
            device = _find_usb_audio_device(kind="output")
            if device is not None:
                import sounddevice as sd
                logger.info("Auto-detected USB output device: [%d] %s",
                            device, sd.query_devices(device)["name"])
        self.device = device

        # Check if aplay is available
        if use_aplay:
            import shutil
            if shutil.which("aplay") is None:
                logger.warning("aplay not found, falling back to sounddevice")
                self._use_aplay = False

        if not self._use_aplay:
            try:
                import sounddevice as sd
                self._sd = sd

                # Get device info
                if self.device is None:
                    device_info = sd.query_devices(kind="output")
                else:
                    device_info = sd.query_devices(self.device)
                logger.info("Audio output: %s", device_info['name'])
            except ImportError as e:
                raise ImportError(
                    "sounddevice not installed. "
                    "Install with: pip install sounddevice"
                ) from e
        else:
            logger.info("Audio output: aplay (ALSA)")

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate using scipy."""
        if orig_sr == target_sr:
            return audio

        from scipy import signal

        # Calculate the number of samples in the resampled audio
        target_length = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio.astype(np.float32), target_length)
        return resampled

    def play(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Queue audio for playback.

        Args:
            audio: Audio samples
            sample_rate: Sample rate (uses default if not specified)
        """
        self._queue.put((audio, sample_rate or self.sample_rate))

        if not self._playing:
            self._playing = True
            self._thread = threading.Thread(target=self._playback_thread, daemon=True)
            self._thread.start()

    def _playback_thread(self) -> None:
        """Background thread for audio playback."""
        while self._playing:
            try:
                audio, sr = self._queue.get(timeout=0.1)

                if self._use_aplay:
                    self._play_with_aplay(audio, sr)
                else:
                    # Ensure correct dtype for sounddevice
                    if audio.dtype == np.int16:
                        audio = audio.astype(np.float32) / 32768.0

                    self._sd.play(audio, sr, device=self.device)
                    self._sd.wait()

            except queue.Empty:
                if self._queue.empty():
                    self._playing = False
                    break

    def play_blocking(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play audio and wait for completion.

        Args:
            audio: Audio samples
            sample_rate: Sample rate
        """
        sr = sample_rate or self.sample_rate

        if self._use_aplay:
            self._play_with_aplay(audio, sr)
        else:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            # Let ALSA handle sample rate conversion
            self._sd.play(audio, sr, device=self.device)
            self._sd.wait()

    def _play_with_aplay(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio using aplay (best quality on Linux/ALSA)."""
        import subprocess
        import tempfile
        import os
        from scipy.io import wavfile

        # Ensure int16 format
        if audio.dtype != np.int16:
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)

        # Write to temp file and play
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, sample_rate, audio)
            try:
                # Add timeout to prevent hanging
                cmd = ["aplay", "-q"]
                # Route to specific ALSA device if we auto-detected one
                if self.device is not None:
                    try:
                        import sounddevice as sd
                        dev_name = sd.query_devices(self.device)["name"]
                        # Extract ALSA hw:N,M from device name like "Jabra SPEAK 410 USB: Audio (hw:4,0)"
                        import re
                        m = re.search(r"\(hw:(\d+,\d+)\)", dev_name)
                        if m:
                            cmd.extend(["-D", f"plughw:{m.group(1)}"])
                    except Exception:
                        pass  # Fall back to default ALSA device
                cmd.append(f.name)
                subprocess.run(cmd, check=True, timeout=60)
            except subprocess.TimeoutExpired:
                logger.warning("Audio playback timeout")
            finally:
                os.unlink(f.name)

    def stop(self) -> None:
        """Stop playback and clear queue."""
        self._playing = False

        if not self._use_aplay and hasattr(self, '_sd'):
            self._sd.stop()

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._playing or not self._queue.empty()

    def wait(self) -> None:
        """Wait for all queued audio to finish."""
        if self._thread:
            self._thread.join()


class ChimeSounds:
    """
    Generate feedback sounds for the assistant.

    Provides wake, listening, and error chimes.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def wake_chime(self) -> np.ndarray:
        """Generate a pleasant 'wake up' chime (ascending tones)."""
        duration = 0.15
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # Two ascending tones
        freq1, freq2 = 440, 660  # A4 -> E5
        tone1 = np.sin(2 * np.pi * freq1 * t[:len(t)//2]) * 0.3
        tone2 = np.sin(2 * np.pi * freq2 * t[:len(t)//2]) * 0.3

        # Apply envelope
        envelope1 = np.exp(-3 * np.linspace(0, 1, len(tone1)))
        envelope2 = np.exp(-3 * np.linspace(0, 1, len(tone2)))

        chime = np.concatenate([tone1 * envelope1, tone2 * envelope2])
        return (chime * 32767).astype(np.int16)

    def listening_chime(self) -> np.ndarray:
        """Generate a soft 'listening' indicator."""
        duration = 0.1
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        freq = 880  # A5
        tone = np.sin(2 * np.pi * freq * t) * 0.2
        envelope = np.exp(-5 * np.linspace(0, 1, len(tone)))

        return ((tone * envelope) * 32767).astype(np.int16)

    def done_chime(self) -> np.ndarray:
        """Generate a 'done/understood' chime (descending)."""
        duration = 0.15
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        freq1, freq2 = 660, 440  # E5 -> A4
        tone1 = np.sin(2 * np.pi * freq1 * t[:len(t)//2]) * 0.3
        tone2 = np.sin(2 * np.pi * freq2 * t[:len(t)//2]) * 0.3

        envelope1 = np.exp(-3 * np.linspace(0, 1, len(tone1)))
        envelope2 = np.exp(-3 * np.linspace(0, 1, len(tone2)))

        chime = np.concatenate([tone1 * envelope1, tone2 * envelope2])
        return (chime * 32767).astype(np.int16)

    def thinking_chime(self) -> np.ndarray:
        """Generate a soft 'processing/thinking' indicator (two quick blips)."""
        duration = 0.08
        pause = 0.05
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        pause_samples = int(self.sample_rate * pause)

        freq = 600  # D5
        tone = np.sin(2 * np.pi * freq * t) * 0.15
        envelope = np.exp(-8 * np.linspace(0, 1, len(tone)))
        blip = (tone * envelope * 32767).astype(np.int16)

        # Two blips with pause
        silence = np.zeros(pause_samples, dtype=np.int16)
        return np.concatenate([blip, silence, blip])

    def error_chime(self) -> np.ndarray:
        """Generate an error/problem indicator."""
        duration = 0.3
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # Dissonant tone
        freq = 220  # A3
        tone = np.sin(2 * np.pi * freq * t) * 0.3
        tone += np.sin(2 * np.pi * freq * 1.05 * t) * 0.2  # Slight detuning

        envelope = np.exp(-2 * np.linspace(0, 1, len(tone)))

        return ((tone * envelope) * 32767).astype(np.int16)
