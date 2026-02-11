"""
Audio processing utilities.
"""

import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np


def save_audio(audio: np.ndarray, sample_rate: int, path: str | Path) -> None:
    """
    Save audio to WAV file.

    Args:
        audio: Audio data (int16 PCM)
        sample_rate: Sample rate in Hz
        path: Output file path
    """
    from scipy.io import wavfile

    wavfile.write(str(path), sample_rate, audio)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load audio from file.

    Args:
        path: Path to audio file

    Returns:
        Tuple of (audio data, sample rate)
    """
    from scipy.io import wavfile

    sample_rate, audio = wavfile.read(str(path))

    # Convert to int16 if needed
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    return audio, sample_rate


def audio_to_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert audio array to WAV bytes.

    Args:
        audio: Audio data (int16 PCM)
        sample_rate: Sample rate in Hz

    Returns:
        WAV file bytes
    """
    from scipy.io import wavfile

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio)
    return buffer.getvalue()


def bytes_to_audio(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert WAV bytes to audio array.

    Args:
        wav_bytes: WAV file bytes

    Returns:
        Tuple of (audio data, sample rate)
    """
    from scipy.io import wavfile

    buffer = io.BytesIO(wav_bytes)
    sample_rate, audio = wavfile.read(buffer)
    return audio, sample_rate


def concatenate_audio(
    audio_list: list[np.ndarray],
    sample_rate: int,
    silence_duration: float = 0.3,
) -> np.ndarray:
    """
    Concatenate multiple audio arrays with silence between them.

    Args:
        audio_list: List of audio arrays
        sample_rate: Sample rate in Hz
        silence_duration: Silence between chunks in seconds

    Returns:
        Concatenated audio array
    """
    if not audio_list:
        return np.array([], dtype=np.int16)

    # Create silence buffer
    silence_samples = int(sample_rate * silence_duration)
    silence = np.zeros(silence_samples, dtype=np.int16)

    # Concatenate with silence
    result = []
    for i, audio in enumerate(audio_list):
        result.append(audio)
        if i < len(audio_list) - 1:
            result.append(silence)

    return np.concatenate(result)


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    """
    Play audio using system audio player.

    Args:
        audio: Audio data (int16 PCM)
        sample_rate: Sample rate in Hz
    """
    # Save to temp file and play
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        save_audio(audio, sample_rate, f.name)
        temp_path = f.name

    try:
        # Try different audio players
        players = ["aplay", "paplay", "ffplay", "play"]

        for player in players:
            try:
                if player == "ffplay":
                    subprocess.run(
                        [player, "-nodisp", "-autoexit", "-loglevel", "quiet", temp_path],
                        check=True,
                    )
                else:
                    subprocess.run([player, "-q", temp_path], check=True)
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        else:
            raise RuntimeError("No audio player found. Install aplay, paplay, or ffplay.")
    finally:
        os.unlink(temp_path)


def play_audio_stream(audio_stream: BinaryIO, sample_rate: int) -> None:
    """
    Play audio from a stream.

    Args:
        audio_stream: Stream of WAV data
        sample_rate: Sample rate in Hz
    """
    # Read all data and play
    wav_bytes = audio_stream.read()
    audio, _ = bytes_to_audio(wav_bytes)
    play_audio(audio, sample_rate)


def resample_audio(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    """
    Resample audio to a different sample rate.

    Args:
        audio: Audio data
        original_rate: Original sample rate
        target_rate: Target sample rate

    Returns:
        Resampled audio
    """
    if original_rate == target_rate:
        return audio

    from scipy import signal

    # Calculate number of samples in resampled audio
    num_samples = int(len(audio) * target_rate / original_rate)

    # Resample
    resampled = signal.resample(audio, num_samples)

    # Convert back to int16 if needed
    if audio.dtype == np.int16:
        resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

    return resampled


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level.

    Args:
        audio: Audio data
        target_db: Target dB level (default -3 dB)

    Returns:
        Normalized audio
    """
    # Convert to float for processing
    audio_float = audio.astype(np.float64)

    # Calculate current peak
    peak = np.max(np.abs(audio_float))
    if peak == 0:
        return audio

    # Calculate gain needed
    target_linear = 10 ** (target_db / 20) * 32767
    gain = target_linear / peak

    # Apply gain
    normalized = audio_float * gain

    # Clip and convert back to int16
    return np.clip(normalized, -32768, 32767).astype(np.int16)


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Get audio duration in seconds.

    Args:
        audio: Audio data
        sample_rate: Sample rate in Hz

    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate
