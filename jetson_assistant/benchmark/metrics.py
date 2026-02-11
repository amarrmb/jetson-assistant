"""
Benchmark metrics collection.
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkMetrics:
    """Collected metrics from a benchmark run."""

    backend: str
    benchmark_type: str  # "tts" or "stt"
    iterations: int = 0

    # Timing metrics (in seconds)
    times: list[float] = field(default_factory=list)
    ttfb: list[float] = field(default_factory=list)  # Time to first byte

    # TTS-specific
    audio_durations: list[float] = field(default_factory=list)
    chars_processed: list[int] = field(default_factory=list)

    # STT-specific
    audio_lengths: list[float] = field(default_factory=list)  # Input audio duration

    # Resource metrics
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    gpu_memory_before_mb: float = 0.0
    gpu_memory_after_mb: float = 0.0

    # Extra info
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_time(self) -> float:
        """Average processing time."""
        return sum(self.times) / len(self.times) if self.times else 0.0

    @property
    def min_time(self) -> float:
        """Minimum processing time."""
        return min(self.times) if self.times else 0.0

    @property
    def max_time(self) -> float:
        """Maximum processing time."""
        return max(self.times) if self.times else 0.0

    @property
    def avg_ttfb(self) -> float:
        """Average time to first byte."""
        return sum(self.ttfb) / len(self.ttfb) if self.ttfb else 0.0

    @property
    def avg_rtf(self) -> float:
        """
        Average Real-Time Factor.

        RTF < 1 means faster than real-time.
        For TTS: processing_time / audio_duration
        For STT: processing_time / audio_length
        """
        if self.benchmark_type == "tts" and self.audio_durations:
            total_time = sum(self.times)
            total_duration = sum(self.audio_durations)
            return total_time / total_duration if total_duration > 0 else 0.0
        elif self.benchmark_type == "stt" and self.audio_lengths:
            total_time = sum(self.times)
            total_length = sum(self.audio_lengths)
            return total_time / total_length if total_length > 0 else 0.0
        return 0.0

    @property
    def chars_per_second(self) -> float:
        """Characters processed per second (TTS only)."""
        if not self.chars_processed or not self.times:
            return 0.0
        total_chars = sum(self.chars_processed)
        total_time = sum(self.times)
        return total_chars / total_time if total_time > 0 else 0.0

    @property
    def memory_delta_mb(self) -> float:
        """Memory usage increase in MB."""
        return self.memory_after_mb - self.memory_before_mb

    @property
    def gpu_memory_delta_mb(self) -> float:
        """GPU memory usage increase in MB."""
        return self.gpu_memory_after_mb - self.gpu_memory_before_mb

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend,
            "type": self.benchmark_type,
            "iterations": self.iterations,
            "timing": {
                "avg_time": self.avg_time,
                "min_time": self.min_time,
                "max_time": self.max_time,
                "avg_ttfb": self.avg_ttfb,
                "times": self.times,
            },
            "performance": {
                "avg_rtf": self.avg_rtf,
                "chars_per_second": self.chars_per_second,
            },
            "memory": {
                "before_mb": self.memory_before_mb,
                "after_mb": self.memory_after_mb,
                "delta_mb": self.memory_delta_mb,
                "gpu_before_mb": self.gpu_memory_before_mb,
                "gpu_after_mb": self.gpu_memory_after_mb,
                "gpu_delta_mb": self.gpu_memory_delta_mb,
            },
            "metadata": self.metadata,
        }


class MetricsCollector:
    """Context manager for collecting timing metrics."""

    def __init__(self):
        self.start_time: float = 0.0
        self.first_byte_time: float | None = None
        self.end_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    def mark_first_byte(self):
        """Mark when first output byte is available."""
        if self.first_byte_time is None:
            self.first_byte_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Total elapsed time in seconds."""
        return self.end_time - self.start_time

    @property
    def ttfb(self) -> float:
        """Time to first byte in seconds."""
        if self.first_byte_time is None:
            return self.elapsed
        return self.first_byte_time - self.start_time


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0
