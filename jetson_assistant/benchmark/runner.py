"""
Benchmark runner for TTS and STT backends.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

from jetson_assistant.benchmark.metrics import (
    BenchmarkMetrics,
    MetricsCollector,
    get_gpu_memory_usage,
    get_memory_usage,
)


def run_benchmark(
    benchmark_type: str,
    backends: list[str] | None = None,
    text: str = "The quick brown fox jumps over the lazy dog.",
    audio_path: str | None = None,
    iterations: int = 3,
    warmup: int = 1,
) -> list[BenchmarkMetrics]:
    """
    Run benchmarks on TTS or STT backends.

    Args:
        benchmark_type: "tts" or "stt"
        backends: List of backend names (None = all available)
        text: Text for TTS benchmarks
        audio_path: Audio file path for STT benchmarks
        iterations: Number of test iterations
        warmup: Number of warmup iterations

    Returns:
        List of BenchmarkMetrics for each backend
    """
    results = []

    if benchmark_type == "tts":
        results = _benchmark_tts(backends, text, iterations, warmup)
    elif benchmark_type == "stt":
        if not audio_path:
            raise ValueError("audio_path required for STT benchmarks")
        results = _benchmark_stt(backends, audio_path, iterations, warmup)
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    return results


def _benchmark_tts(
    backends: list[str] | None,
    text: str,
    iterations: int,
    warmup: int,
) -> list[BenchmarkMetrics]:
    """Run TTS benchmarks."""
    from jetson_assistant.core.engine import Engine
    from jetson_assistant.tts.registry import list_tts_backends

    # Get backends to test
    if backends is None:
        backends = [b["name"] for b in list_tts_backends()]

    results = []

    for backend_name in backends:
        logger.info("Benchmarking TTS: %s...", backend_name)

        metrics = BenchmarkMetrics(
            backend=backend_name,
            benchmark_type="tts",
            iterations=iterations,
        )

        try:
            # Initialize engine
            engine = Engine()

            # Record memory before
            metrics.memory_before_mb = get_memory_usage()
            metrics.gpu_memory_before_mb = get_gpu_memory_usage()

            # Load backend
            engine.load_tts_backend(backend_name)

            # Warmup
            for _ in range(warmup):
                engine.synthesize(text)

            # Benchmark iterations
            for i in range(iterations):
                with MetricsCollector() as collector:
                    result = engine.synthesize(text)
                    collector.mark_first_byte()

                metrics.times.append(collector.elapsed)
                metrics.ttfb.append(collector.ttfb)
                metrics.audio_durations.append(result.duration)
                metrics.chars_processed.append(len(text))

                logger.info("  [%d/%d] %.3fs", i + 1, iterations, collector.elapsed)

            # Record memory after
            metrics.memory_after_mb = get_memory_usage()
            metrics.gpu_memory_after_mb = get_gpu_memory_usage()

            # Cleanup
            engine.unload_tts_backend()

            # Store metadata
            metrics.metadata = {
                "text_length": len(text),
                "model_info": engine.get_tts_info() if engine.get_tts_info().get("loaded") else {},
            }

        except Exception as e:
            logger.error("  Error: %s", e)
            metrics.metadata["error"] = str(e)

        results.append(metrics)

    return results


def _benchmark_stt(
    backends: list[str] | None,
    audio_path: str,
    iterations: int,
    warmup: int,
) -> list[BenchmarkMetrics]:
    """Run STT benchmarks."""
    from jetson_assistant.core.audio import load_audio
    from jetson_assistant.core.engine import Engine
    from jetson_assistant.stt.registry import list_stt_backends

    # Load audio once
    audio, sample_rate = load_audio(audio_path)
    audio_duration = len(audio) / sample_rate

    # Get backends to test
    if backends is None:
        backends = [b["name"] for b in list_stt_backends()]

    results = []

    for backend_name in backends:
        logger.info("Benchmarking STT: %s...", backend_name)

        metrics = BenchmarkMetrics(
            backend=backend_name,
            benchmark_type="stt",
            iterations=iterations,
        )

        try:
            # Initialize engine
            engine = Engine()

            # Record memory before
            metrics.memory_before_mb = get_memory_usage()
            metrics.gpu_memory_before_mb = get_gpu_memory_usage()

            # Load backend
            engine.load_stt_backend(backend_name)

            # Warmup
            for _ in range(warmup):
                engine.transcribe(audio, sample_rate)

            # Benchmark iterations
            for i in range(iterations):
                with MetricsCollector() as collector:
                    result = engine.transcribe(audio, sample_rate)
                    collector.mark_first_byte()

                metrics.times.append(collector.elapsed)
                metrics.ttfb.append(collector.ttfb)
                metrics.audio_lengths.append(audio_duration)

                logger.info("  [%d/%d] %.3fs", i + 1, iterations, collector.elapsed)

            # Record memory after
            metrics.memory_after_mb = get_memory_usage()
            metrics.gpu_memory_after_mb = get_gpu_memory_usage()

            # Cleanup
            engine.unload_stt_backend()

            # Store metadata
            metrics.metadata = {
                "audio_path": audio_path,
                "audio_duration": audio_duration,
                "sample_rate": sample_rate,
            }

        except Exception as e:
            logger.error("  Error: %s", e)
            metrics.metadata["error"] = str(e)

        results.append(metrics)

    return results
