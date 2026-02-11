#!/usr/bin/env python3
"""
Comparative benchmark for jetson-assistant TTS and STT backends.

Produces publishable results in Markdown and JSON formats.
Run on Jetson Thor for representative numbers.

Usage:
    python scripts/benchmark_backends.py
    python scripts/benchmark_backends.py --output results/
    python scripts/benchmark_backends.py --tts-only
    python scripts/benchmark_backends.py --stt-only
    python scripts/benchmark_backends.py --iterations 10
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BackendResult:
    """Result from a single backend benchmark."""

    name: str
    backend_type: str  # "tts" or "stt"
    times_ms: list[float] = field(default_factory=list)
    sample_rate: int = 0
    memory_mb: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def avg_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0


# Standard test sentences (short, medium, long)
TTS_SENTENCES = [
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "In a groundbreaking development, researchers at the university have discovered a new method "
    "for synthesizing complex organic molecules that could revolutionize pharmaceutical manufacturing.",
    "Welcome to the demo.",
    "I can see you're holding a red cup in your left hand.",
]


def get_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def benchmark_tts(
    backends: list[str] | None = None,
    iterations: int = 5,
    warmup: int = 1,
) -> list[BackendResult]:
    """Benchmark TTS backends."""
    from jetson_assistant.tts.registry import get_tts_backend, list_tts_backends

    if backends is None:
        backends = [b["name"] for b in list_tts_backends()]

    results = []

    for name in backends:
        print(f"\n--- TTS: {name} ---", file=sys.stderr)
        result = BackendResult(name=name, backend_type="tts")

        try:
            backend = get_tts_backend(name)

            # Load with defaults
            mem_before = get_memory_mb()
            if name == "kokoro":
                backend.load(voice="af_heart")
            elif name == "piper":
                backend.load(voice="en_US-amy-medium")
            else:
                backend.load()
            result.memory_mb = get_memory_mb() - mem_before

            # Warmup
            for _ in range(warmup):
                backend.synthesize("Warmup sentence.")

            # Benchmark each sentence
            for sentence in TTS_SENTENCES:
                for i in range(iterations):
                    start = time.perf_counter()
                    synth_result = backend.synthesize(sentence)
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    result.times_ms.append(elapsed_ms)
                    result.sample_rate = synth_result.sample_rate

                    if i == 0:
                        print(
                            f"  \"{sentence[:40]}...\" "
                            f"{elapsed_ms:.1f}ms ({len(synth_result.audio)} samples)",
                            file=sys.stderr,
                        )

            result.metadata["sentences"] = len(TTS_SENTENCES)
            result.metadata["sample_rate"] = result.sample_rate
            backend.unload()

        except Exception as e:
            result.error = str(e)
            print(f"  ERROR: {e}", file=sys.stderr)

        results.append(result)

    return results


def benchmark_stt(
    backends: list[str] | None = None,
    iterations: int = 5,
    warmup: int = 1,
    audio_path: str | None = None,
) -> list[BackendResult]:
    """Benchmark STT backends."""
    from jetson_assistant.stt.registry import get_stt_backend, list_stt_backends

    if backends is None:
        # Only benchmark locally-loadable backends (not vllm which needs a server)
        all_backends = [b["name"] for b in list_stt_backends()]
        backends = [b for b in all_backends if b != "vllm"]

    # Generate or load test audio
    if audio_path:
        from scipy.io import wavfile
        sr, audio = wavfile.read(audio_path)
    else:
        # Generate 3 seconds of speech-like audio (sine wave + noise)
        sr = 16000
        t = np.linspace(0, 3.0, sr * 3)
        audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        print("  Using synthetic audio (no test WAV provided)", file=sys.stderr)

    results = []

    for name in backends:
        print(f"\n--- STT: {name} ---", file=sys.stderr)
        result = BackendResult(name=name, backend_type="stt")

        try:
            backend = get_stt_backend(name)

            mem_before = get_memory_mb()
            if name == "nemotron":
                backend.load(model_size="0.6b")
            else:
                backend.load(model_size="base")
            result.memory_mb = get_memory_mb() - mem_before

            # Warmup
            for _ in range(warmup):
                backend.transcribe(audio, sr)

            # Benchmark
            for i in range(iterations):
                start = time.perf_counter()
                transcription = backend.transcribe(audio, sr)
                elapsed_ms = (time.perf_counter() - start) * 1000

                result.times_ms.append(elapsed_ms)

                if i == 0:
                    print(
                        f"  Transcription: \"{transcription.text[:60]}\" ({elapsed_ms:.1f}ms)",
                        file=sys.stderr,
                    )

            result.metadata["audio_duration_s"] = len(audio) / sr
            backend.unload()

        except Exception as e:
            result.error = str(e)
            print(f"  ERROR: {e}", file=sys.stderr)

        results.append(result)

    return results


def format_markdown(tts_results: list[BackendResult], stt_results: list[BackendResult]) -> str:
    """Format results as publishable Markdown."""
    lines = ["# Jetson Assistant Backend Benchmarks\n"]

    if tts_results:
        lines.append("## TTS Backends\n")
        lines.append("| Backend | Min (ms) | Avg (ms) | P95 (ms) | Max (ms) | Sample Rate | Memory (MB) |")
        lines.append("|---------|----------|----------|----------|----------|-------------|-------------|")
        for r in tts_results:
            if r.error:
                lines.append(f"| {r.name} | ERROR | - | - | - | - | - |")
            else:
                lines.append(
                    f"| {r.name} | {r.min_ms:.0f} | {r.avg_ms:.0f} | "
                    f"{r.p95_ms:.0f} | {r.max_ms:.0f} | {r.sample_rate} | "
                    f"{r.memory_mb:.0f} |"
                )
        lines.append("")

    if stt_results:
        lines.append("## STT Backends\n")
        lines.append("| Backend | Min (ms) | Avg (ms) | P95 (ms) | Max (ms) | Memory (MB) |")
        lines.append("|---------|----------|----------|----------|----------|-------------|")
        for r in stt_results:
            if r.error:
                lines.append(f"| {r.name} | ERROR | - | - | - | - |")
            else:
                lines.append(
                    f"| {r.name} | {r.min_ms:.0f} | {r.avg_ms:.0f} | "
                    f"{r.p95_ms:.0f} | {r.max_ms:.0f} | {r.memory_mb:.0f} |"
                )
        lines.append("")

    lines.append(f"*Benchmarked on {time.strftime('%Y-%m-%d')}*\n")
    return "\n".join(lines)


def format_json(tts_results: list[BackendResult], stt_results: list[BackendResult]) -> str:
    """Format results as JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tts": [
            {
                "name": r.name,
                "min_ms": r.min_ms,
                "avg_ms": r.avg_ms,
                "p95_ms": r.p95_ms,
                "max_ms": r.max_ms,
                "sample_rate": r.sample_rate,
                "memory_mb": r.memory_mb,
                "error": r.error,
                "times_ms": r.times_ms,
                "metadata": r.metadata,
            }
            for r in tts_results
        ],
        "stt": [
            {
                "name": r.name,
                "min_ms": r.min_ms,
                "avg_ms": r.avg_ms,
                "p95_ms": r.p95_ms,
                "max_ms": r.max_ms,
                "memory_mb": r.memory_mb,
                "error": r.error,
                "times_ms": r.times_ms,
                "metadata": r.metadata,
            }
            for r in stt_results
        ],
    }
    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark jetson-assistant backends")
    parser.add_argument("--tts-only", action="store_true", help="Only benchmark TTS")
    parser.add_argument("--stt-only", action="store_true", help="Only benchmark STT")
    parser.add_argument("--tts-backends", type=str, help="Comma-separated TTS backends")
    parser.add_argument("--stt-backends", type=str, help="Comma-separated STT backends")
    parser.add_argument("--iterations", "-n", type=int, default=5, help="Iterations per test")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--audio", type=str, help="Audio file for STT benchmarks")
    parser.add_argument("--output", "-o", type=str, help="Output directory for results")
    args = parser.parse_args()

    tts_backends = args.tts_backends.split(",") if args.tts_backends else None
    stt_backends = args.stt_backends.split(",") if args.stt_backends else None

    tts_results = []
    stt_results = []

    if not args.stt_only:
        print("=" * 50, file=sys.stderr)
        print("TTS BENCHMARKS", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        tts_results = benchmark_tts(tts_backends, args.iterations, args.warmup)

    if not args.tts_only:
        print("\n" + "=" * 50, file=sys.stderr)
        print("STT BENCHMARKS", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        stt_results = benchmark_stt(stt_backends, args.iterations, args.warmup, args.audio)

    # Output
    md = format_markdown(tts_results, stt_results)
    print("\n" + md)

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "benchmark.md").write_text(md)
        (output_dir / "benchmark.json").write_text(format_json(tts_results, stt_results))
        print(f"\nResults saved to {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
