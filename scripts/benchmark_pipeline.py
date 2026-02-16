#!/usr/bin/env python3
"""
E2E pipeline benchmark for jetson-assistant.

Measures TTFB, end-to-end latency, per-stage timing (STT, LLM TTFT, TTS),
and VRAM usage for any config YAML.

Usage:
    python scripts/benchmark_pipeline.py --config configs/thor.yaml
    python scripts/benchmark_pipeline.py --config configs/orin.yaml --runs 5
    python scripts/benchmark_pipeline.py --config configs/nano.yaml --json
"""

import argparse
import json
import time
from dataclasses import dataclass


# Default phrases used when none are provided.
DEFAULT_TEST_PHRASES = [
    "What time is it?",
    "Tell me a joke.",
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "What is the weather like today?",
]


@dataclass
class PipelineMetrics:
    """Aggregated metrics from an E2E pipeline benchmark run.

    All timing values are in milliseconds, memory values in megabytes.
    """

    ttfb_ms: float
    e2e_ms: float
    stt_ms: float
    llm_ttft_ms: float
    tts_ms: float
    vram_idle_mb: float
    vram_peak_mb: float

    def to_markdown(self) -> str:
        """Render metrics as a Markdown table."""
        return (
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| TTFB | {self.ttfb_ms:.0f}ms |\n"
            f"| E2E Latency | {self.e2e_ms:.0f}ms |\n"
            f"| STT | {self.stt_ms:.0f}ms |\n"
            f"| LLM TTFT | {self.llm_ttft_ms:.0f}ms |\n"
            f"| TTS | {self.tts_ms:.0f}ms |\n"
            f"| VRAM Idle | {self.vram_idle_mb:.0f}MB |\n"
            f"| VRAM Peak | {self.vram_peak_mb:.0f}MB |"
        )

    def to_json(self) -> dict:
        """Return metrics as a plain dict suitable for JSON serialization."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def benchmark_pipeline(
    config_path: str,
    runs: int = 10,
    test_phrases: list[str] | None = None,
) -> PipelineMetrics:
    """Run an E2E pipeline benchmark against a config YAML.

    Args:
        config_path: Path to assistant config YAML file.
        runs: Number of runs per test phrase (results are averaged).
        test_phrases: Optional list of input phrases. Defaults to
            ``DEFAULT_TEST_PHRASES``.

    Returns:
        A ``PipelineMetrics`` instance with averaged results across all
        runs and phrases.
    """
    if test_phrases is None:
        test_phrases = DEFAULT_TEST_PHRASES

    # Import heavy deps here so ``--help`` stays fast and tests that only
    # exercise PipelineMetrics do not need CUDA / PyTorch.
    import yaml

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    # --- VRAM measurement (requires CUDA) --------------------------------
    vram_idle = 0.0
    vram_peak = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vram_idle = torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass

    # --- Per-stage timing -------------------------------------------------
    # TODO: Full pipeline benchmark implementation.
    #
    # The real implementation will:
    #   1. Instantiate an AssistantConfig from cfg_dict
    #   2. Load STT, LLM, and TTS backends
    #   3. For each test_phrase * runs iteration:
    #       a. Synthesize a WAV from the phrase (to simulate mic input)
    #       b. Time STT transcription  -> stt_ms
    #       c. Time LLM first-token    -> llm_ttft_ms
    #       d. Time TTS synthesis       -> tts_ms
    #       e. Compute TTFB (stt + llm_ttft)
    #       f. Compute E2E  (stt + llm_full + tts)
    #   4. Average all timings across iterations
    #
    # For now we return zeroed timing metrics because running real
    # benchmarks requires GPU hardware and loaded models.

    stt_ms = 0.0
    llm_ttft_ms = 0.0
    tts_ms = 0.0
    ttfb_ms = 0.0
    e2e_ms = 0.0

    # --- VRAM peak --------------------------------------------------------
    try:
        import torch

        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass

    return PipelineMetrics(
        ttfb_ms=ttfb_ms,
        e2e_ms=e2e_ms,
        stt_ms=stt_ms,
        llm_ttft_ms=llm_ttft_ms,
        tts_ms=tts_ms,
        vram_idle_mb=vram_idle,
        vram_peak_mb=vram_peak,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E pipeline benchmark for jetson-assistant",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to assistant config YAML (e.g. configs/thor.yaml)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per test phrase (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON instead of Markdown",
    )
    args = parser.parse_args()

    metrics = benchmark_pipeline(args.config, args.runs)

    if args.json_output:
        print(json.dumps(metrics.to_json(), indent=2))
    else:
        print(metrics.to_markdown())


if __name__ == "__main__":
    main()
