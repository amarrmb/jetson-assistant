"""
Tests for benchmarking functionality.
"""

import pytest

from jetson_assistant.benchmark.metrics import BenchmarkMetrics, MetricsCollector
from jetson_assistant.benchmark.report import format_results


class TestBenchmarkMetrics:
    """Test benchmark metrics."""

    def test_metrics_creation(self):
        """Test creating benchmark metrics."""
        metrics = BenchmarkMetrics(
            backend="test",
            benchmark_type="tts",
            iterations=3,
        )

        assert metrics.backend == "test"
        assert metrics.benchmark_type == "tts"
        assert metrics.iterations == 3

    def test_timing_metrics(self):
        """Test timing metric calculations."""
        metrics = BenchmarkMetrics(
            backend="test",
            benchmark_type="tts",
            iterations=3,
        )

        metrics.times = [1.0, 2.0, 3.0]

        assert metrics.avg_time == 2.0
        assert metrics.min_time == 1.0
        assert metrics.max_time == 3.0

    def test_rtf_calculation(self):
        """Test RTF calculation."""
        metrics = BenchmarkMetrics(
            backend="test",
            benchmark_type="tts",
            iterations=2,
        )

        # 2 seconds processing for 4 seconds audio = 0.5 RTF
        metrics.times = [1.0, 1.0]
        metrics.audio_durations = [2.0, 2.0]

        assert metrics.avg_rtf == 0.5

    def test_chars_per_second(self):
        """Test chars per second calculation."""
        metrics = BenchmarkMetrics(
            backend="test",
            benchmark_type="tts",
            iterations=2,
        )

        # 100 chars in 1 second = 100 chars/s
        metrics.times = [0.5, 0.5]
        metrics.chars_processed = [50, 50]

        assert metrics.chars_per_second == 100.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = BenchmarkMetrics(
            backend="test",
            benchmark_type="tts",
            iterations=1,
        )
        metrics.times = [1.0]

        d = metrics.to_dict()

        assert d["backend"] == "test"
        assert d["type"] == "tts"
        assert "timing" in d
        assert "performance" in d
        assert "memory" in d


class TestMetricsCollector:
    """Test metrics collector."""

    def test_timing_collection(self):
        """Test collecting timing metrics."""
        import time

        with MetricsCollector() as collector:
            time.sleep(0.1)

        assert collector.elapsed >= 0.1
        assert collector.elapsed < 0.2

    def test_first_byte_marking(self):
        """Test marking first byte time."""
        import time

        with MetricsCollector() as collector:
            time.sleep(0.05)
            collector.mark_first_byte()
            time.sleep(0.05)

        assert collector.ttfb >= 0.05
        assert collector.ttfb < collector.elapsed


class TestReportFormatting:
    """Test report generation."""

    def test_markdown_format(self):
        """Test Markdown report format."""
        metrics = BenchmarkMetrics(
            backend="test",
            benchmark_type="tts",
            iterations=1,
        )
        metrics.times = [1.0]
        metrics.audio_durations = [2.0]
        metrics.chars_processed = [100]

        report = format_results([metrics], format_type="markdown")

        assert "# TTS Benchmark Results" in report
        assert "test" in report
        assert "|" in report

    def test_json_format(self):
        """Test JSON report format."""
        import json

        metrics = BenchmarkMetrics(
            backend="test",
            benchmark_type="tts",
            iterations=1,
        )
        metrics.times = [1.0]

        report = format_results([metrics], format_type="json")

        # Should be valid JSON
        data = json.loads(report)
        assert data["type"] == "tts"
        assert len(data["results"]) == 1

    def test_html_format(self):
        """Test HTML report format."""
        metrics = BenchmarkMetrics(
            backend="test",
            benchmark_type="tts",
            iterations=1,
        )
        metrics.times = [1.0]

        report = format_results([metrics], format_type="html")

        assert "<html>" in report
        assert "<table>" in report
        assert "test" in report


class TestPipelineMetrics:
    """Test E2E pipeline benchmark metrics."""

    def test_pipeline_benchmark_produces_metrics(self):
        """Test PipelineMetrics creation and field access."""
        from scripts.benchmark_pipeline import PipelineMetrics

        m = PipelineMetrics(
            ttfb_ms=450.0,
            e2e_ms=700.0,
            stt_ms=24.0,
            llm_ttft_ms=200.0,
            tts_ms=280.0,
            vram_idle_mb=4096.0,
            vram_peak_mb=6144.0,
        )
        assert m.ttfb_ms == 450.0
        assert m.e2e_ms == 700.0
        assert m.stt_ms == 24.0
        assert m.llm_ttft_ms == 200.0
        assert m.tts_ms == 280.0
        assert m.vram_idle_mb == 4096.0
        assert m.vram_peak_mb == 6144.0

    def test_to_markdown_contains_all_metrics(self):
        """Test that to_markdown produces a table with all metric names and values."""
        from scripts.benchmark_pipeline import PipelineMetrics

        m = PipelineMetrics(
            ttfb_ms=450.0,
            e2e_ms=700.0,
            stt_ms=24.0,
            llm_ttft_ms=200.0,
            tts_ms=280.0,
            vram_idle_mb=4096.0,
            vram_peak_mb=6144.0,
        )
        table = m.to_markdown()

        # Check table structure
        assert "| Metric" in table
        assert "|-----" in table

        # Check all metric labels present
        assert "TTFB" in table
        assert "E2E Latency" in table
        assert "STT" in table
        assert "LLM TTFT" in table
        assert "TTS" in table
        assert "VRAM Idle" in table
        assert "VRAM Peak" in table

        # Check values present
        assert "450" in table
        assert "700" in table
        assert "24" in table
        assert "200" in table
        assert "280" in table
        assert "4096" in table
        assert "6144" in table

    def test_to_json_returns_dict(self):
        """Test that to_json returns a dict with all fields."""
        from scripts.benchmark_pipeline import PipelineMetrics

        m = PipelineMetrics(
            ttfb_ms=450.0,
            e2e_ms=700.0,
            stt_ms=24.0,
            llm_ttft_ms=200.0,
            tts_ms=280.0,
            vram_idle_mb=4096.0,
            vram_peak_mb=6144.0,
        )
        d = m.to_json()

        assert isinstance(d, dict)
        assert d["ttfb_ms"] == 450.0
        assert d["e2e_ms"] == 700.0
        assert d["stt_ms"] == 24.0
        assert d["llm_ttft_ms"] == 200.0
        assert d["tts_ms"] == 280.0
        assert d["vram_idle_mb"] == 4096.0
        assert d["vram_peak_mb"] == 6144.0
        assert len(d) == 7

    def test_to_json_roundtrip(self):
        """Test that to_json output can be serialized to JSON."""
        import json as json_mod
        from scripts.benchmark_pipeline import PipelineMetrics

        m = PipelineMetrics(
            ttfb_ms=123.4,
            e2e_ms=567.8,
            stt_ms=10.0,
            llm_ttft_ms=100.0,
            tts_ms=200.0,
            vram_idle_mb=2048.0,
            vram_peak_mb=3072.0,
        )
        serialized = json_mod.dumps(m.to_json())
        deserialized = json_mod.loads(serialized)
        assert deserialized["ttfb_ms"] == 123.4
        assert deserialized["vram_peak_mb"] == 3072.0

    def test_default_test_phrases(self):
        """Test that benchmark_pipeline has sensible default test phrases."""
        from scripts.benchmark_pipeline import DEFAULT_TEST_PHRASES

        assert len(DEFAULT_TEST_PHRASES) == 5
        assert "What time is it?" in DEFAULT_TEST_PHRASES
        assert "Tell me a joke." in DEFAULT_TEST_PHRASES
