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
