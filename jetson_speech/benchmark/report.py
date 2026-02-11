"""
Benchmark report generation.
"""

import json
from typing import Any

from jetson_speech.benchmark.metrics import BenchmarkMetrics


def format_results(
    results: list[BenchmarkMetrics],
    format_type: str = "markdown",
) -> str:
    """
    Format benchmark results.

    Args:
        results: List of BenchmarkMetrics
        format_type: "markdown", "json", or "html"

    Returns:
        Formatted report string
    """
    if format_type == "json":
        return _format_json(results)
    elif format_type == "html":
        return _format_html(results)
    else:
        return _format_markdown(results)


def _format_markdown(results: list[BenchmarkMetrics]) -> str:
    """Format as Markdown table."""
    if not results:
        return "No benchmark results."

    benchmark_type = results[0].benchmark_type.upper()

    lines = [
        f"# {benchmark_type} Benchmark Results\n",
        "| Backend | Avg Time | Min | Max | RTF | Memory |",
        "|---------|----------|-----|-----|-----|--------|",
    ]

    for r in results:
        error = r.metadata.get("error")
        if error:
            lines.append(f"| {r.backend} | ERROR | - | - | - | - |")
            continue

        memory = f"{r.memory_delta_mb:.1f} MB"
        if r.gpu_memory_delta_mb > 0:
            memory += f" (+{r.gpu_memory_delta_mb:.1f} GPU)"

        lines.append(
            f"| {r.backend} | "
            f"{r.avg_time:.3f}s | "
            f"{r.min_time:.3f}s | "
            f"{r.max_time:.3f}s | "
            f"{r.avg_rtf:.2f}x | "
            f"{memory} |"
        )

    # Add notes
    lines.append("\n## Notes\n")
    lines.append("- **RTF** (Real-Time Factor): < 1 means faster than real-time")
    lines.append(f"- Iterations per backend: {results[0].iterations}")

    if results[0].benchmark_type == "tts":
        text_len = results[0].metadata.get("text_length", 0)
        lines.append(f"- Text length: {text_len} characters")

        # Add chars/second comparison
        lines.append("\n### Speed (chars/second)\n")
        for r in results:
            if not r.metadata.get("error"):
                lines.append(f"- {r.backend}: {r.chars_per_second:.1f} chars/s")

    elif results[0].benchmark_type == "stt":
        audio_dur = results[0].metadata.get("audio_duration", 0)
        lines.append(f"- Audio duration: {audio_dur:.1f} seconds")

    return "\n".join(lines)


def _format_json(results: list[BenchmarkMetrics]) -> str:
    """Format as JSON."""
    data = {
        "type": results[0].benchmark_type if results else "unknown",
        "results": [r.to_dict() for r in results],
    }
    return json.dumps(data, indent=2)


def _format_html(results: list[BenchmarkMetrics]) -> str:
    """Format as HTML."""
    if not results:
        return "<p>No benchmark results.</p>"

    benchmark_type = results[0].benchmark_type.upper()

    html = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<style>",
        "body { font-family: sans-serif; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        "</style>",
        "</head><body>",
        f"<h1>{benchmark_type} Benchmark Results</h1>",
        "<table>",
        "<tr><th>Backend</th><th>Avg Time</th><th>Min</th><th>Max</th><th>RTF</th><th>Memory</th></tr>",
    ]

    for r in results:
        error = r.metadata.get("error")
        if error:
            html.append(f"<tr><td>{r.backend}</td><td colspan='5'>ERROR: {error}</td></tr>")
            continue

        memory = f"{r.memory_delta_mb:.1f} MB"
        if r.gpu_memory_delta_mb > 0:
            memory += f" (+{r.gpu_memory_delta_mb:.1f} GPU)"

        html.append(
            f"<tr>"
            f"<td>{r.backend}</td>"
            f"<td>{r.avg_time:.3f}s</td>"
            f"<td>{r.min_time:.3f}s</td>"
            f"<td>{r.max_time:.3f}s</td>"
            f"<td>{r.avg_rtf:.2f}x</td>"
            f"<td>{memory}</td>"
            f"</tr>"
        )

    html.extend([
        "</table>",
        "<p><small>RTF (Real-Time Factor): &lt; 1 means faster than real-time</small></p>",
        "</body></html>",
    ])

    return "\n".join(html)


def save_report(
    results: list[BenchmarkMetrics],
    path: str,
    format_type: str | None = None,
) -> None:
    """
    Save benchmark report to file.

    Args:
        results: List of BenchmarkMetrics
        path: Output file path
        format_type: Format (auto-detected from extension if None)
    """
    if format_type is None:
        if path.endswith(".json"):
            format_type = "json"
        elif path.endswith(".html"):
            format_type = "html"
        else:
            format_type = "markdown"

    report = format_results(results, format_type)

    with open(path, "w") as f:
        f.write(report)
