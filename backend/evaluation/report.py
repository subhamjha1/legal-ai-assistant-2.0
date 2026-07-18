"""
Evaluation report generation (Milestone 9).

Charts are hand-rolled inline SVG rather than matplotlib output, on
purpose: matplotlib needs a rendering backend (fiddly in headless CI
containers) and produces raster images that bloat the report file. Plain
SVG strings need no rendering library at all, stay crisp at any zoom level,
and keep the whole HTML report a single self-contained file - easy to
upload as a CI artifact or drop into a PR comment.
"""
from datetime import datetime

from evaluation.schema import EvaluationSummary

_CHART_METRIC_ORDER = [
    "avg_answer_correctness",
    "avg_citation_precision",
    "avg_citation_recall",
    "avg_mrr",
    "avg_faithfulness",
]


def generate_markdown_report(summary: EvaluationSummary) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- **Run started:** {summary.run_started_at}",
        f"- **Run finished:** {summary.run_finished_at}",
        f"- **Model:** {summary.model_used}",
        f"- **Total questions:** {summary.total_questions}",
        f"- **Errored questions:** {int(summary.aggregate_metrics.get('errored_questions', 0))}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in sorted(summary.aggregate_metrics):
        value = summary.aggregate_metrics[key]
        lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")

    lines += ["", "## Per-Question Results", ""]
    lines += [
        "| ID | Category | Answer Correctness | Citation P | Citation R | MRR | Faithfulness | Latency (s) | Error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.results:
        m = r.metrics
        lines.append(
            f"| {r.question_id} | {r.category or '-'} | "
            f"{m.get('answer_correctness', 0):.2f} | {m.get('citation_precision', 0):.2f} | "
            f"{m.get('citation_recall', 0):.2f} | {m.get('mrr', 0):.2f} | "
            f"{m.get('faithfulness', 0):.2f} | {m.get('latency_seconds', 0):.2f} | "
            f"{'⚠️ ' + r.error if r.error else ''} |"
        )

    lines += ["", "## Notes on Metric Rigor", ""]
    lines += [
        "- **Answer correctness** uses token-F1 (SQuAD-style), a deterministic stand-in for an "
        "LLM-judge or semantic-similarity metric. See `evaluation/metrics.py` docstrings for the "
        "full rationale.",
        "- **Faithfulness** checks that each answer sentence carries a citation tag - it does NOT "
        "verify the citation's claim is actually entailed by the source text (that would need an "
        "LLM judge, as in RAGAS's real faithfulness metric).",
        "- Citation precision/recall use page-**range overlap** matching against the golden "
        "dataset's `expected_citations`, not exact equality.",
    ]

    return "\n".join(lines)


def generate_html_report(summary: EvaluationSummary) -> str:
    chart_svg = _bar_chart(summary.aggregate_metrics)
    latency_svg = _latency_chart(summary.results)

    rows = "\n".join(
        f"<tr>"
        f"<td>{r.question_id}</td><td>{r.category or '-'}</td>"
        f"<td>{r.metrics.get('answer_correctness', 0):.2f}</td>"
        f"<td>{r.metrics.get('citation_precision', 0):.2f}</td>"
        f"<td>{r.metrics.get('citation_recall', 0):.2f}</td>"
        f"<td>{r.metrics.get('mrr', 0):.2f}</td>"
        f"<td>{r.metrics.get('faithfulness', 0):.2f}</td>"
        f"<td>{r.metrics.get('latency_seconds', 0):.2f}</td>"
        f"<td class='err'>{r.error or ''}</td>"
        f"</tr>"
        for r in summary.results
    )

    metric_rows = "\n".join(
        f"<tr><td>{key}</td><td>{value:.4f}</td></tr>" if isinstance(value, float) else f"<tr><td>{key}</td><td>{value}</td></tr>"
        for key, value in sorted(summary.aggregate_metrics.items())
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, "Public Sans", Arial, sans-serif; background: #0b0d10; color: #e8e6df; padding: 32px; max-width: 1000px; margin: 0 auto; }}
  h1, h2 {{ font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; }}
  th, td {{ border: 1px solid #262b33; padding: 6px 10px; font-size: 13px; text-align: left; }}
  th {{ background: #1b1f26; }}
  .err {{ color: #c0563d; }}
  .meta {{ color: #9a988f; font-size: 13px; margin-bottom: 24px; }}
  .chart-wrap {{ background: #14171c; border: 1px solid #262b33; border-radius: 8px; padding: 16px; margin-bottom: 32px; }}
</style>
</head>
<body>
  <h1>Evaluation Report</h1>
  <p class="meta">
    Run: {summary.run_started_at} &rarr; {summary.run_finished_at}<br>
    Model: {summary.model_used} &middot; Questions: {summary.total_questions} &middot;
    Errored: {int(summary.aggregate_metrics.get('errored_questions', 0))}
  </p>

  <h2>Aggregate Metrics</h2>
  <div class="chart-wrap">{chart_svg}</div>
  <table><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>

  <h2>Latency Distribution</h2>
  <div class="chart-wrap">{latency_svg}</div>

  <h2>Per-Question Results</h2>
  <table>
    <tr><th>ID</th><th>Category</th><th>Answer Correctness</th><th>Citation P</th><th>Citation R</th><th>MRR</th><th>Faithfulness</th><th>Latency (s)</th><th>Error</th></tr>
    {rows}
  </table>

  <p class="meta">Generated {datetime.now().isoformat()}</p>
</body>
</html>"""


def _bar_chart(aggregate_metrics: dict[str, float], width: int = 640, height: int = 220) -> str:
    values = [(key.replace("avg_", ""), aggregate_metrics.get(key, 0.0)) for key in _CHART_METRIC_ORDER if key in aggregate_metrics]
    if not values:
        return "<p>No metrics to chart.</p>"

    bar_width = width / (len(values) * 2)
    bars = []
    labels = []
    for i, (label, value) in enumerate(values):
        x = i * 2 * bar_width + bar_width * 0.5
        bar_height = max(value, 0.01) * (height - 40)
        y = height - 30 - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
            f'fill="#c9a227" rx="2" />'
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" fill="#e8e6df" font-size="11" '
            f'text-anchor="middle" font-family="IBM Plex Mono, monospace">{value:.2f}</text>'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 10}" fill="#9a988f" font-size="10" '
            f'text-anchor="middle" font-family="Public Sans, sans-serif">{label}</text>'
        )

    baseline = f'<line x1="0" y1="{height - 30}" x2="{width}" y2="{height - 30}" stroke="#262b33" />'
    return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{baseline}{"".join(bars)}{"".join(labels)}</svg>'


def _latency_chart(results: list, width: int = 640, height: int = 160) -> str:
    if not results:
        return "<p>No latency data.</p>"

    latencies = [r.metrics.get("latency_seconds", 0.0) for r in results]
    max_latency = max(latencies) or 1.0
    bar_width = width / (len(latencies) * 1.4)

    bars = []
    for i, latency in enumerate(latencies):
        x = i * 1.4 * bar_width
        bar_height = (latency / max_latency) * (height - 30)
        y = height - 20 - bar_height
        color = "#c0563d" if latency > max_latency * 0.8 else "#4c8a71"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="1" />'
        )

    baseline = f'<line x1="0" y1="{height - 20}" x2="{width}" y2="{height - 20}" stroke="#262b33" />'
    axis_label = (
        f'<text x="4" y="{height - 4}" fill="#9a988f" font-size="10" '
        f'font-family="Public Sans, sans-serif">max: {max_latency:.2f}s</text>'
    )
    return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{baseline}{"".join(bars)}{axis_label}</svg>'
