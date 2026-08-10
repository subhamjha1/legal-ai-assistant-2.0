from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.config.settings import settings

logger = logging.getLogger(__name__)


def plot_metrics_bar_chart(metrics: dict, title: str, out_path: Path) -> Path:
    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != "num_queries"}
    if not numeric_metrics:
        return out_path

    fig, ax = plt.subplots(figsize=(max(6, len(numeric_metrics) * 1.2), 5))
    names = list(numeric_metrics.keys())
    values = list(numeric_metrics.values())
    bars = ax.bar(names, values, color="#2f5d62")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(document_types: list[str], predicted: list[str], golden: list[str], out_path: Path) -> Path:
    idx = {t: i for i, t in enumerate(document_types)}
    matrix = np.zeros((len(document_types), len(document_types)), dtype=int)
    for p, g in zip(predicted, golden):
        if p in idx and g in idx:
            matrix[idx[g]][idx[p]] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(document_types)))
    ax.set_yticks(range(len(document_types)))
    ax.set_xticklabels(document_types, rotation=45, ha="right")
    ax.set_yticklabels(document_types)
    ax.set_xlabel("Predicted (top-1 retrieved) document type")
    ax.set_ylabel("Golden document type")
    ax.set_title("Retrieval Document-Type Confusion Matrix")
    for i in range(len(document_types)):
        for j in range(len(document_types)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_markdown_report(
    retrieval_metrics: dict,
    generation_metrics: dict,
    retrieval_chart_path: Path,
    confusion_matrix_path: Path,
    num_golden_rows: int,
) -> Path:
    settings.eval_report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    report_path = settings.eval_report_dir / f"eval_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"

    lines = [
        f"# Legal & Tax RAG — Evaluation Report",
        f"_Generated {timestamp} — {num_golden_rows} golden queries_",
        "",
        "## Retrieval Metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for k, v in retrieval_metrics.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        f"![Retrieval Metrics]({retrieval_chart_path.name})",
        "",
        "## Generation Quality (RAGAS / DeepEval)",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    if generation_metrics:
        for k, v in generation_metrics.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("| _(RAGAS/DeepEval not run — see logs)_ | — |")

    lines += [
        "",
        "## Document-Type Confusion Matrix (top-1 retrieved chunk)",
        "",
        f"![Confusion Matrix]({confusion_matrix_path.name})",
        "",
        "## Notes",
        "- Recall@K here is binary hit-or-miss per query (golden set has one relevant chunk per query).",
        "- A golden row counts as a 'hit' when the retrieved chunk's filename matches AND the golden "
        "page falls within that chunk's page range.",
        "- RAGAS/DeepEval metrics require live LLM judge calls; if the LLM API key is unset these are skipped.",
    ]

    report_path.write_text("\n".join(lines))
    logger.info("Wrote evaluation report to %s", report_path)
    return report_path
