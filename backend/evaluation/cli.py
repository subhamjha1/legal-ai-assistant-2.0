"""
Evaluation CLI (Milestone 9).

Single-command usage:
    python -m evaluation.cli --dataset evaluation/golden_dataset.json --output-dir evaluation/results

CI usage (non-zero exit code on quality regression):
    python -m evaluation.cli --dataset evaluation/golden_dataset.json \\
        --min-answer-correctness 0.6 --min-faithfulness 0.7
"""
import argparse
import json
import sys
from pathlib import Path

from evaluation.dataset_loader import load_dataset
from evaluation.report import generate_html_report, generate_markdown_report
from evaluation.runner import run_evaluation
from evaluation.schema import EvaluationConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden-dataset evaluation harness.")
    parser.add_argument("--dataset", required=True, help="Path to a .json or .csv golden dataset.")
    parser.add_argument("--output-dir", default="evaluation/results", help="Where to write reports.")
    parser.add_argument("--top-k", type=int, default=5, help="top_k passed to the QA pipeline.")
    parser.add_argument(
        "--retrieval-k", type=int, default=5, help="K used for Recall@K / nDCG@K."
    )
    parser.add_argument(
        "--min-answer-correctness",
        type=float,
        default=0.0,
        help="CI gate: exit 1 if avg_answer_correctness falls below this (0.0 = no gate).",
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.0,
        help="CI gate: exit 1 if avg_faithfulness falls below this (0.0 = no gate).",
    )
    args = parser.parse_args(argv)

    try:
        questions = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        return 1

    if not questions:
        print(f"No questions found in {args.dataset}", file=sys.stderr)
        return 1

    config = EvaluationConfig(
        top_k=args.top_k,
        retrieval_k_for_recall=args.retrieval_k,
        min_avg_answer_correctness=args.min_answer_correctness,
        min_avg_faithfulness=args.min_faithfulness,
    )

    print(f"Running evaluation: {len(questions)} questions from {args.dataset}...")
    summary = run_evaluation(questions, config=config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report.md").write_text(generate_markdown_report(summary), encoding="utf-8")
    (output_dir / "report.html").write_text(generate_html_report(summary), encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    print(f"\nReports written to {output_dir}/ (report.md, report.html, results.json)")
    print("\nAggregate metrics:")
    for key in sorted(summary.aggregate_metrics):
        value = summary.aggregate_metrics[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # CI gates - only enforced if explicitly set above 0.0.
    failures = []
    avg_correctness = summary.aggregate_metrics.get("avg_answer_correctness", 0.0)
    avg_faithfulness = summary.aggregate_metrics.get("avg_faithfulness", 0.0)

    if config.min_avg_answer_correctness > 0.0 and avg_correctness < config.min_avg_answer_correctness:
        failures.append(
            f"avg_answer_correctness {avg_correctness:.4f} < required {config.min_avg_answer_correctness:.4f}"
        )
    if config.min_avg_faithfulness > 0.0 and avg_faithfulness < config.min_avg_faithfulness:
        failures.append(
            f"avg_faithfulness {avg_faithfulness:.4f} < required {config.min_avg_faithfulness:.4f}"
        )

    if failures:
        print("\nCI GATE FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nEvaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
