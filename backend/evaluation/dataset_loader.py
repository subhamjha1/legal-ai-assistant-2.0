"""
Golden dataset loader (Milestone 9).

Supports two formats:
- JSON: full fidelity, including nested `expected_citations` (a question
  can cite multiple passages, each with its own document/page range).
  This is the recommended format and what golden_dataset.json uses.
- CSV: flat/simplified - one row per question, `relevant_pages` as a
  semicolon-separated list of ints, and a single implicit expected citation
  spanning `source_document` + the full `relevant_pages` range (CSV isn't a
  natural fit for nested structures; this is a deliberate, documented
  limitation, not an oversight - use JSON when a question needs multiple
  distinct cited passages).

Scaling to 100+ questions requires zero code changes: append more objects
to the JSON array (or rows to the CSV) and re-run the same command.
"""
import csv
import json
from pathlib import Path

from evaluation.schema import ExpectedCitation, GoldenQuestion


def load_dataset(path: str | Path) -> list[GoldenQuestion]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    if path.suffix == ".json":
        return _load_json(path)
    if path.suffix == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported dataset format '{path.suffix}'. Use .json or .csv.")


def _load_json(path: Path) -> list[GoldenQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("JSON golden dataset must be a top-level array of question objects.")
    return [GoldenQuestion.model_validate(item) for item in raw]


def _load_csv(path: Path) -> list[GoldenQuestion]:
    questions = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pages = [int(p) for p in row["relevant_pages"].split(";") if p.strip()]
            expected_citations = []
            if pages:
                expected_citations = [
                    ExpectedCitation(
                        document=row["source_document"],
                        page_start=min(pages),
                        page_end=max(pages),
                    )
                ]
            questions.append(
                GoldenQuestion(
                    id=row["id"],
                    query=row["query"],
                    ground_truth_answer=row["ground_truth_answer"],
                    source_document=row["source_document"],
                    relevant_pages=pages,
                    expected_citations=expected_citations,
                    category=row.get("category") or None,
                    notes=row.get("notes") or None,
                )
            )
    return questions


def save_dataset_json(questions: list[GoldenQuestion], path: str | Path) -> None:
    path = Path(path)
    path.write_text(
        json.dumps([q.model_dump(mode="json") for q in questions], indent=2), encoding="utf-8"
    )
