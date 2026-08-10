from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.config.settings import settings


@dataclass
class GoldenRow:
    query: str
    ground_truth: str
    document: str
    page: int


def load_golden_dataset(path: Path | None = None) -> list[GoldenRow]:
    path = path or settings.golden_dataset_path
    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {path}. Create a CSV with columns: "
            "query, ground_truth, document, page. See data/golden/golden_dataset.example.csv."
        )
    df = pd.read_csv(path)
    required = {"query", "ground_truth", "document", "page"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Golden dataset missing required columns: {missing}")

    return [
        GoldenRow(query=row["query"], ground_truth=row["ground_truth"], document=row["document"], page=int(row["page"]))
        for _, row in df.iterrows()
    ]
