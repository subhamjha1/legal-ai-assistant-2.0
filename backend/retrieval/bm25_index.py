from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.config.settings import settings

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9§]+")

# keep legal words that normal search tokenizers might throw away
_STOPWORDS = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "be", "this", "that"}


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class Bm25Index:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        assert len(chunk_ids) == len(texts)
        self.chunk_ids = list(chunk_ids)
        tokenized_corpus = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index over %d documents", len(chunk_ids))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.chunk_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]

    def save(self, path: Path | None = None) -> None:
        path = path or settings.bm25_index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self._bm25}, f)
        logger.info("Saved BM25 index to %s", path)

    @classmethod
    def load(cls, path: Path | None = None) -> "Bm25Index":
        path = path or settings.bm25_index_path
        index = cls()
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            index.chunk_ids = data["chunk_ids"]
            index._bm25 = data["bm25"]
            logger.info("Loaded BM25 index from %s (%d docs)", path, len(index.chunk_ids))
        else:
            logger.warning("No existing BM25 index at %s — starting empty", path)
        return index
