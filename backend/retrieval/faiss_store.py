from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from backend.config.settings import settings

logger = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(self, dimension: int, index_dir: Path | None = None):
        self.dimension = dimension
        self.index_dir = Path(index_dir or settings.faiss_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.index_dir / "index.faiss"
        self._ids_path = self.index_dir / "id_to_chunk_id.json"

        base_index = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap2(base_index)
        self.id_to_chunk_id: dict[int, str] = {}
        self._next_id = 0

    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        assert vectors.shape[0] == len(chunk_ids)
        assert vectors.shape[1] == self.dimension
        int_ids = np.arange(self._next_id, self._next_id + len(chunk_ids), dtype="int64")
        self.index.add_with_ids(vectors, int_ids)
        for iid, cid in zip(int_ids.tolist(), chunk_ids):
            self.id_to_chunk_id[iid] = cid
        self._next_id += len(chunk_ids)
        logger.info("Added %d vectors to FAISS (total=%d)", len(chunk_ids), self.index.ntotal)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        query_vector = query_vector.reshape(1, -1).astype("float32")
        scores, ids = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for score, iid in zip(scores[0].tolist(), ids[0].tolist()):
            if iid == -1:
                continue
            chunk_id = self.id_to_chunk_id.get(iid)
            if chunk_id:
                results.append((chunk_id, float(score)))
        return results

    def save(self) -> None:
        faiss.write_index(self.index, str(self._index_path))
        with open(self._ids_path, "w") as f:
            json.dump({"next_id": self._next_id, "id_to_chunk_id": self.id_to_chunk_id}, f)
        logger.info("Saved FAISS index to %s (%d vectors)", self._index_path, self.index.ntotal)

    @classmethod
    def load(cls, dimension: int, index_dir: Path | None = None) -> "FaissVectorStore":
        store = cls(dimension, index_dir)
        if store._index_path.exists() and store._ids_path.exists():
            store.index = faiss.read_index(str(store._index_path))
            with open(store._ids_path) as f:
                data = json.load(f)
            store._next_id = data["next_id"]
            store.id_to_chunk_id = {int(k): v for k, v in data["id_to_chunk_id"].items()}
            logger.info("Loaded FAISS index from %s (%d vectors)", store._index_path, store.index.ntotal)
        else:
            logger.warning("No existing FAISS index found at %s — starting empty", store.index_dir)
        return store
