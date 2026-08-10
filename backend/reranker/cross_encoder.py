from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from backend.config.settings import settings
from backend.schemas.document import Chunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.reranker_model_name
        self.device = device or settings.reranker_device
        logger.info("Loading cross-encoder reranker %s on %s", self.model_name, self.device)
        self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int | None = None) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        top_k = top_k or settings.final_top_k
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        scored = sorted(zip(chunks, scores.tolist()), key=lambda x: x[1], reverse=True)
        return scored[:top_k]


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
