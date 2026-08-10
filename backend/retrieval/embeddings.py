from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config.settings import settings

logger = logging.getLogger(__name__)

_BGE_QUERY_INSTRUCTION = "Represent this legal question for retrieving relevant legal passages: "


class EmbeddingModel:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device
        logger.info("Loading embedding model %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._is_e5 = "e5" in self.model_name.lower()
        self._is_bge = "bge" in self.model_name.lower()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def _prep_queries(self, texts: list[str]) -> list[str]:
        if self._is_e5:
            return [f"query: {t}" for t in texts]
        if self._is_bge:
            return [_BGE_QUERY_INSTRUCTION + t for t in texts]
        return texts

    def _prep_passages(self, texts: list[str]) -> list[str]:
        if self._is_e5:
            return [f"passage: {t}" for t in texts]
        return texts

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        prepped = self._prep_queries(texts)
        emb = self._model.encode(
            prepped, batch_size=settings.embedding_batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(emb, dtype="float32")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        prepped = self._prep_passages(texts)
        emb = self._model.encode(
            prepped,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return np.asarray(emb, dtype="float32")


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
