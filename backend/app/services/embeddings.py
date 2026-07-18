"""
Embedding providers (Milestone 3).

Why one interface, two providers:
BGE (BAAI/bge-large-en-v1.5) runs locally via sentence-transformers - free,
no API key, no per-call latency to a third party, but requires downloading
~1.3GB of model weights once and running inference on this machine's CPU/GPU.
text-embedding-3-large runs via the OpenAI API - no local compute or download,
generally strong quality, but costs money per token and adds network latency.

Neither is strictly better for every deployment, so both are built behind
`EmbeddingProvider`. Swapping providers is a one-line config change
(`EMBEDDING_PROVIDER=bge|openai`), and nothing in the vector store or
retrieval layer needs to know which one is active - it only needs vectors of
a known, fixed dimension.
"""
from abc import ABC, abstractmethod

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Common interface every embedding backend must implement."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality this provider produces. The vector store
        collection is created with this exact size, so it must be accurate
        and stable for a given provider/model."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text, in
        the same order."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query. Kept separate from embed_texts
        because some models (BGE included) recommend a distinct instruction
        prefix for queries vs. documents to improve retrieval quality."""


class BGEEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding via BAAI/bge-large-en-v1.5 (sentence-transformers).

    BGE models are trained with an asymmetric convention: document chunks are
    embedded as-is, but queries should be prefixed with an instruction
    ("Represent this sentence for searching relevant passages: ") for best
    retrieval accuracy. We apply that prefix only in embed_query.
    """

    _QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None  # lazy-loaded: avoids the ~1.3GB download/load
        # cost at import time, e.g. for tests that never call embed_texts.

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading BGE model '%s' (first use; this downloads weights if not cached)...", self.settings.bge_model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.settings.bge_model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.settings.bge_embedding_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,  # required for cosine similarity in Qdrant
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([self._QUERY_INSTRUCTION + query])[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding via OpenAI's text-embedding-3-large API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to use the 'openai' embedding provider. "
                "Either set it in .env or switch EMBEDDING_PROVIDER to 'bge'."
            )
        from openai import OpenAI
        self._client = OpenAI(api_key=self.settings.openai_api_key)

    @property
    def dimension(self) -> int:
        return self.settings.openai_embedding_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # OpenAI's embeddings endpoint accepts batches directly; no manual
        # batching loop needed for the sizes this system deals with, but we
        # still chunk defensively for very large chunk sets.
        results: list[list[float]] = []
        batch_size = self.settings.embedding_batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
            )
            results.extend(item.embedding for item in response.data)
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic, dependency-free, no-network embedding using the
    "hashing trick" over character trigrams. Semantically much weaker than
    BGE or OpenAI - it captures shared word-roots and surface overlap, not
    true meaning - but needs no model download and no API key, making it a
    genuinely useful offline fallback for CI, demos, and network-restricted
    environments (like this project's own development sandbox, which
    cannot reach HuggingFace or OpenAI - see README). Hybrid search's BM25
    side compensates significantly for its weaker semantic signal, since
    exact-term precision doesn't depend on embedding quality at all.

    NOT recommended for production retrieval quality - use 'bge' or
    'openai' whenever the environment has real network access.
    """

    def __init__(self, dim: int = 512) -> None:
        self.settings = get_settings()
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)

    def _embed_one(self, text: str) -> list[float]:
        import hashlib
        import math
        import re

        vec = [0.0] * self._dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            grams = [token[i : i + 3] for i in range(max(len(token) - 2, 1))] or [token]
            for gram in grams:
                digest = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                index = digest % self._dim
                sign = 1.0 if (digest // self._dim) % 2 == 0 else -1.0
                vec[index] += sign

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def get_embedding_provider(provider_name: str | None = None) -> EmbeddingProvider:
    """Factory selecting the configured provider. Explicit `provider_name`
    override is mainly for tests; production code should rely on the
    EMBEDDING_PROVIDER env var."""
    settings = get_settings()
    name = provider_name or settings.embedding_provider

    if name == "bge":
        return BGEEmbeddingProvider()
    if name == "openai":
        return OpenAIEmbeddingProvider()
    if name == "hash":
        return HashEmbeddingProvider()
    raise ValueError(f"Unknown embedding provider '{name}'. Expected 'bge', 'openai', or 'hash'.")
