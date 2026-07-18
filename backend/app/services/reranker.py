"""
Re-ranking providers (Milestone 6).

Why one interface, two providers:
BAAI/bge-reranker-large is a cross-encoder - it jointly encodes the query
and each candidate passage together, letting it capture semantic relevance
a bi-encoder (embedding similarity) or BM25 can miss (e.g. that a passage
*answers* a question rather than merely sharing vocabulary with it). It's
the higher-quality option, but - like BGE embeddings - it requires
downloading model weights from HuggingFace, which this development sandbox
cannot reach.

Rather than ship untested rerank code, `LightweightReranker` provides a
genuinely-testable, dependency-free stand-in: it blends each candidate's
(min-max normalized) RRF score from hybrid search with its query-term
overlap. This is a real, defensible heuristic - not a placeholder that
returns nothing - but it is NOT a substitute for a cross-encoder's semantic
judgment, and should be swapped for `cross_encoder` once you can reach
HuggingFace.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


@dataclass
class RerankResult:
    index: int  # index into the original candidates list passed to rerank()
    confidence: float  # 0-1, higher is more relevant


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidate_texts: list[str], candidate_scores: list[float] | None = None) -> list[RerankResult]:
        """
        Score and order candidates by relevance to `query`.

        `candidate_scores`, if provided, are each candidate's upstream
        relevance signal (e.g. RRF score) - implementations may use this as
        an additional feature (LightweightReranker does; a real
        cross-encoder typically doesn't need to).

        Returns RerankResult objects sorted by confidence, descending.
        """


class LightweightReranker(Reranker):
    """
    Dependency-free re-ranking stand-in: blends normalized upstream
    relevance score with query-term overlap (Jaccard similarity between
    query tokens and candidate tokens).

    This directly addresses one real weakness of raw RRF order: RRF ranks
    purely by *position* in each ranker's list, with no sense of *how much*
    better one match is over another, and no direct signal about literal
    term overlap once both rankers are already fused. Blending in Jaccard
    overlap gives a cheap, real boost to candidates that literally contain
    the query's key terms (e.g. a specific section number), on top of
    whatever upstream ranking they already had - without needing a model
    download.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def rerank(
        self, query: str, candidate_texts: list[str], candidate_scores: list[float] | None = None
    ) -> list[RerankResult]:
        if not candidate_texts:
            return []

        alpha = self.settings.lightweight_rerank_alpha
        query_tokens = _tokenize(query)

        scores = candidate_scores or [0.0] * len(candidate_texts)
        normalized_scores = self._min_max_normalize(scores)

        results = []
        for i, text in enumerate(candidate_texts):
            overlap = self._jaccard(query_tokens, _tokenize(text))
            confidence = alpha * normalized_scores[i] + (1 - alpha) * overlap
            results.append(RerankResult(index=i, confidence=confidence))

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0

    @staticmethod
    def _min_max_normalize(values: list[float]) -> list[float]:
        if not values:
            return values
        lo, hi = min(values), max(values)
        if hi == lo:
            return [1.0 for _ in values]  # all equal -> no information to differentiate, treat as max
        return [(v - lo) / (hi - lo) for v in values]


class CrossEncoderReranker(Reranker):
    """
    Real cross-encoder re-ranking via BAAI/bge-reranker-large.

    NOTE: requires downloading model weights from HuggingFace on first use.
    This is complete production code but could not be exercised in this
    development sandbox (no network path to huggingface.co - see Milestone 3
    for the same constraint on BGE embeddings). Verify on a machine with
    normal internet access:
        RERANKER_PROVIDER=cross_encoder pytest tests/test_reranker.py
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None  # lazy-loaded

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading cross-encoder reranker '%s' (first use)...", self.settings.reranker_model_name)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.settings.reranker_model_name)
        return self._model

    def rerank(
        self, query: str, candidate_texts: list[str], candidate_scores: list[float] | None = None
    ) -> list[RerankResult]:
        if not candidate_texts:
            return []

        pairs = [[query, text] for text in candidate_texts]
        raw_scores = self.model.predict(pairs)  # typically unbounded logits

        # Squash to 0-1 via sigmoid so downstream confidence thresholding
        # (rerank_min_confidence) behaves consistently regardless of provider.
        import math
        confidences = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]

        results = [RerankResult(index=i, confidence=c) for i, c in enumerate(confidences)]
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results


def get_reranker(provider_name: str | None = None) -> Reranker:
    settings = get_settings()
    name = provider_name or settings.reranker_provider

    if name == "lightweight":
        return LightweightReranker()
    if name == "cross_encoder":
        return CrossEncoderReranker()
    raise ValueError(f"Unknown reranker provider '{name}'. Expected 'lightweight' or 'cross_encoder'.")
