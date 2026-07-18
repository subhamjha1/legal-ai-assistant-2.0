"""
Hybrid search service (Milestone 5): Reciprocal Rank Fusion.

Why RRF over a weighted score blend:
Vector search produces cosine similarities (roughly 0-1, dense near the top),
BM25 produces unbounded scores that scale with query length and corpus
statistics. Blending them with fixed weights (e.g. 0.6*vector + 0.4*keyword)
requires normalizing two incomparable distributions, and that normalization
tends to need re-tuning per corpus. RRF sidesteps this entirely: it only
looks at *rank position* in each list, not the raw score value, so it's
robust to two rankers that score on completely different scales - exactly
our situation (BGE/OpenAI cosine similarity vs. BM25).

Formula: for each ranker, a chunk at rank r (1-indexed) contributes
1 / (rrf_k + r) to its fused score. Contributions from every ranker that
returned the chunk are summed; a chunk missing from a ranker's results
simply contributes 0 from that ranker, rather than being penalized further.
"""
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.search import HybridSearchHit
from app.services.indexing import get_cached_embedding_provider, get_cached_vector_store
from app.services.keyword_search import get_keyword_search_provider

logger = get_logger(__name__)


class HybridSearchService:
    """
    Combines vector search (Milestone 3) and keyword/BM25 search
    (Milestone 4) via Reciprocal Rank Fusion.

    Takes explicit vector_store/embedding_provider/keyword_provider
    dependencies (rather than only reaching for the cached singletons
    internally) so tests can inject fakes without touching real models,
    Qdrant, or Elasticsearch.
    """

    def __init__(
        self,
        embedding_provider=None,
        vector_store=None,
        keyword_provider=None,
    ) -> None:
        self.settings = get_settings()
        self.embedding_provider = embedding_provider or get_cached_embedding_provider()
        self.vector_store = vector_store or get_cached_vector_store()
        self.keyword_provider = keyword_provider or get_keyword_search_provider()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> list[HybridSearchHit]:
        top_k = top_k or self.settings.hybrid_search_top_k
        pool_size = self.settings.hybrid_candidate_pool_size

        query_vector = self.embedding_provider.embed_query(query)
        vector_hits = self.vector_store.search(
            query_vector, top_k=pool_size, document_id=document_id, document_type=document_type
        )
        keyword_hits = self.keyword_provider.search(
            query, top_k=pool_size, document_id=document_id, document_type=document_type
        )

        fused = self._reciprocal_rank_fusion(vector_hits, keyword_hits)
        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self, vector_hits: list[dict], keyword_hits: list[dict]
    ) -> list[HybridSearchHit]:
        k = self.settings.rrf_k
        candidates: dict[str, dict] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            chunk_id = hit["chunk_id"]
            entry = candidates.setdefault(chunk_id, {"payload": hit, "rrf_score": 0.0, "matched_by": []})
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["vector_rank"] = rank
            entry["vector_score"] = hit.get("score")
            entry["matched_by"].append("vector")

        for rank, hit in enumerate(keyword_hits, start=1):
            chunk_id = hit["chunk_id"]
            entry = candidates.setdefault(chunk_id, {"payload": hit, "rrf_score": 0.0, "matched_by": []})
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["keyword_rank"] = rank
            entry["keyword_score"] = hit.get("score")
            entry["matched_by"].append("keyword")

        ranked = sorted(candidates.values(), key=lambda e: e["rrf_score"], reverse=True)

        results = []
        for entry in ranked:
            payload = entry["payload"]
            results.append(
                HybridSearchHit(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    text=payload["text"],
                    structural_label=payload.get("structural_label"),
                    page_start=payload["page_start"],
                    page_end=payload["page_end"],
                    original_filename=payload["original_filename"],
                    document_type=payload["document_type"],
                    rrf_score=entry["rrf_score"],
                    vector_rank=entry.get("vector_rank"),
                    keyword_rank=entry.get("keyword_rank"),
                    vector_score=entry.get("vector_score"),
                    keyword_score=entry.get("keyword_score"),
                    matched_by=entry["matched_by"],
                )
            )
        return results
