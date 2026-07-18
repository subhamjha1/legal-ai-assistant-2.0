"""
Retrieval service (Milestone 6): the final retrieval-quality gate before
generation.

Pipeline:
    hybrid search (Milestone 5, wide candidate pool)
        -> MMR selection (diversify: drop near-duplicate chunks)
        -> re-ranking (score genuine query-relevance, not just fusion rank)
        -> confidence filtering (drop anything below rerank_min_confidence)

Why this order:
Hybrid search casts a wide net (mmr_pool_size candidates) so nothing
relevant is missed at the fusion stage. MMR then narrows that pool down
while actively avoiding redundancy - there's no point re-ranking 5 chunks
that all say the same thing. Re-ranking then scores true relevance on the
now-diverse, now-smaller set (re-ranking is more expensive per-candidate
than fusion, so doing it after MMR narrows the pool keeps cost bounded).
Confidence filtering last means the LLM (Milestone 7) never receives a
context chunk this pipeline itself doesn't trust - directly serving the
"minimize hallucination" requirement.
"""
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.search import RetrievedChunk
from app.services.hybrid_search import HybridSearchService
from app.services.indexing import get_cached_embedding_provider
from app.services.mmr import mmr_select
from app.services.reranker import Reranker, get_reranker

logger = get_logger(__name__)


class RetrievalService:
    def __init__(
        self,
        hybrid_search_service: HybridSearchService | None = None,
        reranker: Reranker | None = None,
        embedding_provider=None,
    ) -> None:
        self.settings = get_settings()
        self.hybrid_search = hybrid_search_service or HybridSearchService()
        self.reranker = reranker or get_reranker()
        self.embedding_provider = embedding_provider or get_cached_embedding_provider()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> tuple[list[RetrievedChunk], int, int]:
        """Returns (results, candidates_considered, dropped_for_low_confidence)."""
        final_top_k = top_k or self.settings.rerank_final_top_k

        # 1. Hybrid search - wide candidate pool.
        candidates = self.hybrid_search.search(
            query, top_k=self.settings.mmr_pool_size, document_id=document_id, document_type=document_type
        )
        candidates_considered = len(candidates)
        if not candidates:
            return [], 0, 0

        # 2. MMR diversification. Needs embeddings for each candidate; we
        # re-embed candidate texts here rather than trying to recover
        # per-hit vectors from Qdrant (which would require an extra
        # with_vectors round trip and wouldn't cover keyword-only hits
        # anyway, since BM25 hits have no vector at all).
        query_vector = self.embedding_provider.embed_query(query)
        candidate_vectors = self.embedding_provider.embed_texts([c.text for c in candidates])

        mmr_indices = mmr_select(
            query_vector=query_vector,
            candidate_vectors=candidate_vectors,
            top_k=self.settings.retriever_top_k,
            lambda_param=self.settings.mmr_lambda,
        )
        diversified = [candidates[i] for i in mmr_indices]

        # 3. Re-ranking.
        rerank_results = self.reranker.rerank(
            query=query,
            candidate_texts=[c.text for c in diversified],
            candidate_scores=[c.rrf_score for c in diversified],
        )

        # 4. Confidence filtering + shaping into the final response schema.
        results: list[RetrievedChunk] = []
        dropped = 0
        for rr in rerank_results:
            if rr.confidence < self.settings.rerank_min_confidence:
                dropped += 1
                continue
            chunk = diversified[rr.index]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    structural_label=chunk.structural_label,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    original_filename=chunk.original_filename,
                    document_type=chunk.document_type,
                    confidence=rr.confidence,
                    rrf_score=chunk.rrf_score,
                )
            )
            if len(results) >= final_top_k:
                break

        logger.info(
            "Retrieval for query '%s...': %d candidates -> %d after MMR -> %d final (dropped %d below confidence %.2f)",
            query[:50],
            candidates_considered,
            len(diversified),
            len(results),
            dropped,
            self.settings.rerank_min_confidence,
        )
        return results, candidates_considered, dropped
