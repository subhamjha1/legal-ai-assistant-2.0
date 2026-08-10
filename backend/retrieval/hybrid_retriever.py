from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import settings
from backend.database.repository import ChunkRepository
from backend.reranker.cross_encoder import get_reranker
from backend.retrieval.bm25_index import Bm25Index
from backend.retrieval.context_compression import compress_context
from backend.retrieval.elasticsearch_store import get_elasticsearch_store
from backend.retrieval.embeddings import get_embedding_model
from backend.retrieval.faiss_store import FaissVectorStore
from backend.retrieval.query_expansion import build_lexical_query_string, expand_query, generate_multi_queries
from backend.retrieval.rrf import reciprocal_rank_fusion
from backend.schemas.document import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalFilters:
    document_type: str | None = None
    year: int | None = None
    court: str | None = None
    act_name: str | None = None
    judge: str | None = None

    def to_es_filters(self) -> dict:
        return {
            "document_type": self.document_type,
            "year": self.year,
            "court": self.court,
            "act_name": self.act_name,
        }


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    parent_text: str | None = None
    neighbor_chunk_ids: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, faiss_store: FaissVectorStore, bm25_index: Bm25Index):
        self.faiss_store = faiss_store
        self.bm25_index = bm25_index
        self.embedding_model = get_embedding_model()
        self.es_store = get_elasticsearch_store()
        self.reranker = get_reranker()

    async def retrieve(
        self,
        session: AsyncSession,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.final_top_k
        filters = filters or RetrievalFilters()

        expansion_terms = expand_query(query)
        multi_queries = generate_multi_queries(query)
        semantic_query_variants = [query] + multi_queries
        lexical_query = build_lexical_query_string(query, expansion_terms)

        faiss_ranked_lists: list[list[tuple[str, float]]] = []
        query_vectors = self.embedding_model.embed_queries(semantic_query_variants)
        for i in range(len(semantic_query_variants)):
            results = self.faiss_store.search(query_vectors[i], top_k=settings.faiss_top_k)
            faiss_ranked_lists.append(results)

        bm25_results = self.bm25_index.search(lexical_query, top_k=settings.bm25_top_k)

        es_results = self.es_store.search(
            lexical_query, top_k=settings.elasticsearch_top_k, filters=filters.to_es_filters()
        )

        all_ranked_lists = faiss_ranked_lists + [bm25_results, es_results]
        all_ranked_lists = [rl for rl in all_ranked_lists if rl]
        if not all_ranked_lists:
            logger.warning("All retrieval backends returned empty results for query %r", query)
            return []

        fused = reciprocal_rank_fusion(all_ranked_lists)
        candidate_ids = [cid for cid, _ in fused[: settings.fusion_top_k]]

        chunk_repo = ChunkRepository(session)
        chunk_map = await chunk_repo.get_by_ids(candidate_ids)
        candidates = [chunk_map[cid] for cid in candidate_ids if cid in chunk_map]
        if not candidates:
            logger.warning("No candidate chunks hydrated from Postgres for query %r", query)
            return []

        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        compressed = compress_context(reranked)

        enriched: list[RetrievedChunk] = []
        for chunk, score in compressed:
            neighbors = await chunk_repo.get_neighbors(
                chunk.document_id, chunk.chunk_index, settings.neighbor_window
            )
            neighbor_ids = [n.chunk_id for n in neighbors if n.chunk_id != chunk.chunk_id]

            parent_text = None
            if chunk.parent_chunk_id:
                parent = await chunk_repo.get_parent(chunk.parent_chunk_id)
                if parent:
                    parent_text = parent.text

            enriched.append(
                RetrievedChunk(chunk=chunk, score=score, parent_text=parent_text, neighbor_chunk_ids=neighbor_ids)
            )

        logger.info(
            "Retrieved %d final chunks for query %r (fused_candidates=%d)",
            len(enriched),
            query,
            len(candidate_ids),
        )
        return enriched
