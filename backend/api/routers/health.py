from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.deps import get_bm25_index, get_faiss_store
from backend.api.schemas import HealthResponse
from backend.config.settings import settings
from backend.retrieval.bm25_index import Bm25Index
from backend.retrieval.elasticsearch_store import get_elasticsearch_store
from backend.retrieval.faiss_store import FaissVectorStore
from backend.graph.neo4j_client import get_graph_store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    faiss_store: FaissVectorStore = Depends(get_faiss_store),
    bm25_index: Bm25Index = Depends(get_bm25_index),
):
    return HealthResponse(
        status="ok",
        faiss_vectors=faiss_store.index.ntotal,
        bm25_documents=len(bm25_index.chunk_ids),
        elasticsearch_enabled=get_elasticsearch_store().enabled,
        neo4j_enabled=get_graph_store().enabled,
    )
