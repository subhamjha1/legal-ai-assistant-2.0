from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import evaluation, health, ingestion, query, retrieval
from backend.config.settings import settings
from backend.database.session import init_db
from backend.retrieval.bm25_index import Bm25Index
from backend.retrieval.embeddings import get_embedding_model
from backend.retrieval.faiss_store import FaissVectorStore
from backend.retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.app_name, settings.environment)

    await init_db()

    embedder = get_embedding_model()
    faiss_store = FaissVectorStore.load(dimension=embedder.dimension)
    bm25_index = Bm25Index.load()

    app.state.faiss_store = faiss_store
    app.state.bm25_index = bm25_index
    app.state.hybrid_retriever = HybridRetriever(faiss_store, bm25_index)

    logger.info(
        "Startup complete: faiss_vectors=%d bm25_docs=%d",
        faiss_store.index.ntotal,
        len(bm25_index.chunk_ids),
    )
    yield

    faiss_store.save()
    bm25_index.save()
    logger.info("Shutdown complete — indices persisted")


app = FastAPI(
    title=settings.app_name,
    description="Production Legal & Tax RAG system for US legal documents with hybrid retrieval, "
    "GraphRAG, and citation-grounded answer generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(ingestion.router, prefix=settings.api_prefix)
app.include_router(query.router, prefix=settings.api_prefix)
app.include_router(retrieval.router, prefix=settings.api_prefix)
app.include_router(evaluation.router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {"service": settings.app_name, "docs": "/docs", "api_prefix": settings.api_prefix}
