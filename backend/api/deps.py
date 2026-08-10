from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import SessionLocal
from backend.retrieval.bm25_index import Bm25Index
from backend.retrieval.faiss_store import FaissVectorStore
from backend.retrieval.hybrid_retriever import HybridRetriever


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_faiss_store(request: Request) -> FaissVectorStore:
    return request.app.state.faiss_store


def get_bm25_index(request: Request) -> Bm25Index:
    return request.app.state.bm25_index


def get_hybrid_retriever(request: Request) -> HybridRetriever:
    return request.app.state.hybrid_retriever
