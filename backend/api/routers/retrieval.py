from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db_session, get_hybrid_retriever
from backend.api.routers.query import _to_filters
from backend.api.schemas import DocumentOut, QueryRequest, RetrieveResponse, SummarizeRequest, SummarizeResponse
from backend.database.repository import ChunkRepository, DocumentRepository
from backend.llm.summarizer import map_reduce_summarize
from backend.retrieval.hybrid_retriever import HybridRetriever

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=list[RetrieveResponse])
async def retrieve(
    req: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
):
    retrieved = await retriever.retrieve(session, req.query, filters=_to_filters(req), top_k=req.top_k)
    return [
        RetrieveResponse(
            chunk_id=rc.chunk.chunk_id,
            filename=rc.chunk.filename,
            document_type=rc.chunk.document_type.value,
            page_start=rc.chunk.page_start,
            page_end=rc.chunk.page_end,
            section=rc.chunk.section,
            heading=rc.chunk.heading,
            text=rc.chunk.text,
            score=rc.score,
        )
        for rc in retrieved
    ]


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(document_type: str | None = None, session: AsyncSession = Depends(get_db_session)):
    repo = DocumentRepository(session)
    docs = await repo.list_all(document_type=document_type)
    return [
        DocumentOut(
            document_id=d.document_id,
            filename=d.filename,
            title=d.title,
            document_type=d.document_type.value,
            year=d.year,
            court=d.court,
            page_count=d.page_count,
        )
        for d in docs
    ]


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest, session: AsyncSession = Depends(get_db_session)):
    doc_repo = DocumentRepository(session)
    chunk_repo = ChunkRepository(session)

    doc = await doc_repo.get(req.document_id)
    if not doc:
        raise HTTPException(404, f"Document {req.document_id} not found")

    chunks = await chunk_repo.get_all_children(req.document_id)
    if not chunks:
        raise HTTPException(404, f"No chunks found for document {req.document_id}")

    summary = map_reduce_summarize(doc.title, chunks, max_summary_words=req.max_summary_words)
    return SummarizeResponse(document_id=doc.document_id, title=doc.title, summary=summary)
