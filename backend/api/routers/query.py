from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db_session, get_hybrid_retriever
from backend.api.schemas import QueryRequest, QueryResponse
from backend.database.models import FeedbackORM
from backend.graph.graph_retriever import expand_with_graph
from backend.llm.answer_generator import generate_answer
from backend.retrieval.hybrid_retriever import HybridRetriever, RetrievalFilters

router = APIRouter(tags=["query"])


def _to_filters(req: QueryRequest) -> RetrievalFilters:
    if not req.filters:
        return RetrievalFilters()
    f = req.filters
    return RetrievalFilters(document_type=f.document_type, year=f.year, court=f.court, act_name=f.act_name, judge=f.judge)


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
):
    retrieved = await retriever.retrieve(session, req.query, filters=_to_filters(req), top_k=req.top_k)

    extra_graph_chunks = []
    if req.use_graph_expansion and retrieved:
        extra_graph_chunks = await expand_with_graph(session, retrieved)

    result = generate_answer(req.query, retrieved, extra_graph_chunks)
    return QueryResponse(**result)


@router.post("/query/stream")
async def query_stream(
    req: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
):
    from backend.llm.answer_generator import SYSTEM_PROMPT, USER_TEMPLATE, _format_context_block
    from backend.llm.client import get_llm_client

    retrieved = await retriever.retrieve(session, req.query, filters=_to_filters(req), top_k=req.top_k)
    extra_graph_chunks = await expand_with_graph(session, retrieved) if req.use_graph_expansion else []
    context_block = _format_context_block(retrieved, extra_graph_chunks)
    prompt = USER_TEMPLATE.format(query=req.query, context_block=context_block)

    client = get_llm_client()

    def token_generator():
        for delta in client.stream(prompt, system=SYSTEM_PROMPT):
            yield delta

    return StreamingResponse(token_generator(), media_type="text/plain")


@router.post("/feedback")
async def submit_feedback(payload: dict, session: AsyncSession = Depends(get_db_session)):
    session.add(
        FeedbackORM(
            query=payload["query"],
            answer=payload["answer"],
            rating=payload["rating"],
            comment=payload.get("comment"),
            retrieved_chunk_ids=payload.get("retrieved_chunk_ids", []),
        )
    )
    return {"status": "recorded"}
