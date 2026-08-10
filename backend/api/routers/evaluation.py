from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db_session, get_hybrid_retriever
from backend.evaluation.runner import run_full_evaluation
from backend.retrieval.hybrid_retriever import HybridRetriever

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate")
async def evaluate(
    run_generation: bool = True,
    session: AsyncSession = Depends(get_db_session),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
):
    return await run_full_evaluation(session, retriever, run_generation=run_generation)
