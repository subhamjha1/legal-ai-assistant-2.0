"""
Retrieval API route (Milestone 6).
"""
from fastapi import APIRouter, Depends

from app.schemas.search import RetrievalResponse, SearchRequest
from app.services.retriever import RetrievalService

router = APIRouter(tags=["retrieval"])


def get_retrieval_service() -> RetrievalService:
    return RetrievalService()


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    request: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    results, candidates_considered, dropped = service.retrieve(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        document_type=request.document_type.value if request.document_type else None,
    )
    return RetrievalResponse(
        query=request.query,
        results=results,
        total_results=len(results),
        candidates_considered=candidates_considered,
        below_confidence_threshold_count=dropped,
    )
