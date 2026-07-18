"""
Hybrid search API route (Milestone 5).
"""
from fastapi import APIRouter, Depends

from app.schemas.search import HybridSearchResponse, SearchRequest
from app.services.hybrid_search import HybridSearchService

router = APIRouter(tags=["hybrid-search"])


def get_hybrid_search_service() -> HybridSearchService:
    return HybridSearchService()


@router.post("/hybrid-search", response_model=HybridSearchResponse)
async def hybrid_search(
    request: SearchRequest,
    service: HybridSearchService = Depends(get_hybrid_search_service),
) -> HybridSearchResponse:
    results = service.search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        document_type=request.document_type.value if request.document_type else None,
    )
    return HybridSearchResponse(query=request.query, results=results, total_results=len(results))
