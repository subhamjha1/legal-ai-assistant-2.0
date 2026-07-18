"""
Keyword search API routes (Milestone 4).
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.search import SearchRequest, SearchResponse
from app.services.keyword_search import (
    KeywordSearchProvider,
    get_cached_keyword_provider,
)
from app.services.storage import ChunkRepository, DocumentRepository

router = APIRouter(tags=["keyword-search"])


def get_document_repository() -> DocumentRepository:
    return DocumentRepository()


def get_chunk_repository() -> ChunkRepository:
    return ChunkRepository()


@router.post("/documents/{document_id}/keyword-index", status_code=status.HTTP_201_CREATED)
async def keyword_index_document(
    document_id: str,
    doc_repo: DocumentRepository = Depends(get_document_repository),
    chunk_repo: ChunkRepository = Depends(get_chunk_repository),
    provider: KeywordSearchProvider = Depends(get_cached_keyword_provider),
):
    document = doc_repo.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    chunking_result = chunk_repo.get(document_id)
    if chunking_result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks found for this document. POST /documents/{id}/chunks first.",
        )

    indexed_count = provider.index_chunks(document, chunking_result)
    return {"document_id": document_id, "chunks_indexed": indexed_count}


@router.post("/keyword-search", response_model=SearchResponse)
async def keyword_search(
    request: SearchRequest,
    provider: KeywordSearchProvider = Depends(get_cached_keyword_provider),
) -> SearchResponse:
    results = provider.search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        document_type=request.document_type.value if request.document_type else None,
    )
    return SearchResponse(query=request.query, results=results, total_results=len(results))
