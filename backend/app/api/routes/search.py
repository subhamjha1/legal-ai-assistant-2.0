"""
Vector search API routes (Milestone 3).
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.search import SearchRequest, SearchResponse
from app.services.indexing import IndexingService
from app.services.storage import ChunkRepository, DocumentRepository

router = APIRouter(tags=["vector-search"])


def get_indexing_service() -> IndexingService:
    return IndexingService()


def get_document_repository() -> DocumentRepository:
    return DocumentRepository()


def get_chunk_repository() -> ChunkRepository:
    return ChunkRepository()


@router.post("/documents/{document_id}/index", status_code=status.HTTP_201_CREATED)
async def index_document(
    document_id: str,
    indexing_service: IndexingService = Depends(get_indexing_service),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    chunk_repo: ChunkRepository = Depends(get_chunk_repository),
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

    indexed_count = indexing_service.index_document(document, chunking_result)
    return {"document_id": document_id, "chunks_indexed": indexed_count}


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> SearchResponse:
    results = indexing_service.search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        document_type=request.document_type.value if request.document_type else None,
    )
    return SearchResponse(query=request.query, results=results, total_results=len(results))
