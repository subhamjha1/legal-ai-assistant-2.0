"""
Chunking API routes (Milestone 2).

Thin HTTP layer over SemanticChunker + ChunkRepository, mirroring the same
dependency-injection pattern used in upload.py.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.chunk import ChunkingResult
from app.services.chunker import SemanticChunker
from app.services.storage import ChunkRepository, DocumentRepository

router = APIRouter(prefix="/documents", tags=["chunking"])


def get_chunker() -> SemanticChunker:
    return SemanticChunker()


def get_chunk_repository() -> ChunkRepository:
    return ChunkRepository()


def get_document_repository() -> DocumentRepository:
    return DocumentRepository()


@router.post("/{document_id}/chunks", response_model=ChunkingResult, status_code=status.HTTP_201_CREATED)
async def create_chunks(
    document_id: str,
    chunker: SemanticChunker = Depends(get_chunker),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    chunk_repo: ChunkRepository = Depends(get_chunk_repository),
) -> ChunkingResult:
    document = doc_repo.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    result = chunker.chunk(document)
    chunk_repo.save(result)
    return result


@router.get("/{document_id}/chunks", response_model=ChunkingResult)
async def get_chunks(
    document_id: str,
    chunk_repo: ChunkRepository = Depends(get_chunk_repository),
) -> ChunkingResult:
    result = chunk_repo.get(document_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No chunks found for this document. POST to this endpoint first to generate them.",
        )
    return result
