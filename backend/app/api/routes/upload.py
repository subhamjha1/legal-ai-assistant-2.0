"""
Upload / ingestion API routes.

Why this exists:
This is the HTTP boundary for Milestone 1. It stays thin on purpose - all
real logic lives in services (parser, classifier, repository) so those
services stay testable without spinning up FastAPI at all.
"""
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.schemas.document import (
    DocumentType,
    UploadResponse,
)
from app.services.classifier import DocumentClassifier
from app.services.parser import DocumentParser, PDFParsingError
from app.services.storage import DocumentRepository

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["ingestion"])


# --------------------------------------------------------------------- #
# Dependency providers (constructor injection at the FastAPI layer)
# --------------------------------------------------------------------- #
def get_parser() -> DocumentParser:
    return DocumentParser()


def get_classifier() -> DocumentClassifier:
    return DocumentClassifier()


def get_repository() -> DocumentRepository:
    return DocumentRepository()


# --------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------- #
@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    document_type: Optional[DocumentType] = None,
    settings: Settings = Depends(get_settings),
    parser: DocumentParser = Depends(get_parser),
    classifier: DocumentClassifier = Depends(get_classifier),
    repository: DocumentRepository = Depends(get_repository),
) -> UploadResponse:
    """
    Upload a PDF for ingestion.

    - If `document_type` is provided, it is trusted as-is (user_tagged).
    - If omitted, the system suggests a type from content (auto_suggested),
      per the classification service's heuristic-then-LLM strategy.
    """
    _validate_upload(file, settings)
    start = time.monotonic()

    # Persist the raw upload first so parsing failures don't lose the file.
    temp_path = settings.upload_dir / f"{uuid.uuid4()}_{file.filename}"
    raw_bytes = await file.read()
    temp_path.write_bytes(raw_bytes)

    try:
        type_suggestion = None
        resolved_type = document_type
        type_source = "user_tagged"

        if resolved_type is None:
            # Quick first-pass parse just to get text for classification.
            # We reuse the same parser so we don't duplicate PDF-reading logic.
            preview_doc = parser.parse(
                temp_path,
                original_filename=file.filename,
                document_type=DocumentType.UNKNOWN,
            )
            preview_text = " ".join(p.text for p in preview_doc.pages[:2])
            type_suggestion = classifier.suggest_type(preview_text)
            resolved_type = type_suggestion.suggested_type
            type_source = "auto_suggested"

            # Reuse the preview parse result instead of parsing twice.
            parsed = preview_doc.model_copy(
                update={
                    "metadata": preview_doc.metadata.model_copy(
                        update={
                            "document_type": resolved_type,
                            "document_type_confidence": type_suggestion.confidence,
                            "document_type_source": type_source,
                        }
                    )
                }
            )
        else:
            parsed = parser.parse(
                temp_path,
                original_filename=file.filename,
                document_type=resolved_type,
                document_type_source=type_source,
            )

        repository.save(parsed)

        pages_needing_ocr = sum(
            1 for p in parsed.pages if p.metadata.extraction_method.value == "ocr"
        )
        return UploadResponse(
            document_id=parsed.document_id,
            original_filename=parsed.metadata.original_filename,
            document_type=parsed.metadata.document_type,
            document_type_source=parsed.metadata.document_type_source,
            type_suggestion=type_suggestion,
            total_pages=parsed.metadata.total_pages,
            pages_requiring_ocr=pages_needing_ocr,
            processing_time_seconds=round(time.monotonic() - start, 2),
        )

    except PDFParsingError as exc:
        logger.error("Parsing failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@router.get("")
async def list_documents(repository: DocumentRepository = Depends(get_repository)):
    documents = repository.list_all()
    return [
        {
            "document_id": d.document_id,
            "original_filename": d.metadata.original_filename,
            "document_type": d.metadata.document_type,
            "total_pages": d.metadata.total_pages,
            "uploaded_at": d.metadata.uploaded_at,
        }
        for d in documents
    ]


@router.get("/{document_id}")
async def get_document(document_id: str, repository: DocumentRepository = Depends(get_repository)):
    document = repository.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    repository: DocumentRepository = Depends(get_repository),
):
    deleted = repository.delete(document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    # Clean up any derived chunks so re-uploading the same doc later doesn't
    # resurrect stale chunk data under the same document_id.
    from app.services.storage import ChunkRepository
    ChunkRepository().delete(document_id)

    # Clean up vector index entries too, but only if the vector store has
    # already been constructed this process (avoid paying model-load cost
    # just to delete from an index that was never populated).
    try:
        from app.services.indexing import get_cached_vector_store
        if get_cached_vector_store.cache_info().currsize > 0:
            get_cached_vector_store().delete_document(document_id)
    except Exception:
        pass

    # Same for the keyword search index.
    try:
        from app.services.keyword_search import get_cached_keyword_provider
        if get_cached_keyword_provider.cache_info().currsize > 0:
            get_cached_keyword_provider().delete_document(document_id)
    except Exception:
        pass


# --------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------- #
def _validate_upload(file: UploadFile, settings: Settings) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Allowed: {settings.allowed_extensions}",
        )
