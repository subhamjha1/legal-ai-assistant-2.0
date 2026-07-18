"""
Document storage service (Repository Pattern).

Why this exists:
Milestone 1 only requires persisting parsed JSON to disk, but every later
milestone (vector indexing, keyword indexing, deletion) needs to read/write/
delete the same documents. By putting storage behind a repository interface
now, swapping disk storage for Postgres-backed metadata later (Milestone 3+)
means changing this one file, not every caller.
"""
from pathlib import Path

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.chunk import ChunkingResult
from app.schemas.document import ParsedDocument

logger = get_logger(__name__)


class DocumentRepository:
    """Persists and retrieves ParsedDocument records."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.settings.processed_dir.mkdir(parents=True, exist_ok=True)
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)

    def save(self, document: ParsedDocument) -> Path:
        """Persist a parsed document as JSON, keyed by its document_id."""
        out_path = self.settings.processed_dir / f"{document.document_id}.json"
        out_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved parsed document %s -> %s", document.document_id, out_path)
        return out_path

    def get(self, document_id: str) -> ParsedDocument | None:
        path = self.settings.processed_dir / f"{document_id}.json"
        if not path.exists():
            return None
        return ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))

    def list_all(self) -> list[ParsedDocument]:
        documents = []
        for path in sorted(self.settings.processed_dir.glob("*.json")):
            try:
                documents.append(ParsedDocument.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Skipping corrupt processed file %s: %s", path, exc)
        return documents

    def delete(self, document_id: str) -> bool:
        path = self.settings.processed_dir / f"{document_id}.json"
        if not path.exists():
            return False
        path.unlink()
        logger.info("Deleted document %s", document_id)
        return True

    def find_by_hash(self, file_hash: str) -> ParsedDocument | None:
        """Used to detect duplicate uploads before re-processing an identical file."""
        for document in self.list_all():
            if document.metadata.file_hash == file_hash:
                return document
        return None


class ChunkRepository:
    """Persists and retrieves ChunkingResult records (Milestone 2 output).

    Kept as a separate repository from DocumentRepository (rather than one
    god-object) because chunks and documents have different lifecycles - a
    document can be re-chunked with different parameters without touching
    its parsed pages, and Milestone 3 will read only from this repository
    when building the vector index.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._chunks_dir = self.settings.processed_dir / "chunks"
        self._chunks_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: ChunkingResult) -> Path:
        out_path = self._chunks_dir / f"{result.document_id}.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved %d chunks for document %s -> %s", result.total_chunks, result.document_id, out_path)
        return out_path

    def get(self, document_id: str) -> ChunkingResult | None:
        path = self._chunks_dir / f"{document_id}.json"
        if not path.exists():
            return None
        return ChunkingResult.model_validate_json(path.read_text(encoding="utf-8"))

    def delete(self, document_id: str) -> bool:
        path = self._chunks_dir / f"{document_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True
