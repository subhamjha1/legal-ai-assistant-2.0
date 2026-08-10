from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from backend.ingestion.chunker import chunk_document
from backend.ingestion.citation_extractor import extract_citations
from backend.ingestion.classifier import extract_document_metadata
from backend.ingestion.pdf_parser import parse_pdf
from backend.ingestion.unstructured_fallback import needs_ocr_fallback, parse_pdf_with_unstructured
from backend.schemas.document import Chunk, DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    document_metadata: DocumentMetadata
    child_chunks: list[Chunk]
    parent_chunks: list[Chunk]

    @property
    def all_chunks(self) -> list[Chunk]:
        return self.child_chunks + self.parent_chunks


def ingest_single_pdf(path: str | Path) -> IngestionResult:
    path = Path(path)
    logger.info("Ingesting %s", path)

    parsed = parse_pdf(path)
    if needs_ocr_fallback(parsed):
        logger.info("%s looks scanned/low-text — retrying with OCR fallback", path.name)
        parsed = parse_pdf_with_unstructured(path)

    doc_meta = extract_document_metadata(parsed, path)

    child_chunks, parent_chunks = chunk_document(parsed, doc_meta)

    for chunk in child_chunks:
        chunk.citations = extract_citations(chunk.text)

    return IngestionResult(document_metadata=doc_meta, child_chunks=child_chunks, parent_chunks=parent_chunks)


def ingest_directory(directory: str | Path) -> list[IngestionResult]:
    directory = Path(directory)
    pdf_paths = sorted(directory.rglob("*.pdf"))
    logger.info("Found %d PDFs under %s", len(pdf_paths), directory)

    results: list[IngestionResult] = []
    for path in pdf_paths:
        try:
            results.append(ingest_single_pdf(path))
        except Exception:
            logger.exception("Failed to ingest %s — skipping", path)
    return results
