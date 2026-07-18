"""
Pydantic schemas for document ingestion.

Why this exists:
This is the contract between the parser and every downstream consumer
(chunker, indexer, citation formatter). Getting this right once means the
rest of the pipeline (Milestones 2-7) can be built against a stable interface
instead of ad-hoc dicts. Every field a citation will ever need must live here.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    ACT = "act"
    JUDGMENT = "judgment"
    TAX_DOCUMENT = "tax_document"
    POV_DOCUMENT = "pov_document"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    NATIVE_TEXT = "native_text"      # extracted directly via PyMuPDF
    TABLE_AWARE = "table_aware"      # extracted via pdfplumber (table-heavy pages)
    OCR = "ocr"                      # extracted via Tesseract OCR (scanned pages)
    FAILED = "failed"                # extraction produced nothing usable


class DocumentTypeSuggestion(BaseModel):
    """
    Result of automatic document-type classification, used when the user
    leaves the document type blank on upload.
    """
    suggested_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    method: str  # "keyword_heuristic" or "llm"
    reasoning: Optional[str] = None


class PageMetadata(BaseModel):
    """Per-page extraction metadata, kept separate from page text for clarity."""
    extraction_method: ExtractionMethod
    char_count: int
    has_tables: bool = False
    ocr_confidence: Optional[float] = None  # 0-100, only set when method == OCR
    warnings: list[str] = Field(default_factory=list)


class Page(BaseModel):
    """A single extracted page. Page numbers are 1-indexed to match how a
    human would cite ('see page 12'), not how Python indexes lists."""
    page_number: int = Field(ge=1)
    text: str
    metadata: PageMetadata


class DocumentMetadata(BaseModel):
    """Document-level metadata, independent of any single page."""
    original_filename: str
    document_type: DocumentType
    document_type_confidence: Optional[float] = None
    document_type_source: str = "user_tagged"  # "user_tagged" | "auto_suggested"
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_pages: int
    file_size_bytes: int
    file_hash: str  # sha256, used for dedup and as a stable reference key
    custom_tags: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """
    The full structured output of the ingestion pipeline for one document.
    This is what gets persisted to storage/processed/ and what Milestone 2
    (chunking) consumes as input.
    """
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: DocumentMetadata
    pages: list[Page]

    @field_validator("pages")
    @classmethod
    def pages_must_be_sequential(cls, pages: list[Page]) -> list[Page]:
        numbers = [p.page_number for p in pages]
        if numbers != sorted(numbers):
            raise ValueError("Pages must be in ascending page_number order")
        return pages


class DocumentTypeTagRequest(BaseModel):
    """What the user optionally sends alongside an upload."""
    document_type: Optional[DocumentType] = None
    custom_tags: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """API response returned immediately after ingestion completes."""
    document_id: str
    original_filename: str
    document_type: DocumentType
    document_type_source: str
    type_suggestion: Optional[DocumentTypeSuggestion] = None
    total_pages: int
    pages_requiring_ocr: int
    processing_time_seconds: float
