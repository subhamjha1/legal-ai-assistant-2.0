"""
PDF parsing service.

Why this exists:
This is the single point of truth for turning a raw PDF into page-indexed,
citation-ready text. Every design decision here is driven by one constraint:
downstream citations must be exact, so we cannot afford to lose or misalign
page numbers, and we cannot afford to silently return empty text for a
scanned page.

Strategy per page:
1. Try native text extraction (PyMuPDF) - fast, high fidelity for digital PDFs.
2. If the page looks table-heavy, re-extract with pdfplumber, which handles
   tabular layout better than PyMuPDF's plain text mode.
3. If native extraction yields near-nothing (a scanned/image page), fall back
   to Tesseract OCR on a rasterized image of that page.

We never mix strategies within the *same* page silently - each page's
metadata records exactly which method produced its text, so evaluation
(Milestone 9) can measure OCR-vs-native accuracy separately.
"""
import hashlib
import io
import time
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.document import (
    DocumentMetadata,
    DocumentType,
    ExtractionMethod,
    Page,
    PageMetadata,
    ParsedDocument,
)

logger = get_logger(__name__)


class PDFParsingError(Exception):
    """Raised when a PDF cannot be parsed at all (corrupt file, encrypted, etc.)."""


class DocumentParser:
    """
    Parses a single PDF file into a ParsedDocument.

    Kept as a class (not free functions) because it holds no state across
    calls but benefits from grouping related private helpers and settings
    together, and because it gives Milestone 2+ a clean, mockable interface
    to depend on (dependency injection into the ingestion route).
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse(
        self,
        file_path: Path,
        original_filename: str,
        document_type: DocumentType,
        document_type_source: str = "user_tagged",
        document_type_confidence: float | None = None,
        custom_tags: list[str] | None = None,
    ) -> ParsedDocument:
        """
        Parse a PDF at `file_path` into a fully structured ParsedDocument.

        Raises:
            PDFParsingError: if the file cannot be opened at all.
        """
        start = time.monotonic()
        file_bytes = file_path.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        try:
            doc = fitz.open(file_path)
        except Exception as exc:  # fitz raises its own exception types
            raise PDFParsingError(f"Could not open PDF '{original_filename}': {exc}") from exc

        if doc.needs_pass:
            doc.close()
            raise PDFParsingError(f"'{original_filename}' is password-protected; cannot parse.")

        pages: list[Page] = []
        try:
            for page_index in range(doc.page_count):
                page = self._parse_single_page(doc, file_path, page_index)
                pages.append(page)
        finally:
            doc.close()

        elapsed = time.monotonic() - start
        logger.info(
            "Parsed '%s': %d pages in %.2fs (%d via OCR)",
            original_filename,
            len(pages),
            elapsed,
            sum(1 for p in pages if p.metadata.extraction_method == ExtractionMethod.OCR),
        )

        metadata = DocumentMetadata(
            original_filename=original_filename,
            document_type=document_type,
            document_type_confidence=document_type_confidence,
            document_type_source=document_type_source,
            total_pages=len(pages),
            file_size_bytes=len(file_bytes),
            file_hash=file_hash,
            custom_tags=custom_tags or [],
        )
        return ParsedDocument(metadata=metadata, pages=pages)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _parse_single_page(self, doc: "fitz.Document", file_path: Path, page_index: int) -> Page:
        """
        Extract one page, trying native text first, then table-aware, then OCR.
        `page_index` is 0-indexed internally (PyMuPDF convention); we convert
        to 1-indexed page_number for the human-facing schema.
        """
        page_number = page_index + 1
        warnings: list[str] = []
        fitz_page = doc[page_index]

        native_text = fitz_page.get_text("text").strip()
        has_tables = self._page_likely_has_tables(fitz_page)

        # Case 1: page has real text and doesn't look like a table -> done.
        if len(native_text) >= self.settings.ocr_trigger_char_threshold and not has_tables:
            return Page(
                page_number=page_number,
                text=native_text,
                metadata=PageMetadata(
                    extraction_method=ExtractionMethod.NATIVE_TEXT,
                    char_count=len(native_text),
                    has_tables=False,
                    warnings=warnings,
                ),
            )

        # Case 2: page looks table-heavy -> re-extract with pdfplumber.
        if has_tables:
            table_text = self._extract_with_pdfplumber(file_path, page_index)
            if len(table_text.strip()) >= self.settings.ocr_trigger_char_threshold:
                return Page(
                    page_number=page_number,
                    text=table_text.strip(),
                    metadata=PageMetadata(
                        extraction_method=ExtractionMethod.TABLE_AWARE,
                        char_count=len(table_text.strip()),
                        has_tables=True,
                        warnings=warnings,
                    ),
                )
            warnings.append("Table detected but pdfplumber extraction was weak; falling back to OCR.")

        # Case 3: little/no native text -> treat as scanned, run OCR.
        ocr_text, ocr_confidence = self._extract_with_ocr(fitz_page)
        if len(ocr_text.strip()) < self.settings.ocr_trigger_char_threshold:
            warnings.append("Page yielded near-empty text after OCR; may be blank or corrupt.")
            method = ExtractionMethod.FAILED
        else:
            method = ExtractionMethod.OCR

        return Page(
            page_number=page_number,
            text=ocr_text.strip(),
            metadata=PageMetadata(
                extraction_method=method,
                char_count=len(ocr_text.strip()),
                has_tables=has_tables,
                ocr_confidence=ocr_confidence,
                warnings=warnings,
            ),
        )

    @staticmethod
    def _page_likely_has_tables(fitz_page: "fitz.Page") -> bool:
        """
        Cheap heuristic: PyMuPDF's table finder is fast enough to run on every
        page as a pre-check, avoiding a full pdfplumber pass unless needed.
        """
        try:
            tables = fitz_page.find_tables()
            return len(tables.tables) > 0
        except Exception:
            return False

    @staticmethod
    def _extract_with_pdfplumber(file_path: Path, page_index: int) -> str:
        """Re-extract a single page with pdfplumber, which linearizes tables
        into readable rows far better than PyMuPDF's default text mode."""
        with pdfplumber.open(file_path) as pdf:
            plumber_page = pdf.pages[page_index]
            text_parts = [plumber_page.extract_text() or ""]
            for table in plumber_page.extract_tables():
                rows = ["\t".join(cell or "" for cell in row) for row in table]
                text_parts.append("\n".join(rows))
            return "\n\n".join(part for part in text_parts if part)

    def _extract_with_ocr(self, fitz_page: "fitz.Page") -> tuple[str, float]:
        """Rasterize the page to an image and run Tesseract OCR on it.
        Returns (text, mean_confidence)."""
        # 300 DPI-equivalent zoom for reliable OCR accuracy.
        pix = fitz_page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.open(io.BytesIO(pix.tobytes("png")))

        data = pytesseract.image_to_data(
            image, lang=self.settings.ocr_language, output_type=pytesseract.Output.DICT
        )
        words = [w for w in data["text"] if w.strip()]
        confidences = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
        text = " ".join(words)
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return text, round(mean_confidence, 2)
