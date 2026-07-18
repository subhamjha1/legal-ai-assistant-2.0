"""
Tests for DocumentParser and DocumentClassifier.

These are integration-style tests against a real synthetic PDF (rather than
mocks) because the whole point of Milestone 1 is correctness of page-level
extraction across native / table / OCR paths - mocking fitz/pdfplumber/
tesseract would test nothing meaningful.
"""
from pathlib import Path

import pytest

from app.schemas.document import DocumentType, ExtractionMethod
from app.services.classifier import DocumentClassifier
from app.services.parser import DocumentParser, PDFParsingError

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "sample_docs" / "sample_legal_doc_final.pdf"


@pytest.fixture(scope="module")
def parsed_document():
    parser = DocumentParser()
    return parser.parse(
        SAMPLE_PDF,
        original_filename="sample_legal_doc_final.pdf",
        document_type=DocumentType.JUDGMENT,
    )


def test_parses_all_pages(parsed_document):
    assert parsed_document.metadata.total_pages == 3
    assert len(parsed_document.pages) == 3


def test_page_numbers_are_sequential_and_one_indexed(parsed_document):
    assert [p.page_number for p in parsed_document.pages] == [1, 2, 3]


def test_page_1_uses_native_extraction(parsed_document):
    page1 = parsed_document.pages[0]
    assert page1.metadata.extraction_method == ExtractionMethod.NATIVE_TEXT
    assert "HIGH COURT OF DELHI" in page1.text
    assert page1.metadata.char_count > 0


def test_page_2_contains_expected_legal_text(parsed_document):
    page2 = parsed_document.pages[1]
    assert "Section 80G" in page2.text
    assert page2.metadata.extraction_method in (
        ExtractionMethod.NATIVE_TEXT,
        ExtractionMethod.TABLE_AWARE,
    )


def test_page_3_falls_back_to_ocr(parsed_document):
    """Page 3 is an image-only page with no text layer - the parser must
    detect near-zero native text and route it through Tesseract."""
    page3 = parsed_document.pages[2]
    assert page3.metadata.extraction_method == ExtractionMethod.OCR
    assert page3.metadata.ocr_confidence is not None
    # OCR is not perfect, so we assert on a distinctive substring rather than
    # exact equality.
    assert "80G" in page3.text or "Commissioner" in page3.text


def test_file_hash_is_deterministic(parsed_document):
    parser = DocumentParser()
    second_parse = parser.parse(
        SAMPLE_PDF,
        original_filename="sample_legal_doc_final.pdf",
        document_type=DocumentType.JUDGMENT,
    )
    assert parsed_document.metadata.file_hash == second_parse.metadata.file_hash


def test_password_protected_pdf_raises(tmp_path):
    """We don't have a real encrypted sample, so this documents expected
    behavior via a corrupt-file proxy (empty file), which fitz also rejects."""
    bad_pdf = tmp_path / "broken.pdf"
    bad_pdf.write_bytes(b"not a real pdf")
    parser = DocumentParser()
    with pytest.raises(PDFParsingError):
        parser.parse(bad_pdf, original_filename="broken.pdf", document_type=DocumentType.UNKNOWN)


class TestDocumentClassifier:
    def test_classifies_judgment_from_keywords(self):
        classifier = DocumentClassifier()
        text = (
            "IN THE HIGH COURT OF DELHI. CASE NO. 55/2024. PETITIONER: X. "
            "RESPONDENT: Y. BEFORE THE HON'BLE JUSTICE."
        )
        result = classifier.suggest_type(text)
        assert result.suggested_type == DocumentType.JUDGMENT
        assert result.confidence > 0.5

    def test_classifies_tax_document_from_keywords(self):
        classifier = DocumentClassifier()
        text = "FORM 1040. INTERNAL REVENUE SERVICE. TAXABLE INCOME. DEDUCTIONS."
        result = classifier.suggest_type(text)
        assert result.suggested_type == DocumentType.TAX_DOCUMENT

    def test_low_signal_text_returns_unknown_or_low_confidence(self):
        classifier = DocumentClassifier()
        result = classifier.suggest_type("Hello, this is just a regular letter with no legal terms.")
        assert result.suggested_type == DocumentType.UNKNOWN
        assert result.confidence == 0.0
