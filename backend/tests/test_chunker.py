"""
Tests for SemanticChunker.

Uses the same synthetic 3-page PDF as Milestone 1 (parsed fresh here) so we
test against real extracted text, including the OCR-derived page 3 text,
not a hand-crafted string that wouldn't catch real-world extraction noise.
"""
from pathlib import Path

import pytest

from app.schemas.chunk import SplitReason
from app.schemas.document import DocumentType
from app.services.chunker import SemanticChunker
from app.services.parser import DocumentParser

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "sample_docs" / "sample_legal_doc_final.pdf"


@pytest.fixture(scope="module")
def parsed_document():
    parser = DocumentParser()
    return parser.parse(SAMPLE_PDF, original_filename="sample.pdf", document_type=DocumentType.JUDGMENT)


@pytest.fixture(scope="module")
def chunking_result(parsed_document):
    chunker = SemanticChunker()
    return chunker.chunk(parsed_document)


def test_finds_structural_markers(chunking_result):
    # The sample doc has 5 numbered paragraphs ("1." through "5.").
    assert chunking_result.structural_markers_found >= 5


def test_produces_multiple_chunks(chunking_result):
    # Small numbered-paragraph sections get merged (chunk_min_chars) into
    # coherent multi-paragraph chunks rather than 5 tiny fragments - 3 is
    # the correct outcome here, not a bug.
    assert chunking_result.total_chunks >= 3


def test_chunks_are_sequentially_indexed(chunking_result):
    indices = [c.chunk_index for c in chunking_result.chunks]
    assert indices == list(range(len(indices)))


def test_every_chunk_has_valid_page_range(chunking_result):
    for chunk in chunking_result.chunks:
        assert chunk.page_start <= chunk.page_end
        assert chunk.page_start >= 1
        assert len(chunk.page_spans) >= 1
        # page_spans char counts should roughly cover the chunk (allowing for
        # stripped whitespace at edges)
        assert sum(s.char_count for s in chunk.page_spans) >= chunk.char_count - 5


def test_paragraph_markers_produce_labeled_chunks(chunking_result):
    labeled = [c for c in chunking_result.chunks if c.structural_label]
    assert len(labeled) > 0
    # At least one chunk should be labeled from a numbered paragraph like "4."
    assert any(c.structural_label.strip().startswith(("1.", "2.", "3.", "4.", "5.")) for c in labeled)


def test_section_80g_paragraph_is_a_coherent_chunk(chunking_result):
    """The most important correctness check: the paragraph discussing
    Section 80G(5)(iv) should not be split mid-sentence away from its
    condition clause, since that's exactly the failure mode structure-aware
    chunking is meant to prevent."""
    matches = [c for c in chunking_result.chunks if "80G(5)(iv)" in c.text]
    assert len(matches) == 1
    chunk = matches[0]
    assert "registration number" in chunk.text
    assert "Commissioner" in chunk.text


def test_chunk_crossing_page_boundary_has_multiple_page_spans(chunking_result):
    """At least verify the mechanism works if any chunk does span pages -
    not all documents will have this, so we check conditionally."""
    multi_page_chunks = [c for c in chunking_result.chunks if c.page_start != c.page_end]
    for chunk in multi_page_chunks:
        assert len(chunk.page_spans) >= 2


def test_size_cap_applied_to_oversized_section():
    """Construct an artificially long single-paragraph document to verify
    the size-cap sub-splitting path (not exercised by the short sample doc)."""
    from app.schemas.document import DocumentMetadata, Page, PageMetadata, ParsedDocument, ExtractionMethod
    import uuid

    long_sentence = "This is a legal clause about tax deduction eligibility criteria. "
    long_text = "1. " + (long_sentence * 60)  # comfortably exceeds chunk_max_chars

    doc = ParsedDocument(
        document_id=str(uuid.uuid4()),
        metadata=DocumentMetadata(
            original_filename="long.pdf",
            document_type=DocumentType.ACT,
            total_pages=1,
            file_size_bytes=1000,
            file_hash="deadbeef",
        ),
        pages=[
            Page(
                page_number=1,
                text=long_text,
                metadata=PageMetadata(extraction_method=ExtractionMethod.NATIVE_TEXT, char_count=len(long_text)),
            )
        ],
    )

    result = SemanticChunker().chunk(doc)
    assert result.total_chunks > 1
    assert any(c.split_reason == SplitReason.SIZE_CAP for c in result.chunks)
    # Verify overlap: consecutive size-capped chunks should share some tail/head text
    size_capped = [c for c in result.chunks if c.overlaps_previous]
    assert len(size_capped) > 0
