"""
Tests for citation_formatter.py (Milestone 7).

These are the most important tests in the whole Q&A pipeline: they prove
that citations are built entirely from OUR retrieval metadata, never from
anything the model claims about pages or documents - even when the model's
output is malformed, duplicated, or refers to passages it was never given.
"""
from app.core.config import get_settings
from app.schemas.search import RetrievedChunk
from app.services.citation_formatter import extract_citations, is_no_evidence_response


def _chunk(page_start: int, page_end: int, doc="test.pdf", label=None, text="chunk text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="id",
        document_id="doc-1",
        text=text,
        structural_label=label,
        page_start=page_start,
        page_end=page_end,
        original_filename=doc,
        document_type="judgment",
        confidence=0.9,
        rrf_score=0.03,
    )


class TestExtractCitations:
    def test_single_tag_maps_to_correct_chunk(self):
        chunks = [_chunk(3, 3, label="4.")]
        answer = "The deduction requires a registration number [C1]."
        citations = extract_citations(answer, chunks)
        assert len(citations) == 1
        assert citations[0].chunk_ref == "C1"
        assert citations[0].page_start == 3
        assert citations[0].structural_label == "4."

    def test_multiple_distinct_tags_produce_multiple_citations(self):
        chunks = [_chunk(3, 3, label="4."), _chunk(3, 3, label="5.")]
        answer = "The court disallowed the deduction [C1], then reversed on appeal [C2]."
        citations = extract_citations(answer, chunks)
        assert len(citations) == 2
        assert [c.chunk_ref for c in citations] == ["C1", "C2"]

    def test_duplicate_tag_produces_one_citation_not_two(self):
        chunks = [_chunk(3, 3)]
        answer = "First point [C1]. Related point, same source [C1]."
        citations = extract_citations(answer, chunks)
        assert len(citations) == 1

    def test_multiple_tags_on_one_claim_both_extracted(self):
        chunks = [_chunk(1, 1), _chunk(3, 3)]
        answer = "This is supported by two sources [C1][C2]."
        citations = extract_citations(answer, chunks)
        assert len(citations) == 2

    def test_out_of_range_tag_is_skipped_not_crashed(self):
        """A model hallucinating a citation to a passage it was never given
        should degrade gracefully to 'no citation for that claim', not
        raise or fabricate a page number."""
        chunks = [_chunk(1, 1)]
        answer = "This claim cites a passage that doesn't exist [C7]."
        citations = extract_citations(answer, chunks)
        assert citations == []

    def test_no_tags_returns_empty_list(self):
        chunks = [_chunk(1, 1)]
        answer = "This answer has no citations at all."
        assert extract_citations(answer, chunks) == []

    def test_citation_page_always_comes_from_retrieval_not_model_text(self):
        """Even if the model's answer TEXT mentions a different page number
        in prose, the structured citation must reflect our own chunk
        metadata, not whatever the model typed."""
        chunks = [_chunk(page_start=42, page_end=42)]
        answer = "According to page 999, the rule applies [C1]."  # model is "wrong" in prose
        citations = extract_citations(answer, chunks)
        assert citations[0].page_start == 42  # our data wins, not the model's prose claim

    def test_document_name_always_comes_from_retrieval(self):
        chunks = [_chunk(1, 1, doc="real_source.pdf")]
        answer = "Some claim [C1]."
        citations = extract_citations(answer, chunks)
        assert citations[0].document == "real_source.pdf"


class TestNoEvidenceDetection:
    def test_exact_phrase_is_detected(self):
        settings = get_settings()
        assert is_no_evidence_response(settings.no_evidence_phrase) is True

    def test_exact_phrase_with_surrounding_whitespace_is_detected(self):
        settings = get_settings()
        assert is_no_evidence_response(f"  {settings.no_evidence_phrase}  \n") is True

    def test_similar_but_inexact_phrase_is_not_detected(self):
        """Deliberately strict: a paraphrase of 'no evidence' should NOT be
        treated as the mandated fallback, since evaluation (Milestone 9)
        needs this to be a mechanically checkable exact-match property."""
        assert is_no_evidence_response("I could not find any supporting evidence for this.") is False

    def test_normal_answer_is_not_detected_as_no_evidence(self):
        assert is_no_evidence_response("The deduction requires a valid registration number [C1].") is False
