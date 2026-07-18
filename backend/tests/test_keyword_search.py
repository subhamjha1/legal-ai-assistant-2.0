"""
Tests for keyword search (Milestone 4).

BM25LocalProvider is tested fully and for real - in-process rank_bm25, real
scoring, real persistence to disk, real filtering and deletion. No mocks.

ElasticsearchProvider is NOT tested here: this sandbox has no Docker and no
network path to a registry to pull a real ES image, so there is nothing to
connect to. Its code is real production code (see keyword_search.py) meant
to be verified against a real cluster - see docker-compose.yml and the
README for how to do that on a machine with Docker available.
"""
import uuid

import pytest

from app.core.config import get_settings
from app.schemas.chunk import Chunk, ChunkingResult, PageSpan, SplitReason
from app.schemas.document import DocumentMetadata, DocumentType, ExtractionMethod, Page, PageMetadata, ParsedDocument
from app.services.keyword_search import BM25LocalProvider, get_keyword_search_provider


@pytest.fixture
def fake_document() -> ParsedDocument:
    return ParsedDocument(
        document_id=str(uuid.uuid4()),
        metadata=DocumentMetadata(
            original_filename="test.pdf",
            document_type=DocumentType.JUDGMENT,
            total_pages=1,
            file_size_bytes=100,
            file_hash="abc123",
        ),
        pages=[
            Page(
                page_number=1,
                text="dummy",
                metadata=PageMetadata(extraction_method=ExtractionMethod.NATIVE_TEXT, char_count=5),
            )
        ],
    )


@pytest.fixture
def fake_chunking_result(fake_document) -> ChunkingResult:
    chunks = [
        Chunk(
            document_id=fake_document.document_id,
            chunk_index=0,
            text="Section 80G(5)(iv) requires a valid registration number on the donation receipt issued by the trust.",
            structural_label="4.",
            split_reason=SplitReason.STRUCTURAL_MARKER,
            page_start=3,
            page_end=3,
            page_spans=[PageSpan(page_number=3, char_count=100)],
            char_count=100,
        ),
        Chunk(
            document_id=fake_document.document_id,
            chunk_index=1,
            text="The court held that the disallowance of the tax deduction was not in accordance with law.",
            structural_label="5.",
            split_reason=SplitReason.STRUCTURAL_MARKER,
            page_start=3,
            page_end=3,
            page_spans=[PageSpan(page_number=3, char_count=90)],
            char_count=90,
        ),
        Chunk(
            document_id=fake_document.document_id,
            chunk_index=2,
            text="Case No. 1234/2024 was heard before the Hon'ble Justice A. Sharma at the High Court of Delhi.",
            structural_label=None,
            split_reason=SplitReason.DOCUMENT_BOUNDARY,
            page_start=1,
            page_end=1,
            page_spans=[PageSpan(page_number=1, char_count=95)],
            char_count=95,
        ),
    ]
    return ChunkingResult(
        document_id=fake_document.document_id,
        total_chunks=len(chunks),
        chunks=chunks,
        structural_markers_found=2,
    )


@pytest.fixture
def bm25_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_LOCAL_STORAGE_PATH", str(tmp_path / "bm25_test"))
    get_settings.cache_clear()
    provider = BM25LocalProvider()
    yield provider
    get_settings.cache_clear()


class TestBM25LocalProvider:
    def test_index_chunks_returns_correct_count(self, bm25_provider, fake_document, fake_chunking_result):
        indexed = bm25_provider.index_chunks(fake_document, fake_chunking_result)
        assert indexed == 3
        assert bm25_provider.count() == 3

    def test_exact_term_match_ranks_correctly(self, bm25_provider, fake_document, fake_chunking_result):
        """A query containing the exact statutory term '80G' should surface
        the chunk mentioning it above unrelated chunks - this is precisely
        what BM25 keyword search is for, and where pure vector search on
        rare identifiers like section numbers can under-perform."""
        bm25_provider.index_chunks(fake_document, fake_chunking_result)
        results = bm25_provider.search("Section 80G registration number", top_k=5)

        assert len(results) > 0
        assert "80G" in results[0]["text"]
        assert results[0]["structural_label"] == "4."

    def test_case_number_query_matches_correct_chunk(self, bm25_provider, fake_document, fake_chunking_result):
        bm25_provider.index_chunks(fake_document, fake_chunking_result)
        results = bm25_provider.search("Case No. 1234/2024", top_k=5)
        assert len(results) > 0
        assert "1234" in results[0]["text"]

    def test_search_filters_by_document_id(self, bm25_provider, fake_document, fake_chunking_result):
        bm25_provider.index_chunks(fake_document, fake_chunking_result)
        results = bm25_provider.search("Section 80G", top_k=5, document_id="nonexistent")
        assert len(results) == 0

    def test_irrelevant_query_returns_no_results(self, bm25_provider, fake_document, fake_chunking_result):
        bm25_provider.index_chunks(fake_document, fake_chunking_result)
        results = bm25_provider.search("quantum physics photosynthesis", top_k=5)
        assert len(results) == 0

    def test_delete_document_removes_all_entries(self, bm25_provider, fake_document, fake_chunking_result):
        bm25_provider.index_chunks(fake_document, fake_chunking_result)
        assert bm25_provider.count() == 3
        bm25_provider.delete_document(fake_document.document_id)
        assert bm25_provider.count() == 0

    def test_reindexing_same_document_replaces_not_duplicates(self, bm25_provider, fake_document, fake_chunking_result):
        bm25_provider.index_chunks(fake_document, fake_chunking_result)
        bm25_provider.index_chunks(fake_document, fake_chunking_result)  # re-index
        assert bm25_provider.count() == 3  # not 6

    def test_corpus_persists_across_provider_instances(self, tmp_path, monkeypatch, fake_document, fake_chunking_result):
        monkeypatch.setenv("BM25_LOCAL_STORAGE_PATH", str(tmp_path / "bm25_persist_test"))
        get_settings.cache_clear()

        provider1 = BM25LocalProvider()
        provider1.index_chunks(fake_document, fake_chunking_result)

        provider2 = BM25LocalProvider()  # simulates a fresh process reloading from disk
        assert provider2.count() == 3
        results = provider2.search("Section 80G", top_k=5)
        assert len(results) > 0

        get_settings.cache_clear()


class TestKeywordSearchFactory:
    def test_factory_returns_bm25_provider(self):
        provider = get_keyword_search_provider("bm25_local")
        assert isinstance(provider, BM25LocalProvider)

    def test_factory_raises_on_unknown_provider(self):
        with pytest.raises(ValueError):
            get_keyword_search_provider("not_a_real_provider")
