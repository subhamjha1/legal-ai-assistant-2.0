"""
Tests for re-ranking and the full RetrievalService pipeline (Milestone 6).

LightweightReranker is fully tested (real, dependency-free logic).
CrossEncoderReranker is NOT tested here - it requires downloading
BAAI/bge-reranker-large from HuggingFace, unreachable from this sandbox
(same constraint as BGE embeddings in Milestone 3). See README.
"""
import uuid

import pytest

from app.core.config import get_settings
from app.schemas.chunk import Chunk, ChunkingResult, PageSpan, SplitReason
from app.schemas.document import DocumentMetadata, DocumentType, ExtractionMethod, Page, PageMetadata, ParsedDocument
from app.services.hybrid_search import HybridSearchService
from app.services.keyword_search import BM25LocalProvider
from app.services.reranker import LightweightReranker, get_reranker
from app.services.retriever import RetrievalService
from app.services.vector_store import VectorStore
from tests.test_vector_store import FakeEmbeddingProvider


class TestLightweightReranker:
    def test_exact_term_match_scores_higher_than_no_overlap(self):
        reranker = LightweightReranker()
        results = reranker.rerank(
            query="Section 80G registration number",
            candidate_texts=[
                "Section 80G(5)(iv) requires a valid registration number on the receipt.",
                "The weather in Delhi was pleasant during the monsoon season.",
            ],
            candidate_scores=[0.02, 0.02],  # equal upstream score, isolate term-overlap effect
        )
        by_index = {r.index: r.confidence for r in results}
        assert by_index[0] > by_index[1]

    def test_upstream_score_contributes_when_term_overlap_is_tied(self):
        reranker = LightweightReranker()
        results = reranker.rerank(
            query="tax deduction",
            candidate_texts=[
                "tax deduction claim under review",
                "tax deduction claim under review",  # identical text, different upstream score
            ],
            candidate_scores=[0.05, 0.01],
        )
        by_index = {r.index: r.confidence for r in results}
        assert by_index[0] > by_index[1]

    def test_no_candidates_returns_empty(self):
        reranker = LightweightReranker()
        assert reranker.rerank("query", [], []) == []

    def test_results_sorted_descending_by_confidence(self):
        reranker = LightweightReranker()
        results = reranker.rerank(
            query="registration certificate",
            candidate_texts=["registration certificate", "unrelated text", "registration"],
            candidate_scores=[0.03, 0.03, 0.02],
        )
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_factory_returns_lightweight_by_default(self):
        assert isinstance(get_reranker("lightweight"), LightweightReranker)

    def test_factory_raises_on_unknown_provider(self):
        with pytest.raises(ValueError):
            get_reranker("not_a_real_reranker")


# ---------------------------------------------------------------------- #
# Full pipeline integration test
# ---------------------------------------------------------------------- #
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
        pages=[Page(page_number=1, text="dummy", metadata=PageMetadata(extraction_method=ExtractionMethod.NATIVE_TEXT, char_count=5))],
    )


@pytest.fixture
def fake_chunking_result(fake_document) -> ChunkingResult:
    chunks = [
        Chunk(
            document_id=fake_document.document_id, chunk_index=0,
            text="Section 80G(5)(iv) requires a valid registration number on the donation receipt.",
            structural_label="4.", split_reason=SplitReason.STRUCTURAL_MARKER,
            page_start=3, page_end=3, page_spans=[PageSpan(page_number=3, char_count=80)], char_count=80,
        ),
        Chunk(
            document_id=fake_document.document_id, chunk_index=1,
            text="The court held that the disallowance of the tax deduction was not in accordance with law.",
            structural_label="5.", split_reason=SplitReason.STRUCTURAL_MARKER,
            page_start=3, page_end=3, page_spans=[PageSpan(page_number=3, char_count=90)], char_count=90,
        ),
        Chunk(
            document_id=fake_document.document_id, chunk_index=2,
            text="Case No. 1234/2024 was heard before the Hon'ble Justice A. Sharma at the High Court of Delhi.",
            structural_label=None, split_reason=SplitReason.DOCUMENT_BOUNDARY,
            page_start=1, page_end=1, page_spans=[PageSpan(page_number=1, char_count=95)], char_count=95,
        ),
    ]
    return ChunkingResult(document_id=fake_document.document_id, total_chunks=3, chunks=chunks, structural_markers_found=2)


class TestRetrievalServiceIntegration:
    def test_full_pipeline_returns_high_confidence_results_with_citations(
        self, tmp_path, monkeypatch, fake_document, fake_chunking_result
    ):
        monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant_retr_test"))
        monkeypatch.setenv("BM25_LOCAL_STORAGE_PATH", str(tmp_path / "bm25_retr_test"))
        monkeypatch.setenv("RERANK_MIN_CONFIDENCE", "0.0")  # don't filter anything for this test
        get_settings.cache_clear()

        embedder = FakeEmbeddingProvider(dim=8)
        vector_store = VectorStore(vector_dimension=8)
        keyword_provider = BM25LocalProvider()

        vectors = embedder.embed_texts([c.text for c in fake_chunking_result.chunks])
        vector_store.index_chunks(fake_document, fake_chunking_result, vectors)
        keyword_provider.index_chunks(fake_document, fake_chunking_result)

        hybrid_service = HybridSearchService(
            embedding_provider=embedder, vector_store=vector_store, keyword_provider=keyword_provider
        )
        retrieval_service = RetrievalService(
            hybrid_search_service=hybrid_service,
            reranker=get_reranker("lightweight"),
            embedding_provider=embedder,
        )

        results, candidates_considered, dropped = retrieval_service.retrieve(
            "Section 80G registration number requirement", top_k=3
        )

        assert candidates_considered > 0
        assert len(results) > 0
        # Every result must carry full citation-ready fields.
        for r in results:
            assert r.page_start >= 1
            assert r.original_filename == "test.pdf"
            assert 0.0 <= r.confidence <= 1.0
        # The exact statutory match should be the top result.
        assert results[0].structural_label == "4."

        get_settings.cache_clear()

    def test_confidence_filtering_drops_low_confidence_results(
        self, tmp_path, monkeypatch, fake_document, fake_chunking_result
    ):
        monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant_filter_test"))
        monkeypatch.setenv("BM25_LOCAL_STORAGE_PATH", str(tmp_path / "bm25_filter_test"))
        monkeypatch.setenv("RERANK_MIN_CONFIDENCE", "0.99")  # near-impossible threshold
        get_settings.cache_clear()

        embedder = FakeEmbeddingProvider(dim=8)
        vector_store = VectorStore(vector_dimension=8)
        keyword_provider = BM25LocalProvider()
        vectors = embedder.embed_texts([c.text for c in fake_chunking_result.chunks])
        vector_store.index_chunks(fake_document, fake_chunking_result, vectors)
        keyword_provider.index_chunks(fake_document, fake_chunking_result)

        hybrid_service = HybridSearchService(
            embedding_provider=embedder, vector_store=vector_store, keyword_provider=keyword_provider
        )
        retrieval_service = RetrievalService(
            hybrid_search_service=hybrid_service, reranker=get_reranker("lightweight"), embedding_provider=embedder
        )

        results, candidates_considered, dropped = retrieval_service.retrieve("Section 80G", top_k=3)
        assert dropped > 0
        assert len(results) == 0  # threshold is deliberately unreachable

        get_settings.cache_clear()
