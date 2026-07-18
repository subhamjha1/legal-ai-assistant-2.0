"""
Tests for HybridSearchService / Reciprocal Rank Fusion (Milestone 5).

Two levels of testing:
1. Unit-level: call _reciprocal_rank_fusion directly with hand-constructed
   rank lists, so the fusion math itself is verified precisely and
   independent of any real embedding/BM25 behavior.
2. Integration-level: wire a real VectorStore (embedded Qdrant) and a real
   BM25LocalProvider together with a deterministic FakeEmbeddingProvider,
   index a small corpus, and confirm the whole pipeline produces sensible
   fused results end-to-end.
"""
import uuid

import pytest

from app.core.config import get_settings
from app.schemas.chunk import Chunk, ChunkingResult, PageSpan, SplitReason
from app.schemas.document import DocumentMetadata, DocumentType, ExtractionMethod, Page, PageMetadata, ParsedDocument
from app.services.hybrid_search import HybridSearchService
from app.services.keyword_search import BM25LocalProvider
from app.services.vector_store import VectorStore
from tests.test_vector_store import FakeEmbeddingProvider


def _hit(chunk_id: str, score: float, **extra) -> dict:
    base = {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "text": f"text for {chunk_id}",
        "structural_label": None,
        "page_start": 1,
        "page_end": 1,
        "original_filename": "test.pdf",
        "document_type": "judgment",
        "score": score,
    }
    base.update(extra)
    return base


class TestReciprocalRankFusionUnit:
    """Directly tests the fusion math with constructed rank lists."""

    def _service(self) -> HybridSearchService:
        # Dependencies are never called in these unit tests - only
        # _reciprocal_rank_fusion is exercised - so None placeholders are
        # safe as long as we don't call .search().
        service = HybridSearchService.__new__(HybridSearchService)
        service.settings = get_settings()
        return service

    def test_chunk_found_by_both_rankers_outranks_single_ranker_hits(self):
        service = self._service()
        vector_hits = [_hit("A", 0.9), _hit("B", 0.8)]
        keyword_hits = [_hit("A", 5.0), _hit("C", 4.0)]

        fused = service._reciprocal_rank_fusion(vector_hits, keyword_hits)

        assert fused[0].chunk_id == "A"
        assert set(fused[0].matched_by) == {"vector", "keyword"}
        assert fused[0].rrf_score > fused[1].rrf_score

    def test_rrf_score_matches_formula(self):
        service = self._service()
        k = service.settings.rrf_k
        vector_hits = [_hit("A", 0.9)]
        keyword_hits = [_hit("A", 5.0)]

        fused = service._reciprocal_rank_fusion(vector_hits, keyword_hits)
        expected = 1.0 / (k + 1) + 1.0 / (k + 1)  # rank 1 in both lists
        assert fused[0].rrf_score == pytest.approx(expected)

    def test_rank_position_matters_not_raw_score_scale(self):
        """A chunk ranked #1 by keyword search with a huge BM25 score should
        not automatically dominate a chunk ranked #1 by vector search with a
        'small' cosine score - RRF only cares about rank, not magnitude."""
        service = self._service()
        vector_hits = [_hit("A", 0.55)]       # rank 1, modest cosine score
        keyword_hits = [_hit("B", 999.0)]     # rank 1, huge BM25 score

        fused = service._reciprocal_rank_fusion(vector_hits, keyword_hits)
        scores = {hit.chunk_id: hit.rrf_score for hit in fused}
        # Both are rank-1 in their respective single list, so their RRF
        # contribution should be identical despite wildly different raw scores.
        assert scores["A"] == pytest.approx(scores["B"])

    def test_chunk_missing_from_one_ranker_still_included(self):
        service = self._service()
        vector_hits = [_hit("A", 0.9)]
        keyword_hits: list[dict] = []

        fused = service._reciprocal_rank_fusion(vector_hits, keyword_hits)
        assert len(fused) == 1
        assert fused[0].chunk_id == "A"
        assert fused[0].matched_by == ["vector"]
        assert fused[0].keyword_rank is None

    def test_lower_rank_position_contributes_less(self):
        service = self._service()
        vector_hits = [_hit("A", 0.9), _hit("B", 0.85), _hit("C", 0.8)]
        keyword_hits: list[dict] = []

        fused = service._reciprocal_rank_fusion(vector_hits, keyword_hits)
        rrf_scores = [hit.rrf_score for hit in fused]
        assert rrf_scores == sorted(rrf_scores, reverse=True)
        assert fused[0].chunk_id == "A"
        assert fused[2].chunk_id == "C"


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
            document_id=fake_document.document_id,
            chunk_index=0,
            text="Section 80G(5)(iv) requires a valid registration number on the donation receipt.",
            structural_label="4.",
            split_reason=SplitReason.STRUCTURAL_MARKER,
            page_start=3,
            page_end=3,
            page_spans=[PageSpan(page_number=3, char_count=80)],
            char_count=80,
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
    # Note: a 3rd, term-distinct chunk is included deliberately - with only 2
    # documents, a term appearing in exactly 1 of them hits rank_bm25's
    # IDF=0 edge case (n == N/2), zeroing its score contribution entirely.
    # Real corpora (many chunks) essentially never hit this exactly; 3 is
    # enough to avoid it in this fixture too.
    return ChunkingResult(document_id=fake_document.document_id, total_chunks=3, chunks=chunks, structural_markers_found=2)


class TestHybridSearchIntegration:
    """Real VectorStore + real BM25LocalProvider + fake (deterministic)
    embeddings, wired together through HybridSearchService end-to-end."""

    def test_hybrid_search_returns_fused_results(self, tmp_path, monkeypatch, fake_document, fake_chunking_result):
        monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant_hybrid_test"))
        monkeypatch.setenv("BM25_LOCAL_STORAGE_PATH", str(tmp_path / "bm25_hybrid_test"))
        get_settings.cache_clear()

        embedder = FakeEmbeddingProvider(dim=8)
        vector_store = VectorStore(vector_dimension=8)
        keyword_provider = BM25LocalProvider()

        vectors = embedder.embed_texts([c.text for c in fake_chunking_result.chunks])
        vector_store.index_chunks(fake_document, fake_chunking_result, vectors)
        keyword_provider.index_chunks(fake_document, fake_chunking_result)

        service = HybridSearchService(
            embedding_provider=embedder, vector_store=vector_store, keyword_provider=keyword_provider
        )
        results = service.search("Section 80G registration number", top_k=5)

        assert len(results) > 0
        # The chunk with the exact keyword match should be present and
        # should have a keyword_rank set (found by BM25).
        top_labels = [r.structural_label for r in results]
        assert "4." in top_labels
        matched = next(r for r in results if r.structural_label == "4.")
        assert matched.keyword_rank is not None

        get_settings.cache_clear()
