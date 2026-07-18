"""
Tests for VectorStore and the embedding provider factory (Milestone 3).

Important limitation, stated explicitly: this sandbox has no network access
to huggingface.co or api.openai.com, so we cannot download BGE's weights or
call the OpenAI embeddings API here. Rather than skip vector-store testing
entirely, we test the real Qdrant integration (real embedded Qdrant, real
collection creation, real upsert, real cosine search, real filtering, real
deletion) using a small deterministic FakeEmbeddingProvider. This proves the
indexing/search/delete logic is correct; it does NOT prove BGE or OpenAI
embedding calls work, which must be verified in an environment with normal
internet access (see README).
"""
import uuid

import pytest

from app.core.config import get_settings
from app.schemas.chunk import Chunk, ChunkingResult, PageSpan, SplitReason
from app.schemas.document import DocumentMetadata, DocumentType, ExtractionMethod, Page, PageMetadata, ParsedDocument
from app.services.embeddings import (
    BGEEmbeddingProvider,
    EmbeddingProvider,
    HashEmbeddingProvider,
    get_embedding_provider,
)
from app.services.vector_store import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embedding for testing the vector store
    integration without any model download or API call. Not semantically
    meaningful - only used to verify Qdrant plumbing (indexing, filtering,
    deletion) works correctly."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _fake_vector(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        raw = [b / 255.0 for b in h[: self._dim]]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._fake_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._fake_vector(query)


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
            text="Section 80G requires a valid registration number on the donation receipt.",
            structural_label="4.",
            split_reason=SplitReason.STRUCTURAL_MARKER,
            page_start=3,
            page_end=3,
            page_spans=[PageSpan(page_number=3, char_count=75)],
            char_count=75,
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
    ]
    return ChunkingResult(
        document_id=fake_document.document_id,
        total_chunks=len(chunks),
        chunks=chunks,
        structural_markers_found=2,
    )


@pytest.fixture
def vector_store(tmp_path, monkeypatch):
    """A VectorStore backed by an isolated, temporary embedded Qdrant
    instance - each test gets a clean collection with no cross-test state."""
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant_test"))
    get_settings.cache_clear()
    store = VectorStore(vector_dimension=8)
    yield store
    get_settings.cache_clear()


class TestVectorStoreIndexingAndSearch:
    def test_index_chunks_returns_correct_count(self, vector_store, fake_document, fake_chunking_result):
        embedder = FakeEmbeddingProvider(dim=8)
        vectors = embedder.embed_texts([c.text for c in fake_chunking_result.chunks])
        indexed = vector_store.index_chunks(fake_document, fake_chunking_result, vectors)
        assert indexed == 2
        assert vector_store.count() == 2

    def test_search_returns_payload_with_citation_fields(self, vector_store, fake_document, fake_chunking_result):
        embedder = FakeEmbeddingProvider(dim=8)
        vectors = embedder.embed_texts([c.text for c in fake_chunking_result.chunks])
        vector_store.index_chunks(fake_document, fake_chunking_result, vectors)

        query_vector = embedder.embed_query("Section 80G requires a valid registration number on the donation receipt.")
        results = vector_store.search(query_vector, top_k=5)

        assert len(results) == 2
        top_hit = results[0]
        # The exact-text query should match its source chunk with the top score.
        assert top_hit["structural_label"] == "4."
        assert top_hit["page_start"] == 3
        assert top_hit["document_id"] == fake_document.document_id
        assert top_hit["original_filename"] == "test.pdf"
        assert "score" in top_hit

    def test_search_filters_by_document_id(self, vector_store, fake_document, fake_chunking_result):
        embedder = FakeEmbeddingProvider(dim=8)
        vectors = embedder.embed_texts([c.text for c in fake_chunking_result.chunks])
        vector_store.index_chunks(fake_document, fake_chunking_result, vectors)

        results = vector_store.search(
            embedder.embed_query("anything"), top_k=5, document_id="nonexistent-id"
        )
        assert len(results) == 0

    def test_delete_document_removes_all_its_points(self, vector_store, fake_document, fake_chunking_result):
        embedder = FakeEmbeddingProvider(dim=8)
        vectors = embedder.embed_texts([c.text for c in fake_chunking_result.chunks])
        vector_store.index_chunks(fake_document, fake_chunking_result, vectors)
        assert vector_store.count() == 2

        vector_store.delete_document(fake_document.document_id)
        assert vector_store.count() == 0

    def test_vectors_must_align_with_chunks(self, vector_store, fake_document, fake_chunking_result):
        with pytest.raises(ValueError):
            vector_store.index_chunks(fake_document, fake_chunking_result, vectors=[[0.1] * 8])  # only 1, need 2


class TestEmbeddingProviderFactory:
    def test_factory_returns_bge_provider(self):
        provider = get_embedding_provider("bge")
        assert isinstance(provider, BGEEmbeddingProvider)
        # dimension is a config value, not a model call - safe without network.
        assert provider.dimension == get_settings().bge_embedding_dim

    def test_factory_raises_on_unknown_provider(self):
        with pytest.raises(ValueError):
            get_embedding_provider("not_a_real_provider")

    def test_openai_provider_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        get_settings.cache_clear()
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_embedding_provider("openai")
        get_settings.cache_clear()


class TestHashEmbeddingProvider:
    """The offline, dependency-free fallback - fully testable with no
    network at all, unlike BGE/OpenAI."""

    def test_factory_returns_hash_provider(self):
        assert isinstance(get_embedding_provider("hash"), HashEmbeddingProvider)

    def test_embedding_is_deterministic(self):
        provider = HashEmbeddingProvider(dim=64)
        v1 = provider.embed_query("Section 80G registration")
        v2 = provider.embed_query("Section 80G registration")
        assert v1 == v2

    def test_embedding_is_normalized(self):
        provider = HashEmbeddingProvider(dim=64)
        vec = provider.embed_query("some legal text about deductions")
        norm = sum(v * v for v in vec) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_similar_text_more_similar_than_dissimilar_text(self):
        """Not a semantic embedding, but shared word-roots should still push
        cosine similarity up relative to completely unrelated text."""
        provider = HashEmbeddingProvider(dim=256)
        base = provider.embed_query("registration number for donation receipt")
        similar = provider.embed_query("registration numbers for donation receipts")
        different = provider.embed_query("quantum mechanics wave function collapse")

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            return dot  # both already normalized to unit length

        assert cosine(base, similar) > cosine(base, different)

    def test_dimension_matches_configured_size(self):
        provider = HashEmbeddingProvider(dim=128)
        assert provider.dimension == 128
        assert len(provider.embed_query("test")) == 128
