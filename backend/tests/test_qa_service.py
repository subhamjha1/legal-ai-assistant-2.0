"""
Tests for QAService (Milestone 7).

Two levels, matching the pattern used throughout this project:
1. Full pipeline test with a FakeLLMProvider - real retrieval (Qdrant + BM25
   + MMR + lightweight rerank), fake-but-deterministic LLM response, proving
   the whole orchestration (including the zero-evidence short-circuit that
   skips the LLM call entirely) works correctly.
2. A REAL AnthropicProvider test, auto-skipped if no ANTHROPIC_API_KEY is
   configured. Unlike Milestones 3/4/6's HuggingFace/Docker gaps, this one
   IS network-reachable from this sandbox (api.anthropic.com is not
   blocked) - the only missing piece is a credential. If you run this suite
   with a real key set, this test executes for real and proves the full
   citation-grounded pipeline against an actual Claude call.
"""
import uuid

import pytest

from app.core.config import get_settings
from app.schemas.chunk import Chunk, ChunkingResult, PageSpan, SplitReason
from app.schemas.document import DocumentMetadata, DocumentType, ExtractionMethod, Page, PageMetadata, ParsedDocument
from app.services.hybrid_search import HybridSearchService
from app.services.keyword_search import BM25LocalProvider
from app.services.llm_provider import LLMProvider, get_llm_provider
from app.services.qa_service import QAService
from app.services.reranker import get_reranker
from app.services.retriever import RetrievalService
from app.services.vector_store import VectorStore
from tests.test_vector_store import FakeEmbeddingProvider


class FakeLLMProvider(LLMProvider):
    """Deterministic stand-in LLM for testing prompt->answer->citation
    orchestration without any real API call. Records every call it
    received so tests can assert on call count (e.g. verifying the
    zero-evidence short-circuit never calls the LLM at all)."""

    def __init__(self, fixed_response: str):
        self.fixed_response = fixed_response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_message: str) -> str:
        self.calls.append((system_prompt, user_message))
        return self.fixed_response

    def generate_stream(self, system_prompt: str, user_message: str):
        self.calls.append((system_prompt, user_message))
        # Yield word-by-word to simulate real token streaming.
        words = self.fixed_response.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word


@pytest.fixture
def fake_document() -> ParsedDocument:
    return ParsedDocument(
        document_id=str(uuid.uuid4()),
        metadata=DocumentMetadata(
            original_filename="test.pdf", document_type=DocumentType.JUDGMENT,
            total_pages=1, file_size_bytes=100, file_hash="abc123",
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


def _build_retrieval_service(tmp_path, monkeypatch, fake_document, fake_chunking_result, min_confidence="0.0"):
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant_qa_test"))
    monkeypatch.setenv("BM25_LOCAL_STORAGE_PATH", str(tmp_path / "bm25_qa_test"))
    monkeypatch.setenv("RERANK_MIN_CONFIDENCE", min_confidence)
    get_settings.cache_clear()

    embedder = FakeEmbeddingProvider(dim=8)
    vector_store = VectorStore(vector_dimension=8)
    keyword_provider = BM25LocalProvider()
    vectors = embedder.embed_texts([c.text for c in fake_chunking_result.chunks])
    vector_store.index_chunks(fake_document, fake_chunking_result, vectors)
    keyword_provider.index_chunks(fake_document, fake_chunking_result)

    hybrid_service = HybridSearchService(embedding_provider=embedder, vector_store=vector_store, keyword_provider=keyword_provider)
    return RetrievalService(hybrid_search_service=hybrid_service, reranker=get_reranker("lightweight"), embedding_provider=embedder)


class TestQAServiceWithFakeLLM:
    def test_answer_with_evidence_produces_correct_citations(self, tmp_path, monkeypatch, fake_document, fake_chunking_result):
        retrieval_service = _build_retrieval_service(tmp_path, monkeypatch, fake_document, fake_chunking_result)
        fake_llm = FakeLLMProvider(
            fixed_response="A valid registration number is required on the donation receipt [C1]."
        )
        qa = QAService(retrieval_service=retrieval_service, llm_provider=fake_llm)

        response = qa.answer("What does Section 80G require for donation receipts?", top_k=3)

        assert response.has_sufficient_evidence is True
        assert len(response.citations) == 1
        assert response.citations[0].page_start == 3
        assert len(fake_llm.calls) == 1  # LLM was actually called since evidence existed

        get_settings.cache_clear()

    def test_no_evidence_response_produces_no_citations(self, tmp_path, monkeypatch, fake_document, fake_chunking_result):
        retrieval_service = _build_retrieval_service(tmp_path, monkeypatch, fake_document, fake_chunking_result)
        settings = get_settings()
        fake_llm = FakeLLMProvider(fixed_response=settings.no_evidence_phrase)
        qa = QAService(retrieval_service=retrieval_service, llm_provider=fake_llm)

        response = qa.answer("What is the airspeed velocity of an unladen swallow?", top_k=3)

        assert response.has_sufficient_evidence is False
        assert response.citations == []
        assert response.answer == settings.no_evidence_phrase

        get_settings.cache_clear()

    def test_zero_retrieved_chunks_never_calls_llm(self, tmp_path, monkeypatch, fake_document, fake_chunking_result):
        """Unreachable confidence threshold -> retrieval returns nothing ->
        QAService must short-circuit to the no-evidence answer WITHOUT
        calling the LLM at all - deterministic, free, and impossible for
        the model to get wrong by guessing."""
        retrieval_service = _build_retrieval_service(
            tmp_path, monkeypatch, fake_document, fake_chunking_result, min_confidence="0.999"
        )
        fake_llm = FakeLLMProvider(fixed_response="this should never be returned")
        qa = QAService(retrieval_service=retrieval_service, llm_provider=fake_llm)

        response = qa.answer("Section 80G registration", top_k=3)

        assert response.has_sufficient_evidence is False
        assert response.answer == get_settings().no_evidence_phrase
        assert len(fake_llm.calls) == 0  # the key assertion: LLM was never invoked
        assert response.model_used == "none (no evidence retrieved)"

        get_settings.cache_clear()

    def test_uncited_claim_treated_as_insufficient_evidence(self, tmp_path, monkeypatch, fake_document, fake_chunking_result):
        """If the model answers without any [Cx] tag at all - a prompt
        violation - the system should not present that as a trustworthy,
        cited answer."""
        retrieval_service = _build_retrieval_service(tmp_path, monkeypatch, fake_document, fake_chunking_result)
        fake_llm = FakeLLMProvider(fixed_response="A registration number is required.")  # no citation tag
        qa = QAService(retrieval_service=retrieval_service, llm_provider=fake_llm)

        response = qa.answer("What does Section 80G require?", top_k=3)
        assert response.has_sufficient_evidence is False
        assert response.citations == []

        get_settings.cache_clear()


class TestQAServiceStreaming:
    def test_stream_yields_tokens_then_final_done_event_with_citations(
        self, tmp_path, monkeypatch, fake_document, fake_chunking_result
    ):
        retrieval_service = _build_retrieval_service(tmp_path, monkeypatch, fake_document, fake_chunking_result)
        fake_llm = FakeLLMProvider(
            fixed_response="A valid registration number is required on the donation receipt [C1]."
        )
        qa = QAService(retrieval_service=retrieval_service, llm_provider=fake_llm)

        events = list(qa.answer_stream("What does Section 80G require?", top_k=3))

        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(token_events) > 1  # streamed as multiple deltas, not one blob
        assert len(done_events) == 1
        # Reassembling the streamed tokens must equal the full answer.
        assert "".join(e["text"] for e in token_events) == fake_llm.fixed_response

        done = done_events[0]
        assert done["has_sufficient_evidence"] is True
        assert len(done["citations"]) == 1
        assert done["citations"][0]["page_start"] == 3

        get_settings.cache_clear()

    def test_stream_zero_evidence_never_calls_llm_stream(
        self, tmp_path, monkeypatch, fake_document, fake_chunking_result
    ):
        retrieval_service = _build_retrieval_service(
            tmp_path, monkeypatch, fake_document, fake_chunking_result, min_confidence="0.999"
        )
        fake_llm = FakeLLMProvider(fixed_response="should never be reached")
        qa = QAService(retrieval_service=retrieval_service, llm_provider=fake_llm)

        events = list(qa.answer_stream("Section 80G registration", top_k=3))

        assert len(fake_llm.calls) == 0
        done_events = [e for e in events if e["type"] == "done"]
        assert done_events[0]["has_sufficient_evidence"] is False
        assert done_events[0]["model_used"] == "none (no evidence retrieved)"

        get_settings.cache_clear()


@pytest.mark.skipif(
    not get_settings().anthropic_api_key,
    reason="ANTHROPIC_API_KEY not configured - network path is open in this sandbox, but no credential is available. "
    "Set ANTHROPIC_API_KEY to run this real end-to-end test.",
)
class TestRealAnthropicProvider:
    def test_real_anthropic_call_produces_grounded_citation(self, tmp_path, monkeypatch, fake_document, fake_chunking_result):
        retrieval_service = _build_retrieval_service(tmp_path, monkeypatch, fake_document, fake_chunking_result)
        real_llm = get_llm_provider("anthropic")
        qa = QAService(retrieval_service=retrieval_service, llm_provider=real_llm)

        response = qa.answer("What does Section 80G require for a donation receipt?", top_k=3)

        assert response.has_sufficient_evidence is True
        assert len(response.citations) >= 1
        assert response.citations[0].page_start == 3

        get_settings.cache_clear()
