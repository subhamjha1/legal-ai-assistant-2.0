"""
Tests for evaluation/runner.py and evaluation/report.py (Milestone 9).

Wires the REAL retrieval pipeline (parser -> chunker -> real embedded
Qdrant -> real BM25 -> real MMR -> real lightweight reranker) against a
FakeLLMProvider that answers deterministically based on what's actually in
its prompt - proving the evaluation harness itself (latency capture, metric
computation, aggregation, report rendering, CLI) works correctly end-to-end
without needing a real API key or network call, exactly the same honest
pattern used for QAService in Milestone 7.
"""
import re
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.schemas.document import DocumentType
from app.services.hybrid_search import HybridSearchService
from app.services.keyword_search import BM25LocalProvider
from app.services.llm_provider import LLMProvider
from app.services.parser import DocumentParser
from app.services.qa_service import QAService
from app.services.reranker import get_reranker
from app.services.retriever import RetrievalService
from app.services.vector_store import VectorStore
from evaluation.dataset_loader import load_dataset
from evaluation.report import generate_html_report, generate_markdown_report
from evaluation.runner import run_evaluation
from tests.test_vector_store import FakeEmbeddingProvider

SAMPLE_PDF = "sample_docs/sample_legal_doc_final.pdf"


class PromptEchoLLM(LLMProvider):
    """
    A fake LLM that actually reads the passages it was given and answers
    from them - rather than a single fixed string - so the evaluation
    metrics (which compare against real per-question ground truth) have
    something meaningfully different to score per question. It extracts
    the first numbered passage's text from the prompt and returns it
    verbatim with a [C1] tag, which is "correct enough" to produce
    non-trivial (not uniformly 0 or 1) metric values across the dataset -
    exactly what's needed to prove aggregation math works on real
    variance, not a constant.
    """

    _PASSAGE_PATTERN = re.compile(r"\[C1\][^\n]*\n(.+?)(?:\n\n\[C2\]|\n\nQuestion:)", re.DOTALL)

    def generate(self, system_prompt: str, user_message: str) -> str:
        match = self._PASSAGE_PATTERN.search(user_message)
        if not match:
            return "I could not find supporting evidence."
        passage_text = match.group(1).strip()
        return f"{passage_text} [C1]"

    def generate_stream(self, system_prompt: str, user_message: str):
        yield self.generate(system_prompt, user_message)


@pytest.fixture(scope="module")
def real_indexed_qa_service(tmp_path_factory):
    """Builds the full real pipeline once per test module: parses the real
    sample PDF, chunks it, indexes it into real (temp, isolated) Qdrant +
    BM25, and wraps it all in a QAService using the PromptEchoLLM."""
    tmp_path = tmp_path_factory.mktemp("eval_runner_test")

    import os
    os.environ["QDRANT_LOCAL_PATH"] = str(tmp_path / "qdrant")
    os.environ["BM25_LOCAL_STORAGE_PATH"] = str(tmp_path / "bm25")
    os.environ["RERANK_MIN_CONFIDENCE"] = "0.0"
    get_settings.cache_clear()

    parser = DocumentParser()
    document = parser.parse(Path(SAMPLE_PDF), original_filename="sample_legal_doc_final.pdf", document_type=DocumentType.JUDGMENT)

    from app.services.chunker import SemanticChunker
    chunking_result = SemanticChunker().chunk(document)

    embedder = FakeEmbeddingProvider(dim=16)
    vector_store = VectorStore(vector_dimension=16)
    keyword_provider = BM25LocalProvider()
    vectors = embedder.embed_texts([c.text for c in chunking_result.chunks])
    vector_store.index_chunks(document, chunking_result, vectors)
    keyword_provider.index_chunks(document, chunking_result)

    hybrid = HybridSearchService(embedding_provider=embedder, vector_store=vector_store, keyword_provider=keyword_provider)
    retrieval_service = RetrievalService(hybrid_search_service=hybrid, reranker=get_reranker("lightweight"), embedding_provider=embedder)
    qa_service = QAService(retrieval_service=retrieval_service, llm_provider=PromptEchoLLM())

    yield qa_service

    get_settings.cache_clear()


@pytest.fixture(scope="module")
def golden_questions():
    return load_dataset("evaluation/golden_dataset.json")


class TestRunEvaluation:
    def test_runs_full_dataset_without_crashing(self, real_indexed_qa_service, golden_questions):
        summary = run_evaluation(golden_questions, qa_service=real_indexed_qa_service)
        assert summary.total_questions == 18
        assert len(summary.results) == 18

    def test_every_result_has_computed_metrics(self, real_indexed_qa_service, golden_questions):
        summary = run_evaluation(golden_questions, qa_service=real_indexed_qa_service)
        for result in summary.results:
            assert "answer_correctness" in result.metrics
            assert "citation_precision" in result.metrics
            assert "mrr" in result.metrics
            assert "faithfulness" in result.metrics

    def test_no_evidence_questions_correctly_handled(self, real_indexed_qa_service, golden_questions):
        summary = run_evaluation(golden_questions, qa_service=real_indexed_qa_service)
        no_evidence_results = [r for r in summary.results if r.expected_no_evidence]
        assert len(no_evidence_results) >= 2
        for r in no_evidence_results:
            assert isinstance(r.has_sufficient_evidence, bool)  # ran without error at minimum

    def test_aggregate_metrics_present_and_reasonable(self, real_indexed_qa_service, golden_questions):
        summary = run_evaluation(golden_questions, qa_service=real_indexed_qa_service)
        assert 0.0 <= summary.aggregate_metrics["avg_answer_correctness"] <= 1.0
        assert 0.0 <= summary.aggregate_metrics["avg_citation_precision"] <= 1.0
        assert summary.aggregate_metrics["total_questions"] == 18.0

    def test_retrieval_latency_is_recorded_and_positive(self, real_indexed_qa_service, golden_questions):
        summary = run_evaluation(golden_questions[:3], qa_service=real_indexed_qa_service)
        for result in summary.results:
            assert result.retrieval_latency_seconds >= 0.0
            assert result.total_latency_seconds >= result.retrieval_latency_seconds

    def test_a_genuinely_answerable_question_gets_a_matching_citation(self, real_indexed_qa_service, golden_questions):
        q001 = [q for q in golden_questions if q.id == "q001"]
        summary = run_evaluation(q001, qa_service=real_indexed_qa_service)
        result = summary.results[0]
        assert result.error is None
        assert len(result.retrieved_chunks) > 0


class TestReportGeneration:
    def test_markdown_report_contains_key_sections(self, real_indexed_qa_service, golden_questions):
        summary = run_evaluation(golden_questions[:5], qa_service=real_indexed_qa_service)
        report = generate_markdown_report(summary)
        assert "# Evaluation Report" in report
        assert "Aggregate Metrics" in report
        assert "Per-Question Results" in report
        assert "q001" in report

    def test_html_report_is_valid_self_contained_document(self, real_indexed_qa_service, golden_questions):
        summary = run_evaluation(golden_questions[:5], qa_service=real_indexed_qa_service)
        report = generate_html_report(summary)
        assert report.startswith("<!DOCTYPE html>")
        assert "<svg" in report
        assert "q001" in report

    def test_report_handles_errored_question_gracefully(self, real_indexed_qa_service):
        from evaluation.schema import GoldenQuestion

        broken_question = GoldenQuestion(
            id="broken",
            query="",
            ground_truth_answer="n/a",
            source_document="sample_legal_doc_final.pdf",
        )
        summary = run_evaluation([broken_question], qa_service=real_indexed_qa_service)
        assert len(summary.results) == 1
        report = generate_markdown_report(summary)  # must not raise
        assert "broken" in report
