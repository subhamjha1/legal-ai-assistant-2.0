"""
Evaluation runner (Milestone 9).

Runs each golden question through the real pipeline (RetrievalService +
QAService) exactly as production traffic would, capturing everything
metrics.py needs: the retrieved chunk ranking (for Recall@K/MRR/nDCG), the
generated answer and its citations (for answer correctness and citation
precision/recall), latency (retrieval vs. generation, separately timed),
and token usage when the provider reports it.

Accepts injected RetrievalService/QAService (same dependency-injection
pattern used throughout this project) so:
  - production use just calls run_evaluation(dataset) with no arguments
    and gets the real, configured pipeline.
  - tests can inject a QAService built on a FakeLLMProvider (as used
    throughout Milestones 5-7's test suites) to prove the runner itself -
    latency capture, error handling, metric computation, aggregation -
    works correctly without needing a real API key or network call.
"""
import time
from datetime import datetime, timezone

from app.services.qa_service import QAService
from app.services.retriever import RetrievalService
from evaluation import metrics
from evaluation.schema import (
    EvaluationConfig,
    EvaluationSummary,
    GoldenQuestion,
    PerQuestionResult,
    RetrievedChunkRef,
)


def run_evaluation(
    questions: list[GoldenQuestion],
    config: EvaluationConfig | None = None,
    qa_service: QAService | None = None,
    retrieval_service: RetrievalService | None = None,
) -> EvaluationSummary:
    config = config or EvaluationConfig()
    qa_service = qa_service or QAService()
    retrieval_service = retrieval_service or getattr(qa_service, "retrieval_service", None) or RetrievalService()

    run_started_at = datetime.now(timezone.utc).isoformat()
    results: list[PerQuestionResult] = []

    for question in questions:
        result = _run_one_question(question, config, qa_service, retrieval_service)
        result.metrics = metrics.compute_all_metrics(result, retrieval_k=config.retrieval_k_for_recall)
        results.append(result)

    run_finished_at = datetime.now(timezone.utc).isoformat()
    aggregate_metrics = metrics.aggregate(results)
    model_used = type(qa_service.llm_provider).__name__

    return EvaluationSummary(
        config=config,
        total_questions=len(questions),
        results=results,
        aggregate_metrics=aggregate_metrics,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        model_used=model_used,
    )


def _run_one_question(
    question: GoldenQuestion,
    config: EvaluationConfig,
    qa_service: QAService,
    retrieval_service: RetrievalService,
) -> PerQuestionResult:
    total_start = time.monotonic()

    try:
        retrieval_start = time.monotonic()
        retrieved_chunks, _considered, _dropped = retrieval_service.retrieve(
            question.query, top_k=config.retrieval_k_for_recall
        )
        retrieval_latency = time.monotonic() - retrieval_start

        retrieved_refs = [
            RetrievedChunkRef(document=c.original_filename, page_start=c.page_start, page_end=c.page_end, rank=i + 1)
            for i, c in enumerate(retrieved_chunks)
        ]

        generation_start = time.monotonic()
        answer_response = qa_service.answer(question.query, top_k=config.top_k)
        generation_latency = time.monotonic() - generation_start

        total_latency = time.monotonic() - total_start

        return PerQuestionResult(
            question_id=question.id,
            query=question.query,
            category=question.category,
            generated_answer=answer_response.answer,
            ground_truth_answer=question.ground_truth_answer,
            has_sufficient_evidence=answer_response.has_sufficient_evidence,
            expected_no_evidence=question.expects_no_evidence,
            actual_citations=answer_response.citations,
            expected_citations=question.expected_citations,
            retrieved_chunks=retrieved_refs,
            total_latency_seconds=total_latency,
            retrieval_latency_seconds=retrieval_latency,
            generation_latency_seconds=generation_latency,
            token_usage=answer_response.token_usage,
        )
    except Exception as exc:  # noqa: BLE001 - one bad question must not kill the whole run
        total_latency = time.monotonic() - total_start
        return PerQuestionResult(
            question_id=question.id,
            query=question.query,
            category=question.category,
            generated_answer="",
            ground_truth_answer=question.ground_truth_answer,
            has_sufficient_evidence=False,
            expected_no_evidence=question.expects_no_evidence,
            actual_citations=[],
            expected_citations=question.expected_citations,
            retrieved_chunks=[],
            total_latency_seconds=total_latency,
            retrieval_latency_seconds=0.0,
            error=str(exc),
        )
