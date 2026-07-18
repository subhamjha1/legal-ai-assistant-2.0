"""
Tests for evaluation/metrics.py (Milestone 9).

All pure functions, so every metric is tested directly against
hand-constructed PerQuestionResult objects - no retrieval, no LLM, no
network needed at all.
"""
import pytest

from app.schemas.qa import Citation, TokenUsage
from evaluation.metrics import (
    aggregate,
    answer_correctness,
    citation_precision,
    citation_recall,
    compute_all_metrics,
    exact_match,
    faithfulness,
    hallucinated,
    mean_reciprocal_rank,
    ndcg_at_k,
    retrieval_recall_at_k,
    token_f1,
)
from evaluation.schema import ExpectedCitation, PerQuestionResult, RetrievedChunkRef


def _citation(doc="test.pdf", start=3, end=3, ref="C1") -> Citation:
    return Citation(chunk_ref=ref, document=doc, page_start=start, page_end=end, snippet="snippet")


def _expected(doc="test.pdf", start=3, end=3) -> ExpectedCitation:
    return ExpectedCitation(document=doc, page_start=start, page_end=end)


def _chunk_ref(rank, doc="test.pdf", start=3, end=3) -> RetrievedChunkRef:
    return RetrievedChunkRef(document=doc, page_start=start, page_end=end, rank=rank)


def _result(**overrides) -> PerQuestionResult:
    defaults = dict(
        question_id="q1",
        query="What does Section 80G require?",
        category=None,
        generated_answer="A registration number is required [C1].",
        ground_truth_answer="A registration number is required on the receipt.",
        has_sufficient_evidence=True,
        expected_no_evidence=False,
        actual_citations=[_citation()],
        expected_citations=[_expected()],
        retrieved_chunks=[_chunk_ref(1)],
        total_latency_seconds=1.2,
        retrieval_latency_seconds=0.3,
    )
    defaults.update(overrides)
    return PerQuestionResult(**defaults)


class TestTokenF1AndExactMatch:
    def test_identical_text_scores_one(self):
        assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)
        assert exact_match("The Cat Sat", "the cat sat") == 1.0  # case-insensitive

    def test_completely_different_text_scores_zero(self):
        assert token_f1("quantum physics", "legal deduction rules") == 0.0

    def test_partial_overlap_scores_between_zero_and_one(self):
        score = token_f1("the cat sat on the mat", "the cat sat")
        assert 0.0 < score < 1.0

    def test_empty_strings_score_zero(self):
        assert token_f1("", "something") == 0.0
        assert token_f1("something", "") == 0.0


class TestAnswerCorrectness:
    def test_no_evidence_check_rewards_correct_refusal(self):
        result = _result(expected_no_evidence=True, has_sufficient_evidence=False)
        assert answer_correctness(result) == 1.0

    def test_no_evidence_check_penalizes_hallucinated_answer(self):
        result = _result(expected_no_evidence=True, has_sufficient_evidence=True)
        assert answer_correctness(result) == 0.0

    def test_normal_question_uses_token_f1(self):
        result = _result(generated_answer="exact match text", ground_truth_answer="exact match text")
        assert answer_correctness(result) == pytest.approx(1.0)


class TestCitationPrecisionRecall:
    def test_perfect_match_scores_one_both_ways(self):
        result = _result()
        assert citation_precision(result) == 1.0
        assert citation_recall(result) == 1.0

    def test_wrong_document_scores_zero_precision(self):
        result = _result(actual_citations=[_citation(doc="wrong.pdf")])
        assert citation_precision(result) == 0.0

    def test_non_overlapping_page_scores_zero(self):
        result = _result(actual_citations=[_citation(start=99, end=99)])
        assert citation_precision(result) == 0.0

    def test_overlapping_but_not_identical_page_range_still_matches(self):
        result = _result(
            expected_citations=[_expected(start=3, end=5)],
            actual_citations=[_citation(start=4, end=4)],
        )
        assert citation_precision(result) == 1.0

    def test_extra_uncited_expected_passage_reduces_recall(self):
        result = _result(
            expected_citations=[_expected(start=3, end=3), _expected(start=10, end=10)],
            actual_citations=[_citation(start=3, end=3)],
        )
        assert citation_recall(result) == 0.5

    def test_no_evidence_check_with_no_citations_scores_perfect_precision(self):
        result = _result(expected_no_evidence=True, actual_citations=[])
        assert citation_precision(result) == 1.0

    def test_no_evidence_check_that_hallucinates_a_citation_scores_zero_precision(self):
        result = _result(expected_no_evidence=True, actual_citations=[_citation()])
        assert citation_precision(result) == 0.0


class TestRetrievalMetrics:
    def test_recall_at_k_finds_relevant_chunk_within_k(self):
        result = _result(retrieved_chunks=[_chunk_ref(1, start=3, end=3)])
        assert retrieval_recall_at_k(result, k=5) == 1.0

    def test_recall_at_k_misses_relevant_chunk_beyond_k(self):
        result = _result(retrieved_chunks=[_chunk_ref(10, start=3, end=3)])
        assert retrieval_recall_at_k(result, k=5) == 0.0

    def test_mrr_rewards_earlier_rank(self):
        result_rank_1 = _result(retrieved_chunks=[_chunk_ref(1, start=3, end=3)])
        result_rank_4 = _result(retrieved_chunks=[_chunk_ref(4, start=3, end=3)])
        assert mean_reciprocal_rank(result_rank_1) == pytest.approx(1.0)
        assert mean_reciprocal_rank(result_rank_4) == pytest.approx(0.25)

    def test_mrr_zero_when_nothing_relevant_retrieved(self):
        result = _result(retrieved_chunks=[_chunk_ref(1, start=99, end=99)])
        assert mean_reciprocal_rank(result) == 0.0

    def test_ndcg_perfect_when_relevant_chunk_ranked_first(self):
        result = _result(retrieved_chunks=[_chunk_ref(1, start=3, end=3), _chunk_ref(2, start=99, end=99)])
        assert ndcg_at_k(result, k=5) == pytest.approx(1.0)

    def test_ndcg_lower_when_relevant_chunk_ranked_later(self):
        result_first = _result(retrieved_chunks=[_chunk_ref(1, start=3, end=3)])
        result_later = _result(
            retrieved_chunks=[_chunk_ref(1, start=99, end=99), _chunk_ref(2, start=3, end=3)]
        )
        assert ndcg_at_k(result_later, k=5) < ndcg_at_k(result_first, k=5)


class TestFaithfulness:
    def test_every_sentence_cited_scores_one(self):
        result = _result(generated_answer="Claim one is true [C1]. Claim two is also true [C2].")
        assert faithfulness(result) == 1.0

    def test_uncited_sentence_lowers_score(self):
        result = _result(generated_answer="Claim one is true [C1]. This claim has no citation at all.")
        assert faithfulness(result) == 0.5

    def test_no_evidence_check_correctly_refusing_scores_one(self):
        result = _result(
            expected_no_evidence=True, has_sufficient_evidence=False, generated_answer="I could not find supporting evidence."
        )
        assert faithfulness(result) == 1.0


class TestHallucinationRate:
    def test_fully_cited_correct_answer_is_not_hallucinated(self):
        result = _result()  # default fixture: fully cited, citations match
        assert hallucinated(result) == 0.0

    def test_uncited_claim_counts_as_hallucination(self):
        result = _result(generated_answer="This claim has no citation tag at all.")
        assert hallucinated(result) == 1.0

    def test_wrong_citation_counts_as_hallucination(self):
        result = _result(
            generated_answer="A registration number is required [C1].",
            actual_citations=[_citation(doc="wrong_document.pdf")],
        )
        assert hallucinated(result) == 1.0

    def test_correct_refusal_on_no_evidence_check_is_not_hallucination(self):
        result = _result(expected_no_evidence=True, has_sufficient_evidence=False)
        assert hallucinated(result) == 0.0

    def test_answering_when_should_have_refused_is_hallucination(self):
        result = _result(expected_no_evidence=True, has_sufficient_evidence=True)
        assert hallucinated(result) == 1.0


class TestAggregate:
    def test_averages_computed_correctly_across_questions(self):
        r1 = _result(question_id="q1")
        r1.metrics = compute_all_metrics(r1, retrieval_k=5)
        r2 = _result(question_id="q2", actual_citations=[])
        r2.metrics = compute_all_metrics(r2, retrieval_k=5)

        summary = aggregate([r1, r2])
        assert summary["total_questions"] == 2.0
        assert "avg_citation_precision" in summary
        # r1 has perfect citation precision (1.0), r2 has zero (no citations) -> avg 0.5
        assert summary["avg_citation_precision"] == pytest.approx(0.5)

    def test_token_usage_totals_summed_correctly(self):
        r1 = _result(question_id="q1", token_usage=TokenUsage(input_tokens=100, output_tokens=50))
        r1.metrics = compute_all_metrics(r1, retrieval_k=5)
        r2 = _result(question_id="q2", token_usage=TokenUsage(input_tokens=200, output_tokens=80))
        r2.metrics = compute_all_metrics(r2, retrieval_k=5)

        summary = aggregate([r1, r2])
        assert summary["total_input_tokens"] == 300.0
        assert summary["total_output_tokens"] == 130.0
        assert summary["questions_with_token_usage"] == 2.0

    def test_empty_results_returns_empty_dict(self):
        assert aggregate([]) == {}
