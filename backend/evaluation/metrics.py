"""
Evaluation metrics (Milestone 9).

All functions here are pure (no I/O, no LLM calls) so they're fully
unit-testable with hand-constructed inputs. Where a metric is a genuine
heuristic stand-in rather than the "textbook" version (e.g. answer
correctness via token-F1 rather than an LLM-judge), that's called out
explicitly in the docstring - this project's consistent pattern of never
quietly downgrading a metric's rigor without saying so.
"""
import math
import re

from evaluation.schema import ExpectedCitation, PerQuestionResult, RetrievedChunkRef

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _pages_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


# ---------------------------------------------------------------------- #
# Answer correctness
# ---------------------------------------------------------------------- #
def exact_match(generated: str, ground_truth: str) -> float:
    return 1.0 if generated.strip().lower() == ground_truth.strip().lower() else 0.0


def token_f1(generated: str, ground_truth: str) -> float:
    """
    Token-overlap F1 between generated and ground-truth answers - the same
    style of metric used by classic extractive-QA benchmarks (SQuAD).

    This is a DELIBERATE STAND-IN for "true" answer correctness, which in a
    production RAGAS/DeepEval setup would typically use an LLM-as-judge or
    semantic embedding similarity. Token-F1 is fully deterministic and
    requires no model call at all, so it can run in any environment
    (including this one) - but it will under-score a correct answer that's
    phrased very differently from the ground truth, and can over-score a
    wrong answer that happens to reuse a lot of the same words. Treat it as
    a floor, not a ceiling, on real answer quality.
    """
    gen_tokens = _tokenize(generated)
    gt_tokens = _tokenize(ground_truth)
    if not gen_tokens or not gt_tokens:
        return 0.0

    gen_counts: dict[str, int] = {}
    for t in gen_tokens:
        gen_counts[t] = gen_counts.get(t, 0) + 1
    gt_counts: dict[str, int] = {}
    for t in gt_tokens:
        gt_counts[t] = gt_counts.get(t, 0) + 1

    overlap = sum(min(gen_counts.get(t, 0), c) for t, c in gt_counts.items())
    if overlap == 0:
        return 0.0

    precision = overlap / len(gen_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_correctness(result: PerQuestionResult) -> float:
    """
    For 'no_evidence_check' questions, correctness is binary: did the
    system correctly decline (has_sufficient_evidence == False) rather than
    guess. For every other question, correctness is token-F1 against the
    ground truth answer.
    """
    if result.expected_no_evidence:
        return 0.0 if result.has_sufficient_evidence else 1.0
    return token_f1(result.generated_answer, result.ground_truth_answer)


# ---------------------------------------------------------------------- #
# Citation precision / recall
# ---------------------------------------------------------------------- #
def _citation_matches_expected(actual_doc: str, actual_start: int, actual_end: int, expected: ExpectedCitation) -> bool:
    return actual_doc == expected.document and _pages_overlap(actual_start, actual_end, expected.page_start, expected.page_end)


def citation_precision(result: PerQuestionResult) -> float:
    """Fraction of the system's actual citations that match some expected
    citation (same document, overlapping page range)."""
    if result.expected_no_evidence:
        return 1.0 if not result.actual_citations else 0.0  # citing anything here is a false positive
    if not result.actual_citations:
        return 0.0
    matched = sum(
        1
        for c in result.actual_citations
        if any(_citation_matches_expected(c.document, c.page_start, c.page_end, exp) for exp in result.expected_citations)
    )
    return matched / len(result.actual_citations)


def citation_recall(result: PerQuestionResult) -> float:
    """Fraction of expected citations the system actually cited."""
    if result.expected_no_evidence:
        return 1.0  # nothing was expected to be cited, so recall is vacuously perfect
    if not result.expected_citations:
        return 1.0  # nothing to recall
    matched = sum(
        1
        for exp in result.expected_citations
        if any(_citation_matches_expected(c.document, c.page_start, c.page_end, exp) for c in result.actual_citations)
    )
    return matched / len(result.expected_citations)


# ---------------------------------------------------------------------- #
# Retrieval quality: Recall@K, MRR, nDCG@K
# ---------------------------------------------------------------------- #
def _is_relevant(chunk: RetrievedChunkRef, expected_citations: list[ExpectedCitation]) -> bool:
    return any(_citation_matches_expected(chunk.document, chunk.page_start, chunk.page_end, exp) for exp in expected_citations)


def retrieval_recall_at_k(result: PerQuestionResult, k: int) -> float:
    """Of the expected citations, what fraction were found somewhere in the
    top-K retrieved chunks (regardless of the final answer's citations -
    this measures the RETRIEVER, not the LLM)."""
    if result.expected_no_evidence or not result.expected_citations:
        return 1.0
    top_k_chunks = [c for c in result.retrieved_chunks if c.rank <= k]
    matched = sum(
        1 for exp in result.expected_citations if any(_is_relevant(c, [exp]) for c in top_k_chunks)
    )
    return matched / len(result.expected_citations)


def mean_reciprocal_rank(result: PerQuestionResult) -> float:
    """Reciprocal rank of the FIRST retrieved chunk that matches any
    expected citation. 0 if none of the retrieved chunks match."""
    if result.expected_no_evidence or not result.expected_citations:
        return 1.0
    for chunk in sorted(result.retrieved_chunks, key=lambda c: c.rank):
        if _is_relevant(chunk, result.expected_citations):
            return 1.0 / chunk.rank
    return 0.0


def ndcg_at_k(result: PerQuestionResult, k: int) -> float:
    """Normalized Discounted Cumulative Gain over the top-K retrieved
    chunks, using binary relevance (1 if a chunk matches any expected
    citation, else 0) - standard formula, ideal ordering puts every
    relevant chunk first."""
    if result.expected_no_evidence or not result.expected_citations:
        return 1.0

    top_k_chunks = sorted([c for c in result.retrieved_chunks if c.rank <= k], key=lambda c: c.rank)
    relevances = [1.0 if _is_relevant(c, result.expected_citations) else 0.0 for c in top_k_chunks]

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))  # i=0 -> rank 1 -> log2(2)

    num_relevant = min(len(result.expected_citations), k)
    ideal_relevances = [1.0] * num_relevant
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------- #
# Faithfulness / groundedness
# ---------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION_TAG = re.compile(r"\[C\d+\]")


def faithfulness(result: PerQuestionResult) -> float:
    """
    Fraction of the generated answer's sentences that carry at least one
    [Cx] citation tag - a deterministic, offline-computable PROXY for
    faithfulness/groundedness.

    This is explicitly NOT the same as RAGAS's faithfulness metric, which
    uses an LLM judge to check whether each claim is actually ENTAILED by
    its cited source (catching cases where a sentence has a citation tag
    but still misrepresents what the source says). This proxy only checks
    that a citation is present, not that it's accurate - accuracy of the
    citation's page/document is separately guaranteed by the citation
    formatter itself (Milestone 7), but whether the cited passage truly
    supports the specific claim is not verified by this metric. A
    production setup would add an LLM-judge faithfulness pass alongside
    this cheaper structural proxy.
    """
    if result.expected_no_evidence:
        return 1.0 if not result.has_sufficient_evidence else 0.0

    text = result.generated_answer.strip()
    if not text:
        return 0.0

    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return 0.0

    cited = sum(1 for s in sentences if _CITATION_TAG.search(s))
    return cited / len(sentences)


def hallucinated(result: PerQuestionResult) -> float:
    """
    Binary per-question hallucination flag (1.0 = hallucinated, 0.0 = not),
    used to compute the dataset-wide 'Hallucination Rate' for the final
    evaluation report.

    Definition used here (explicitly a defined heuristic, not an industry-
    standard formula - there isn't a single agreed-upon one):
    - For 'no_evidence_check' questions: hallucination = the system
      answered instead of correctly refusing (has_sufficient_evidence is
      True when it should have been False).
    - For every other question: hallucination = the answer contains at
      least one uncited sentence (faithfulness < 1.0) OR at least one
      citation that doesn't actually match an expected passage
      (citation_precision < 1.0 despite having citations at all) - i.e.
      the model asserted something this pipeline's own citation formatter
      cannot verify against real retrieved evidence.
    """
    if result.expected_no_evidence:
        return 1.0 if result.has_sufficient_evidence else 0.0

    has_uncited_claim = faithfulness(result) < 1.0
    has_wrong_citation = bool(result.actual_citations) and citation_precision(result) < 1.0
    return 1.0 if (has_uncited_claim or has_wrong_citation) else 0.0


# ---------------------------------------------------------------------- #
# Orchestration: compute every metric for one result
# ---------------------------------------------------------------------- #
def compute_all_metrics(result: PerQuestionResult, retrieval_k: int) -> dict[str, float]:
    return {
        "exact_match": exact_match(result.generated_answer, result.ground_truth_answer),
        "answer_correctness": answer_correctness(result),
        "citation_precision": citation_precision(result),
        "citation_recall": citation_recall(result),
        f"retrieval_recall_at_{retrieval_k}": retrieval_recall_at_k(result, retrieval_k),
        "mrr": mean_reciprocal_rank(result),
        f"ndcg_at_{retrieval_k}": ndcg_at_k(result, retrieval_k),
        "faithfulness": faithfulness(result),
        "hallucinated": hallucinated(result),
        "latency_seconds": result.total_latency_seconds,
    }


def aggregate(results: list[PerQuestionResult]) -> dict[str, float]:
    """Dataset-wide averages of every per-question metric, plus token usage
    totals where available."""
    if not results:
        return {}

    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.metrics.keys())

    averages = {}
    for key in sorted(all_keys):
        values = [r.metrics[key] for r in results if key in r.metrics]
        averages[f"avg_{key}"] = sum(values) / len(values) if values else 0.0

    total_input_tokens = sum(r.token_usage.input_tokens for r in results if r.token_usage)
    total_output_tokens = sum(r.token_usage.output_tokens for r in results if r.token_usage)
    questions_with_usage = sum(1 for r in results if r.token_usage)

    averages["total_input_tokens"] = float(total_input_tokens)
    averages["total_output_tokens"] = float(total_output_tokens)
    averages["questions_with_token_usage"] = float(questions_with_usage)
    averages["total_questions"] = float(len(results))
    averages["errored_questions"] = float(sum(1 for r in results if r.error))

    return averages
