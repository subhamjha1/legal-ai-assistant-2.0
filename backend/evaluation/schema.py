"""
Golden dataset schema (Milestone 9).

Field names deliberately mirror the assignment brief's own vocabulary
(Sample Query, Ground Truth Answer, Source Document, Relevant Page
Number(s)) so the dataset is directly traceable to that spec, while adding
the richer `expected_citations` structure this project's citation formatter
actually needs to be evaluated precisely (a claim can legitimately draw on
more than one passage, and page *ranges* matter, not just single pages).
"""
from pydantic import BaseModel, Field

from app.schemas.document import DocumentType
from app.schemas.qa import Citation, TokenUsage


class ExpectedCitation(BaseModel):
    """One passage the ground-truth answer should be able to cite. Page
    matching in metrics.py is range-overlap based (an actual citation
    "hits" this if it names the same document and its page range overlaps
    this one), not exact-equality - a real chunk boundary rarely lines up
    exactly with a hand-authored expectation."""
    document: str
    page_start: int
    page_end: int
    structural_label: str | None = None


class GoldenQuestion(BaseModel):
    id: str
    query: str = Field(description="Sample Query")
    ground_truth_answer: str = Field(description="Ground Truth Answer")
    source_document: str = Field(description="Source Document (primary)")
    relevant_pages: list[int] = Field(
        default_factory=list, description="Relevant Page Number(s), flattened"
    )
    expected_citations: list[ExpectedCitation] = Field(default_factory=list)
    document_type: DocumentType | None = None
    category: str | None = Field(
        default=None,
        description="e.g. 'fact_lookup', 'statutory_requirement', 'holding', 'no_evidence_check'",
    )
    notes: str | None = None

    @property
    def expects_no_evidence(self) -> bool:
        """A 'no evidence' check question deliberately asks something the
        corpus cannot answer, to verify the system's fallback behavior
        (Milestone 7) rather than its retrieval quality. These are tagged
        by category and have no expected citations at all."""
        return self.category == "no_evidence_check"


class EvaluationConfig(BaseModel):
    """Runtime configuration for one evaluation pass - kept separate from
    the dataset itself so the same golden set can be run under different
    top_k / provider settings without editing the dataset file."""
    top_k: int = 5
    retrieval_k_for_recall: int = 5  # the "K" in Recall@K / nDCG@K
    latency_budget_seconds: float = 15.0  # flagged (not failed) if exceeded
    min_avg_answer_correctness: float = 0.0  # CI gate; 0.0 = no gate by default
    min_avg_faithfulness: float = 0.0  # CI gate; 0.0 = no gate by default


class RetrievedChunkRef(BaseModel):
    """Minimal record of one retrieved chunk, kept in the per-question
    result for recall/MRR/nDCG computation without depending on the full
    RetrievedChunk schema (keeps evaluation/ decoupled from app/ internals
    beyond what it actually needs)."""
    document: str
    page_start: int
    page_end: int
    rank: int  # 1-indexed position in the retrieved list


class PerQuestionResult(BaseModel):
    """Everything computed for one golden question - the raw material
    metrics.py's pure functions operate on, and what gets serialized to the
    JSON export for later re-analysis."""
    question_id: str
    query: str
    category: str | None = None

    generated_answer: str
    ground_truth_answer: str
    has_sufficient_evidence: bool
    expected_no_evidence: bool

    actual_citations: list[Citation]
    expected_citations: list[ExpectedCitation]
    retrieved_chunks: list[RetrievedChunkRef]

    total_latency_seconds: float
    retrieval_latency_seconds: float
    generation_latency_seconds: float | None = None
    token_usage: TokenUsage | None = None

    error: str | None = None

    # Populated by metrics.py after the raw result is collected.
    metrics: dict[str, float] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    """Aggregate report: per-question results plus dataset-wide averages,
    ready for report.py to render and cli.py to gate CI on."""
    config: EvaluationConfig
    total_questions: int
    results: list[PerQuestionResult]
    aggregate_metrics: dict[str, float]
    run_started_at: str
    run_finished_at: str
    model_used: str
