"""
Pydantic schemas for vector search (Milestone 3).
"""
from pydantic import BaseModel, Field

from app.schemas.document import DocumentType


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    document_id: str | None = None
    document_type: DocumentType | None = None


class SearchHit(BaseModel):
    score: float
    document_id: str
    chunk_id: str
    text: str
    structural_label: str | None = None
    page_start: int
    page_end: int
    original_filename: str
    document_type: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    total_results: int


class HybridSearchHit(BaseModel):
    """A fused result, transparent about how it was found - useful both for
    debugging retrieval quality and for Milestone 6's re-ranker, which may
    want to weigh chunks differently depending on whether they were found
    by both rankers or just one."""
    chunk_id: str
    document_id: str
    text: str
    structural_label: str | None = None
    page_start: int
    page_end: int
    original_filename: str
    document_type: str
    rrf_score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    matched_by: list[str]  # subset of ["vector", "keyword"]


class HybridSearchResponse(BaseModel):
    query: str
    results: list[HybridSearchHit]
    total_results: int


class RetrievedChunk(BaseModel):
    """The final, retrieval-pipeline output: MMR-diversified, re-ranked, and
    confidence-filtered. This is what Milestone 7's LLM prompt will actually
    receive as context - every field it needs for a citation is present."""
    chunk_id: str
    document_id: str
    text: str
    structural_label: str | None = None
    page_start: int
    page_end: int
    original_filename: str
    document_type: str
    confidence: float  # final re-rank confidence, 0-1
    rrf_score: float  # upstream hybrid-search score, kept for transparency


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    total_results: int
    candidates_considered: int  # size of the pool before MMR + rerank + filtering
    below_confidence_threshold_count: int  # how many were dropped for low confidence
