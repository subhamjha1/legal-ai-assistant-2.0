from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryFilters(BaseModel):
    document_type: Optional[str] = None
    year: Optional[int] = None
    court: Optional[str] = None
    act_name: Optional[str] = None
    judge: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    filters: Optional[QueryFilters] = None
    top_k: Optional[int] = None
    use_graph_expansion: bool = True


class SupportingCitation(BaseModel):
    document_name: str
    page: str
    section: Optional[str] = None
    quote_or_paraphrase: str


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    filename: str
    page_start: int
    page_end: int
    section: Optional[str] = None
    rerank_score: float
    citation_label: str


class QueryResponse(BaseModel):
    answer: str
    summary: str
    supporting_citations: list[dict]
    confidence_note: str
    insufficient_evidence: bool
    retrieval_confidence: float
    retrieved_chunks: list[dict]


class RetrieveResponse(BaseModel):
    chunk_id: str
    filename: str
    document_type: str
    page_start: int
    page_end: int
    section: Optional[str]
    heading: Optional[str]
    text: str
    score: float


class SummarizeRequest(BaseModel):
    document_id: str
    max_summary_words: int = 400


class SummarizeResponse(BaseModel):
    document_id: str
    title: str
    summary: str


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    title: str
    document_type: str
    year: Optional[int]
    court: Optional[str]
    page_count: int


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    document_type: str
    num_child_chunks: int
    num_parent_chunks: int
    status: str


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int
    comment: Optional[str] = None
    retrieved_chunk_ids: list[str] = []


class HealthResponse(BaseModel):
    status: str
    faiss_vectors: int
    bm25_documents: int
    elasticsearch_enabled: bool
    neo4j_enabled: bool
