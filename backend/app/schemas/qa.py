"""
Pydantic schemas for Q&A / answer generation (Milestone 7).
"""
from pydantic import BaseModel, Field

from app.schemas.document import DocumentType


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None
    document_type: DocumentType | None = None


class Citation(BaseModel):
    """A citation is always built from OUR retrieval data, never from text
    the LLM generated - the LLM only ever refers to a chunk by its [Cx] tag;
    document name, page numbers, and snippet all come from the retrieval
    pipeline's own trusted records. This is the core anti-hallucination
    guarantee of the citation formatter."""
    chunk_ref: str  # e.g. "C1" - the tag the LLM used inline in its answer
    document: str
    page_start: int
    page_end: int
    structural_label: str | None = None
    snippet: str  # short excerpt of the cited chunk, for the user to verify


class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    has_sufficient_evidence: bool
    chunks_considered: int
    model_used: str
    token_usage: TokenUsage | None = None  # None when the provider doesn't report usage
