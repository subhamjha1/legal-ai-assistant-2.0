"""
Pydantic schemas for chunking (Milestone 2).

Why this exists:
A chunk is what actually gets embedded and retrieved - it is the unit the
LLM sees at answer time. If a chunk doesn't know exactly which page(s) it
came from, the citation formatter downstream has nothing accurate to cite.
This schema is deliberately redundant with page info (both a range and the
per-page breakdown) because a chunk can legitimately span a page boundary,
and "page 4-5" vs "mostly page 4, one sentence spills to page 5" are both
things a careful citation might need.
"""
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class SplitReason(str, Enum):
    STRUCTURAL_MARKER = "structural_marker"  # split at a Section/Article/¶ boundary
    SIZE_CAP = "size_cap"                    # split because a section exceeded max size
    DOCUMENT_BOUNDARY = "document_boundary"  # the whole doc had no structure, one pass


class PageSpan(BaseModel):
    """How much of a chunk's text falls on each page it touches."""
    page_number: int
    char_count: int


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_index: int  # 0-indexed position within the document's chunk sequence
    text: str
    structural_label: str | None = None  # e.g. "Section 80G(5)(iv)", "Paragraph 4", None if unlabeled
    split_reason: SplitReason
    page_start: int
    page_end: int
    page_spans: list[PageSpan]
    char_count: int
    overlaps_previous: bool = False


class ChunkingResult(BaseModel):
    document_id: str
    total_chunks: int
    chunks: list[Chunk]
    structural_markers_found: int
