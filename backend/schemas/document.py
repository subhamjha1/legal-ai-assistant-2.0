from __future__ import annotations

import hashlib
import uuid
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    ACT = "act"
    JUDGMENT = "judgment"
    COMMENTARY = "commentary"
    IRS_REGULATION = "irs_regulation"
    OTHER = "other"


class RelationType(str, Enum):
    CITES = "cites"
    AMENDS = "amends"
    REFERENCES = "references"
    IMPLEMENTS = "implements"
    OVERRULES = "overrules"
    INTERPRETS = "interprets"


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    title: str
    document_type: DocumentType
    year: Optional[int] = None
    court: Optional[str] = None
    judge: Optional[str] = None
    jurisdiction: Optional[str] = "US"
    act_name: Optional[str] = None
    citation_string: Optional[str] = None
    source_path: str
    page_count: int = 0
    ingested_at: str = Field(default_factory=lambda: date.today().isoformat())
    checksum: Optional[str] = None
    version: int = 1


class Citation(BaseModel):
    raw_text: str
    normalized: Optional[str] = None
    target_document_id: Optional[str] = None


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    parent_chunk_id: Optional[str] = None
    is_parent: bool = False

    text: str
    token_count: int = 0

    filename: str
    document_type: DocumentType
    page_start: int
    page_end: int
    section: Optional[str] = None
    subsection: Optional[str] = None
    heading: Optional[str] = None

    chunk_index: int

    citations: list[Citation] = Field(default_factory=list)

    metadata: dict = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk text must not be empty")
        return v

    @field_validator("page_start", "page_end")
    @classmethod
    def page_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Page numbers must be 1-indexed and positive")
        return v

    def citation_label(self) -> str:
        loc = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        parts = [self.filename, loc]
        if self.section:
            parts.append(f"§{self.section}")
        return ", ".join(parts)


class GraphRelation(BaseModel):
    source_document_id: str
    target_document_id: str
    relation_type: RelationType
    evidence_chunk_id: Optional[str] = None
    confidence: float = 1.0


def new_document_id(filename: str) -> str:
    return "doc_" + hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]


def new_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}_chunk_{chunk_index:05d}"


def new_uuid() -> str:
    return str(uuid.uuid4())
