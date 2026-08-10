from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocumentORM(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), index=True)
    title: Mapped[str] = mapped_column(String(1024))
    document_type: Mapped[str] = mapped_column(String(32), index=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    court: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    judge: Mapped[str | None] = mapped_column(String(256), nullable=True)
    jurisdiction: Mapped[str | None] = mapped_column(String(64), nullable=True)
    act_name: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)
    citation_string: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_path: Mapped[str] = mapped_column(String(1024))
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[str] = mapped_column(String(32))
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chunks: Mapped[list["ChunkORM"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class ChunkORM(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"), index=True)
    parent_chunk_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    is_parent: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    filename: Mapped[str] = mapped_column(String(512))
    document_type: Mapped[str] = mapped_column(String(32), index=True)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    subsection: Mapped[str | None] = mapped_column(String(32), nullable=True)
    heading: Mapped[str | None] = mapped_column(String(512), nullable=True)

    chunk_index: Mapped[int] = mapped_column(Integer, index=True)

    citations: Mapped[list] = mapped_column(JSON, default=list)
    extra_metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    document: Mapped["DocumentORM"] = relationship(back_populates="chunks")


class FeedbackORM(Base):
    
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    rating: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PromptVersionORM(Base):
    
    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer)
    template: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
