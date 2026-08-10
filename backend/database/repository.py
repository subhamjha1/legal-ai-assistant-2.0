from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import ChunkORM, DocumentORM
from backend.schemas.document import Chunk, Citation, DocumentMetadata, DocumentType


def _chunk_orm_to_schema(row: ChunkORM) -> Chunk:
    return Chunk(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        parent_chunk_id=row.parent_chunk_id,
        is_parent=row.is_parent,
        text=row.text,
        token_count=row.token_count,
        filename=row.filename,
        document_type=DocumentType(row.document_type),
        page_start=row.page_start,
        page_end=row.page_end,
        section=row.section,
        subsection=row.subsection,
        heading=row.heading,
        chunk_index=row.chunk_index,
        citations=[Citation(**c) for c in (row.citations or [])],
        metadata=row.extra_metadata or {},
    )


def _document_orm_to_schema(row: DocumentORM) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=row.document_id,
        filename=row.filename,
        title=row.title,
        document_type=DocumentType(row.document_type),
        year=row.year,
        court=row.court,
        judge=row.judge,
        jurisdiction=row.jurisdiction,
        act_name=row.act_name,
        citation_string=row.citation_string,
        source_path=row.source_path,
        page_count=row.page_count,
        ingested_at=row.ingested_at,
        checksum=row.checksum,
        version=row.version,
    )


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, meta: DocumentMetadata) -> None:
        existing = await self.session.get(DocumentORM, meta.document_id)
        fields = dict(
            filename=meta.filename,
            title=meta.title,
            document_type=meta.document_type.value,
            year=meta.year,
            court=meta.court,
            judge=meta.judge,
            jurisdiction=meta.jurisdiction,
            act_name=meta.act_name,
            citation_string=meta.citation_string,
            source_path=meta.source_path,
            page_count=meta.page_count,
            ingested_at=meta.ingested_at,
            checksum=meta.checksum,
            version=meta.version,
        )
        if existing:
            for k, v in fields.items():
                setattr(existing, k, v)
        else:
            self.session.add(DocumentORM(document_id=meta.document_id, **fields))

    async def get(self, document_id: str) -> DocumentMetadata | None:
        row = await self.session.get(DocumentORM, document_id)
        return _document_orm_to_schema(row) if row else None

    async def list_all(self, document_type: str | None = None) -> list[DocumentMetadata]:
        stmt = select(DocumentORM)
        if document_type:
            stmt = stmt.where(DocumentORM.document_type == document_type)
        result = await self.session.execute(stmt)
        return [_document_orm_to_schema(r) for r in result.scalars().all()]


class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_upsert(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            existing = await self.session.get(ChunkORM, c.chunk_id)
            fields = dict(
                document_id=c.document_id,
                parent_chunk_id=c.parent_chunk_id,
                is_parent=c.is_parent,
                text=c.text,
                token_count=c.token_count,
                filename=c.filename,
                document_type=c.document_type.value,
                page_start=c.page_start,
                page_end=c.page_end,
                section=c.section,
                subsection=c.subsection,
                heading=c.heading,
                chunk_index=c.chunk_index,
                citations=[cit.model_dump() for cit in c.citations],
                extra_metadata=c.metadata,
            )
            if existing:
                for k, v in fields.items():
                    setattr(existing, k, v)
            else:
                self.session.add(ChunkORM(chunk_id=c.chunk_id, **fields))

    async def get_by_id(self, chunk_id: str) -> Chunk | None:
        row = await self.session.get(ChunkORM, chunk_id)
        return _chunk_orm_to_schema(row) if row else None

    async def get_by_ids(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        stmt = select(ChunkORM).where(ChunkORM.chunk_id.in_(chunk_ids))
        result = await self.session.execute(stmt)
        return {r.chunk_id: _chunk_orm_to_schema(r) for r in result.scalars().all()}

    async def get_neighbors(self, document_id: str, chunk_index: int, window: int) -> list[Chunk]:
        stmt = (
            select(ChunkORM)
            .where(
                ChunkORM.document_id == document_id,
                ChunkORM.is_parent.is_(False),
                ChunkORM.chunk_index.between(chunk_index - window, chunk_index + window),
            )
            .order_by(ChunkORM.chunk_index)
        )
        result = await self.session.execute(stmt)
        return [_chunk_orm_to_schema(r) for r in result.scalars().all()]

    async def get_parent(self, parent_chunk_id: str) -> Chunk | None:
        return await self.get_by_id(parent_chunk_id)

    async def get_all_children(self, document_id: str) -> list[Chunk]:
        stmt = (
            select(ChunkORM)
            .where(ChunkORM.document_id == document_id, ChunkORM.is_parent.is_(False))
            .order_by(ChunkORM.chunk_index)
        )
        result = await self.session.execute(stmt)
        return [_chunk_orm_to_schema(r) for r in result.scalars().all()]
