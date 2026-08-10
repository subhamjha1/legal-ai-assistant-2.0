from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import tiktoken

from backend.config.settings import settings
from backend.ingestion.pdf_parser import PageBlock, ParsedDocument
from backend.schemas.document import Chunk, DocumentMetadata, new_chunk_id

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass
class _Section:
    heading: str | None
    section: str | None
    subsection: str | None
    page_start: int
    page_end: int
    text_parts: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.text_parts).strip()


def _group_blocks_into_sections(blocks: list[PageBlock]) -> list[_Section]:
    sections: list[_Section] = []
    current: _Section | None = None

    for block in blocks:
        starts_new_section = block.is_heading and (block.section is not None or len(block.text) < 120)

        if starts_new_section or current is None:
            if current is not None and current.text:
                sections.append(current)
            current = _Section(
                heading=block.text if block.is_heading else None,
                section=block.section,
                subsection=block.subsection,
                page_start=block.page_number,
                page_end=block.page_number,
                text_parts=[block.text] if not block.is_heading else [],
            )
            continue

        current.page_end = max(current.page_end, block.page_number)
        current.text_parts.append(block.text)
        if block.section and not current.section:
            current.section = block.section
        if block.subsection and not current.subsection:
            current.subsection = block.subsection

    if current is not None and current.text:
        sections.append(current)

    return sections


def _recursive_split(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    if count_tokens(text) <= target_tokens:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > target_tokens:
            # last resort for huge paragraphs
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            tokens = _ENCODING.encode(para)
            step = max(target_tokens - overlap_tokens, 1)
            for i in range(0, len(tokens), step):
                window = tokens[i : i + target_tokens]
                chunks.append(_ENCODING.decode(window))
            continue

        if current_tokens + para_tokens > target_tokens and current:
            chunks.append(" ".join(current))
            # carry a little context into the next chunk
            overlap_text = _ENCODING.decode(_ENCODING.encode(" ".join(current))[-overlap_tokens:])
            current = [overlap_text] if overlap_text.strip() else []
            current_tokens = count_tokens(" ".join(current))

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _build_chunks_from_sections(
    sections: list[_Section],
    doc_meta: DocumentMetadata,
    target_tokens: int,
    overlap_tokens: int,
    is_parent: bool,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        pieces = _recursive_split(section.text, target_tokens, overlap_tokens)
        for piece in pieces:
            if count_tokens(piece) < settings.min_chunk_size_tokens and len(pieces) > 1:
                # avoid indexing tiny leftovers
                if chunks and chunks[-1].document_id == doc_meta.document_id:
                    chunks[-1].text += " " + piece
                    chunks[-1].token_count = count_tokens(chunks[-1].text)
                    continue

            chunk_id = new_chunk_id(doc_meta.document_id, idx)
            if is_parent:
                # parent and child chunks share indexes, so parent ids need a prefix
                chunk_id = f"parent_{chunk_id}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc_meta.document_id,
                    is_parent=is_parent,
                    text=piece,
                    token_count=count_tokens(piece),
                    filename=doc_meta.filename,
                    document_type=doc_meta.document_type,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    section=section.section,
                    subsection=section.subsection,
                    heading=section.heading,
                    chunk_index=idx,
                    metadata={
                        "act_name": doc_meta.act_name,
                        "court": doc_meta.court,
                        "year": doc_meta.year,
                    },
                )
            )
            idx += 1
    return chunks


def chunk_document(parsed: ParsedDocument, doc_meta: DocumentMetadata) -> tuple[list[Chunk], list[Chunk]]:
    sections = _group_blocks_into_sections(parsed.blocks)
    if not sections:
        logger.warning("No sections detected for %s — document may be empty or unparseable", doc_meta.filename)
        return [], []

    child_chunks = _build_chunks_from_sections(
        sections, doc_meta, settings.chunk_size_tokens, settings.chunk_overlap_tokens, is_parent=False
    )
    parent_chunks = _build_chunks_from_sections(
        sections, doc_meta, settings.parent_chunk_size_tokens, settings.chunk_overlap_tokens, is_parent=True
    )

    for child in child_chunks:
        best_parent = None
        for parent in parent_chunks:
            if parent.page_start <= child.page_start and child.page_end <= parent.page_end:
                best_parent = parent
                break
        if best_parent is None and parent_chunks:
            # page ranges can be imperfect after PDF parsing
            best_parent = min(
                parent_chunks,
                key=lambda p: abs(p.page_start - child.page_start),
            )
        if best_parent is not None:
            child.parent_chunk_id = best_parent.chunk_id

    logger.info(
        "%s -> %d sections, %d child chunks, %d parent chunks",
        doc_meta.filename,
        len(sections),
        len(child_chunks),
        len(parent_chunks),
    )
    return child_chunks, parent_chunks
