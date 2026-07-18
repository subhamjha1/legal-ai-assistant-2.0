"""
Semantic chunking service (Milestone 2).

Why this exists, and why structure-aware:
A fixed-size chunker cuts a legal document at arbitrary character counts,
which frequently slices a section in half - e.g. "Section 80G(5)(iv)
requires that the donation..." gets cut mid-clause, and the LLM retrieves a
chunk with the obligation but not its condition, or vice versa. This is a
direct cause of both hallucination and wrong citations in legal RAG systems.

Strategy:
1. Find legal structural markers in the document (numbered paragraphs like
   "4.", Section/Article/Clause/Chapter/Part headers).
2. Split the document at those markers first - each resulting section is a
   coherent legal unit.
3. Only if a section is still larger than `chunk_max_chars` (e.g. a very long
   judgment paragraph) do we sub-split it, using a sliding window that
   prefers to break at sentence boundaries, with character overlap between
   sub-chunks so context isn't lost across the cut.
4. Sections smaller than `chunk_min_chars` are merged into the next section,
   to avoid embedding near-empty fragments (e.g. a lone "IN THE HIGH COURT
   OF DELHI" header) that add index noise without retrieval value.

Every chunk tracks exactly which page(s) it was drawn from, via a
character-offset-to-page map built while concatenating page text - this is
what lets Milestone 7's citation formatter say "page 4" and be correct even
when a chunk straddles a page boundary.
"""
import re
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.chunk import Chunk, ChunkingResult, PageSpan, SplitReason
from app.schemas.document import ParsedDocument

logger = get_logger(__name__)

# Ordered by specificity; each is checked at line-start (re.MULTILINE) since
# legal structural markers almost always begin a line.
_STRUCTURAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("section", re.compile(r"(?m)^\s*(SECTION|Section)\s+\d+[A-Za-z]*(\(\d+\))?[A-Za-z]*")),
    ("article", re.compile(r"(?m)^\s*(ARTICLE|Article)\s+\d+")),
    ("chapter", re.compile(r"(?m)^\s*(CHAPTER|Chapter)\s+\d+")),
    ("clause", re.compile(r"(?m)^\s*(CLAUSE|Clause)\s+\d+")),
    ("part", re.compile(r"(?m)^\s*(PART|Part)\s+[IVXLC]+")),
    # Numbered paragraphs, e.g. judgment style "4. In view of...". Deliberately
    # last/lowest priority since it's the most general pattern and most prone
    # to false positives (e.g. a numbered list inside a table).
    ("paragraph", re.compile(r"(?m)^\s*(\d{1,3})\.\s+(?=[A-Z(])")),
]

# Sentence-boundary heuristic used only when sub-splitting an oversized
# section - we prefer to cut after ". ", "; ", or a newline.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;])\s+")


@dataclass
class _PageOffset:
    page_number: int
    start: int  # inclusive char offset into the concatenated full_text
    end: int    # exclusive


@dataclass
class _Marker:
    label: str
    start: int  # char offset where the marker (and its section) begins


class SemanticChunker:
    """Chunks a ParsedDocument into page-aware, structure-respecting Chunks."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        full_text, page_offsets = self._build_full_text(document)
        markers = self._find_markers(full_text)

        sections = self._split_by_markers(full_text, markers)
        sections = self._merge_small_sections(sections)

        chunks: list[Chunk] = []
        for label, start, end, split_reason in sections:
            section_text = full_text[start:end]
            sub_chunks = self._size_cap_split(section_text, start)
            for i, (sub_text, sub_start, sub_end) in enumerate(sub_chunks):
                chunk = self._build_chunk(
                    document_id=document.document_id,
                    chunk_index=len(chunks),
                    text=sub_text,
                    structural_label=self._label_for_subchunk(label, i, len(sub_chunks)),
                    split_reason=split_reason if i == 0 else SplitReason.SIZE_CAP,
                    global_start=sub_start,
                    global_end=sub_end,
                    page_offsets=page_offsets,
                    overlaps_previous=(i > 0),
                )
                chunks.append(chunk)

        logger.info(
            "Chunked document %s into %d chunks (%d structural markers found)",
            document.document_id,
            len(chunks),
            len(markers),
        )
        return ChunkingResult(
            document_id=document.document_id,
            total_chunks=len(chunks),
            chunks=chunks,
            structural_markers_found=len(markers),
        )

    # ------------------------------------------------------------------ #
    # Text assembly + page offset mapping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_full_text(document: ParsedDocument) -> tuple[str, list[_PageOffset]]:
        """Concatenate all page texts into one string, recording the exact
        char range each page occupies so chunk offsets can be mapped back to
        page numbers later."""
        parts: list[str] = []
        offsets: list[_PageOffset] = []
        cursor = 0
        separator = "\n\n"

        for page in document.pages:
            start = cursor
            parts.append(page.text)
            cursor += len(page.text)
            offsets.append(_PageOffset(page_number=page.page_number, start=start, end=cursor))
            parts.append(separator)
            cursor += len(separator)

        return "".join(parts), offsets

    @staticmethod
    def _pages_for_range(start: int, end: int, page_offsets: list[_PageOffset]) -> list[PageSpan]:
        """Given a [start, end) char range in the full concatenated text,
        return how many characters of it fall on each page it overlaps."""
        spans: list[PageSpan] = []
        for po in page_offsets:
            overlap_start = max(start, po.start)
            overlap_end = min(end, po.end)
            if overlap_start < overlap_end:
                spans.append(PageSpan(page_number=po.page_number, char_count=overlap_end - overlap_start))
        return spans

    # ------------------------------------------------------------------ #
    # Structural marker detection + splitting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_markers(full_text: str) -> list[_Marker]:
        found: list[_Marker] = []
        for _, pattern in _STRUCTURAL_PATTERNS:
            for match in pattern.finditer(full_text):
                label = match.group(0).strip()
                found.append(_Marker(label=label, start=match.start()))
        found.sort(key=lambda m: m.start)

        # Drop markers that are within a few characters of an already-found
        # marker (avoids double-splitting when two patterns match the same
        # spot, e.g. "Section 5" also loosely matching a paragraph number).
        deduped: list[_Marker] = []
        for marker in found:
            if deduped and marker.start - deduped[-1].start < 5:
                continue
            deduped.append(marker)
        return deduped

    @staticmethod
    def _split_by_markers(
        full_text: str, markers: list[_Marker]
    ) -> list[tuple[str | None, int, int, SplitReason]]:
        """Returns list of (label, start, end, split_reason) sections."""
        if not markers:
            return [(None, 0, len(full_text), SplitReason.DOCUMENT_BOUNDARY)]

        sections: list[tuple[str | None, int, int, SplitReason]] = []

        # Text before the first marker (e.g. a title block) becomes its own
        # unlabeled section if non-trivial.
        if markers[0].start > 0:
            sections.append((None, 0, markers[0].start, SplitReason.DOCUMENT_BOUNDARY))

        for i, marker in enumerate(markers):
            end = markers[i + 1].start if i + 1 < len(markers) else len(full_text)
            sections.append((marker.label, marker.start, end, SplitReason.STRUCTURAL_MARKER))

        return sections

    def _merge_small_sections(
        self, sections: list[tuple[str | None, int, int, SplitReason]]
    ) -> list[tuple[str | None, int, int, SplitReason]]:
        """Merge sections shorter than chunk_min_chars into the following
        section, so trivial fragments (like a lone header) don't become
        their own low-value embedding."""
        if not sections:
            return sections

        merged: list[tuple[str | None, int, int, SplitReason]] = []
        pending: tuple[str | None, int, int, SplitReason] | None = None

        for label, start, end, reason in sections:
            if pending is not None:
                # Extend the pending small section to include this one.
                pending_label = pending[0] or label
                pending = (pending_label, pending[1], end, pending[3])
            else:
                pending = (label, start, end, reason)

            if (pending[2] - pending[1]) >= self.settings.chunk_min_chars:
                merged.append(pending)
                pending = None

        if pending is not None:
            # Trailing small section: attach to the previous chunk if one
            # exists, otherwise keep it as-is (it's the whole document).
            if merged:
                last = merged.pop()
                merged.append((last[0], last[1], pending[2], last[3]))
            else:
                merged.append(pending)

        return merged

    # ------------------------------------------------------------------ #
    # Size-cap sub-splitting for oversized sections
    # ------------------------------------------------------------------ #
    def _size_cap_split(self, text: str, global_offset: int) -> list[tuple[str, int, int]]:
        """Split `text` into pieces no larger than chunk_max_chars, preferring
        sentence boundaries, with chunk_overlap_chars of overlap between
        consecutive pieces. Returns (sub_text, global_start, global_end)."""
        max_chars = self.settings.chunk_max_chars
        overlap = self.settings.chunk_overlap_chars

        if len(text) <= max_chars:
            return [(text, global_offset, global_offset + len(text))]

        pieces: list[tuple[str, int, int]] = []
        cursor = 0
        text_len = len(text)

        while cursor < text_len:
            tentative_end = min(cursor + max_chars, text_len)
            if tentative_end < text_len:
                # Try to snap to the nearest sentence boundary before
                # tentative_end (search backwards within a reasonable window).
                window_start = max(cursor + int(max_chars * 0.5), cursor)
                search_region = text[window_start:tentative_end]
                boundaries = list(_SENTENCE_BOUNDARY.finditer(search_region))
                if boundaries:
                    tentative_end = window_start + boundaries[-1].end()

            piece = text[cursor:tentative_end]
            pieces.append((piece, global_offset + cursor, global_offset + tentative_end))

            if tentative_end >= text_len:
                break
            cursor = max(tentative_end - overlap, cursor + 1)  # guarantee forward progress

        return pieces

    @staticmethod
    def _label_for_subchunk(label: str | None, index: int, total: int) -> str | None:
        if label is None:
            return None
        if total == 1:
            return label
        return f"{label} (part {index + 1}/{total})"

    def _build_chunk(
        self,
        document_id: str,
        chunk_index: int,
        text: str,
        structural_label: str | None,
        split_reason: SplitReason,
        global_start: int,
        global_end: int,
        page_offsets: list[_PageOffset],
        overlaps_previous: bool,
    ) -> Chunk:
        page_spans = self._pages_for_range(global_start, global_end, page_offsets)
        if not page_spans:
            # Defensive fallback; should not happen given how offsets are built.
            page_spans = [PageSpan(page_number=page_offsets[0].page_number, char_count=len(text))]

        return Chunk(
            document_id=document_id,
            chunk_index=chunk_index,
            text=text.strip(),
            structural_label=structural_label,
            split_reason=split_reason,
            page_start=page_spans[0].page_number,
            page_end=page_spans[-1].page_number,
            page_spans=page_spans,
            char_count=len(text.strip()),
            overlaps_previous=overlaps_previous,
        )
