from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

# heading patterns go from specific to broad
_SECTION_PATTERNS = [
    re.compile(r"^\s*(SEC(?:TION)?\.?\s*\d+[A-Za-z]?)\b", re.IGNORECASE),
    re.compile(r"^\s*(§+\s*\d+[\w.\-]*)"),
    re.compile(r"^\s*(Section\s+\d+(?:\([a-zA-Z0-9]+\))*)", re.IGNORECASE),
    re.compile(r"^\s*(Article\s+[IVXLCDM\d]+)", re.IGNORECASE),
    re.compile(r"^\s*(Rule\s+\d+[\w.\-]*)", re.IGNORECASE),
    re.compile(r"^\s*(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)\s+[A-Z]"),
]

_SUBSECTION_PATTERN = re.compile(r"^\s*\(([a-zA-Z0-9]{1,4})\)\s")


@dataclass
class TextSpan:
    text: str
    font_size: float
    is_bold: bool


@dataclass
class PageBlock:
    
    page_number: int
    text: str
    is_heading: bool = False
    section: str | None = None
    subsection: str | None = None
    avg_font_size: float = 0.0


@dataclass
class ParsedDocument:
    filename: str
    source_path: str
    page_count: int
    checksum: str
    blocks: list[PageBlock] = field(default_factory=list)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_section(line: str) -> tuple[str | None, str | None]:
    section = None
    for pattern in _SECTION_PATTERNS:
        m = pattern.match(line)
        if m:
            section = m.group(1).strip()
            break
    subsection = None
    m = _SUBSECTION_PATTERN.match(line)
    if m:
        subsection = m.group(1)
    return section, subsection


def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    doc = fitz.open(path)

    all_font_sizes: list[float] = []
    raw_pages: list[list[tuple[str, float, bool]]] = []

    for page in doc:
        page_dict = page.get_text("dict")
        lines_out: list[tuple[str, float, bool]] = []
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue
                avg_size = sum(s["size"] for s in spans) / len(spans)
                is_bold = any((s["flags"] & 2**4) for s in spans)
                lines_out.append((line_text, avg_size, is_bold))
                all_font_sizes.append(avg_size)
        raw_pages.append(lines_out)

    body_median = sorted(all_font_sizes)[len(all_font_sizes) // 2] if all_font_sizes else 10.0
    heading_size_threshold = body_median * 1.15

    blocks: list[PageBlock] = []
    for page_idx, lines in enumerate(raw_pages, start=1):
        buffer: list[str] = []
        buffer_size = 0.0
        buffer_bold = False

        def flush():
            nonlocal buffer, buffer_size, buffer_bold
            if not buffer:
                return
            text = " ".join(buffer).strip()
            if not text:
                buffer, buffer_size, buffer_bold = [], 0.0, False
                return
            section, subsection = _detect_section(text)
            is_heading = bool(section) or (
                buffer_size >= heading_size_threshold and len(text) < 150
            )
            blocks.append(
                PageBlock(
                    page_number=page_idx,
                    text=text,
                    is_heading=is_heading,
                    section=section,
                    subsection=subsection,
                    avg_font_size=buffer_size,
                )
            )
            buffer, buffer_size, buffer_bold = [], 0.0, False

        prev_was_heading = False
        for line_text, size, bold in lines:
            section, _ = _detect_section(line_text)
            looks_like_heading = bool(section) or (size >= heading_size_threshold and len(line_text) < 150)

            if looks_like_heading:
                flush()
                blocks.append(
                    PageBlock(
                        page_number=page_idx,
                        text=line_text,
                        is_heading=True,
                        section=section,
                        subsection=_detect_section(line_text)[1],
                        avg_font_size=size,
                    )
                )
                prev_was_heading = True
                continue

            buffer.append(line_text)
            buffer_size = max(buffer_size, size)
            buffer_bold = buffer_bold or bold
            prev_was_heading = False

        flush()

    parsed = ParsedDocument(
        filename=path.name,
        source_path=str(path),
        page_count=doc.page_count,
        checksum=_sha256_of_file(path),
        blocks=blocks,
    )
    doc.close()
    logger.info(
        "Parsed %s: %d pages, %d blocks (%d headings detected)",
        path.name,
        parsed.page_count,
        len(blocks),
        sum(1 for b in blocks if b.is_heading),
    )
    return parsed
