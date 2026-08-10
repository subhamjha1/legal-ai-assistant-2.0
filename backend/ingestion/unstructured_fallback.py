from __future__ import annotations

import logging
from pathlib import Path

from backend.ingestion.pdf_parser import PageBlock, ParsedDocument, _sha256_of_file

logger = logging.getLogger(__name__)

_TITLE_LIKE = {"Title", "Header"}


def parse_pdf_with_unstructured(path: str | Path) -> ParsedDocument:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "unstructured[pdf] is not installed. `pip install 'unstructured[pdf]'`"
        ) from e

    path = Path(path)
    elements = partition_pdf(
        filename=str(path),
        strategy="hi_res",
        infer_table_structure=True,
    )

    blocks: list[PageBlock] = []
    max_page = 1
    for el in elements:
        text = (el.text or "").strip()
        if not text:
            continue
        page_number = getattr(el.metadata, "page_number", None) or 1
        max_page = max(max_page, page_number)
        category = el.category
        blocks.append(
            PageBlock(
                page_number=page_number,
                text=text,
                is_heading=category in _TITLE_LIKE,
                section=None,
                subsection=None,
                avg_font_size=14.0 if category in _TITLE_LIKE else 10.0,
            )
        )

    parsed = ParsedDocument(
        filename=path.name,
        source_path=str(path),
        page_count=max_page,
        checksum=_sha256_of_file(path),
        blocks=blocks,
    )
    logger.info("Parsed %s via unstructured fallback: %d blocks", path.name, len(blocks))
    return parsed


def needs_ocr_fallback(parsed: ParsedDocument, min_chars_per_page: int = 40) -> bool:
    if parsed.page_count == 0:
        return True
    total_chars = sum(len(b.text) for b in parsed.blocks)
    return (total_chars / parsed.page_count) < min_chars_per_page
