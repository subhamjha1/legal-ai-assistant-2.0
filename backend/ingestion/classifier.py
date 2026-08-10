from __future__ import annotations

import re
from pathlib import Path

from backend.ingestion.pdf_parser import ParsedDocument
from backend.schemas.document import DocumentMetadata, DocumentType, new_document_id

_FOLDER_HINTS = {
    "act": DocumentType.ACT,
    "acts": DocumentType.ACT,
    "judgment": DocumentType.JUDGMENT,
    "judgments": DocumentType.JUDGMENT,
    "case": DocumentType.JUDGMENT,
    "cases": DocumentType.JUDGMENT,
    "commentary": DocumentType.COMMENTARY,
    "commentaries": DocumentType.COMMENTARY,
    "pov": DocumentType.COMMENTARY,
    "irs": DocumentType.IRS_REGULATION,
    "tax_regulations": DocumentType.IRS_REGULATION,
    "regulations": DocumentType.IRS_REGULATION,
}

_KEYWORD_HINTS: list[tuple[re.Pattern, DocumentType]] = [
    (re.compile(r"\bv\.\s|\bversus\b|\bU\.S\.\s+\d+|\bF\.\d?d\b|\bSupreme Court\b", re.I), DocumentType.JUDGMENT),
    (re.compile(r"\bAn Act\b|\bBe it enacted\b|\bPublic Law\b", re.I), DocumentType.ACT),
    (re.compile(r"\bTreasury Regulation\b|\bInternal Revenue\b|\bIRS\b|\b26 CFR\b", re.I), DocumentType.IRS_REGULATION),
    (re.compile(r"\bcommentary\b|\banalysis\b|\bin our view\b|\bwe argue\b", re.I), DocumentType.COMMENTARY),
]

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _classify_by_folder(path: Path) -> DocumentType | None:
    for part in path.parts:
        hint = _FOLDER_HINTS.get(part.lower())
        if hint:
            return hint
    return None


def _classify_by_keywords(sample_text: str) -> DocumentType:
    scores = {dt: 0 for dt in DocumentType}
    for pattern, dt in _KEYWORD_HINTS:
        scores[dt] += len(pattern.findall(sample_text))
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else DocumentType.OTHER


def classify_with_llm(sample_text: str) -> DocumentType:
    from backend.llm.client import get_llm_client

    client = get_llm_client()
    prompt = (
        "Classify this US legal document excerpt into exactly one category: "
        "act, judgment, commentary, irs_regulation, other. "
        "Reply with only the category word.\n\n"
        f"Excerpt:\n{sample_text[:2000]}"
    )
    reply = client.complete(prompt, max_tokens=5, temperature=0.0).strip().lower()
    try:
        return DocumentType(reply)
    except ValueError:
        return DocumentType.OTHER


def extract_document_metadata(parsed: ParsedDocument, source_path: Path) -> DocumentMetadata:
    doc_type = _classify_by_folder(source_path)
    sample_text = "\n".join(b.text for b in parsed.blocks[:40])

    if doc_type is None:
        doc_type = _classify_by_keywords(sample_text)

    year_match = _YEAR_PATTERN.search(sample_text)
    year = int(year_match.group(0)) if year_match else None

    title = parsed.filename.rsplit(".", 1)[0]
    for block in parsed.blocks[:10]:
        if block.is_heading and len(block.text) > 8:
            title = block.text
            break

    return DocumentMetadata(
        document_id=new_document_id(parsed.filename),
        filename=parsed.filename,
        title=title,
        document_type=doc_type,
        year=year,
        source_path=str(source_path),
        page_count=parsed.page_count,
        checksum=parsed.checksum,
    )
