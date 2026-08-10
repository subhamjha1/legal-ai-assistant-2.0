from __future__ import annotations

import re

from backend.schemas.document import Citation

_PATTERNS = [
    # Case law: "410 U.S. 113", "347 U.S. 483 (1954)"
    re.compile(r"\b\d{1,3}\s+U\.S\.\s+\d{1,4}(?:\s*\(\d{4}\))?"),
    # Federal reporter: "410 F.2d 113", "410 F.3d 113", "410 F. Supp. 2d 113"
    re.compile(r"\b\d{1,4}\s+F\.(?:2d|3d|Supp\.\s?2d)?\s+\d{1,5}"),
    # U.S. Code: "26 U.S.C. § 61", "42 U.S.C. §§ 1983-1985"
    re.compile(r"\b\d{1,3}\s+U\.S\.C\.\s+§+\s*\d+[\w.\-]*"),
    # CFR: "26 C.F.R. § 1.61-1"
    re.compile(r"\b\d{1,3}\s+C\.F\.R\.\s+§+\s*[\w.\-]*"),
    # Public Law: "Public Law 117-169"
    re.compile(r"\bPublic Law\s+\d+-\d+", re.IGNORECASE),
    # Internal Revenue Code section shorthand: "IRC § 162", "Section 162(a)"
    re.compile(r"\bIRC\s+§+\s*\d+[\w.\-]*", re.IGNORECASE),
    re.compile(r"\bSection\s+\d+\([a-zA-Z0-9]+\)(?:\([a-zA-Z0-9]+\))*", re.IGNORECASE),
]


def extract_citations(text: str) -> list[Citation]:
    found: list[Citation] = []
    seen: set[str] = set()
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(Citation(raw_text=raw, normalized=_normalize(raw)))
    return found


def _normalize(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip().rstrip(".")
