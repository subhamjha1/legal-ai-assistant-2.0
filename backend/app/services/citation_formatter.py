"""
Citation formatter (Milestone 7).

This is the single most important anti-hallucination component in the
system, and it is deliberately "dumb": it does not ask the LLM what page or
document it used. It scans the LLM's answer text for [Cx] tags, and for
every tag found, looks up the corresponding chunk in OUR OWN retrieval
results - the same RetrievedChunk objects the LLM was given, whose page
numbers and document names came straight from Milestone 1's page-preserving
parser. The LLM's only job is to place the tag correctly; it never gets a
chance to invent a page number, because we never ask it for one.

Kept as pure, dependency-free functions (no LLM call inside) so this is
fully unit-testable against hand-written answer strings.
"""
import re

from app.core.config import get_settings
from app.schemas.qa import Citation
from app.schemas.search import RetrievedChunk

_TAG_PATTERN = re.compile(r"\[C(\d+)\]")


def extract_citations(answer_text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """
    Parse [Cx] tags out of `answer_text` (in order of first appearance,
    deduplicated) and map each to the corresponding chunk in `chunks`
    (1-indexed, matching how build_user_message numbered them).

    Tags referring to an out-of-range index (a model error - citing a
    passage number it was never given) are silently skipped rather than
    raising, since a malformed citation should degrade to "uncited claim",
    not crash the response.
    """
    seen_indices: list[int] = []
    for match in _TAG_PATTERN.finditer(answer_text):
        index = int(match.group(1))
        if index not in seen_indices:
            seen_indices.append(index)

    citations = []
    for index in seen_indices:
        chunk = _chunk_for_tag(index, chunks)
        if chunk is None:
            continue  # model cited a passage number it was never given
        citations.append(
            Citation(
                chunk_ref=f"C{index}",
                document=chunk.original_filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                structural_label=chunk.structural_label,
                snippet=_snippet(chunk.text),
            )
        )
    return citations


def is_no_evidence_response(answer_text: str) -> bool:
    """Checks whether the model correctly used the exact mandated fallback
    phrase, rather than a loose 'contains similar words' heuristic - this
    needs to be a mechanically checkable property for evaluation
    (Milestone 9), not a fuzzy judgment call."""
    settings = get_settings()
    return answer_text.strip() == settings.no_evidence_phrase


def _chunk_for_tag(index: int, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    if 1 <= index <= len(chunks):
        return chunks[index - 1]
    return None


def _snippet(text: str, max_length: int = 200) -> str:
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."
