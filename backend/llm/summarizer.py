from __future__ import annotations

import logging

from backend.ingestion.chunker import count_tokens
from backend.llm.client import get_llm_client
from backend.schemas.document import Chunk

logger = logging.getLogger(__name__)

_MAP_PROMPT = """Summarize the following excerpt from a legal document in 3-5 \
sentences. Preserve any section numbers, holdings, or defined terms. Do not \
add outside information.

Source: {filename}, page {page_start}-{page_end}, section {section}

Excerpt:
{text}"""

_REDUCE_PROMPT = """Combine the following partial summaries of "{title}" into \
a single coherent summary (max {max_words} words). Preserve section/page \
references where they appear. Organize chronologically or by legal issue, \
whichever is clearer.

Partial summaries:
{summaries}"""

_REDUCE_GROUP_TOKEN_BUDGET = 3000


def _group_for_reduce(summaries: list[str], budget: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for s in summaries:
        t = count_tokens(s)
        if current and current_tokens + t > budget:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(s)
        current_tokens += t
    if current:
        groups.append(current)
    return groups


def map_reduce_summarize(title: str, chunks: list[Chunk], max_summary_words: int = 400) -> str:
    client = get_llm_client()
    ordered = sorted(chunks, key=lambda c: c.chunk_index)

    map_summaries: list[str] = []
    for c in ordered:
        prompt = _MAP_PROMPT.format(
            filename=c.filename, page_start=c.page_start, page_end=c.page_end, section=c.section or "N/A", text=c.text
        )
        try:
            summary = client.complete(prompt, max_tokens=200, temperature=0.0)
            map_summaries.append(f"[p.{c.page_start}-{c.page_end}, §{c.section or 'N/A'}] {summary.strip()}")
        except Exception:
            logger.exception("Map step failed for chunk %s — skipping", c.chunk_id)

    if not map_summaries:
        return "Unable to generate summary: no chunks summarized successfully."

    current_summaries = map_summaries
    while True:
        joined = "\n".join(current_summaries)
        if count_tokens(joined) <= _REDUCE_GROUP_TOKEN_BUDGET:
            prompt = _REDUCE_PROMPT.format(title=title, max_words=max_summary_words, summaries=joined)
            return client.complete(prompt, max_tokens=max_summary_words * 2, temperature=0.0).strip()

        groups = _group_for_reduce(current_summaries, _REDUCE_GROUP_TOKEN_BUDGET)
        logger.info("Reduce pass over %d groups (document too large for single reduce call)", len(groups))
        next_summaries = []
        for group in groups:
            prompt = _REDUCE_PROMPT.format(title=title, max_words=max_summary_words, summaries="\n".join(group))
            next_summaries.append(client.complete(prompt, max_tokens=max_summary_words * 2, temperature=0.0).strip())
        current_summaries = next_summaries
