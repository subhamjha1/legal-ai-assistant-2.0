from __future__ import annotations

import json
import logging

from backend.llm.client import get_llm_client
from backend.retrieval.hybrid_retriever import RetrievedChunk
from backend.schemas.document import Chunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful US legal and tax research assistant.

Rules you must always follow:
1. Answer ONLY using the provided context chunks. Never use outside knowledge.
2. Never fabricate a citation, statute, case name, or fact that is not in the context.
3. Every factual claim in your answer must be attributable to a specific context chunk.
4. If the context does not contain enough information to answer confidently, say so explicitly: "Insufficient evidence in the provided documents to answer this fully," and explain what is missing.
5. Always cite the document name, page number, and section (if available) for each claim.
6. Return your response as a single JSON object matching this schema exactly:

{
  "answer": "<the direct answer>",
  "summary": "<1-3 sentence plain-English summary>",
  "supporting_citations": [
    {"document_name": "...", "page": "...", "section": "...", "quote_or_paraphrase": "..."}
  ],
  "confidence_note": "<brief note on evidence strength, e.g. 'directly supported by IRC section text' or 'inferred from commentary only'>",
  "insufficient_evidence": false
}

Do not include any text outside the JSON object."""

USER_TEMPLATE = """Question: {query}

Context chunks (each labeled with its source):
{context_block}

Answer the question following all rules above. Respond with the JSON object only."""


def _format_context_block(retrieved: list[RetrievedChunk], extra_graph_chunks: list[Chunk] | None = None) -> str:
    parts = []
    for i, rc in enumerate(retrieved, start=1):
        c = rc.chunk
        parts.append(
            f"[Chunk {i}] Source: {c.filename} | Page: {c.page_start}-{c.page_end} | "
            f"Section: {c.section or 'N/A'} | Type: {c.document_type.value}\n{c.text}"
        )
    if extra_graph_chunks:
        for i, c in enumerate(extra_graph_chunks, start=len(retrieved) + 1):
            parts.append(
                f"[Chunk {i} - graph-connected document] Source: {c.filename} | Page: {c.page_start}-{c.page_end} | "
                f"Section: {c.section or 'N/A'} | Type: {c.document_type.value}\n{c.text}"
            )
    return "\n\n".join(parts)


def _compute_retrieval_confidence(retrieved: list[RetrievedChunk]) -> float:
    if not retrieved:
        return 0.0
    top_score = retrieved[0].score
    normalized_top = max(0.0, min(1.0, (top_score + 5) / 10))

    top_doc = retrieved[0].chunk.document_id
    agreement = sum(1 for rc in retrieved if rc.chunk.document_id == top_doc) / len(retrieved)

    return round(0.7 * normalized_top + 0.3 * agreement, 3)


def generate_answer(
    query: str,
    retrieved: list[RetrievedChunk],
    extra_graph_chunks: list[Chunk] | None = None,
) -> dict:
    if not retrieved:
        return {
            "answer": "",
            "summary": "",
            "supporting_citations": [],
            "confidence_note": "No relevant documents were retrieved.",
            "insufficient_evidence": True,
            "retrieval_confidence": 0.0,
            "retrieved_chunks": [],
        }

    context_block = _format_context_block(retrieved, extra_graph_chunks)
    prompt = USER_TEMPLATE.format(query=query, context_block=context_block)

    client = get_llm_client()
    raw = client.complete(prompt, system=SYSTEM_PROMPT, temperature=0.0)

    parsed = _safe_parse_json(raw)
    parsed["retrieval_confidence"] = _compute_retrieval_confidence(retrieved)
    parsed["retrieved_chunks"] = [
        {
            "chunk_id": rc.chunk.chunk_id,
            "filename": rc.chunk.filename,
            "page_start": rc.chunk.page_start,
            "page_end": rc.chunk.page_end,
            "section": rc.chunk.section,
            "rerank_score": rc.score,
            "citation_label": rc.chunk.citation_label(),
        }
        for rc in retrieved
    ]
    return parsed


def _safe_parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM did not return valid JSON, wrapping raw text. Raw: %r", raw[:300])
        return {
            "answer": raw,
            "summary": "",
            "supporting_citations": [],
            "confidence_note": "LLM output was not valid JSON — returned raw text as a fallback.",
            "insufficient_evidence": False,
        }
