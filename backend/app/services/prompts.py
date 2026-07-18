"""
Prompt engineering for citation-grounded Q&A (Milestone 7).

Design principles baked into this prompt:
- The model is given ONLY the retrieved chunks as its knowledge source -
  never told to "use your knowledge of law" as a fallback, since that's
  exactly how a legal assistant hallucinates a plausible-sounding but
  unsupported answer.
- Every claim must carry an inline [Cx] tag pointing at a specific
  retrieved chunk. This isn't just a formatting nicety - it's what lets the
  citation formatter (see citation_formatter.py) build citations entirely
  from OUR retrieval data rather than trusting the model to type out
  correct page numbers itself (which LLMs reliably get wrong).
- An exact, mechanically-checkable fallback phrase
  ("I could not find supporting evidence.") is mandated for insufficient
  evidence, rather than leaving the model to phrase a hedge however it
  likes - this makes "did the system admit it doesn't know" a checkable
  property in evaluation (Milestone 9), not a fuzzy judgment call.
"""
from app.core.config import get_settings
from app.schemas.search import RetrievedChunk


def build_system_prompt() -> str:
    settings = get_settings()
    return f"""You are a legal research assistant. You answer questions ONLY using the numbered context passages provided in the user's message. You do not use any outside legal knowledge, even if you are confident it is correct.

Rules you must follow exactly:
1. Every factual claim in your answer must be immediately followed by the tag of the passage that supports it, in the form [C1], [C2], etc. A sentence with multiple sources may carry multiple tags, e.g. [C1][C3].
2. Never cite a passage number that was not given to you.
3. Do not combine information from multiple passages into a claim that no single passage actually supports - if passage synthesis is needed, cite every passage the synthesis draws from.
4. If the provided passages do not contain enough information to answer the question, respond with EXACTLY this sentence and nothing else: "{settings.no_evidence_phrase}"
5. Do not hedge with phrases like "it seems" or "likely" to cover for a claim you cannot actually support with a citation - either cite it or leave it out.
6. Be concise. Answer the question directly; do not restate the question or add unsolicited commentary.
"""


def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    """Builds the numbered-context user message. Tags are simple ordinals
    (C1, C2, ...) rather than the chunks' real UUIDs, since short ordinal
    tags are far less error-prone for a model to reproduce exactly than
    long UUIDs - accuracy of the *tag* matters because the citation
    formatter maps it straight back to trusted metadata."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        page_ref = f"page {chunk.page_start}" if chunk.page_start == chunk.page_end else f"pages {chunk.page_start}-{chunk.page_end}"
        label = f", {chunk.structural_label}" if chunk.structural_label else ""
        context_blocks.append(f"[C{i}] (Source: {chunk.original_filename}, {page_ref}{label})\n{chunk.text}")

    context_text = "\n\n".join(context_blocks)
    return f"""Context passages:

{context_text}

Question: {query}

Answer the question using only the passages above, with inline [Cx] citation tags as instructed."""
