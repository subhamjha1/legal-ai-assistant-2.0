from __future__ import annotations

import re

from backend.schemas.document import Chunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentence_set(text: str) -> set[str]:
    return {s.strip().lower() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 20}


def deduplicate_chunks(scored_chunks: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    seen_ids: set[str] = set()
    seen_text_hashes: set[str] = set()
    result: list[tuple[Chunk, float]] = []

    for chunk, score in scored_chunks:
        if chunk.chunk_id in seen_ids:
            continue
        text_key = re.sub(r"\s+", " ", chunk.text.strip().lower())[:500]
        if text_key in seen_text_hashes:
            continue
        seen_ids.add(chunk.chunk_id)
        seen_text_hashes.add(text_key)
        result.append((chunk, score))

    return result


def compress_context(scored_chunks: list[tuple[Chunk, float]], overlap_threshold: float = 0.6) -> list[tuple[Chunk, float]]:
    deduped = deduplicate_chunks(scored_chunks)
    kept: list[tuple[Chunk, float]] = []
    kept_sentences: list[set[str]] = []

    for chunk, score in deduped:
        sentences = _sentence_set(chunk.text)
        if not sentences:
            kept.append((chunk, score))
            kept_sentences.append(sentences)
            continue

        is_redundant = False
        for prior_sentences in kept_sentences:
            if not prior_sentences:
                continue
            overlap = len(sentences & prior_sentences) / len(sentences)
            if overlap >= overlap_threshold:
                is_redundant = True
                break

        if not is_redundant:
            kept.append((chunk, score))
            kept_sentences.append(sentences)

    return kept
