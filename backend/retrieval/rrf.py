from __future__ import annotations

from backend.config.settings import settings


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int | None = None,
) -> list[tuple[str, float]]:
    k = k if k is not None else settings.rrf_k_constant
    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, _original_score) in enumerate(ranked_list, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
