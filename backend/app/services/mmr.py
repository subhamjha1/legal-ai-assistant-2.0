"""
Maximal Marginal Relevance (MMR) selection (Milestone 6).

Why this exists:
Top-K retrieval alone can return 5 chunks that are all near-paraphrases of
the same sentence (common in legal documents that restate a holding several
times) - which wastes the LLM's context budget and gives a false sense of
corroboration without adding real information. MMR re-selects a smaller,
genuinely diverse subset from a larger relevance-ranked candidate pool: it
picks the next chunk that is both relevant to the query AND dissimilar to
what's already been selected, trading a `mmr_lambda` amount of relevance for
diversity.

Kept as pure vector math (no embedding provider dependency) so the selection
algorithm itself is testable with synthetic vectors, independent of whether
BGE, OpenAI, or a fake embedder produced them.
"""
import math


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_select(
    query_vector: list[float],
    candidate_vectors: list[list[float]],
    top_k: int,
    lambda_param: float = 0.5,
) -> list[int]:
    """
    Select `top_k` indices from `candidate_vectors` via MMR.

    Returns indices into `candidate_vectors`, in selection order (index 0 of
    the return value is the first, most-relevance-favored pick).

    lambda_param=1.0 reduces to plain top-K-by-relevance (no diversity
    penalty). lambda_param=0.0 ignores relevance entirely after the first
    pick, selecting purely for dissimilarity to what's already chosen.
    """
    n = len(candidate_vectors)
    if n == 0:
        return []
    top_k = min(top_k, n)

    relevance = [_cosine_similarity(query_vector, v) for v in candidate_vectors]
    remaining = set(range(n))
    selected: list[int] = []

    while len(selected) < top_k and remaining:
        if not selected:
            # First pick: pure relevance, no diversity term possible yet.
            next_idx = max(remaining, key=lambda i: relevance[i])
        else:
            def mmr_score(i: int) -> float:
                diversity_penalty = max(
                    _cosine_similarity(candidate_vectors[i], candidate_vectors[j]) for j in selected
                )
                return lambda_param * relevance[i] - (1 - lambda_param) * diversity_penalty

            next_idx = max(remaining, key=mmr_score)

        selected.append(next_idx)
        remaining.discard(next_idx)

    return selected
