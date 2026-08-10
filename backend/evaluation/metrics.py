from __future__ import annotations

import math


def recall_at_k(hits: list[bool], k: int) -> float:
    return 1.0 if any(hits[:k]) else 0.0


def precision_at_k(hits: list[bool], k: int) -> float:
    top_k = hits[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / len(top_k)


def mrr(hits: list[bool]) -> float:
    for i, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / i
    return 0.0


def ndcg_at_k(hits: list[bool], k: int) -> float:
    top_k = hits[:k]
    dcg = sum((1.0 if hit else 0.0) / math.log2(i + 2) for i, hit in enumerate(top_k))
    ideal_hits = sorted(top_k, reverse=True)
    idcg = sum((1.0 if hit else 0.0) / math.log2(i + 2) for i, hit in enumerate(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_metrics(per_query_hits: list[list[bool]], k_values: list[int]) -> dict:
    n = len(per_query_hits)
    if n == 0:
        return {}

    results: dict = {"num_queries": n}
    for k in k_values:
        results[f"recall@{k}"] = round(sum(recall_at_k(h, k) for h in per_query_hits) / n, 4)
        results[f"precision@{k}"] = round(sum(precision_at_k(h, k) for h in per_query_hits) / n, 4)
        results[f"ndcg@{k}"] = round(sum(ndcg_at_k(h, k) for h in per_query_hits) / n, 4)

    results["mrr"] = round(sum(mrr(h) for h in per_query_hits) / n, 4)
    return results
