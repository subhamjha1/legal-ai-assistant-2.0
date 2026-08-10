from backend.evaluation.metrics import aggregate_metrics, mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_recall_at_k_hit_and_miss():
    assert recall_at_k([False, True, False], k=2) == 1.0
    assert recall_at_k([False, False, True], k=2) == 0.0
    assert recall_at_k([False, False, True], k=3) == 1.0


def test_precision_at_k():
    assert precision_at_k([True, False, True, False], k=4) == 0.5
    assert precision_at_k([True, True], k=2) == 1.0
    assert precision_at_k([], k=5) == 0.0


def test_mrr_rewards_earlier_hits():
    assert mrr([True, False, False]) == 1.0
    assert mrr([False, True, False]) == 0.5
    assert mrr([False, False, False]) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([True, True, False], k=3) == 1.0


def test_ndcg_zero_when_no_hits():
    assert ndcg_at_k([False, False, False], k=3) == 0.0


def test_aggregate_metrics_shape():
    per_query_hits = [[True, False], [False, True], [False, False]]
    result = aggregate_metrics(per_query_hits, k_values=[1, 2])
    assert result["num_queries"] == 3
    assert "recall@1" in result and "recall@2" in result
    assert "precision@1" in result
    assert "ndcg@2" in result
    assert "mrr" in result
    assert 0.0 <= result["mrr"] <= 1.0
