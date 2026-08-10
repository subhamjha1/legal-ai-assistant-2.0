from backend.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_prefers_items_ranked_highly_across_multiple_lists():
    list_a = [("chunk_1", 0.9), ("chunk_2", 0.8), ("chunk_3", 0.7)]
    list_b = [("chunk_2", 50.0), ("chunk_1", 40.0), ("chunk_4", 30.0)]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    fused_ids = [cid for cid, _ in fused]

    assert fused_ids.index("chunk_1") < fused_ids.index("chunk_3")
    assert fused_ids.index("chunk_2") < fused_ids.index("chunk_4")


def test_rrf_ignores_raw_score_scale():
    list_a = [("chunk_1", 0.99)]
    list_b = [("chunk_2", 1000.0), ("chunk_1", 1.0)]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    fused_ids = [cid for cid, _ in fused]

    assert fused_ids[0] == "chunk_1"


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
