"""
Tests for MMR selection (Milestone 6).

Uses hand-crafted synthetic vectors (not real embeddings) so the algorithm
itself is verified precisely, independent of any embedding provider's
semantic behavior.
"""
import pytest

from app.services.mmr import mmr_select, _cosine_similarity


def test_cosine_similarity_identical_vectors():
    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_safe():
    assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0


class TestMMRSelect:
    def test_pure_relevance_when_lambda_is_one(self):
        """lambda=1.0 should reduce to plain top-K-by-relevance, ignoring
        diversity entirely - even if candidate 0 and 1 are identical."""
        query = [1, 0, 0]
        candidates = [
            [1, 0, 0],   # rank 1 by relevance
            [1, 0, 0],   # identical to above - would be penalized at lower lambda
            [0, 1, 0],   # rank 3 by relevance (orthogonal to query)
        ]
        selected = mmr_select(query, candidates, top_k=2, lambda_param=1.0)
        assert selected == [0, 1]  # the two near-identical, most-relevant picks

    def test_diversity_preferred_over_near_duplicate_at_low_lambda(self):
        """At a balanced/low lambda, once the most relevant chunk is picked,
        a near-duplicate of it should be passed over in favor of a
        still-relevant-but-more-different chunk."""
        query = [1, 0, 0]
        candidates = [
            [1, 0, 0],      # most relevant, picked first
            [0.99, 0.01, 0],  # near-duplicate of candidate 0
            [0.6, 0.6, 0],    # still relevant to query, but distinct from candidate 0
        ]
        selected = mmr_select(query, candidates, top_k=2, lambda_param=0.3)
        assert selected[0] == 0
        # The near-duplicate (index 1) should lose out to the more diverse,
        # still-reasonably-relevant candidate (index 2).
        assert selected[1] == 2

    def test_top_k_larger_than_candidates_returns_all(self):
        query = [1, 0, 0]
        candidates = [[1, 0, 0], [0, 1, 0]]
        selected = mmr_select(query, candidates, top_k=10, lambda_param=0.5)
        assert len(selected) == 2
        assert set(selected) == {0, 1}

    def test_empty_candidates_returns_empty(self):
        assert mmr_select([1, 0, 0], [], top_k=5, lambda_param=0.5) == []

    def test_first_pick_is_always_most_relevant_regardless_of_lambda(self):
        query = [1, 0, 0]
        candidates = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        for lam in (0.0, 0.3, 0.7, 1.0):
            selected = mmr_select(query, candidates, top_k=1, lambda_param=lam)
            assert selected == [1]  # candidate 1 is the only one aligned with the query
