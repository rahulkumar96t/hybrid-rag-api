import pytest

from hybrid_rag_api.fusion import reciprocal_rank_fusion


def test_rrf_scores_follow_the_formula():
    fused = dict(reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y", "z"]}, k=60))
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["z"] == pytest.approx(1 / 62)


def test_items_found_by_both_retrievers_rank_first():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "dense": ["c", "d", "a"]})
    assert [item for item, _ in fused][:2] == ["a", "c"]


def test_weights_shift_the_balance():
    rankings = {"bm25": ["a", "b"], "dense": ["b", "a"]}
    assert reciprocal_rank_fusion(rankings, weights={"bm25": 2.0})[0][0] == "a"
    assert reciprocal_rank_fusion(rankings, weights={"dense": 2.0})[0][0] == "b"


def test_duplicates_in_one_list_count_once_at_best_rank():
    fused = dict(reciprocal_rank_fusion({"a": ["x", "x", "y"]}, k=0))
    assert fused == {"x": 1.0, "y": pytest.approx(1 / 3)}


def test_ties_keep_first_seen_order():
    fused = reciprocal_rank_fusion({"a": ["p", "q"], "b": ["q", "p"]})
    assert [item for item, _ in fused] == ["p", "q"]


def test_empty_and_invalid_input():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"a": []}) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": ["x"]}, k=-1)
