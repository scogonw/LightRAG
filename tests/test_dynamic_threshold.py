import pytest

from lightrag.utils import apply_dynamic_threshold


def _hits(*scores, key="distance"):
    return [{"id": f"r{i}", key: s} for i, s in enumerate(scores)]


class TestApplyDynamicThreshold:
    def test_empty_input_returns_empty(self):
        assert apply_dynamic_threshold([], floor=0.4, gap=0.10) == []

    def test_single_hit_above_floor_kept(self):
        hits = _hits(0.72)
        out = apply_dynamic_threshold(hits, floor=0.4, gap=0.10)
        assert len(out) == 1
        assert out[0]["id"] == "r0"

    def test_single_hit_below_floor_dropped(self):
        hits = _hits(0.3)
        assert apply_dynamic_threshold(hits, floor=0.4, gap=0.10) == []

    def test_strong_top_drops_weak_tail_via_gap(self):
        hits = _hits(0.9, 0.5, 0.45)
        out = apply_dynamic_threshold(hits, floor=0.4, gap=0.10)
        assert [r["id"] for r in out] == ["r0"]

    def test_close_cluster_all_kept(self):
        hits = _hits(0.9, 0.85, 0.82)
        out = apply_dynamic_threshold(hits, floor=0.4, gap=0.10)
        assert [r["id"] for r in out] == ["r0", "r1", "r2"]

    def test_weak_query_floor_dominates(self):
        # Top score below floor + gap window: cutoff = floor itself.
        hits = _hits(0.42, 0.41, 0.40)
        out = apply_dynamic_threshold(hits, floor=0.4, gap=0.10)
        assert [r["id"] for r in out] == ["r0", "r1", "r2"]

    def test_top_below_floor_returns_empty(self):
        hits = _hits(0.35, 0.30, 0.20)
        assert apply_dynamic_threshold(hits, floor=0.4, gap=0.10) == []

    def test_gap_zero_disables_relative_filter(self):
        hits = _hits(0.9, 0.5, 0.45)
        out = apply_dynamic_threshold(hits, floor=0.4, gap=0.0)
        assert [r["id"] for r in out] == ["r0", "r1", "r2"]

    def test_preserves_input_order(self):
        # Even when scores are out of order, surviving hits keep input order.
        hits = _hits(0.5, 0.9, 0.6)
        out = apply_dynamic_threshold(hits, floor=0.4, gap=0.10)
        assert [r["id"] for r in out] == ["r1"]

    def test_custom_score_key(self):
        hits = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.5}]
        out = apply_dynamic_threshold(hits, floor=0.4, gap=0.10, score_key="score")
        assert [r["id"] for r in out] == ["a"]

    def test_missing_score_treated_as_zero(self):
        # Defensive: a hit without the score field is treated as 0 and dropped
        # by the floor (unless floor is also 0).
        hits = [{"id": "a", "distance": 0.9}, {"id": "b"}]
        out = apply_dynamic_threshold(hits, floor=0.4, gap=0.10)
        assert [r["id"] for r in out] == ["a"]

    def test_user_real_world_chunk_at_0_72(self):
        # Regression: the chunk at 0.7207 from the bug report must survive.
        hits = _hits(0.7207)
        out = apply_dynamic_threshold(hits, floor=0.4, gap=0.10)
        assert len(out) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
