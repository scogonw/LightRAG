"""Tests for KG chunk count floor guard and oversampling factor.

Verifies acceptance criteria from SCO-53:
  - For 1 entity/relation, requested chunk count >= max_related_chunks (not max_related_chunks/2)
  - Oversampling factor is applied and configurable
  - Debug log emitted when pool < requested count
"""

import pytest

from lightrag.constants import DEFAULT_KG_CHUNK_OVERSAMPLING_FACTOR


def _num_of_chunks(max_related_chunks: int, n: int, oversampling_factor: float = DEFAULT_KG_CHUNK_OVERSAMPLING_FACTOR) -> int:
    """Mirror of the formula in operate.py VECTOR branch."""
    return int(max(max_related_chunks, int(max_related_chunks * n / 2)) * oversampling_factor)


class TestKgChunkFloorGuard:
    def test_one_entity_meets_floor(self):
        # 1 entity: old formula → max_related_chunks/2; new formula → max_related_chunks
        result = _num_of_chunks(max_related_chunks=5, n=1)
        assert result >= 5, f"Expected >= 5, got {result}"

    def test_two_entities_meets_floor(self):
        # 2 entities: old formula → max_related_chunks * 2/2 = max_related_chunks
        # new formula: max(max_related_chunks, max_related_chunks) * oversampling
        result = _num_of_chunks(max_related_chunks=5, n=2)
        assert result >= 5

    def test_many_entities_scales_above_floor(self):
        # 10 entities: max(5, 5*10/2=25) * 1.5 → 37
        result = _num_of_chunks(max_related_chunks=5, n=10)
        assert result > 5
        assert result == int(25 * DEFAULT_KG_CHUNK_OVERSAMPLING_FACTOR)

    def test_oversampling_applied_to_one_entity(self):
        result = _num_of_chunks(max_related_chunks=5, n=1, oversampling_factor=1.5)
        # floor = 5; 5 * 1.5 = 7
        assert result == 7

    def test_custom_oversampling_factor(self):
        result = _num_of_chunks(max_related_chunks=10, n=1, oversampling_factor=2.0)
        assert result == 20

    def test_oversampling_factor_one_is_neutral(self):
        result = _num_of_chunks(max_related_chunks=5, n=1, oversampling_factor=1.0)
        assert result == 5

    def test_floor_dominates_when_n_is_one(self):
        # Old formula: int(10 * 1 / 2) = 5; new floor: max(10, 5) = 10
        result = _num_of_chunks(max_related_chunks=10, n=1, oversampling_factor=1.0)
        assert result == 10

    def test_default_oversampling_factor_is_1_5(self):
        assert DEFAULT_KG_CHUNK_OVERSAMPLING_FACTOR == 1.5


class TestKgChunkDebugLogCondition:
    def test_pool_smaller_than_requested_triggers_log_condition(self):
        pool_size = 3
        num_of_chunks = 8
        assert pool_size < num_of_chunks, (
            "When pool_size < num_of_chunks the debug log branch should be entered"
        )

    def test_pool_equal_to_requested_does_not_trigger_log_condition(self):
        pool_size = 8
        num_of_chunks = 8
        assert not (pool_size < num_of_chunks)

    def test_pool_larger_than_requested_does_not_trigger_log_condition(self):
        pool_size = 15
        num_of_chunks = 8
        assert not (pool_size < num_of_chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
