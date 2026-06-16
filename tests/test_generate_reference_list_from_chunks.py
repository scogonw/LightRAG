"""Offline unit tests for generate_reference_list_from_chunks (utils.py).

Regression suite for SCO-102: reference list must be ordered by max rerank_score
per file (desc) when reranking is active, and must fall back to the original
frequency → first-appearance ordering when reranking is off.
"""

import pytest

from lightrag.utils import generate_reference_list_from_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_order(reference_list: list[dict]) -> list[str]:
    """Return the file_paths in reference_id order (1, 2, 3…)."""
    return [r["file_path"] for r in reference_list]


def _chunk(file_path: str, **extra) -> dict:
    return {"file_path": file_path, "content": "x", **extra}


# ---------------------------------------------------------------------------
# Rerank ON
# ---------------------------------------------------------------------------


@pytest.mark.offline
class TestRerankOn:
    def test_single_high_score_beats_many_low_score_chunks(self):
        """Core bug regression: file A contributes one high-score chunk; file B
        contributes three low-score chunks.  Before the fix file B ranked [1]
        (more chunks).  After the fix file A must rank [1] (higher score)."""
        chunks = [
            _chunk("file_a.txt", rerank_score=0.95),
            _chunk("file_b.txt", rerank_score=0.20),
            _chunk("file_b.txt", rerank_score=0.15),
            _chunk("file_b.txt", rerank_score=0.10),
        ]
        ref_list, updated = generate_reference_list_from_chunks(chunks)

        assert _ref_order(ref_list) == ["file_a.txt", "file_b.txt"]
        assert ref_list[0]["reference_id"] == "1"
        assert ref_list[1]["reference_id"] == "2"

        # Updated chunks carry the correct reference_id
        assert updated[0]["reference_id"] == "1"  # file_a
        for c in updated[1:]:
            assert c["reference_id"] == "2"  # file_b

    def test_ordering_tracks_max_score_not_average(self):
        """Max score per file is used, not total or average."""
        chunks = [
            _chunk("low_avg.txt", rerank_score=0.90),   # max 0.90
            _chunk("low_avg.txt", rerank_score=0.01),
            _chunk("high_avg.txt", rerank_score=0.50),  # max 0.50
            _chunk("high_avg.txt", rerank_score=0.45),
            _chunk("high_avg.txt", rerank_score=0.40),
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)
        assert _ref_order(ref_list) == ["low_avg.txt", "high_avg.txt"]

    def test_multiple_files_ordered_by_descending_score(self):
        chunks = [
            _chunk("c.txt", rerank_score=0.30),
            _chunk("a.txt", rerank_score=0.90),
            _chunk("b.txt", rerank_score=0.60),
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)
        assert _ref_order(ref_list) == ["a.txt", "b.txt", "c.txt"]

    def test_tie_in_rerank_score_falls_back_to_frequency_then_first_appearance(self):
        """When two files share the same max score the tiebreaker is frequency
        (desc), then first-appearance index (asc)."""
        chunks = [
            _chunk("first.txt", rerank_score=0.80),   # first seen, count=1
            _chunk("second.txt", rerank_score=0.80),  # second seen, count=2
            _chunk("second.txt", rerank_score=0.70),
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)
        # second.txt: max=0.80, count=2; first.txt: max=0.80, count=1
        # Same max score → freq tiebreak → second.txt first
        assert _ref_order(ref_list) == ["second.txt", "first.txt"]

    def test_reference_list_structure(self):
        chunks = [
            _chunk("x.pdf", rerank_score=0.5),
            _chunk("y.pdf", rerank_score=0.8),
        ]
        ref_list, updated = generate_reference_list_from_chunks(chunks)
        for item in ref_list:
            assert "reference_id" in item
            assert "file_path" in item
        for c in updated:
            assert "reference_id" in c

    def test_updated_chunks_not_mutated_in_place(self):
        chunks = [_chunk("f.txt", rerank_score=0.5)]
        original_keys = set(chunks[0].keys())
        _, updated = generate_reference_list_from_chunks(chunks)
        # Original dict must not gain reference_id
        assert set(chunks[0].keys()) == original_keys
        assert "reference_id" in updated[0]


# ---------------------------------------------------------------------------
# Rerank OFF — byte-identical guard
# ---------------------------------------------------------------------------


@pytest.mark.offline
class TestRerankOff:
    def test_frequency_ordering_no_rerank_score(self):
        """When no chunk carries rerank_score, the sort must match the original
        frequency (desc) → first-appearance (asc) ordering exactly."""
        chunks = [
            _chunk("rare.txt"),          # first seen, count=1
            _chunk("common.txt"),        # second seen, count=3
            _chunk("common.txt"),
            _chunk("common.txt"),
            _chunk("middle.txt"),        # third seen, count=2
            _chunk("middle.txt"),
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)
        assert _ref_order(ref_list) == ["common.txt", "middle.txt", "rare.txt"]

    def test_first_appearance_tiebreaker_no_rerank(self):
        """Equal frequency: earlier first-appearance wins (lower index → [1])."""
        chunks = [
            _chunk("alpha.txt"),
            _chunk("beta.txt"),
            _chunk("alpha.txt"),
            _chunk("beta.txt"),
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)
        # Both have count=2; alpha first seen at index 0 → [1]
        assert _ref_order(ref_list) == ["alpha.txt", "beta.txt"]

    def test_single_file_no_rerank(self):
        chunks = [_chunk("only.txt"), _chunk("only.txt")]
        ref_list, updated = generate_reference_list_from_chunks(chunks)
        assert len(ref_list) == 1
        assert ref_list[0]["file_path"] == "only.txt"
        assert ref_list[0]["reference_id"] == "1"
        for c in updated:
            assert c["reference_id"] == "1"

    def test_rerank_score_none_treated_as_off(self):
        """An explicit rerank_score=None must not activate the rerank path."""
        chunks = [
            _chunk("rare.txt", rerank_score=None),
            _chunk("common.txt", rerank_score=None),
            _chunk("common.txt", rerank_score=None),
        ]
        ref_list, _ = generate_reference_list_from_chunks(chunks)
        # No valid scores → frequency path: common first
        assert _ref_order(ref_list) == ["common.txt", "rare.txt"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.offline
class TestEdgeCases:
    def test_empty_chunks_returns_empty(self):
        ref_list, updated = generate_reference_list_from_chunks([])
        assert ref_list == []
        assert updated == []

    def test_unknown_source_excluded_from_reference_list(self):
        chunks = [
            _chunk("real.txt"),
            {"file_path": "unknown_source", "content": "x"},
        ]
        ref_list, updated = generate_reference_list_from_chunks(chunks)
        file_paths = [r["file_path"] for r in ref_list]
        assert "unknown_source" not in file_paths
        assert "real.txt" in file_paths
        # The unknown_source chunk gets an empty reference_id
        unknown_chunk = next(c for c in updated if c["file_path"] == "unknown_source")
        assert unknown_chunk["reference_id"] == ""

    def test_missing_file_path_key_excluded_from_reference_list(self):
        chunks = [
            {"content": "no file path key"},
            _chunk("valid.txt"),
        ]
        ref_list, updated = generate_reference_list_from_chunks(chunks)
        assert len(ref_list) == 1
        assert ref_list[0]["file_path"] == "valid.txt"
        no_path = next(c for c in updated if "file_path" not in c or not c.get("file_path"))
        assert no_path["reference_id"] == ""

    def test_all_unknown_source_returns_empty_reference_list(self):
        chunks = [
            {"file_path": "unknown_source", "content": "x"},
            {"file_path": "unknown_source", "content": "y"},
        ]
        ref_list, updated = generate_reference_list_from_chunks(chunks)
        assert ref_list == []
        for c in updated:
            assert c["reference_id"] == ""

    def test_unknown_source_ignored_in_rerank_score_path(self):
        """Chunks with unknown_source must not contribute to file_path_max_score
        even when they carry a rerank_score — they must not appear in the
        reference list and must not affect the ordering of real files."""
        chunks = [
            {"file_path": "unknown_source", "content": "x", "rerank_score": 0.99},
            _chunk("real_low.txt", rerank_score=0.10),
            _chunk("real_high.txt", rerank_score=0.80),
        ]
        ref_list, updated = generate_reference_list_from_chunks(chunks)
        paths = _ref_order(ref_list)
        assert "unknown_source" not in paths
        assert paths[0] == "real_high.txt"
        assert paths[1] == "real_low.txt"

    def test_tie_rerank_and_tie_frequency_uses_first_appearance(self):
        """Ultimate tiebreaker: lower first-appearance index wins."""
        chunks = [
            _chunk("a.txt", rerank_score=0.5),
            _chunk("b.txt", rerank_score=0.5),
        ]
        # Both count=1, same score → first-appearance: a.txt at index 0
        ref_list, _ = generate_reference_list_from_chunks(chunks)
        assert _ref_order(ref_list) == ["a.txt", "b.txt"]

    def test_rerank_on_with_single_chunk(self):
        chunks = [_chunk("solo.txt", rerank_score=0.75)]
        ref_list, updated = generate_reference_list_from_chunks(chunks)
        assert len(ref_list) == 1
        assert ref_list[0]["reference_id"] == "1"
        assert ref_list[0]["file_path"] == "solo.txt"
        assert updated[0]["reference_id"] == "1"
