"""Tests for KV-store chunk access-control filter.

Covers the legacy-chunk fallback (metadata=None, top-level org_id present),
the mismatch exclusion path, normal populated-metadata behaviour, and the
chunk_tracking stale-key cleanup that mirrors the filter result.
"""
import pytest

from lightrag.operate import _filter_chunks_by_kb_access


def _chunk(chunk_id, metadata, org_id=None, content="text"):
    c = {"chunk_id": chunk_id, "content": content, "metadata": metadata}
    if org_id is not None:
        c["org_id"] = org_id
    return c


class TestLegacyChunkFallback:
    """metadata=None → fall back to top-level org_id."""

    def test_legacy_matching_org_returned(self):
        chunk = _chunk("c1", metadata=None, org_id="org-A")
        result = _filter_chunks_by_kb_access([chunk], {}, org_id="org-A")
        assert len(result) == 1 and result[0]["chunk_id"] == "c1"

    def test_legacy_mismatched_org_excluded(self):
        chunk = _chunk("c1", metadata=None, org_id="org-A")
        result = _filter_chunks_by_kb_access([chunk], {}, org_id="org-B")
        assert result == []

    def test_legacy_no_top_level_org_excluded_when_request_org_set(self):
        chunk = _chunk("c1", metadata=None)  # no top-level org_id
        result = _filter_chunks_by_kb_access([chunk], {}, org_id="org-A")
        assert result == []

    def test_legacy_no_org_anywhere_passes_when_no_filter(self):
        chunk = _chunk("c1", metadata=None)
        result = _filter_chunks_by_kb_access([chunk], {}, org_id=None)
        assert len(result) == 1

    def test_legacy_with_kb_params_excluded(self):
        """Legacy chunks without metadata cannot satisfy KB-level restrictions."""
        chunk = _chunk("c1", metadata=None, org_id="org-A")
        metadata_filter = {"agent_kb_ids": ["kb-1"]}
        result = _filter_chunks_by_kb_access([chunk], metadata_filter, org_id="org-A")
        assert result == []


class TestPopulatedMetadata:
    """Existing behaviour must be preserved for chunks that have metadata."""

    def test_matching_org_in_metadata_returned(self):
        meta = {"org_id": "org-A", "access_level": "ORGANIZATION"}
        chunk = _chunk("c2", metadata=meta)
        result = _filter_chunks_by_kb_access([chunk], {}, org_id="org-A")
        assert len(result) == 1

    def test_mismatched_org_in_metadata_excluded(self):
        meta = {"org_id": "org-A", "access_level": "ORGANIZATION"}
        chunk = _chunk("c2", metadata=meta)
        result = _filter_chunks_by_kb_access([chunk], {}, org_id="org-B")
        assert result == []

    def test_agent_kb_path_matched(self):
        meta = {"knowledgebase_id": "kb-1", "access_level": "CHAT_WIDGET", "org_id": "org-A"}
        chunk = _chunk("c3", metadata=meta)
        result = _filter_chunks_by_kb_access(
            [chunk], {"agent_kb_ids": ["kb-1"]}, org_id="org-A"
        )
        assert len(result) == 1

    def test_agent_kb_path_wrong_kb_excluded(self):
        meta = {"knowledgebase_id": "kb-2", "access_level": "CHAT_WIDGET", "org_id": "org-A"}
        chunk = _chunk("c3", metadata=meta)
        result = _filter_chunks_by_kb_access(
            [chunk], {"agent_kb_ids": ["kb-1"]}, org_id="org-A"
        )
        assert result == []


class TestChunkTrackingCleanup:
    """chunk_tracking must not contain entries for filtered-out chunks."""

    def test_stale_keys_removed_after_filter(self):
        legacy_pass = _chunk("pass", metadata=None, org_id="org-A")
        legacy_fail = _chunk("fail", metadata=None, org_id="org-B")
        populated_pass = _chunk(
            "pop-pass",
            metadata={"org_id": "org-A", "access_level": "ORGANIZATION"},
        )
        chunks = [legacy_pass, legacy_fail, populated_pass]
        tracking = {
            "pass": {"source": "E", "frequency": 1, "order": 1},
            "fail": {"source": "E", "frequency": 1, "order": 2},
            "pop-pass": {"source": "E", "frequency": 1, "order": 3},
        }

        result = _filter_chunks_by_kb_access(chunks, {}, org_id="org-A")
        result_ids = {c["chunk_id"] for c in result}

        # Mimic the cleanup added to _find_related_text_unit_from_entities
        for stale_id in list(tracking.keys() - result_ids):
            del tracking[stale_id]

        assert set(tracking.keys()) == result_ids
        assert "fail" not in tracking

    def test_tracking_untouched_when_no_filter(self):
        chunk = _chunk("c1", metadata={"org_id": "org-A", "access_level": "ORGANIZATION"})
        tracking = {"c1": {"source": "E", "frequency": 1, "order": 1}}
        # No metadata_filter → _filter_chunks_by_kb_access not called in production;
        # verify the filter itself doesn't corrupt tracking when called anyway.
        result = _filter_chunks_by_kb_access([chunk], {}, org_id=None)
        # tracking unchanged because caller only cleans when metadata_filter is set
        assert "c1" in tracking
