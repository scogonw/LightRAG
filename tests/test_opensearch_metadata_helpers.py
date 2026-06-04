"""Offline unit tests for the OpenSearch metadata-cascade helpers.

These cover the pure building blocks shared by the chunk and graph metadata
cascades (script construction + bulk-error classification) without requiring a
running OpenSearch cluster.
"""

import pytest

# opensearch_impl imports opensearch-py at module load; skip the whole module
# if the optional dependency isn't installed in this environment.
opensearch_impl = pytest.importorskip("lightrag.kg.opensearch_impl")

_metadata_upsert_script = opensearch_impl._metadata_upsert_script
_summarize_bulk_update_errors = opensearch_impl._summarize_bulk_update_errors


def test_metadata_upsert_script_shape():
    entry = {"resource_id": "r1", "access_level": "ORGANIZATION"}
    script = _metadata_upsert_script("r1", entry)
    assert script["lang"] == "painless"
    assert script["params"] == {"rid": "r1", "entry": entry}
    # The source must handle null, single-dict, and list-of-dicts shapes and
    # anchor on resource_id (replace matching entry, append when absent).
    src = script["source"]
    assert "m == null" in src
    assert "instanceof Map" in src
    assert "instanceof List" in src
    assert "resource_id" in src
    assert "m.add(entry)" in src  # append-when-absent (Option A injection)
    assert "rid == null" in src  # no-op guard when no anchor


def test_summarize_counts_pure_success():
    assert _summarize_bulk_update_errors(5, []) == {
        "updated": 5,
        "failures": 0,
        "not_found": 0,
    }


def test_summarize_counts_none_errors():
    assert _summarize_bulk_update_errors(3, None) == {
        "updated": 3,
        "failures": 0,
        "not_found": 0,
    }


def test_summarize_classifies_not_found_by_result():
    errors = [{"update": {"result": "not_found"}}]
    assert _summarize_bulk_update_errors(2, errors) == {
        "updated": 2,
        "failures": 0,
        "not_found": 1,
    }


def test_summarize_classifies_not_found_by_status_404():
    errors = [{"update": {"status": 404}}]
    assert _summarize_bulk_update_errors(0, errors) == {
        "updated": 0,
        "failures": 0,
        "not_found": 1,
    }


def test_summarize_classifies_real_failures():
    errors = [
        {"update": {"status": 500, "error": "boom"}},
        {"update": {"result": "not_found"}},
        {"something_else": True},
    ]
    assert _summarize_bulk_update_errors(1, errors) == {
        "updated": 1,
        "failures": 2,
        "not_found": 1,
    }


def _apply_upsert(metadata, resource_id, entry):
    """Python port of _METADATA_UPSERT_PAINLESS, for verifying the algorithm.

    Mirrors the Painless source's branches exactly so the intended upsert
    behaviour is checked offline (the script itself only runs in OpenSearch).
    """
    if resource_id is None:
        return metadata  # no-op guard
    if metadata is None:
        return entry
    if isinstance(metadata, dict):
        if metadata.get("resource_id") == resource_id:
            return entry
        return [metadata, entry]
    if isinstance(metadata, list):
        m = list(metadata)
        for i, e in enumerate(m):
            if isinstance(e, dict) and e.get("resource_id") == resource_id:
                m[i] = entry
                return m
        m.append(entry)
        return m
    return metadata


def test_upsert_null_sets_entry():
    entry = {"resource_id": "r1", "access_level": "ORGANIZATION"}
    assert _apply_upsert(None, "r1", entry) == entry


def test_upsert_dict_same_doc_replaced():
    old = {"resource_id": "r1", "access_level": "TEAM_MEMBERS"}
    entry = {"resource_id": "r1", "access_level": "ORGANIZATION"}
    assert _apply_upsert(old, "r1", entry) == entry


def test_upsert_dict_other_doc_keeps_both():
    other = {"resource_id": "r2", "access_level": "ONLY_ME"}
    entry = {"resource_id": "r1", "access_level": "ORGANIZATION"}
    assert _apply_upsert(other, "r1", entry) == [other, entry]


def test_upsert_list_replaces_matching_entry_only():
    a = {"resource_id": "r1", "access_level": "TEAM_MEMBERS"}
    b = {"resource_id": "r2", "access_level": "ONLY_ME"}
    entry = {"resource_id": "r1", "access_level": "ORGANIZATION"}
    assert _apply_upsert([a, b], "r1", entry) == [entry, b]


def test_upsert_list_injects_when_absent():
    # Option A: doc extracted this entity but never accumulated its entry.
    b = {"resource_id": "r2", "access_level": "ONLY_ME"}
    entry = {"resource_id": "r1", "access_level": "ORGANIZATION"}
    assert _apply_upsert([b], "r1", entry) == [b, entry]


def test_upsert_is_idempotent():
    entry = {"resource_id": "r1", "access_level": "ORGANIZATION"}
    once = _apply_upsert([{"resource_id": "r2"}], "r1", entry)
    twice = _apply_upsert(once, "r1", entry)
    assert twice == once
    assert twice.count(entry) == 1


def test_upsert_noop_when_resource_id_missing():
    old = [{"resource_id": "r2"}]
    assert _apply_upsert(old, None, {"x": 1}) == old
