"""Offline unit tests for the OpenSearch metadata-cascade helpers.

These cover the pure building blocks shared by the chunk and graph metadata
cascades (script construction + bulk-error classification) without requiring a
running OpenSearch cluster.
"""

import pytest

# opensearch_impl imports opensearch-py at module load; skip the whole module
# if the optional dependency isn't installed in this environment.
opensearch_impl = pytest.importorskip("lightrag.kg.opensearch_impl")

_metadata_replace_script = opensearch_impl._metadata_replace_script
_summarize_bulk_update_errors = opensearch_impl._summarize_bulk_update_errors


def test_metadata_replace_script_shape():
    script = _metadata_replace_script({"a": 1}, {"a": 2})
    assert script["lang"] == "painless"
    assert script["params"] == {"old": {"a": 1}, "new": {"a": 2}}
    # The source must handle null, single-dict, and list-of-dicts shapes.
    src = script["source"]
    assert "m == null" in src
    assert "instanceof Map" in src
    assert "instanceof List" in src
    # List branch replaces only the entry equal to params.old.
    assert "equals(params.old)" in src


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
