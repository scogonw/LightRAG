"""Offline unit tests for the OpenSearch metadata-cascade helpers.

These cover the pure building blocks shared by the chunk and graph metadata
cascades (script construction + bulk-error classification) without requiring a
running OpenSearch cluster.
"""

import pytest

# opensearch_impl imports opensearch-py at module load; skip the whole module
# if the optional dependency isn't installed in this environment.
opensearch_impl = pytest.importorskip("lightrag.kg.opensearch_impl")

from lightrag.utils import merge_metadata_entry, remove_metadata_entry

_metadata_upsert_script = opensearch_impl._metadata_upsert_script
_metadata_remove_script = opensearch_impl._metadata_remove_script
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
    """Exercise the production merge helper under this module's argument order.

    This was a hand-written Python port of _METADATA_UPSERT_PAINLESS. It is now
    a thin adapter over ``utils.merge_metadata_entry``, which ingestion uses to
    merge a document's entry into a shared chunk — so these cases verify the
    real implementation rather than a copy of it that could drift from it.
    """
    return merge_metadata_entry(metadata, entry, resource_id)


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


# ---------------------------------------------------------------------------
# OpenSearchKVStorage.get_metadata / set_metadata
#
# These back the metadata PATCH route, which reads and writes a document's
# tenant metadata (org_id / knowledgebase_id / access_level / resource_id) in
# full_docs. Both must avoid touching the document's ``content``.
# ---------------------------------------------------------------------------


class _FakeClient:
    """Records the OpenSearch calls made against it."""

    def __init__(self, *, mget_response=None, update_error=None):
        self.mget_response = mget_response
        self.update_error = update_error
        self.mget_calls = []
        self.update_calls = []

    async def mget(self, index, body):
        self.mget_calls.append({"index": index, "body": body})
        return self.mget_response

    async def update(self, index, id, body, refresh=None):
        self.update_calls.append(
            {"index": index, "id": id, "body": body, "refresh": refresh}
        )
        if self.update_error is not None:
            raise self.update_error
        return {"result": "updated"}


def _make_kv(client):
    kv = opensearch_impl.OpenSearchKVStorage.__new__(
        opensearch_impl.OpenSearchKVStorage
    )
    kv.client = client
    kv._index_name = "idx"
    kv._index_ready = True
    kv.workspace = "ws"
    return kv


async def test_get_metadata_source_filters_to_metadata_only():
    """Must not pull the document's full content off the cluster."""
    client = _FakeClient(
        mget_response={
            "docs": [
                {
                    "found": True,
                    "_id": "doc-1",
                    "_source": {"metadata": {"resource_id": "r1"}},
                }
            ]
        }
    )
    kv = _make_kv(client)

    assert await kv.get_metadata("doc-1") == {"resource_id": "r1"}
    assert client.mget_calls[0]["body"] == {
        "docs": [{"_id": "doc-1", "_source": ["metadata"]}]
    }


async def test_get_metadata_missing_record_is_none():
    client = _FakeClient(mget_response={"docs": [{"found": False, "_id": "doc-1"}]})
    assert await _make_kv(client).get_metadata("doc-1") is None


async def test_get_metadata_record_without_metadata_is_empty_dict():
    """Distinct from a missing record: the document exists, it just has no
    metadata. The route treats these differently (409 vs. mergeable)."""
    client = _FakeClient(
        mget_response={"docs": [{"found": True, "_id": "doc-1", "_source": {}}]}
    )
    assert await _make_kv(client).get_metadata("doc-1") == {}


async def test_set_metadata_uses_partial_update():
    client = _FakeClient()
    kv = _make_kv(client)

    assert await kv.set_metadata("doc-1", {"access_level": "ONLY_ME"}) is True
    call = client.update_calls[0]
    assert call["id"] == "doc-1"
    assert call["refresh"] is True
    # A partial "doc" update, so content is left untouched.
    assert set(call["body"]) == {"doc"}
    assert call["body"]["doc"]["metadata"] == {"access_level": "ONLY_ME"}
    assert "content" not in call["body"]["doc"]


async def test_set_metadata_missing_record_returns_false():
    client = _FakeClient(update_error=opensearch_impl.NotFoundError(404, "missing", {}))
    assert await _make_kv(client).set_metadata("doc-1", {"a": 1}) is False


async def test_metadata_methods_noop_when_index_missing():
    kv = _make_kv(_FakeClient())
    kv._index_ready = False
    assert await kv.get_metadata("doc-1") is None
    assert await kv.set_metadata("doc-1", {"a": 1}) is False


# ---------------------------------------------------------------------------
# Withdrawing a document's entry (the delete path)
#
# A chunk id is a content hash, so one record can belong to several documents.
# Deleting a document must withdraw only its own entry; deleting the record
# outright empties every sibling that shares the content.
# ---------------------------------------------------------------------------


def test_metadata_remove_script_shape():
    script = _metadata_remove_script("r1")
    assert script["lang"] == "painless"
    assert script["params"] == {"rid": "r1"}
    src = script["source"]
    assert "m == null" in src
    assert "instanceof Map" in src
    assert "instanceof List" in src
    # Collapses to a bare dict when one entry survives, matching
    # utils.remove_metadata_entry.
    assert "keep.size() == 1" in src
    assert "ctx._source.metadata = null" in src


def test_remove_last_entry_marks_record_unreferenced():
    entry = {"resource_id": "r1", "access_level": "ORGANIZATION"}
    remaining, unreferenced = remove_metadata_entry(entry, "r1")
    assert remaining is None
    assert unreferenced is True


def test_remove_keeps_sibling_and_collapses_to_dict():
    mine = {"resource_id": "r1", "access_level": "CHAT_WIDGET"}
    theirs = {"resource_id": "r2", "access_level": "TEAM_MEMBERS"}
    remaining, unreferenced = remove_metadata_entry([mine, theirs], "r1")
    assert remaining == theirs
    assert unreferenced is False


def test_remove_keeps_multiple_siblings_as_list():
    mine = {"resource_id": "r1"}
    a = {"resource_id": "r2"}
    b = {"resource_id": "r3"}
    remaining, unreferenced = remove_metadata_entry([mine, a, b], "r1")
    assert remaining == [a, b]
    assert unreferenced is False


def test_remove_unknown_resource_leaves_record_intact():
    theirs = {"resource_id": "r2", "access_level": "ONLY_ME"}
    remaining, unreferenced = remove_metadata_entry(theirs, "r1")
    assert remaining == theirs
    assert unreferenced is False


def test_remove_without_metadata_reports_unreferenced():
    """Records carrying no metadata keep the original delete-outright path."""
    _, unreferenced = remove_metadata_entry(None, "r1")
    assert unreferenced is True


def test_upsert_then_remove_round_trips():
    """Two documents share a chunk; deleting one leaves the other untouched."""
    a = {"resource_id": "r1", "access_level": "CHAT_WIDGET"}
    b = {"resource_id": "r2", "access_level": "ORGANIZATION"}
    shared = merge_metadata_entry(merge_metadata_entry(None, a, "r1"), b, "r2")
    assert shared == [a, b]
    remaining, unreferenced = remove_metadata_entry(shared, "r1")
    assert remaining == b
    assert unreferenced is False
    remaining, unreferenced = remove_metadata_entry(remaining, "r2")
    assert remaining is None
    assert unreferenced is True
