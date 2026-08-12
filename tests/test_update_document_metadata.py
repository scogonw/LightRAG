"""Offline unit tests for the update-document-metadata route's helpers."""

import sys

sys.argv = sys.argv[:1]

from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from fastapi import APIRouter, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from lightrag.api.routers import document_routes  # noqa: E402
from lightrag.api.routers.document_routes import (  # noqa: E402
    _clean_cascade_entry,
    _shallow_merge_metadata,
    create_document_routes,
)
from lightrag.base import DocStatus  # noqa: E402


def test_shallow_merge_adds_keys():
    assert _shallow_merge_metadata({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_shallow_merge_overwrites_keys():
    assert _shallow_merge_metadata({"a": 1}, {"a": 2}) == {"a": 2}


def test_shallow_merge_null_deletes_key():
    assert _shallow_merge_metadata({"a": 1, "b": 2}, {"a": None}) == {"b": 2}


def test_shallow_merge_null_on_missing_key_is_noop():
    assert _shallow_merge_metadata({"a": 1}, {"missing": None}) == {"a": 1}


def test_shallow_merge_empty_patch():
    assert _shallow_merge_metadata({"a": 1}, {}) == {"a": 1}


def test_shallow_merge_existing_none():
    assert _shallow_merge_metadata(None, {"a": 1}) == {"a": 1}


def test_shallow_merge_does_not_mutate_existing():
    existing = {"a": 1}
    _shallow_merge_metadata(existing, {"b": 2})
    assert existing == {"a": 1}


def test_clean_cascade_entry_strips_bookkeeping_keys():
    entry = _clean_cascade_entry(
        {
            "access_level": "ORGANIZATION",
            "knowledgebase_id": "kb1",
            "resource_id": "r1",
            "processing_start_time": 123,
            "processing_end_time": 456,
        }
    )
    assert entry == {
        "access_level": "ORGANIZATION",
        "knowledgebase_id": "kb1",
        "resource_id": "r1",
    }


def test_clean_cascade_entry_keeps_user_keys():
    entry = _clean_cascade_entry({"department": "eng", "year": 2026, "resource_id": "r1"})
    assert entry == {"department": "eng", "year": 2026, "resource_id": "r1"}


def test_clean_cascade_entry_none_is_empty():
    assert _clean_cascade_entry(None) == {}


def _fresh_router(monkeypatch):
    """Swap in a clean APIRouter for the duration of a test.

    ``create_document_routes`` registers its handlers on a module-level router,
    so two calls in one process append duplicate routes and the first-registered
    closure — with the first test's ``rag`` — answers every request.
    """
    monkeypatch.setattr(
        document_routes,
        "router",
        APIRouter(prefix="/documents", tags=["documents"]),
    )


def _make_test_client_with_non_opensearch_backend(monkeypatch):
    """Build a FastAPI app with a LightRAG mock whose storages are NOT
    OpenSearch instances, so the 501 backend guard fires."""
    app = FastAPI()

    rag = MagicMock()
    # MagicMock() instances are not OpenSearchDocStatusStorage /
    # OpenSearchVectorDBStorage, so the isinstance checks return False.
    doc_manager = MagicMock()
    api_key = None

    _fresh_router(monkeypatch)
    app.include_router(create_document_routes(rag, doc_manager, api_key))
    return TestClient(app)


def test_patch_returns_501_when_backend_not_opensearch(monkeypatch):
    # Disable auth so we can exercise the 501 path directly.
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")

    client = _make_test_client_with_non_opensearch_backend(monkeypatch)
    response = client.patch(
        "/documents/doc-123/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"a": 1}},
    )
    assert response.status_code == 501
    assert "OpenSearch" in response.json()["detail"]


# ---------------------------------------------------------------------------
# End-to-end route tests against stubbed OpenSearch storages.
#
# The metadata a document actually carries lives in full_docs; doc_status only
# ever holds the pipeline's own processing timestamps, because every status
# transition overwrites that field. The stubs reproduce that split faithfully,
# which is what lets them exercise the consequences of the route having sourced
# its metadata from doc_status:
#
#   - full_docs was never written, so a reprocess rebuilt the document's chunks
#     from the stale ingest-time metadata and reverted the change. This was the
#     failure that actually occurred in production — verified against a patched
#     document whose chunks, nodes, entities and relations all carried the new
#     access_level while full_docs still held the old one.
#   - a patch that omits resource_id left the cascade with no anchor, so it
#     silently no-opped behind a 200. Clients that send the complete tenant blob
#     on every patch never hit this; clients that send only changed keys do.
#   - a patch that omits any other key wrote a partial entry onto the records,
#     since the painless upsert replaces a matched entry wholesale.
# ---------------------------------------------------------------------------

TENANT_METADATA = {
    "org_id": "org-test",
    "knowledgebase_id": "kb-1",
    "access_level": "ORGANIZATION",
    "resource_id": "res-1",
}


class _StubKV:
    """Stands in for OpenSearchKVStorage (full_docs and text_chunks).

    Both roles are the same class in production, and the chunk cascade has to
    reach text_chunks as well as the vector index — chunks live in both.
    """

    def __init__(self, records):
        self.records = records
        self.calls = []

    async def get_metadata(self, doc_id):
        record = self.records.get(doc_id)
        if record is None:
            return None
        return dict(record.get("metadata") or {})

    async def set_metadata(self, doc_id, metadata):
        if doc_id not in self.records:
            return False
        self.records[doc_id]["metadata"] = dict(metadata)
        return True

    async def update_metadata_for_ids(self, record_ids, resource_id, entry):
        self.calls.append(
            {
                "record_ids": list(record_ids),
                "resource_id": resource_id,
                "entry": dict(entry),
            }
        )
        return {"updated": len(record_ids), "failures": 0, "not_found": 0}


class _StubDocStatus:
    """Stands in for OpenSearchDocStatusStorage."""

    def __init__(self, records):
        self.records = records

    async def get_by_id(self, doc_id):
        record = self.records.get(doc_id)
        return dict(record) if record is not None else None

    async def upsert(self, data):
        for doc_id, value in data.items():
            self.records[doc_id] = dict(value)


class _StubVectorDB:
    """Stands in for OpenSearchVectorDBStorage (chunks/entities/relations)."""

    def __init__(self):
        self.calls = []

    async def update_metadata_for_ids(self, record_ids, resource_id, entry):
        self.calls.append(
            {
                "record_ids": list(record_ids),
                "resource_id": resource_id,
                "entry": dict(entry),
            }
        )
        return {"updated": len(record_ids), "failures": 0, "not_found": 0}


class _StubGraph:
    """Stands in for OpenSearchGraphStorage."""

    def __init__(self):
        self.calls = []

    async def update_metadata_by_chunk_ids(self, chunk_ids, resource_id, entry):
        self.calls.append(
            {
                "chunk_ids": list(chunk_ids),
                "resource_id": resource_id,
                "entry": dict(entry),
            }
        )
        return {
            "entity_names": ["Ent A"],
            "edge_pairs": [("Ent A", "Ent B")],
            "nodes": {"updated": 1, "failures": 0, "not_found": 0},
            "edges": {"updated": 1, "failures": 0, "not_found": 0},
        }


def _make_patch_client(
    monkeypatch,
    *,
    full_doc_metadata=TENANT_METADATA,
    doc_status_metadata=None,
    chunks_list=("chunk-1", "chunk-2"),
    full_doc_exists=True,
    status="processed",
):
    """Build a TestClient whose rag has OpenSearch-shaped stub storages.

    The route resolves the OpenSearch classes by importing them at call time, so
    swapping the module attributes makes the stubs satisfy its isinstance guard.
    """
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")

    import lightrag.kg.opensearch_impl as osi

    monkeypatch.setattr(osi, "OpenSearchKVStorage", _StubKV)
    monkeypatch.setattr(osi, "OpenSearchDocStatusStorage", _StubDocStatus)
    monkeypatch.setattr(osi, "OpenSearchVectorDBStorage", _StubVectorDB)
    monkeypatch.setattr(osi, "OpenSearchGraphStorage", _StubGraph)

    full_docs_records = {}
    if full_doc_exists:
        full_docs_records["doc-1"] = {
            "content": "body text",
            "metadata": dict(full_doc_metadata) if full_doc_metadata else {},
        }

    doc_status_records = {
        "doc-1": {
            "status": status,
            "org_id": "org-test",
            "chunks_list": list(chunks_list),
            # What the pipeline actually leaves behind here.
            "metadata": dict(
                doc_status_metadata
                if doc_status_metadata is not None
                else {"processing_start_time": 1, "processing_end_time": 2}
            ),
        }
    }

    rag = SimpleNamespace(
        workspace="test-ws",
        full_docs=_StubKV(full_docs_records),
        doc_status=_StubDocStatus(doc_status_records),
        text_chunks=_StubKV({}),
        chunks_vdb=_StubVectorDB(),
        entities_vdb=_StubVectorDB(),
        relationships_vdb=_StubVectorDB(),
        chunk_entity_relation_graph=_StubGraph(),
    )

    app = FastAPI()
    _fresh_router(monkeypatch)
    app.include_router(create_document_routes(rag, MagicMock(), None))
    return TestClient(app), rag


def test_patch_cascades_using_full_docs_metadata(monkeypatch):
    """A partial patch must still cascade.

    doc_status carries only processing timestamps, so a route sourcing its
    metadata from there sees no resource_id in a patch that does not resend one,
    and cascades nothing while still answering 200. Not the failure production
    hit — the caller there always resent the full tenant blob — but the one any
    client sending only its changed keys would.
    """
    client, rag = _make_patch_client(monkeypatch)

    response = client.patch(
        "/documents/doc-1/metadata?wait=true",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "update_started"
    assert body["metadata"]["access_level"] == "ONLY_ME"

    # The cascade actually ran, against the document's chunks.
    assert len(rag.chunks_vdb.calls) == 1
    call = rag.chunks_vdb.calls[0]
    assert call["resource_id"] == "res-1"
    assert call["record_ids"] == ["chunk-1", "chunk-2"]

    # Chunks live in text_chunks KV as well as the vector index, and KG-derived
    # chunks are read from KV. Updating only the vector index is what made every
    # entity/relation-derived chunk fail the query-side access filter.
    assert len(rag.text_chunks.calls) == 1
    assert rag.text_chunks.calls[0] == call

    # ... and reached the graph and both graph-side vector indices.
    assert len(rag.chunk_entity_relation_graph.calls) == 1
    assert len(rag.entities_vdb.calls) == 1
    assert len(rag.relationships_vdb.calls) == 1

    # Counts are reported back per stage.
    assert body["cascade"]["chunks"]["updated"] == 2
    assert body["cascade"]["text_chunks"]["updated"] == 2
    assert set(body["cascade"]) == {
        "chunks",
        "text_chunks",
        "nodes",
        "edges",
        "entities_vdb",
        "relations_vdb",
    }


def test_patch_cascade_entry_keeps_untouched_keys(monkeypatch):
    """A partial patch must write the COMPLETE merged metadata to records.

    The painless upsert replaces a matched entry wholesale, so cascading only
    the patched keys would strip knowledgebase_id off every chunk/entity/
    relation and silently detach the document from its knowledge base.
    """
    client, rag = _make_patch_client(monkeypatch)

    client.patch(
        "/documents/doc-1/metadata?wait=true",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    entry = rag.chunks_vdb.calls[0]["entry"]
    assert entry == {
        "org_id": "org-test",
        "knowledgebase_id": "kb-1",
        "access_level": "ONLY_ME",
        "resource_id": "res-1",
    }
    # Every cascade stage receives the same complete entry.
    assert rag.entities_vdb.calls[0]["entry"] == entry
    assert rag.relationships_vdb.calls[0]["entry"] == entry
    assert rag.chunk_entity_relation_graph.calls[0]["entry"] == entry


def test_patch_cascade_entry_stamps_org_id_when_metadata_lacks_it(monkeypatch):
    """The cascade must stamp org_id onto an entry that arrives without one.

    The query-side check reads org_id from *inside* the entry
    (``_chunk_meta_matches_kb_filter``). Ingestion stamps it now too (via
    ``utils.build_metadata_entry``), but entries written before that change have
    only a top-level record field — and records predating *that* have no
    top-level org for ``_entry_with_record_org`` to borrow either. Without the
    entry carrying the org, an org-path caller is still rejected after a
    perfectly successful cascade — the shape of the KG-derived chunk drop.

    So this stays the regression test for the legacy shape the cascade repairs.
    Note the default fixture metadata *does* include org_id, so this case needs
    its own document.
    """
    client, rag = _make_patch_client(
        monkeypatch,
        full_doc_metadata={
            "knowledgebase_id": "kb-1",
            "access_level": "ORGANIZATION",
            "resource_id": "res-1",
        },
    )

    client.patch(
        "/documents/doc-1/metadata?wait=true",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    for stage in (
        rag.chunks_vdb,
        rag.text_chunks,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunk_entity_relation_graph,
    ):
        assert stage.calls[0]["entry"]["org_id"] == "org-test"


def test_patch_writes_back_to_full_docs(monkeypatch):
    """The production failure this fix exists for.

    full_docs is what chunks are rebuilt from, so it must carry the patch or a
    reprocess resurrects the pre-patch access_level. Before the fix it was never
    written: patched documents were observed with the new access_level on every
    chunk, node, entity and relation, and the old one still sitting in full_docs
    — one reprocess away from silently reverting.
    """
    client, rag = _make_patch_client(monkeypatch)

    client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    assert rag.full_docs.records["doc-1"]["metadata"] == {
        "org_id": "org-test",
        "knowledgebase_id": "kb-1",
        "access_level": "ONLY_ME",
        "resource_id": "res-1",
    }
    # Content is left alone.
    assert rag.full_docs.records["doc-1"]["content"] == "body text"


def test_patch_mirrors_onto_doc_status_without_losing_bookkeeping(monkeypatch):
    client, rag = _make_patch_client(monkeypatch)

    client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    mirrored = rag.doc_status.records["doc-1"]["metadata"]
    assert mirrored["access_level"] == "ONLY_ME"
    assert mirrored["knowledgebase_id"] == "kb-1"
    assert mirrored["processing_start_time"] == 1
    assert mirrored["processing_end_time"] == 2


def test_patch_409_when_metadata_has_no_resource_id(monkeypatch):
    """Without an anchor the cascade cannot locate the document's entry, so the
    request must fail rather than report a success it did not deliver."""
    client, rag = _make_patch_client(
        monkeypatch,
        full_doc_metadata={"org_id": "org-test", "knowledgebase_id": "kb-1"},
    )

    response = client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    assert response.status_code == 409
    assert "resource_id" in response.json()["detail"]
    # Nothing was written anywhere.
    assert rag.chunks_vdb.calls == []
    assert "access_level" not in rag.full_docs.records["doc-1"]["metadata"]


def test_patch_409_when_full_docs_record_missing(monkeypatch):
    client, rag = _make_patch_client(monkeypatch, full_doc_exists=False)

    response = client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    assert response.status_code == 409
    assert "full_docs" in response.json()["detail"]
    assert rag.chunks_vdb.calls == []


def test_patch_422_when_changing_resource_id(monkeypatch):
    client, rag = _make_patch_client(monkeypatch)

    response = client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"resource_id": "res-2"}},
    )

    assert response.status_code == 422
    assert rag.chunks_vdb.calls == []
    assert rag.full_docs.records["doc-1"]["metadata"]["resource_id"] == "res-1"


def test_patch_422_when_removing_resource_id(monkeypatch):
    client, rag = _make_patch_client(monkeypatch)

    response = client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"resource_id": None}},
    )

    assert response.status_code == 422
    assert rag.chunks_vdb.calls == []


def test_patch_resource_id_echoed_unchanged_is_allowed(monkeypatch):
    """Re-sending the same resource_id is a no-op, not a rejection — clients
    that PUT back the whole blob should not be broken."""
    client, rag = _make_patch_client(monkeypatch)

    response = client.patch(
        "/documents/doc-1/metadata?wait=true",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"resource_id": "res-1", "access_level": "ONLY_ME"}},
    )

    assert response.status_code == 200
    assert rag.chunks_vdb.calls[0]["resource_id"] == "res-1"


def test_patch_without_chunks_reports_metadata_updated(monkeypatch):
    """A document that has not been chunked yet has nothing to cascade to, but
    the metadata write itself still has to land."""
    client, rag = _make_patch_client(
        monkeypatch, chunks_list=(), status="pending"
    )

    response = client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "metadata_updated"
    assert rag.chunks_vdb.calls == []
    assert rag.full_docs.records["doc-1"]["metadata"]["access_level"] == "ONLY_ME"


def test_patch_busy_document_is_not_written(monkeypatch):
    client, rag = _make_patch_client(monkeypatch, status=DocStatus.PROCESSING)

    response = client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    assert response.json()["status"] == "busy"
    assert rag.full_docs.records["doc-1"]["metadata"]["access_level"] == "ORGANIZATION"
    assert rag.chunks_vdb.calls == []


def test_patch_404_on_org_mismatch(monkeypatch):
    client, rag = _make_patch_client(monkeypatch)

    response = client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-other"},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    assert response.status_code == 404
    assert rag.chunks_vdb.calls == []


def test_empty_patch_reports_full_docs_metadata(monkeypatch):
    client, _ = _make_patch_client(monkeypatch)

    response = client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {}},
    )

    assert response.json()["status"] == "no_change"
    # The tenant blob, not doc_status' processing timestamps.
    assert response.json()["metadata"] == TENANT_METADATA


async def test_enqueue_puts_tenant_metadata_in_full_docs_not_doc_status():
    """Ingest-side half of the contract the route depends on: the tenant blob
    lands in full_docs and doc_status gets none of it."""
    from lightrag import LightRAG

    full_docs = _StubKV({})
    full_docs.upsert_calls = []

    async def _full_docs_upsert(data):
        full_docs.upsert_calls.append(data)
        for doc_id, value in data.items():
            full_docs.records[doc_id] = dict(value)

    async def _noop():
        return None

    full_docs.upsert = _full_docs_upsert
    full_docs.index_done_callback = _noop

    doc_status = _StubDocStatus({})

    async def _filter_keys(keys):
        return set(keys)

    doc_status.filter_keys = _filter_keys

    stub_self = SimpleNamespace(full_docs=full_docs, doc_status=doc_status)

    await LightRAG.apipeline_enqueue_documents(
        stub_self,
        input="hello world",
        file_paths="policy.pdf",
        metadata=dict(TENANT_METADATA),
        org_id="org-test",
    )

    (doc_id,) = full_docs.records
    assert full_docs.records[doc_id]["metadata"] == TENANT_METADATA
    assert "metadata" not in doc_status.records[doc_id]


def test_patch_deleting_key_clears_it_from_doc_status_mirror(monkeypatch):
    """A deleted key must not survive in the doc-status mirror."""
    client, rag = _make_patch_client(
        monkeypatch,
        full_doc_metadata={**TENANT_METADATA, "tag": "temporary"},
        doc_status_metadata={
            "processing_start_time": 1,
            "processing_end_time": 2,
            "tag": "temporary",
        },
    )

    client.patch(
        "/documents/doc-1/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"tag": None}},
    )

    mirrored = rag.doc_status.records["doc-1"]["metadata"]
    assert "tag" not in mirrored
    assert mirrored["processing_start_time"] == 1
    assert "tag" not in rag.full_docs.records["doc-1"]["metadata"]
