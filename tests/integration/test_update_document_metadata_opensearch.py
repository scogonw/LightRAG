"""Integration tests for PATCH /documents/{doc_id}/metadata against a real
OpenSearch cluster. Skipped unless --run-integration is passed AND
LIGHTRAG_RUN_INTEGRATION=true is set.

These tests exercise the full route: synchronous metadata write to full_docs
(the source of truth chunks are rebuilt from) plus the cascade to the chunks
vector index and to the document's entities and relations.
"""

import asyncio
import os
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# sys.argv shim so importing document_routes (which calls parse_args() at
# import time) doesn't choke on pytest's CLI flags.
import sys

sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import create_document_routes  # noqa: E402


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.getenv("LIGHTRAG_RUN_INTEGRATION", "").lower() != "true",
        reason="set LIGHTRAG_RUN_INTEGRATION=true to run integration tests",
    ),
]


def _make_client(rag) -> TestClient:
    app = FastAPI()

    class _Stub:
        input_dir = None

    app.include_router(create_document_routes(rag, _Stub(), api_key=None))
    return TestClient(app)


async def _ingest_one_doc(
    rag,
    *,
    content: str,
    file_path: str,
    org_id: str,
    metadata: dict,
) -> str:
    """Insert a document and return its doc_id once it is PROCESSED.

    Uses ``pipeline_index_texts`` (the same path the /documents/text route
    uses) because it accepts ``org_id`` whereas ``rag.ainsert`` does not.
    """
    from lightrag.api.routers.document_routes import pipeline_index_texts
    from lightrag.base import DocStatus
    from lightrag.utils import generate_track_id

    track_id = generate_track_id("test")
    await pipeline_index_texts(
        rag,
        texts=[content],
        file_sources=[file_path],
        track_id=track_id,
        metadata=metadata,
        org_id=org_id,
    )

    for _ in range(60):
        docs = await rag.doc_status.get_docs_by_track_id(track_id)
        for doc_id, info in docs.items():
            if info.status == DocStatus.PROCESSED:
                return doc_id
        await asyncio.sleep(1)
    raise AssertionError(
        f"Document for {file_path} did not reach PROCESSED within 60s"
    )


@pytest.mark.asyncio
async def test_patch_updates_full_docs_metadata(opensearch_rag):
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="The quick brown fox jumps over the lazy dog.",
        file_path=f"happy-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={
            "department": "engineering",
            "year": 2025,
            "resource_id": "res-happy",
        },
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"year": 2026, "tag": "added"}},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "update_started"
    assert body["doc_id"] == doc_id
    expected = {
        "department": "engineering",
        "year": 2026,
        "tag": "added",
        "resource_id": "res-happy",
    }
    assert body["metadata"] == expected

    # full_docs is the source of truth: it is what chunks are rebuilt from, so
    # the patch has to land here or a reprocess resurrects the old values.
    assert await rag.full_docs.get_metadata(doc_id) == expected

    # doc-status carries the same keys mirrored alongside the pipeline's own
    # bookkeeping, which the mirror must not drop.
    stored = await rag.doc_status.get_by_id(doc_id)
    assert stored["metadata"].items() >= expected.items()


@pytest.mark.asyncio
async def test_patch_without_resource_id_returns_409(opensearch_rag):
    """No anchor means the cascade cannot find the document's entry on its
    chunks/entities/relations, so the request must fail rather than report a
    success it did not deliver."""
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="Legacy document ingested before resource_id existed.",
        file_path=f"legacy-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"department": "engineering"},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    assert response.status_code == 409, response.text
    assert "resource_id" in response.json()["detail"]
    # Nothing was written.
    assert await rag.full_docs.get_metadata(doc_id) == {"department": "engineering"}


@pytest.mark.asyncio
async def test_patch_changing_resource_id_returns_422(opensearch_rag):
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="Document whose identity key must stay put.",
        file_path=f"anchor-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"resource_id": "res-anchor", "access_level": "ORGANIZATION"},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"resource_id": "res-other"}},
    )

    assert response.status_code == 422, response.text
    stored = await rag.full_docs.get_metadata(doc_id)
    assert stored["resource_id"] == "res-anchor"


@pytest.mark.asyncio
async def test_patch_wait_reports_cascade_counts(opensearch_rag):
    """With wait=true the cascade runs inline and reports what it touched, so a
    caller can tell a real propagation from a silent no-op."""
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content=(
            "Alice works at Acme Corporation in Berlin. "
            "Bob reports to Alice on the platform team."
        ),
        file_path=f"wait-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={
            "resource_id": "res-wait",
            "knowledgebase_id": "kb-wait",
            "access_level": "ORGANIZATION",
        },
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata?wait=true",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"access_level": "ONLY_ME"}},
    )

    assert response.status_code == 200, response.text
    cascade = response.json()["cascade"]
    assert set(cascade) == {
        "chunks",
        "nodes",
        "edges",
        "entities_vdb",
        "relations_vdb",
    }
    assert cascade["chunks"]["updated"] > 0
    assert cascade["chunks"]["failures"] == 0

    # The cascade wrote the COMPLETE merged blob, not just the patched key:
    # a partial entry would have stripped knowledgebase_id off every record and
    # silently detached the document from its knowledge base.
    stored = await rag.doc_status.get_by_id(doc_id)
    chunks = await rag.chunks_vdb.get_by_ids(stored["chunks_list"])
    for chunk in chunks:
        assert chunk is not None
        assert chunk["metadata"] == {
            "resource_id": "res-wait",
            "knowledgebase_id": "kb-wait",
            "access_level": "ONLY_ME",
        }, chunk


@pytest.mark.asyncio
async def test_patch_propagates_to_single_source_chunks(opensearch_rag):
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content=(
            "Distinct sentence one for chunk A. "
            "Distinct sentence two for chunk B. "
            "Distinct sentence three for chunk C."
        ),
        file_path=f"single-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"label": "before", "resource_id": "res-single"},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"label": "after"}},
    )
    assert response.status_code == 200, response.text

    await asyncio.sleep(2)
    await rag.chunks_vdb.index_done_callback()

    stored = await rag.doc_status.get_by_id(doc_id)
    chunk_ids = stored["chunks_list"]
    assert chunk_ids, "doc has no chunks; ingestion may have failed"

    chunks = await rag.chunks_vdb.get_by_ids(chunk_ids)
    for chunk in chunks:
        assert chunk is not None, "chunk missing from vector store"
        # Anchored upsert replaces this doc's entry; resource_id is retained.
        assert chunk["metadata"] == {
            "label": "after",
            "resource_id": "res-single",
        }, chunk


@pytest.mark.asyncio
async def test_patch_null_value_removes_key(opensearch_rag):
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="Content for null-deletion test, sufficiently long.",
        file_path=f"null-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"keep": "yes", "remove": "yes", "resource_id": "res-null"},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"remove": None}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"] == {"keep": "yes", "resource_id": "res-null"}

    assert await rag.full_docs.get_metadata(doc_id) == {
        "keep": "yes",
        "resource_id": "res-null",
    }

    await asyncio.sleep(2)
    await rag.chunks_vdb.index_done_callback()

    stored = await rag.doc_status.get_by_id(doc_id)
    chunks = await rag.chunks_vdb.get_by_ids(stored["chunks_list"])
    for chunk in chunks:
        assert "remove" not in (chunk["metadata"] or {})
        assert (chunk["metadata"] or {}).get("keep") == "yes"


@pytest.mark.asyncio
async def test_patch_preserves_other_docs_metadata_on_shared_chunks(
    opensearch_rag,
):
    """When two docs share identical chunk content (same content hash), the
    chunk's metadata is stored as a list of two dicts. PATCHing only doc A's
    metadata must replace ONLY doc A's entry; doc B's entry must stay intact.
    """
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"

    shared_content = (
        "This sentence is identical across docs. "
        "And so is this one. They will produce the same chunks."
    )

    doc_a = await _ingest_one_doc(
        rag, content=shared_content, file_path=f"shareA-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id, metadata={"src": "A", "resource_id": "res-A"},
    )
    await _ingest_one_doc(
        rag, content=shared_content, file_path=f"shareB-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id, metadata={"src": "B", "resource_id": "res-B"},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_a}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"src": "A2"}},
    )
    assert response.status_code == 200, response.text

    await asyncio.sleep(2)
    await rag.chunks_vdb.index_done_callback()

    stored_a = await rag.doc_status.get_by_id(doc_a)
    chunks = await rag.chunks_vdb.get_by_ids(stored_a["chunks_list"])
    for chunk in chunks:
        meta = chunk["metadata"]
        # Anchored on resource_id: doc A's entry (res-A) becomes src=A2;
        # doc B's entry (res-B) is untouched.
        if isinstance(meta, list):
            by_res = {m.get("resource_id"): m.get("src") for m in meta if isinstance(m, dict)}
            assert by_res.get("res-A") == "A2", meta
            assert by_res.get("res-B") == "B", meta
        else:
            assert meta.get("src") == "A2", meta


async def _nodes_edges_for_doc(rag, chunk_ids):
    """Scan the graph nodes/edges indices for records sourced from chunk_ids.

    Returns (node_hits, edge_hits) as raw OpenSearch hits (full _source) so
    tests can assert on the stored ``metadata`` of the document's entities and
    relations.
    """
    graph = rag.chunk_entity_relation_graph
    await graph._refresh_graph_indices_if_dirty(refresh_nodes=True, refresh_edges=True)
    node_hits = await graph._scan_by_source_ids(
        graph._nodes_index, list(chunk_ids), source_fields=True
    )
    edge_hits = await graph._scan_by_source_ids(
        graph._edges_index, list(chunk_ids), source_fields=True
    )
    return node_hits, edge_hits


def _meta_has_value(meta, key, value):
    """True if metadata (dict or list-of-dicts) contains key==value somewhere."""
    if isinstance(meta, dict):
        return meta.get(key) == value
    if isinstance(meta, list):
        return any(isinstance(m, dict) and m.get(key) == value for m in meta)
    return False


@pytest.mark.asyncio
async def test_patch_propagates_to_entities_and_relations(opensearch_rag):
    """PATCH must cascade onto graph nodes/edges and the entity/relation vector
    indices, not just doc-status and chunks."""
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content=(
            "Alice Johnson works at Acme Corporation in Paris. "
            "Acme Corporation was founded by Bob Smith in 1998. "
            "Alice Johnson manages the research team at Acme Corporation."
        ),
        file_path=f"graph-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"label": "before", "resource_id": "res-graph"},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"label": "after"}},
    )
    assert response.status_code == 200, response.text

    # Let the background graph cascade run, then make search views current.
    await asyncio.sleep(3)
    await rag.chunk_entity_relation_graph.index_done_callback()
    await rag.entities_vdb.index_done_callback()
    await rag.relationships_vdb.index_done_callback()

    stored = await rag.doc_status.get_by_id(doc_id)
    chunk_ids = stored["chunks_list"]
    assert chunk_ids, "doc has no chunks; ingestion may have failed"

    node_hits, edge_hits = await _nodes_edges_for_doc(rag, chunk_ids)
    assert node_hits, "no entities were extracted; cannot verify graph cascade"

    # Every graph node/edge sourced from this doc must reflect the new value.
    for hit in node_hits:
        meta = hit["_source"].get("metadata")
        assert _meta_has_value(meta, "label", "after"), hit["_source"]
        assert not _meta_has_value(meta, "label", "before"), hit["_source"]
    for hit in edge_hits:
        meta = hit["_source"].get("metadata")
        assert _meta_has_value(meta, "label", "after"), hit["_source"]

    # Entity/relation vector records (IDs derived from names/pairs) must match.
    from lightrag.utils import compute_mdhash_id

    entity_ids = [
        compute_mdhash_id(str(hit["_id"]), prefix="ent-") for hit in node_hits
    ]
    entity_records = await rag.entities_vdb.get_by_ids(entity_ids)
    seen_entity = False
    for rec in entity_records:
        if rec is None:
            continue
        seen_entity = True
        assert _meta_has_value(rec.get("metadata"), "label", "after"), rec
    assert seen_entity, "no entity vector records found for derived IDs"

    rel_ids = set()
    for hit in edge_hits:
        src = hit["_source"].get("source_node_id")
        tgt = hit["_source"].get("target_node_id")
        if src is None or tgt is None:
            continue
        rel_ids.add(compute_mdhash_id(f"{src}{tgt}", prefix="rel-"))
        rel_ids.add(compute_mdhash_id(f"{tgt}{src}", prefix="rel-"))
    if rel_ids:
        rel_records = await rag.relationships_vdb.get_by_ids(list(rel_ids))
        for rec in rel_records:
            if rec is None:
                continue
            assert _meta_has_value(rec.get("metadata"), "label", "after"), rec


@pytest.mark.asyncio
async def test_patch_preserves_other_docs_metadata_on_shared_entities(
    opensearch_rag,
):
    """An entity extracted from two docs carries a list of per-doc metadata.
    PATCHing only doc A must replace doc A's entry and leave doc B's intact."""
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"

    shared_content = (
        "Acme Corporation is headquartered in Paris. "
        "Acme Corporation employs thousands of people worldwide."
    )

    doc_a = await _ingest_one_doc(
        rag, content=shared_content, file_path=f"entA-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id, metadata={"src": "A", "resource_id": "res-entA"},
    )
    await _ingest_one_doc(
        rag, content=shared_content, file_path=f"entB-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id, metadata={"src": "B", "resource_id": "res-entB"},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_a}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"src": "A2"}},
    )
    assert response.status_code == 200, response.text

    await asyncio.sleep(3)
    await rag.chunk_entity_relation_graph.index_done_callback()

    stored_a = await rag.doc_status.get_by_id(doc_a)
    node_hits, _ = await _nodes_edges_for_doc(rag, stored_a["chunks_list"])
    assert node_hits, "no shared entities found"

    for hit in node_hits:
        meta = hit["_source"].get("metadata")
        # Anchored on resource_id: doc A's entry (res-entA) becomes src=A2.
        # Doc B's entry (res-entB), if present, must survive. Every entity
        # extracted from doc A must end up with an entry for res-entA (replaced
        # or injected), so src=A2 is always present and the old src=A is gone.
        if isinstance(meta, list):
            by_res = {m.get("resource_id"): m.get("src") for m in meta if isinstance(m, dict)}
            assert by_res.get("res-entA") == "A2", meta
            assert "A" not in by_res.values(), meta
            if "res-entB" in by_res:
                assert by_res["res-entB"] == "B", meta
        else:
            assert meta.get("src") == "A2", meta


@pytest.mark.asyncio
async def test_patch_returns_404_for_nonexistent_doc(opensearch_rag):
    rag = opensearch_rag
    client = _make_client(rag)
    response = client.patch(
        "/documents/does-not-exist/metadata",
        headers={"X-Org-Id": "org-anything"},
        json={"metadata": {"a": 1}},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


@pytest.mark.asyncio
async def test_patch_returns_404_for_org_mismatch(opensearch_rag):
    rag = opensearch_rag
    org_a = f"orgA-{uuid.uuid4().hex[:6]}"
    org_b = f"orgB-{uuid.uuid4().hex[:6]}"

    doc_id = await _ingest_one_doc(
        rag,
        content="Content for org-mismatch test.",
        file_path=f"orgmismatch-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_a,
        metadata={"x": 1},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_b},  # wrong org
        json={"metadata": {"x": 2}},
    )
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "Document not found"


@pytest.mark.asyncio
async def test_patch_missing_org_header_returns_422(opensearch_rag):
    rag = opensearch_rag
    client = _make_client(rag)
    response = client.patch(
        "/documents/anything/metadata",
        json={"metadata": {"a": 1}},
        # No X-Org-Id header
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_patch_empty_metadata_returns_no_change(opensearch_rag):
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="Content for empty-patch test.",
        file_path=f"empty-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"original": True, "resource_id": "res-empty"},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "no_change"
    # Reports the document's real metadata, not doc-status' processing keys.
    assert body["metadata"] == {"original": True, "resource_id": "res-empty"}

    assert await rag.full_docs.get_metadata(doc_id) == {
        "original": True,
        "resource_id": "res-empty",
    }


@pytest.mark.asyncio
async def test_patch_returns_busy_when_doc_processing(opensearch_rag):
    """Force a doc into PROCESSING and confirm PATCH returns busy."""
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="Content for busy test.",
        file_path=f"busy-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"v": 1},
    )

    # Force the doc back into PROCESSING for the duration of this test.
    stored = await rag.doc_status.get_by_id(doc_id)
    stored = {k: v for k, v in stored.items() if k != "_id"}
    stored["status"] = "processing"
    await rag.doc_status.upsert({doc_id: stored})

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"v": 2}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "busy"
    # Metadata should NOT have been updated
    assert await rag.full_docs.get_metadata(doc_id) == {"v": 1}


@pytest.mark.asyncio
async def test_patch_idempotent(opensearch_rag):
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="Content for idempotency test.",
        file_path=f"idem-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"v": 1, "resource_id": "res-idem"},
    )

    client = _make_client(rag)
    headers = {"X-Org-Id": org_id}
    payload = {"metadata": {"v": 2, "tag": "x"}}

    r1 = client.patch(
        f"/documents/{doc_id}/metadata", headers=headers, json=payload
    )
    assert r1.status_code == 200
    r2 = client.patch(
        f"/documents/{doc_id}/metadata", headers=headers, json=payload
    )
    assert r2.status_code == 200

    await asyncio.sleep(2)
    await rag.chunks_vdb.index_done_callback()

    stored = await rag.doc_status.get_by_id(doc_id)
    assert await rag.full_docs.get_metadata(doc_id) == {
        "v": 2,
        "tag": "x",
        "resource_id": "res-idem",
    }

    expected = {"v": 2, "tag": "x", "resource_id": "res-idem"}
    chunks = await rag.chunks_vdb.get_by_ids(stored["chunks_list"])
    for chunk in chunks:
        meta = chunk["metadata"]
        if isinstance(meta, dict):
            # Idempotent: re-running the same patch must not duplicate entries.
            assert meta == expected
        elif isinstance(meta, list):
            assert expected in meta
            assert meta.count(expected) == 1


@pytest.mark.asyncio
async def test_patch_busy_does_not_block_other_docs(opensearch_rag):
    """The pipeline-busy guard is target-doc-only: PROCESSING status on doc A
    must not prevent a successful PATCH on doc B."""
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"

    doc_a = await _ingest_one_doc(
        rag,
        content="Content for busy-doc-A.",
        file_path=f"busyA-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"name": "A", "resource_id": "res-busyA"},
    )
    doc_b = await _ingest_one_doc(
        rag,
        content="Different content for doc-B that should remain editable.",
        file_path=f"busyB-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"name": "B", "resource_id": "res-busyB"},
    )

    # Force doc A into PROCESSING.
    stored_a = await rag.doc_status.get_by_id(doc_a)
    stored_a = {k: v for k, v in stored_a.items() if k != "_id"}
    stored_a["status"] = "processing"
    await rag.doc_status.upsert({doc_a: stored_a})

    client = _make_client(rag)

    # PATCH on doc A -> busy
    r_a = client.patch(
        f"/documents/{doc_a}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"name": "A2"}},
    )
    assert r_a.status_code == 200, r_a.text
    assert r_a.json()["status"] == "busy"

    # PATCH on doc B -> succeeds
    r_b = client.patch(
        f"/documents/{doc_b}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"name": "B2"}},
    )
    assert r_b.status_code == 200, r_b.text
    assert r_b.json()["status"] == "update_started"
    assert r_b.json()["metadata"] == {"name": "B2", "resource_id": "res-busyB"}

    assert await rag.full_docs.get_metadata(doc_b) == {
        "name": "B2",
        "resource_id": "res-busyB",
    }


@pytest.mark.asyncio
async def test_patch_unauthenticated_returns_401_or_403(opensearch_rag, monkeypatch):
    """When auth is configured (API key required), a request without the
    Authorization header / api key must be rejected before reaching the
    handler.
    """
    rag = opensearch_rag

    # Configure an API key so combined_auth enforces it.
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-secret-key")

    app = FastAPI()

    class _Stub:
        input_dir = None

    app.include_router(
        create_document_routes(rag, _Stub(), api_key="test-secret-key")
    )
    client = TestClient(app)

    response = client.patch(
        "/documents/any-doc-id/metadata",
        headers={"X-Org-Id": "org-anything"},
        json={"metadata": {"a": 1}},
    )
    assert response.status_code in (401, 403), response.text
