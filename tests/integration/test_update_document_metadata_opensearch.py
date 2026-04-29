"""Integration tests for PATCH /documents/{doc_id}/metadata against a real
OpenSearch cluster. Skipped unless --run-integration is passed AND
LIGHTRAG_RUN_INTEGRATION=true is set.

These tests exercise the full route: synchronous doc-status update +
background cascade to chunks vector index.
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
async def test_patch_updates_doc_status_metadata(opensearch_rag):
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="The quick brown fox jumps over the lazy dog.",
        file_path=f"happy-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"department": "engineering", "year": 2025},
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
    assert body["metadata"] == {
        "department": "engineering",
        "year": 2026,
        "tag": "added",
    }

    stored = await rag.doc_status.get_by_id(doc_id)
    assert stored["metadata"] == {
        "department": "engineering",
        "year": 2026,
        "tag": "added",
    }


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
        metadata={"label": "before"},
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
        assert chunk["metadata"] == {"label": "after"}, chunk


@pytest.mark.asyncio
async def test_patch_null_value_removes_key(opensearch_rag):
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="Content for null-deletion test, sufficiently long.",
        file_path=f"null-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"keep": "yes", "remove": "yes"},
    )

    client = _make_client(rag)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        headers={"X-Org-Id": org_id},
        json={"metadata": {"remove": None}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"] == {"keep": "yes"}

    stored = await rag.doc_status.get_by_id(doc_id)
    assert stored["metadata"] == {"keep": "yes"}

    await asyncio.sleep(2)
    await rag.chunks_vdb.index_done_callback()

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
        org_id=org_id, metadata={"src": "A"},
    )
    await _ingest_one_doc(
        rag, content=shared_content, file_path=f"shareB-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id, metadata={"src": "B"},
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
        if isinstance(meta, list):
            keys_present = sorted(
                m.get("src") for m in meta if isinstance(m, dict)
            )
            assert keys_present == ["A2", "B"], meta
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
        metadata={"original": True},
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
    assert body["metadata"] == {"original": True}

    stored = await rag.doc_status.get_by_id(doc_id)
    assert stored["metadata"] == {"original": True}


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
    after = await rag.doc_status.get_by_id(doc_id)
    assert after["metadata"] == {"v": 1}


@pytest.mark.asyncio
async def test_patch_idempotent(opensearch_rag):
    rag = opensearch_rag
    org_id = f"org-{uuid.uuid4().hex[:6]}"
    doc_id = await _ingest_one_doc(
        rag,
        content="Content for idempotency test.",
        file_path=f"idem-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id,
        metadata={"v": 1},
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
    assert stored["metadata"] == {"v": 2, "tag": "x"}

    chunks = await rag.chunks_vdb.get_by_ids(stored["chunks_list"])
    for chunk in chunks:
        meta = chunk["metadata"]
        if isinstance(meta, dict):
            assert meta == {"v": 2, "tag": "x"}
        elif isinstance(meta, list):
            assert {"v": 2, "tag": "x"} in meta
