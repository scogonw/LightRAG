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
