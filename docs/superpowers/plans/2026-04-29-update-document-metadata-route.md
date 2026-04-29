# Update Document Metadata Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PATCH /documents/{doc_id}/metadata` that updates a document's `metadata` field (shallow merge, `null` deletes a key) and propagates the change to the OpenSearch chunks vector index in the background. OpenSearch backends only — other backends return 501.

**Architecture:** New endpoint in `document_routes.py` performs synchronous validation, ownership check, pipeline-busy guard, and doc-status update via the existing `OpenSearchDocStatusStorage.upsert()`. A FastAPI `BackgroundTasks` callback then bulk-updates the chunks vector index by chunk IDs (read from the doc-status row's `chunks_list`) using a Painless script that handles three metadata shapes (null / dict / list-of-dicts). No cascade to entities/relations — those are intrinsically multi-source and lack per-entry doc tagging.

**Tech Stack:** Python (async), FastAPI, Pydantic v2, opensearchpy (`AsyncOpenSearch`, `helpers.async_bulk`), Painless scripting (server-side on OpenSearch). Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-04-29-update-document-metadata-route-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `lightrag/kg/opensearch_impl.py` (modify) | Add one method on `OpenSearchVectorDBStorage`: `update_metadata_for_ids(chunk_ids, old_metadata, new_metadata)`. Self-contained — uses existing client + `helpers.async_bulk`. |
| `lightrag/api/routers/document_routes.py` (modify) | Add Pydantic models `UpdateDocumentMetadataRequest`, `UpdateDocumentMetadataResponse`. Add helper `_shallow_merge_metadata(existing, patch)`. Add background-task function `cascade_metadata_to_chunks(rag, doc_id, chunk_ids, old_metadata, new_metadata)`. Add route handler `update_document_metadata`. Place near the existing `delete_document` handler (around current line 2862). |
| `tests/test_update_document_metadata.py` (create) | Offline unit tests for the merge helper and the 501 backend guard. |
| `tests/integration/test_update_document_metadata_opensearch.py` (create) | Integration tests against a real OpenSearch instance. Marked `@pytest.mark.integration` and skipped unless `LIGHTRAG_RUN_INTEGRATION=true`. |

---

## Task 1: Add `_shallow_merge_metadata` helper + offline tests

**Files:**
- Create: `tests/test_update_document_metadata.py`
- Modify: `lightrag/api/routers/document_routes.py` (add helper near top of file, after the existing `normalize_file_path` function around line 102)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_update_document_metadata.py` with the following content:

```python
"""Offline unit tests for the update-document-metadata route's helpers."""

import pytest

from lightrag.api.routers.document_routes import _shallow_merge_metadata


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_update_document_metadata.py -v`

Expected: All tests FAIL with `ImportError` or `AttributeError` because `_shallow_merge_metadata` does not exist.

- [ ] **Step 3: Implement `_shallow_merge_metadata`**

Open `lightrag/api/routers/document_routes.py`. After the existing `normalize_file_path` function (ends around line 101), add:

```python
def _shallow_merge_metadata(
    existing: dict | None, patch: dict
) -> dict:
    """Shallow-merge ``patch`` into ``existing`` and return a new dict.

    A key whose patch value is ``None`` is removed from the result; any other
    value overwrites or adds. ``existing`` is treated as ``{}`` when ``None``
    and is never mutated.
    """
    result = dict(existing or {})
    for k, v in patch.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = v
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_update_document_metadata.py -v`

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_update_document_metadata.py lightrag/api/routers/document_routes.py
git commit -m "feat(api): add _shallow_merge_metadata helper for update-doc route"
```

---

## Task 2: Add `update_metadata_for_ids` to `OpenSearchVectorDBStorage`

**Files:**
- Modify: `lightrag/kg/opensearch_impl.py` (add method to `OpenSearchVectorDBStorage` class — after `delete_entity_relation`, around line 3181)

- [ ] **Step 1: Inspect surrounding code**

Read `lightrag/kg/opensearch_impl.py` lines 2750–3200 to confirm:
- The class is `OpenSearchVectorDBStorage`
- It already imports `helpers` from `opensearchpy` (used by `upsert`)
- Existing methods follow the pattern: check `self._index_ready`, try/except `OpenSearchException`, log on error
- `_is_missing_index_error` is already imported / available in module scope (used elsewhere in the file)

No code change yet — context only.

- [ ] **Step 2: Add the method**

In `lightrag/kg/opensearch_impl.py`, inside `class OpenSearchVectorDBStorage`, immediately after the `delete_entity_relation` method (which ends around line 3181) and before `drop` (which begins around line 3183), add:

```python
    async def update_metadata_for_ids(
        self,
        chunk_ids: list[str],
        old_metadata: dict,
        new_metadata: dict,
    ) -> dict:
        """Replace ``metadata`` on the listed records via a Painless script.

        Handles three shapes the stored ``metadata`` field can take:
          - missing / null   -> set to ``new_metadata``
          - dict (single src) -> replace with ``new_metadata``
          - list of dicts    -> replace the list entry that equals
                                ``old_metadata`` (multi-source chunks)

        Args:
            chunk_ids: Record IDs to update. Empty list is a no-op.
            old_metadata: The doc's metadata snapshot before the patch was
                applied. Used to find the matching list entry on
                multi-source chunks.
            new_metadata: The merged metadata to write.

        Returns:
            ``{"updated": int, "failures": int, "not_found": int}`` —
            ``updated`` counts successful updates, ``not_found`` counts
            chunk IDs that no longer exist in the index, ``failures`` counts
            other errors.
        """
        if not chunk_ids:
            return {"updated": 0, "failures": 0, "not_found": 0}
        if not self._index_ready:
            return {"updated": 0, "failures": 0, "not_found": len(chunk_ids)}

        painless = (
            "def m = ctx._source.metadata; "
            "if (m == null) { ctx._source.metadata = params.new; } "
            "else if (m instanceof Map) { ctx._source.metadata = params.new; } "
            "else if (m instanceof List) { "
            "  for (int i = 0; i < m.size(); i++) { "
            "    if (m.get(i).equals(params.old)) { m.set(i, params.new); break; } "
            "  } "
            "}"
        )
        script = {
            "lang": "painless",
            "source": painless,
            "params": {"old": old_metadata, "new": new_metadata},
        }
        actions = [
            {
                "_op_type": "update",
                "_index": self._index_name,
                "_id": cid,
                "script": script,
            }
            for cid in chunk_ids
        ]
        try:
            success, errors = await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh=True
            )
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return {"updated": 0, "failures": 0, "not_found": len(chunk_ids)}
            logger.error(
                f"[{self.workspace}] Error updating chunk metadata: {e}"
            )
            return {"updated": 0, "failures": len(chunk_ids), "not_found": 0}

        not_found = 0
        failures = 0
        for err in errors or []:
            update_info = err.get("update") if isinstance(err, dict) else None
            if isinstance(update_info, dict) and (
                update_info.get("result") == "not_found"
                or update_info.get("status") == 404
            ):
                not_found += 1
            else:
                failures += 1
        return {"updated": success, "failures": failures, "not_found": not_found}
```

- [ ] **Step 3: Verify the file still parses**

Run: `python -c "import lightrag.kg.opensearch_impl"`

Expected: no output (silent success). Any `SyntaxError` or `ImportError` indicates the edit is malformed.

- [ ] **Step 4: Run lint to confirm**

Run: `ruff check lightrag/kg/opensearch_impl.py`

Expected: pre-existing lint state preserved (no new errors introduced by this change).

- [ ] **Step 5: Commit**

```bash
git add lightrag/kg/opensearch_impl.py
git commit -m "feat(opensearch): add update_metadata_for_ids on vector storage"
```

---

## Task 3: Add Pydantic request/response models

**Files:**
- Modify: `lightrag/api/routers/document_routes.py` (add models near the other request/response models, after `DeleteDocRequest` around line 449)

- [ ] **Step 1: Add the models**

In `lightrag/api/routers/document_routes.py`, after the `DeleteDocRequest` class (ends around line 449) and before `DeleteEntityRequest`, add:

```python
class UpdateDocumentMetadataRequest(BaseModel):
    """Request model for partial metadata update on a document.

    The ``metadata`` field is a patch dict applied with shallow-merge
    semantics: keys with non-null values are added or overwritten, keys
    whose value is ``null`` are removed. An empty dict is a no-op.
    """

    metadata: Dict[str, Any] = Field(
        description=(
            "Patch dict applied to the document's metadata. Non-null values "
            "are merged in (added or overwritten); null values delete the "
            "corresponding key. Empty dict is a no-op."
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "metadata": {
                    "department": "marketing",
                    "year": 2026,
                    "old_tag": None,
                }
            }
        },
    )


class UpdateDocumentMetadataResponse(BaseModel):
    """Response model for the update-document-metadata endpoint."""

    status: Literal["update_started", "no_change", "busy"] = Field(
        description="Outcome of the update request"
    )
    message: str = Field(description="Human-readable status message")
    doc_id: str = Field(description="The document ID that was targeted")
    metadata: Dict[str, Any] = Field(
        description="The document's metadata after applying the patch"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "update_started",
                "message": (
                    "Metadata updated. Cascade to chunks scheduled in "
                    "background."
                ),
                "doc_id": "doc-abc123",
                "metadata": {"department": "marketing", "year": 2026},
            }
        }
    )
```

- [ ] **Step 2: Verify the file still imports**

Run: `python -c "from lightrag.api.routers.document_routes import UpdateDocumentMetadataRequest, UpdateDocumentMetadataResponse"`

Expected: no output (silent success).

- [ ] **Step 3: Run lint**

Run: `ruff check lightrag/api/routers/document_routes.py`

Expected: no new errors compared to baseline.

- [ ] **Step 4: Commit**

```bash
git add lightrag/api/routers/document_routes.py
git commit -m "feat(api): add request/response models for update-document-metadata"
```

---

## Task 4: Add `cascade_metadata_to_chunks` background task

**Files:**
- Modify: `lightrag/api/routers/document_routes.py` (add module-level function above `create_document_routes` factory — before the existing `background_delete_documents` helper, search for that name to find the right area)

- [ ] **Step 1: Find the right location**

Run: `grep -n "background_delete_documents" lightrag/api/routers/document_routes.py`

This locates the existing background helper. Add the new helper *immediately above* the first definition of `background_delete_documents`.

- [ ] **Step 2: Add the function**

Add this function above `background_delete_documents`:

```python
async def cascade_metadata_to_chunks(
    rag: "LightRAG",
    doc_id: str,
    chunk_ids: list[str],
    old_metadata: dict,
    new_metadata: dict,
) -> None:
    """Background task: propagate a metadata change to the chunks vector index.

    Idempotent. Single-source chunks are overwritten outright; multi-source
    chunks have only the entry equal to ``old_metadata`` replaced. If no
    entry matches (drift), that chunk is left unchanged and counted in
    ``failures``/``not_found`` for the log line.
    """
    # Defensive: route should already have guarded this, but if backend was
    # swapped at runtime we still want to fail soft.
    from lightrag.kg.opensearch_impl import OpenSearchVectorDBStorage

    if not isinstance(rag.chunks_vdb, OpenSearchVectorDBStorage):
        logger.warning(
            f"cascade_metadata_to_chunks skipped for {doc_id}: chunks_vdb is "
            f"not OpenSearchVectorDBStorage ({type(rag.chunks_vdb).__name__})"
        )
        return

    if not chunk_ids:
        logger.info(
            f"cascade_metadata_to_chunks: doc_id={doc_id} no chunks to update"
        )
        return

    try:
        result = await rag.chunks_vdb.update_metadata_for_ids(
            chunk_ids=chunk_ids,
            old_metadata=old_metadata,
            new_metadata=new_metadata,
        )
        logger.info(
            f"cascade_metadata_to_chunks: doc_id={doc_id} "
            f"updated={result['updated']} "
            f"failures={result['failures']} "
            f"not_found={result['not_found']} "
            f"total_chunks={len(chunk_ids)}"
        )
        if result["failures"] > 0:
            logger.warning(
                f"cascade_metadata_to_chunks: doc_id={doc_id} had "
                f"{result['failures']} failures — see prior logs for details"
            )
    except Exception as e:
        # Never propagate — response was already sent. Log loudly.
        logger.error(
            f"cascade_metadata_to_chunks failed for doc_id={doc_id}: {e}"
        )
        logger.error(traceback.format_exc())
```

- [ ] **Step 3: Verify imports**

The function uses `LightRAG` (forward-reference string), `logger`, and `traceback` — all already imported at the top of the file (lines 9, 11, 27). No new imports needed.

Run: `python -c "from lightrag.api.routers.document_routes import cascade_metadata_to_chunks"`

Expected: no output (silent success).

- [ ] **Step 4: Run lint**

Run: `ruff check lightrag/api/routers/document_routes.py`

Expected: no new errors.

- [ ] **Step 5: Commit**

```bash
git add lightrag/api/routers/document_routes.py
git commit -m "feat(api): add cascade_metadata_to_chunks background task"
```

---

## Task 5: Add the route handler

**Files:**
- Modify: `lightrag/api/routers/document_routes.py` (add route inside `create_document_routes(...)` factory, near the `delete_document` handler around line 2862)

- [ ] **Step 1: Locate insertion point**

Run: `grep -n "async def delete_document" lightrag/api/routers/document_routes.py`

Insert the new route definition *immediately after* the `delete_document` function ends (search forward from that line for the next `@router.` decorator — insert before it).

- [ ] **Step 2: Add the route**

Inside the `create_document_routes` factory, immediately after the closing of `delete_document` and before the next route, add:

```python
    @router.patch(
        "/{doc_id}/metadata",
        response_model=UpdateDocumentMetadataResponse,
        dependencies=[Depends(combined_auth)],
        summary="Update a document's metadata (OpenSearch only).",
    )
    async def update_document_metadata(
        doc_id: str,
        body: UpdateDocumentMetadataRequest,
        background_tasks: BackgroundTasks,
        x_org_id: str = Header(
            ...,
            alias="X-Org-Id",
            description="Organization ID for multi-tenancy (required)",
        ),
    ) -> UpdateDocumentMetadataResponse:
        """
        Partially update a document's ``metadata`` via shallow merge.

        Semantics:
          - Keys in the request body with non-null values are added or
            overwritten on the document's metadata.
          - Keys with ``null`` values are removed.
          - An empty patch is a no-op (status="no_change").

        The doc-status row is updated synchronously. A background task
        propagates the same change to the chunks vector index so chunk-level
        ``metadata_filter`` queries reflect the update.

        Limitations:
          - Only supported when both ``doc_status`` and ``chunks_vdb`` are
            backed by OpenSearch. Other backends return 501.
          - Updates do **not** cascade to entities or relations: those
            indices are intrinsically multi-source and lack per-entry doc
            tagging, so a safe cascade is not possible.
          - When the same chunk is shared across multiple documents, the
            cascade replaces only the metadata entry equal to the doc's
            previous metadata snapshot. If the snapshot has drifted from
            what's stored on the chunk, that chunk is left unchanged.

        Args:
            doc_id: The document to update.
            body: ``{"metadata": <patch dict>}``.

        Returns:
            UpdateDocumentMetadataResponse:
              - status="update_started": doc-status updated, cascade dispatched.
              - status="no_change": empty patch, no writes performed.
              - status="busy": target doc is mid-ingestion (PROCESSING /
                PREPROCESSED), no writes performed.

        Raises:
            HTTPException 404: Document not found, or org_id mismatch.
            HTTPException 501: Backend is not OpenSearch.
            HTTPException 500: Unexpected internal error.
        """
        from lightrag.kg.opensearch_impl import (
            OpenSearchDocStatusStorage,
            OpenSearchVectorDBStorage,
        )

        # 1. Backend guard
        if not isinstance(rag.doc_status, OpenSearchDocStatusStorage) or not isinstance(
            rag.chunks_vdb, OpenSearchVectorDBStorage
        ):
            raise HTTPException(
                status_code=501,
                detail="Document metadata updates are only supported on OpenSearch storage.",
            )

        # 2. Empty-patch short-circuit (don't even read the doc — return current
        #    metadata if we have it, else an empty dict).
        if not body.metadata:
            existing = await rag.doc_status.get_by_id(doc_id)
            if existing is None or existing.get("org_id", "") != x_org_id:
                raise HTTPException(status_code=404, detail="Document not found")
            return UpdateDocumentMetadataResponse(
                status="no_change",
                message="Empty patch; no changes applied.",
                doc_id=doc_id,
                metadata=existing.get("metadata") or {},
            )

        try:
            # 3. Load + ownership check
            existing = await rag.doc_status.get_by_id(doc_id)
            if existing is None or existing.get("org_id", "") != x_org_id:
                raise HTTPException(status_code=404, detail="Document not found")

            # 4. Pipeline race guard (target-doc only)
            current_status = existing.get("status")
            if current_status in (
                DocStatus.PROCESSING.value,
                DocStatus.PREPROCESSED.value,
                DocStatus.PROCESSING,
                DocStatus.PREPROCESSED,
            ):
                return UpdateDocumentMetadataResponse(
                    status="busy",
                    message=(
                        "Document is currently being processed by the "
                        "ingestion pipeline. Retry once it leaves "
                        "PROCESSING/PREPROCESSED state."
                    ),
                    doc_id=doc_id,
                    metadata=existing.get("metadata") or {},
                )

            # 5. Snapshot old metadata BEFORE mutating
            old_metadata = dict(existing.get("metadata") or {})

            # 6. Compute new metadata
            new_metadata = _shallow_merge_metadata(old_metadata, body.metadata)

            # 7. Persist doc-status synchronously
            #    upsert() takes a {doc_id: data} dict. Strip the synthetic _id
            #    field that get_by_id attaches.
            updated_record = {k: v for k, v in existing.items() if k != "_id"}
            updated_record["metadata"] = new_metadata
            updated_record["updated_at"] = datetime.now(timezone.utc).isoformat()
            await rag.doc_status.upsert({doc_id: updated_record})

            # 8. Dispatch background cascade
            chunk_ids = list(existing.get("chunks_list") or [])
            background_tasks.add_task(
                cascade_metadata_to_chunks,
                rag,
                doc_id,
                chunk_ids,
                old_metadata,
                new_metadata,
            )

            return UpdateDocumentMetadataResponse(
                status="update_started",
                message=(
                    f"Metadata updated. Cascade to {len(chunk_ids)} chunks "
                    f"scheduled in background."
                ),
                doc_id=doc_id,
                metadata=new_metadata,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Error PATCH /documents/{doc_id}/metadata: {e}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 3: Verify the route is registered**

Run:
```bash
python -c "
from fastapi import FastAPI
from lightrag.api.routers.document_routes import create_document_routes
import unittest.mock as m

app = FastAPI()
rag = m.MagicMock()
doc_manager = m.MagicMock()
api_key = None

app.include_router(create_document_routes(rag, doc_manager, api_key))

paths = sorted(str(r.path) + ':' + ','.join(sorted(r.methods)) for r in app.routes if hasattr(r, 'methods'))
for p in paths:
    if 'metadata' in p or 'documents' in p and ('PATCH' in p or 'document' in p):
        print(p)
"
```

Expected: at least one line containing `/documents/{doc_id}/metadata:PATCH`.

- [ ] **Step 4: Run lint**

Run: `ruff check lightrag/api/routers/document_routes.py`

Expected: no new errors.

- [ ] **Step 5: Commit**

```bash
git add lightrag/api/routers/document_routes.py
git commit -m "feat(api): add PATCH /documents/{doc_id}/metadata route"
```

---

## Task 6: Add the 501-on-non-OpenSearch offline test

**Files:**
- Modify: `tests/test_update_document_metadata.py` (add a route-level test using FastAPI's TestClient)

- [ ] **Step 1: Add the test**

Append to `tests/test_update_document_metadata.py`:

```python
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.routers.document_routes import create_document_routes


def _make_test_client_with_non_opensearch_backend():
    """Build a FastAPI app with a LightRAG mock whose storages are NOT
    OpenSearch instances, so the 501 backend guard fires."""
    app = FastAPI()

    rag = MagicMock()
    # MagicMock() instances are not OpenSearchDocStatusStorage /
    # OpenSearchVectorDBStorage, so the isinstance checks return False.
    doc_manager = MagicMock()
    api_key = None

    app.include_router(create_document_routes(rag, doc_manager, api_key))
    return TestClient(app)


def test_patch_returns_501_when_backend_not_opensearch(monkeypatch):
    # Disable auth so we can exercise the 501 path directly.
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")

    client = _make_test_client_with_non_opensearch_backend()
    response = client.patch(
        "/documents/doc-123/metadata",
        headers={"X-Org-Id": "org-test"},
        json={"metadata": {"a": 1}},
    )
    assert response.status_code == 501
    assert "OpenSearch" in response.json()["detail"]
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_update_document_metadata.py::test_patch_returns_501_when_backend_not_opensearch -v`

Expected: PASS. If it fails because `combined_auth` rejects the request, the test fixture's monkeypatch needs adjusting; check `lightrag/api/utils_api.py` for the auth env var names and update.

- [ ] **Step 3: Run the full offline test file**

Run: `python -m pytest tests/test_update_document_metadata.py -v`

Expected: all 8 tests PASS (7 from Task 1 + 1 from Task 6).

- [ ] **Step 4: Commit**

```bash
git add tests/test_update_document_metadata.py
git commit -m "test(api): assert PATCH /documents/.../metadata returns 501 on non-OpenSearch"
```

---

## Task 7: Add OpenSearch integration tests — happy path

**Files:**
- Create: `tests/integration/test_update_document_metadata_opensearch.py`

- [ ] **Step 1: Inspect existing integration test patterns**

Run:
```bash
ls tests/integration/ 2>/dev/null || find tests -name '*opensearch*' -type f
```

If `tests/integration/` doesn't exist, look for any existing OpenSearch-backed test file (e.g., via `grep -rln "OpenSearchDocStatusStorage\|opensearch" tests/`) to mirror its fixture pattern.

- [ ] **Step 2: Identify the shared fixture for an OpenSearch-backed `LightRAG`**

Likely candidates: `tests/conftest.py`, `tests/test_graph_storage.py`, or `tests/lightrag_storage_test.py`. Look for a fixture that yields a `LightRAG` instance configured with OpenSearch storages and gates on `LIGHTRAG_RUN_INTEGRATION`.

If no such fixture exists, create one in `tests/integration/conftest.py` that:
1. Skips when `os.getenv("LIGHTRAG_RUN_INTEGRATION") != "true"`.
2. Reads OpenSearch connection from env (`OPENSEARCH_HOST`, `OPENSEARCH_PORT`, `OPENSEARCH_USERNAME`, `OPENSEARCH_PASSWORD`).
3. Uses a unique random workspace per test session (`f"test_meta_update_{uuid.uuid4().hex[:8]}"`) so concurrent test runs don't collide.
4. Constructs `LightRAG(..., kv_storage="OpenSearchKVStorage", vector_storage="OpenSearchVectorDBStorage", graph_storage="OpenSearchGraphStorage", doc_status_storage="OpenSearchDocStatusStorage", workspace=...)`, calls `await rag.initialize_storages()`, yields, then calls `await rag.finalize_storages()` and drops the workspace.

Spelling out the conftest fixture template (use as-is if no fixture exists):

```python
# tests/integration/conftest.py
import os
import uuid

import pytest_asyncio

from lightrag import LightRAG
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete


def _integration_disabled() -> bool:
    return os.getenv("LIGHTRAG_RUN_INTEGRATION", "").lower() != "true"


@pytest_asyncio.fixture
async def opensearch_rag(monkeypatch):
    import pytest
    if _integration_disabled():
        pytest.skip("set LIGHTRAG_RUN_INTEGRATION=true to run")

    workspace = f"test_meta_{uuid.uuid4().hex[:8]}"

    rag = LightRAG(
        working_dir=f"/tmp/{workspace}",
        workspace=workspace,
        kv_storage="OpenSearchKVStorage",
        vector_storage="OpenSearchVectorDBStorage",
        graph_storage="OpenSearchGraphStorage",
        doc_status_storage="OpenSearchDocStatusStorage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
    )
    await rag.initialize_storages()
    try:
        yield rag
    finally:
        # Drop test indices to leave the cluster clean
        for store in (rag.doc_status, rag.chunks_vdb, rag.entities_vdb,
                      rag.relationships_vdb, rag.full_docs, rag.text_chunks,
                      rag.llm_response_cache):
            try:
                await store.drop()
            except Exception:
                pass
        await rag.finalize_storages()
```

If a working fixture already exists, reuse it instead of creating this one — the tests below only need a `LightRAG` instance and a way to issue HTTP requests against the routes.

- [ ] **Step 3: Write the happy-path test file**

Create `tests/integration/test_update_document_metadata_opensearch.py`:

```python
"""Integration tests for PATCH /documents/{doc_id}/metadata against a real
OpenSearch cluster. Skipped unless LIGHTRAG_RUN_INTEGRATION=true.

These tests exercise the full route: synchronous doc-status update +
background cascade to chunks vector index.
"""

import asyncio
import os
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.routers.document_routes import create_document_routes


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
    # doc_manager is only used by upload-related routes; pass a stub.
    class _Stub:
        input_dir = None
    app.include_router(create_document_routes(rag, _Stub(), api_key=None))
    return TestClient(app)


async def _ingest_one_doc(rag, *, content: str, file_path: str, org_id: str,
                          metadata: dict) -> str:
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

    # Poll until the document is PROCESSED (or timeout after 60s).
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

    # Verify the doc-status row reflects the merge
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

    # Background cascade is dispatched but we already returned; wait for it
    # plus the OpenSearch refresh.
    await asyncio.sleep(2)
    await rag.chunks_vdb.index_done_callback()

    # Read the chunk records by ID and confirm metadata
    stored = await rag.doc_status.get_by_id(doc_id)
    chunk_ids = stored["chunks_list"]
    assert chunk_ids, "doc has no chunks; ingestion may have failed"

    chunks = await rag.chunks_vdb.get_by_ids(chunk_ids)
    for chunk in chunks:
        assert chunk is not None, "chunk missing from vector store"
        # Single-source chunk -> metadata is a dict
        assert chunk["metadata"] == {"label": "after"}, chunk
```

- [ ] **Step 4: Run the integration tests (only if you have OpenSearch running)**

Run:
```bash
LIGHTRAG_RUN_INTEGRATION=true python -m pytest \
    tests/integration/test_update_document_metadata_opensearch.py -v
```

Expected: 2 tests PASS.

If you don't have an OpenSearch instance handy, run without the env var to confirm they skip cleanly:
```bash
python -m pytest tests/integration/test_update_document_metadata_opensearch.py -v
```
Expected: 2 tests SKIPPED.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_update_document_metadata_opensearch.py
# also commit conftest.py if you created one in step 2
git add tests/integration/conftest.py 2>/dev/null || true
git commit -m "test(integration): add happy-path tests for update-document-metadata route"
```

---

## Task 8: Integration tests — null deletion + multi-source preservation

**Files:**
- Modify: `tests/integration/test_update_document_metadata_opensearch.py`

- [ ] **Step 1: Append the null-deletion test**

Add to `tests/integration/test_update_document_metadata_opensearch.py`:

```python
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
```

- [ ] **Step 2: Append the multi-source preservation test**

Add to the same file:

```python
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

    # Identical content -> identical chunk content hash -> deduped chunk record
    # whose metadata becomes a list of two entries.
    shared_content = (
        "This sentence is identical across docs. "
        "And so is this one. They will produce the same chunks."
    )

    doc_a = await _ingest_one_doc(
        rag, content=shared_content, file_path=f"shareA-{uuid.uuid4().hex[:6]}.txt",
        org_id=org_id, metadata={"src": "A"},
    )
    doc_b = await _ingest_one_doc(
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
        # Multi-source chunk: metadata should be a list with both entries
        if isinstance(meta, list):
            keys_present = sorted(
                m.get("src") for m in meta if isinstance(m, dict)
            )
            # A's contribution updated to A2; B's contribution untouched
            assert keys_present == ["A2", "B"], meta
        else:
            # Some chunks may not actually be shared if chunking produced
            # different boundaries; for those, just confirm A's update
            # propagated.
            assert meta.get("src") == "A2", meta
```

- [ ] **Step 3: Run the new tests**

Run:
```bash
LIGHTRAG_RUN_INTEGRATION=true python -m pytest \
    tests/integration/test_update_document_metadata_opensearch.py::test_patch_null_value_removes_key \
    tests/integration/test_update_document_metadata_opensearch.py::test_patch_preserves_other_docs_metadata_on_shared_chunks \
    -v
```

Expected: 2 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_update_document_metadata_opensearch.py
git commit -m "test(integration): assert null-deletion and multi-source preservation"
```

---

## Task 9: Integration tests — error paths

**Files:**
- Modify: `tests/integration/test_update_document_metadata_opensearch.py`

- [ ] **Step 1: Append the error-path tests**

Add to the same integration test file:

```python
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
    # updated_at should NOT have been bumped; metadata unchanged
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
```

- [ ] **Step 2: Run the new tests**

Run:
```bash
LIGHTRAG_RUN_INTEGRATION=true python -m pytest \
    tests/integration/test_update_document_metadata_opensearch.py -v -k "404 or 422 or no_change or busy or idempotent"
```

Expected: 6 tests PASS.

- [ ] **Step 3: Run the entire integration file end-to-end**

Run:
```bash
LIGHTRAG_RUN_INTEGRATION=true python -m pytest \
    tests/integration/test_update_document_metadata_opensearch.py -v
```

Expected: all 8 integration tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_update_document_metadata_opensearch.py
git commit -m "test(integration): cover 404/422/busy/no_change/idempotent paths"
```

---

## Task 10: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full offline test suite**

Run: `python -m pytest tests/test_update_document_metadata.py -v`

Expected: all 8 tests PASS.

- [ ] **Step 2: Lint everything we touched**

Run:
```bash
ruff check \
    lightrag/api/routers/document_routes.py \
    lightrag/kg/opensearch_impl.py \
    tests/test_update_document_metadata.py \
    tests/integration/test_update_document_metadata_opensearch.py
```

Expected: no errors introduced by this work (pre-existing issues elsewhere in those files are out of scope).

- [ ] **Step 3: Sanity-check OpenAPI surface**

Run:
```bash
python -c "
from fastapi import FastAPI
from unittest.mock import MagicMock
from lightrag.api.routers.document_routes import create_document_routes
app = FastAPI()
app.include_router(create_document_routes(MagicMock(), MagicMock(), api_key=None))
import json
schema = app.openapi()
# Print every documents/* path with its methods
for path, ops in sorted(schema['paths'].items()):
    if path.startswith('/documents'):
        print(path, sorted(ops.keys()))
"
```

Expected output includes a line like: `/documents/{doc_id}/metadata ['patch']`.

- [ ] **Step 4: Run integration suite if OpenSearch is available**

Run:
```bash
LIGHTRAG_RUN_INTEGRATION=true python -m pytest \
    tests/integration/test_update_document_metadata_opensearch.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Spot-check the route end-to-end via curl** (optional, requires running `lightrag-server`)

```bash
# Start the server (in another terminal)
lightrag-server

# In this terminal, ingest a doc, then PATCH its metadata
curl -X POST http://localhost:9621/documents/text \
    -H 'Content-Type: application/json' \
    -H 'X-Org-Id: org-demo' \
    -d '{"text":"Hello world.","metadata":{"phase":"alpha"}}'

# Find the doc_id (e.g. via /documents/paginated), then:
curl -X PATCH "http://localhost:9621/documents/<DOC_ID>/metadata" \
    -H 'Content-Type: application/json' \
    -H 'X-Org-Id: org-demo' \
    -d '{"metadata":{"phase":"beta","priority":"high"}}'
```

Expected: 200 with `status: "update_started"` and the merged metadata.

- [ ] **Step 6: Final commit (if anything outstanding)**

```bash
git status
```

If clean: nothing to commit. Otherwise, commit any straggler test/lint fixes with a descriptive message.
