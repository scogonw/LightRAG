# Update Document Metadata Route — Design

**Date:** 2026-04-29
**Scope:** Add a new API route that updates a document's `metadata` field, with cascade to the OpenSearch chunks vector index. OpenSearch backend only.

---

## 1. Motivation

LightRAG ingestion pipelines accept arbitrary `metadata` per document, which is later used at query time via `metadata_filter`. Today, once a document is ingested, the only way to change its metadata is to delete and re-ingest the document. That is expensive (re-extraction, re-embedding) and disruptive (chunk IDs change, KG nodes/edges churn).

This route gives callers a cheap, in-place way to re-tag a document — the most common reason being post-ingestion classification, access-level changes, or correcting metadata that was missing/wrong at upload time.

## 2. Scope

In scope:
- New endpoint `PATCH /documents/{doc_id}/metadata` that updates the `metadata` field on a single document.
- Shallow-merge semantics: keys in the patch are added/overwritten on the existing metadata; a key whose value is `null` is removed.
- Background cascade to the OpenSearch `chunks_vdb` index, so chunk-level `metadata_filter` queries reflect the change.

Out of scope (explicitly):
- Updating any other document field (`file_path`, `org_id`, `status`, `error_msg`, etc.).
- Cascading metadata changes to `entities_vdb` or `relations_vdb`. Entities and relations are intrinsically multi-source and their `metadata` lists do not tag entries by originating doc, so a safe cascade is not possible without a schema change. Updating doc-level metadata will *not* affect KG-side filtering.
- Batch updates across multiple documents.
- Non-OpenSearch storage backends. The endpoint returns 501 when configured backends are anything else.
- Updating the LLM cache (cache key does not include metadata).

## 3. Endpoint contract

**Route:** `PATCH /documents/{doc_id}/metadata`
**Auth:** `combined_auth` dependency (existing) + required `X-Org-Id` header.

### Request

```json
{ "metadata": { "department": "marketing", "year": 2026, "old_tag": null } }
```

- `metadata` (required, dict): non-null values are added/overwritten on the doc's metadata; `null` values delete that key. Empty dict is treated as a no-op.
- The body schema rejects unknown top-level fields.

### Response (200)

```json
{
  "status": "update_started" | "no_change" | "busy",
  "message": "...",
  "doc_id": "doc-123",
  "metadata": { "department": "marketing", "year": 2026 }
}
```

| `status` | Meaning |
|---|---|
| `update_started` | doc-status row updated synchronously; cascade to `chunks_vdb` dispatched in the background. |
| `no_change` | Patch was empty; no writes performed. |
| `busy` | Target doc is currently in PROCESSING or PREPROCESSED; refused to avoid racing with the ingestion pipeline. |

`metadata` in the response is the *resulting merged metadata* (after applying the patch).

### Errors

| Code | Condition |
|---|---|
| 404 | Document not found OR `org_id` mismatch (same response — no existence leak across tenants) |
| 422 | Pydantic validation (e.g., missing `metadata`, missing `X-Org-Id` header, non-JSON-serializable values, malformed body) |
| 501 | Configured backend is not OpenSearch (either `doc_status` or `chunks_vdb`) |
| 500 | Unexpected error during synchronous doc-status write |

Errors that occur in the background cascade do not surface as HTTP errors (the response has already been sent). They are logged and appended to `pipeline_status["history_messages"]`.

## 4. Control flow

```
PATCH /documents/{doc_id}/metadata     (X-Org-Id: <org>, body: {metadata: {...}})

1. Backend guard
   if not isinstance(rag.doc_status, OpenSearchDocStatusStorage)
      or not isinstance(rag.chunks_vdb, OpenSearchVectorDBStorage):
         return 501

2. Empty-patch short-circuit
   if request.metadata == {}:
        return 200 {status: "no_change", doc_id, metadata: <current>}

3. Load + ownership check
   existing = await rag.doc_status.get_by_id(doc_id)
   if existing is None or existing["org_id"] != x_org_id:
        return 404 "Document not found"

4. Pipeline race guard (target-doc only)
   if existing["status"] in {PROCESSING, PREPROCESSED}:
        return 200 {status: "busy", ...}

5. Snapshot old metadata BEFORE mutating
   old_metadata = existing.get("metadata") or {}

6. Compute new metadata (shallow merge; null = delete key)
   new_metadata = _shallow_merge_metadata(old_metadata, request.metadata)

7. Persist doc-status synchronously
   existing["metadata"] = new_metadata
   existing["updated_at"] = now_iso()
   await rag.doc_status.upsert({doc_id: existing})
   (any error here -> 500; cascade has not been dispatched yet)

8. Dispatch background cascade
   background_tasks.add_task(
       cascade_metadata_to_chunks,
       rag, doc_id, x_org_id, old_metadata, new_metadata
   )

9. Return 200 {status: "update_started", doc_id, metadata: new_metadata}
```

### Background cascade (`cascade_metadata_to_chunks`)

Calls a new method on `OpenSearchVectorDBStorage`:

```python
await rag.chunks_vdb.update_metadata_by_full_doc_id(
    full_doc_id=doc_id,
    org_id=x_org_id,
    old_metadata=old_metadata,
    new_metadata=new_metadata,
)
```

The method runs a single OpenSearch `update_by_query` against the chunks index (with `refresh=True`, so a separate `index_done_callback()` is unnecessary), scoped to `full_doc_id == doc_id AND org_id == x_org_id`, with a Painless script that handles three shapes the chunk's `metadata` field can take:

```painless
def m = ctx._source.metadata;
if (m == null) {
    ctx._source.metadata = params.new;
} else if (m instanceof Map) {
    // single-source chunk: replace whole
    ctx._source.metadata = params.new;
} else if (m instanceof List) {
    // multi-source chunk: replace the entry that equals old_metadata
    for (int i = 0; i < m.size(); i++) {
        if (m.get(i).equals(params.old)) {
            m.set(i, params.new);
            break;
        }
    }
}
```

The script is invoked with `params.old = old_metadata` and `params.new = new_metadata`.

The method returns `{"updated": int, "failures": int}`. The cascade task logs:

```
chunk metadata cascade: doc_id=X org_id=Y updated=N failures=K
```

If `failures > 0` or `updated == 0` and chunks for that doc do exist, append a warning to `pipeline_status["history_messages"]` so it surfaces in the existing pipeline status UI.

Errors raised inside the cascade task are caught, logged with traceback, and appended to `pipeline_status["history_messages"]`. They never propagate (FastAPI background tasks have already returned the response).

## 5. Multi-source metadata handling — known limitation

Chunks in LightRAG are content-deduped by hash. When two documents share the same chunk content, the chunk's `metadata` field is stored as a **list of dicts** — one entry per contributing document. Crucially, the entries are **not tagged with their originating `doc_id`** (see `_chunk_meta_matches_kb_filter` in `lightrag/operate.py:147` and the entity-merge logic in `lightrag/operate.py:2026-2053`).

Implications for this design:

1. **Match-by-value is the only mechanism.** The cascade script identifies "this doc's contribution" by comparing each list entry to `old_metadata`. If two docs happened to ship the *exact same* metadata, both contributions are indistinguishable, and only the first match is updated. This is acceptable: the rest of the system already treats those entries as equivalent (they're `frozenset`-deduped at merge time).

2. **Drift between doc-status and chunks**: if a chunk's metadata list contains an entry that differs from the doc's current metadata snapshot (e.g., the doc was updated through some other path in the past), the script's `equals(old)` check will not match, and that chunk is left alone. The cascade logs `unchanged > 0` so the operator can investigate.

3. **No cascade to entities/relations.** Their `metadata` lists are even more entangled (an entity merges across all chunks across all docs). A correct cascade would require either schema migration (adding a per-entry `doc_id` tag during ingestion) or accepting incorrect overwrites. Both are out of scope. Operators should be aware that KG-derived `metadata_filter` results will not reflect post-ingestion metadata changes.

These limitations are documented in the route's docstring so API consumers see them in the OpenAPI schema.

## 6. Concurrency & races

- **Pipeline busy on unrelated doc:** allowed. The guard is target-doc-only.
- **Pipeline busy on the target doc** (status = PROCESSING/PREPROCESSED): refused with `status="busy"` and HTTP 200 (consistent with `delete_document`'s busy response).
- **Concurrent PATCH on the same doc:** last write wins on the doc-status row. Cascades may interleave at the chunk level; each chunk ends up reflecting whichever cascade ran last. Acceptable for a metadata field; not worth distributed locking.
- **PATCH then immediate query:** the synchronous doc-status write is committed via `wait_for` refresh (already the default in `OpenSearchDocStatusStorage.upsert`). Chunk-level changes lag by the duration of the background task plus an `index_done_callback()` refresh — typically sub-second.

## 7. Code structure

| File | Change |
|---|---|
| `lightrag/api/routers/document_routes.py` | Add Pydantic models `UpdateDocumentMetadataRequest` and `UpdateDocumentMetadataResponse`. Add the route handler `update_document_metadata`. Add the helper `_shallow_merge_metadata` and the background-task function `cascade_metadata_to_chunks`. Place the route near `delete_document` (around line 2862) for thematic grouping. |
| `lightrag/kg/opensearch_impl.py` | Add `OpenSearchVectorDBStorage.update_metadata_by_full_doc_id(self, full_doc_id, org_id, old_metadata, new_metadata) -> dict`. |

No changes to `lightrag/base.py`. The new method is OpenSearch-specific by design; the route guards with `isinstance` and returns 501 for other backends.

### Pydantic models (sketch)

```python
class UpdateDocumentMetadataRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={...})
    metadata: Dict[str, Any] = Field(
        description="Patch dict. Non-null values are merged into the doc's "
                    "metadata; null values delete that key. Empty dict is a no-op."
    )

class UpdateDocumentMetadataResponse(BaseModel):
    status: Literal["update_started", "no_change", "busy"]
    message: str
    doc_id: str
    metadata: Dict[str, Any]
```

### Helper (sketch)

```python
def _shallow_merge_metadata(existing: dict | None, patch: dict) -> dict:
    """Merge patch into existing. null values in patch delete the key."""
    result = dict(existing or {})
    for k, v in patch.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = v
    return result
```

### Storage method (sketch)

```python
# In OpenSearchVectorDBStorage
async def update_metadata_by_full_doc_id(
    self,
    full_doc_id: str,
    org_id: str,
    old_metadata: dict,
    new_metadata: dict,
) -> dict:
    """Replace metadata on chunks belonging to (full_doc_id, org_id).

    Handles three metadata shapes on the chunk record:
      - missing/null      -> set to new_metadata
      - dict (single src) -> replace with new_metadata
      - list of dicts     -> replace the entry that equals old_metadata

    Returns: {"updated": int, "failures": int}
    """
    body = {
        "query": {"bool": {"must": [
            {"term": {"full_doc_id": full_doc_id}},
            {"term": {"org_id": org_id}},
        ]}},
        "script": {
            "lang": "painless",
            "source": "<painless source from section 4>",
            "params": {"old": old_metadata, "new": new_metadata},
        },
    }
    response = await self.client.update_by_query(
        index=self._index_name,
        body=body,
        refresh=True,
        conflicts="proceed",
    )
    return {
        "updated": response.get("updated", 0),
        "failures": len(response.get("failures", [])),
    }
```

## 8. Testing

### Offline unit tests (no OpenSearch)

Place in `tests/test_update_document_metadata.py`. Cover the merge helper and the 501 backend guard. No DB required.

| Test | Asserts |
|---|---|
| `test_shallow_merge_adds_keys` | New keys are added. |
| `test_shallow_merge_overwrites_keys` | Existing keys are overwritten by patch values. |
| `test_shallow_merge_null_deletes_key` | `{k: None}` removes `k`. |
| `test_shallow_merge_null_on_missing_key_is_noop` | Deleting a key that does not exist is silently OK. |
| `test_shallow_merge_empty_patch` | Empty patch returns the original dict unchanged. |
| `test_shallow_merge_existing_none` | Treats `None` existing as `{}`. |
| `test_patch_returns_501_when_backend_not_opensearch` | Configure LightRAG with JSON storages; PATCH returns 501. |

### Integration tests (require OpenSearch)

Marked `@pytest.mark.integration` and `@pytest.mark.requires_db`. Use the existing OpenSearch test fixtures.

| Test | Asserts |
|---|---|
| `test_patch_updates_doc_status_metadata` | After PATCH, `doc_status.get_by_id(doc_id)["metadata"]` reflects merged metadata; `updated_at` advances. |
| `test_patch_propagates_to_single_source_chunks` | Ingest 1 doc, PATCH metadata, all chunks for that `full_doc_id` show new metadata. |
| `test_patch_preserves_other_docs_metadata_on_shared_chunks` | Ingest 2 docs that share chunk content, PATCH only doc A's metadata, chunk's `metadata` list still contains doc B's entry intact. |
| `test_patch_null_value_removes_key_from_chunks` | PATCH `{"tag": null}` removes the key from chunk metadata too. |
| `test_patch_returns_404_for_nonexistent_doc` | Unknown `doc_id` -> 404. |
| `test_patch_returns_404_for_org_mismatch` | Doc belongs to org A; PATCH with `X-Org-Id: B` -> 404. |
| `test_patch_missing_org_header_returns_422` | No `X-Org-Id` -> 422. |
| `test_patch_returns_busy_when_doc_processing` | Force doc to PROCESSING; PATCH -> 200 `status="busy"`; no chunk changes. |
| `test_patch_busy_does_not_block_other_docs` | Doc A is PROCESSING; PATCH on doc B succeeds. |
| `test_patch_empty_metadata_returns_no_change` | PATCH `{"metadata": {}}` -> 200 `status="no_change"`; no writes. |
| `test_patch_idempotent` | Same PATCH twice yields the same final state on chunks and doc-status. |
| `test_patch_unauthenticated_returns_401_or_403` | Missing/invalid auth (per `combined_auth` config) -> 401/403. |

### Manual / out-of-test verification

The Painless script requires scripting to be enabled on the cluster. This is the default in OSS OpenSearch and Amazon OpenSearch Service. Operators running on a managed offering with scripting disabled by policy must enable it; the endpoint will surface the failure as a 500 (synchronous path will not hit it; cascade failures are logged).

## 9. Operational notes

- **OpenAPI docstring.** The route's docstring will spell out the multi-source limitation (Section 5) and the entities/relations non-cascade so API consumers know the bound.
- **Existing routes touched:** none modified; only additions.
- **Migration:** none required. New endpoint operates on existing data shapes.
- **Roll-out:** safe to ship behind no flag — endpoint returns 501 on non-OpenSearch deployments; OpenSearch deployments gain a new optional capability with no behavior change to existing endpoints.

## 10. Decisions captured

| # | Decision |
|---|---|
| 1 | Updatable fields: `metadata` only |
| 2 | Update semantics: shallow merge, `null` value = delete key |
| 3 | Cascade: background task |
| 4 | Cascade target: chunks_vdb only (not entities/relations — see Section 5) |
| 5 | Auth: `X-Org-Id` required; mismatch returns 404 (no existence leak) |
| 6 | Scope: single doc per request |
| 7 | Concurrency: refuse only when target doc is PROCESSING/PREPROCESSED |
| 8 | Non-OpenSearch backends: 501 at the route layer |
| 9 | Cascade implementation: OpenSearch `update_by_query` with Painless script (single round trip) |
| 10 | Doc-status update path: reuse existing `upsert()` (we already need to read for ownership/status checks) |
