# Remove Soft-Delete Fields and Switch to Path-Based Doc IDs

**Date:** 2026-04-22

## Summary

Remove the `is_deleted` and `deleted_at` fields from the document-status subsystem entirely, convert the soft-delete code path into a hard-delete, and change document ID generation from `md5(content + file_path)` to `md5(file_path)` so that filename becomes the collision dimension.

**After this change:**
- Two uploads of the *same* filename are rejected as duplicates (captured as `dup-*` FAILED status records, not hard errors).
- Two uploads of the *same content* under *different* filenames are allowed.
- No notion of soft-deletion exists anywhere in the stack.

## Motivation

Today, `doc_id = md5(content + file_path)`. A user can upload the same file twice and have it silently accepted as long as the content shifts by a single byte. The deployment needs the opposite guarantee: each filename should exist at most once (per workspace).

The soft-delete machinery was added to preserve deletion history, but that audit trail is no longer needed. Keeping the fields means every query in every storage backend carries an `is_deleted` filter, and the schemas drift out of sync across backends (postgres has a column, mongo uses `{"$or": [...]}` patterns, json/redis check a boolean on each record). Removing the fields eliminates this complexity.

## Scope

All of the following are in scope:

- `lightrag/lightrag.py` — doc_id hashing + soft-delete call sites
- `lightrag/base.py` — `DocProcessingStatus` dataclass + `DocStatusStorage` abstract methods
- All 5 doc-status storage implementations: `kg/opensearch_impl.py`, `kg/postgres_impl.py`, `kg/mongo_impl.py`, `kg/redis_impl.py`, `kg/json_doc_status_impl.py`
- `lightrag/api/routers/document_routes.py` — response models and endpoint wiring
- `lightrag_webui/src/api/lightrag.ts` — TypeScript types
- `lightrag_webui/src/features/DocumentManager.tsx` — UI filter and display

## Behavior Changes

### Doc ID generation

In `lightrag.py:1419`:

```python
# Before
doc_id = compute_mdhash_id(cleaned_content + path, prefix="doc-")

# After
doc_id = compute_mdhash_id(path, prefix="doc-")
```

### Missing `file_path` fallback

`lightrag.py:1397` currently falls back to the literal string `"unknown_source"` when no path is provided. Under path-based hashing, every such upload would collide. Change the fallback to a unique placeholder per upload:

```python
# Before
file_paths = [path if path else "unknown_source" for path in file_paths]

# After
file_paths = [path if path else f"unknown_source_{uuid.uuid4()}" for path in file_paths]
```

### Duplicate detection

The existing duplicate-handling block at `lightrag.py:1450-1495` continues to apply; its behavior just re-interprets. When `filter_keys()` returns a doc_id as already-present, a `dup-*` FAILED record is created with an updated error message:

```python
"error_msg": f"File already exists. Original doc_id: {doc_id}, Status: {existing_status}",
```

Bulk-insert semantics are unchanged: some documents succeed, duplicates produce tracking records, the overall API response reports `partial_success` / `duplicated`.

### Delete becomes hard-delete

At `lightrag.py:3471-3478` and `lightrag.py:4064-4076`, replace:

```python
soft_delete_data = {
    doc_id: {
        **doc_status_data,
        "is_deleted": True,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
    }
}
await self.doc_status.upsert(soft_delete_data)
await self.full_docs.delete([doc_id])
```

with:

```python
await self.doc_status.delete([doc_id])
await self.full_docs.delete([doc_id])
```

The surrounding variables (`soft_delete_data`, the `**doc_status_data` spread) are no longer needed.

## Transitional Behavior (Accepted Divergence)

Pre-upgrade documents carry old IDs of the form `md5(content + file_path)`. After the upgrade, new uploads compute `md5(file_path)`. **These IDs do not match for the same file.**

Consequence: a file uploaded before the upgrade, then re-uploaded after the upgrade, is *not* caught by the collision check. The first post-upgrade upload of that file is treated as new; every subsequent re-upload (under the new scheme) is then caught.

This is accepted. No data migration re-hashes existing `doc_id`s.

## Data Model Changes (`base.py`)

`DocProcessingStatus` dataclass at line 820 — remove:

```python
is_deleted: bool = False
deleted_at: str | None = None
```

`DocStatusStorage` abstract class:

- Remove `is_deleted: bool = False` parameter from `get_docs_paginated()`.
- Remove `get_deleted_count()` abstract method at line 918.

## Storage Backend Changes

All 5 doc-status backends share the same set of edits:

1. Drop `is_deleted` / `deleted_at` from index mappings / table schemas / default dicts.
2. Remove every `must_not: {term: {is_deleted: True}}`, `is_deleted IS NULL OR is_deleted = FALSE`, and `"$or": [{"is_deleted": {"$exists": False}}, {"is_deleted": False}]` filter from read queries and aggregations.
3. Delete the `get_deleted_count()` implementation.
4. Remove the `is_deleted` parameter from `get_docs_paginated()` and strip the conditional filter body it guards.

### Specific call sites

- `opensearch_impl.py`: mapping at lines 705-706; filters at lines 824, 912, 1033, 1061; `get_deleted_count` at line 1053; `get_docs_paginated` at line 930.
- `postgres_impl.py`: `_add_is_deleted_columns()` at line 1276; filters at lines 4150, 4231; `get_docs_paginated` at line 4350; `get_deleted_count` at line 4573.
- `mongo_impl.py`: filters at lines 453, 482, 698-701, 760; `get_docs_paginated` at line 667; `get_deleted_count` at line 777.
- `redis_impl.py`: skip conditions at lines 715, 769, 1090; `get_docs_paginated` at line 936 (filter at 990); `get_deleted_count` at line 1070.
- `json_doc_status_impl.py`: skip conditions at lines 101, 128, 370; `get_docs_paginated` at line 253 (filter at 288); `get_deleted_count` at line 363.

## Migration (Startup, Idempotent)

Each backend performs a one-shot migration on first boot after the upgrade. All migrations are idempotent — running again is a no-op.

- **postgres_impl.py**: replace the existing `_add_is_deleted_columns()` method (called during init) with a `_drop_is_deleted_columns()` method that runs:
  ```sql
  DELETE FROM LIGHTRAG_DOC_STATUS WHERE is_deleted = TRUE;
  ALTER TABLE LIGHTRAG_DOC_STATUS DROP COLUMN IF EXISTS is_deleted;
  ALTER TABLE LIGHTRAG_DOC_STATUS DROP COLUMN IF EXISTS deleted_at;
  ```
  Guard with a `information_schema.columns` check so subsequent boots short-circuit.

- **mongo_impl.py**: during init,
  ```python
  await self._data.delete_many({"is_deleted": True})
  await self._data.update_many({}, {"$unset": {"is_deleted": "", "deleted_at": ""}})
  ```

- **opensearch_impl.py**: during init (wrapped in a try/except that tolerates a missing index),
  ```python
  await self.client.delete_by_query(index=self._index_name,
      body={"query": {"term": {"is_deleted": True}}}, refresh=True)
  await self.client.update_by_query(index=self._index_name,
      body={"script": {"source": "ctx._source.remove('is_deleted'); ctx._source.remove('deleted_at')"}},
      refresh=True)
  ```

- **redis_impl.py**: during init, scan `doc_status:*` keys. For each, load JSON; if `is_deleted=True`, delete the key; otherwise pop both fields and write back.

- **json_doc_status_impl.py**: on load, drop entries where `is_deleted=True`, strip the two fields from every remaining entry, and write the file back.

## API Changes (`document_routes.py`)

- `DocStatusResponse` (line 474): remove `is_deleted` and `deleted_at` fields, strip them from the example payload.
- `DocStatusUpdateRequest` (line 662): remove the `is_deleted` field.
- Endpoint that currently aggregates `get_deleted_count` (line 3274): remove the call and the `deleted_count` field from its response body.
- `is_deleted=request.is_deleted` (line 3262) and `is_deleted=doc.is_deleted` (line 3314): delete these passthroughs.
- Duplicate-response error message: align with the new filename-based semantics (wording flows from the message changed in `lightrag.py:1482`).

## WebUI Changes

`lightrag_webui/src/api/lightrag.ts`:

- Remove `is_deleted: boolean` and `deleted_at?: string | null` from `DocStatusResponse` (lines 201-202).
- Remove `is_deleted?: boolean` from whichever request type it appears on (line 223).

`lightrag_webui/src/features/DocumentManager.tsx`:

- Remove the `'deleted'` option from `statusFilter` (the type union, the dropdown `<Select>` options, and any label constant).
- Remove the branch at line 619 that passes `is_deleted: query.statusFilter === 'deleted'` to the backend.
- Remove the `deleted_at` display cell at lines 1569-1570.

## Testing

- **Unit/integration tests that reference `is_deleted`**: all tests that currently set up soft-deleted fixtures (`tests/test_opensearch_storage.py`, postgres/mongo integration tests) need updating — assertions against `get_deleted_count`, paginated queries with `is_deleted=True`, and direct reads of the fields must be removed or reshaped.
- **New test coverage**: add a test per backend that inserts the same file twice, asserts the second attempt produces a `dup-*` FAILED record (not a second active doc), and asserts that inserting the same content under a different path succeeds.
- **Migration tests**: for each backend, set up fixtures with legacy `is_deleted=True` rows, run init, assert the rows are gone and the fields are absent.
- **WebUI lint + typecheck** (`bun run lint`) must pass with the type changes.

## Rollout

1. Code change + migration ships together. No feature flag.
2. On first boot after the upgrade, each backend runs its idempotent migration automatically. **Ordering within init matters:** the migration (which purges `is_deleted=True` rows) must complete before the first read query runs. Otherwise read queries — whose `must_not`/`WHERE` filters are now removed — would surface soft-deleted rows as active until the migration catches up. All backends already perform init synchronously before accepting traffic, so honoring this just means running the migration step inside the existing `initialize()` / `_create_index_if_not_exists()` flow rather than on first query.
3. Users who re-upload a pre-upgrade file once will "register" its new doc_id; subsequent uploads are then caught.

## Out of Scope

- **Re-hashing existing doc_ids to the new scheme.** Considered and rejected in favor of the accepted divergence described above (simpler, no cross-table ID rewrites). Rejected because the migration would have to rewrite `doc_id` primary keys in `doc_status`, `full_docs`, chunk back-references, vector-store metadata, and graph `source_id`s — high risk, moderate value.
- **Adding a `file_path`-field lookup as a second collision check** (i.e., query `doc_status` by `file_path` in addition to by `doc_id`) to catch pre-upgrade files on first re-upload. Rejected to keep the insert path simple; the transitional gap self-heals after one re-upload.
- Audit trail / deletion history in any alternate form.
- Cross-workspace filename uniqueness (remains workspace-scoped, which is the current behavior).
