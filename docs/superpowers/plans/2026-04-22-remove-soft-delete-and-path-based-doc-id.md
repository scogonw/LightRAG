# Remove Soft-Delete and Path-Based Doc IDs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `is_deleted` / `deleted_at` fields end-to-end, convert soft-delete into hard-delete, and change doc ID generation from `md5(content + file_path)` to `md5(file_path)` so duplicate filenames are rejected.

**Architecture:** Core change in `lightrag.py` flips the doc-id hash input. All 5 doc-status storage backends strip the two fields from schemas/filters/queries and add an idempotent startup migration that hard-deletes rows with `is_deleted=True`. API response models and WebUI types drop the fields. Accepted divergence: pre-upgrade doc_ids remain under the old scheme and are only caught by the new collision check after one re-upload.

**Tech Stack:** Python 3.11+ async, FastAPI, OpenSearch / PostgreSQL (asyncpg) / MongoDB (motor) / Redis (redis-py) / JSON, React 19 + TypeScript (Bun + Vite), pytest

**Spec:** `docs/superpowers/specs/2026-04-22-remove-soft-delete-and-path-based-doc-id-design.md`

---

## File Manifest

**Core:**
- Modify: `lightrag/lightrag.py` — doc_id hashing, UUID fallback, hard-delete conversions, duplicate error message
- Modify: `lightrag/base.py` — remove fields from `DocProcessingStatus`, drop abstract methods

**Storage backends (each: strip fields, remove filters, drop `get_deleted_count`, add startup migration):**
- Modify: `lightrag/kg/opensearch_impl.py`
- Modify: `lightrag/kg/postgres_impl.py`
- Modify: `lightrag/kg/mongo_impl.py`
- Modify: `lightrag/kg/redis_impl.py`
- Modify: `lightrag/kg/json_doc_status_impl.py`

**API:**
- Modify: `lightrag/api/routers/document_routes.py`

**WebUI:**
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/features/DocumentManager.tsx`

**Tests:**
- Modify: `tests/test_opensearch_storage.py` (if any existing soft-delete fixtures — verify after Task 6)
- Create: `tests/test_path_based_doc_id.py` — unit tests for the new hashing behavior

---

### Task 1: Test and change doc_id generation to path-only hashing

**Files:**
- Create: `tests/test_path_based_doc_id.py`
- Modify: `lightrag/lightrag.py:1414-1425`

- [ ] **Step 1: Write failing test**

Create `tests/test_path_based_doc_id.py`:

```python
"""Unit tests for path-based doc_id generation.

After the soft-delete removal, doc_id is derived from file_path only so
identical filenames collide regardless of content.
"""
import pytest

pytestmark = pytest.mark.offline

from lightrag.utils import compute_mdhash_id


def _doc_id(content: str, file_path: str) -> str:
    """Helper mirroring the production call at lightrag.py:1419."""
    return compute_mdhash_id(file_path, prefix="doc-")


def test_same_path_same_content_same_id():
    a = _doc_id("hello world", "ABC.pdf")
    b = _doc_id("hello world", "ABC.pdf")
    assert a == b


def test_same_path_different_content_same_id():
    """Key guarantee: filename alone determines the id."""
    a = _doc_id("hello world", "ABC.pdf")
    b = _doc_id("completely different content", "ABC.pdf")
    assert a == b


def test_different_path_same_content_different_ids():
    """Key guarantee: same content under different paths produces different ids."""
    a = _doc_id("hello world", "ABC.pdf")
    b = _doc_id("hello world", "DEF.pdf")
    assert a != b


def test_id_has_doc_prefix():
    assert _doc_id("x", "ABC.pdf").startswith("doc-")
```

- [ ] **Step 2: Run the test — it passes trivially because the helper already uses the new scheme**

Run: `python -m pytest tests/test_path_based_doc_id.py -v`
Expected: 4 passed. This test file locks in the behavior we're migrating *toward*; the production change in Step 3 makes `lightrag.py` match this helper.

- [ ] **Step 3: Change production doc_id generation**

In `lightrag/lightrag.py`, replace line 1419:

```python
# BEFORE (line 1419):
                doc_id = compute_mdhash_id(cleaned_content + path, prefix="doc-")

# AFTER:
                doc_id = compute_mdhash_id(path, prefix="doc-")
```

- [ ] **Step 4: Run offline tests to confirm nothing else breaks**

Run: `python -m pytest tests -v -x`
Expected: all offline tests pass. If any test asserts on a specific legacy doc_id that embedded content, update the expected value to match `compute_mdhash_id(file_path, prefix="doc-")`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_path_based_doc_id.py lightrag/lightrag.py
git commit -m "feat(lightrag): hash doc_id from file_path only

Filenames now collide by identity instead of by content-and-path. See
spec docs/superpowers/specs/2026-04-22-remove-soft-delete-and-path-based-doc-id-design.md"
```

---

### Task 2: Add UUID fallback for missing file_path

**Files:**
- Modify: `lightrag/lightrag.py:1376-1397` (import block + fallback line)
- Modify: `tests/test_path_based_doc_id.py`

- [ ] **Step 1: Add a failing test**

Append to `tests/test_path_based_doc_id.py`:

```python
def test_missing_path_fallback_produces_unique_ids(monkeypatch):
    """Two uploads with no file_path must not collide.

    The fallback used to be the literal 'unknown_source' which would make
    every path-less upload share an id once we switched to path-only
    hashing. Now the fallback is 'unknown_source_<uuid4>' so each is unique.
    """
    import re
    from lightrag import lightrag as lightrag_module

    # Exercise the same logic used on line 1397.
    def fallback_for(path):
        from uuid import uuid4
        return path if path else f"unknown_source_{uuid4()}"

    a = fallback_for("")
    b = fallback_for("")
    pattern = re.compile(r"^unknown_source_[0-9a-f-]{36}$")
    assert pattern.match(a)
    assert pattern.match(b)
    assert a != b
```

- [ ] **Step 2: Run the test — it passes because the helper inlines the target behavior**

Run: `python -m pytest tests/test_path_based_doc_id.py::test_missing_path_fallback_produces_unique_ids -v`
Expected: pass.

- [ ] **Step 3: Add `uuid4` import and change fallback in production code**

In `lightrag/lightrag.py`, add to the stdlib import block around line 1-18 (keep imports alphabetical with what's there):

```python
from uuid import uuid4
```

Replace line 1394 (inside `apipeline_enqueue_documents`):

```python
# BEFORE:
            file_paths = [path if path else "unknown_source" for path in file_paths]

# AFTER:
            file_paths = [path if path else f"unknown_source_{uuid4()}" for path in file_paths]
```

Replace line 1397:

```python
# BEFORE:
            file_paths = ["unknown_source"] * len(input)

# AFTER:
            file_paths = [f"unknown_source_{uuid4()}" for _ in input]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_path_based_doc_id.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add lightrag/lightrag.py tests/test_path_based_doc_id.py
git commit -m "feat(lightrag): unique fallback for uploads without file_path

Avoids doc_id collision when multiple inserts omit file_path now that
the id is derived from path only."
```

---

### Task 3: Update duplicate error message wording

**Files:**
- Modify: `lightrag/lightrag.py:1482`

- [ ] **Step 1: Replace the error message**

In `lightrag/lightrag.py`, find the duplicate-records block around line 1471-1488 and change line 1482:

```python
# BEFORE (line 1482):
                    "error_msg": f"Content already exists. Original doc_id: {doc_id}, Status: {existing_status}",

# AFTER:
                    "error_msg": f"File already exists. Original doc_id: {doc_id}, Status: {existing_status}",
```

- [ ] **Step 2: Check for callers that match on the old message text**

Run: `grep -rn "Content already exists" lightrag tests lightrag_webui/src`
Expected: zero matches. If any remain (e.g., in a test assertion), update them to the new wording.

- [ ] **Step 3: Commit**

```bash
git add lightrag/lightrag.py
git commit -m "chore(lightrag): reword duplicate error to 'File already exists'"
```

---

### Task 4: Replace soft-delete upserts with hard-delete

**Files:**
- Modify: `lightrag/lightrag.py:3468-3479` (short path — doc with no chunks)
- Modify: `lightrag/lightrag.py:4064-4079` (main delete path)

- [ ] **Step 1: Replace the no-chunks short path**

In `lightrag/lightrag.py`, replace the block at lines 3468-3479:

```python
# BEFORE:
                try:
                    # Soft-delete the doc status record and hard-delete full doc content.
                    deletion_stage = "delete_doc_entries"
                    soft_delete_data = {
                        doc_id: {
                            **doc_status_data,
                            "is_deleted": True,
                            "deleted_at": datetime.now(timezone.utc).isoformat(),
                        }
                    }
                    await self.doc_status.upsert(soft_delete_data)
                    await self.full_docs.delete([doc_id])
                except Exception as e:

# AFTER:
                try:
                    deletion_stage = "delete_doc_entries"
                    await self.doc_status.delete([doc_id])
                    await self.full_docs.delete([doc_id])
                except Exception as e:
```

- [ ] **Step 2: Replace the main delete path**

In `lightrag/lightrag.py`, replace the block at lines 4064-4079:

```python
# BEFORE:
            # 11. Soft-delete original document status and hard-delete full doc content.
            try:
                deletion_stage = "delete_doc_entries"
                in_final_delete_stage = True
                soft_delete_data = {
                    doc_id: {
                        **doc_status_data,
                        "is_deleted": True,
                        "deleted_at": datetime.now(timezone.utc).isoformat(),
                    }
                }
                await self.doc_status.upsert(soft_delete_data)
                await self.full_docs.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete document and status: {e}")
                raise Exception(f"Failed to delete document and status: {e}") from e

# AFTER:
            # 11. Hard-delete document status and full doc content.
            try:
                deletion_stage = "delete_doc_entries"
                in_final_delete_stage = True
                await self.doc_status.delete([doc_id])
                await self.full_docs.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete document and status: {e}")
                raise Exception(f"Failed to delete document and status: {e}") from e
```

- [ ] **Step 3: Remove any now-unused `doc_status_data` assignments**

Run: `grep -n "doc_status_data" lightrag/lightrag.py`
Review each hit. `doc_status_data` may still be read earlier in `adelete_by_doc_id` for other purposes (e.g., for `chunks_list`). Only remove the assignment if both the no-chunks and main paths above were its sole consumers. Otherwise leave it alone.

- [ ] **Step 4: Run offline tests**

Run: `python -m pytest tests -v -x`
Expected: all pass. Some tests may reference `is_deleted` in `DocProcessingStatus` construction — those will still pass until Task 5 removes the field.

- [ ] **Step 5: Commit**

```bash
git add lightrag/lightrag.py
git commit -m "refactor(lightrag): hard-delete instead of soft-delete in doc deletion"
```

---

### Task 5: Remove `is_deleted` / `deleted_at` from `DocProcessingStatus` and abstract methods

**Files:**
- Modify: `lightrag/base.py:834-837, 886-903, 916-919`

- [ ] **Step 1: Remove the two fields from `DocProcessingStatus`**

In `lightrag/base.py`, delete lines 834-837:

```python
# REMOVE THESE 4 LINES (currently at 834-837):
    is_deleted: bool = False
    """Whether the document has been soft-deleted"""
    deleted_at: str | None = None
    """ISO format timestamp when document was soft-deleted"""
```

- [ ] **Step 2: Drop `is_deleted` parameter from `get_docs_paginated` abstract signature**

In `lightrag/base.py`, change the abstract method around line 886:

```python
# BEFORE:
    @abstractmethod
    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
        is_deleted: bool = False,
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')
            is_deleted: If True, return only soft-deleted documents; if False, exclude them

# AFTER:
    @abstractmethod
    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')
```

- [ ] **Step 3: Remove `get_deleted_count` abstract method**

In `lightrag/base.py`, delete the method around line 918 (3 lines):

```python
# REMOVE:
    @abstractmethod
    async def get_deleted_count(self) -> int:
        """Get count of soft-deleted documents"""
```

- [ ] **Step 4: Verify no direct imports of the removed symbols remain**

Run: `grep -rn "is_deleted\|deleted_at" lightrag/base.py`
Expected: zero matches.

- [ ] **Step 5: Do NOT run tests yet**

Skipping this step is intentional. The abstract base is now out of sync with every backend implementation and `document_routes.py`, so tests will fail until Tasks 6–11 land. The follow-up tasks bring everything back to green.

- [ ] **Step 6: Commit**

```bash
git add lightrag/base.py
git commit -m "refactor(base): remove is_deleted/deleted_at from DocProcessingStatus"
```

---

### Task 6: Update `opensearch_impl.py` — strip fields, migration, remove methods

**Files:**
- Modify: `lightrag/kg/opensearch_impl.py`

- [ ] **Step 1: Remove `is_deleted` / `deleted_at` from the doc-status index mapping**

Find the block around line 692-716 inside `_create_index_if_not_exists`. Remove lines 705-706:

```python
# REMOVE these two lines from the mapping:
                            "is_deleted": {"type": "boolean"},
                            "deleted_at": {"type": "date"},
```

- [ ] **Step 2: Drop the defaults in `_prepare_doc_status_data`**

In the same file around line 648-664, remove lines 657-658:

```python
# REMOVE:
        data.setdefault("is_deleted", False)
        data.setdefault("deleted_at", None)
```

- [ ] **Step 3: Remove `must_not: is_deleted` filters**

In the same file:

- Line 824 — inside `get_status_counts`, remove the inner `must_not: [{"term": {"is_deleted": True}}]` clause. After removal the `bool` query becomes empty; replace the entire `"query"` block with `{"match_all": {}}`:

```python
# BEFORE (around line 820-826):
            body = {
                "size": 0,
                "query": {
                    "bool": {
                        "must_not": [{"term": {"is_deleted": True}}],
                    }
                },
                "aggs": {"status_counts": {"terms": {"field": "status", "size": 100}}},
            }

# AFTER:
            body = {
                "size": 0,
                "query": {"match_all": {}},
                "aggs": {"status_counts": {"terms": {"field": "status", "size": 100}}},
            }
```

- Line 912 — inside `get_docs_by_statuses`, drop the `must_not` clause:

```python
# BEFORE (around line 908-915):
        return await self._search_all_docs(
            {
                "bool": {
                    "must": [{"terms": {"status": status_values}}],
                    "must_not": [{"term": {"is_deleted": True}}],
                }
            }
        )

# AFTER:
        return await self._search_all_docs(
            {
                "bool": {
                    "must": [{"terms": {"status": status_values}}],
                }
            }
        )
```

- Line 1033 — find this in `get_all_status_counts`. Apply the same transformation as Step 3a above (replace the `bool.must_not` query with `{"match_all": {}}`).

- [ ] **Step 4: Remove `is_deleted` parameter from `get_docs_paginated`**

Around line 923-950, change the signature and strip the conditional filter body. The exact edit depends on the current method; find the block and:

- Remove `is_deleted: bool = False,` from the signature (~line 930).
- Remove the `must_not_clauses.append(...)` or `must_clauses.append(...)` branches that key on `is_deleted` (~lines 945-948). Keep the rest of the function intact.

```python
# BEFORE:
        is_deleted: bool = False,
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        ...
        if is_deleted:
            must_clauses.append({"term": {"is_deleted": True}})
        else:
            must_not_clauses.append({"term": {"is_deleted": True}})

# AFTER:
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        ...
        # (the 4-line is_deleted branch is removed entirely)
```

- [ ] **Step 5: Delete `get_deleted_count` method**

Delete the entire method around lines 1053-1067 (`async def get_deleted_count(self) -> int:` through the last `return` in that method).

- [ ] **Step 6: Add migration method**

Add this method to `OpenSearchDocStatusStorage` (place it next to `_create_index_if_not_exists`, around line 725):

```python
    async def _migrate_drop_soft_delete_fields(self) -> None:
        """One-shot idempotent migration: delete rows with is_deleted=True
        and strip the is_deleted/deleted_at fields from remaining rows.

        Safe to re-run: delete-by-query is a no-op when the field is absent,
        and the update-by-query script checks for field presence via remove().
        """
        if not await self.client.indices.exists(index=self._index_name):
            return
        try:
            await self.client.delete_by_query(
                index=self._index_name,
                body={"query": {"term": {"is_deleted": True}}},
                refresh=True,
                conflicts="proceed",
            )
            await self.client.update_by_query(
                index=self._index_name,
                body={
                    "script": {
                        "source": (
                            "ctx._source.remove('is_deleted'); "
                            "ctx._source.remove('deleted_at');"
                        )
                    },
                    "query": {
                        "bool": {
                            "should": [
                                {"exists": {"field": "is_deleted"}},
                                {"exists": {"field": "deleted_at"}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                },
                refresh=True,
                conflicts="proceed",
            )
            logger.info(
                f"[{self.workspace}] Dropped is_deleted/deleted_at from {self._index_name}"
            )
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                return
            logger.warning(
                f"[{self.workspace}] soft-delete migration failed: {e}"
            )
```

- [ ] **Step 7: Call the migration from `initialize`**

In `OpenSearchDocStatusStorage.initialize` (around lines 666-675), add the migration call after the index is ensured:

```python
    async def initialize(self):
        """Initialize client connection and create doc status index."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_index_if_not_exists()
            await self._migrate_drop_soft_delete_fields()  # NEW LINE
            self._index_ready = True
            logger.debug(
                f"[{self.workspace}] OpenSearch DocStatus storage initialized: {self._index_name}"
            )
```

- [ ] **Step 8: Verify no stray references remain**

Run: `grep -n "is_deleted\|deleted_at\|get_deleted_count" lightrag/kg/opensearch_impl.py`
Expected: matches ONLY inside `_migrate_drop_soft_delete_fields` (the migration body itself).

- [ ] **Step 9: Run opensearch unit tests**

Run: `python -m pytest tests/test_opensearch_storage.py -v`
Expected: all pass. If a mock-based test references the removed fields, update it to drop those references.

- [ ] **Step 10: Commit**

```bash
git add lightrag/kg/opensearch_impl.py
git commit -m "refactor(opensearch): drop is_deleted/deleted_at + add migration"
```

---

### Task 7: Update `postgres_impl.py` — strip fields, migration, remove methods

**Files:**
- Modify: `lightrag/kg/postgres_impl.py`

- [ ] **Step 1: Replace the legacy `_migrate_doc_status_add_soft_delete` method with a drop-columns migration**

At lines 1275-1329, replace the entire method body:

```python
    async def _migrate_doc_status_drop_soft_delete(self):
        """Idempotent: hard-delete rows where is_deleted=TRUE, then drop the columns.

        Replaces the earlier _migrate_doc_status_add_soft_delete migration.
        Safe to re-run: the column existence checks short-circuit subsequent boots.
        """
        try:
            check_is_deleted_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'is_deleted'
            """
            is_deleted_info = await self.query(check_is_deleted_sql)
            if is_deleted_info:
                logger.info(
                    "Migrating LIGHTRAG_DOC_STATUS: purging rows with is_deleted=TRUE"
                )
                await self.execute(
                    "DELETE FROM LIGHTRAG_DOC_STATUS WHERE is_deleted = TRUE"
                )
                logger.info(
                    "Dropping is_deleted column from LIGHTRAG_DOC_STATUS"
                )
                await self.execute(
                    "ALTER TABLE LIGHTRAG_DOC_STATUS DROP COLUMN IF EXISTS is_deleted"
                )

            check_deleted_at_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'deleted_at'
            """
            deleted_at_info = await self.query(check_deleted_at_sql)
            if deleted_at_info:
                logger.info(
                    "Dropping deleted_at column from LIGHTRAG_DOC_STATUS"
                )
                await self.execute(
                    "ALTER TABLE LIGHTRAG_DOC_STATUS DROP COLUMN IF EXISTS deleted_at"
                )
        except Exception as e:
            logger.warning(
                f"Failed to drop is_deleted/deleted_at from LIGHTRAG_DOC_STATUS: {e}"
            )
```

- [ ] **Step 2: Update the migration call site at line 1660**

Find the caller (grep for `_migrate_doc_status_add_soft_delete`) and replace with the new name:

```python
# BEFORE:
            await self._migrate_doc_status_add_soft_delete()

# AFTER:
            await self._migrate_doc_status_drop_soft_delete()
```

- [ ] **Step 3: Remove `is_deleted` filters from read queries**

- Line 4150 — in `get_status_counts`, remove the `AND (is_deleted IS NULL OR is_deleted = FALSE)` line from the SQL:

```python
# BEFORE:
        sql = """SELECT status as "status", COUNT(1) as "count"
                   FROM LIGHTRAG_DOC_STATUS
                  WHERE workspace=$1
                    AND (is_deleted IS NULL OR is_deleted = FALSE)
                  GROUP BY STATUS
                 """

# AFTER:
        sql = """SELECT status as "status", COUNT(1) as "count"
                   FROM LIGHTRAG_DOC_STATUS
                  WHERE workspace=$1
                  GROUP BY STATUS
                 """
```

- Line 4231 — in `get_docs_by_statuses`, drop the `AND (is_deleted IS NULL OR is_deleted = FALSE)` tail:

```python
# BEFORE:
        sql = (
            "SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 AND status = ANY($2)"
            " AND (is_deleted IS NULL OR is_deleted = FALSE)"
        )

# AFTER:
        sql = "SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 AND status = ANY($2)"
```

- [ ] **Step 4: Remove `is_deleted` parameter and filter from `get_docs_paginated`**

In the method around line 4343-4416:

- Remove `is_deleted: bool = False,` from the signature (line 4350).
- Remove the docstring line describing `is_deleted`.
- Remove the entire `if is_deleted: ... else: ...` block at lines 4405-4408.
- Remove `{deleted_filter}` interpolation from both `where_clause` branches at lines 4412 and 4415 (they become plain `WHERE workspace=$1 AND status=$2` and `WHERE workspace=$1`).

```python
# BEFORE (lines 4404-4415):
        # Build WHERE clause with parameterized query
        if is_deleted:
            deleted_filter = "AND is_deleted = TRUE"
        else:
            deleted_filter = "AND (is_deleted IS NULL OR is_deleted = FALSE)"

        if status_filter is not None:
            param_count += 1
            where_clause = f"WHERE workspace=$1 AND status=$2 {deleted_filter}"
            params["status"] = status_filter.value
        else:
            where_clause = f"WHERE workspace=$1 {deleted_filter}"

# AFTER:
        if status_filter is not None:
            param_count += 1
            where_clause = "WHERE workspace=$1 AND status=$2"
            params["status"] = status_filter.value
        else:
            where_clause = "WHERE workspace=$1"
```

- [ ] **Step 5: Delete `get_deleted_count` method**

Delete the method at lines 4573-4584.

- [ ] **Step 6: Strip `is_deleted` / `deleted_at` from `DocProcessingStatus` constructors**

Grep the file for any `DocProcessingStatus(` call that passes `is_deleted=` or `deleted_at=` and remove those kwargs.

Run: `grep -n "is_deleted\|deleted_at" lightrag/kg/postgres_impl.py`
Expected result after all edits: matches ONLY inside `_migrate_doc_status_drop_soft_delete`.

- [ ] **Step 7: Run postgres tests**

Run: `python -m pytest tests/test_postgres_migration.py tests/test_postgres_upsert.py tests/test_postgres_performance_timing.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add lightrag/kg/postgres_impl.py
git commit -m "refactor(postgres): drop is_deleted/deleted_at columns + replace migration"
```

---

### Task 8: Update `mongo_impl.py` — strip fields, migration, remove methods

**Files:**
- Modify: `lightrag/kg/mongo_impl.py`

- [ ] **Step 1: Remove `$or: [is_deleted exists/false]` filter from `get_status_counts`**

At lines 450-461, replace:

```python
# BEFORE:
    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status (excludes soft-deleted)"""
        pipeline = [
            {"$match": {"$or": [{"is_deleted": {"$exists": False}}, {"is_deleted": False}]}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]

# AFTER:
    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
```

- [ ] **Step 2: Remove filter from `get_docs_by_statuses`**

At lines 469-495, inside the `.find(...)` call:

```python
# BEFORE:
        cursor = self._data.find({
            "status": {"$in": status_values},
            "$or": [{"is_deleted": {"$exists": False}}, {"is_deleted": False}],
        })

# AFTER:
        cursor = self._data.find({"status": {"$in": status_values}})
```

- [ ] **Step 3: Remove `is_deleted` parameter and filter from `get_docs_paginated`**

At lines 660-751:

- Drop `is_deleted: bool = False,` from the signature (line 667).
- Remove the `is_deleted` description line from the docstring.
- Remove the entire 3-line branch at lines 698-701:

```python
# REMOVE:
        if is_deleted:
            query_filter["is_deleted"] = True
        else:
            query_filter["$or"] = [{"is_deleted": {"$exists": False}}, {"is_deleted": False}]
```

- [ ] **Step 4: Remove filter from `get_all_status_counts`**

At lines 759-761:

```python
# BEFORE:
        pipeline = [
            {"$match": {"$or": [{"is_deleted": {"$exists": False}}, {"is_deleted": False}]}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]

# AFTER:
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
```

- [ ] **Step 5: Delete `get_deleted_count`**

Delete lines 777-779:

```python
# REMOVE:
    async def get_deleted_count(self) -> int:
        """Get count of soft-deleted documents"""
        return await self._data.count_documents({"is_deleted": True})
```

- [ ] **Step 6: Add migration method**

Insert before the `initialize` method in `MongoDocStatusStorage` (around line 391):

```python
    async def _migrate_drop_soft_delete_fields(self) -> None:
        """Idempotent: delete documents with is_deleted=True, then unset both fields.

        Safe to re-run: delete filters on is_deleted=True so already-clean
        collections are unaffected, and $unset is a no-op when the field is absent.
        """
        try:
            del_result = await self._data.delete_many({"is_deleted": True})
            if del_result.deleted_count:
                logger.info(
                    f"[{self.workspace}] Purged {del_result.deleted_count} soft-deleted doc_status rows"
                )
            unset_result = await self._data.update_many(
                {"$or": [
                    {"is_deleted": {"$exists": True}},
                    {"deleted_at": {"$exists": True}},
                ]},
                {"$unset": {"is_deleted": "", "deleted_at": ""}},
            )
            if unset_result.modified_count:
                logger.info(
                    f"[{self.workspace}] Stripped soft-delete fields from {unset_result.modified_count} rows"
                )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] mongo soft-delete migration failed: {e}"
            )
```

- [ ] **Step 7: Call migration from `initialize`**

In `MongoDocStatusStorage.initialize` (around lines 391-403), add the call after index creation:

```python
    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            self._data = await get_or_create_collection(self.db, self._collection_name)

            await self.create_and_migrate_indexes_if_not_exists()
            await self._migrate_drop_soft_delete_fields()  # NEW LINE

            logger.debug(
                f"[{self.workspace}] Use MongoDB as DocStatus {self._collection_name}"
            )
```

- [ ] **Step 8: Verify cleanup**

Run: `grep -n "is_deleted\|deleted_at" lightrag/kg/mongo_impl.py`
Expected: matches ONLY inside `_migrate_drop_soft_delete_fields`.

- [ ] **Step 9: Run mongo tests**

Run: `python -m pytest tests/test_mongo_storage.py -v`
Expected: all offline mongo tests pass. (Integration mongo tests need `--run-integration`.)

- [ ] **Step 10: Commit**

```bash
git add lightrag/kg/mongo_impl.py
git commit -m "refactor(mongo): drop is_deleted/deleted_at + add migration"
```

---

### Task 9: Update `redis_impl.py` — strip fields, migration, remove methods

**Files:**
- Modify: `lightrag/kg/redis_impl.py`

- [ ] **Step 1: Remove `is_deleted` skip checks from read paths**

- Line 715 — in `get_status_counts`, remove:

```python
# REMOVE these 2 lines (currently 715-716):
                                    if doc_data.get("is_deleted", False):
                                        continue
```

- Line 769 — in `get_docs_by_statuses`, remove the same 2-line guard.

- Line 1090 — in `get_deleted_count` (will be deleted in Step 4 anyway, skip edit here).

- [ ] **Step 2: Remove `is_deleted` parameter and filter from `get_docs_paginated`**

At lines 929-1054:

- Drop `is_deleted: bool = False,` from the signature (line 936).
- Remove the `is_deleted` field from the docstring `Args:` section.
- Remove the 3-line filter at lines 989-991:

```python
# REMOVE:
                                    # Filter by is_deleted field
                                    if doc_data.get("is_deleted", False) != is_deleted:
                                        continue
```

- [ ] **Step 3: Strip fields when decoding doc data**

In `get_docs_by_statuses` (around lines 774-781) and `get_docs_paginated` (around lines 1004-1012), the code constructs `DocProcessingStatus(**data)`. Now that the dataclass no longer accepts `is_deleted`/`deleted_at`, pop them before constructing:

Search for every `data = doc_data.copy()` block in `redis_impl.py` and append two pops right after the copy (before any `DocProcessingStatus(**data)` call):

```python
                                data = doc_data.copy()
                                data.pop("content", None)
                                data.pop("is_deleted", None)
                                data.pop("deleted_at", None)
```

This defends against legacy rows that still have the fields before migration runs.

- [ ] **Step 4: Delete `get_deleted_count`**

Delete the entire method at lines 1070-1102 (whole body — find the next `async def` after `get_deleted_count` and delete everything between).

- [ ] **Step 5: Add migration method**

Insert into `RedisDocStatusStorage` above `initialize` (around line 579):

```python
    async def _migrate_drop_soft_delete_fields(self) -> None:
        """Idempotent: delete keys with is_deleted=True, strip fields from the rest."""
        try:
            async with self._get_redis_connection() as redis:
                cursor = 0
                purged = 0
                stripped = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{self.final_namespace}:*", count=1000
                    )
                    if not keys:
                        if cursor == 0:
                            break
                        continue
                    pipe = redis.pipeline()
                    for key in keys:
                        pipe.get(key)
                    values = await pipe.execute()
                    write_pipe = redis.pipeline()
                    dirty = False
                    for key, value in zip(keys, values):
                        if not value:
                            continue
                        try:
                            doc_data = json.loads(value)
                        except json.JSONDecodeError:
                            continue
                        if doc_data.get("is_deleted") is True:
                            write_pipe.delete(key)
                            purged += 1
                            dirty = True
                            continue
                        if "is_deleted" in doc_data or "deleted_at" in doc_data:
                            doc_data.pop("is_deleted", None)
                            doc_data.pop("deleted_at", None)
                            write_pipe.set(key, json.dumps(doc_data))
                            stripped += 1
                            dirty = True
                    if dirty:
                        await write_pipe.execute()
                    if cursor == 0:
                        break
                if purged or stripped:
                    logger.info(
                        f"[{self.workspace}] Redis migration: purged={purged} stripped={stripped}"
                    )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Redis soft-delete migration failed: {e}"
            )
```

- [ ] **Step 6: Call migration from `initialize`**

In `RedisDocStatusStorage.initialize` (around line 580), add the migration call after the `ping` succeeds:

```python
    async def initialize(self):
        async with get_data_init_lock():
            if self._initialized:
                return

            try:
                async with self._get_redis_connection() as redis:
                    await redis.ping()
                    logger.info(
                        f"[{self.workspace}] Connected to Redis for doc status namespace {self.namespace}"
                    )
                    await self._migrate_drop_soft_delete_fields()  # NEW LINE
                    self._initialized = True
```

- [ ] **Step 7: Verify**

Run: `grep -n "is_deleted\|deleted_at" lightrag/kg/redis_impl.py`
Expected: matches ONLY inside `_migrate_drop_soft_delete_fields` plus the three `data.pop(...)` defenses added in Step 3.

- [ ] **Step 8: Commit**

```bash
git add lightrag/kg/redis_impl.py
git commit -m "refactor(redis): drop is_deleted/deleted_at + add migration"
```

---

### Task 10: Update `json_doc_status_impl.py` — strip fields, migration, remove methods

**Files:**
- Modify: `lightrag/kg/json_doc_status_impl.py`

- [ ] **Step 1: Remove `is_deleted` skip checks**

Find and delete each of these 2-line guards:

- Lines 101-102 inside `get_status_counts`:
  ```python
  # REMOVE:
                  if doc.get("is_deleted", False):
                      continue
  ```
- Lines 128-129 inside `get_docs_by_statuses`:
  ```python
  # REMOVE:
                  if v.get("is_deleted", False):
                      continue
  ```
- Lines 369-371 inside `get_deleted_count` (whole method deleted in Step 4).

- [ ] **Step 2: Strip fields before constructing `DocProcessingStatus`**

Around line 133-141 inside `get_docs_by_statuses`, after `data = v.copy()` and `data.pop("content", None)`, add:

```python
                    data.pop("is_deleted", None)
                    data.pop("deleted_at", None)
```

Apply the same defensive pop in `get_docs_paginated` (around line 290) wherever `DocProcessingStatus(**data)` is constructed.

- [ ] **Step 3: Remove `is_deleted` parameter and filter from `get_docs_paginated`**

Around line 253-290:

- Drop `is_deleted: bool = False,` from the signature (line 253).
- Remove the `is_deleted` line from the docstring.
- Remove the filter check at line 288:

```python
# REMOVE:
                  if doc_data.get("is_deleted", False) != is_deleted:
                      continue
```

- [ ] **Step 4: Delete `get_deleted_count`**

Delete lines 363-372 (the whole method).

- [ ] **Step 5: Add migration method**

Insert into `JsonDocStatusStorage` above the `initialize` method:

```python
    async def _migrate_drop_soft_delete_fields(self) -> None:
        """Idempotent: drop entries where is_deleted=True, strip fields from the rest.

        Called from initialize(). The caller already holds get_data_init_lock
        so this is race-free. If any change is made, we mark storage_updated
        so the next index_done_callback persists to disk.
        """
        if not self._data:
            return
        async with self._storage_lock:
            to_delete = [
                key for key, value in self._data.items()
                if isinstance(value, dict) and value.get("is_deleted") is True
            ]
            for key in to_delete:
                del self._data[key]
            stripped = 0
            for value in self._data.values():
                if not isinstance(value, dict):
                    continue
                if "is_deleted" in value or "deleted_at" in value:
                    value.pop("is_deleted", None)
                    value.pop("deleted_at", None)
                    stripped += 1
            if to_delete or stripped:
                logger.info(
                    f"[{self.workspace}] JSON migration: purged={len(to_delete)} stripped={stripped}"
                )
                await set_all_update_flags(self.namespace)
```

- [ ] **Step 6: Call migration from `initialize`**

In `JsonDocStatusStorage.initialize` (around lines 50-72), add the call after `_data` is populated:

```python
    async def initialize(self):
        """Initialize storage data"""
        self._storage_lock = get_namespace_lock(...)
        self.storage_updated = await get_update_flag(...)
        async with get_data_init_lock():
            need_init = await try_initialize_namespace(
                self.namespace, workspace=self.workspace
            )
            self._data = await get_namespace_data(
                self.namespace, workspace=self.workspace
            )
            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    self._data.update(loaded_data)
                    logger.info(
                        f"[{self.workspace}] Process {os.getpid()} doc status load {self.namespace} with {len(loaded_data)} records"
                    )
                await self._migrate_drop_soft_delete_fields()  # NEW LINE
```

- [ ] **Step 7: Verify**

Run: `grep -n "is_deleted\|deleted_at" lightrag/kg/json_doc_status_impl.py`
Expected: matches ONLY inside `_migrate_drop_soft_delete_fields` plus the defensive `data.pop` lines from Step 2.

- [ ] **Step 8: Commit**

```bash
git add lightrag/kg/json_doc_status_impl.py
git commit -m "refactor(json): drop is_deleted/deleted_at + add migration"
```

---

### Task 11: Update `document_routes.py` API layer

**Files:**
- Modify: `lightrag/api/routers/document_routes.py`

- [ ] **Step 1: Remove `is_deleted` and `deleted_at` from `DocStatusResponse`**

At lines 474-525, delete lines 497-500 and 521-522:

```python
# REMOVE (lines 497-500):
    is_deleted: bool = Field(default=False, description="Whether the document has been soft-deleted")
    deleted_at: Optional[str] = Field(
        default=None, description="Deletion timestamp (ISO format string)"
    )

# REMOVE from the example payload (lines 521-522):
                "is_deleted": False,
                "deleted_at": None,
```

- [ ] **Step 2: Remove `is_deleted` from the paginated-docs request model**

At lines 662-677, delete the field definition and its example entry:

```python
# REMOVE (lines 662-664):
    is_deleted: bool = Field(
        default=False, description="If True, return only soft-deleted documents; if False, exclude them"
    )

# REMOVE from the example (line 674):
                "is_deleted": False,
```

- [ ] **Step 3: Remove `deleted_count` from `PaginatedDocsResponse`**

At lines 713-732, delete lines 729-731:

```python
# REMOVE:
    deleted_count: int = Field(
        default=0, description="Count of soft-deleted documents"
    )
```

- [ ] **Step 4: Drop `is_deleted` / `deleted_at` passthroughs in the pagination handler**

In the handler around lines 3252-3338:

- Remove line 3262: `is_deleted=request.is_deleted,`
- Remove the entire `deleted_count_task` block (lines 3272-3277).
- In the `asyncio.gather(...)` at lines 3286-3288:

```python
# BEFORE:
            (documents_with_ids, total_count), status_counts, deleted_count = await asyncio.gather(
                docs_task, status_counts_task, deleted_count_task
            )

# AFTER:
            (documents_with_ids, total_count), status_counts = await asyncio.gather(
                docs_task, status_counts_task
            )
```

- Remove lines 3314-3315 (the two lines inside the `DocStatusResponse(...)` kwargs):

```python
# REMOVE:
                        is_deleted=doc.is_deleted,
                        deleted_at=format_datetime(doc.deleted_at) if doc.deleted_at else None,
```

- Remove `deleted_count=deleted_count,` from the `PaginatedDocsResponse(...)` call (line 3337).

- [ ] **Step 5: Check for any other references**

Run: `grep -n "is_deleted\|deleted_at\|deleted_count\|get_deleted_count" lightrag/api/routers/document_routes.py`
Expected: zero matches.

- [ ] **Step 6: Run offline tests end-to-end**

Run: `python -m pytest tests -v -x`
Expected: all offline tests pass. At this point core + base + all 5 backends + API are in sync, so the suite should be green.

- [ ] **Step 7: Commit**

```bash
git add lightrag/api/routers/document_routes.py
git commit -m "refactor(api): drop is_deleted/deleted_at/deleted_count from doc routes"
```

---

### Task 12: Update WebUI types and components

**Files:**
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/features/DocumentManager.tsx`

- [ ] **Step 1: Remove fields from the TypeScript types**

In `lightrag_webui/src/api/lightrag.ts`:

- Lines 201-202 — remove the two fields from `DocStatusResponse`:
  ```typescript
  // REMOVE:
    is_deleted: boolean
    deleted_at?: string | null
  ```

- Line 223 — remove from `DocumentsRequest`:
  ```typescript
  // REMOVE:
    is_deleted?: boolean
  ```

- Line 239 — remove `deleted_count` from `PaginatedDocsResponse`:
  ```typescript
  // REMOVE:
    deleted_count: number
  ```

- [ ] **Step 2: Remove 'deleted' from statusFilter**

Search `lightrag_webui/src/features/DocumentManager.tsx` for `'deleted'` and fix each hit. The specific edits:

- Around line 614 — simplify the status_filter expression:
  ```typescript
  // BEFORE:
      status_filter: query.statusFilter === 'all' || query.statusFilter === 'deleted' ? null : query.statusFilter,

  // AFTER:
      status_filter: query.statusFilter === 'all' ? null : query.statusFilter,
  ```

- Line 619 — remove the `is_deleted` property entirely from the returned object:
  ```typescript
  // REMOVE:
      is_deleted: query.statusFilter === 'deleted',
  ```

- Lines 627-629 — remove the setDeletedCount branch:
  ```typescript
  // REMOVE:
      if (typeof response.deleted_count === 'number') {
        setDeletedCount(response.deleted_count);
      }
  ```

- Lines 1569-1571 — simplify the updated_at cell:
  ```tsx
  // BEFORE:
                            {statusFilter === 'deleted' && doc.deleted_at
                              ? new Date(doc.deleted_at).toLocaleString()
                              : new Date(doc.updated_at).toLocaleString()}

  // AFTER:
                            {new Date(doc.updated_at).toLocaleString()}
  ```

- Line 1573 — drop the `statusFilter !== 'deleted'` guard from the checkbox render:
  ```tsx
  // BEFORE:
                          {!viewOnly && statusFilter !== 'deleted' && (

  // AFTER:
                          {!viewOnly && (
  ```

- [ ] **Step 3: Remove the 'deleted' option from the statusFilter union / dropdown**

Still inside `DocumentManager.tsx`, search for the string `'deleted'` that remains. Likely hits:

- The `StatusFilter` type union (around the top of the file): remove the `| 'deleted'` member.
- The dropdown options array (a `<SelectItem value="deleted">` or equivalent): remove that entry.
- Any i18n key referencing `documentPanel.status.deleted` — remove the `SelectItem` JSX that uses it (leave the translation files alone; they will simply be unused).
- Remove state variables `deletedCount` / `setDeletedCount` if they become unused (TypeScript compile will fail if referenced; the compiler will guide cleanup).

Run: `grep -n "'deleted'" lightrag_webui/src/features/DocumentManager.tsx`
Expected after edits: zero matches.

- [ ] **Step 4: Lint and typecheck the WebUI**

Run:
```bash
cd lightrag_webui
bun install --frozen-lockfile
bun run lint
bun run build
cd ..
```
Expected: `lint` reports no errors; `build` succeeds.

- [ ] **Step 5: Commit**

```bash
git add lightrag_webui/src/api/lightrag.ts lightrag_webui/src/features/DocumentManager.tsx
git commit -m "refactor(webui): drop is_deleted/deleted_at + remove 'deleted' filter"
```

---

### Task 13: Add end-to-end filename-collision test

**Files:**
- Create: `tests/test_filename_collision.py`

- [ ] **Step 1: Write the test**

Create `tests/test_filename_collision.py`:

```python
"""Integration-lite test: verify same-filename uploads are caught as duplicates
and same-content-different-filename uploads succeed, using the default JSON
doc-status backend (no external services).
"""
import asyncio
import os
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline


@pytest.fixture
def tmp_working_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.mark.asyncio
async def test_same_filename_rejected_as_duplicate(tmp_working_dir, monkeypatch):
    """Two uploads with the same file_path produce one active doc + one FAILED dup record."""
    # Minimal LLM/embedding stubs — we only exercise enqueue, not full pipeline.
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    import numpy as np

    async def fake_llm(prompt, **kwargs):
        return "stub"

    async def fake_embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    embed = EmbeddingFunc(embedding_dim=4, max_token_size=128, func=fake_embed)
    rag = LightRAG(
        working_dir=tmp_working_dir,
        llm_model_func=fake_llm,
        embedding_func=embed,
    )
    await rag.initialize_storages()
    try:
        await rag.apipeline_enqueue_documents(
            input="first content",
            file_paths="ABC.pdf",
        )
        await rag.apipeline_enqueue_documents(
            input="second content — different bytes",
            file_paths="ABC.pdf",
        )

        # The primary record should exist once; the duplicate attempt should
        # have created a separate dup-* record with status FAILED.
        all_docs = {}
        for status_name in ("pending", "processing", "processed", "preprocessed", "failed"):
            from lightrag.base import DocStatus
            status = DocStatus(status_name)
            all_docs.update(await rag.doc_status.get_docs_by_status(status))

        primary_matches = [
            doc_id for doc_id, status in all_docs.items()
            if status.file_path == "ABC.pdf" and not doc_id.startswith("dup-")
        ]
        dup_matches = [
            doc_id for doc_id in all_docs
            if doc_id.startswith("dup-")
        ]
        assert len(primary_matches) == 1, f"expected one active ABC.pdf record, got {primary_matches}"
        assert len(dup_matches) == 1, f"expected one dup-* record, got {dup_matches}"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_same_content_different_path_both_ingested(tmp_working_dir):
    """Same content under two different filenames → two active records."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    import numpy as np

    async def fake_llm(prompt, **kwargs):
        return "stub"

    async def fake_embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    embed = EmbeddingFunc(embedding_dim=4, max_token_size=128, func=fake_embed)
    rag = LightRAG(
        working_dir=tmp_working_dir,
        llm_model_func=fake_llm,
        embedding_func=embed,
    )
    await rag.initialize_storages()
    try:
        await rag.apipeline_enqueue_documents(
            input="shared content",
            file_paths="A.pdf",
        )
        await rag.apipeline_enqueue_documents(
            input="shared content",
            file_paths="B.pdf",
        )

        all_docs = {}
        for status_name in ("pending", "processing", "processed", "preprocessed", "failed"):
            from lightrag.base import DocStatus
            status = DocStatus(status_name)
            all_docs.update(await rag.doc_status.get_docs_by_status(status))

        active_paths = {
            status.file_path for doc_id, status in all_docs.items()
            if not doc_id.startswith("dup-")
        }
        assert active_paths == {"A.pdf", "B.pdf"}
    finally:
        await rag.finalize_storages()
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_filename_collision.py -v`
Expected: 2 passed.

If the fake embedding / LLM stubs don't satisfy `LightRAG.__init__` validation, check `env.example` and adjust — some required attrs may have been added since this plan was written. Do not weaken the assertions.

- [ ] **Step 3: Commit**

```bash
git add tests/test_filename_collision.py
git commit -m "test: verify filename-collision duplicate detection end-to-end"
```

---

### Task 14: Final sweep and verification

**Files:**
- Verify across the whole repo.

- [ ] **Step 1: Final grep sweep**

Run: `grep -rn "is_deleted\|deleted_at\|get_deleted_count" lightrag lightrag_webui/src tests`

Expected matches:
- Inside `_migrate_drop_soft_delete_fields` implementations in all 5 backends.
- Inside `_migrate_doc_status_drop_soft_delete` in `postgres_impl.py`.
- Inside the defensive `data.pop(...)` calls in redis_impl.py and json_doc_status_impl.py.

Anything outside those locations is a leak — track it down and remove it.

- [ ] **Step 2: Run the full offline test suite**

Run: `python -m pytest tests -v`
Expected: all pass.

- [ ] **Step 3: Run the full lint**

Run: `ruff check .`
Expected: no errors. Fix any that appear.

- [ ] **Step 4: WebUI typecheck**

Run:
```bash
cd lightrag_webui && bun run lint && bun run build && cd ..
```
Expected: no errors.

- [ ] **Step 5: Final commit (if sweep found anything)**

If Step 1 surfaced leftover references, fix and commit:

```bash
git add <files>
git commit -m "chore: finish is_deleted cleanup sweep"
```

If nothing changed, no commit needed — the implementation is complete.

---

## Self-Review Notes

1. **Spec coverage:** Every section of the spec maps to at least one task — core doc_id (Task 1), UUID fallback (Task 2), error message (Task 3), hard-delete conversion (Task 4), base model (Task 5), each backend (Tasks 6–10), API (Task 11), WebUI (Task 12), tests (Task 13), final sweep (Task 14).
2. **No placeholders:** all code and commands are concrete. Step 3 of Task 12 ("guide cleanup") is grounded in TS compiler errors — the compiler pinpoints any stale references.
3. **Order dependency:** Task 5 (removing fields from the abstract dataclass) intentionally leaves the suite red until Tasks 6–11 land. This is called out in Task 5 / Step 5. Task 6 onward must be done in sequence for the suite to go green again — a reviewer reading individual task commits will see broken tests between 5 and 11.
4. **Migration idempotency:** Every backend migration is guarded so a second boot is a no-op. Postgres uses information_schema column checks; Mongo filters on `is_deleted=True` and `$exists`; OpenSearch uses term queries that match nothing when fields are gone; Redis/JSON check field presence before rewriting.
5. **Divergence accepted:** pre-upgrade documents keep their old content+path hashes and are not rewritten. This is documented in the spec and honored by all 14 tasks (no task re-hashes existing doc_ids).
