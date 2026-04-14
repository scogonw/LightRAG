# Document Token Usage Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track every LLM and embedding token consumed during document ingestion and store it per-document in `DocProcessingStatus`, broken down by stage.

**Architecture:** Add a `DocumentTokenTracker` that composes the existing `TokenTracker` with per-stage sub-trackers. Thread it through `use_llm_func_with_cache` and vector storage `upsert` methods. Persist to a new `token_usage` field on `DocProcessingStatus`.

**Tech Stack:** Python async, dataclasses, existing LightRAG storage backends

---

### Task 1: Add `DocumentTokenTracker` class to utils.py

**Files:**
- Modify: `lightrag/utils.py:2614` (after existing `TokenTracker` class)

- [ ] **Step 1: Add `DocumentTokenTracker` class after `TokenTracker`**

In `lightrag/utils.py`, add the following class right after the `TokenTracker.__str__` method (after line 2614):

```python
class DocumentTokenTracker:
    """Track token usage per document, broken down by stage.

    Composes multiple TokenTracker instances, one per stage
    (e.g. entity_extraction, graph_summary, embedding).
    """

    def __init__(self):
        self.stages: dict[str, TokenTracker] = {}

    def get_stage(self, stage: str) -> TokenTracker:
        """Get or create a TokenTracker for a specific stage."""
        if stage not in self.stages:
            self.stages[stage] = TokenTracker()
        return self.stages[stage]

    def get_usage(self) -> dict[str, Any]:
        """Return usage dict with per-stage breakdown + total."""
        result = {}
        total_prompt = 0
        total_completion = 0
        total_total = 0
        for stage, tracker in self.stages.items():
            usage = tracker.get_usage()
            result[stage] = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }
            total_prompt += usage["prompt_tokens"]
            total_completion += usage["completion_tokens"]
            total_total += usage["total_tokens"]
        result["total"] = {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_total,
        }
        return result
```

- [ ] **Step 2: Add `DocumentTokenTracker` to module exports**

Verify `DocumentTokenTracker` is importable. No `__all__` in utils.py, so it's accessible by default.

- [ ] **Step 3: Commit**

```bash
git add lightrag/utils.py
git commit -m "feat: add DocumentTokenTracker for per-stage token tracking"
```

---

### Task 2: Add `token_usage` field to `DocProcessingStatus`

**Files:**
- Modify: `lightrag/base.py:836-837` (after `deleted_at` field)

- [ ] **Step 1: Add `token_usage` field to `DocProcessingStatus`**

In `lightrag/base.py`, add the new field after `deleted_at` (after line 837):

```python
    token_usage: dict[str, Any] | None = None
    """Token usage accumulated during document ingestion, broken down by stage"""
```

This is a dataclass field with default `None`, so all existing code that creates `DocProcessingStatus` without this field will still work.

- [ ] **Step 2: Commit**

```bash
git add lightrag/base.py
git commit -m "feat: add token_usage field to DocProcessingStatus"
```

---

### Task 3: Thread `token_tracker` through `use_llm_func_with_cache`

**Files:**
- Modify: `lightrag/utils.py:1984-2123`

- [ ] **Step 1: Add `token_tracker` parameter to `use_llm_func_with_cache`**

Add `token_tracker=None` to the function signature. In `lightrag/utils.py`, change line 1993:

```python
    cache_keys_collector: list = None,
```

to:

```python
    cache_keys_collector: list = None,
    token_tracker=None,
```

- [ ] **Step 2: Pass `token_tracker` in the cache-enabled LLM call path**

In the cache-enabled path (around line 2071-2078), the existing code builds a `kwargs` dict and calls `use_llm_func`. Add `token_tracker` to kwargs. Change lines 2071-2078 from:

```python
        # Call LLM with sanitized input
        kwargs = {}
        if safe_history_messages:
            kwargs["history_messages"] = safe_history_messages
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        res: str = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )
```

to:

```python
        # Call LLM with sanitized input
        kwargs = {}
        if safe_history_messages:
            kwargs["history_messages"] = safe_history_messages
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if token_tracker is not None:
            kwargs["token_tracker"] = token_tracker

        res: str = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )
```

- [ ] **Step 3: Pass `token_tracker` in the no-cache LLM call path**

In the no-cache path (around lines 2105-2113), apply the same change. Change lines 2105-2113 from:

```python
    # When cache is disabled, directly call LLM with sanitized input
    kwargs = {}
    if safe_history_messages:
        kwargs["history_messages"] = safe_history_messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        res = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )
```

to:

```python
    # When cache is disabled, directly call LLM with sanitized input
    kwargs = {}
    if safe_history_messages:
        kwargs["history_messages"] = safe_history_messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if token_tracker is not None:
        kwargs["token_tracker"] = token_tracker

    try:
        res = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )
```

- [ ] **Step 4: Commit**

```bash
git add lightrag/utils.py
git commit -m "feat: thread token_tracker through use_llm_func_with_cache"
```

---

### Task 4: Thread `token_tracker` through entity extraction in operate.py

**Files:**
- Modify: `lightrag/operate.py` (functions: `_summarize_descriptions`, `_merge_nodes_then_upsert`, `_merge_edges_then_upsert`, `extract_entities`, `merge_nodes_and_edges`, and `_process_single_content`)

- [ ] **Step 1: Add `token_tracker` to `_summarize_descriptions`**

In `lightrag/operate.py`, change the signature at line 304-310 from:

```python
async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
```

to:

```python
async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
    token_tracker=None,
) -> str:
```

Then change the `use_llm_func_with_cache` call at lines 363-368 from:

```python
    summary, _ = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="summary",
    )
```

to:

```python
    summary, _ = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="summary",
        token_tracker=token_tracker,
    )
```

- [ ] **Step 2: Add `token_tracker` to `_merge_nodes_then_upsert`**

Change the signature at line 1648-1658 from:

```python
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
```

to:

```python
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    token_tracker=None,
):
```

Then find where `_summarize_descriptions` is called inside this function. Search for the call and add `token_tracker=token_tracker` to it. The call pattern will look like:

```python
await _summarize_descriptions(
    ...,
    llm_response_cache=llm_response_cache,
    token_tracker=token_tracker,
)
```

Also find where `entity_vdb.upsert(payload)` is called (line 1994) inside the lambda. Change:

```python
operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
```

to:

```python
operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload, token_tracker=token_tracker),
```

- [ ] **Step 3: Add `token_tracker` to `_merge_edges_then_upsert`**

Change the signature at line 2009-2023 from:

```python
async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    added_entities: list = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
```

to:

```python
async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    added_entities: list = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    token_tracker=None,
):
```

Then find and update the `_summarize_descriptions` call and `relationships_vdb.upsert()` call inside this function similarly to Step 2. Also find any `entity_vdb.upsert()` calls in the "add missing entity" logic and add `token_tracker=token_tracker`.

- [ ] **Step 4: Add `token_tracker` to `extract_entities`**

Change the signature at line 2984-2991 from:

```python
async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
) -> list:
```

to:

```python
async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
    token_tracker=None,
) -> list:
```

Then inside the nested `_process_single_content` function (line 3032), pass `token_tracker` to both `use_llm_func_with_cache` calls:

At line 3065-3073, change:

```python
        final_result, timestamp = await use_llm_func_with_cache(
            entity_extraction_user_prompt,
            use_llm_func,
            system_prompt=entity_extraction_system_prompt,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )
```

to:

```python
        final_result, timestamp = await use_llm_func_with_cache(
            entity_extraction_user_prompt,
            use_llm_func,
            system_prompt=entity_extraction_system_prompt,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
            token_tracker=token_tracker,
        )
```

At line 3113-3122 (gleaning call), change similarly:

```python
                glean_result, timestamp = await use_llm_func_with_cache(
                    entity_continue_extraction_user_prompt,
                    use_llm_func,
                    system_prompt=entity_extraction_system_prompt,
                    llm_response_cache=llm_response_cache,
                    history_messages=history,
                    cache_type="extract",
                    chunk_id=chunk_key,
                    cache_keys_collector=cache_keys_collector,
                )
```

to:

```python
                glean_result, timestamp = await use_llm_func_with_cache(
                    entity_continue_extraction_user_prompt,
                    use_llm_func,
                    system_prompt=entity_extraction_system_prompt,
                    llm_response_cache=llm_response_cache,
                    history_messages=history,
                    cache_type="extract",
                    chunk_id=chunk_key,
                    cache_keys_collector=cache_keys_collector,
                    token_tracker=token_tracker,
                )
```

- [ ] **Step 5: Add `token_tracker` to `merge_nodes_and_edges`**

Change the signature at line 2602-2618 to add `token_tracker=None`:

```python
async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    full_entities_storage: BaseKVStorage = None,
    full_relations_storage: BaseKVStorage = None,
    doc_id: str = None,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
    token_tracker=None,
) -> None:
```

Then pass `token_tracker=token_tracker` to the `_merge_nodes_then_upsert` call at line 2704-2714:

```python
                    entity_data = await _merge_nodes_then_upsert(
                        entity_name,
                        entities,
                        knowledge_graph_inst,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        entity_chunks_storage,
                        token_tracker=token_tracker,
                    )
```

And pass `token_tracker=token_tracker` to the `_merge_edges_then_upsert` call in the Phase 2 section (similar pattern).

- [ ] **Step 6: Commit**

```bash
git add lightrag/operate.py
git commit -m "feat: thread token_tracker through entity extraction and merge"
```

---

### Task 5: Add `token_tracker` parameter to vector storage `upsert`

**Files:**
- Modify: `lightrag/base.py:293` (abstract method)
- Modify: `lightrag/kg/nano_vector_db_impl.py:142`
- Modify: `lightrag/kg/faiss_impl.py:128`
- Modify: `lightrag/kg/qdrant_impl.py:655`
- Modify: `lightrag/kg/milvus_impl.py:1486`
- Modify: `lightrag/kg/opensearch_impl.py:2894`
- Modify: `lightrag/kg/postgres_impl.py:2610,3485`
- Modify: `lightrag/kg/mongo_impl.py:2366`

- [ ] **Step 1: Update abstract method in `BaseVectorStorage`**

In `lightrag/base.py`, change line 293 from:

```python
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
```

to:

```python
    async def upsert(self, data: dict[str, dict[str, Any]], token_tracker=None) -> None:
```

- [ ] **Step 2: Update `nano_vector_db_impl.py`**

Change the upsert signature at line 142 to add `token_tracker=None`:

```python
    async def upsert(self, data: dict[str, dict[str, Any]], token_tracker=None) -> None:
```

Change line 169 from:

```python
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
```

to:

```python
        embedding_kwargs = {}
        if token_tracker is not None:
            embedding_kwargs["token_tracker"] = token_tracker
        embedding_tasks = [self.embedding_func(batch, **embedding_kwargs) for batch in batches]
```

- [ ] **Step 3: Update `faiss_impl.py`**

Same pattern as Step 2. Change signature at line 128 and embedding call at line 169.

- [ ] **Step 4: Update `qdrant_impl.py`**

Same pattern. Change signature at line 655 and embedding call at line 679.

- [ ] **Step 5: Update `milvus_impl.py`**

Same pattern. Change signature at line 1486 and embedding call at line 1512.

- [ ] **Step 6: Update `opensearch_impl.py`**

Same pattern. Change signature at line 2894 and embedding call at line 2921. Also check for any other `upsert` methods in this file that call `self.embedding_func`.

- [ ] **Step 7: Update `postgres_impl.py`**

Two vector storage classes. Change signatures at lines 2610 and 3485, and embedding calls at line 3530. The upsert at line 2610 may not call embedding directly; check and update accordingly.

- [ ] **Step 8: Update `mongo_impl.py`**

Change signature and embedding call at line 2366/2390.

- [ ] **Step 9: Commit**

```bash
git add lightrag/base.py lightrag/kg/nano_vector_db_impl.py lightrag/kg/faiss_impl.py lightrag/kg/qdrant_impl.py lightrag/kg/milvus_impl.py lightrag/kg/opensearch_impl.py lightrag/kg/postgres_impl.py lightrag/kg/mongo_impl.py
git commit -m "feat: add token_tracker parameter to vector storage upsert"
```

---

### Task 6: Wire `DocumentTokenTracker` in `lightrag.py` orchestration

**Files:**
- Modify: `lightrag/lightrag.py` (functions: `process_document`, `_process_extract_entities`)

- [ ] **Step 1: Add import**

Add at the top of `lightrag/lightrag.py` in the imports from utils:

```python
from lightrag.utils import DocumentTokenTracker
```

(Find the existing `from lightrag.utils import ...` line and add `DocumentTokenTracker` to it.)

- [ ] **Step 2: Add `token_tracker` to `_process_extract_entities`**

Change the signature at line 2356-2357 from:

```python
    async def _process_extract_entities(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
```

to:

```python
    async def _process_extract_entities(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None, token_tracker=None
    ) -> list:
```

And pass it through to `extract_entities` at lines 2360-2367:

```python
            chunk_results = await extract_entities(
                chunk,
                global_config=asdict(self),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
                text_chunks_storage=self.text_chunks,
                token_tracker=token_tracker,
            )
```

- [ ] **Step 3: Create `DocumentTokenTracker` in `process_document` and pass to chunks_vdb.upsert**

Inside `process_document` (line 1886), after `processing_start_time = int(time.time())` (line 1900), add:

```python
                    doc_token_tracker = DocumentTokenTracker()
```

Then change the `chunks_vdb.upsert` call at line 2059-2061 from:

```python
                            chunks_vdb_task = asyncio.create_task(
                                self.chunks_vdb.upsert(chunks)
                            )
```

to:

```python
                            chunks_vdb_task = asyncio.create_task(
                                self.chunks_vdb.upsert(chunks, token_tracker=doc_token_tracker.get_stage("embedding"))
                            )
```

- [ ] **Step 4: Pass `DocumentTokenTracker` to `_process_extract_entities`**

Change the call at lines 2078-2081 from:

```python
                            entity_relation_task = asyncio.create_task(
                                self._process_extract_entities(
                                    chunks, pipeline_status, pipeline_status_lock
                                )
                            )
```

to:

```python
                            entity_relation_task = asyncio.create_task(
                                self._process_extract_entities(
                                    chunks, pipeline_status, pipeline_status_lock,
                                    token_tracker=doc_token_tracker.get_stage("entity_extraction"),
                                )
                            )
```

- [ ] **Step 5: Pass `DocumentTokenTracker` to `merge_nodes_and_edges`**

Change the call at lines 2172-2189 to add `token_tracker=doc_token_tracker`:

```python
                                await merge_nodes_and_edges(
                                    chunk_results=chunk_results,
                                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                                    entity_vdb=self.entities_vdb,
                                    relationships_vdb=self.relationships_vdb,
                                    global_config=asdict(self),
                                    full_entities_storage=self.full_entities,
                                    full_relations_storage=self.full_relations,
                                    doc_id=doc_id,
                                    pipeline_status=pipeline_status,
                                    pipeline_status_lock=pipeline_status_lock,
                                    llm_response_cache=self.llm_response_cache,
                                    entity_chunks_storage=self.entity_chunks,
                                    relation_chunks_storage=self.relation_chunks,
                                    current_file_number=current_file_number,
                                    total_files=total_files,
                                    file_path=file_path,
                                    token_tracker=doc_token_tracker,
                                )
```

Note: We pass the full `doc_token_tracker` here (not a stage), because `merge_nodes_and_edges` internally calls both `_summarize_descriptions` (which should use `graph_summary` stage) and vector `upsert` (which should use `embedding` stage). The `merge_nodes_and_edges` function will extract the appropriate stages.

Wait — looking at this again, we need to reconsider. In Task 4 Step 5, we passed a single `token_tracker` to `merge_nodes_and_edges`, but that function needs to route to different stages. We have two choices:
1. Pass the full `DocumentTokenTracker` and have `merge_nodes_and_edges` call `.get_stage()` internally.
2. Pass individual stage trackers.

Option 1 requires `merge_nodes_and_edges` to know about `DocumentTokenTracker` (tight coupling). Option 2 means passing two trackers.

**Resolution:** Pass the full `DocumentTokenTracker` as `token_tracker`. Inside `merge_nodes_and_edges`, when forwarding to `_merge_nodes_then_upsert` / `_merge_edges_then_upsert`, pass it as-is. Inside those functions, when calling `_summarize_descriptions`, use `token_tracker.get_stage("graph_summary") if token_tracker else None`, and when calling `entity_vdb.upsert()`, use `token_tracker.get_stage("embedding") if token_tracker else None`.

This means updating the code in Task 4 Steps 2 and 3: the `token_tracker` passed to `_merge_nodes_then_upsert` and `_merge_edges_then_upsert` is a `DocumentTokenTracker`, and those functions extract stages.

Update Task 4 Step 2 — inside `_merge_nodes_then_upsert`, the `_summarize_descriptions` call should use:

```python
token_tracker=token_tracker.get_stage("graph_summary") if token_tracker else None,
```

And the `entity_vdb.upsert` lambda should use:

```python
operation=lambda payload=data_for_vdb: entity_vdb.upsert(
    payload,
    token_tracker=token_tracker.get_stage("embedding") if token_tracker else None,
),
```

Similarly for Task 4 Step 3 (`_merge_edges_then_upsert`).

- [ ] **Step 6: Persist `token_usage` in the final PROCESSED status update**

Change the final doc status upsert at lines 2194-2215. Add `"token_usage"` to the data dict. Change from:

```python
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSED,
                                            "chunks_count": len(chunks),
                                            "chunks_list": list(chunks.keys()),
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,
                                            "org_id": status_doc.org_id,
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                            },
                                        }
                                    }
                                )
```

to:

```python
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSED,
                                            "chunks_count": len(chunks),
                                            "chunks_list": list(chunks.keys()),
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,
                                            "org_id": status_doc.org_id,
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                            },
                                            "token_usage": doc_token_tracker.get_usage(),
                                        }
                                    }
                                )
```

- [ ] **Step 7: Commit**

```bash
git add lightrag/lightrag.py
git commit -m "feat: wire DocumentTokenTracker through document ingestion pipeline"
```

---

### Task 7: Handle `token_usage` in storage backends

**Files:**
- Modify: `lightrag/kg/json_doc_status_impl.py` (multiple reconstruction sites)
- Modify: `lightrag/kg/postgres_impl.py` (upsert SQL, reconstruction, migration)
- Modify: `lightrag/kg/mongo_impl.py` (reconstruction sites)
- Modify: `lightrag/kg/redis_impl.py` (reconstruction sites)
- Modify: `lightrag/kg/opensearch_impl.py` (reconstruction sites)

- [ ] **Step 1: JSON doc status — no changes needed for storage**

The JSON backend stores raw dicts and reconstructs via `DocProcessingStatus(**data)`. Since `token_usage` has a default of `None`, existing data without it will work fine. New data will include it if present.

However, we need to ensure the reconstruction sites don't strip it. Check `json_doc_status_impl.py` — the reconstruction uses `data = v.copy()` then `DocProcessingStatus(**data)`. Since `token_usage` will be in the dict when present, it passes through naturally. No changes needed.

- [ ] **Step 2: Postgres — add `token_usage` column migration**

Add a new migration method in `lightrag/kg/postgres_impl.py` after `_migrate_doc_status_add_soft_delete` (after line 1325):

```python
    async def _migrate_doc_status_add_token_usage(self):
        """Add token_usage column to LIGHTRAG_DOC_STATUS table if it doesn't exist"""
        try:
            check_token_usage_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'token_usage'
            """

            token_usage_info = await self.query(check_token_usage_sql)
            if not token_usage_info:
                logger.info("Adding token_usage column to LIGHTRAG_DOC_STATUS table")
                add_token_usage_sql = """
                ALTER TABLE LIGHTRAG_DOC_STATUS
                ADD COLUMN token_usage JSONB NULL
                """
                await self.execute(add_token_usage_sql)
                logger.info(
                    "Successfully added token_usage column to LIGHTRAG_DOC_STATUS table"
                )
            else:
                logger.info(
                    "token_usage column already exists in LIGHTRAG_DOC_STATUS table"
                )

        except Exception as e:
            logger.warning(
                f"Failed to add token_usage column to LIGHTRAG_DOC_STATUS: {e}"
            )
```

- [ ] **Step 3: Postgres — call migration in initialization**

Add the migration call after the soft-delete migration (after line 1633):

```python
        # Migrate doc status to add token_usage field if needed
        try:
            await self._migrate_doc_status_add_token_usage()
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to migrate doc status token_usage field: {e}"
            )
```

- [ ] **Step 4: Postgres — update upsert SQL**

In the `PGDocStatusStorage.upsert` method (line 4596), update the SQL to include `token_usage`. Change the INSERT statement from:

```sql
insert into LIGHTRAG_DOC_STATUS(workspace,id,org_id,content_summary,content_length,chunks_count,status,file_path,chunks_list,track_id,metadata,error_msg,created_at,updated_at)
values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
on conflict(id,workspace) do update set ...
```

to include `token_usage` as parameter `$15`:

```sql
insert into LIGHTRAG_DOC_STATUS(workspace,id,org_id,content_summary,content_length,chunks_count,status,file_path,chunks_list,track_id,metadata,error_msg,created_at,updated_at,token_usage)
values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
on conflict(id,workspace) do update set
  ...existing columns...,
  token_usage = EXCLUDED.token_usage
```

And add the value to the batch tuple:

```python
json.dumps(v.get("token_usage")) if v.get("token_usage") else None,
```

- [ ] **Step 5: Postgres — update all `DocProcessingStatus` reconstruction sites**

In every place in `postgres_impl.py` where `DocProcessingStatus(...)` is constructed, add:

```python
token_usage=element.get("token_usage"),
```

This applies to the reconstruction at lines ~4155, ~4214, ~4280, ~4446 (and any others found by searching for `DocProcessingStatus(` in the file). The `token_usage` field from Postgres will be a dict (JSONB auto-parses) or None.

- [ ] **Step 6: MongoDB/Redis/OpenSearch — update reconstruction sites**

These backends use `DocProcessingStatus(**data)` which passes through all dict keys. Since `token_usage` defaults to `None`, existing documents without it work fine, and new documents with it pass through. Verify no reconstruction site explicitly lists fields (which would drop `token_usage`).

For MongoDB (`mongo_impl.py`): Check lines ~489, ~507, ~743. If they use `DocProcessingStatus(**data)`, no change needed.

For Redis (`redis_impl.py`): Check lines ~782, ~842, ~1024. If they use `DocProcessingStatus(**data)`, no change needed.

For OpenSearch (`opensearch_impl.py`): Check lines ~870, ~1008. If they use `DocProcessingStatus(**data)`, no change needed.

- [ ] **Step 7: Commit**

```bash
git add lightrag/kg/postgres_impl.py lightrag/kg/json_doc_status_impl.py lightrag/kg/mongo_impl.py lightrag/kg/redis_impl.py lightrag/kg/opensearch_impl.py
git commit -m "feat: handle token_usage field in storage backends with Postgres migration"
```

---

### Task 8: Run tests and verify

**Files:**
- No new files

- [ ] **Step 1: Run offline tests**

```bash
python -m pytest tests -x -v 2>&1 | tail -30
```

Expected: All existing tests pass. The new `token_usage` field defaults to `None` so existing tests won't break.

- [ ] **Step 2: Run linting**

```bash
ruff check lightrag/utils.py lightrag/base.py lightrag/operate.py lightrag/lightrag.py lightrag/kg/
```

Expected: No new linting errors.

- [ ] **Step 3: Fix any issues found**

Address any test failures or lint errors.

- [ ] **Step 4: Commit fixes if needed**

```bash
git add -u
git commit -m "fix: address test/lint issues from token tracking implementation"
```
