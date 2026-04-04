# Metadata Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add user-defined metadata to documents at upload time and enable filtering by that metadata during vector and graph queries, exposed through the REST API at both insertion and query endpoints.

**Architecture:** Metadata is a flat `dict[str, Any]` attached to documents at insert time. It propagates from documents -> chunks -> entities -> relations (via `file_path`-style inheritance). At query time, a `metadata_filter: dict[str, Any]` parameter allows exact-match filtering applied as a post-retrieval filter in the base layer (works across all backends) with optional pre-filtering for SQL-based backends. The filtering is applied to vector search results and graph node/edge retrieval.

**Tech Stack:** Python 3.10+, FastAPI (Pydantic v2), dataclasses, NanoVectorDB, PostgreSQL/pgvector, NetworkX, Neo4j

---

## Current Architecture Summary

### Data Flow (Insert)
```
API (InsertTextRequest) 
  -> LightRAG.ainsert(input, file_paths, ...) 
  -> apipeline_enqueue_documents() -- stores {doc_id: {content, file_path}} in full_docs
  -> apipeline_process_enqueue_documents()
    -> chunking -- creates chunks with file_path from parent doc
    -> chunks_vdb.upsert({chunk_id: {content, file_path, ...}})
    -> entity extraction -> _merge_nodes_then_upsert() 
      -> graph.upsert_node(name, {file_path, source_id, ...})
      -> entities_vdb.upsert({ent_id: {content, entity_name, file_path, ...}})
    -> relation extraction -> _merge_edges_then_upsert()
      -> graph.upsert_edge(src, tgt, {file_path, source_id, ...})
      -> relationships_vdb.upsert({rel_id: {content, file_path, ...}})
```

### Data Flow (Query)
```
API (QueryRequest) -> QueryParam -> kg_query()
  -> _perform_kg_search()
    -> chunks_vdb.query(query, top_k) -- naive/mix modes
    -> entities_vdb.query(ll_keywords, top_k) -- local/hybrid/mix
    -> relationships_vdb.query(hl_keywords, top_k) -- global/hybrid/mix
    -> graph.get_node() / get_edge() for property retrieval
  -> _build_query_context() -> LLM response
```

### Key Files
| File | Purpose |
|------|---------|
| `lightrag/base.py:84-170` | `QueryParam` dataclass |
| `lightrag/base.py:217-352` | `BaseVectorStorage` abstract class |
| `lightrag/base.py:405-703` | `BaseGraphStorage` abstract class |
| `lightrag/lightrag.py:1236-1269` | `ainsert()` method |
| `lightrag/lightrag.py:1343-1528` | `apipeline_enqueue_documents()` |
| `lightrag/operate.py:1100-1170` | `_rebuild_single_entity()` - entity VDB upsert |
| `lightrag/operate.py:1575-1590` | relationship VDB upsert |
| `lightrag/operate.py:1623-1945` | `_merge_nodes_then_upsert()` |
| `lightrag/operate.py:1948-2500` | `_merge_edges_then_upsert()` |
| `lightrag/operate.py:3525-3570` | `_get_vector_context()` - chunk vector search |
| `lightrag/operate.py:3573-3780` | `_perform_kg_search()` - main search orchestrator |
| `lightrag/api/routers/document_routes.py:228-242` | `InsertTextRequest` model |
| `lightrag/api/routers/document_routes.py:264-278` | `InsertTextsRequest` model |
| `lightrag/api/routers/query_routes.py:16-111` | `QueryRequest` model |
| `lightrag/kg/nano_vector_db_impl.py:144-172` | NanoVectorDB `query()` |
| `lightrag/kg/postgres_impl.py:3428-3457` | PGVector `query()` |
| `lightrag/kg/postgres_impl.py:6484-6513` | SQL query templates for entities/relations/chunks |

---

## File Structure

### Files to modify:
1. **`lightrag/base.py`** - Add `metadata_filter` to `QueryParam`; add `metadata` to `BaseVectorStorage.query()` signature
2. **`lightrag/lightrag.py`** - Add `metadata` parameter to `ainsert()`, `apipeline_enqueue_documents()`; propagate through chunking
3. **`lightrag/operate.py`** - Propagate metadata through entity/relation upserts; apply metadata filtering in `_get_vector_context()` and `_perform_kg_search()`
4. **`lightrag/api/routers/document_routes.py`** - Add `metadata` field to `InsertTextRequest`, `InsertTextsRequest`; pass through to pipeline
5. **`lightrag/api/routers/query_routes.py`** - Add `metadata_filter` field to `QueryRequest`; map to `QueryParam`
6. **`lightrag/kg/nano_vector_db_impl.py`** - Implement post-retrieval metadata filtering in `query()`
7. **`lightrag/kg/postgres_impl.py`** - Implement SQL WHERE clause metadata filtering in `query()` + add `metadata` JSONB column

### Files to create:
- None (all changes fit in existing files)

---

## Task 1: Add `metadata` field to `QueryParam` and `BaseVectorStorage.query()`

**Files:**
- Modify: `lightrag/base.py:84-170` (QueryParam)
- Modify: `lightrag/base.py:261-272` (BaseVectorStorage.query signature)

- [ ] **Step 1: Add `metadata_filter` field to `QueryParam`**

In `lightrag/base.py`, add after the `include_references` field (line 169):

```python
    metadata_filter: dict[str, Any] | None = None
    """Optional metadata filter for retrieval. Only returns results whose stored metadata
    contains all key-value pairs specified here (exact match).
    Example: {"department": "engineering", "year": 2024}
    """
```

Also ensure `Any` is imported from `typing` at the top of the file (it likely already is).

- [ ] **Step 2: Add `metadata_filter` parameter to `BaseVectorStorage.query()`**

Change the abstract `query()` method signature at line 262:

```python
    @abstractmethod
    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Query the vector storage and retrieve top_k results.

        Args:
            query: The query string to search for
            top_k: Number of top results to return
            query_embedding: Optional pre-computed embedding for the query.
                           If provided, skips embedding computation for better performance.
            metadata_filter: Optional dict of key-value pairs for exact-match filtering.
                           Only results whose stored metadata contains all specified pairs are returned.
        """
```

- [ ] **Step 3: Commit**

```bash
git add lightrag/base.py
git commit -m "feat: add metadata_filter to QueryParam and BaseVectorStorage.query()"
```

---

## Task 2: Add `metadata` to insert API request models

**Files:**
- Modify: `lightrag/api/routers/document_routes.py:228-242` (InsertTextRequest)
- Modify: `lightrag/api/routers/document_routes.py:264-278` (InsertTextsRequest)

- [ ] **Step 1: Add `metadata` field to `InsertTextRequest`**

After the `file_source` field (line 242), add:

```python
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata dict attached to the document. Propagates to chunks, entities, and relations. Used for filtering during queries. Example: {\"department\": \"engineering\", \"year\": 2024}",
    )
```

Ensure `Dict` and `Any` are imported from `typing` at the top of the file. Check existing imports first.

- [ ] **Step 2: Add `metadata` field to `InsertTextsRequest`**

After the `file_sources` field (line 278), add:

```python
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata dict attached to all documents in this batch. Propagates to chunks, entities, and relations. Used for filtering during queries.",
    )
```

- [ ] **Step 3: Add `metadata_filter` field to `QueryRequest`**

In `lightrag/api/routers/query_routes.py`, after the `stream` field (line 111), add:

```python
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filter for retrieval. Only returns results whose metadata contains all specified key-value pairs (exact match). Example: {\"department\": \"engineering\"}",
    )
```

Ensure `Dict` and `Any` are imported from `typing`.

- [ ] **Step 4: Commit**

```bash
git add lightrag/api/routers/document_routes.py lightrag/api/routers/query_routes.py
git commit -m "feat: add metadata fields to insert and query API request models"
```

---

## Task 3: Wire `metadata` through the insert API endpoints

**Files:**
- Modify: `lightrag/api/routers/document_routes.py:2286-2362` (insert_text endpoint)
- Modify: `lightrag/api/routers/document_routes.py:2364-2450` (insert_texts endpoint)

- [ ] **Step 1: Pass metadata from insert_text endpoint to pipeline**

In the `insert_text` endpoint (around line 2346), modify the `pipeline_index_texts` call to include metadata:

```python
            background_tasks.add_task(
                pipeline_index_texts,
                rag,
                [request.text],
                file_sources=[request.file_source],
                track_id=track_id,
                metadata=request.metadata,
            )
```

You'll also need to update `pipeline_index_texts` to accept and forward `metadata`. Find this function (search for `def pipeline_index_texts` or `async def pipeline_index_texts`) and add `metadata: dict[str, Any] | None = None` parameter, forwarding it to `rag.ainsert()`.

- [ ] **Step 2: Do the same for insert_texts endpoint**

Apply the equivalent change to the `insert_texts` endpoint, passing `request.metadata` through.

- [ ] **Step 3: Commit**

```bash
git add lightrag/api/routers/document_routes.py
git commit -m "feat: wire metadata through insert API endpoints to pipeline"
```

---

## Task 4: Propagate `metadata` through `LightRAG.ainsert()` and `apipeline_enqueue_documents()`

**Files:**
- Modify: `lightrag/lightrag.py:1236-1269` (ainsert)
- Modify: `lightrag/lightrag.py:1343-1528` (apipeline_enqueue_documents)

- [ ] **Step 1: Add `metadata` param to `ainsert()`**

At `lightrag/lightrag.py:1236`, add `metadata` parameter:

```python
    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
```

Forward it to `apipeline_enqueue_documents`:

```python
        await self.apipeline_enqueue_documents(input, ids, file_paths, track_id, metadata=metadata)
```

- [ ] **Step 2: Add `metadata` param to `apipeline_enqueue_documents()`**

At `lightrag/lightrag.py:1343`, add `metadata` parameter:

```python
    async def apipeline_enqueue_documents(
        self,
        input: str | list[str],
        ids: list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
```

- [ ] **Step 3: Store metadata in the `contents` dict**

In the contents building logic (lines 1404-1430), add metadata to each entry. For the custom-IDs branch (~line 1411):

```python
            contents = {
                id_: {"content": content, "file_path": file_path, "metadata": metadata}
                for content, (id_, file_path) in unique_contents.items()
            }
```

And for the auto-IDs branch (~line 1424):

```python
            contents = {
                compute_mdhash_id(content, prefix="doc-"): {
                    "content": content,
                    "file_path": path,
                    "metadata": metadata,
                }
                for content, path in unique_content_with_paths.items()
            }
```

- [ ] **Step 4: Store metadata in `full_docs_data`**

At line 1513, add metadata to the full_docs data:

```python
        full_docs_data = {
            doc_id: {
                "content": contents[doc_id]["content"],
                "file_path": contents[doc_id]["file_path"],
                "metadata": contents[doc_id].get("metadata"),
            }
            for doc_id in new_docs.keys()
        }
```

- [ ] **Step 5: Commit**

```bash
git add lightrag/lightrag.py
git commit -m "feat: propagate metadata through ainsert and enqueue pipeline"
```

---

## Task 5: Propagate metadata from documents to chunks

**Files:**
- Modify: `lightrag/lightrag.py` (find the chunking section in `apipeline_process_enqueue_documents`)

- [ ] **Step 1: Find where chunks are created and upserted to chunks_vdb**

Search for `chunks_vdb.upsert` in `lightrag.py`. The chunk data dict includes `content`, `full_doc_id`, `tokens`, `chunk_order_index`, `file_path`. Add `metadata` from the parent document.

When building chunk data for upsert, read the parent doc's metadata from `full_docs` and include it:

```python
# In the chunk building section, for each chunk:
# Get metadata from the parent document's full_docs entry
doc_data = await self.full_docs.get_by_id(full_doc_id)
chunk_metadata = doc_data.get("metadata") if doc_data else None

# Add to chunk upsert data
chunk_data[chunk_id] = {
    "content": chunk_content,
    "full_doc_id": full_doc_id,
    "tokens": token_count,
    "chunk_order_index": order_index,
    "file_path": file_path,
    "metadata": chunk_metadata,
}
```

Note: The exact code structure depends on the chunking loop. Read the section carefully and insert metadata propagation at the right place. The key principle is: each chunk inherits its parent document's metadata.

- [ ] **Step 2: Commit**

```bash
git add lightrag/lightrag.py
git commit -m "feat: propagate document metadata to chunks during chunking"
```

---

## Task 6: Propagate metadata to entities and relations in operate.py

**Files:**
- Modify: `lightrag/operate.py:1130-1156` (_rebuild_single_entity / _update_entity_storage)
- Modify: `lightrag/operate.py:1575-1590` (relationship VDB upsert in _rebuild_single_relationship)
- Modify: `lightrag/operate.py:1906-1938` (_merge_nodes_then_upsert entity upsert)
- Modify: `lightrag/operate.py:2446-2490` (_merge_edges_then_upsert relation upsert)

- [ ] **Step 1: Understand the metadata flow from chunks to entities/relations**

Entities and relations are extracted from chunks. Each entity/relation has `source_id` (chunk IDs joined by GRAPH_FIELD_SEP). The metadata should be derived from the source chunks' metadata. Since all chunks from the same document share the same metadata, and entities can span multiple documents, the entity's metadata should be the **union** of its source chunks' metadata values (stored as a list per key if they differ, or a single value if all the same).

**Simpler approach for v1:** Store metadata as-is from the first source chunk. Entities that span multiple documents with different metadata will inherit metadata from the first chunk encountered. This avoids complex merging logic.

**Even simpler approach (recommended for v1):** Don't store metadata directly on entities/relations. Instead, during query-time filtering, filter the **chunks** by metadata first, then only use entities/relations that are connected to matching chunks (via `source_id` field which contains chunk IDs). This keeps the insertion path unchanged and centralizes filtering logic.

- [ ] **Step 2: Choose the query-time filtering approach**

For v1, the recommended approach is:
1. Store metadata on **chunks only** (done in Task 5)
2. During query, filter vector search results (chunks, entities, relations) by checking if their source chunks match the metadata filter
3. For entities/relations, resolve `source_id` -> chunk IDs -> check chunk metadata

However, this requires extra lookups. A pragmatic middle ground:
- Store metadata on chunks in the vector DB
- Store metadata on entities and relations in the vector DB too (inherited from source chunks)
- Apply post-retrieval filtering on the `metadata` field

Let's go with storing metadata on entities/relations too. In `_merge_nodes_then_upsert`, when building entity data, collect metadata from source chunks. Add to the VDB upsert data.

- [ ] **Step 3: Add metadata to entity VDB upsert in `_update_entity_storage` (line ~1147)**

In `_rebuild_single_entity`, the `_update_entity_storage` inner function builds `vdb_data`. Add metadata:

```python
            vdb_data = {
                entity_vdb_id: {
                    "content": entity_content,
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                    "metadata": updated_entity_data.get("metadata"),
                }
            }
```

Similarly update the graph node data in `_merge_nodes_then_upsert` to carry metadata. The metadata should come from the chunks that this entity was extracted from. Read the chunk metadata from the text_chunks KV store.

- [ ] **Step 4: Add metadata to relation VDB upsert**

Apply the same pattern to `_merge_edges_then_upsert` and `_rebuild_single_relationship`.

- [ ] **Step 5: Commit**

```bash
git add lightrag/operate.py
git commit -m "feat: propagate metadata to entity and relation VDB upserts"
```

---

## Task 7: Implement post-retrieval metadata filtering in NanoVectorDB

**Files:**
- Modify: `lightrag/kg/nano_vector_db_impl.py:144-172` (query method)

- [ ] **Step 1: Add `metadata_filter` parameter and post-filter logic**

Update the `query()` method:

```python
    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        # Use provided embedding or compute it
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding = await self.embedding_func(
                [query], _priority=5
            )
            embedding = embedding[0]

        client = await self._get_client()
        # Fetch extra results when filtering to compensate for filtered-out items
        fetch_top_k = top_k * 3 if metadata_filter else top_k
        results = client.query(
            query=embedding,
            top_k=fetch_top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {
                **{k: v for k, v in dp.items() if k != "vector"},
                "id": dp["__id__"],
                "distance": dp["__metrics__"],
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]

        # Apply metadata filter (post-retrieval exact match)
        if metadata_filter:
            results = _apply_metadata_filter(results, metadata_filter)

        return results[:top_k]
```

- [ ] **Step 2: Add the `_apply_metadata_filter` helper function**

Add this at module level in `nano_vector_db_impl.py`:

```python
def _apply_metadata_filter(
    results: list[dict[str, Any]], metadata_filter: dict[str, Any]
) -> list[dict[str, Any]]:
    """Filter results by exact-match on metadata key-value pairs.
    
    A result matches if its 'metadata' dict contains all key-value pairs
    from metadata_filter. Results without metadata are excluded when a filter is active.
    """
    filtered = []
    for result in results:
        result_metadata = result.get("metadata")
        if not isinstance(result_metadata, dict):
            continue
        if all(result_metadata.get(k) == v for k, v in metadata_filter.items()):
            filtered.append(result)
    return filtered
```

- [ ] **Step 3: Commit**

```bash
git add lightrag/kg/nano_vector_db_impl.py
git commit -m "feat: implement post-retrieval metadata filtering in NanoVectorDB"
```

---

## Task 8: Implement metadata filtering in PGVectorStorage

**Files:**
- Modify: `lightrag/kg/postgres_impl.py:3428-3457` (PGVectorStorage.query)
- Modify: `lightrag/kg/postgres_impl.py:6448-6513` (SQL templates)
- Note: Schema migration needed to add `metadata` JSONB column to vector tables

- [ ] **Step 1: Add `metadata` JSONB column to vector tables**

Find the table creation SQL for vector storage tables (search for `CREATE TABLE` with `content_vector` in `postgres_impl.py`). Add a `metadata JSONB` column to the chunks, entities, and relationships tables.

Also update the upsert SQL templates at lines 6448-6483 to include the `metadata` column.

- [ ] **Step 2: Update `query()` to accept and apply `metadata_filter`**

```python
    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embeddings = await self.embedding_func(
                [query], _priority=5
            )
            embedding = embeddings[0]

        embedding_string = ",".join(map(str, embedding))

        vector_cast = (
            "halfvec"
            if getattr(self.db, "vector_index_type", None) == "HNSW_HALFVEC"
            else "vector"
        )
        
        # Build SQL with optional metadata filter
        if metadata_filter:
            sql_key = self.namespace + "_with_metadata"
            sql = SQL_TEMPLATES[sql_key].format(
                embedding_string=embedding_string,
                table_name=self.table_name,
                vector_cast=vector_cast,
            )
            import json
            params = {
                "workspace": self.workspace,
                "closer_than_threshold": 1 - self.cosine_better_than_threshold,
                "top_k": top_k,
                "metadata_filter": json.dumps(metadata_filter),
            }
        else:
            sql = SQL_TEMPLATES[self.namespace].format(
                embedding_string=embedding_string,
                table_name=self.table_name,
                vector_cast=vector_cast,
            )
            params = {
                "workspace": self.workspace,
                "closer_than_threshold": 1 - self.cosine_better_than_threshold,
                "top_k": top_k,
            }
        results = await self.db.query(sql, params=list(params.values()), multirows=True)
        return results
```

- [ ] **Step 3: Add metadata-filtered SQL templates**

Add new SQL templates with `metadata @> $4::jsonb` clause (PostgreSQL JSONB containment operator):

```python
    "chunks_with_metadata": """
              SELECT c.id,
                     c.content,
                     c.file_path,
                     EXTRACT(EPOCH FROM c.create_time)::BIGINT AS created_at
              FROM {table_name} c
              WHERE c.workspace = $1
                AND c.content_vector <=> '[{embedding_string}]'::{vector_cast} < $2
                AND c.metadata @> $4::jsonb
              ORDER BY c.content_vector <=> '[{embedding_string}]'::{vector_cast}
              LIMIT $3;
              """,
    "entities_with_metadata": """
                SELECT e.entity_name,
                       EXTRACT(EPOCH FROM e.create_time)::BIGINT AS created_at
                FROM {table_name} e
                WHERE e.workspace = $1
                  AND e.content_vector <=> '[{embedding_string}]'::{vector_cast} < $2
                  AND e.metadata @> $4::jsonb
                ORDER BY e.content_vector <=> '[{embedding_string}]'::{vector_cast}
                LIMIT $3;
                """,
    "relationships_with_metadata": """
                     SELECT r.source_id AS src_id,
                            r.target_id AS tgt_id,
                            EXTRACT(EPOCH FROM r.create_time)::BIGINT AS created_at
                     FROM {table_name} r
                     WHERE r.workspace = $1
                       AND r.content_vector <=> '[{embedding_string}]'::{vector_cast} < $2
                       AND r.metadata @> $4::jsonb
                     ORDER BY r.content_vector <=> '[{embedding_string}]'::{vector_cast}
                     LIMIT $3;
                     """,
```

- [ ] **Step 4: Update upsert methods to store metadata**

In the PGVectorStorage `_upsert_chunks()`, `_upsert_entities()`, `_upsert_relationships()` methods, include the metadata field in the upsert SQL and parameters.

- [ ] **Step 5: Commit**

```bash
git add lightrag/kg/postgres_impl.py
git commit -m "feat: implement metadata filtering in PGVectorStorage with JSONB containment"
```

---

## Task 9: Wire `metadata_filter` through query pipeline in operate.py

**Files:**
- Modify: `lightrag/operate.py:3525-3570` (_get_vector_context)
- Modify: `lightrag/operate.py:3573-3780` (_perform_kg_search)
- Modify: `lightrag/operate.py:4359-4388` (_get_node_data / _get_edge_data)

- [ ] **Step 1: Pass `metadata_filter` from `QueryParam` to `_get_vector_context`**

In `_get_vector_context()` (line ~3542), pass `metadata_filter` to the VDB query:

```python
        results = await chunks_vdb.query(
            query, top_k=search_top_k, query_embedding=query_embedding,
            metadata_filter=query_param.metadata_filter
        )
```

- [ ] **Step 2: Pass `metadata_filter` to entity VDB query in `_get_node_data`**

Find `_get_node_data()` (line ~4359). It queries `entities_vdb`. Pass `metadata_filter`:

```python
        results = await entities_vdb.query(
            ll_keywords, top_k=..., query_embedding=ll_embedding,
            metadata_filter=query_param.metadata_filter
        )
```

- [ ] **Step 3: Pass `metadata_filter` to relationship VDB query in `_get_edge_data`**

Find the relationship VDB query call. Pass `metadata_filter`:

```python
        results = await relationships_vdb.query(
            hl_keywords, top_k=..., query_embedding=hl_embedding,
            metadata_filter=query_param.metadata_filter
        )
```

- [ ] **Step 4: Commit**

```bash
git add lightrag/operate.py
git commit -m "feat: wire metadata_filter through query pipeline to VDB queries"
```

---

## Task 10: Wire `metadata_filter` from API QueryRequest to QueryParam

**Files:**
- Modify: `lightrag/api/routers/query_routes.py` (where QueryRequest is mapped to QueryParam)

- [ ] **Step 1: Find where QueryRequest fields are mapped to QueryParam**

Search for where `QueryParam` is constructed from `QueryRequest` fields in `query_routes.py`. Add:

```python
    if request.metadata_filter is not None:
        param.metadata_filter = request.metadata_filter
```

Or if `QueryParam` is constructed via keyword arguments:

```python
    QueryParam(
        ...,
        metadata_filter=request.metadata_filter,
    )
```

- [ ] **Step 2: Commit**

```bash
git add lightrag/api/routers/query_routes.py
git commit -m "feat: map metadata_filter from API request to QueryParam"
```

---

## Task 11: Update remaining vector storage backends

**Files:**
- Modify: All other vector storage implementations in `lightrag/kg/`:
  - `faiss_impl.py`
  - `milvus_impl.py`
  - `qdrant_impl.py`
  - `redis_impl.py`
  - `mongo_impl.py`
  - `opensearch_impl.py`
  - `chroma_impl.py`
  - `tidb_impl.py`

- [ ] **Step 1: Update each backend's `query()` signature**

For each backend, add `metadata_filter: dict[str, Any] | None = None` to the `query()` method signature.

- [ ] **Step 2: Add post-retrieval filtering**

For backends that don't have native metadata filtering support, use the same `_apply_metadata_filter` pattern as NanoVectorDB. Extract the helper to a shared utility or duplicate it in each file.

For backends with native metadata filtering (Milvus has `expr`, Qdrant has `payload` filters, Redis has `FILTER`, etc.), implement native pre-filtering where possible.

- [ ] **Step 3: Commit**

```bash
git add lightrag/kg/
git commit -m "feat: add metadata_filter support to all vector storage backends"
```

---

## Task 12: Add metadata to NanoVectorDB upsert to actually store it

**Files:**
- Modify: `lightrag/kg/nano_vector_db_impl.py:96-142` (upsert method)

- [ ] **Step 1: Ensure `metadata` is included in meta_fields**

In the NanoVectorDB `upsert()` method, the `meta_fields` set determines which fields from the data dict are stored alongside vectors. The `metadata` field must be included.

Check how `meta_fields` is populated. If it's auto-populated from the data dict keys, metadata will flow automatically. If it's a fixed set, add `"metadata"` to it.

In the upsert method (~line 108-115), metadata from the data dict should flow through as a stored field. Verify this by checking what fields are extracted:

```python
# Existing code stores fields from data that are in meta_fields
# Make sure "metadata" gets stored
```

If the implementation filters by `meta_fields`, ensure `"metadata"` is added to the set during initialization or upsert.

- [ ] **Step 2: Commit**

```bash
git add lightrag/kg/nano_vector_db_impl.py
git commit -m "feat: ensure metadata field is stored in NanoVectorDB"
```

---

## Summary of Changes by Layer

| Layer | Change | Files |
|-------|--------|-------|
| **API (Request)** | Add `metadata` to insert models, `metadata_filter` to query model | document_routes.py, query_routes.py |
| **API (Wiring)** | Pass metadata/filter through endpoints | document_routes.py, query_routes.py |
| **Core (Insert)** | Add `metadata` param to `ainsert()`, propagate to enqueue | lightrag.py |
| **Core (Chunks)** | Attach doc metadata to chunks during chunking | lightrag.py |
| **Core (KG)** | Propagate metadata to entity/relation VDB upserts | operate.py |
| **Core (Query)** | Pass `metadata_filter` to VDB queries | operate.py |
| **Storage (Base)** | Add `metadata_filter` to `query()` signature | base.py |
| **Storage (Impl)** | Implement filtering (post-retrieval or native) | nano_vector_db_impl.py, postgres_impl.py, others |

## Notes

- **Graph storage filtering**: For v1, metadata filtering is applied at the **vector search** layer, not directly on graph traversal. Since entity/relation retrieval starts with vector search (finding relevant entities/relations by embedding similarity), filtering at the vector layer effectively gates what enters the graph traversal. Direct graph filtering (e.g., Cypher WHERE clauses in Neo4j) can be added in a follow-up.

- **Metadata schema**: Metadata is an unstructured `dict[str, Any]`. Values should be primitives (str, int, float, bool) for reliable filtering. Nested objects are not supported for filtering in v1.

- **Backward compatibility**: All metadata parameters are optional with `None` defaults. Existing documents without metadata will have `metadata=None` and will be excluded from filtered queries (by design - if you filter, only documents with matching metadata are returned).
