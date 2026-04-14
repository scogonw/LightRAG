# Document Token Usage Tracking

Track every token consumed (LLM and embedding) during document ingestion and store the accumulated usage against the document.

## Decisions

- **Storage:** Dedicated `token_usage` field on `DocProcessingStatus` (not in `metadata`)
- **Granularity:** Broken down by stage (`entity_extraction`, `graph_summary`, `embedding`) plus a `total` rollup
- **Cache hits:** Record zero tokens (only actual API calls are tracked)
- **Missing provider data:** Store zeros (no null distinction)
- **API exposure:** Included automatically in existing doc status endpoints

## Data Model

### New field on `DocProcessingStatus` (base.py)

```python
token_usage: dict[str, Any] | None = None
"""Token usage accumulated during document ingestion, broken down by stage"""
```

Default `None` ensures backward compatibility with existing stored documents.

### Stored shape

```json
{
    "entity_extraction": {
        "prompt_tokens": 12500,
        "completion_tokens": 3200,
        "total_tokens": 15700
    },
    "graph_summary": {
        "prompt_tokens": 800,
        "completion_tokens": 200,
        "total_tokens": 1000
    },
    "embedding": {
        "prompt_tokens": 5000,
        "completion_tokens": 0,
        "total_tokens": 5000
    },
    "total": {
        "prompt_tokens": 18300,
        "completion_tokens": 3400,
        "total_tokens": 21700
    }
}
```

## New Class: `DocumentTokenTracker` (utils.py)

Composes the existing `TokenTracker` class. One instance per document, with sub-trackers per stage.

```python
class DocumentTokenTracker:
    """Track token usage per document, broken down by stage."""

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
        total_prompt = total_completion = total_total = 0
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

## Integration Points

### 1. `use_llm_func_with_cache` (utils.py, line 1984)

Add `token_tracker: Any | None = None` parameter. On cache miss, pass it as a kwarg to `use_llm_func()`. On cache hit, do nothing (zero tokens consumed).

### 2. Entity extraction (operate.py)

**`extract_entities()` (line 2984):** Add `token_tracker: DocumentTokenTracker | None = None` parameter. Pass `token_tracker.get_stage("entity_extraction")` to `use_llm_func_with_cache()` for extraction and gleaning calls.

**`_process_single_content()` (line 3032):** Add `token_tracker` parameter, forward to `use_llm_func_with_cache()`.

**`merge_nodes_and_edges()` (line 2602):** Add `token_tracker` parameter. Pass `token_tracker.get_stage("graph_summary")` to `_summarize_descriptions()` for summary LLM calls.

**`_summarize_descriptions()` (line ~323):** Add `token_tracker` parameter, forward to `use_llm_func_with_cache()`.

**`_merge_nodes_then_upsert()` (line ~1940) and `_merge_edges_then_upsert()` (line ~2500):** Add `token_tracker` parameter. Pass `token_tracker.get_stage("embedding")` to entity/relationship VDB `upsert()` calls.

### 3. Vector storage `upsert()` (base.py + all implementations)

**`BaseVectorStorage.upsert()` (line 293):** Add `token_tracker: Any | None = None` parameter.

**All implementations** (15 files): Pass `token_tracker=token_tracker` when calling `self.embedding_func(batch)`. The kwargs flow through `EmbeddingFunc.__call__()` -> `priority_limit_async_func_call` wrapper -> actual embedding function (e.g., `openai_embed`).

Implementations to update:
- `nano_vector_db_impl.py` (line 142)
- `faiss_impl.py` (line 128)
- `qdrant_impl.py` (line 655)
- `milvus_impl.py` (line 1486)
- `opensearch_impl.py` (lines 515, 2894)
- `postgres_impl.py` (lines 2610, 3485)
- `mongo_impl.py` (lines 211, 434, 2366)
- `redis_impl.py` (line 319)

### 4. Document processing orchestration (lightrag.py)

**`process_document()` (line 1886):** Create `DocumentTokenTracker()` at start. Pass stage-specific trackers to:
- `chunks_vdb.upsert(chunks, token_tracker=doc_tracker.get_stage("embedding"))`
- `_process_extract_entities(chunk, ..., token_tracker=doc_tracker)`

**`_process_extract_entities()` (line 2356):** Add `token_tracker` parameter, forward to `extract_entities()`.

**Final persistence:** After all stages complete, set `status_doc.token_usage = doc_tracker.get_usage()` before the final `doc_status.upsert()`.

### 5. Storage backend changes

**Doc status storage backends:** The `token_usage` field is `dict | None`, same serialization as the existing `metadata` field. All backends reconstruct via `DocProcessingStatus(**data)` so the new optional field with default `None` is backward-compatible. No storage-specific code changes needed.

**Postgres migration:** Add `token_usage JSONB NULL` column to the doc status table (follows same pattern as the existing `metadata JSONB NULL` column).

## Provider Coverage

| Provider | LLM tracking | Embedding tracking |
|----------|-------------|-------------------|
| OpenAI | Supported (openai.py:460, 612) | Supported (openai.py:844) |
| Gemini | Supported (gemini.py:376, 425) | Supported (gemini.py:591) |
| Others (Ollama, Bedrock, etc.) | Records zeros | Records zeros |

No changes needed in provider files. They already check `if token_tracker` before recording.

## API Exposure

No new endpoints needed. `DocProcessingStatus` is already serialized (via `dataclasses.asdict()`) in doc status API responses. The new `token_usage` field will appear automatically. For documents ingested before this feature, `token_usage` will be `null`.

## Scope Exclusions

- Token tracking during queries (only ingestion)
- Retroactive tracking for already-ingested documents
- Cost calculation or pricing logic
- New API endpoints (relies on existing doc status endpoints)
- Changes to LLM provider files (they already support `token_tracker`)
