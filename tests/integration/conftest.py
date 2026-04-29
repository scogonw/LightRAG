"""Conftest for integration tests that need a real OpenSearch instance.

These tests are auto-skipped by the project's tests/conftest.py unless
pytest is invoked with --run-integration. The fixture in this file
additionally requires LIGHTRAG_RUN_INTEGRATION=true so that running
--run-integration without a configured OpenSearch cluster doesn't fail
unexpectedly.
"""

import os
import uuid

import pytest_asyncio


def _integration_disabled() -> bool:
    return os.getenv("LIGHTRAG_RUN_INTEGRATION", "").lower() != "true"


@pytest_asyncio.fixture
async def opensearch_rag():
    """Yield a fully-initialised LightRAG instance backed by OpenSearch.

    Workspace is randomised per test to avoid index-name collisions on
    parallel or repeated runs. Storages are dropped in teardown.
    """
    import pytest
    if _integration_disabled():
        pytest.skip(
            "set LIGHTRAG_RUN_INTEGRATION=true (and ensure OpenSearch is running) to run"
        )

    # Imports are inside the fixture so collection-time import doesn't fail
    # if OpenSearch deps aren't installed.
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete

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
        for store in (
            rag.doc_status,
            rag.chunks_vdb,
            rag.entities_vdb,
            rag.relationships_vdb,
            rag.full_docs,
            rag.text_chunks,
            rag.llm_response_cache,
        ):
            try:
                await store.drop()
            except Exception:
                pass
        await rag.finalize_storages()
