"""Integration-lite test: verify same-filename uploads are caught as duplicates
and same-content-different-filename uploads succeed, using the default JSON
doc-status backend (no external services).
"""
import tempfile
import uuid

import pytest

pytestmark = pytest.mark.offline


@pytest.fixture
def tmp_working_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


async def _build_rag(working_dir):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    import numpy as np

    async def fake_llm(prompt, **kwargs):
        return "stub"

    async def fake_embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    embed = EmbeddingFunc(embedding_dim=4, max_token_size=128, func=fake_embed)
    # Use a unique workspace per test so the shared in-memory namespace cache
    # (see lightrag.kg.shared_storage.get_namespace_data) does not leak state
    # between tests sharing the same default workspace.
    rag = LightRAG(
        working_dir=working_dir,
        workspace=f"test-{uuid.uuid4().hex}",
        llm_model_func=fake_llm,
        embedding_func=embed,
    )
    await rag.initialize_storages()
    return rag


async def _collect_all_doc_statuses(rag):
    from lightrag.base import DocStatus

    all_docs = {}
    for status in DocStatus:
        all_docs.update(await rag.doc_status.get_docs_by_status(status))
    return all_docs


async def test_same_filename_rejected_as_duplicate(tmp_working_dir):
    """Identical filename with different content should be flagged as a
    duplicate: one primary (doc-*) record, one duplicate (dup-*) record."""
    rag = await _build_rag(tmp_working_dir)
    try:
        await rag.apipeline_enqueue_documents(
            input="first content",
            file_paths="ABC.pdf",
        )
        await rag.apipeline_enqueue_documents(
            input="second content - different bytes",
            file_paths="ABC.pdf",
        )

        all_docs = await _collect_all_doc_statuses(rag)

        primary = [
            doc_id
            for doc_id, status in all_docs.items()
            if status.file_path == "ABC.pdf" and not doc_id.startswith("dup-")
        ]
        dup = [doc_id for doc_id in all_docs if doc_id.startswith("dup-")]
        assert len(primary) == 1, f"expected one active ABC.pdf record, got {primary}"
        assert len(dup) == 1, f"expected one dup-* record, got {dup}"
    finally:
        await rag.finalize_storages()


async def test_same_content_different_path_both_ingested(tmp_working_dir):
    """Identical content uploaded under two different filenames should yield
    two independent active records with no duplicate markers."""
    rag = await _build_rag(tmp_working_dir)
    try:
        await rag.apipeline_enqueue_documents(
            input="shared content",
            file_paths="A.pdf",
        )
        await rag.apipeline_enqueue_documents(
            input="shared content",
            file_paths="B.pdf",
        )

        all_docs = await _collect_all_doc_statuses(rag)

        active_paths = {
            status.file_path
            for doc_id, status in all_docs.items()
            if not doc_id.startswith("dup-")
        }
        assert active_paths == {"A.pdf", "B.pdf"}
    finally:
        await rag.finalize_storages()
