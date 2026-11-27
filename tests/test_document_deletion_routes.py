import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from lightrag.base import DeletionResult, strict_storage_errors
from lightrag.kg import shared_storage

pytestmark = pytest.mark.offline


@pytest.fixture
def deletion_routes(monkeypatch):
    # API config reads argv when the document router is first imported.
    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    from lightrag.api.routers import document_routes

    namespaces = {"pipeline_status": {"history_messages": []}, "deletion_jobs": {}}
    lock = asyncio.Lock()

    async def get_namespace_data(namespace, **kwargs):
        return namespaces[namespace]

    async def allow_request():
        return None

    monkeypatch.setattr(shared_storage, "get_namespace_data", get_namespace_data)
    monkeypatch.setattr(shared_storage, "get_namespace_lock", lambda *args, **kw: lock)
    monkeypatch.setattr(
        document_routes, "get_combined_auth_dependency", lambda _: allow_request
    )
    monkeypatch.setattr(document_routes, "router", APIRouter(prefix="/documents"))

    rag = SimpleNamespace(
        workspace="test-deletion",
        adelete_by_doc_id=AsyncMock(),
        doc_status=SimpleNamespace(get_by_id=AsyncMock()),
    )
    app = FastAPI()
    app.include_router(document_routes.create_document_routes(rag, None))
    return TestClient(app), rag, namespaces


@pytest.mark.parametrize("second_fails", [True, False])
def test_deletion_job_reports_the_background_result(deletion_routes, second_fails):
    client, rag, namespaces = deletion_routes

    async def delete(doc_id, **kwargs):
        [job] = namespaces["deletion_jobs"].values()
        assert job["status"] == "running"
        status = "fail" if doc_id == "doc-b" and second_fails else "success"
        return DeletionResult(status=status, doc_id=doc_id, message="")

    rag.adelete_by_doc_id.side_effect = delete
    accepted = client.request(
        "DELETE", "/documents/delete_document", json={"doc_ids": ["doc-a", "doc-b"]}
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "deletion_started"

    job_id = accepted.json()["job_id"]
    result = client.get(f"/documents/deletion_jobs/{job_id}")
    assert result.status_code == 200
    assert result.headers["Cache-Control"] == "no-store"
    assert result.json() == {
        "status": "failed" if second_fails else "completed",
        "deleted_count": 1 if second_fails else 2,
        "failed_count": int(second_fails),
    }
    assert namespaces["pipeline_status"]["busy"] is False


def test_status_lookup_distinguishes_storage_failure_from_missing_document(
    deletion_routes,
):
    client, rag, _ = deletion_routes

    async def unavailable(doc_id):
        assert strict_storage_errors.get()
        raise RuntimeError("storage unavailable")

    rag.doc_status.get_by_id.side_effect = unavailable
    assert client.get("/documents/deletion_status/doc-a").status_code == 503

    rag.doc_status.get_by_id.side_effect = None
    rag.doc_status.get_by_id.return_value = None
    result = client.get("/documents/deletion_status/doc-a")
    assert result.status_code == 200
    assert result.json() == {"doc_id": "doc-a", "deleted": True}
