"""Offline unit tests for the update-document-metadata route's helpers."""

import sys

sys.argv = sys.argv[:1]

from unittest.mock import MagicMock  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from lightrag.api.routers.document_routes import (  # noqa: E402
    _shallow_merge_metadata,
    create_document_routes,
)


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
