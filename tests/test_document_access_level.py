"""Offline coverage for the document listing's ``access_level`` column.

``access_level`` is not a doc-status field: ingestion never writes it there and
every status transition overwrites ``doc_status.metadata`` with its own
processing timestamps. ``full_docs`` is the authoritative store, so the paginated
listing joins it in. These tests cover the two pieces that join relies on — the
``BaseKVStorage.get_metadata_batch`` default (exercised through JsonKVStorage, no
external service needed) and the ``_extract_access_level`` reader — plus the
response model's default.
"""

import sys
import tempfile

import pytest

sys.argv = sys.argv[:1]

from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from fastapi import APIRouter, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from lightrag.api.routers import document_routes  # noqa: E402
from lightrag.api.routers.document_routes import (  # noqa: E402
    DocStatusResponse,
    _extract_access_level,
    create_document_routes,
)
from lightrag.base import DocProcessingStatus, DocStatus  # noqa: E402
from lightrag.kg.json_kv_impl import JsonKVStorage  # noqa: E402
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402

pytestmark = pytest.mark.offline

NOW = "2026-01-01T00:00:00"

# d3 has a record but no metadata at all; d4's metadata omits access_level.
# Both must be distinguishable from "no record" (d_missing).
DOCS = {
    "d1": {
        "content": "first",
        "metadata": {"access_level": "ORGANIZATION", "knowledgebase_id": "kb-1"},
    },
    "d2": {"content": "second", "metadata": {"access_level": "ONLY_ME"}},
    "d3": {"content": "third"},
    "d4": {"content": "fourth", "metadata": {"knowledgebase_id": "kb-2"}},
}


@pytest.fixture
async def full_docs():
    initialize_share_data()
    store = JsonKVStorage(
        namespace="full_docs",
        global_config={"working_dir": tempfile.mkdtemp()},
        embedding_func=None,
        workspace="test_ws",
    )
    await store.initialize()
    await store.upsert(DOCS)
    return store


# --- _extract_access_level -------------------------------------------------


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"access_level": "ORGANIZATION"}, "ORGANIZATION"),
        ({"access_level": "ONLY_ME", "knowledgebase_id": "kb-1"}, "ONLY_ME"),
        (None, None),
        ({}, None),
        ({"knowledgebase_id": "kb-1"}, None),
        # Metadata is client-supplied and unvalidated: anything that is not a
        # non-empty string is reported as absent rather than rendered.
        ({"access_level": ""}, None),
        ({"access_level": None}, None),
        ({"access_level": 42}, None),
        ({"access_level": ["ORGANIZATION"]}, None),
        ({"access_level": {"value": "ORGANIZATION"}}, None),
        ({"access_level": True}, None),
    ],
)
def test_extract_access_level(metadata, expected):
    assert _extract_access_level(metadata) == expected


def test_extract_access_level_does_not_normalize_unknown_values():
    """Unknown values pass through — the API reports what is stored rather than
    silently hiding a misconfigured document from an operator."""
    assert _extract_access_level({"access_level": "TYPO_LEVEL"}) == "TYPO_LEVEL"


# --- get_metadata_batch default -------------------------------------------


async def test_get_metadata_batch_returns_metadata_by_id(full_docs):
    result = await full_docs.get_metadata_batch(["d1", "d2"])
    assert result == {
        "d1": {"access_level": "ORGANIZATION", "knowledgebase_id": "kb-1"},
        "d2": {"access_level": "ONLY_ME"},
    }


async def test_get_metadata_batch_omits_absent_records(full_docs):
    result = await full_docs.get_metadata_batch(["d1", "d_missing"])
    assert "d_missing" not in result
    assert set(result) == {"d1"}


async def test_get_metadata_batch_maps_record_without_metadata_to_empty_dict(full_docs):
    """A record that exists but carries no metadata must still be a key, so
    callers can tell "no metadata" apart from "no record"."""
    result = await full_docs.get_metadata_batch(["d3"])
    assert result == {"d3": {}}


async def test_get_metadata_batch_empty_ids_skips_lookup(full_docs):
    assert await full_docs.get_metadata_batch([]) == {}


async def test_get_metadata_batch_feeds_extract_access_level(full_docs):
    """The two halves of the join, composed the way the endpoint composes them."""
    ids = ["d1", "d2", "d3", "d4", "d_missing"]
    by_id = await full_docs.get_metadata_batch(ids)
    assert [_extract_access_level(by_id.get(i)) for i in ids] == [
        "ORGANIZATION",
        "ONLY_ME",
        None,
        None,
        None,
    ]


# --- response model --------------------------------------------------------


def _doc_status_response(**overrides):
    fields = {
        "id": "doc-1",
        "content_summary": "summary",
        "content_length": 5,
        "status": "processed",
        "created_at": NOW,
        "updated_at": NOW,
        "file_path": "report.pdf",
    }
    fields.update(overrides)
    return DocStatusResponse(**fields)


def test_doc_status_response_access_level_defaults_to_none():
    """The three non-paginated construction sites omit the field entirely, so it
    must be optional rather than required."""
    assert _doc_status_response().access_level is None


def test_doc_status_response_serializes_access_level():
    response = _doc_status_response(access_level="TEAM_MEMBERS")
    assert response.model_dump()["access_level"] == "TEAM_MEMBERS"


# --- POST /documents/paginated --------------------------------------------


class _StubDocStatus:
    """Stands in for a doc-status backend serving one page of documents."""

    def __init__(self, docs):
        self.docs = docs

    async def get_docs_paginated(self, **kwargs):
        return list(self.docs.items()), len(self.docs)

    async def get_all_status_counts(self, **kwargs):
        return {"all": len(self.docs)}


class _StubFullDocs:
    """Stands in for full_docs. ``failure`` reproduces a backend that raises."""

    def __init__(self, metadata_by_id, failure=None):
        self.metadata_by_id = metadata_by_id
        self.failure = failure
        self.seen_ids = None

    async def get_metadata_batch(self, ids):
        self.seen_ids = list(ids)
        if self.failure is not None:
            raise self.failure
        return {i: self.metadata_by_id[i] for i in ids if i in self.metadata_by_id}


def _status(doc_id):
    return DocProcessingStatus(
        content_summary=doc_id,
        content_length=1,
        file_path=f"{doc_id}.pdf",
        status=DocStatus.PROCESSED,
        created_at=NOW,
        updated_at=NOW,
        org_id="org-test",
        # What the pipeline actually leaves behind: its own bookkeeping only,
        # never the tenant metadata the column needs.
        metadata={"processing_start_time": 1, "processing_end_time": 2},
    )


def _paginated_client(monkeypatch, metadata_by_id, failure=None):
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")

    # create_document_routes registers on a module-level router, so a shared
    # router would let an earlier test's rag answer this test's requests.
    monkeypatch.setattr(
        document_routes,
        "router",
        APIRouter(prefix="/documents", tags=["documents"]),
    )

    full_docs = _StubFullDocs(metadata_by_id, failure=failure)
    rag = SimpleNamespace(
        workspace="test-ws",
        doc_status=_StubDocStatus({d: _status(d) for d in ("d1", "d2", "d3")}),
        full_docs=full_docs,
    )

    app = FastAPI()
    app.include_router(create_document_routes(rag, MagicMock(), None))
    return TestClient(app), full_docs


def _post_page(client):
    response = client.post(
        "/documents/paginated",
        json={
            "page": 1,
            "page_size": 10,
            "sort_field": "updated_at",
            "sort_direction": "desc",
        },
    )
    assert response.status_code == 200, response.text
    return {d["id"]: d["access_level"] for d in response.json()["documents"]}


def test_paginated_populates_access_level_from_full_docs(monkeypatch):
    client, full_docs = _paginated_client(
        monkeypatch,
        {
            "d1": {"access_level": "ORGANIZATION", "resource_id": "r1"},
            "d2": {"access_level": "ONLY_ME"},
            # d3 has metadata but no access_level.
            "d3": {"knowledgebase_id": "kb-1"},
        },
    )

    assert _post_page(client) == {
        "d1": "ORGANIZATION",
        "d2": "ONLY_ME",
        "d3": None,
    }
    # Exactly one batched lookup, scoped to the page's documents.
    assert full_docs.seen_ids == ["d1", "d2", "d3"]


def test_paginated_reports_none_when_full_docs_record_missing(monkeypatch):
    client, _ = _paginated_client(monkeypatch, {"d1": {"access_level": "ONLY_ME"}})

    assert _post_page(client) == {"d1": "ONLY_ME", "d2": None, "d3": None}


def test_paginated_survives_metadata_lookup_failure(monkeypatch):
    """A full_docs outage must degrade the column, not 500 the whole listing."""
    client, _ = _paginated_client(
        monkeypatch, {}, failure=RuntimeError("opensearch unavailable")
    )

    assert _post_page(client) == {"d1": None, "d2": None, "d3": None}
