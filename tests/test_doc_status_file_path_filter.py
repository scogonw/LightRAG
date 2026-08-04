"""Offline coverage for the doc-status ``file_path_filter`` search.

Exercises JsonDocStatusStorage end to end (no external service needed) plus the
pure helpers and the request-model validator that every backend relies on.
"""

import sys
import tempfile

import pytest

sys.argv = sys.argv[:1]

from lightrag.base import DocStatus  # noqa: E402
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage  # noqa: E402
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.utils import (  # noqa: E402
    matches_file_path_filter,
    normalize_search_filter,
)

pytestmark = pytest.mark.offline

NOW = "2026-01-01T00:00:00"

# d4 carries SQL/OpenSearch metacharacters so escaping is covered by the same
# fixtures the plain substring cases use.
DOCS = {
    "d1": ("Quarterly_REPORT.pdf", DocStatus.PROCESSED),
    "d2": ("notes/report_draft.md", DocStatus.PENDING),
    "d3": ("invoice.txt", DocStatus.PROCESSED),
    "d4": ("100%_special_report.txt", DocStatus.FAILED),
}


@pytest.fixture
async def storage():
    initialize_share_data()
    working_dir = tempfile.mkdtemp()
    store = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": working_dir},
        embedding_func=None,
        workspace="test_ws",
    )
    await store.initialize()
    await store.upsert(
        {
            doc_id: {
                "content_summary": doc_id,
                "content_length": 1,
                "file_path": file_path,
                "status": status.value,
                "created_at": NOW,
                "updated_at": NOW,
            }
            for doc_id, (file_path, status) in DOCS.items()
        }
    )
    return store


async def _ids(store, **kwargs):
    rows, total = await store.get_docs_paginated(**kwargs)
    return sorted(doc_id for doc_id, _ in rows), total


# --- helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("", None), ("   ", None), ("\t\n", None), ("  a ", "a")],
)
def test_normalize_search_filter(value, expected):
    assert normalize_search_filter(value) == expected


@pytest.mark.parametrize(
    "file_path,term,expected",
    [
        ("Quarterly_REPORT.pdf", "report", True),
        ("Quarterly_REPORT.pdf", "REPORT", True),
        ("Quarterly_REPORT.pdf", "quarterly", True),
        ("invoice.txt", "report", False),
        (None, "report", False),
        ("", "report", False),
    ],
)
def test_matches_file_path_filter(file_path, term, expected):
    assert matches_file_path_filter(file_path, term) is expected


# --- listing ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", [None, "", "   "])
async def test_blank_filter_is_not_a_filter(storage, blank):
    """Blank input must list everything, not match-nothing or match-literally."""
    ids, total = await _ids(storage, file_path_filter=blank)
    assert ids == ["d1", "d2", "d3", "d4"]
    assert total == 4


@pytest.mark.asyncio
async def test_substring_match_is_case_insensitive(storage):
    lower, lower_total = await _ids(storage, file_path_filter="report")
    upper, upper_total = await _ids(storage, file_path_filter="REPORT")

    assert lower == ["d1", "d2", "d4"]
    assert lower_total == 3
    assert (upper, upper_total) == (lower, lower_total)


@pytest.mark.asyncio
async def test_matches_on_path_segment(storage):
    ids, total = await _ids(storage, file_path_filter="notes/")
    assert ids == ["d2"]
    assert total == 1


@pytest.mark.asyncio
async def test_wildcard_metacharacters_are_matched_literally(storage):
    """'%' and '_' must not behave as SQL LIKE wildcards."""
    ids, _ = await _ids(storage, file_path_filter="100%")
    assert ids == ["d4"]

    # A bare '%' would match every row if treated as a wildcard.
    ids, total = await _ids(storage, file_path_filter="%")
    assert ids == ["d4"]
    assert total == 1

    # '_' is a single-char wildcard in LIKE; here it must be a literal.
    ids, _ = await _ids(storage, file_path_filter="Quarterly_")
    assert ids == ["d1"]


@pytest.mark.asyncio
async def test_no_match_returns_empty_page_and_zero_total(storage):
    ids, total = await _ids(storage, file_path_filter="nomatch")
    assert ids == []
    assert total == 0


@pytest.mark.asyncio
async def test_combines_with_status_filter(storage):
    ids, total = await _ids(
        storage, status_filter=DocStatus.PROCESSED, file_path_filter="report"
    )
    assert ids == ["d1"]
    assert total == 1


@pytest.mark.asyncio
async def test_total_count_reflects_filter_not_page_size(storage):
    """total_count must describe the whole filtered set, not the returned slice."""
    rows, total = await storage.get_docs_paginated(
        file_path_filter="report", page=1, page_size=10
    )
    assert total == 3
    assert len(rows) == 3


# --- status counts ---------------------------------------------------------


@pytest.mark.asyncio
async def test_status_counts_reflect_the_filter(storage):
    """Tab counts must describe matching docs, else they contradict the listing."""
    counts = await storage.get_all_status_counts(file_path_filter="report")

    assert counts["all"] == 3
    assert counts["processed"] == 1
    assert counts["pending"] == 1
    assert counts["failed"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", [None, "", "   "])
async def test_status_counts_unfiltered(storage, blank):
    counts = await storage.get_all_status_counts(file_path_filter=blank)
    assert counts["all"] == 4
    assert counts["processed"] == 2


@pytest.mark.asyncio
async def test_status_counts_agree_with_listing_total(storage):
    for term in ["report", "invoice", "nomatch", "100%"]:
        _, total = await _ids(storage, file_path_filter=term)
        counts = await storage.get_all_status_counts(file_path_filter=term)
        assert counts["all"] == total, f"mismatch for {term!r}"


# --- request model ---------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("   ", None), ("  Report.PDF ", "Report.PDF")],
)
def test_documents_request_normalizes_filter(value, expected):
    from lightrag.api.routers.document_routes import DocumentsRequest

    assert DocumentsRequest(file_path_filter=value).file_path_filter == expected


def test_documents_request_defaults_to_no_filter():
    from lightrag.api.routers.document_routes import DocumentsRequest

    assert DocumentsRequest().file_path_filter is None
