"""Offline unit tests for the document-health classifier.

The classifier decides which documents an operator is asked to look at, so a
silent change in its meaning is expensive: ``orphan`` once approximated "no live
upstream row" purely because the metadata cascade was the only writer that
stamped ``org_id``, and the cascade only ran for a resource that existed
upstream. Ingestion stamps it too now, so these tests pin what each class does
and does not claim.
"""

import pytest

from lightrag.kg.opensearch_audit import ACTIONABLE_CLASSES, CLASSES, classify
from lightrag.utils import build_metadata_entry, normalize_metadata_entries

pytestmark = pytest.mark.offline


def _marked(entry: dict) -> bool:
    """Reproduce the audit's marking rule for a single-entry record."""
    return any(e.get("org_id") for e in normalize_metadata_entries(entry))


# ---------------------------------------------------------------------------
# The ingest fix and the audit must agree. If ingestion stamps org_id but the
# audit still reads that as "a cascade ran", every new document is misfiled.
# ---------------------------------------------------------------------------


def test_freshly_ingested_document_is_healthy_not_orphan():
    entry = build_metadata_entry(
        {"resource_id": "r1", "knowledgebase_id": "kb-1"}, "org-1"
    )
    assert _marked(entry), "ingest entry must carry org_id for the audit to accept it"
    assert (
        classify(owned=3, declared=3, chunk_list=["a"], alive_ids={"a"}, marked=True)
        == "healthy"
    )


def test_legacy_entry_without_org_id_is_still_orphan():
    """Documents ingested before the fix keep landing in orphan until backfilled."""
    entry = build_metadata_entry({"resource_id": "r1"}, None)
    assert not _marked(entry)
    assert (
        classify(owned=3, declared=3, chunk_list=["a"], alive_ids={"a"}, marked=False)
        == "orphan"
    )


# ---------------------------------------------------------------------------
# inert is the class that still carries a real signal — it keys on chunks being
# gone, not on metadata shape, so the ingest change must not disturb it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marked", [True, False])
def test_inert_is_independent_of_metadata_shape(marked):
    assert (
        classify(
            owned=0, declared=2, chunk_list=["a", "b"], alive_ids=set(), marked=marked
        )
        == "inert"
    )


@pytest.mark.parametrize("marked", [True, False])
def test_partially_dead_chunk_list_is_inert(marked):
    assert (
        classify(
            owned=0, declared=2, chunk_list=["a", "b"], alive_ids={"a"}, marked=marked
        )
        == "inert"
    )


@pytest.mark.parametrize("marked", [True, False])
def test_shared_is_independent_of_metadata_shape(marked):
    """Owns nothing, but every chunk it lists is alive: a sibling owns them."""
    assert (
        classify(
            owned=0,
            declared=2,
            chunk_list=["a", "b"],
            alive_ids={"a", "b"},
            marked=marked,
        )
        == "shared"
    )


@pytest.mark.parametrize("marked", [True, False])
def test_empty_declares_nothing(marked):
    assert (
        classify(owned=0, declared=0, chunk_list=[], alive_ids=set(), marked=marked)
        == "empty"
    )


def test_owning_chunks_always_beats_the_chunk_list_check():
    """A document that owns chunks is never shared/inert, whatever its list says."""
    assert (
        classify(
            owned=1, declared=2, chunk_list=["a", "b"], alive_ids=set(), marked=True
        )
        == "healthy"
    )


def test_actionable_classes_are_real_classes():
    assert set(ACTIONABLE_CLASSES) <= set(CLASSES)
