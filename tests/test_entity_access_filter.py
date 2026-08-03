"""Offline unit tests for per-entry access control on entities and relations.

Entities and relations are merged across every document they were extracted
from, so their ``metadata`` is a LIST of per-document entries. The OpenSearch
knowledgebase filter evaluates that list under ``object`` mapping, where the
``knowledgebase_id`` and ``access_level`` clauses are matched against flattened
arrays and can therefore be satisfied by *different* entries. These tests pin
the Python-side per-entry re-check that closes that gap.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

from lightrag.operate import (
    _chunk_meta_matches_kb_filter,
    _filter_records_by_kb_access,
)

FIXTURE = Path(__file__).parent / "fixtures" / "entity_access_trace.json"

ORG = "org-1"
TEAM_KB = "drive-team"
USER_KB = "drive-user"
OTHER_KB = "drive-forbidden"

FILTER = {
    "user_id": "user-1",
    "agent_kb_ids": [],
    "user_kb_ids": [USER_KB],
    "team_kb_ids": [TEAM_KB],
}


def _entry(kb, access, resource="res-x"):
    return {
        "resource_id": resource,
        "knowledgebase_id": kb,
        "access_level": access,
    }


def test_cross_entry_match_is_rejected():
    """The exact shape the object-mapping bug lets through: the permitted
    knowledgebase_id comes from one entry and the permitted access_level from
    another, so neither entry on its own grants access."""
    record = {
        "entity_name": "E",
        "metadata": [
            _entry(TEAM_KB, "ONLY_ME"),  # right KB, wrong level for a team KB
            _entry(OTHER_KB, "TEAM_MEMBERS"),  # right level, forbidden KB
        ],
    }
    assert _filter_records_by_kb_access([record], FILTER, ORG) == []


def test_record_kept_when_one_entry_grants_access():
    record = {
        "entity_name": "E",
        "metadata": [_entry(OTHER_KB, "TEAM_MEMBERS"), _entry(TEAM_KB, "TEAM_MEMBERS")],
    }
    (kept,) = _filter_records_by_kb_access([record], FILTER, ORG)
    assert kept["entity_name"] == "E"


def test_inaccessible_entries_are_pruned_from_the_kept_record():
    """The kept record must not carry the resource_ids and knowledgebase_ids of
    documents the caller cannot access — those reach the API response."""
    allowed = _entry(TEAM_KB, "TEAM_MEMBERS", "res-ok")
    record = {
        "entity_name": "E",
        "metadata": [
            _entry(OTHER_KB, "TEAM_MEMBERS", "res-secret-1"),
            allowed,
            _entry(OTHER_KB, "ONLY_ME", "res-secret-2"),
        ],
    }
    (kept,) = _filter_records_by_kb_access([record], FILTER, ORG)
    assert kept["metadata"] == [allowed]


def test_input_record_is_not_mutated():
    record = {
        "entity_name": "E",
        "metadata": [_entry(OTHER_KB, "TEAM_MEMBERS"), _entry(TEAM_KB, "TEAM_MEMBERS")],
    }
    _filter_records_by_kb_access([record], FILTER, ORG)
    assert len(record["metadata"]) == 2


def test_user_owned_and_org_paths():
    user_owned = {"metadata": [_entry(USER_KB, "ONLY_ME")]}
    org_wide = {"metadata": [{**_entry(OTHER_KB, "ORGANIZATION"), "org_id": ORG}]}
    wrong_org = {"metadata": [{**_entry(OTHER_KB, "ORGANIZATION"), "org_id": "org-2"}]}

    assert len(_filter_records_by_kb_access([user_owned], FILTER, ORG)) == 1
    # Org-wide content is readable by anyone in the org, regardless of KB.
    assert len(_filter_records_by_kb_access([org_wide], FILTER, ORG)) == 1
    assert _filter_records_by_kb_access([wrong_org], FILTER, ORG) == []


def test_dict_metadata_is_preserved_as_dict():
    """Chunk-shaped (single-dict) metadata must not be turned into a list."""
    record = {"metadata": _entry(TEAM_KB, "TEAM_MEMBERS")}
    (kept,) = _filter_records_by_kb_access([record], FILTER, ORG)
    assert isinstance(kept["metadata"], dict)


def test_records_without_metadata_are_dropped():
    assert _filter_records_by_kb_access([{"entity_name": "E"}], FILTER, ORG) == []
    assert _filter_records_by_kb_access([{"metadata": None}], FILTER, ORG) == []
    assert _filter_records_by_kb_access([{"metadata": "junk"}], FILTER, ORG) == []


def test_agent_path_is_exclusive():
    """When agent_kb_ids is set, only CHAT_WIDGET entries in those KBs match."""
    agent_filter = {"agent_kb_ids": [TEAM_KB], "user_id": None}
    widget = {"metadata": [_entry(TEAM_KB, "CHAT_WIDGET")]}
    team = {"metadata": [_entry(TEAM_KB, "TEAM_MEMBERS")]}
    assert len(_filter_records_by_kb_access([widget], agent_filter, ORG)) == 1
    assert _filter_records_by_kb_access([team], agent_filter, ORG) == []


def test_matches_chunk_filter_semantics_entry_by_entry():
    """The record filter must agree with the chunk filter on every single entry,
    so entities and chunks cannot disagree about who may read a document."""
    entries = [
        _entry(TEAM_KB, "TEAM_MEMBERS"),
        _entry(TEAM_KB, "ONLY_ME"),
        _entry(USER_KB, "ONLY_ME"),
        _entry(OTHER_KB, "ORGANIZATION"),
        _entry(OTHER_KB, "TEAM_MEMBERS"),
    ]
    for entry in entries:
        expected = _chunk_meta_matches_kb_filter(entry, FILTER, ORG)
        got = bool(_filter_records_by_kb_access([{"metadata": [entry]}], FILTER, ORG))
        assert got == expected, entry


# ---------------------------------------------------------------------------
# Grounded in a captured (pseudonymised) query response
#
# The synthetic cases above pin the logic; this one pins it against the shape
# real data takes — entities merged across hundreds of documents, metadata lists
# well over a hundred entries long, spanning far more knowledge bases than any
# one caller can read.
# ---------------------------------------------------------------------------


def test_captured_response_entries_are_pruned():
    """Replays the entity metadata a two-knowledge-base user actually received.

    Every one of the 13 entities legitimately keeps at least one accessible
    entry, so none is dropped — that is correct, not a miss. What the filter
    removes is the 770 entries belonging to knowledge bases the user has no
    access to, which were being disclosed in the query response.
    """
    fixture = json.loads(FIXTURE.read_text())
    records = fixture["entities"]
    metadata_filter = fixture["metadata_filter"]
    org_id = fixture["org_id"]

    entries_before = sum(len(r["metadata"]) for r in records)
    assert len(records) == 13
    assert entries_before == 894

    kept = _filter_records_by_kb_access(records, metadata_filter, org_id)
    entries_after = sum(len(r["metadata"]) for r in kept)

    assert len(kept) == 13
    assert entries_after == 124
    assert entries_before - entries_after == 770

    allowed_kbs = set(metadata_filter["team_kb_ids"]) | set(
        metadata_filter["user_kb_ids"]
    )
    for record in kept:
        for entry in record["metadata"]:
            assert entry["knowledgebase_id"] in allowed_kbs, entry


# ---------------------------------------------------------------------------
# OpenSearchGraphStorage.get_edges_batch / edge_degrees_batch
#
# Without these overrides the base implementations run, and they accept
# metadata_filter and org_id then ignore both — so every relation fetched by
# pair reaches the caller with no access check at all.
# ---------------------------------------------------------------------------

opensearch_impl = pytest.importorskip("lightrag.kg.opensearch_impl")


class _FakeGraphClient:
    def __init__(self, hits=(), docs=()):
        self._hits = list(hits)
        self._docs = list(docs)
        self.search_bodies = []
        self.mget_bodies = []

    async def search(self, index, body):
        self.search_bodies.append(body)
        if "aggs" in body:
            return {
                "aggregations": {
                    "source_degrees": {"buckets": [{"key": "A", "doc_count": 2}]},
                    "target_degrees": {"buckets": [{"key": "B", "doc_count": 3}]},
                }
            }
        return {"hits": {"hits": self._hits}}

    async def mget(self, index, body):
        self.mget_bodies.append(body)
        return {"docs": self._docs}


def _make_graph(client):
    g = opensearch_impl.OpenSearchGraphStorage.__new__(
        opensearch_impl.OpenSearchGraphStorage
    )
    g.client = client
    g._nodes_index = "nodes"
    g._edges_index = "edges"
    g._indices_ready = True
    g.workspace = "ws"

    async def _noop(*args, **kwargs):
        return None

    g._refresh_graph_indices_if_dirty = _noop
    return g


def _edge_hit(src, tgt, metadata):
    edge_id = opensearch_impl.compute_mdhash_id(f"{src}-{tgt}", prefix="edge-")
    return {
        "_id": edge_id,
        "_source": {
            "source_node_id": src,
            "target_node_id": tgt,
            "weight": 1.0,
            "metadata": metadata,
        },
    }


async def test_get_edges_batch_applies_the_filter():
    client = _FakeGraphClient(hits=[_edge_hit("A", "B", [_entry(TEAM_KB, "TEAM_MEMBERS")])])
    graph = _make_graph(client)

    result = await graph.get_edges_batch(
        [{"src": "A", "tgt": "B"}], metadata_filter=FILTER, org_id=ORG
    )

    assert list(result) == [("A", "B")]
    # It went through search (filtered), not the unfiltered mget path.
    assert client.mget_bodies == []
    must = client.search_bodies[0]["query"]["bool"]["must"]
    assert any("ids" in clause for clause in must)
    assert len(must) == 2, "the knowledgebase filter clause is missing"


async def test_get_edges_batch_resolves_reverse_orientation():
    """Edges are bidirectional and stored under one orientation only; a request
    for (A, B) must still find an edge indexed as B-A."""
    client = _FakeGraphClient(hits=[_edge_hit("B", "A", [_entry(TEAM_KB, "TEAM_MEMBERS")])])
    graph = _make_graph(client)

    result = await graph.get_edges_batch(
        [{"src": "A", "tgt": "B"}], metadata_filter=FILTER, org_id=ORG
    )
    assert list(result) == [("A", "B")]


async def test_get_edges_batch_without_filter_uses_mget():
    src, tgt = "A", "B"
    edge_id = opensearch_impl.compute_mdhash_id(f"{src}-{tgt}", prefix="edge-")
    client = _FakeGraphClient(
        docs=[{"found": True, "_id": edge_id, "_source": {"weight": 1.0}}]
    )
    graph = _make_graph(client)

    result = await graph.get_edges_batch([{"src": src, "tgt": tgt}])
    assert list(result) == [(src, tgt)]
    assert client.search_bodies == []


async def test_get_edges_batch_empty_input():
    graph = _make_graph(_FakeGraphClient())
    assert await graph.get_edges_batch([]) == {}


async def test_edge_degrees_batch_sums_endpoint_degrees():
    """Matches BaseGraphStorage.edge_degree: source degree + target degree."""
    graph = _make_graph(_FakeGraphClient())
    result = await graph.edge_degrees_batch(
        [("A", "B")], metadata_filter=FILTER, org_id=ORG
    )
    assert result == {("A", "B"): 5}  # 2 + 3 from the aggregation buckets


async def test_edge_degrees_batch_passes_the_filter_through():
    client = _FakeGraphClient()
    graph = _make_graph(client)
    await graph.edge_degrees_batch([("A", "B")], metadata_filter=FILTER, org_id=ORG)
    query = client.search_bodies[0]["query"]
    assert "must" in query["bool"], "filter was dropped; degrees would count unreadable edges"


def test_opensearch_overrides_the_unfiltered_base_methods():
    base = opensearch_impl.BaseGraphStorage
    impl = opensearch_impl.OpenSearchGraphStorage
    assert impl.get_edges_batch is not base.get_edges_batch
    assert impl.edge_degrees_batch is not base.edge_degrees_batch


# ---------------------------------------------------------------------------
# aquery_data must carry org_id onto the QueryParam it builds
#
# Without it the org condition is never added to the knowledgebase filter, and
# a /query/data request with no metadata_filter produces no filter at all —
# i.e. cross-org results. The omission is a single missing keyword in a large
# literal, so pin it structurally.
# ---------------------------------------------------------------------------


def test_aquery_data_copies_access_control_fields():
    import ast
    import inspect

    from lightrag.lightrag import LightRAG

    tree = ast.parse(inspect.getsource(LightRAG.aquery_data).lstrip())
    # The QueryParam copy assigned to data_param — not the signature default.
    calls = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "QueryParam"
        and any(isinstance(t, ast.Name) and t.id == "data_param" for t in node.targets)
    ]
    assert calls, "aquery_data no longer assigns a QueryParam to data_param"

    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        for field in ("metadata_filter", "org_id"):
            assert field in passed, (
                f"aquery_data drops {field!r} when copying QueryParam, "
                f"which disables access-control filtering on /query/data"
            )
