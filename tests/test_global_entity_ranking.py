"""Tests for global-mode entity ranking in _find_most_related_entities_from_relationships.

Acceptance criteria for SCO-54:
- Entities are sorted by max introducing-edge similarity (highest first).
- Full set of entities is unchanged — only order changes.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from lightrag.base import QueryParam
from lightrag.operate import _find_most_related_entities_from_relationships


def _make_graph(node_data: dict):
    """Return a mock graph whose get_nodes_batch resolves from the given dict."""
    graph = MagicMock()

    async def _get_nodes_batch(node_ids, metadata_filter=None, org_id=None):
        return {nid: node_data[nid] for nid in node_ids if nid in node_data}

    graph.get_nodes_batch = AsyncMock(side_effect=_get_nodes_batch)
    return graph


def _edge(src, tgt):
    return {"src_id": src, "tgt_id": tgt}


def _nodes(*names):
    return {n: {"entity_name": n, "description": f"desc-{n}"} for n in names}


@pytest.mark.offline
class TestGlobalEntityRankingOrder:
    """Entities from higher-similarity edges must appear before lower-similarity ones."""

    async def test_entity_from_higher_rank_edge_precedes_lower_rank(self):
        """Position 0 in edge_datas has the highest cosine similarity.
        Both entities from that edge must appear before entities from position 1."""
        # edge_datas is sorted by VDB cosine similarity: position 0 = highest
        edge_datas = [_edge("A", "B"), _edge("C", "D")]
        query_param = QueryParam(mode="global", top_k=10)
        graph = _make_graph(_nodes("A", "B", "C", "D"))

        result = await _find_most_related_entities_from_relationships(
            edge_datas, query_param, graph
        )

        names = [n["entity_name"] for n in result]
        # A and B (vdb_score=1.0, position 0) must both precede C and D (vdb_score=0.9, position 1)
        assert set(names[:2]) == {"A", "B"}, f"Top-2 were {names[:2]!r}, expected A and B"
        assert set(names[2:]) == {"C", "D"}, f"Bottom-2 were {names[2:]!r}, expected C and D"

    async def test_three_edges_ordering(self):
        """Verify correct descending order across three distinct similarity levels."""
        edge_datas = [_edge("Hi", "X"), _edge("Mid", "X"), _edge("Lo", "X")]
        query_param = QueryParam(mode="global", top_k=10)
        # X is shared; Hi/Mid/Lo are unique to their positions
        graph = _make_graph(_nodes("Hi", "Mid", "Lo", "X"))

        result = await _find_most_related_entities_from_relationships(
            edge_datas, query_param, graph
        )

        names = [n["entity_name"] for n in result]
        # Hi (score 1.0) must appear before Mid (0.9) which must appear before Lo (0.8)
        assert names.index("Hi") < names.index("Mid") < names.index("Lo")


@pytest.mark.offline
class TestGlobalEntityRankingCompleteness:
    """The set of returned entities must equal the set of unique src/tgt across all edges."""

    async def test_all_entities_returned(self):
        edge_datas = [_edge("X", "Y"), _edge("Y", "Z"), _edge("P", "Q")]
        query_param = QueryParam(mode="global", top_k=10)
        all_names = {"X", "Y", "Z", "P", "Q"}
        graph = _make_graph(_nodes(*all_names))

        result = await _find_most_related_entities_from_relationships(
            edge_datas, query_param, graph
        )

        assert {n["entity_name"] for n in result} == all_names

    async def test_empty_edge_datas_returns_empty(self):
        graph = _make_graph({})
        result = await _find_most_related_entities_from_relationships(
            [], QueryParam(mode="global", top_k=10), graph
        )
        assert result == []

    async def test_missing_node_excluded_silently(self):
        """Nodes absent from the graph are silently excluded; present nodes still appear."""
        edge_datas = [_edge("Known", "Missing")]
        graph = _make_graph(_nodes("Known"))

        result = await _find_most_related_entities_from_relationships(
            edge_datas, QueryParam(mode="global", top_k=10), graph
        )

        assert [n["entity_name"] for n in result] == ["Known"]


@pytest.mark.offline
class TestGlobalEntityRankingSharedEntities:
    """An entity introduced by multiple edges keeps the score of its highest-ranked edge."""

    async def test_shared_entity_ranked_by_max_introducing_edge(self):
        """Entity B appears in both edge 0 (score 1.0) and edge 1 (score 0.9).
        It should keep score 1.0 and rank ahead of C (score 0.9 only)."""
        # Edge 0 (position 0, score=1.0): A→B
        # Edge 1 (position 1, score=0.9): B→C
        # B was introduced at position 0, so max_score(B) = 1.0 > score at position 1
        edge_datas = [_edge("A", "B"), _edge("B", "C")]
        query_param = QueryParam(mode="global", top_k=10)
        graph = _make_graph(_nodes("A", "B", "C"))

        result = await _find_most_related_entities_from_relationships(
            edge_datas, query_param, graph
        )

        names = [n["entity_name"] for n in result]
        # C (only introduced at position 1) must come after A and B (both at position 0)
        assert names[-1] == "C", f"Expected C last, got {names}"
        assert set(names) == {"A", "B", "C"}

    async def test_entity_score_not_downgraded_by_later_lower_edge(self):
        """Re-encountering an entity at a lower-ranked edge must not reduce its score."""
        edge_datas = [_edge("Anchor", "Shared"), _edge("Shared", "Tail")]
        query_param = QueryParam(mode="global", top_k=20)
        graph = _make_graph(_nodes("Anchor", "Shared", "Tail"))

        result = await _find_most_related_entities_from_relationships(
            edge_datas, query_param, graph
        )

        names = [n["entity_name"] for n in result]
        # Shared introduced at position 0 (same as Anchor); Tail only at position 1
        # Both Anchor and Shared must precede Tail
        assert names.index("Tail") > names.index("Anchor")
        assert names.index("Tail") > names.index("Shared")
