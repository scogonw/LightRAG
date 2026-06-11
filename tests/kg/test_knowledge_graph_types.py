"""Unit tests for KnowledgeGraph.filter_by_org is_truncated recomputation."""

from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode


def _node(nid: str, org: str) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(id=nid, labels=[nid], properties={"org_id": org})


def _edge(src: str, tgt: str, org: str) -> KnowledgeGraphEdge:
    eid = f"{src}-{tgt}"
    return KnowledgeGraphEdge(id=eid, type="rel", source=src, target=tgt, properties={"org_id": org})


class TestFilterByOrgIsTruncated:
    """filter_by_org must recompute is_truncated, not pass it through blindly."""

    def test_no_filter_no_truncation(self):
        """Single-org graph, no budget hit — is_truncated stays False."""
        kg = KnowledgeGraph(
            nodes=[_node("A", "org_X"), _node("B", "org_X")],
            edges=[_edge("A", "B", "org_X")],
            is_truncated=False,
        )
        result = kg.filter_by_org("org_X")
        assert {n.id for n in result.nodes} == {"A", "B"}
        assert result.is_truncated is False

    def test_budget_hit_single_org_preserves_truncated(self):
        """Budget was hit but no cross-org nodes shed — is_truncated preserved."""
        kg = KnowledgeGraph(
            nodes=[_node("A", "org_X"), _node("B", "org_X"), _node("C", "org_X")],
            edges=[_edge("A", "B", "org_X")],
            is_truncated=True,
        )
        result = kg.filter_by_org("org_X")
        assert {n.id for n in result.nodes} == {"A", "B", "C"}
        assert result.is_truncated is True

    def test_budget_hit_with_wrong_org_nodes_clears_truncated(self):
        """BFS hit budget, but filtering removed wrong-org nodes — is_truncated cleared."""
        kg = KnowledgeGraph(
            nodes=[_node("A", "org_X"), _node("B", "org_X"), _node("C", "org_Y")],
            edges=[_edge("A", "B", "org_X"), _edge("A", "C", "org_Y")],
            is_truncated=True,
        )
        result = kg.filter_by_org("org_X")
        assert {n.id for n in result.nodes} == {"A", "B"}
        assert "C" not in {n.id for n in result.nodes}
        # Pre-filter is_truncated was True, but nodes were shed → must be cleared
        assert result.is_truncated is False

    def test_edges_from_wrong_org_nodes_dropped(self):
        """Edges whose source or target was filtered out are removed."""
        kg = KnowledgeGraph(
            nodes=[_node("A", "org_X"), _node("B", "org_Y")],
            edges=[_edge("A", "B", "org_X")],
            is_truncated=False,
        )
        result = kg.filter_by_org("org_X")
        assert result.nodes == [kg.nodes[0]]
        assert result.edges == []

    def test_no_org_id_property_excluded(self):
        """Nodes without org_id property are excluded (fail-closed)."""
        no_org = KnowledgeGraphNode(id="Z", labels=["Z"], properties={})
        kg = KnowledgeGraph(
            nodes=[_node("A", "org_X"), no_org],
            edges=[],
            is_truncated=False,
        )
        result = kg.filter_by_org("org_X")
        assert {n.id for n in result.nodes} == {"A"}
