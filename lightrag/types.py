from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Optional


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: list[str]
    low_level_keywords: list[str]


class KnowledgeGraphNode(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraphEdge(BaseModel):
    id: str
    type: Optional[str]
    source: str  # id of source node
    target: str  # id of target node
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraph(BaseModel):
    nodes: list[KnowledgeGraphNode] = []
    edges: list[KnowledgeGraphEdge] = []
    is_truncated: bool = False

    def filter_by_org(self, org_id: str) -> "KnowledgeGraph":
        """
        Return a copy restricted to nodes/edges belonging to the given org.

        Used by graph storage backends as a uniform post-filter when the
        underlying query language can't easily push down a property-level
        org_id predicate. Edges are dropped if either endpoint was filtered
        out, or if the edge itself carries a non-matching ``org_id``.
        Nodes/edges with no ``org_id`` property are treated as not matching
        (fail-closed) so cross-tenant data cannot leak.
        """
        kept_nodes = [
            n for n in self.nodes if n.properties.get("org_id", "") == org_id
        ]
        kept_node_ids = {n.id for n in kept_nodes}
        kept_edges = [
            e
            for e in self.edges
            if e.source in kept_node_ids
            and e.target in kept_node_ids
            and e.properties.get("org_id", "") == org_id
        ]
        return KnowledgeGraph(
            nodes=kept_nodes, edges=kept_edges, is_truncated=self.is_truncated
        )
