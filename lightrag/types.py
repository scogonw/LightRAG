from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Optional


class ExtractedEntity(BaseModel):
    """A single entity extracted from text by the LLM."""

    entity_name: str = Field(
        description="Name of the entity. Use title case for case-insensitive names."
    )
    entity_type: str = Field(description="Type/category of the entity.")
    entity_description: str = Field(
        description="Concise yet comprehensive description of the entity based on the input text."
    )


class ExtractedRelationship(BaseModel):
    """A single relationship between two entities extracted from text."""

    source_entity: str = Field(
        description="Name of the source entity in the relationship."
    )
    target_entity: str = Field(
        description="Name of the target entity in the relationship."
    )
    relationship_keywords: str = Field(
        description="Comma-separated high-level keywords summarizing the relationship."
    )
    relationship_description: str = Field(
        description="Concise explanation of the relationship between source and target entities."
    )


class EntityExtractionResult(BaseModel):
    """Structured output format for entity and relationship extraction from text."""

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="List of entities extracted from the input text.",
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list,
        description="List of relationships between entities extracted from the input text.",
    )


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
        # Preserve is_truncated only when no nodes were shed by the filter.
        # If filtering removed nodes the BFS budget was consumed partly by wrong-org
        # nodes, so the pre-filter flag is unreliable and must be cleared.
        is_truncated = self.is_truncated and len(kept_nodes) == len(self.nodes)
        return KnowledgeGraph(
            nodes=kept_nodes, edges=kept_edges, is_truncated=is_truncated
        )
