"""Diagnose why a metadata PATCH did (not) cascade to entities/relations.

Traces the full cascade chain for a single document against the live
OpenSearch cluster, printing exactly where it breaks:

    doc-status metadata + chunks_list
      -> graph nodes/edges located by source_ids  (the scan)
      -> derived ent-/rel- vector IDs             (the join)
      -> entity/relation vector records           (the vdb update target)

Usage:
    # Uses the same OPENSEARCH_* env vars / config.ini as the server.
    python scripts/diagnose_metadata_cascade.py <doc_id> [--workspace WS]

Reads only — it does not modify anything.
"""

import argparse
import asyncio
import json

# Load OPENSEARCH_* connection settings from .env, exactly like the server does
# (lightrag.api.config / lightrag_server both call load_dotenv at import time).
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

from lightrag.kg.opensearch_impl import ClientManager, _build_index_name  # noqa: E402
from lightrag.namespace import NameSpace  # noqa: E402
from lightrag.utils import compute_mdhash_id  # noqa: E402


def _summarize_metadata(meta):
    """Compact one-line description of a record's metadata field."""
    if meta is None:
        return "None"
    if isinstance(meta, dict):
        return f"dict {json.dumps(meta, sort_keys=True, default=str)}"
    if isinstance(meta, list):
        return f"list[{len(meta)}] {json.dumps(meta, sort_keys=True, default=str)}"
    return f"{type(meta).__name__} {meta!r}"


async def _scan_source_ids(client, index, chunk_ids, source_fields):
    """Return all hits in *index* whose source_ids match any of chunk_ids."""
    try:
        resp = await client.search(
            index=index,
            body={
                "query": {"terms": {"source_ids": chunk_ids}},
                "_source": source_fields,
                "size": 1000,
            },
        )
    except Exception as e:  # noqa: BLE001
        print(f"    !! search on {index} failed: {e}")
        return []
    return resp["hits"]["hits"]


async def main(doc_id: str, workspace: str) -> None:
    client = await ClientManager.get_client()
    try:
        _, _, doc_status_index = _build_index_name(workspace, NameSpace.DOC_STATUS)
        _, _, base_graph = _build_index_name(
            workspace, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
        )
        nodes_index = f"{base_graph}-nodes"
        edges_index = f"{base_graph}-edges"
        _, _, entities_index = _build_index_name(
            workspace, NameSpace.VECTOR_STORE_ENTITIES
        )
        _, _, relations_index = _build_index_name(
            workspace, NameSpace.VECTOR_STORE_RELATIONSHIPS
        )
        _, _, chunks_index = _build_index_name(
            workspace, NameSpace.VECTOR_STORE_CHUNKS
        )

        print(f"workspace          : {workspace!r}")
        print(f"doc_status index   : {doc_status_index}")
        print(f"nodes / edges      : {nodes_index} / {edges_index}")
        print(f"entities / rels vdb: {entities_index} / {relations_index}")
        print("=" * 72)

        # 1. doc-status
        ds = await client.mget(index=doc_status_index, body={"ids": [doc_id]})
        ds_doc = ds["docs"][0]
        if not ds_doc.get("found"):
            print(f"[1] doc-status: doc_id={doc_id} NOT FOUND. Wrong workspace?")
            return
        ds_src = ds_doc["_source"]
        chunk_ids = list(ds_src.get("chunks_list") or [])
        print(f"[1] doc-status metadata: {_summarize_metadata(ds_src.get('metadata'))}")
        print(f"    chunks_list: {len(chunk_ids)} chunk(s)")
        if not chunk_ids:
            print("    !! No chunks -> nothing to locate entities/relations from.")
            return

        # 1b. chunk metadata — confirm whether the chunk cascade actually landed.
        cv = await client.mget(index=chunks_index, body={"ids": chunk_ids})
        for d in cv["docs"][:5]:
            if d.get("found"):
                print(
                    f"    chunk {d['_id']!r}: "
                    f"{_summarize_metadata(d['_source'].get('metadata'))}"
                )
            else:
                print(f"    chunk {d['_id']!r}: NOT FOUND in {chunks_index}")

        # 2. graph nodes/edges located by source_ids (this is the cascade's scan)
        node_hits = await _scan_source_ids(client, nodes_index, chunk_ids, False)
        edge_hits = await _scan_source_ids(
            client, edges_index, chunk_ids, ["source_node_id", "target_node_id"]
        )
        print(
            f"[2] graph scan by source_ids: "
            f"{len(node_hits)} node(s), {len(edge_hits)} edge(s)"
        )
        if not node_hits and not edge_hits:
            print(
                "    !! BREAK: no nodes/edges matched source_ids. Either this doc's\n"
                "       entities carry source_ids that don't include these chunk_ids\n"
                "       (truncation / older ingest), or source_ids isn't populated."
            )

        # Show node metadata samples (read full _source for a few)
        sample_ids = [h["_id"] for h in node_hits[:5]]
        if sample_ids:
            node_full = await client.mget(
                index=nodes_index, body={"ids": sample_ids}
            )
            for d in node_full["docs"]:
                if d.get("found"):
                    print(
                        f"    node {d['_id']!r}: "
                        f"{_summarize_metadata(d['_source'].get('metadata'))}"
                    )

        # 3. derive vector IDs and check the vdb records
        entity_ids = [compute_mdhash_id(str(h["_id"]), prefix="ent-") for h in node_hits]
        rel_ids = set()
        for h in edge_hits:
            s = h["_source"].get("source_node_id")
            t = h["_source"].get("target_node_id")
            if s is None or t is None:
                continue
            rel_ids.add(compute_mdhash_id(f"{s}{t}", prefix="rel-"))
            rel_ids.add(compute_mdhash_id(f"{t}{s}", prefix="rel-"))

        print(
            f"[3] derived vector IDs: {len(entity_ids)} ent-, "
            f"{len(rel_ids)} rel- (both orientations)"
        )
        if entity_ids:
            ev = await client.mget(index=entities_index, body={"ids": entity_ids})
            found = [d for d in ev["docs"] if d.get("found")]
            print(f"    entities_vdb: {len(found)}/{len(entity_ids)} derived IDs exist")
            for d in found[:5]:
                print(
                    f"    ent {d['_id']!r}: "
                    f"{_summarize_metadata(d['_source'].get('metadata'))}"
                )
            if not found:
                print(
                    "    !! BREAK: derived ent- IDs don't exist in entities_vdb.\n"
                    "       Entity-name -> vdb-ID derivation mismatch."
                )
        if rel_ids:
            rv = await client.mget(index=relations_index, body={"ids": list(rel_ids)})
            found = [d for d in rv["docs"] if d.get("found")]
            print(
                f"    relationships_vdb: {len(found)}/{len(rel_ids)} derived IDs exist"
            )
            for d in found[:5]:
                print(
                    f"    rel {d['_id']!r}: "
                    f"{_summarize_metadata(d['_source'].get('metadata'))}"
                )
    finally:
        await ClientManager.release_client(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_id", help="The document id to trace")
    parser.add_argument(
        "--workspace",
        default="",
        help="Workspace (defaults to '' / OPENSEARCH_WORKSPACE env if set)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.doc_id, args.workspace))
