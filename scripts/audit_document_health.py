"""Audit every document's chunk ownership and tenant metadata.

Chunk IDs are content hashes, so one chunk record can belong to several
documents — the same file uploaded into two knowledgebases lands on the same
records. That makes a document's health more than "does it exist": it can own
its chunks, share them with a sibling, or point at chunks that are gone.

Classifies every document in a workspace:

    healthy   owns chunks, and every chunk carries a tenant entry
    orphan    owns chunks, but no chunk carries a cascaded entry
              -> probably has no live upstream row; confirm before deleting
    shared    owns no chunks, but its chunks_list is fully alive
              -> benign duplicate. Its resource_id may still be the ONLY anchor
                 for one audience, so deleting it can silently revoke access
    inert     owns no chunks and its chunks_list is dead or partial
              -> content is gone; delete if the resource is dead, re-embed if live
    empty     declares no chunks at all

Also reports KV/vector divergence and cross-org entry contamination, and emits
SQL listing the resource_ids that need adjudicating — LightRAG can describe a
document's *shape*, only the system of record knows whether it is *live*.

The same audit is available in the WebUI (Documents → Health check) and over the
API at ``GET /documents/health``; the logic lives in
``lightrag.kg.opensearch_audit`` so all three agree.

Usage:
    python scripts/audit_document_health.py [--workspace WS] [--list CLASS]

Reads only — it does not modify anything.
"""

import argparse
import asyncio

# Load OPENSEARCH_* connection settings from .env, exactly like the server does.
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

from lightrag.kg.opensearch_audit import (
    CLASSES,
    audit_document_health,
)
from lightrag.kg.opensearch_impl import ClientManager, _build_index_name
from lightrag.namespace import NameSpace


def _print_report(report: dict, workspace: str) -> None:
    print(f"workspace                 : {workspace or '(none)'}")
    print(f"documents                 : {report['documents']}")
    print(
        f"chunks  kv / vector       : "
        f"{report['chunks']['kv']} / {report['chunks']['vector']}"
    )
    print()
    for klass in CLASSES:
        stats = report["classes"][klass]
        print(
            f"  {klass:<9} {stats['documents']:>5} docs  "
            f"{stats['chunks_owned']:>6} chunks owned"
        )
    print()
    integrity = report["integrity"]
    print(f"  KV/vector access mismatches : {integrity['access_mismatches']}")
    print(f"  chunks with no metadata     : {integrity['chunks_without_metadata']}")
    print(f"  cross-org entry leakage     : {integrity['cross_org_entries']}")

    actionable = report["actionable"]
    if actionable:
        print()
        print("Needs adjudicating upstream (orphan + inert):")
        for row in actionable:
            print(
                f"  {row['doc_id']}  {row['created_at']}  "
                f"owned={row['owned']:<4} {','.join(row['levels']):<13} "
                f"{str(row['file_path'])[:44]}"
            )
        resource_ids = report["resource_ids"]
        if resource_ids:
            print()
            print("-- A row means the document is LIVE: re-embed it, do not delete.")
            print("-- No row means the resource is gone: safe to delete.")
            print("SELECT id, title, is_deleted,")
            print("       metadata #>> '{lightrag_doc_id}' AS lightrag_doc_id")
            print("FROM resources WHERE id IN (")
            print(
                "\n".join(f"  '{r}'," for r in resource_ids[:-1])
                + f"\n  '{resource_ids[-1]}'"
            )
            print(");")

    listing = report.get("class_listing")
    if listing:
        print()
        print(f"All '{listing['klass']}' documents:")
        for row in listing["documents"]:
            print(
                f"  {row['doc_id']}  {row['created_at']}  owned={row['owned']:<4} "
                f"declared={row['declared']:<4} {str(row['file_path'])[:44]}  "
                f"{row['resource_id']}"
            )

    samples = report["divergent_samples"]
    if samples:
        print()
        print(
            f"KV/vector divergence on {integrity['access_mismatches']} chunk(s), "
            f"first {len(samples)}:"
        )
        for sample in samples:
            print(
                f"  {sample['chunk_id']}  kv={sample['kv']} vector={sample['vector']}"
            )
    print()
    print(
        "Note: 'shared' documents own no chunks but may still be the only "
        "anchor for one audience — check upstream before deleting them."
    )


async def main_async(workspace: str, list_class: str | None) -> None:
    client = await ClientManager.get_client()
    try:
        _, _, status_index = _build_index_name(workspace, NameSpace.DOC_STATUS)
        _, _, kv_index = _build_index_name(workspace, NameSpace.KV_STORE_TEXT_CHUNKS)
        _, _, vdb_index = _build_index_name(workspace, NameSpace.VECTOR_STORE_CHUNKS)
        _, _, full_index = _build_index_name(workspace, NameSpace.KV_STORE_FULL_DOCS)
        report = await audit_document_health(
            client,
            status_index=status_index,
            kv_index=kv_index,
            vdb_index=vdb_index,
            full_index=full_index,
            include_class=list_class,
        )
        _print_report(report, workspace)
    finally:
        await ClientManager.release_client(client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", default="", help="Workspace prefix (defaults to unprefixed)"
    )
    parser.add_argument(
        "--list",
        dest="list_class",
        choices=list(CLASSES),
        help="Print every document in one class",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.workspace, args.list_class))


if __name__ == "__main__":
    main()
