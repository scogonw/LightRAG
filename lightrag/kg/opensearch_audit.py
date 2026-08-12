"""Document-health audit shared by the CLI script and the API route.

Chunk IDs are content hashes, so one chunk record can belong to several
documents — the same file uploaded into two knowledgebases lands on the same
records, while ``full_doc_id`` can only name one owner. "The document exists"
therefore says very little: it may own its chunks, share them with a sibling, or
point at chunks that are gone.

This classifies every document by that distinction, and reports the integrity
invariants the access-control work depends on. It deliberately does **not**
decide whether a document is live — LightRAG knows a document's *shape*, only
the upstream system of record knows whether its resource still exists. Callers
get the ``resource_id`` so they can adjudicate that themselves.

That caveat now carries real weight. ``orphan`` used to double as a rough
liveness hint, because only the metadata cascade stamped ``org_id`` onto an
entry and the cascade only ran for a resource that existed upstream. Ingestion
stamps it too now, so the hint is gone by design: **adjudicate ``resource_id``
upstream, in both directions, for every document you intend to act on.** The CLI
grew ``--adjudicate-all`` for exactly that. ``inert`` is untouched and remains a
real signal.

Read-only.
"""

from collections import defaultdict
from typing import Any

from lightrag.utils import normalize_metadata_entries

_SCROLL = "3m"
_PAGE = 2000

#: Documents that own no chunks and whose chunks_list is dead, or that own chunks
#: whose entries lack org_id. A starting point for adjudication, not a verdict —
#: see the module docstring on why ``orphan`` no longer implies "probably dead".
ACTIONABLE_CLASSES = ("orphan", "inert")

CLASSES = ("healthy", "orphan", "shared", "inert", "empty")


async def _scroll(client, index, source):
    """Yield every hit in an index, paging with the scroll API."""
    resp = await client.search(
        index=index, scroll=_SCROLL, body={"size": _PAGE, "_source": source}
    )
    scroll_id = resp["_scroll_id"]
    while resp["hits"]["hits"]:
        for hit in resp["hits"]["hits"]:
            yield hit
        resp = await client.scroll(scroll_id=scroll_id, scroll=_SCROLL)
        scroll_id = resp["_scroll_id"]


def classify(owned: int, declared: int, chunk_list: list, alive_ids: set, marked: bool) -> str:
    """Bucket one document.

    ``healthy`` owns chunks that carry a tenant entry. ``orphan`` owns chunks
    whose entries carry no ``org_id``. ``shared`` owns nothing yet every chunk it
    lists is alive, so a sibling owns them: benign, and its resource may still be
    the only anchor for one audience. ``inert`` owns nothing and its chunks are
    gone.

    ``orphan`` is **not** a liveness signal. It once approximated one, because
    only the cascade stamped ``org_id`` and the cascade only ran for a resource
    that existed upstream. Ingestion now stamps it too, so ``orphan`` means
    "ingested before that change and never cascaded since" — a shrinking legacy
    set, not evidence a document is dead. Decide liveness by adjudicating
    ``resource_id`` upstream in both directions; ``--adjudicate-all`` on the CLI
    emits the SQL for every document, not just the actionable ones.

    ``inert`` is unaffected by any of this: it keys on chunks being gone, so it
    remains the live detector for zero-chunk dead documents.
    """
    if declared == 0 and owned == 0:
        return "empty"
    if owned > 0:
        return "healthy" if marked else "orphan"
    if chunk_list and all(cid in alive_ids for cid in chunk_list):
        return "shared"
    return "inert"


async def audit_document_health(
    client,
    *,
    status_index: str,
    kv_index: str,
    vdb_index: str,
    full_index: str,
    include_class: str | None = None,
) -> dict[str, Any]:
    """Classify every document and check the storage integrity invariants.

    ``include_class`` additionally returns the full listing for one class; the
    default response carries only the actionable ones, because ``healthy`` can
    run to thousands of rows.
    """
    docs: dict[str, dict[str, Any]] = {}
    async for hit in _scroll(
        client, status_index, ["file_path", "chunks_count", "chunks_list", "created_at"]
    ):
        src = hit["_source"]
        docs[hit["_id"]] = {
            "file_path": src.get("file_path"),
            "declared": src.get("chunks_count") or 0,
            "chunks_list": src.get("chunks_list") or [],
            "created_at": str(src.get("created_at"))[:10],
        }

    owned: dict[str, int] = defaultdict(int)
    marked: dict[str, bool] = defaultdict(bool)
    levels: dict[str, set] = defaultdict(set)
    alive_ids: set[str] = set()
    kv_levels: dict[str, frozenset] = {}
    contaminated: list[str] = []
    no_metadata: list[str] = []

    async for hit in _scroll(client, kv_index, ["full_doc_id", "metadata", "org_id"]):
        src = hit["_source"]
        alive_ids.add(hit["_id"])
        doc_id = src.get("full_doc_id")
        owned[doc_id] += 1
        entries = normalize_metadata_entries(src.get("metadata"))
        if not entries:
            no_metadata.append(hit["_id"])
        record_org = src.get("org_id")
        kv_levels[hit["_id"]] = frozenset(
            str(e.get("access_level")) for e in entries
        )
        for entry in entries:
            levels[doc_id].add(entry.get("access_level"))
            entry_org = entry.get("org_id")
            if entry_org:
                # Both writers now stamp org_id: ingestion via
                # utils.build_metadata_entry, and the metadata cascade. So this
                # marks "the entry has a complete shape", NOT "a cascade ran" and
                # therefore NOT "a live upstream row exists" — see classify().
                marked[doc_id] = True
                if record_org and entry_org != record_org:
                    contaminated.append(hit["_id"])

    vdb_levels: dict[str, frozenset] = {}
    async for hit in _scroll(client, vdb_index, ["metadata"]):
        entries = normalize_metadata_entries(hit["_source"].get("metadata"))
        vdb_levels[hit["_id"]] = frozenset(
            str(e.get("access_level")) for e in entries
        )

    anchors: dict[str, str | None] = {}
    async for hit in _scroll(client, full_index, ["metadata"]):
        meta = hit["_source"].get("metadata") or {}
        anchors[hit["_id"]] = meta.get("resource_id")

    buckets: dict[str, list] = defaultdict(list)
    for doc_id, info in docs.items():
        klass = classify(
            owned.get(doc_id, 0),
            info["declared"],
            info["chunks_list"],
            alive_ids,
            marked.get(doc_id, False),
        )
        row = {
            "doc_id": doc_id,
            "file_path": info["file_path"],
            "created_at": info["created_at"],
            "declared": info["declared"],
            "owned": owned.get(doc_id, 0),
            "levels": sorted(str(x) for x in levels.get(doc_id, [])),
            "resource_id": anchors.get(doc_id),
            "klass": klass,
        }
        buckets[klass].append(row)

    common = set(kv_levels) & set(vdb_levels)
    divergent = [c for c in common if kv_levels[c] != vdb_levels[c]]

    actionable = sorted(
        (row for klass in ACTIONABLE_CLASSES for row in buckets.get(klass, [])),
        key=lambda r: -r["owned"],
    )

    result: dict[str, Any] = {
        "documents": len(docs),
        "chunks": {"kv": len(kv_levels), "vector": len(vdb_levels)},
        "classes": {
            klass: {
                "documents": len(buckets.get(klass, [])),
                "chunks_owned": sum(r["owned"] for r in buckets.get(klass, [])),
            }
            for klass in CLASSES
        },
        "integrity": {
            "access_mismatches": len(divergent),
            "chunks_without_metadata": len(no_metadata),
            "cross_org_entries": len(contaminated),
        },
        "actionable": actionable,
        "resource_ids": sorted(
            {r["resource_id"] for r in actionable if r["resource_id"]}
        ),
        "divergent_samples": [
            {
                "chunk_id": c,
                "kv": sorted(kv_levels[c]),
                "vector": sorted(vdb_levels[c]),
            }
            for c in divergent[:5]
        ],
    }
    if include_class:
        # "all" spans every class, for adjudicating the whole workspace upstream
        # rather than only the classes this module guesses are interesting.
        rows = (
            [row for klass in CLASSES for row in buckets.get(klass, [])]
            if include_class == "all"
            else buckets.get(include_class, [])
        )
        result["class_listing"] = {
            "klass": include_class,
            "documents": sorted(rows, key=lambda r: -r["owned"]),
        }
    return result
