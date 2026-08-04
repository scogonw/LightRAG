"""Offline unit tests for the org clause in the knowledgebase access filter.

Regression cover for a query-time failure that dropped every entity and
relation for callers whose only qualifying path was org-wide access.

``_build_knowledgebase_filter`` is shared by the k-NN vector indices and the
graph ``-nodes``/``-edges`` indices, but the two map ``org_id`` differently:
the vector indices declare it as ``keyword``, while the graph indices declared
nothing and picked it up through ``dynamic: True`` as ``text`` + a ``.keyword``
subfield. An unanalysed ``term`` query against the analysed ``text`` field never
matched an id like ``SCOGO`` (indexed as ``scogo``), so ``get_nodes_batch``
returned nothing, every node read as missing, and the caller got 0 entities /
0 relations while chunks — served from the vector index — still came back.

The filter therefore has to match ``org_id`` *and* ``org_id.keyword``.
"""

import pytest

# opensearch_impl imports opensearch-py at module load; skip the whole module
# if the optional dependency isn't installed in this environment.
opensearch_impl = pytest.importorskip("lightrag.kg.opensearch_impl")

pytestmark = pytest.mark.offline

_build_knowledgebase_filter = opensearch_impl._build_knowledgebase_filter
_org_id_clause = opensearch_impl._org_id_clause

ORG = "SCOGO"


def _collect_org_terms(node):
    """Every ``{"term": {<path>: value}}`` in *node* whose path starts with org_id."""
    found = []
    if isinstance(node, dict):
        term = node.get("term")
        if isinstance(term, dict):
            for path, value in term.items():
                if path.split(".")[0] == "org_id":
                    found.append((path, value))
        for value in node.values():
            found.extend(_collect_org_terms(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_collect_org_terms(item))
    return found


def _org_paths(filter_body):
    return {path for path, _ in _collect_org_terms(filter_body)}


# --- the clause itself -------------------------------------------------------


def test_org_clause_matches_both_mappings():
    clause = _org_id_clause(ORG)
    assert clause == {
        "bool": {
            "should": [
                {"term": {"org_id": ORG}},
                {"term": {"org_id.keyword": ORG}},
            ],
            "minimum_should_match": 1,
        }
    }


def test_org_clause_is_a_should_not_a_must():
    """Both paths must be alternatives — a ``must`` would match nothing anywhere,
    since no single index carries both field paths."""
    inner = _org_id_clause(ORG)["bool"]
    assert "must" not in inner
    assert inner["minimum_should_match"] == 1
    assert len(inner["should"]) == 2


# --- the three call sites ----------------------------------------------------


def test_no_metadata_filter_falls_back_to_both_org_paths():
    body = _build_knowledgebase_filter(None, ORG)
    assert _org_paths(body) == {"org_id", "org_id.keyword"}


def test_empty_kb_params_falls_back_to_both_org_paths():
    body = _build_knowledgebase_filter(
        {"user_id": None, "agent_kb_ids": [], "user_kb_ids": [], "team_kb_ids": []},
        ORG,
    )
    assert _org_paths(body) == {"org_id", "org_id.keyword"}


def test_org_condition_covers_both_paths_for_empty_kb_lists():
    """The reported failure: user_id set, every KB list empty. The org clause is
    the only qualifying condition, so it alone decides whether anything matches."""
    body = _build_knowledgebase_filter(
        {
            "user_id": "user_2wcLXOeP9lz3mNO68qbQnwQTPQX",
            "agent_kb_ids": [],
            "user_kb_ids": [],
            "team_kb_ids": [],
        },
        ORG,
    )
    assert _org_paths(body) == {"org_id", "org_id.keyword"}
    # ...and it stays paired with the access-level restriction.
    must = body["bool"]["must"]
    assert {"terms": {"metadata.access_level.keyword": ["ORGANIZATION", "CHAT_WIDGET"]}} in must


def test_org_condition_covers_both_paths_alongside_team_and_user():
    body = _build_knowledgebase_filter(
        {
            "user_id": "user_37pAuaNdIjXu1nM6oGP6BC5rNn2",
            "agent_kb_ids": [],
            "user_kb_ids": ["drive_user"],
            "team_kb_ids": ["drive_team"],
        },
        ORG,
    )
    assert _org_paths(body) == {"org_id", "org_id.keyword"}
    # Three OR'd conditions: org, team, user-owned.
    assert len(body["bool"]["should"]) == 3
    assert body["bool"]["minimum_should_match"] == 1


# --- paths that must NOT gain an org clause ----------------------------------


def test_agent_path_is_exclusive_and_has_no_org_clause():
    body = _build_knowledgebase_filter(
        {
            "user_id": "user_x",
            "agent_kb_ids": ["drive_agent"],
            "user_kb_ids": ["drive_user"],
            "team_kb_ids": ["drive_team"],
        },
        ORG,
    )
    assert _org_paths(body) == set()
    assert body["bool"]["must"] == [
        {"terms": {"metadata.knowledgebase_id.keyword": ["drive_agent"]}},
        {"term": {"metadata.access_level.keyword": "CHAT_WIDGET"}},
    ]


def test_no_org_and_no_kb_params_yields_no_filter():
    assert _build_knowledgebase_filter(None, None) is None
    assert (
        _build_knowledgebase_filter(
            {"user_id": None, "agent_kb_ids": [], "user_kb_ids": [], "team_kb_ids": []},
            None,
        )
        is None
    )


def test_kb_paths_without_org_carry_no_org_clause():
    body = _build_knowledgebase_filter(
        {
            "user_id": "user_x",
            "agent_kb_ids": [],
            "user_kb_ids": [],
            "team_kb_ids": ["drive_team"],
        },
        None,
    )
    assert _org_paths(body) == set()


# --- the graph index mappings that caused it ---------------------------------


@pytest.mark.parametrize("index_kind", ["nodes", "edges"])
def test_graph_indices_declare_org_id_as_keyword(index_kind):
    """Freshly created graph indices must not leave ``org_id`` to dynamic mapping.

    Read out of the source rather than a live cluster: the mapping bodies are
    built inline in ``_create_indices_if_not_exist``.
    """
    import inspect

    source = inspect.getsource(
        opensearch_impl.OpenSearchGraphStorage._create_indices_if_not_exist
    )
    # Both mapping bodies live in this one method; each must declare org_id.
    assert source.count('"org_id": {"type": "keyword"}') == 2, (
        "expected an explicit keyword mapping for org_id in both the nodes and "
        "edges index bodies"
    )
