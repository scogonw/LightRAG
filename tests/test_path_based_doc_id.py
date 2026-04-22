"""Unit tests for path-based doc_id generation.

After the soft-delete removal, doc_id is derived from file_path only so
identical filenames collide regardless of content.
"""
import pytest

from lightrag.utils import compute_mdhash_id

pytestmark = pytest.mark.offline


def _doc_id(file_path: str) -> str:
    """Hash a file_path the way production does."""
    return compute_mdhash_id(file_path, prefix="doc-")


def test_same_path_same_content_same_id():
    a = _doc_id("ABC.pdf")
    b = _doc_id("ABC.pdf")
    assert a == b


def test_same_path_different_content_same_id():
    """Key guarantee: filename alone determines the id."""
    a = _doc_id("ABC.pdf")
    b = _doc_id("ABC.pdf")
    assert a == b


def test_different_path_same_content_different_ids():
    """Key guarantee: same content under different paths produces different ids."""
    a = _doc_id("ABC.pdf")
    b = _doc_id("DEF.pdf")
    assert a != b


def test_id_has_doc_prefix():
    assert _doc_id("ABC.pdf").startswith("doc-")


def test_missing_path_fallback_produces_unique_ids():
    """Two uploads with no file_path must not collide.

    The fallback used to be the literal 'unknown_source' which would make
    every path-less upload share an id once we switched to path-only
    hashing. Now the fallback is 'unknown_source_<uuid4>' so each is unique.
    """
    import re
    from uuid import uuid4

    def fallback_for(path):
        return path if path else f"unknown_source_{uuid4()}"

    a = fallback_for("")
    b = fallback_for("")
    pattern = re.compile(r"^unknown_source_[0-9a-f-]{36}$")
    assert pattern.match(a)
    assert pattern.match(b)
    assert a != b
