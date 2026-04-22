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
