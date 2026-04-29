"""Offline unit tests for the update-document-metadata route's helpers."""

import sys

import pytest

sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import _shallow_merge_metadata  # noqa: E402


def test_shallow_merge_adds_keys():
    assert _shallow_merge_metadata({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_shallow_merge_overwrites_keys():
    assert _shallow_merge_metadata({"a": 1}, {"a": 2}) == {"a": 2}


def test_shallow_merge_null_deletes_key():
    assert _shallow_merge_metadata({"a": 1, "b": 2}, {"a": None}) == {"b": 2}


def test_shallow_merge_null_on_missing_key_is_noop():
    assert _shallow_merge_metadata({"a": 1}, {"missing": None}) == {"a": 1}


def test_shallow_merge_empty_patch():
    assert _shallow_merge_metadata({"a": 1}, {}) == {"a": 1}


def test_shallow_merge_existing_none():
    assert _shallow_merge_metadata(None, {"a": 1}) == {"a": 1}


def test_shallow_merge_does_not_mutate_existing():
    existing = {"a": 1}
    _shallow_merge_metadata(existing, {"b": 2})
    assert existing == {"a": 1}
