import pytest

from lightrag.operate import (
    chunking_by_token_size,
    chunking_by_token_size_table_aware,
)
from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="dummy", tokenizer=DummyTokenizer())


HEADER = (
    "| Particulars | For the quarter ended March 31, 2026 | For the quarter ended December 31, 2025 |\n"
    "|---|---|---|"
)


def statement(rows: int) -> str:
    body = "\n".join(
        f"| Line item {i} | {1000 + i} | {2000 + i} |" for i in range(rows)
    )
    return f"{HEADER}\n{body}"


def contents(chunks):
    return [chunk["content"] for chunk in chunks]


@pytest.mark.offline
def test_text_without_tables_is_chunked_exactly_as_before():
    tokenizer = make_tokenizer()
    content = "word " * 120

    assert chunking_by_token_size_table_aware(
        tokenizer, content, chunk_overlap_token_size=10, chunk_token_size=100
    ) == chunking_by_token_size(
        tokenizer, content, chunk_overlap_token_size=10, chunk_token_size=100
    )


@pytest.mark.offline
def test_split_by_character_is_delegated_unchanged():
    tokenizer = make_tokenizer()
    content = f"intro\n\n{statement(3)}\n\noutro"

    assert chunking_by_token_size_table_aware(
        tokenizer, content, "\n\n", True, chunk_token_size=500
    ) == chunking_by_token_size(tokenizer, content, "\n\n", True, chunk_token_size=500)


@pytest.mark.offline
def test_large_table_pieces_each_carry_the_header_and_every_row_once():
    tokenizer = make_tokenizer()
    table = statement(40)
    content = (
        f"## Statement of standalone financial results\n\n{table}\n\nNotes follow."
    )

    chunks = chunking_by_token_size_table_aware(
        tokenizer, content, chunk_overlap_token_size=20, chunk_token_size=400
    )

    table_chunks = [c for c in contents(chunks) if "| Line item" in c]
    assert len(table_chunks) > 1
    for chunk in table_chunks:
        lines = [line for line in chunk.split("\n") if line.startswith("|")]
        assert "\n".join(lines[:2]) == HEADER
    rows = [
        line
        for chunk in table_chunks
        for line in chunk.split("\n")
        if line.startswith("| Line item")
    ]
    assert rows == [f"| Line item {i} | {1000 + i} | {2000 + i} |" for i in range(40)]
    assert all(chunk["tokens"] <= 400 for chunk in chunks)
    assert [chunk["chunk_order_index"] for chunk in chunks] == list(range(len(chunks)))


@pytest.mark.offline
def test_table_that_fits_starts_a_new_chunk_instead_of_straddling_two():
    tokenizer = make_tokenizer()
    table = statement(4)
    limit = len(table) + 20
    intro = "x" * (limit - 50)  # fits on its own, but not together with the table
    content = f"{intro}\n\n{table}"

    chunks = contents(
        chunking_by_token_size_table_aware(
            tokenizer, content, chunk_overlap_token_size=10, chunk_token_size=limit
        )
    )

    assert chunks == [intro, table]


@pytest.mark.offline
def test_small_text_and_tables_are_packed_together():
    tokenizer = make_tokenizer()
    content = f"Heading\n\n{statement(2)}\n\nFootnote"

    chunks = contents(
        chunking_by_token_size_table_aware(tokenizer, content, chunk_token_size=1200)
    )

    assert chunks == [f"Heading\n\n{statement(2)}\n\nFootnote"]


@pytest.mark.offline
def test_oversized_row_falls_back_to_token_windows():
    tokenizer = make_tokenizer()
    huge_row = "| " + "y" * 500 + " |"
    content = f"{HEADER}\n| small | 1 | 2 |\n{huge_row}\n| tail | 3 | 4 |"

    chunks = chunking_by_token_size_table_aware(
        tokenizer, content, chunk_overlap_token_size=10, chunk_token_size=200
    )

    assert all(chunk["tokens"] <= 200 for chunk in chunks)
    joined = "".join(contents(chunks))
    assert "| small | 1 | 2 |" in joined and "| tail | 3 | 4 |" in joined
    assert joined.count("y") >= 500


@pytest.mark.offline
def test_single_pipe_line_in_prose_is_not_a_table():
    tokenizer = make_tokenizer()
    content = "a | b is prose\n| lone line\nmore prose " + "z " * 100

    assert chunking_by_token_size_table_aware(
        tokenizer, content, chunk_overlap_token_size=10, chunk_token_size=80
    ) == chunking_by_token_size(
        tokenizer, content, chunk_overlap_token_size=10, chunk_token_size=80
    )


@pytest.mark.offline
def test_lightrag_uses_table_aware_chunking_by_default():
    from lightrag.lightrag import LightRAG

    default = LightRAG.__dataclass_fields__["chunking_func"].default_factory()
    assert default is chunking_by_token_size_table_aware
