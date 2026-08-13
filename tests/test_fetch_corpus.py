# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch-layer tests. The network is not touched: row iteration is injected."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from train.corpus import CorpusSource
from scripts.fetch_corpus import write_documents, fetch_gutenberg_batch


def test_write_documents_writes_one_json_object_per_line(tmp_path: Path):
    rows = [{"text": "alpha"}, {"text": "beta"}]
    n = write_documents(iter(rows), tmp_path / "text.jsonl")
    assert n == 2
    lines = (tmp_path / "text.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["text"] for x in lines] == ["alpha", "beta"]


def test_write_documents_skips_empty_and_whitespace_only(tmp_path: Path):
    rows = [{"text": "alpha"}, {"text": "   "}, {"text": ""}, {"text": None}, {"text": "beta"}]
    n = write_documents(iter(rows), tmp_path / "text.jsonl")
    assert n == 2


def test_write_documents_creates_parent_directories(tmp_path: Path):
    dest = tmp_path / "deep" / "nested" / "text.jsonl"
    assert write_documents(iter([{"text": "x"}]), dest) == 1
    assert dest.is_file()


def test_fetch_gutenberg_batch_routes_to_matching_sources(tmp_path: Path, monkeypatch):
    """Verify routing logic: each row writes to all matching sources, none to non-matching."""
    # Create test sources matching the real selectors
    spine = CorpusSource(
        name="spine", slice="spine", target_share=0.1,
        hf_repo="sedthh/gutenberg_english", hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        authors=["Fabre, Jean-Henri", "Maeterlinck, Maurice", "Browne, Thomas, Sir"],
    )
    weird = CorpusSource(
        name="weird", slice="weird", target_share=0.1,
        hf_repo="sedthh/gutenberg_english", hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        authors=["Blackwood, Algernon", "Machen, Arthur", "Browne, Thomas, Sir"],
    )

    # Inject fake rows: text and metadata that test different matching cases
    fake_rows = [
        # Row 1: Matches spine only (Fabre author)
        {
            "TEXT": "Observation of ants.",
            "METADATA": json.dumps({"authors": "Fabre, Jean-Henri", "bookshelves": ""}),
        },
        # Row 2: Matches both spine and weird (Browne in both selectors)
        {
            "TEXT": "On the nature of things.",
            "METADATA": json.dumps({"authors": "Browne, Thomas, Sir", "bookshelves": ""}),
        },
        # Row 3: Matches weird only (Blackwood author)
        {
            "TEXT": "The wendigo.",
            "METADATA": json.dumps({"authors": "Blackwood, Algernon", "bookshelves": ""}),
        },
        # Row 4: Matches neither (unknown author)
        {
            "TEXT": "Generic book.",
            "METADATA": json.dumps({"authors": "Unknown Author", "bookshelves": ""}),
        },
        # Row 5: Empty text should be skipped
        {
            "TEXT": "   ",
            "METADATA": json.dumps({"authors": "Fabre, Jean-Henri", "bookshelves": ""}),
        },
    ]

    # Mock load_dataset to return fake rows
    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter(fake_rows)

    # Patch shared_dir to use tmp_path
    def mock_shared_dir(kind):
        if kind == "raw":
            return tmp_path
        raise ValueError(f"unexpected kind: {kind}")

    with patch("datasets.load_dataset", return_value=mock_ds):
        with patch("scripts.fetch_corpus.shared_dir", side_effect=mock_shared_dir):
            counts = fetch_gutenberg_batch([spine, weird], limit_rows=0)

    # Verify returned counts: spine has 2 (rows 1+2), weird has 2 (rows 2+3)
    assert counts == {"spine": 2, "weird": 2}, f"Expected spine:2, weird:2; got {counts}"

    # Verify files were created with correct content
    spine_file = tmp_path / "spine" / "text.jsonl"
    weird_file = tmp_path / "weird" / "text.jsonl"

    assert spine_file.exists(), "spine file not created"
    assert weird_file.exists(), "weird file not created"

    spine_texts = [json.loads(line)["text"] for line in spine_file.read_text().splitlines()]
    weird_texts = [json.loads(line)["text"] for line in weird_file.read_text().splitlines()]

    # Spine should have rows 1 and 2
    assert spine_texts == ["Observation of ants.", "On the nature of things."], \
        f"Expected rows 1+2 in spine, got {spine_texts}"

    # Weird should have rows 2 and 3
    assert weird_texts == ["On the nature of things.", "The wendigo."], \
        f"Expected rows 2+3 in weird, got {weird_texts}"


def test_fetch_gutenberg_batch_limit_rows(tmp_path: Path, monkeypatch):
    """Verify limit_rows stops scanning after specified number of rows."""
    spine = CorpusSource(
        name="spine", slice="spine", target_share=0.1,
        hf_repo="sedthh/gutenberg_english", hf_revision="28973b04f28fd7be4a6186a042bc26159d4366ca",
        authors=["Fabre, Jean-Henri"],
    )

    # Create more rows than the limit
    fake_rows = [
        {"TEXT": "Row 1", "METADATA": json.dumps({"authors": "Fabre, Jean-Henri"})},
        {"TEXT": "Row 2", "METADATA": json.dumps({"authors": "Fabre, Jean-Henri"})},
        {"TEXT": "Row 3", "METADATA": json.dumps({"authors": "Fabre, Jean-Henri"})},
        {"TEXT": "Row 4", "METADATA": json.dumps({"authors": "Fabre, Jean-Henri"})},
    ]

    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter(fake_rows)

    def mock_shared_dir(kind):
        if kind == "raw":
            return tmp_path
        raise ValueError(f"unexpected kind: {kind}")

    with patch("datasets.load_dataset", return_value=mock_ds):
        with patch("scripts.fetch_corpus.shared_dir", side_effect=mock_shared_dir):
            # limit_rows=2 should stop after scanning 2 rows
            counts = fetch_gutenberg_batch([spine], limit_rows=2)

    assert counts["spine"] == 2, f"Expected 2 documents with limit_rows=2, got {counts['spine']}"

    spine_file = tmp_path / "spine" / "text.jsonl"
    spine_texts = [json.loads(line)["text"] for line in spine_file.read_text().splitlines()]
    assert spine_texts == ["Row 1", "Row 2"], f"Expected first 2 rows, got {spine_texts}"
