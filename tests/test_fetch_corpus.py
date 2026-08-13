# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch-layer tests. The network is not touched: row iteration is injected."""
import json
from pathlib import Path

from scripts.fetch_corpus import write_documents


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
