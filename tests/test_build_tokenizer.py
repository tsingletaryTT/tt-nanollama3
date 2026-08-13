# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""``build_tokenizer.py`` decides WHAT the model is trained on. Nothing downstream checks.

The corpus is a text file; its name is the only claim anyone makes about its contents. So
the branch that picks between "train on what is already there" and "fetch TinyStories and
prepare it" is the highest-blast-radius code in the corpus pipeline, and it shipped
untested: the default ``--corpus`` was ``artifacts/corpus/blend.txt``, and when that file
did not exist the legacy path wrote a TinyStories-only corpus INTO it. Every later run
then found the file, skipped the fetch, and trained on TinyStories while the operator
believed it was the nine-source blend.

These tests cover both arms. ``train_bpe`` and the fetch/prepare pair are patched out --
what is under test is which of them gets called, with what path, not BPE itself.
"""
import sys
from pathlib import Path

import pytest

import scripts.build_tokenizer as bt


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path):
    """Record what the script would have done, without downloading or training."""
    calls = {"fetch": [], "prepare": [], "train": []}

    def fake_fetch(dest_dir, split="train"):
        calls["fetch"].append(Path(dest_dir))
        raw = tmp_path / "raw.txt"
        raw.write_text("tinystories raw\n", encoding="utf-8")
        return raw

    def fake_prepare(src, dest, max_bytes=None):
        calls["prepare"].append((Path(src), Path(dest), max_bytes))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text("prepared tinystories\n", encoding="utf-8")

        class Stats:
            line_count, bytes_written, truncated = 1, 21, False
        return Stats()

    def fake_train(corpus, out_dir, vocab_size=32000, show_progress=False):
        calls["train"].append((Path(corpus), Path(out_dir), vocab_size))

    class FakeTok:
        def __init__(self, size):
            self._size = size

        def get_vocab(self):
            return {str(i): i for i in range(self._size)}

    monkeypatch.setattr(bt, "fetch_corpus", fake_fetch)
    monkeypatch.setattr(bt, "prepare_corpus", fake_prepare)
    monkeypatch.setattr(bt, "train_bpe", fake_train)
    monkeypatch.setattr(bt, "load_exported", lambda out: FakeTok(bt.VOCAB_SIZE))
    monkeypatch.setattr(bt, "ARTIFACTS", tmp_path / "artifacts")
    return calls


def _run(monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["build_tokenizer.py", *argv])
    return bt.main()


# --- arm 1: the corpus already exists.


def test_existing_corpus_is_trained_on_as_is(monkeypatch, tmp_path, stub_pipeline):
    """No fetch, no prepare, no rewrite of the file. The blend took hours to build."""
    corpus = tmp_path / "blend.txt"
    corpus.write_text("the nine-source blend\n", encoding="utf-8")

    assert _run(monkeypatch, "--corpus", str(corpus)) == 0

    assert stub_pipeline["fetch"] == []
    assert stub_pipeline["prepare"] == []
    assert stub_pipeline["train"][0][0] == corpus
    assert corpus.read_text(encoding="utf-8") == "the nine-source blend\n"


def test_corpus_mb_is_ignored_when_the_corpus_exists(monkeypatch, tmp_path, stub_pipeline):
    """A head-truncating byte cap would amputate a blend written one source at a time in
    sorted order -- it would drop whole sources off the end, not sample the blend."""
    corpus = tmp_path / "blend.txt"
    corpus.write_text("x" * 10_000, encoding="utf-8")

    assert _run(monkeypatch, "--corpus", str(corpus), "--corpus-mb", "1") == 0

    assert stub_pipeline["prepare"] == []
    assert len(corpus.read_text(encoding="utf-8")) == 10_000


# --- arm 2: the corpus does not exist.


def test_missing_corpus_takes_the_legacy_path_under_any_other_name(
        monkeypatch, tmp_path, stub_pipeline):
    corpus = tmp_path / "corpus.txt"

    assert _run(monkeypatch, "--corpus", str(corpus), "--corpus-mb", "7") == 0

    assert len(stub_pipeline["fetch"]) == 1
    src, dest, max_bytes = stub_pipeline["prepare"][0]
    assert dest == corpus
    assert max_bytes == 7 * 1024 * 1024, "--corpus-mb applies on the legacy path"
    assert stub_pipeline["train"][0][0] == corpus


def test_missing_blend_refuses_rather_than_fetching_tinystories_into_its_name(
        monkeypatch, tmp_path, stub_pipeline, capsys):
    """THE BUG: on a fresh clone the README sequence produced a 512 MB TinyStories file
    named blend.txt, and every later run trained on it believing it was the blend."""
    corpus = tmp_path / "blend.txt"

    assert _run(monkeypatch, "--corpus", str(corpus)) == 1

    assert not corpus.exists(), "the legacy path must never create blend.txt"
    assert stub_pipeline["fetch"] == []
    assert stub_pipeline["train"] == [], "nothing may be trained on a corpus that is absent"
    err = capsys.readouterr().err
    assert "blend_corpus.py" in err, "the error must say how to build the real blend"
    assert "corpus.txt" in err, "and offer the honestly-named legacy path"


def test_the_default_corpus_is_the_blend(monkeypatch, tmp_path, stub_pipeline):
    """Defaulting elsewhere would train the blend pipeline's tokenizer on something else."""
    blend = tmp_path / "artifacts" / "corpus" / bt.BLEND_NAME
    blend.parent.mkdir(parents=True)
    blend.write_text("blend\n", encoding="utf-8")

    assert _run(monkeypatch) == 0
    assert stub_pipeline["train"][0][0] == blend


def test_the_default_matches_the_tokenizer_step_that_follows_it():
    """The documented sequence is build_tokenizer.py then train/tokenization.py. Two
    different defaults meant step 2 looked for a file step 1 never wrote, and the
    quickstart crashed on a fresh clone."""
    from train.tokenization import _parse_args
    assert Path(_parse_args([]).corpus).name == bt.BLEND_NAME


# --- the vocabulary-size check, which is the script's other silent-failure guard.


def test_a_short_vocabulary_fails_loudly(monkeypatch, tmp_path, stub_pipeline):
    """BPE stops merging when the corpus runs out of pairs, so a small --corpus-mb can
    under-shoot the cap and produce a tokenizer that mismatches the model config's
    vocab_size -- otherwise surfacing as an embedding-shape error much later."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("tiny\n", encoding="utf-8")

    class ShortTok:
        def get_vocab(self):
            return {"a": 0}

    monkeypatch.setattr(bt, "load_exported", lambda out: ShortTok())
    assert _run(monkeypatch, "--corpus", str(corpus)) == 1
