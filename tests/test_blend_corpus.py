# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Blend planning: deterministic, share-faithful, and honest about repetition."""
import pytest
from train.corpus import CorpusSource
from scripts.blend_corpus import plan_blend


def _src(name, share, upsample=1):
    return CorpusSource(name=name, slice="spine", target_share=share,
                        hf_repo="r", hf_revision="a" * 40, upsample=upsample)


def test_plan_allocates_each_source_its_share_of_the_budget(monkeypatch):
    sources = {"a": _src("a", 0.75), "b": _src("b", 0.25)}
    monkeypatch.setattr("scripts.blend_corpus.SOURCES", sources)
    plan = plan_blend({"a": 10_000_000, "b": 10_000_000}, budget=1_000_000)
    assert plan == {"a": 750_000, "b": 250_000}


def test_plan_is_deterministic(monkeypatch):
    sources = {"a": _src("a", 0.5), "b": _src("b", 0.5)}
    monkeypatch.setattr("scripts.blend_corpus.SOURCES", sources)
    avail = {"a": 9_000_000, "b": 9_000_000}
    assert plan_blend(avail, 1_000_000) == plan_blend(avail, 1_000_000)


def test_plan_refuses_when_a_source_cannot_meet_its_share(monkeypatch):
    """Silently emitting less than the share would produce a corpus nobody ordered."""
    sources = {"a": _src("a", 1.0, upsample=1)}
    monkeypatch.setattr("scripts.blend_corpus.SOURCES", sources)
    with pytest.raises(ValueError, match="cannot supply"):
        plan_blend({"a": 100}, budget=1_000_000)


def test_plan_counts_upsample_toward_supply(monkeypatch):
    sources = {"a": _src("a", 1.0, upsample=4)}
    monkeypatch.setattr("scripts.blend_corpus.SOURCES", sources)
    assert plan_blend({"a": 300_000}, budget=1_000_000) == {"a": 1_000_000}


# --- _emit: the planner can be perfect and the blend still wrong, so test the emitter too.


def test_emit_truncates_a_source_far_larger_than_the_want(tmp_path):
    """A big source must NOT be emitted whole. This is the bug that shipped once:
    writing only whole passes made tinystories 53% of the blend against a 30% target."""
    from scripts.blend_corpus import _emit
    big = tmp_path / "big.txt"
    big.write_text(" ".join(f"w{i}" for i in range(100_000)) + "\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        written = _emit(big, want_tokens=1_300, out=fh)
    assert written < 2_000, f"emitted {written}; a whole pass would be ~130,000"
    assert len(out.read_text(encoding="utf-8").split()) < 2_000


def test_emit_repeats_a_source_smaller_than_the_want(tmp_path):
    from scripts.blend_corpus import _emit
    small = tmp_path / "small.txt"
    small.write_text("alpha beta gamma delta\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        written = _emit(small, want_tokens=1_300, out=fh)
    assert written >= 1_300 * 0.99
    assert out.read_text(encoding="utf-8").count("alpha") > 1, "should repeat to reach target"


def test_emit_lands_close_to_the_requested_token_count(tmp_path):
    """Overshoot is bounded by a word or two, not by one whole pass over the file."""
    from scripts.blend_corpus import _emit
    src = tmp_path / "s.txt"
    src.write_text("\n".join(" ".join(f"w{i}" for i in range(10)) for _ in range(50_000)),
                   encoding="utf-8")
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        written = _emit(src, want_tokens=13_000, out=fh)
    assert 13_000 * 0.99 <= written <= 13_000 * 1.02


def test_emit_refuses_an_empty_source(tmp_path):
    """Without this guard the repeat loop never terminates."""
    from scripts.blend_corpus import _emit
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        with pytest.raises(ValueError, match="empty"):
            _emit(empty, want_tokens=100, out=fh)


def test_emit_refuses_a_whitespace_only_source(tmp_path):
    """size > 0 but no words: the size check alone does not stop the repeat loop."""
    from scripts.blend_corpus import _emit
    ws = tmp_path / "ws.txt"
    ws.write_text("   \n\n  \n", encoding="utf-8")
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        with pytest.raises(ValueError, match="no words"):
            _emit(ws, want_tokens=100, out=fh)
