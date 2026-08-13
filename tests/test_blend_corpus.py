# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Blend planning: deterministic, share-faithful, and honest about repetition."""
import json
from pathlib import Path

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
        written = _emit(big, want_tokens=1_300, out=fh, tokens_per_word=1.3).tokens
    assert written < 2_000, f"emitted {written}; a whole pass would be ~130,000"
    assert len(out.read_text(encoding="utf-8").split()) < 2_000


def test_emit_repeats_a_source_smaller_than_the_want(tmp_path):
    from scripts.blend_corpus import _emit
    small = tmp_path / "small.txt"
    small.write_text("alpha beta gamma delta\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        written = _emit(small, want_tokens=1_300, out=fh, tokens_per_word=1.3).tokens
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
        written = _emit(src, want_tokens=13_000, out=fh, tokens_per_word=1.3).tokens
    assert 13_000 * 0.99 <= written <= 13_000 * 1.02


def test_emit_refuses_an_empty_source(tmp_path):
    """Without this guard the repeat loop never terminates."""
    from scripts.blend_corpus import _emit
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        with pytest.raises(ValueError, match="empty"):
            _emit(empty, want_tokens=100, out=fh, tokens_per_word=1.3)


def test_emit_refuses_a_whitespace_only_source(tmp_path):
    """size > 0 but no words: the size check alone does not stop the repeat loop."""
    from scripts.blend_corpus import _emit
    ws = tmp_path / "ws.txt"
    ws.write_text("   \n\n  \n", encoding="utf-8")
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        with pytest.raises(ValueError, match="no words"):
            _emit(ws, want_tokens=100, out=fh, tokens_per_word=1.3)


# --- the emitter must size itself with each source's MEASURED tokens/word.
#
# The shipped blend used a flat 1.3 while plan_blend gated on tokenizer-measured
# availability. Real ratios run 1.194-1.559 across the nine sources, so the emitter
# over-emitted for eight of them: wikipedia_simple (ratio 1.559, upsample=1) made 1.058
# passes and silently duplicated ~5.8% of itself, and procedural made 4.03 passes against
# a 4x limit. These tests pin the ratio path shut.


def test_measure_tokens_per_word_derives_the_ratio_from_the_availability_report(tmp_path):
    from scripts.blend_corpus import _measure_tokens_per_word
    src = tmp_path / "s.txt"
    src.write_text("alpha beta gamma delta\nepsilon zeta\n", encoding="utf-8")
    ratio, words = _measure_tokens_per_word(src, available_tokens=9)
    assert words == 6
    assert ratio == pytest.approx(1.5)


def test_measure_tokens_per_word_refuses_a_source_with_no_measurement(tmp_path):
    """A zero here would silently divide the whole plan by nothing."""
    from scripts.blend_corpus import _measure_tokens_per_word
    src = tmp_path / "s.txt"
    src.write_text("alpha beta\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no measured token count"):
        _measure_tokens_per_word(src, available_tokens=0)


def test_measure_tokens_per_word_refuses_a_wordless_source(tmp_path):
    from scripts.blend_corpus import _measure_tokens_per_word
    src = tmp_path / "s.txt"
    src.write_text("   \n\n \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no words"):
        _measure_tokens_per_word(src, available_tokens=100)


def test_emit_at_a_high_ratio_does_not_repeat_a_source_that_covers_the_want(tmp_path):
    """THE BUG: at the flat 1.3, a source whose real ratio is 1.56 got over-emitted past
    the end of the file and wrapped around, duplicating part of itself while declaring
    upsample=1. With the measured ratio it stays inside one pass."""
    from scripts.blend_corpus import _emit
    src = tmp_path / "s.txt"
    src.write_text("\n".join(f"w{i}" for i in range(10_000)) + "\n", encoding="utf-8")
    # 10,000 words at 1.56 tokens/word = 15,600 tokens available; want 15,000.
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        emission = _emit(src, want_tokens=15_000, out=fh, tokens_per_word=1.56)
    assert emission.words < 10_000, "must not need a second pass"
    assert out.read_text(encoding="utf-8").count("\n\n") == 0, "no pass boundary written"
    assert emission.tokens == pytest.approx(15_000, rel=0.01)

    # The same want at the old flat constant needs 11,538 words: more than the file holds.
    out2 = tmp_path / "out2.txt"
    with out2.open("w", encoding="utf-8") as fh:
        flat = _emit(src, want_tokens=15_000, out=fh, tokens_per_word=1.3)
    assert flat.words > 10_000, "the flat constant is what caused the undeclared repeat"


def test_emit_repetition_stays_within_the_declared_upsample(tmp_path):
    """With a measured ratio, repetition collapses to want/available, which plan_blend's
    gate already holds at or below the declared upsample. That is the invariant the flat
    constant broke for procedural (4.03 passes against a 4x limit)."""
    from scripts.blend_corpus import _emit
    src = tmp_path / "s.txt"
    src.write_text("\n".join(f"w{i}" for i in range(10_000)) + "\n", encoding="utf-8")
    available, upsample = 13_400, 4          # 10,000 words at 1.34 tokens/word
    want = int(available * 3.9)              # procedural's real 3.91x, near the limit
    with (tmp_path / "out.txt").open("w", encoding="utf-8") as fh:
        emission = _emit(src, want_tokens=want, out=fh,
                         tokens_per_word=available / 10_000)
    passes = emission.words / 10_000
    assert passes <= upsample
    assert passes == pytest.approx(want / available, rel=1e-3)


def test_token_meter_counts_the_same_way_measure_corpus_does():
    """emitted_tokens is only comparable with available_tokens if both chunk identically:
    BPE merges do not cross an encode() call."""
    from scripts.blend_corpus import TokenMeter

    class FakeEnc:
        def __init__(self, n):
            self.ids = list(range(n))

    class FakeTok:
        """One 'token' per word, so a mis-chunk shows up as a changed count."""
        def encode_batch(self, chunks):
            return [FakeEnc(len(c.split())) for c in chunks]

    text = "alpha beta\n\ngamma\n\n   \n\ndelta epsilon zeta\n"
    meter = TokenMeter(FakeTok())
    # Fed in arbitrary fragments: the meter must not depend on where the writes split.
    for i in range(0, len(text), 7):
        meter.feed(text[i:i + 7])
    expected = sum(len(c.split()) for c in text.split("\n\n") if c.strip())
    assert meter.close() == expected == 6


def test_emit_reports_its_emission_to_the_meter(tmp_path):
    """The manifest's real token count comes from this callback, so it must see every
    byte written for the source -- including the pass separator and the truncated tail."""
    from scripts.blend_corpus import _emit
    src = tmp_path / "s.txt"
    src.write_text("alpha beta gamma\n", encoding="utf-8")
    seen = []
    out = tmp_path / "out.txt"
    with out.open("w", encoding="utf-8") as fh:
        _emit(src, want_tokens=13, out=fh, tokens_per_word=1.3, on_text=seen.append)
    assert "".join(seen) == out.read_text(encoding="utf-8")


# --- the committed record of the blend that was actually built.
#
# artifacts/ is gitignored, so without a tracked copy the answer to "what was this model
# trained on" would exist only on the machine that ran the blend. These tests hold that
# record to the registry it claims to implement.

RECORD = (Path(__file__).resolve().parents[1]
          / "docs" / "measurements" / "blend_manifest.json")


def _record():
    return json.loads(RECORD.read_text())


def test_recorded_blend_covers_every_registered_source():
    """A source added to the registry without re-blending must break this, not slip
    through into a manifest that no longer describes the artifact."""
    from train.corpus import SOURCES
    rec = _record()["sources"]
    assert set(rec) == set(SOURCES)
    for name, src in SOURCES.items():
        assert rec[name]["target_share"] == src.target_share
        assert rec[name]["declared_upsample"] == src.upsample
        assert rec[name]["hf_revision"] == src.hf_revision


def test_recorded_token_counts_are_the_tokenizers_own():
    """'approx' here would mean the provenance manifest is estimating what it claims to
    record."""
    rec = _record()
    assert rec["token_count_method"] == "tokenizer"
    assert all(s["emitted_tokens_method"] == "tokenizer"
               for s in rec["sources"].values())


def test_recorded_repetition_never_exceeds_the_declared_upsample():
    """THE REGRESSION: the shipped blend put procedural at 4.03 passes against a declared
    4x, and wikipedia_simple at 1.058 against a declared 1x -- i.e. it duplicated ~5.8% of
    Simple Wikipedia while the manifest said it repeated nothing."""
    for name, s in _record()["sources"].items():
        assert s["repetition_factor"] <= s["declared_upsample"], name
        assert s["repetition_within_declared_upsample"], name


def test_recorded_shares_track_their_targets():
    """The shipped blend deviated by up to ~3 points while reporting achieved_share as
    exactly equal to target to 15 decimal places."""
    for name, s in _record()["sources"].items():
        assert abs(s["achieved_share"] - s["target_share"]) < 0.005, name


def test_recorded_total_is_stated_against_the_budget():
    """The 400M-vs-real figure has to exist in the repository, not only in a task report."""
    rec = _record()
    assert rec["total_emitted_tokens"] == sum(
        s["emitted_tokens"] for s in rec["sources"].values())
    assert rec["total_vs_budget_tokens"] == rec["total_emitted_tokens"] - rec["budget"]
    assert abs(rec["total_vs_budget_pct"]) < 1.0
