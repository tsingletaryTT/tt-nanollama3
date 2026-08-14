# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/eval_per_source.py.

Everything that can be exercised without a real model IS -- manifest parsing, source-span
reconstruction, window sampling, and the verification checks are pure numpy/JSON logic. The
handful of tests that need an actual forward pass build a tiny random-initialized
LlamaForCausalLM (transformers.LlamaConfig) rather than depending on
artifacts/hf-tt-tnt-v1 existing, and skip cleanly (pytest.importorskip) if torch is not
installed in this environment -- matching tests/test_generate_samples.py's convention that
this suite must still pass on a machine with no torch at all.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from convert.tokenizer import train_bpe  # noqa: E402
from scripts.eval_per_source import (  # noqa: E402
    LossEstimate,
    build_reports,
    emission_order,
    emitted_words,
    find_word_boundary_line,
    load_manifest,
    mean_and_sem,
    per_window_losses,
    reports_to_json,
    render_markdown,
    sample_windows,
    tokenize_blend_by_source,
    unseen_tail_ids,
    verify_against_disk_arrays,
)


# ---------------------------------------------------------------------------------------
# Fixtures shared across this file
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "train.txt"
    corpus.write_text("\n".join(["the cat sat on the mat", "a dog ran fast"] * 200) + "\n",
                      encoding="utf-8")
    return train_bpe(corpus, d / "tokenizer", vocab_size=400)


@pytest.fixture(scope="module")
def tok(tiny_tokenizer):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(tiny_tokenizer), local_files_only=True)


def _alpha_zebra_lines():
    alpha = [f"alpha word number {i}" for i in range(20)]  # 4 words/line, 80 words total
    zebra = [f"zebra word number {i}" for i in range(20)]
    return alpha, zebra


# ---------------------------------------------------------------------------------------
# Manifest parsing -- pure JSON, no tokenizer needed
# ---------------------------------------------------------------------------------------


def test_load_manifest_raises_clearly_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="blend manifest not found"):
        load_manifest(tmp_path / "nope.json")


def test_emission_order_is_sorted_by_name_not_insertion_order():
    manifest = {"sources": {"zebra": {}, "alpha": {}, "mango": {}}}
    assert emission_order(manifest) == ["alpha", "mango", "zebra"]


def test_emitted_words_extracts_the_right_field():
    manifest = {"sources": {"a": {"emitted_words": 5, "emitted_tokens": 999}}}
    assert emitted_words(manifest) == {"a": 5}


# ---------------------------------------------------------------------------------------
# tokenize_blend_by_source: the exact per-source span reconstruction
# ---------------------------------------------------------------------------------------


def test_tokenize_blend_by_source_matches_manual_tokenization(tmp_path, tok):
    from train.tokenization import encode_batch

    alpha, zebra = _alpha_zebra_lines()
    blend = tmp_path / "blend.txt"
    blend.write_text("\n".join(alpha + zebra) + "\n", encoding="utf-8")

    result = tokenize_blend_by_source(blend, ["alpha", "zebra"], {"alpha": 80, "zebra": 80},
                                      tok)
    assert set(result) == {"alpha", "zebra"}
    assert np.array_equal(result["alpha"], encode_batch(alpha, tok))
    assert np.array_equal(result["zebra"], encode_batch(zebra, tok))


def test_tokenize_blend_by_source_is_insensitive_to_chunk_lines(tmp_path, tok):
    alpha, zebra = _alpha_zebra_lines()
    blend = tmp_path / "blend.txt"
    blend.write_text("\n".join(alpha + zebra) + "\n", encoding="utf-8")

    small_chunks = tokenize_blend_by_source(blend, ["alpha", "zebra"],
                                            {"alpha": 80, "zebra": 80}, tok, chunk_lines=3)
    big_chunks = tokenize_blend_by_source(blend, ["alpha", "zebra"],
                                          {"alpha": 80, "zebra": 80}, tok, chunk_lines=5000)
    assert np.array_equal(small_chunks["alpha"], big_chunks["alpha"])
    assert np.array_equal(small_chunks["zebra"], big_chunks["zebra"])


def test_tokenize_blend_by_source_rejects_overshoot(tmp_path, tok):
    alpha, zebra = _alpha_zebra_lines()
    blend = tmp_path / "blend.txt"
    blend.write_text("\n".join(alpha + zebra) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overshot"):
        tokenize_blend_by_source(blend, ["alpha", "zebra"], {"alpha": 79, "zebra": 81}, tok)


def test_tokenize_blend_by_source_rejects_leftover_lines(tmp_path, tok):
    alpha, zebra = _alpha_zebra_lines()
    blend = tmp_path / "blend.txt"
    blend.write_text("\n".join(alpha + zebra) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="more lines than"):
        tokenize_blend_by_source(blend, ["alpha", "zebra"], {"alpha": 80, "zebra": 40}, tok)


def test_tokenize_blend_by_source_rejects_short_corpus(tmp_path, tok):
    alpha, zebra = _alpha_zebra_lines()
    blend = tmp_path / "blend.txt"
    blend.write_text("\n".join(alpha + zebra) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ended before source"):
        tokenize_blend_by_source(blend, ["alpha", "zebra"], {"alpha": 80, "zebra": 120}, tok)


# ---------------------------------------------------------------------------------------
# Raw-file tail location for genuinely-unseen data (poetry, tinystories-style sources)
# ---------------------------------------------------------------------------------------


def test_find_word_boundary_line_exact_line_end(tmp_path):
    p = tmp_path / "raw.txt"
    p.write_text("aa bb\ncc dd\nee ff\n", encoding="utf-8")  # 2 words/line
    # First 2 lines = 4 words exactly -> boundary line is index 1 (0-based)
    assert find_word_boundary_line(p, 4) == 1


def test_find_word_boundary_line_mid_line(tmp_path):
    p = tmp_path / "raw.txt"
    p.write_text("aa bb cc\ndd ee ff\n", encoding="utf-8")  # 3 words/line
    # Target 4 falls inside line 1 (words 4-6) -- boundary line is index 1.
    assert find_word_boundary_line(p, 4) == 1


def test_find_word_boundary_line_raises_when_file_too_short(tmp_path):
    p = tmp_path / "raw.txt"
    p.write_text("aa bb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than"):
        find_word_boundary_line(p, 100)


def test_unseen_tail_ids_excludes_the_boundary_line(tmp_path, tok):
    from train.tokenization import encode_batch

    p = tmp_path / "raw.txt"
    lines = [f"line {i} word" for i in range(10)]  # 3 words/line, 30 words total
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # emitted_words=15 lands exactly at the end of line index 4 (0-based) -- lines 0..4
    # (5 lines * 3 words = 15) may have been (partially) emitted; lines 5..9 are the tail.
    tail_ids, words_sampled = unseen_tail_ids(p, 15, tok, word_cap=1_000_000)
    expected = encode_batch(lines[5:], tok)
    assert np.array_equal(tail_ids, expected)
    assert words_sampled == 15  # 5 remaining lines * 3 words


def test_unseen_tail_ids_respects_the_word_cap(tmp_path, tok):
    p = tmp_path / "raw.txt"
    lines = [f"line {i} word" for i in range(100)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _, words_sampled = unseen_tail_ids(p, 3, tok, word_cap=9)
    assert words_sampled <= 9 + 3  # at most one line's worth over the cap


# ---------------------------------------------------------------------------------------
# verify_against_disk_arrays: the mechanical confirmation of "val ⊂ one source"
# ---------------------------------------------------------------------------------------


def _u32(*vals):
    return np.array(vals, dtype=np.uint32)


def test_verify_accepts_a_clean_straddling_reconstruction():
    per_source = {"a": _u32(1, 2, 3, 4), "b": _u32(5, 6, 7, 8)}
    train_ids = _u32(1, 2, 3, 4, 5, 6)
    val_ids = _u32(7, 8)
    split = verify_against_disk_arrays(per_source, ["a", "b"], train_ids, val_ids)
    assert split == {"a": (4, 0), "b": (2, 2)}


def test_verify_rejects_a_total_token_count_mismatch():
    per_source = {"a": _u32(1, 2, 3)}
    train_ids = _u32(1, 2)
    val_ids = _u32(9, 9, 9)  # deliberately wrong total
    with pytest.raises(ValueError, match="does not match"):
        verify_against_disk_arrays(per_source, ["a"], train_ids, val_ids)


def test_verify_rejects_a_straddler_whose_val_slice_is_wrong():
    per_source = {"a": _u32(1, 2, 3, 4)}
    train_ids = _u32(1, 2)
    val_ids = _u32(99, 99)  # right length, wrong content
    with pytest.raises(ValueError, match="does not match val_ids"):
        verify_against_disk_arrays(per_source, ["a"], train_ids, val_ids)


def test_verify_returns_exactly_one_straddler_for_contiguous_spans():
    """With per-source spans laid out by cumulative offset (as tokenize_blend_by_source
    always produces them), a single boundary value can only ever fall inside one source's
    span -- this is the ordinary case verify_against_disk_arrays must accept cleanly, and
    what the real pipeline's own sources ("no straddler" or "exactly one") look like."""
    per_source = {"a": _u32(1, 2, 3, 4), "b": _u32(5, 6, 7, 8)}
    train_ids = _u32(1, 2, 3, 4, 5, 6)  # boundary lands inside "b"
    val_ids = _u32(7, 8)
    split = verify_against_disk_arrays(per_source, ["a", "b"], train_ids, val_ids)
    assert sum(1 for _, v in split.values() if v > 0) == 1


# ---------------------------------------------------------------------------------------
# Window sampling and loss statistics -- pure numpy
# ---------------------------------------------------------------------------------------


def test_sample_windows_shapes_and_alignment():
    ids = np.arange(1000, dtype=np.uint32)
    rng = np.random.default_rng(0)
    x, y = sample_windows(ids, seq_len=10, n_windows=5, rng=rng)
    assert x.shape == (5, 10)
    assert y.shape == (5, 10)
    # y is x shifted by exactly one position, since both come from the same ids array.
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_sample_windows_raises_when_too_little_data():
    ids = np.arange(5, dtype=np.uint32)
    with pytest.raises(ValueError, match="only 5"):
        sample_windows(ids, seq_len=10, n_windows=3, rng=np.random.default_rng(0))


def test_mean_and_sem_matches_hand_computation():
    losses = [1.0, 2.0, 3.0, 4.0]
    mean, sem = mean_and_sem(losses)
    assert mean == pytest.approx(2.5)
    assert sem == pytest.approx(np.std(losses, ddof=1) / 2.0)


def test_mean_and_sem_is_zero_not_nan_for_a_single_sample():
    mean, sem = mean_and_sem([3.5])
    assert mean == 3.5
    assert sem == 0.0


# ---------------------------------------------------------------------------------------
# Cross-entropy against a real (tiny) model -- skipped if torch is unavailable
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=400, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).eval()


def test_per_window_losses_matches_manual_cross_entropy(tiny_model):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 400, size=(6, 12)).astype("int64")
    y = rng.integers(0, 400, size=(6, 12)).astype("int64")

    got = per_window_losses(tiny_model, x, y, batch_size=4)
    assert len(got) == 6

    with torch.no_grad():
        logits = tiny_model(torch.from_numpy(x)).logits.float()
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), torch.from_numpy(y).reshape(-1),
            reduction="none",
        ).reshape(6, 12).mean(dim=1).tolist()
    assert got == pytest.approx(expected, abs=1e-5)


def test_per_window_losses_is_insensitive_to_batch_size(tiny_model):
    rng = np.random.default_rng(1)
    x = rng.integers(0, 400, size=(7, 12)).astype("int64")
    y = rng.integers(0, 400, size=(7, 12)).astype("int64")
    one_batch = per_window_losses(tiny_model, x, y, batch_size=100)
    many_batches = per_window_losses(tiny_model, x, y, batch_size=2)
    assert one_batch == pytest.approx(many_batches, abs=1e-5)


# ---------------------------------------------------------------------------------------
# End-to-end smoke test: build_reports / render_markdown / reports_to_json together
# ---------------------------------------------------------------------------------------


@pytest.fixture()
def synthetic_pipeline(tmp_path, tok, tiny_model, monkeypatch):
    """A miniature stand-in for the real artifacts/ layout: two sources, one fully
    repeated (no held-out tail) and one under-consumed (has a held-out tail), plus a real
    (tiny) HF model to run forward passes against."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # "alpha": repetition_factor >= 1 -> no held-out tail possible.
    alpha_raw = [f"alpha raw word {i} filler token here now" for i in range(40)]
    (corpus_dir / "alpha.txt").write_text("\n".join(alpha_raw) + "\n", encoding="utf-8")
    alpha_blend_lines = alpha_raw + alpha_raw[:10]  # more than one full pass

    # "zebra": repetition_factor < 1 -> the raw file's tail beyond what's blended is
    # genuinely unseen.
    zebra_raw = [f"zebra raw word {i} filler token here now" for i in range(100)]
    (corpus_dir / "zebra.txt").write_text("\n".join(zebra_raw) + "\n", encoding="utf-8")
    zebra_blend_lines = zebra_raw[:20]  # only the first 20 of 100 lines ever emitted

    blend_lines = alpha_blend_lines + zebra_blend_lines
    blend_path = corpus_dir / "blend.txt"
    blend_path.write_text("\n".join(blend_lines) + "\n", encoding="utf-8")

    alpha_words = sum(len(line.split()) for line in alpha_blend_lines)
    zebra_words = sum(len(line.split()) for line in zebra_blend_lines)

    manifest = {
        "sources": {
            "alpha": {
                "emitted_words": alpha_words, "repetition_factor": 1.25,
                "target_share": 0.6,
            },
            "zebra": {
                "emitted_words": zebra_words, "repetition_factor": 0.2,
                "target_share": 0.4,
            },
        }
    }

    # Register these synthetic names in train.corpus.SOURCES for the "slice" lookup, the
    # same way the real registry describes flavour/folklore/etc.
    from train.corpus import CorpusSource
    monkeypatch.setitem(sys.modules["scripts.eval_per_source"].SOURCES, "alpha",
                        CorpusSource(name="alpha", slice="test-slice", target_share=0.6,
                                    hf_repo="r", hf_revision="rev"))
    monkeypatch.setitem(sys.modules["scripts.eval_per_source"].SOURCES, "zebra",
                        CorpusSource(name="zebra", slice="test-slice", target_share=0.4,
                                    hf_repo="r", hf_revision="rev"))

    order = ["alpha", "zebra"]
    per_source_ids = tokenize_blend_by_source(
        blend_path, order, {"alpha": alpha_words, "zebra": zebra_words}, tok)

    # A "training run" that held out nothing extra from the whole-stream tail (val_fraction
    # 0 for simplicity) -- this test is about per-source held-out TAILS, not the whole-
    # stream split, so keep that split degenerate and uninvolved.
    train_ids = np.concatenate([per_source_ids["alpha"], per_source_ids["zebra"]])
    val_ids = np.zeros(0, dtype=np.uint32)

    split = verify_against_disk_arrays(per_source_ids, order, train_ids, val_ids)
    return {
        "manifest": manifest, "per_source_ids": per_source_ids, "split": split,
        "order": order, "val_ids": val_ids, "corpus_dir": corpus_dir, "tok": tok,
        "model": tiny_model,
    }


def test_build_reports_labels_repeated_source_as_no_holdout(synthetic_pipeline):
    reports = build_reports(
        synthetic_pipeline["manifest"], synthetic_pipeline["per_source_ids"],
        synthetic_pipeline["split"], synthetic_pipeline["order"],
        synthetic_pipeline["val_ids"], synthetic_pipeline["corpus_dir"],
        synthetic_pipeline["tok"], synthetic_pipeline["model"],
        seq_len=8, n_windows=4, batch_size=4, tail_word_cap=10_000, seed=0,
    )
    by_name = {r.name: r for r in reports}
    assert by_name["alpha"].held_out is None
    assert "no unseen tail exists" in by_name["alpha"].note


def test_build_reports_finds_a_genuine_holdout_for_undersampled_source(synthetic_pipeline):
    reports = build_reports(
        synthetic_pipeline["manifest"], synthetic_pipeline["per_source_ids"],
        synthetic_pipeline["split"], synthetic_pipeline["order"],
        synthetic_pipeline["val_ids"], synthetic_pipeline["corpus_dir"],
        synthetic_pipeline["tok"], synthetic_pipeline["model"],
        seq_len=8, n_windows=4, batch_size=4, tail_word_cap=10_000, seed=0,
    )
    by_name = {r.name: r for r in reports}
    assert by_name["zebra"].held_out is not None
    assert isinstance(by_name["zebra"].held_out, LossEstimate)
    assert "unused tail" in by_name["zebra"].held_out_kind


def test_render_markdown_marks_missing_holdout_as_na_not_a_number(synthetic_pipeline):
    reports = build_reports(
        synthetic_pipeline["manifest"], synthetic_pipeline["per_source_ids"],
        synthetic_pipeline["split"], synthetic_pipeline["order"],
        synthetic_pipeline["val_ids"], synthetic_pipeline["corpus_dir"],
        synthetic_pipeline["tok"], synthetic_pipeline["model"],
        seq_len=8, n_windows=4, batch_size=4, tail_word_cap=10_000, seed=0,
    )
    md = render_markdown(reports, checkpoint_step=1, hf_model=Path("m"), seq_len=8,
                         headline_val_loss=1.23)
    lines = [line for line in md.splitlines() if line.startswith("| alpha")]
    assert len(lines) == 1
    assert "n/a" in lines[0]


def test_build_reports_uses_val_ids_for_a_straddling_undersampled_source(tiny_model, tok,
                                                                          tmp_path):
    """The wikipedia_simple-shaped case: repetition_factor < 1 AND the source straddles
    the train/val boundary -- held-out must come from val_ids directly, never from a
    freshly-derived raw-file tail (there is no "raw file" for this branch to read)."""
    ids = np.arange(200, dtype=np.uint32)
    per_source_ids = {"wiki": ids}
    val_ids = ids[150:]
    split = {"wiki": (150, 50)}
    manifest = {"sources": {"wiki": {"emitted_words": 999, "repetition_factor": 0.7,
                                     "target_share": 0.3}}}

    reports = build_reports(
        manifest, per_source_ids, split, ["wiki"], val_ids, tmp_path, tok, tiny_model,
        seq_len=8, n_windows=4, batch_size=4, tail_word_cap=10_000, seed=0,
    )
    assert len(reports) == 1
    r = reports[0]
    assert r.held_out is not None
    assert "val_ids.npy" in r.held_out_kind
    assert r.train_tokens == 150


def test_reports_to_json_round_trips_through_json_dumps(synthetic_pipeline):
    reports = build_reports(
        synthetic_pipeline["manifest"], synthetic_pipeline["per_source_ids"],
        synthetic_pipeline["split"], synthetic_pipeline["order"],
        synthetic_pipeline["val_ids"], synthetic_pipeline["corpus_dir"],
        synthetic_pipeline["tok"], synthetic_pipeline["model"],
        seq_len=8, n_windows=4, batch_size=4, tail_word_cap=10_000, seed=0,
    )
    payload = reports_to_json(reports, checkpoint_step=1, hf_model="m", seq_len=8)
    text = json.dumps(payload)  # must not raise (no NaN/inf/non-serializable values)
    reloaded = json.loads(text)
    assert reloaded["sources"]["alpha"]["held_out_loss"] is None
    assert reloaded["sources"]["zebra"]["held_out_loss"]["n_windows"] == 4


# ---------------------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------------------


def test_cli_defaults_point_at_the_real_artifact_paths():
    from scripts.eval_per_source import _parse_args

    args = _parse_args([])
    assert args.hf_model.name == "hf-tt-tnt-v1"
    assert args.seq_len == 512
    assert args.n_windows > 0
    assert args.out.name == "per-source-loss-tt-tnt-v1.md"


def test_main_refuses_to_run_without_a_converted_model(tmp_path, monkeypatch, capsys):
    from scripts.eval_per_source import main

    monkeypatch.setattr(sys, "argv", ["eval_per_source.py",
                                      "--hf-model", str(tmp_path / "nonexistent")])
    rc = main()
    assert rc == 1
    assert "no converted model" in capsys.readouterr().err
