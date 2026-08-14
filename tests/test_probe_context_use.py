# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/probe_context_use.py.

Everything that can be exercised without a real model IS -- bucket-edge derivation, the
bucketing/SEM arithmetic, the context-capacity gate (pure config.json/JSON logic), the
stratified per-source span reconstruction, and rendering. The handful of tests that need an
actual forward pass build a tiny random-initialized LlamaForCausalLM
(transformers.LlamaConfig) rather than depending on artifacts/hf-tt-tnt-v1 existing, and skip
cleanly (pytest.importorskip) with an explicit reason if torch/transformers are not installed
-- matching tests/test_eval_per_source.py's convention that this suite must still pass (not
vacuously) on a machine with no torch at all.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_context_use import (  # noqa: E402
    BucketEstimate,
    SourcePositionReport,
    bucket_position_losses,
    build_source_reports,
    default_bucket_edges,
    infer_split_side,
    per_window_position_losses,
    probe_positions,
    report_to_json,
    render_markdown,
    require_context_capacity,
    stratified_source_spans,
    verify_stratified_reconstruction,
)


# ---------------------------------------------------------------------------------------
# default_bucket_edges -- pure arithmetic
# ---------------------------------------------------------------------------------------


def test_default_bucket_edges_matches_the_original_finding_at_512():
    # [0,32) [32,64) [64,128) [128,256) [256,512) -- exactly the buckets that reproduced
    # the original max-context-investigation.md position-wise finding.
    assert default_bucket_edges(512) == [0, 32, 64, 128, 256, 512]


def test_default_bucket_edges_truncates_for_a_shorter_window():
    assert default_bucket_edges(256) == [0, 32, 64, 128, 256]


def test_default_bucket_edges_closes_with_a_partial_final_bucket():
    assert default_bucket_edges(100) == [0, 32, 64, 100]


def test_default_bucket_edges_extends_naturally_past_512():
    assert default_bucket_edges(2048) == [0, 32, 64, 128, 256, 512, 1024, 2048]


def test_default_bucket_edges_handles_a_single_position_window():
    assert default_bucket_edges(1) == [0, 1]


def test_default_bucket_edges_rejects_non_positive_seq_len():
    with pytest.raises(ValueError, match="seq_len must be >= 1"):
        default_bucket_edges(0)


# ---------------------------------------------------------------------------------------
# bucket_position_losses -- the SEM computation, over hand-built numbers
# ---------------------------------------------------------------------------------------


def test_bucket_position_losses_matches_hand_computation():
    # 4 windows x 4 positions. Bucket [0,2) and [2,4).
    per_position = np.array([
        [1.0, 3.0, 5.0, 7.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0, 3.0],
    ])
    edges = [0, 2, 4]
    buckets = bucket_position_losses(per_position, edges)
    assert len(buckets) == 2

    # Bucket [0,2): per-window means of columns 0-1 -> [2.0, 3.0, 1.0, 3.0]
    b0 = buckets[0]
    window_means_0 = np.array([2.0, 3.0, 1.0, 3.0])
    assert b0.lo == 0 and b0.hi == 2
    assert b0.mean == pytest.approx(window_means_0.mean())
    assert b0.sem == pytest.approx(window_means_0.std(ddof=1) / np.sqrt(4))
    assert b0.n_windows == 4

    # Bucket [2,4): per-window means of columns 2-3 -> [6.0, 7.0, 1.0, 3.0]
    b1 = buckets[1]
    window_means_1 = np.array([6.0, 7.0, 1.0, 3.0])
    assert b1.lo == 2 and b1.hi == 4
    assert b1.mean == pytest.approx(window_means_1.mean())
    assert b1.sem == pytest.approx(window_means_1.std(ddof=1) / np.sqrt(4))


def test_bucket_position_losses_reports_zero_sem_not_nan_for_one_window():
    per_position = np.array([[1.0, 2.0, 3.0, 4.0]])
    buckets = bucket_position_losses(per_position, [0, 2, 4])
    assert all(b.sem == 0.0 for b in buckets)
    assert all(b.n_windows == 1 for b in buckets)


def test_bucket_position_losses_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="n_windows, seq_len"):
        bucket_position_losses(np.array([1.0, 2.0, 3.0]), [0, 2])


def test_bucket_position_losses_rejects_edges_not_starting_at_zero():
    per_position = np.zeros((3, 8))
    with pytest.raises(ValueError, match="must start at 0"):
        bucket_position_losses(per_position, [1, 4, 8])


def test_bucket_position_losses_rejects_edges_past_the_window():
    per_position = np.zeros((3, 8))
    with pytest.raises(ValueError, match="past the 8-token window"):
        bucket_position_losses(per_position, [0, 4, 16])


# ---------------------------------------------------------------------------------------
# require_context_capacity -- the too-short-context failure path (no model load needed)
# ---------------------------------------------------------------------------------------


def _write_config(tmp_path: Path, max_position_embeddings) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    config = {"model_type": "llama"}
    if max_position_embeddings is not None:
        config["max_position_embeddings"] = max_position_embeddings
    (d / "config.json").write_text(json.dumps(config))
    return d


def test_require_context_capacity_raises_clearly_when_seq_len_too_long(tmp_path):
    d = _write_config(tmp_path, 256)
    with pytest.raises(ValueError, match=r"exceeds .* max_position_embeddings \(256\)"):
        require_context_capacity(d, 512)


def test_require_context_capacity_passes_when_seq_len_fits(tmp_path):
    d = _write_config(tmp_path, 512)
    assert require_context_capacity(d, 512) == 512
    assert require_context_capacity(d, 100) == 512


def test_require_context_capacity_raises_when_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no config.json"):
        require_context_capacity(tmp_path / "nope", 512)


def test_require_context_capacity_raises_when_field_missing(tmp_path):
    d = _write_config(tmp_path, None)
    with pytest.raises(ValueError, match="no 'max_position_embeddings' field"):
        require_context_capacity(d, 512)


# ---------------------------------------------------------------------------------------
# stratified_source_spans / verify_stratified_reconstruction -- pure numpy
# ---------------------------------------------------------------------------------------


def _u32(*vals):
    return np.array(vals, dtype=np.uint32)


def test_stratified_source_spans_matches_tokenize_stratified_arithmetic():
    # Mirrors train/tokenization.py::_tokenize_stratified's own per-source split exactly:
    # n_val = int(len(arr) * val_fraction); split_point = len(arr) - n_val.
    per_source = {"a": _u32(*range(10)), "b": _u32(*range(100, 107))}
    val_spans = stratified_source_spans(per_source, val_fraction=0.3, side="val")
    train_spans = stratified_source_spans(per_source, val_fraction=0.3, side="train")

    # "a": len 10, n_val = int(10*0.3) = 3 -> train 7, val 3
    assert np.array_equal(train_spans["a"], _u32(0, 1, 2, 3, 4, 5, 6))
    assert np.array_equal(val_spans["a"], _u32(7, 8, 9))
    # "b": len 7, n_val = int(7*0.3) = 2 -> train 5, val 2
    assert np.array_equal(train_spans["b"], _u32(100, 101, 102, 103, 104))
    assert np.array_equal(val_spans["b"], _u32(105, 106))


def test_stratified_source_spans_rejects_bad_side():
    with pytest.raises(ValueError, match="side must be"):
        stratified_source_spans({"a": _u32(1, 2, 3)}, 0.1, "both")


def test_verify_stratified_reconstruction_accepts_a_clean_match():
    spans = {"a": _u32(1, 2), "b": _u32(3, 4, 5)}
    tokens = _u32(1, 2, 3, 4, 5)
    verify_stratified_reconstruction(spans, ["a", "b"], tokens)  # must not raise


def test_verify_stratified_reconstruction_rejects_a_length_mismatch():
    spans = {"a": _u32(1, 2)}
    tokens = _u32(1, 2, 3)
    with pytest.raises(ValueError, match="do not match"):
        verify_stratified_reconstruction(spans, ["a"], tokens)


def test_verify_stratified_reconstruction_rejects_a_content_mismatch():
    spans = {"a": _u32(1, 2, 3)}
    tokens = _u32(1, 2, 9)
    with pytest.raises(ValueError, match="do not match"):
        verify_stratified_reconstruction(spans, ["a"], tokens)


def test_verify_stratified_reconstruction_respects_order():
    spans = {"a": _u32(1, 2), "b": _u32(3, 4)}
    tokens = _u32(3, 4, 1, 2)
    with pytest.raises(ValueError, match="do not match"):
        verify_stratified_reconstruction(spans, ["a", "b"], tokens)
    verify_stratified_reconstruction(spans, ["b", "a"], tokens)  # must not raise


# ---------------------------------------------------------------------------------------
# infer_split_side
# ---------------------------------------------------------------------------------------


def test_infer_split_side_recognises_val():
    assert infer_split_side(Path("artifacts/tokens-stratified/val_ids.npy")) == "val"


def test_infer_split_side_recognises_train():
    assert infer_split_side(Path("artifacts/tokens-stratified/train_ids.npy")) == "train"


def test_infer_split_side_is_none_when_ambiguous():
    assert infer_split_side(Path("artifacts/tokens/ids.npy")) is None
    assert infer_split_side(Path("artifacts/train_val_ids.npy")) is None


# ---------------------------------------------------------------------------------------
# Cross-entropy / bucketed probe against a real (tiny) model -- skipped if torch is missing
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    torch = pytest.importorskip(
        "torch", reason="probe_context_use's forward-pass tests need torch, which is not "
                       "installed in this environment; skipping explicitly rather than "
                       "silently passing.")
    pytest.importorskip(
        "transformers", reason="probe_context_use's forward-pass tests need transformers, "
                              "which is not installed in this environment.")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=400, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).eval()


def test_per_window_position_losses_matches_manual_cross_entropy(tiny_model):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 400, size=(6, 12)).astype("int64")
    y = rng.integers(0, 400, size=(6, 12)).astype("int64")

    got = per_window_position_losses(tiny_model, x, y, batch_size=4)
    assert got.shape == (6, 12)

    with torch.no_grad():
        logits = tiny_model(torch.from_numpy(x)).logits.float()
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), torch.from_numpy(y).reshape(-1),
            reduction="none",
        ).reshape(6, 12).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_per_window_position_losses_is_insensitive_to_batch_size(tiny_model):
    rng = np.random.default_rng(1)
    x = rng.integers(0, 400, size=(7, 12)).astype("int64")
    y = rng.integers(0, 400, size=(7, 12)).astype("int64")
    one_batch = per_window_position_losses(tiny_model, x, y, batch_size=100)
    many_batches = per_window_position_losses(tiny_model, x, y, batch_size=2)
    np.testing.assert_allclose(one_batch, many_batches, atol=1e-5)


def test_probe_positions_returns_one_estimate_per_bucket(tiny_model):
    rng = np.random.default_rng(0)
    ids = np.random.default_rng(2).integers(0, 400, size=2000).astype("uint32")
    buckets = probe_positions(tiny_model, ids, seq_len=16, n_windows=5, batch_size=4, rng=rng)
    assert [ (b.lo, b.hi) for b in buckets ] == list(
        zip(default_bucket_edges(16)[:-1], default_bucket_edges(16)[1:]))
    assert all(b.n_windows == 5 for b in buckets)


def test_probe_positions_raises_clearly_when_ids_too_short(tiny_model):
    ids = np.arange(5, dtype=np.uint32)
    with pytest.raises(ValueError, match="only 5"):
        probe_positions(tiny_model, ids, seq_len=16, n_windows=5, batch_size=4,
                        rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------------------
# build_source_reports -- skip-with-a-note for an under-sized source, real loss for a big one
# ---------------------------------------------------------------------------------------


@pytest.fixture()
def synthetic_source_manifest(monkeypatch):
    from train.corpus import CorpusSource

    manifest = {"sources": {
        "book": {"emitted_words": 1000, "repetition_factor": 2.0, "target_share": 0.6},
        "tiny": {"emitted_words": 10, "repetition_factor": 0.1, "target_share": 0.4},
    }}
    monkeypatch.setitem(sys.modules["scripts.probe_context_use"].SOURCES, "book",
                        CorpusSource(name="book", slice="test-slice", target_share=0.6,
                                    hf_repo="r", hf_revision="rev"))
    monkeypatch.setitem(sys.modules["scripts.probe_context_use"].SOURCES, "tiny",
                        CorpusSource(name="tiny", slice="test-slice", target_share=0.4,
                                    hf_repo="r", hf_revision="rev"))
    return manifest


def test_build_source_reports_skips_a_too_short_source_with_a_note(tiny_model,
                                                                    synthetic_source_manifest):
    order = ["book", "tiny"]
    spans = {
        "book": np.random.default_rng(0).integers(0, 400, size=500).astype("uint32"),
        "tiny": np.array([1, 2, 3], dtype=np.uint32),  # far too short for seq_len=16
    }
    reports = build_source_reports(synthetic_source_manifest, spans, order, tiny_model,
                                   seq_len=16, n_windows=4, batch_size=4, seed=0,
                                   edges=default_bucket_edges(16))
    by_name = {r.name: r for r in reports}
    assert by_name["book"].buckets is not None
    assert by_name["book"].n_tokens_available == 500
    assert by_name["tiny"].buckets is None
    assert "insufficient tokens" in by_name["tiny"].note
    assert by_name["tiny"].n_tokens_available == 3


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


def test_render_markdown_includes_the_overall_table():
    overall = [BucketEstimate(lo=0, hi=32, mean=4.5, sem=0.1, n_windows=64)]
    md = render_markdown(overall, hf_model=Path("m"), tokens_path=Path("t.npy"), seq_len=32,
                         n_windows=64, batch_size=8)
    assert "[0, 32)" in md
    assert "4.5000" in md
    assert "0.1000" in md


def test_render_markdown_marks_a_skipped_source_as_na_not_a_number():
    overall = [BucketEstimate(lo=0, hi=32, mean=1.0, sem=0.0, n_windows=1)]
    source_reports = [
        SourcePositionReport(name="tiny", slice_name="s", n_tokens_available=3, buckets=None,
                             note="insufficient tokens for a 16-token window: only 3 "
                                  "available, need > 17"),
    ]
    md = render_markdown(overall, hf_model=Path("m"), tokens_path=Path("t.npy"), seq_len=32,
                         n_windows=1, batch_size=8, source_reports=source_reports)
    table_lines = [line for line in md.splitlines() if line.startswith("| tiny")]
    assert len(table_lines) == 1
    assert "n/a" in table_lines[0]
    assert "insufficient tokens" in md


def test_report_to_json_round_trips_through_json_dumps():
    overall = [BucketEstimate(lo=0, hi=32, mean=4.5, sem=0.1, n_windows=64)]
    source_reports = [
        SourcePositionReport(name="book", slice_name="s", n_tokens_available=500,
                             buckets=[BucketEstimate(lo=0, hi=32, mean=3.0, sem=0.05,
                                                     n_windows=4)]),
        SourcePositionReport(name="tiny", slice_name="s", n_tokens_available=3, buckets=None,
                             note="insufficient tokens"),
    ]
    payload = report_to_json(overall, hf_model="m", tokens_path="t.npy", seq_len=32,
                             n_windows=64, batch_size=8, seed=0, bucket_edges=[0, 32],
                             source_reports=source_reports)
    text = json.dumps(payload)  # must not raise
    reloaded = json.loads(text)
    assert reloaded["per_source"]["book"]["buckets"][0]["mean"] == pytest.approx(3.0)
    assert reloaded["per_source"]["tiny"]["buckets"] is None
    assert reloaded["overall"][0]["n_windows"] == 64


# ---------------------------------------------------------------------------------------
# CLI defaults and main()'s clear-failure paths
# ---------------------------------------------------------------------------------------


def test_cli_defaults_point_at_the_documented_artifact_paths():
    from scripts.probe_context_use import _parse_args

    args = _parse_args([])
    assert args.hf_model.name == "hf-tt-tnt-v1"
    assert args.tokens.name == "val_ids.npy"
    assert "tokens-stratified" in str(args.tokens)
    assert args.seq_len == 512
    assert args.n_windows > 0


def test_default_output_paths_are_derived_from_the_model_directory_name():
    from scripts.probe_context_use import _default_output_paths

    md, js = _default_output_paths(ROOT / "artifacts" / "hf-tt-tnt-v1")
    assert md.name == "context-use-tt-tnt-v1.md"
    assert js.name == "context-use-tt-tnt-v1.json"


def test_main_refuses_to_run_without_a_converted_model(tmp_path, monkeypatch, capsys):
    from scripts.probe_context_use import main

    monkeypatch.setattr(sys, "argv", ["probe_context_use.py",
                                      "--hf-model", str(tmp_path / "nonexistent")])
    rc = main()
    assert rc == 1
    assert "no converted model" in capsys.readouterr().err


def test_main_refuses_clearly_when_seq_len_exceeds_the_models_trained_context(
        tmp_path, monkeypatch, capsys, tiny_model):
    """End-to-end exercise of the too-short-context failure path via the CLI: a real (tiny)
    saved HF model with a small max_position_embeddings, asked to probe a longer window than
    it was ever trained on. Must fail loudly before doing any forward pass or touching
    --tokens (which is deliberately left pointing at a nonexistent file to prove this)."""
    model_dir = tmp_path / "tiny-model"
    tiny_model.save_pretrained(model_dir)

    monkeypatch.setattr(sys, "argv", [
        "probe_context_use.py",
        "--hf-model", str(model_dir),
        "--tokens", str(tmp_path / "does-not-exist.npy"),
        "--seq-len", "512",
    ])
    from scripts.probe_context_use import main
    rc = main()
    err = capsys.readouterr().err
    assert rc == 1
    assert "max_position_embeddings" in err
    assert "64" in err  # the tiny model's configured max_position_embeddings
