# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/generate_samples.py.

No trained model exists yet for the new corpus, and this suite must pass in environments
where no model (and possibly not even torch) is installed at all -- that's the normal state
until a checkpoint is trained and converted. So everything here that can be tested without a
model IS tested without one: load_prompts() is pure JSON I/O, and resolve_model_dir() is a
pure filesystem check, both importable and callable with no torch/transformers import
triggered (those imports live inside main(), not at module scope).

The one test that exercises a real model end-to-end (test_generation_smoke_test_against_real_hf_model)
is skipped EXPLICITLY, with a reason printed in the pytest report, when artifacts/hf is not
present -- it never just silently passes as a no-op.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_samples import (  # noqa: E402
    PROMPTS,
    load_prompts,
    resolve_model_dir,
    validate_sampling_args,
)


# --- load_prompts(): pure JSON I/O, no model required ----------------------------------

def test_load_prompts_matches_the_frozen_json_file_on_disk():
    prompts = load_prompts()
    raw = json.loads(PROMPTS.read_text())["prompts"]
    assert prompts == raw


def test_load_prompts_returns_all_fifteen_frozen_prompts():
    # Pinning the count here is deliberate: a change in count is either a new probe added
    # correctly (test updated deliberately) or an accidental edit to the frozen set (caught).
    assert len(load_prompts()) == 15


# --- resolve_model_dir(): pure filesystem check, no model required ---------------------

def test_resolve_model_dir_raises_clearly_for_a_nonexistent_directory(tmp_path):
    missing = tmp_path / "no-such-model-dir"
    with pytest.raises(FileNotFoundError, match="no such directory"):
        resolve_model_dir(str(missing))


def test_resolve_model_dir_raises_clearly_when_config_json_is_missing(tmp_path):
    # Directory exists but was never populated by convert_checkpoint.py.
    with pytest.raises(FileNotFoundError, match="config.json"):
        resolve_model_dir(str(tmp_path))


def test_resolve_model_dir_accepts_a_directory_that_has_config_json(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_model_dir(str(tmp_path)) == tmp_path


# --- validate_sampling_args(): pure value check, no model required ---------------------

def test_validate_sampling_args_accepts_the_all_none_default():
    # temperature=None, top_p=None, top_k=None, num_samples=1 is what argparse hands back
    # when no sampling flags are passed at all -- must never raise.
    validate_sampling_args(None, None, None, 1)


def test_validate_sampling_args_accepts_boundary_values():
    # top_k=0 means "disabled" (not "invalid"); top_p=1.0 means "no nucleus filtering".
    validate_sampling_args(0.8, 1.0, 0, 3)


def test_validate_sampling_args_rejects_zero_num_samples():
    with pytest.raises(ValueError, match="--num-samples"):
        validate_sampling_args(None, None, None, 0)


def test_validate_sampling_args_rejects_negative_num_samples():
    with pytest.raises(ValueError, match="--num-samples"):
        validate_sampling_args(None, None, None, -1)


def test_validate_sampling_args_rejects_zero_temperature():
    with pytest.raises(ValueError, match="--temperature"):
        validate_sampling_args(0.0, None, None, 1)


def test_validate_sampling_args_rejects_negative_temperature():
    with pytest.raises(ValueError, match="--temperature"):
        validate_sampling_args(-0.5, None, None, 1)


def test_validate_sampling_args_rejects_top_p_above_one():
    with pytest.raises(ValueError, match="--top-p"):
        validate_sampling_args(None, 1.5, None, 1)


def test_validate_sampling_args_rejects_top_p_of_zero():
    with pytest.raises(ValueError, match="--top-p"):
        validate_sampling_args(None, 0.0, None, 1)


def test_validate_sampling_args_rejects_negative_top_k():
    with pytest.raises(ValueError, match="--top-k"):
        validate_sampling_args(None, None, -1, 1)


# --- main(): bad sampling flags must fail fast, before writing anything, before model load --

def test_main_fails_clearly_for_invalid_temperature_before_touching_the_model(
    tmp_path, monkeypatch, capsys
):
    """Sampling-flag validation runs BEFORE resolve_model_dir(), so a bad --temperature is
    reported as a --temperature problem even when --model also points nowhere useful --
    the user should not have to fix the model path just to discover the flag was the issue.
    """
    import scripts.generate_samples as gs

    missing = tmp_path / "no-such-model-dir"
    label = "pytest-bad-temperature-must-not-be-written"
    out_path = ROOT / "docs" / "measurements" / f"samples-{label}.md"
    monkeypatch.setattr(
        sys, "argv",
        ["generate_samples.py", "--model", str(missing), "--label", label,
         "--temperature", "-1"],
    )

    rc = gs.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "ERROR" in captured.err
    assert "--temperature" in captured.err
    assert not out_path.exists()


def test_main_fails_clearly_for_zero_num_samples(tmp_path, monkeypatch, capsys):
    import scripts.generate_samples as gs

    missing = tmp_path / "no-such-model-dir"
    label = "pytest-bad-num-samples-must-not-be-written"
    out_path = ROOT / "docs" / "measurements" / f"samples-{label}.md"
    monkeypatch.setattr(
        sys, "argv",
        ["generate_samples.py", "--model", str(missing), "--label", label,
         "--num-samples", "0"],
    )

    rc = gs.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "ERROR" in captured.err
    assert "--num-samples" in captured.err
    assert not out_path.exists()


def test_main_passes_valid_sampling_flags_through_to_the_model_check(
    tmp_path, monkeypatch, capsys
):
    """Valid sampling flags must clear validate_sampling_args() cleanly and fall through to
    the next check (the model directory) -- confirming the new validation step doesn't
    misfire on legitimate input. The failure that surfaces here must be about the missing
    model, not about any sampling flag.
    """
    import scripts.generate_samples as gs

    missing = tmp_path / "no-such-model-dir"
    label = "pytest-valid-sampling-flags-must-not-be-written"
    out_path = ROOT / "docs" / "measurements" / f"samples-{label}.md"
    monkeypatch.setattr(
        sys, "argv",
        ["generate_samples.py", "--model", str(missing), "--label", label,
         "--temperature", "0.8", "--top-p", "0.95", "--top-k", "40", "--num-samples", "2"],
    )

    rc = gs.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "no such directory" in captured.err
    assert not out_path.exists()


# --- main(): the missing/invalid model path must fail fast, before writing anything ----

def test_main_fails_clearly_and_writes_nothing_for_a_missing_model_directory(
    tmp_path, monkeypatch, capsys
):
    import scripts.generate_samples as gs

    missing = tmp_path / "no-such-model-dir"
    label = "pytest-missing-model-must-not-be-written"
    out_path = ROOT / "docs" / "measurements" / f"samples-{label}.md"
    monkeypatch.setattr(
        sys, "argv",
        ["generate_samples.py", "--model", str(missing), "--label", label],
    )

    rc = gs.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "ERROR" in captured.err
    # The whole point: a missing model must never produce an empty/partial samples file
    # that could be mistaken for a real result.
    assert not out_path.exists()


def test_main_fails_clearly_and_writes_nothing_for_a_directory_without_config_json(
    tmp_path, monkeypatch, capsys
):
    import scripts.generate_samples as gs

    bad_model_dir = tmp_path / "not-a-converted-model"
    bad_model_dir.mkdir()
    label = "pytest-bad-model-dir-must-not-be-written"
    out_path = ROOT / "docs" / "measurements" / f"samples-{label}.md"
    monkeypatch.setattr(
        sys, "argv",
        ["generate_samples.py", "--model", str(bad_model_dir), "--label", label],
    )

    rc = gs.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "ERROR" in captured.err
    assert "config.json" in captured.err
    assert not out_path.exists()


# --- end-to-end smoke test: only meaningful with a real converted model present --------

def test_generation_smoke_test_against_real_hf_model(tmp_path, monkeypatch):
    """Full generation round-trip against a real converted HF model directory, if one is
    present locally (artifacts/hf/, the TinyStories-era v2 model). Output is redirected to
    tmp_path (by monkeypatching the module's ROOT) so this never writes into the real
    docs/measurements/ directory as a side effect of running the test suite.

    Explicitly skipped -- with a reason that shows up in the pytest report -- when no such
    model is present, which is the normal state for this repo's test environments (no model
    exists yet for the new corpus; artifacts/hf/ is gitignored and machine-local).
    """
    hf_dir = ROOT / "artifacts" / "hf"
    if not (hf_dir / "config.json").is_file():
        pytest.skip(
            "no converted model at artifacts/hf/ in this environment -- generate_samples.py's "
            "parsing/validation logic is still covered by the tests above without one"
        )

    import scripts.generate_samples as gs

    monkeypatch.setattr(gs, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["generate_samples.py", "--model", str(hf_dir), "--label", "pytest-smoke",
         "--max-new-tokens", "5"],
    )

    rc = gs.main()
    assert rc == 0

    out_path = tmp_path / "docs" / "measurements" / "samples-pytest-smoke.md"
    assert out_path.is_file()
    text = out_path.read_text(encoding="utf-8")
    for prompt in load_prompts():
        assert prompt["id"] in text
    # The header must say "greedy" and spell out every knob explicitly (as "n/a") whenever
    # no sampling flag was passed -- a samples file must never be mistakable for a sampled
    # run. See the module docstring and validate_sampling_args()'s callers in main().
    assert "greedy (temperature=n/a, top_p=n/a, top_k=n/a)" in text
    assert "1 sample(s)/prompt" in text


# --- sampling behavior: only meaningful with a real converted model present ------------

def test_num_samples_labels_each_completion_when_sampling(tmp_path, monkeypatch):
    """--num-samples > 1 together with a sampling flag must emit N labeled completions per
    prompt (not just one), so a human reviewer can compare them side by side.
    """
    hf_dir = ROOT / "artifacts" / "hf"
    if not (hf_dir / "config.json").is_file():
        pytest.skip("no converted model at artifacts/hf/ in this environment")

    import scripts.generate_samples as gs

    monkeypatch.setattr(gs, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["generate_samples.py", "--model", str(hf_dir), "--label", "pytest-num-samples",
         "--max-new-tokens", "5", "--temperature", "0.8", "--top-p", "0.95",
         "--num-samples", "2", "--seed", "0"],
    )

    rc = gs.main()
    assert rc == 0

    out_path = tmp_path / "docs" / "measurements" / "samples-pytest-num-samples.md"
    text = out_path.read_text(encoding="utf-8")
    assert "sampling (temperature=0.8, top_p=0.95, top_k=0)" in text
    assert "2 sample(s)/prompt" in text
    for prompt in load_prompts():
        assert f"### {prompt['id']}" in text
    assert text.count("sample 1/2:") == len(load_prompts())
    assert text.count("sample 2/2:") == len(load_prompts())


def test_sampling_is_reproducible_given_a_seed(tmp_path, monkeypatch):
    """Same seed + same sampling flags => byte-identical output. Without this, a sampled
    samples file would be un-diffable and un-reviewable run to run -- the whole point of
    recording a seed in the header (see the module docstring) is that it lets someone else
    reproduce exactly what was read.
    """
    hf_dir = ROOT / "artifacts" / "hf"
    if not (hf_dir / "config.json").is_file():
        pytest.skip("no converted model at artifacts/hf/ in this environment")

    import scripts.generate_samples as gs

    argv = [
        "generate_samples.py", "--model", str(hf_dir), "--label", "pytest-repro",
        "--max-new-tokens", "8", "--temperature", "0.9", "--top-p", "0.9", "--top-k", "40",
        "--num-samples", "2", "--seed", "1234",
    ]
    out_rel = Path("docs") / "measurements" / "samples-pytest-repro.md"

    run_a_dir = tmp_path / "a"
    monkeypatch.setattr(gs, "ROOT", run_a_dir)
    monkeypatch.setattr(sys, "argv", argv)
    assert gs.main() == 0
    text_a = (run_a_dir / out_rel).read_text(encoding="utf-8")

    run_b_dir = tmp_path / "b"
    monkeypatch.setattr(gs, "ROOT", run_b_dir)
    monkeypatch.setattr(sys, "argv", argv)
    assert gs.main() == 0
    text_b = (run_b_dir / out_rel).read_text(encoding="utf-8")

    assert text_a == text_b


def test_default_flags_reproduce_the_committed_v1_samples_file_body(tmp_path, monkeypatch):
    """Regenerating with ONLY --model/--label (no sampling flags) against the exact model
    and settings that produced the committed docs/measurements/samples-tt-tnt-v1.md must
    reproduce that file's body byte-for-byte -- only the header line (which now spells out
    temperature/top_p/top_k/num-samples explicitly) is allowed to differ. This is the
    concrete regression test for the constraint that adding sampling controls must not
    change one byte of existing, already-reviewed greedy output.
    """
    hf_dir = ROOT / "artifacts" / "hf-tt-tnt-v1"
    if not (hf_dir / "config.json").is_file():
        pytest.skip("no converted model at artifacts/hf-tt-tnt-v1/ in this environment")

    committed = ROOT / "docs" / "measurements" / "samples-tt-tnt-v1.md"
    if not committed.is_file():
        pytest.skip("docs/measurements/samples-tt-tnt-v1.md is not present to diff against")

    import scripts.generate_samples as gs

    monkeypatch.setattr(gs, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["generate_samples.py", "--model", str(hf_dir), "--label", "tt-tnt-v1",
         "--seed", "0"],
    )
    assert gs.main() == 0

    fresh_lines = (tmp_path / "docs" / "measurements" / "samples-tt-tnt-v1.md").read_text(
        encoding="utf-8"
    ).splitlines()
    committed_lines = committed.read_text(encoding="utf-8").splitlines()

    # Line 0 is the title, line 2 is the "model: ... " header line (allowed to differ),
    # everything else -- every generated prompt/completion -- must match exactly.
    assert fresh_lines[0] == committed_lines[0]
    assert fresh_lines[3:] == committed_lines[3:]
