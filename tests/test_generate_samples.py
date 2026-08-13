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

from scripts.generate_samples import PROMPTS, load_prompts, resolve_model_dir  # noqa: E402


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
