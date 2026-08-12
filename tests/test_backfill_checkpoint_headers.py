# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""scripts/backfill_checkpoint_headers.py: pure-CPU header rewrite. No ttml, no hardware.

Exercises the rewrite against synthetic fake checkpoints built with plain pickle (never
against the real artifacts/checkpoints/*.pkl -- those are the only trained artifacts that
exist for this project and this test suite must not risk them).
"""

import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from convert.checkpoint_reader import read_checkpoint_meta
from train.checkpoint import validate_header

ROOT = Path(__file__).resolve().parent.parent

# Loaded by file path, not `import scripts.backfill_checkpoint_headers`: this environment
# has an unrelated project's own `scripts/__init__.py` package earlier on sys.path (an
# editable install of a sibling repo), which shadows any bare `scripts` package name from
# this repo. Loading by explicit path sidesteps that collision entirely.
_SCRIPT_PATH = ROOT / "scripts" / "backfill_checkpoint_headers.py"
_spec = importlib.util.spec_from_file_location("backfill_checkpoint_headers", _SCRIPT_PATH)
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)


def _fake_checkpoint(tmp_path: Path, step: int, *, header_extra=None, dtype="BFLOAT16") -> Path:
    """A minimal file matching ttml.checkpointing's on-disk shape: old-schema header
    (total_tokens, no batch_size/tokens_seen) plus a handful of "tensor" records."""
    header = {
        "format": 1,
        "step": step,
        "vocab_size": 32000,
        "seq_len": 256,
        "model_config_path": "/home/ttuser/tt-metal/tt-train/configs/model_configs/nanollama3.yaml",
        "tokenizer_dir": "/home/ttuser/code/tt-nanollama3/artifacts/tokenizer",
        "total_tokens": 127_635_889,
        "created_at": "2026-08-11T17:22:49+00:00",
    }
    if header_extra:
        header.update(header_extra)
    manifest = {
        "model": {"named_parameters": {
            "llama/fc/weight": {"layout": "TILE", "dtype": dtype},
            "llama/ln_fc/gamma": {"layout": "TILE", "dtype": dtype},
        }},
    }
    path = tmp_path / f"nanollama3_step{step:08d}.pkl"
    tensor_payload = b"x" * 4096  # stand-in tensor bytes; content is irrelevant, length isn't
    with open(path, "wb") as f:
        pickle.dump({"format": 1, "header": header, "manifest": manifest}, f)
        pickle.dump(tensor_payload, f)
        pickle.dump(tensor_payload, f)
    return path


def test_backfill_renames_total_tokens_and_derives_tokens_seen(tmp_path):
    path = _fake_checkpoint(tmp_path, step=3000)
    new_header = backfill.backfill_one(path)
    assert "total_tokens" not in new_header
    assert new_header["corpus_tokens"] == 127_635_889
    assert new_header["batch_size"] == backfill.KNOWN_BATCH_SIZE
    assert new_header["tokens_seen"] == 3000 * backfill.KNOWN_BATCH_SIZE * 256


def test_backfill_adds_architecture_fields(tmp_path):
    path = _fake_checkpoint(tmp_path, step=500)
    new_header = backfill.backfill_one(path)
    assert new_header["intermediate_dim"] == 1024
    assert new_header["weight_tying"] is True
    assert new_header["rms_norm_eps"] == 1e-5
    assert new_header["weights_dtype"] == "bfloat16"
    assert new_header["transformer_config"]["embedding_dim"] == 384


def test_backfilled_header_passes_validate_header(tmp_path):
    path = _fake_checkpoint(tmp_path, step=1000)
    new_header = backfill.backfill_one(path)
    validate_header(new_header)  # must not raise


def test_backfill_preserves_step_and_other_original_fields(tmp_path):
    path = _fake_checkpoint(tmp_path, step=2500)
    new_header = backfill.backfill_one(path)
    assert new_header["step"] == 2500
    assert new_header["vocab_size"] == 32000
    assert new_header["created_at"] == "2026-08-11T17:22:49+00:00"


def test_backfill_preserves_tensor_bytes_exactly(tmp_path):
    path = _fake_checkpoint(tmp_path, step=1500)
    before_header, before_manifest = read_checkpoint_meta(path)
    with open(path, "rb") as f:
        pickle.load(f)  # skip record 0
        before_tensor_a = pickle.load(f)
        before_tensor_b = pickle.load(f)

    backfill.backfill_one(path)

    after_header, after_manifest = read_checkpoint_meta(path)
    assert after_manifest == before_manifest  # untouched
    with open(path, "rb") as f:
        pickle.load(f)  # skip the rewritten record 0
        after_tensor_a = pickle.load(f)
        after_tensor_b = pickle.load(f)
    assert after_tensor_a == before_tensor_a
    assert after_tensor_b == before_tensor_b


def test_backfill_is_idempotent(tmp_path):
    """Running the script twice against an already-migrated file must not double-apply
    the rename or corrupt tokens_seen -- the script must be safe to re-run."""
    path = _fake_checkpoint(tmp_path, step=3000)
    once = backfill.backfill_one(path)
    twice = backfill.backfill_one(path)
    assert once == twice


def test_dry_run_writes_nothing(tmp_path):
    path = _fake_checkpoint(tmp_path, step=3000)
    before_bytes = path.read_bytes()
    new_header = backfill.backfill_one(path, dry_run=True)
    assert path.read_bytes() == before_bytes
    assert new_header["corpus_tokens"] == 127_635_889  # still computed, just not written


def test_refuses_to_stamp_bfloat16_over_a_non_bfloat16_manifest(tmp_path):
    path = _fake_checkpoint(tmp_path, step=3000, dtype="FLOAT32")
    with pytest.raises(ValueError, match="not BFLOAT16"):
        backfill.backfill_one(path)


def test_crash_mid_copy_leaves_the_original_checkpoint_intact(tmp_path, monkeypatch):
    """Simulate a short/failed tensor-byte copy and confirm the original file is
    untouched -- os.replace must never be reached when the copied length is wrong."""
    path = _fake_checkpoint(tmp_path, step=3000)
    original_bytes = path.read_bytes()

    real_open = open

    def _truncating_open(file, mode="r", *a, **kw):
        f = real_open(file, mode, *a, **kw)
        if mode == "rb" and str(file) == str(path):
            real_read = f.read

            def _short_read(n=-1):
                data = real_read(n)
                return data[: len(data) // 2] if data else data

            f.read = _short_read
        return f

    monkeypatch.setattr(backfill, "open", _truncating_open, raising=False)
    with pytest.raises(RuntimeError, match="refusing to replace the original"):
        backfill.backfill_one(path)
    monkeypatch.undo()

    assert path.read_bytes() == original_bytes  # untouched
    assert path.with_name(path.name + ".tmp").exists()  # evidence left behind for inspection


def test_main_backfills_a_directory_and_reports_success(tmp_path, capsys):
    _fake_checkpoint(tmp_path, step=500)
    _fake_checkpoint(tmp_path, step=1000)
    # main() parses sys.argv directly; drive it the same way the CLI does.
    argv = sys.argv
    sys.argv = ["backfill_checkpoint_headers.py", str(tmp_path)]
    try:
        rc = backfill.main()
    finally:
        sys.argv = argv
    assert rc == 0
    out = capsys.readouterr().out
    assert "backfilled nanollama3_step00000500.pkl" in out
    assert "backfilled nanollama3_step00001000.pkl" in out
    assert "verified nanollama3_step00000500.pkl" in out
    assert "all 2 checkpoint(s) verified" in out


def test_main_exits_nonzero_for_an_empty_directory(tmp_path):
    argv = sys.argv
    sys.argv = ["backfill_checkpoint_headers.py", str(tmp_path)]
    try:
        rc = backfill.main()
    finally:
        sys.argv = argv
    assert rc == 1


def test_backfill_script_imports_no_tenstorrent():
    """Pure-CPU per the brief: no ttml, no ttnn, even transitively.

    Loaded by file path in the subprocess too, for the same shadowing reason as the
    module-level load above.
    """
    probe = (
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location("
        "'backfill_checkpoint_headers', 'scripts/backfill_checkpoint_headers.py'); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "bad=[mod for mod in ('ttnn', 'ttml') if mod in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True, cwd=str(ROOT),
    )
    assert out.stdout.strip() == "", f"backfill script pulled in: {out.stdout.strip()}"
