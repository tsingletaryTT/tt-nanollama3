# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""convert.checkpoint_reader: pure-CPU header + manifest reads. No ttml, no hardware."""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from convert.checkpoint_reader import read_checkpoint_meta, read_record0, read_tensors, tensor_names

ROOT = Path(__file__).resolve().parent.parent
REAL_CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"


def _fake_checkpoint(tmp_path: Path, *, header=None, manifest=None) -> Path:
    """Write a minimal file matching ttml.checkpointing's on-disk shape, without ttml."""
    record = {
        "format": 1,
        "header": header if header is not None else {"step": 42},
        "manifest": manifest if manifest is not None else {
            "model": {"named_parameters": {
                "a": {"dtype": "BFLOAT16"}, "b": {"dtype": "BFLOAT16"},
            }},
        },
    }
    path = tmp_path / "fake.pkl"
    with open(path, "wb") as f:
        pickle.dump(record, f)
        pickle.dump("tensor-a-bytes", f)  # stand-in for a gathered tensor record
        pickle.dump("tensor-b-bytes", f)
    return path


def _fake_checkpoint_with_tensors(tmp_path: Path, names_and_arrays) -> Path:
    """Write a checkpoint whose model tensor records appear in exactly the given order.

    Unlike ``_fake_checkpoint`` (which uses string stand-ins), this writes real numpy
    arrays so a caller pairing the wrong name to the wrong array is unambiguous by shape.
    """
    manifest = {"model": {"named_parameters": {name: {} for name, _array in names_and_arrays}}}
    record = {"format": 1, "header": {"step": 1}, "manifest": manifest}
    path = tmp_path / "fake_tensors.pkl"
    with open(path, "wb") as f:
        pickle.dump(record, f)
        for _name, array in names_and_arrays:
            pickle.dump(array, f)
    return path


def test_read_checkpoint_meta_returns_header_and_manifest(tmp_path):
    path = _fake_checkpoint(tmp_path, header={"step": 7, "vocab_size": 32000})
    header, manifest = read_checkpoint_meta(path)
    assert header == {"step": 7, "vocab_size": 32000}
    assert "model" in manifest


def test_read_record0_offset_lands_exactly_before_first_tensor_record(tmp_path):
    path = _fake_checkpoint(tmp_path)
    _record, offset = read_record0(path)
    with open(path, "rb") as f:
        f.seek(offset)
        first_tensor = pickle.load(f)
    assert first_tensor == "tensor-a-bytes"


def test_rejects_a_file_that_is_not_a_checkpoint(tmp_path):
    path = tmp_path / "not_a_checkpoint.pkl"
    with open(path, "wb") as f:
        pickle.dump({"nope": True}, f)
    with pytest.raises(ValueError, match="not a ttml checkpoint"):
        read_checkpoint_meta(path)


def test_rejects_unreadable_garbage(tmp_path):
    path = tmp_path / "garbage.pkl"
    path.write_bytes(b"this is not a pickle stream at all")
    with pytest.raises(ValueError, match="could not read checkpoint record 0"):
        read_checkpoint_meta(path)


def test_tensor_names_flattens_named_parameters():
    manifest = {"model": {"named_parameters": {
        "llama/fc/weight": {}, "llama/blocks/0/ln/gamma": {},
    }}}
    assert tensor_names(manifest, group="model") == sorted(
        ["llama/fc/weight", "llama/blocks/0/ln/gamma"]
    )


def test_tensor_names_recurses_nested_optimizer_state():
    """The optimizer group isn't a flat named_parameters leaf -- AdamW nests moments
    under exp_avg/exp_avg_sq, each of which is its own named_parameters leaf."""
    manifest = {
        "optimizer": {
            "exp_avg": {"named_parameters": {"llama/fc/weight": {}}},
            "exp_avg_sq": {"named_parameters": {"llama/fc/weight": {}}},
        }
    }
    names = tensor_names(manifest, group="optimizer")
    assert names == ["llama/fc/weight", "llama/fc/weight"]  # one per sub-state, not deduped


def test_tensor_names_rejects_unknown_group():
    with pytest.raises(ValueError, match="no 'bogus' group"):
        tensor_names({"model": {}}, group="bogus")


def test_read_tensors_preserves_declaration_order_not_sorted(tmp_path):
    """The trap: named_parameters keys are deliberately NOT alphabetical here ("zeta"
    sorts after "mid" and "alpha"). If read_tensors paired sorted names against the record
    stream -- e.g. by using tensor_names(), which sorts -- every array would come back
    attached to the wrong name. Distinct shapes per array make a mis-pairing unmistakable.
    """
    zeta = np.zeros((2, 2), dtype=np.float32)
    alpha = np.ones((3, 3), dtype=np.float32)
    mid = np.full((4, 4), 7.0, dtype=np.float32)
    declared = [("zeta", zeta), ("alpha", alpha), ("mid", mid)]
    path = _fake_checkpoint_with_tensors(tmp_path, declared)

    got = list(read_tensors(path, group="model"))

    assert [name for name, _array in got] == ["zeta", "alpha", "mid"]  # declaration order
    for (name, array), (expected_name, expected_array) in zip(got, declared):
        assert name == expected_name
        assert np.array_equal(array, expected_array)


def test_read_tensors_is_a_lazy_generator(tmp_path):
    """Yields one tensor at a time (peak memory ~one tensor), not a materialized list."""
    array = np.arange(4, dtype=np.float32)
    path = _fake_checkpoint_with_tensors(tmp_path, [("only", array)])

    gen = read_tensors(path, group="model")

    assert hasattr(gen, "__next__"), "read_tensors should return an iterator, not a list"
    name, got = next(gen)
    assert name == "only"
    assert np.array_equal(got, array)


def test_read_tensors_rejects_unknown_group(tmp_path):
    path = _fake_checkpoint_with_tensors(tmp_path, [("a", np.zeros((1,)))])
    with pytest.raises(ValueError, match="no 'bogus' group"):
        list(read_tensors(path, group="bogus"))


@pytest.mark.skipif(
    not REAL_CHECKPOINT_DIR.exists() or not list(REAL_CHECKPOINT_DIR.glob("*.pkl")),
    reason="no real checkpoint present under artifacts/checkpoints/",
)
def test_read_tensors_matches_manifest_declaration_order_on_real_checkpoint():
    """Same trap, against the real artifact: the manifest's named_parameters order is not
    alphabetical (block 5's q_linear is declared before block 0's), so read_tensors must
    walk the manifest dict directly rather than zip against a sorted name list."""
    path = sorted(REAL_CHECKPOINT_DIR.glob("nanollama3_step*.pkl"))[0]
    _header, manifest = read_checkpoint_meta(path)
    declared_order = list(manifest["model"]["named_parameters"].keys())

    got = list(read_tensors(path, group="model"))

    assert [name for name, _array in got] == declared_order
    assert declared_order != sorted(declared_order)  # confirms this is a real ordering test
    # Shape sanity per the brief: fc/weight is the (vocab, hidden) embedding/lm_head tensor.
    by_name = dict(got)
    fc = by_name["llama/fc/weight"]
    while fc.ndim > 2 and fc.shape[0] == 1:
        fc = fc[0]
    assert fc.shape == (32000, 384)


@pytest.mark.skipif(
    not REAL_CHECKPOINT_DIR.exists() or not list(REAL_CHECKPOINT_DIR.glob("*.pkl")),
    reason="no real checkpoint present under artifacts/checkpoints/",
)
def test_reads_a_real_checkpoints_header_and_manifest():
    path = sorted(REAL_CHECKPOINT_DIR.glob("nanollama3_step*.pkl"))[0]
    header, manifest = read_checkpoint_meta(path)
    assert "step" in header
    names = tensor_names(manifest, group="model")
    assert len(names) == 50  # per the brief: "all 50 model tensors"
    assert all(names)  # no empty/falsy names


def test_convert_checkpoint_reader_imports_no_tenstorrent():
    """convert/ must run on a machine with no hardware and no tt-metal checkout.

    Checked in a subprocess: this test session may already have imported plenty, so
    inspecting our own sys.modules would prove nothing.
    """
    probe = (
        "import sys; import convert.checkpoint_reader; "
        "bad=[m for m in ('ttnn','ttml') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True, cwd=str(ROOT),
    )
    assert out.stdout.strip() == "", f"convert.checkpoint_reader pulled in: {out.stdout.strip()}"
