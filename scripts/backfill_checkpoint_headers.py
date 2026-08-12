#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Back-fill enriched headers into the six existing NanoLlama3 checkpoints, in place.

The header schema fix (see ``train/checkpoint.py`` and ``train/run.py``) applies only to
checkpoints written *after* it landed. The six checkpoints already on disk
(``nanollama3_step00000500.pkl`` .. ``nanollama3_step00003000.pkl``) were written under the
old schema and need the same fields added retroactively, or they fail to even load: `load()`
now validates the header before restoring any tensor (see `train/checkpoint.load`), and the
old header is missing fields the new `_REQUIRED` tuple demands (`corpus_tokens`, `batch_size`,
`tokens_seen`).

**This script is deliberately pure-CPU**: it uses only stdlib ``pickle`` to read and rewrite
record 0, and never imports ``ttml``/``ttnn``. A checkpoint file is one pickle record —
``{"format": ..., "header": {...}, "manifest": {...}}`` — followed by one pickled tensor
record per manifest entry (see ``convert/checkpoint_reader.py`` and
``~/tt-metal/tt-train/sources/ttml/ttml/checkpointing.py``). Record 0 is small (a handful of
scalars and short strings); nothing about reading or rewriting it needs a Tenstorrent device,
a tt-metal build, or even the ``ttml`` package to be importable.

**Safety.** Writes to ``<path>.tmp`` then ``os.replace()`` over the original, mirroring
``ttml.checkpointing.save_checkpoint``'s own atomicity, so a crash mid-write leaves the
original checkpoint file untouched. Tensor bytes after record 0 are copied through
byte-for-byte (never deserialized, never re-encoded) and the copy is verified against the
original file's measured length before the replace happens — if the copied byte count doesn't
match, the ``.tmp`` file is left on disk for inspection and the original is never touched.
These are the only trained artifacts that exist for this project; there is no acceptable
failure mode here other than "leave the original exactly as it was."

    python scripts/backfill_checkpoint_headers.py --dry-run   # preview, writes nothing
    python scripts/backfill_checkpoint_headers.py             # rewrite record 0 in place
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from convert.checkpoint_reader import read_record0  # noqa: E402
from train.checkpoint import validate_header  # noqa: E402

#: The single training run that produced all six existing checkpoints used this batch size
#: throughout: `python train/run.py --steps 3000 --save-every 500 --batch-size 64` (see
#: CLAUDE.md's "feat/checkpointing" run log). The old header never recorded batch_size --
#: that gap is exactly what this backfill (and Fix 4) exists to close -- so there is nothing
#: inside the checkpoint itself to read it back from. It is supplied here from the run's own
#: recorded command line, not guessed.
KNOWN_BATCH_SIZE = 64

#: ttml C++ defaults for a LlamaConfig with intermediate_dim left unset, evaluated at
#: embedding_dim=384 (nanollama3.yaml's setting, unchanged since these checkpoints were
#: written): round_up(4 * 384 * 2/3, 256) = round_up(1024, 256) = 1024.
#: ~/tt-metal/tt-train/sources/ttml/modules/llama_block.cpp:15-23.
BACKFILL_INTERMEDIATE_DIM = 1024

#: WeightTyingType::Enabled is LlamaConfig's default
#: (~/tt-metal/tt-train/sources/ttml/models/llama.hpp:35); nanollama3.yaml never overrides
#: it, and the training run's own printed "Weight tying: Enabled" banner
#: (.superpowers/sdd/2026-08-11-checkpointing/task-3-report.md) confirms it was on.
BACKFILL_WEIGHT_TYING = True

#: RMSNormLayer's default epsilon
#: (~/tt-metal/tt-train/sources/ttml/modules/rms_norm_module.hpp:17); never overridden here.
BACKFILL_RMS_NORM_EPS = 1e-5

#: Confirmed against each checkpoint's own manifest below (_assert_all_model_tensors_bfloat16)
#: rather than assumed -- this backfill refuses to stamp weights_dtype on a checkpoint whose
#: manifest disagrees.
BACKFILL_WEIGHTS_DTYPE = "bfloat16"

#: nanollama3.yaml's transformer_config, unchanged since these checkpoints were written.
#: Hardcoded here (rather than re-read from `model_config_path`) deliberately: that path
#: points outside this repo (`/home/ttuser/tt-metal/...`), is unversioned, and could have
#: been edited by a tt-metal update between when these checkpoints were made and when this
#: backfill runs -- reading it now would risk stamping checkpoints with whatever the file
#: says *today*, not what was true when they were trained.
BACKFILL_TRANSFORMER_CONFIG = {
    "model_type": "llama",
    "num_heads": 6,
    "num_groups": 3,
    "embedding_dim": 384,
    "dropout_prob": 0.0,
    "num_blocks": 6,
    "vocab_size": 32000,
    "max_sequence_length": 256,
    "runner_type": "default",
    "theta": 500000.0,
}


def _assert_all_model_tensors_bfloat16(manifest: Dict[str, Any], path: Path) -> None:
    """Verify the weights_dtype claim against this checkpoint's own manifest."""

    def walk(node: Any):
        if not isinstance(node, dict):
            return
        if "named_parameters" in node:
            for name, meta in node["named_parameters"].items():
                dtype = meta.get("dtype")
                if dtype != "BFLOAT16":
                    raise ValueError(
                        f"{path}: tensor {name!r} is {dtype}, not BFLOAT16 -- "
                        f"refusing to backfill weights_dtype={BACKFILL_WEIGHTS_DTYPE!r}"
                    )
            return
        for sub in node.values():
            walk(sub)

    walk(manifest.get("model", {}))


def _migrate_header(header: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new header with Fix 1's architecture fields and Fix 4's token-count
    rename/derivation applied. Idempotent: safe to run twice on an already-migrated file.
    """
    header = dict(header)  # never mutate the caller's dict in place

    # Fix 4: total_tokens (whole corpus) -> corpus_tokens (same value, honest name) +
    # batch_size (newly recorded) + tokens_seen (derived, the number that actually matters).
    if "total_tokens" in header:
        header["corpus_tokens"] = header.pop("total_tokens")
    header.setdefault("batch_size", KNOWN_BATCH_SIZE)
    header["tokens_seen"] = header["step"] * header["batch_size"] * header["seq_len"]

    # Fix 1: architecture facts that exist only as ttml C++ defaults or manifest facts,
    # recoverable from nothing the checkpoint references.
    header.setdefault("transformer_config", BACKFILL_TRANSFORMER_CONFIG)
    header.setdefault("intermediate_dim", BACKFILL_INTERMEDIATE_DIM)
    header.setdefault("weight_tying", BACKFILL_WEIGHT_TYING)
    header.setdefault("rms_norm_eps", BACKFILL_RMS_NORM_EPS)
    header.setdefault("weights_dtype", BACKFILL_WEIGHTS_DTYPE)
    return header


def backfill_one(path: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    """Rewrite record 0 of `path` with an enriched header; copy everything after it
    through byte-for-byte. Returns the new header (computed either way, written only if
    `dry_run` is False).
    """
    orig_size = path.stat().st_size
    record, tail_offset = read_record0(path)
    tail_length = orig_size - tail_offset

    _assert_all_model_tensors_bfloat16(record["manifest"], path)
    new_header = _migrate_header(record["header"])
    validate_header(new_header)  # fail before writing anything to disk

    if dry_run:
        return new_header

    new_record = dict(record)
    new_record["header"] = new_header

    # Write the enriched record 0 plus every tensor-record byte, copied through unchanged,
    # into a temp file first. `path` itself is never opened for writing -- only `os.replace`
    # below touches it, and only after the copy has been byte-counted and verified.
    tmp_path = path.with_name(path.name + ".tmp")
    with open(path, "rb") as src, open(tmp_path, "wb") as dst:
        pickle.dump(new_record, dst)
        src.seek(tail_offset)
        copied = 0
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)

    if copied != tail_length:
        # Deliberately leave tmp_path on disk rather than delete it -- it's the evidence
        # needed to diagnose what went wrong -- and deliberately do NOT call os.replace:
        # `path` (the original checkpoint) must be left exactly as it was.
        raise RuntimeError(
            f"{path}: copied {copied} tensor-record bytes, expected {tail_length} -- "
            f"refusing to replace the original. Left the (incomplete) rewrite at "
            f"{tmp_path} for inspection; {path} is untouched."
        )
    os.replace(tmp_path, path)
    return new_header


def _verify(paths, dry_run: bool) -> int:
    """Re-open each rewritten file with a fresh handle and confirm the new fields stuck
    and `step` didn't move. Skipped entirely in --dry-run, since nothing was written.
    """
    if dry_run:
        return 0
    failures = 0
    for path in paths:
        expected_step = int(path.stem.split("step")[-1])
        header, _manifest = read_record0(path)
        header = header["header"]
        if header["step"] != expected_step:
            print(f"FAIL {path.name}: header step {header['step']} != "
                  f"filename step {expected_step}", file=sys.stderr)
            failures += 1
            continue
        missing = [f for f in ("corpus_tokens", "batch_size", "tokens_seen",
                               "intermediate_dim", "weight_tying", "rms_norm_eps",
                               "weights_dtype") if f not in header]
        if missing:
            print(f"FAIL {path.name}: still missing {missing}", file=sys.stderr)
            failures += 1
            continue
        print(f"  verified {path.name}: step={header['step']} "
              f"tokens_seen={header['tokens_seen']:,} weight_tying={header['weight_tying']}")
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint_dir", nargs="?",
                   default=str(ROOT / "artifacts" / "checkpoints"),
                   help="Directory of nanollama3_step*.pkl files (default: "
                        "artifacts/checkpoints).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and validate the new header for every checkpoint "
                        "without writing anything.")
    args = p.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    paths = sorted(checkpoint_dir.glob("nanollama3_step*.pkl"))
    if not paths:
        print(f"no checkpoints found under {checkpoint_dir}", file=sys.stderr)
        return 1

    for path in paths:
        before_size = path.stat().st_size
        new_header = backfill_one(path, dry_run=args.dry_run)
        after_size = path.stat().st_size if not args.dry_run else before_size
        action = "would backfill" if args.dry_run else "backfilled"
        print(f"{action} {path.name}: step={new_header['step']} "
              f"corpus_tokens={new_header['corpus_tokens']:,} "
              f"batch_size={new_header['batch_size']} "
              f"tokens_seen={new_header['tokens_seen']:,} "
              f"(file size {before_size:,} -> {after_size:,})")

    failures = _verify(paths, args.dry_run)
    if failures:
        print(f"\n{failures} checkpoint(s) failed post-backfill verification", file=sys.stderr)
        return 1
    if not args.dry_run:
        print(f"\nall {len(paths)} checkpoint(s) verified: new header fields present, "
              "step unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
