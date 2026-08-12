#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Convert a NanoLlama3 checkpoint into a loadable Hugging Face model directory.

Thin argparse CLI over ``convert.to_hf.convert_checkpoint``. Reads a ttml checkpoint
(pickle + numpy, no ttml/ttnn import required) and the trained tokenizer, and writes
``config.json``, ``model.safetensors``, and the tokenizer files into an output directory
that ``transformers.AutoModelForCausalLM.from_pretrained`` can load directly.

By default this picks the *newest* checkpoint under ``artifacts/checkpoints`` (highest
step number, from the ``nanollama3_step<N>.pkl`` naming convention) so a bare invocation
converts the most recently trained model:

    python scripts/convert_checkpoint.py                       # newest checkpoint -> artifacts/hf/
    python scripts/convert_checkpoint.py --checkpoint path.pkl  # a specific checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from convert.to_hf import convert_checkpoint  # noqa: E402
from train.paths import (  # noqa: E402
    ProtectedPathError,
    assert_not_protected,
    read_dir,
    shared_dir,
    write_dir,
)
from train.sizes import DEFAULT_SIZE, SIZES  # noqa: E402


def _newest_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the checkpoint with the highest ``step`` under ``checkpoint_dir``.

    Sorts by the numeric step embedded in the filename (``nanollama3_step00003000.pkl``)
    rather than by mtime or filename string order, so it's correct regardless of how the
    files were copied or touched.
    """
    paths = sorted(checkpoint_dir.glob("nanollama3_step*.pkl"),
                    key=lambda p: int(p.stem.split("step")[-1]))
    if not paths:
        raise FileNotFoundError(f"no nanollama3_step*.pkl checkpoints found under {checkpoint_dir}")
    return paths[-1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default=DEFAULT_SIZE, choices=sorted(SIZES),
                   help=f"Model size, which selects the artifact directories "
                        f"(default: {DEFAULT_SIZE}).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Checkpoint file to convert (default: newest under "
                        "--checkpoint-dir).")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Directory to pick the newest checkpoint from, when --checkpoint "
                        "is not given (default: artifacts/<size>/checkpoints, falling back "
                        "to the legacy flat artifacts/checkpoints when that is where the "
                        "artifacts actually are).")
    p.add_argument("--tokenizer-dir", type=Path, default=None,
                   help="Directory holding tokenizer.json etc. (default: "
                        "artifacts/tokenizer — shared by every size).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for the HF model "
                        "(default: artifacts/<size>/hf).")
    args = p.parse_args()

    # Reads tolerate the pre-registry flat layout so the baseline artifacts stay findable
    # where they are; writes are always per-size and are guarded, so a conversion cannot
    # overwrite the published HF artifact.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = read_dir(args.size, "checkpoints")
    if args.tokenizer_dir is None:
        args.tokenizer_dir = shared_dir("tokenizer")
    if args.out_dir is None:
        args.out_dir = write_dir(args.size, "hf")
    else:
        try:
            assert_not_protected(args.out_dir)
        except ProtectedPathError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    checkpoint = args.checkpoint or _newest_checkpoint(args.checkpoint_dir)
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    if not args.tokenizer_dir.is_dir():
        print(f"tokenizer directory not found: {args.tokenizer_dir}", file=sys.stderr)
        return 1

    print(f"converting {checkpoint} -> {args.out_dir}")
    config = convert_checkpoint(checkpoint, args.tokenizer_dir, args.out_dir)
    print(f"wrote {args.out_dir}: hidden_size={config['hidden_size']} "
          f"num_hidden_layers={config['num_hidden_layers']} "
          f"vocab_size={config['vocab_size']} "
          f"tie_word_embeddings={config['tie_word_embeddings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
