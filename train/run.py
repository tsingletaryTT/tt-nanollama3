#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Train NanoLlama3 on Tenstorrent hardware.

We own this entrypoint because tt-train's own Python trainer does not work against the
current tree: ``examples/python/transformers/training.py`` imports a ``trainer`` module
that is not on its path, calls ``train()`` with an extra ``val_ids`` argument the signature
does not accept, and relies on a ``TrainingConfig`` that lacks the ``seq_len`` ``train()``
requires. Its data loader also hardcodes ``$TT_METAL_HOME/tt-train/data/shakespeare.txt``.

What we reuse from ttml (never reimplemented): ``TransformerModelFactory``,
``create_optimizer``, ``initialize_device``, ``set_seed``, and the ``train()`` loop itself.
What we supply: our corpus, our tokenizer, ``seq_len``, and a **real** validation loss —
ttml's ``train()`` fills ``val_losses`` with a copy of the training loss under a comment
calling it placeholder behavior, so a val number from it means nothing.

    python train/run.py --steps 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# CPython auto-prepends the *script's own directory* (train/) to sys.path before this
# module's top-level code runs. That directory holds train/tokenize.py, which then shadows
# the stdlib `tokenize` module for the rest of the process — and numpy imports `tokenize`
# transitively (via inspect/linecache) as soon as it loads, raising a confusing
# "partially initialized module 'inspect'" AttributeError. Strip it before importing numpy.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.config import build_yaml_config, run_config_from_yaml  # noqa: E402


def _default_tt_metal_home() -> str:
    return os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal"))


def _prepare_env(tt_metal_home: str, arch: str) -> None:
    """ttml needs all three of these before import; it aborts without RUNTIME_ROOT."""
    os.environ.setdefault("TT_METAL_HOME", tt_metal_home)
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", tt_metal_home)
    os.environ.setdefault("TT_METAL_ARCH_NAME", arch)
    os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
    sys.path.append(f"{tt_metal_home}/tt-train/sources/ttml")


def evaluate(model, val_ids: np.ndarray, cfg, batches: int = 10) -> float:
    """Real validation loss over ``batches`` sampled windows.

    ttml's train() does not compute this — it appends the last training loss and labels it
    val_loss. We run the model in eval mode over held-out tokens and average properly.
    """
    import ttml
    import ttnn
    # These live in different modules — build_causal_mask is in utils, not trainer.
    from ttml.common.trainer import get_batch_ttml
    from ttml.common.utils import build_causal_mask

    mask = ttml.autograd.Tensor.from_numpy(
        build_causal_mask(cfg.seq_len), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch_ttml(val_ids, cfg.seq_len, cfg.validation_batch_size, False)
        logits = model(x, mask)
        loss = ttml.ops.loss.cross_entropy_loss(logits, y, ttml.ops.ReduceType.MEAN)
        total += float(loss.to_numpy().mean())
        ttml.autograd.AutoContext.get_instance().reset_graph()
    model.train()
    return total / batches


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens-dir", default=str(ROOT / "artifacts" / "tokens"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--arch", default="blackhole", choices=["blackhole", "wormhole_b0"])
    p.add_argument("--tt-metal-home", default=_default_tt_metal_home())
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved config and exit without opening a device.")
    args = p.parse_args()

    tokens = Path(args.tokens_dir)
    train_path, val_path = tokens / "train_ids.npy", tokens / "val_ids.npy"
    if not train_path.is_file():
        print(f"ERROR: {train_path} not found. Run train/tokenize.py first.", file=sys.stderr)
        return 1

    model_config = Path(args.tt_metal_home) / "tt-train/configs/model_configs/nanollama3.yaml"
    if not model_config.is_file():
        print(f"ERROR: model config not found at {model_config}", file=sys.stderr)
        return 1

    yaml_config = build_yaml_config(
        str(ROOT / "artifacts" / "tokenizer"), str(model_config),
        batch_size=args.batch_size, max_steps=args.steps, eval_every=args.eval_every,
    )
    cfg = run_config_from_yaml(yaml_config)

    print(f"NanoLlama3 training — steps={cfg.steps} batch={cfg.batch_size} "
          f"seq_len={cfg.seq_len} arch={args.arch}")
    if args.dry_run:
        print("--dry-run set: not opening a device.")
        return 0

    _prepare_env(args.tt_metal_home, args.arch)

    import ttml  # noqa: E402
    from ttml.common.model_factory import TransformerModelFactory  # noqa: E402
    from ttml.common.trainer import train  # noqa: E402
    from ttml.common.utils import create_optimizer, initialize_device, set_seed  # noqa: E402

    train_ids = np.load(train_path)
    val_ids = np.load(val_path)
    print(f"  train tokens={len(train_ids):,}  val tokens={len(val_ids):,}")

    set_seed(yaml_config["training_config"]["seed"])
    initialize_device(yaml_config)

    # Everything from here to the end of the function runs with the device open, so it
    # all belongs inside this try — model/optimizer construction included. If either
    # raises (bad config, on-device OOM) before train()/evaluate() even start, the device
    # must still be closed in the finally below, or teardown aborts in
    # MetalContext::destroy_all_instances.
    try:
        model = TransformerModelFactory(yaml_config).create_model()
        optimizer = create_optimizer(model, yaml_config)

        # train() takes exactly (cfg, model, optim, train_ids, use_ddp, use_tp) — no val_ids.
        train_losses, _ = train(cfg, model, optimizer, train_ids, False, False)
        val_loss = evaluate(model, val_ids, cfg)
        print(f"\nfirst train loss : {train_losses[0]:.4f}")
        print(f"last  train loss : {train_losses[-1]:.4f}")
        print(f"real  val   loss : {val_loss:.4f}")
    finally:
        # Let ttml close the device — bypassing this triggers a teardown abort in
        # MetalContext::destroy_all_instances.
        ttml.autograd.AutoContext.get_instance().close_device()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
