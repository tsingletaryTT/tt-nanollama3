#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Two paired SFT arms: think vs no-think. Same seed, same cut points, same steps.

PAIRED BY CONSTRUCTION: both arms consume the same traces in the same order, so per-step
noise cancels in the arm-vs-arm delta. The only difference is whether the completion
carries the think-block.

    gozer run --chips 1 --who "claude:improv" --reason "improv SFT arm think" -- \
        python3 scripts/train_improv.py --arm think
    gozer run --chips 1 --who "claude:improv" --reason "improv SFT arm nothink" -- \
        python3 scripts/train_improv.py --arm nothink

CORRECTIONS APPLIED relative to the original task-5 brief (see task-5-report.md for the
full trail -- these were established the hard way by earlier tasks in this plan):

1. ``build_sft_examples`` takes ``pad_token_id`` explicitly (it tile-aligns every example
   to a multiple of 32 -- ttml's SDPA backward kernel mismatches raw-T against
   tile-padded-T otherwise, see that function's docstring). Not bypassed here.

2. The model is built via ``train.model.create_model(yaml_config, transformer_config)``,
   exactly as ``scripts/smoke_sft_trainer.py`` (Task 2) verified working.
   ``ttml.models.llama.create_llama_from_config`` does not exist in the installed ttml.

3. ``ttml.open_device_mesh((1, 1))`` is required before constructing the SFTTrainer --
   this repo's own ``initialize_device()`` (AutoContext-only) does not populate the
   module-global mesh that ``ttml.mesh()`` reads, and SFTTrainer needs that global. A
   (1,1) mesh only -- never (1,2) or a 4-chip mesh (this board hard-froze once on a
   4-chip open with no OOM and no kernel panic).

4. Warm start goes through ``train.enthusiasts.warm_start``, the same function
   ``train/run.py`` uses for every arm (dense or MoE) via its own ``--warm-start`` flag --
   NOT ``ttml.serialization.load_model``, which does not exist. For a dense model warm
   -starting from a dense checkpoint of identical shape, every parameter name is shared,
   so ``moe_block_indices=[]`` and the whole checkpoint transfers with nothing left fresh.
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.derive_traces import build_sft_examples  # noqa: E402

MODEL_YAML = ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue"
WARM_START_CKPT = (ROOT / "artifacts" / "checkpoints-v077-beta2-control"
                   / "tt_tnt_step00010764.pkl")


class LossRecorder:
    """Records every step's (step, loss, lr) both in memory and to a JSONL file on disk.

    ``SFTTrainer.train()`` has no return value and only ever feeds a loss into the tqdm
    progress bar postfix -- a callback hooked on ``on_step_end`` is the only way to get
    the numbers back out. Duck-typed rather than subclassing ``TrainerCallback``: the
    trainer only ever calls the hooks it fires, so the other five no-ops are unnecessary.

    Logs EVERY step to ``curve_path`` (flushed after each write, so the trajectory
    survives a mid-run kill) -- not sampled every N. At 3000 steps this JSONL is a few
    hundred KB at most, and coarser sampling would be exactly the wrong economy here:
    the open question this instrumentation exists to answer is whether the loss falls in
    a fast early collapse (template memorisation) or a smooth decline throughout, and a
    sparse sample can misrepresent the early region where that distinction is sharpest.
    """

    def __init__(self, curve_path: Path) -> None:
        self.history: list[tuple[int, float]] = []
        self.curve_path = curve_path
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = curve_path.open("w")

    def on_train_begin(self, trainer) -> None:
        pass

    def on_after_forward(self, trainer, batch, loss) -> None:
        pass

    def on_after_backward(self, trainer, batch) -> None:
        pass

    def on_step_end(self, trainer, step, *args, **kwargs) -> None:
        # SFTTrainer calls cb.on_step_end(self, self.step, step_loss, lr).
        if not args:
            return
        loss = float(args[0])
        lr = float(args[1]) if len(args) > 1 else None
        self.history.append((step, loss))
        self._fh.write(json.dumps({"step": step, "loss": loss, "lr": lr}) + "\n")
        self._fh.flush()

    def on_eval_end(self, trainer, step, eval_loss) -> None:
        pass

    def on_before_optimizer_step(self, trainer) -> None:
        pass

    def on_save(self, trainer, step, path) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass

    def close(self) -> None:
        self._fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["think", "nothink"], required=True)
    ap.add_argument("--traces", type=Path,
                    default=ROOT / "artifacts" / "improv" / "traces.jsonl")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    import ttml  # noqa: F401 -- import opens the UMD cluster; MUST run under a gozer lease
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig, SFTTrainer

    # See correction #3 above: SFTTrainer reads ttml.mesh(), which only open_device_mesh
    # populates (initialize_device()/AutoContext.open_device() does not).
    yaml_config: dict = {}
    ttml.open_device_mesh((1, 1))
    try:
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        pad_token_id = tok.pad_token_id or 0

        traces = [json.loads(line) for line in args.traces.open()]
        examples = build_sft_examples(traces, tok, with_think=(args.arm == "think"),
                                      pad_token_id=pad_token_id)
        print(f"arm={args.arm}  examples={len(examples):,}  seed={args.seed}  "
              f"traces={args.traces}")

        collate = partial(sft_collate_fn, max_seq_len=512, pad_token_id=pad_token_id)
        loader = InMemoryDataloader(examples, batch_size=args.batch_size,
                                    collate_fn=collate, shuffle=True, seed=args.seed)

        out = ROOT / "artifacts" / "improv" / f"ckpt-{args.arm}"
        out.mkdir(parents=True, exist_ok=True)

        # --- model construction (correction #2) ---
        from train.model import create_model

        model_yaml = yaml.safe_load(MODEL_YAML.read_text())
        transformer_config = model_yaml["transformer_config"]
        model = create_model(yaml_config, transformer_config)

        # --- warm start (correction #4) ---
        from train.enthusiasts import warm_start

        warm_summary = warm_start(
            model, WARM_START_CKPT,
            transformer_config=transformer_config, yaml_config=yaml_config,
            moe_block_indices=[],  # dense model: nothing is exempt from the strict match
        )
        print(f"warm start: {warm_summary}")

        curve_path = out / "loss_curve.jsonl"
        recorder = LossRecorder(curve_path)
        trainer = SFTTrainer(
            model=model, train_dataloader=loader, eval_dataloader=None,
            config=SFTConfig(max_steps=args.steps, learning_rate=args.lr, seed=args.seed,
                             max_seq_len=512, checkpoint_dir=str(out),
                             save_interval=1000, eval_interval=0, max_grad_norm=1.0),
            optimizer={"type": "AdamW", "lr": args.lr, "weight_decay": 0.01},
            callbacks=[recorder],
        )
        trainer.train()
        recorder.close()

        loss_start = recorder.history[0] if recorder.history else (None, None)
        loss_end = recorder.history[-1] if recorder.history else (None, None)
        print(f"loss: step {loss_start[0]}={loss_start[1]}  ->  "
              f"step {loss_end[0]}={loss_end[1]}")

        manifest = {
            "arm": args.arm,
            "paired_with": "nothink" if args.arm == "think" else "think",
            "seed": args.seed,
            "traces": str(args.traces),
            "n_examples": len(examples),
            "steps": args.steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_seq_len": 512,
            "mesh": "(1,1)",  # SFTTrainer opens its own mesh; ddp does not apply here
            "warm_start": str(WARM_START_CKPT.relative_to(ROOT)),
            "warm_start_summary": warm_summary,
            "checkpoint_dir": str(out.relative_to(ROOT)),
            "loss_curve": str(curve_path.relative_to(ROOT)),
            "loss_curve_note": "every step logged (not sampled)",
            "loss_step_start": loss_start[0],
            "loss_start": loss_start[1],
            "loss_step_end": loss_end[0],
            "loss_end": loss_end[1],
        }
        (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"wrote {out / 'train_manifest.json'}")
    finally:
        # Mirror image of open_device_mesh -- also clears ttml's global mesh state.
        # Skipping device teardown triggers an abort in MetalContext::destroy_all_instances.
        ttml.close_device_mesh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
