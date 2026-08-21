#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does ttml's SFTTrainer train OUR model with a loss mask? Eight hand-made examples.

Front-loaded deliberately (Task 2 of the improv-thinking plan). Every later task assumes
this path works, and this repo has only ever trained through train/run.py --
ttml.trainers.SFTTrainer is an unexercised code path here. Failing here changes the plan,
not the code: the fallback (full-sequence loss, no masking) is a design decision, not a
workaround this script should reach for.

TILE ALIGNMENT (added after the first run of this script found a real blocker): the
first attempt crashed in loss.backward() with a TT_FATAL from ttml's SDPA backward kernel
("u_scaler shape mismatch") whenever the collated batch's sequence length T was not a
multiple of 32 (ttml's tile width) -- see task-2-report.md for the full trace and the
diagnostic that isolated this. The coordinator's ruling: fix it in the example builder,
not in a custom collate, so sft_collate_fn and everything downstream in ttml stay
untouched. This script now builds its 8 examples through
scripts.derive_traces.build_sft_examples, the same tile-aligning builder Task 5 will use,
so this smoke test exercises the actual production path rather than a hand-rolled one
that could silently drift from it.

Constructor substitution (see Step 3 in the plan / task brief): the plan's expected entry
point, ``ttml.models.llama.create_llama_from_config``, does not exist in the installed
ttml (verified: ``dir(ttml.models.llama)`` has no such name -- it has ``Llama``,
``LlamaConfig``, ``create_cpp_llama_model``, but no ``create_llama_from_config``). This
repo's own train/run.py builds its Llama via ``train.model.create_model(yaml_config,
transformer_config)`` (see train/model.py's module docstring and its ``create_model``
function) precisely because that wrapper's ``TtTntLlama`` accepts a null attention mask on
the causal path, same requirement SFTTrainer has (``attention_mask=None`` lets the model
build a causal mask on its own). That is the substitution used below.

WARM-START LOSS GUARD (added after Task 5 found what this smoke test's random-init
version could not catch): a smoke test that trains 4 steps from FRESH random weights
cannot tell a correct label convention from a wrong one -- both look like "high loss,
went down a bit" against a randomly initialised model. Task 5 shipped a full plan with
`scripts/derive_traces.py`'s labels in the wrong convention (same-position/HF-style
instead of ttml's pre-shifted `labels[t] == input_ids[t+1]`) for exactly this reason: it
trained from random weights here, then warm-started for real training, and only the
warm-started run's absolute loss (~11, not ~1.7-2.9) exposed the bug -- three tasks and
one full training run later than it should have (see task-5-report.md's addendum for the
full trail). This script now warm-starts from the same production checkpoint Task 5
trains from and asserts the FIRST step's loss is below a threshold chosen to sit
comfortably between "correct" (~1.7-2.9, this checkpoint's own validation range) and
"broken" (~11.5, what the label-shift bug produced) -- a random-init smoke could never
draw this line, because both outcomes start from the same place.

    gozer run --chips 1 --who "claude:improv" --reason "SFTTrainer masked smoke" -- \
        python3 scripts/smoke_sft_trainer.py
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT = "Lily found a needle. She showed it to her mother."
COMPLETION = " Her mother took the needle and sewed the button."

MODEL_YAML = ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue"
WARM_START_CKPT = (ROOT / "artifacts" / "checkpoints-v077-beta2-control"
                   / "tt_tnt_step00010764.pkl")

# Comfortably between this checkpoint's own validation range (~1.7-2.9, see
# task-5-report.md's addendum) and what a broken label convention produced (~11.5, close
# to ln(vocab_size)=10.37, i.e. random-init level). A pass here means the model is
# genuinely warm-started AND the example-building/masking path feeding it is sane; a
# failure here means one of those two things is broken, most likely the labels.
FIRST_STEP_LOSS_THRESHOLD = 5.0


class _FirstStepLoss:
    """Captures the loss reported at the FIRST optimizer step only.

    Duck-typed against ttml.trainers.callback.TrainerCallback's hooks (only the ones
    SFTTrainer actually calls are needed; the rest are no-ops). SFTTrainer calls
    ``cb.on_step_end(self, self.step, step_loss, lr)`` after every optimizer step --
    ``self.value`` latches on the first call and ignores the rest.
    """

    def __init__(self) -> None:
        self.value = None

    def on_train_begin(self, trainer) -> None:
        pass

    def on_after_forward(self, trainer, batch, loss) -> None:
        pass

    def on_after_backward(self, trainer, batch) -> None:
        pass

    def on_step_end(self, trainer, step, *args, **kwargs) -> None:
        if self.value is None and args:
            self.value = float(args[0])

    def on_eval_end(self, trainer, step, eval_loss) -> None:
        pass

    def on_before_optimizer_step(self, trainer) -> None:
        pass

    def on_save(self, trainer, step, path) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass


def _assert_stochastic_rounding(trainer) -> None:
    """Fail loudly if the built optimizer does not actually have stochastic rounding on.

    FIX 1 (task-6-report.md): this smoke's optimizer dict originally omitted
    ``stochastic_rounding`` entirely, which ``optimizer_registry.cpp`` defaults to
    ``false`` -- the exact same omission Task 5's real training arms shipped with, which
    is why THIS smoke never caught it: a 4-step run with a masked-loss assertion has no
    signal at all about whether 17 RMSNorm gammas moved. Reading the flag back out of the
    constructed optimizer's own state dict (not the input dict, which only proves intent)
    and refusing to proceed if it did not take.
    """
    state = trainer.optimizer.get_state_dict()
    enabled = bool(state.get("stochastic_rounding", False))
    print(f"optimizer stochastic_rounding (read back from optimizer state): {enabled}")
    if not enabled:
        raise RuntimeError(
            "stochastic_rounding is NOT enabled on this smoke's optimizer -- RMSNorm "
            "gamma parameters cannot move in bfloat16 (ulp at 1.0 is 0.0039, an order of "
            "magnitude larger than the ~3e-4 updates these gammas receive). Refusing to "
            "train rather than silently repeating FIX 1's bug; see task-6-report.md."
        )


def main() -> int:
    from transformers import AutoTokenizer

    import ttml  # noqa: F401  (import opens the UMD cluster -- must run under a gozer lease)
    import ttnn
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig, SFTTrainer

    # A device must be open before sft_collate_fn below (it builds on-device tensors via
    # ttml.autograd.Tensor.from_numpy). The plan's draft script never opened one at all --
    # a gap, not a deliberate omission.
    #
    # SECOND FINDING (beyond the missing call): train/run.py's own device-init helper,
    # ttml.common.utils.initialize_device(), calls AutoContext.open_device() directly and
    # does NOT populate ttml's module-global mesh. SFTTrainer.__init__ -> _build_loss_fn()
    # calls ttml.mesh(), which reads that separate global and raises "Device mesh is not
    # initialized" even though AutoContext genuinely has a device open (verified: first
    # attempt used initialize_device({}) and hit exactly this, with the loss_mask ratio
    # line printed correctly beforehand, i.e. the device WAS open). The two device-init
    # entry points in this ttml -- AutoContext.open_device (what this repo's train/run.py
    # uses) and ttml.open_device_mesh (what SFTTrainer's mesh-aware code paths need) --
    # are not equivalent; only the latter also sets the global ttml.mesh(). Using
    # open_device_mesh here, which is the one SFTTrainer requires.
    yaml_config: dict = {}
    ttml.open_device_mesh((1, 1))
    try:
        from scripts.derive_traces import build_sft_examples

        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        pad_token_id = tok.pad_token_id or 0

        # with_think=False: this smoke test is about the mask + SFTTrainer + model
        # combination in general, not the think-block feature specifically (that is
        # covered by tests/test_derive_traces.py). "think" is unused when with_think is
        # False but build_sft_examples still reads the key off the trace dict, so it must
        # be present.
        trace = {"prefix": PROMPT, "think": "", "continuation": COMPLETION}
        examples = build_sft_examples([trace] * 8, tok, with_think=False,
                                      pad_token_id=pad_token_id)

        collate = partial(sft_collate_fn, max_seq_len=512, pad_token_id=pad_token_id)
        loader = InMemoryDataloader(examples, batch_size=4, collate_fn=collate, shuffle=False)

        # THE CONTRACT: loss_mask.sum() must equal B*T, or the masked mean is silently wrong.
        # Read B, T from the mask's own shape ([B, 1, T, 1]) rather than assuming batch_size,
        # matching how ttml's own SFTTrainer._compute_loss validates this on the first batch.
        batch = next(iter(loader))
        mask = batch.loss_mask.to_numpy(ttnn.DataType.FLOAT32)
        b, _, t, _ = mask.shape
        ratio = float(mask.sum()) / (b * t)
        print(f"loss_mask.sum()={mask.sum():.2f}  B*T={b * t}  "
              f"ratio={ratio:.4f}   (contract: ratio == 1.0)")

        # --- model construction ---
        # ttml.models.llama.create_llama_from_config does not exist (verified above the
        # docstring). Substituting this repo's own constructor, train.model.create_model.
        from train.model import create_model

        model_yaml = yaml.safe_load(MODEL_YAML.read_text())
        transformer_config = model_yaml["transformer_config"]
        model = create_model(yaml_config, transformer_config)

        # --- warm start (see the module docstring's WARM-START LOSS GUARD section) ---
        from train.enthusiasts import warm_start

        warm_summary = warm_start(model, WARM_START_CKPT,
                                  transformer_config=transformer_config,
                                  yaml_config=yaml_config, moe_block_indices=[])
        print(f"warm start: {warm_summary}")

        first_step_loss = _FirstStepLoss()
        trainer = SFTTrainer(
            model=model, train_dataloader=loader, eval_dataloader=None,
            config=SFTConfig(max_steps=4, learning_rate=1e-5, seed=5489,
                             max_seq_len=512, save_interval=0, eval_interval=0),
            optimizer={"type": "AdamW", "lr": 1e-5, "weight_decay": 0.01,
                       "stochastic_rounding": True},
            callbacks=[first_step_loss],
        )
        _assert_stochastic_rounding(trainer)
        trainer.train()
        print("SFTTrainer completed 4 masked steps")

        if first_step_loss.value is None:
            raise RuntimeError("smoke never recorded a first-step loss -- on_step_end "
                               "was never called; SFTTrainer's callback contract may "
                               "have changed")
        print(f"first step loss: {first_step_loss.value:.4f}  "
              f"(threshold: < {FIRST_STEP_LOSS_THRESHOLD})")
        if first_step_loss.value >= FIRST_STEP_LOSS_THRESHOLD:
            raise AssertionError(
                f"first-step loss {first_step_loss.value:.4f} is >= "
                f"{FIRST_STEP_LOSS_THRESHOLD} on a WARM-STARTED model. A correctly "
                f"warm-started model on ordinary text should start near this "
                f"checkpoint's own validation loss (~1.7-2.9), not near "
                f"ln(vocab_size)~10.37 (random-init level). The most likely cause is a "
                f"label-shift convention mismatch in scripts/derive_traces.py's "
                f"build_sft_examples / _sft_example_unaligned: ttml's cross_entropy_loss "
                f"(via sft_collate_fn) expects labels[t] == input_ids[t+1] (pre-shifted "
                f"for next-token prediction, same as ttml.common.data.get_batch's "
                f"y = split_ids[i+1:...]), NOT same-position HF-style labels[t] == "
                f"input_ids[t]. See task-5-report.md's addendum for the original "
                f"discovery of this exact failure mode (loss 11.5 -> 1.7 from fixing "
                f"exactly this).")
    finally:
        # Mirror image of open_device_mesh -- also clears ttml's global mesh state.
        # Bypassing device teardown entirely triggers an abort in
        # MetalContext::destroy_all_instances (see train/run.py's identical finally block).
        ttml.close_device_mesh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
