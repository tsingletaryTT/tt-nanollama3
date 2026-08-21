#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does ttml's SFTTrainer train OUR model with a loss mask? Eight hand-made examples.

Front-loaded deliberately (Task 2 of the improv-thinking plan). Every later task assumes
this path works, and this repo has only ever trained through train/run.py --
ttml.trainers.SFTTrainer is an unexercised code path here. Failing here changes the plan,
not the code: the fallback (full-sequence loss, no masking) is a design decision, not a
workaround this script should reach for.

Constructor substitution (see Step 3 in the plan / task brief): the plan's expected entry
point, ``ttml.models.llama.create_llama_from_config``, does not exist in the installed
ttml (verified: ``dir(ttml.models.llama)`` has no such name -- it has ``Llama``,
``LlamaConfig``, ``create_cpp_llama_model``, but no ``create_llama_from_config``). This
repo's own train/run.py builds its Llama via ``train.model.create_model(yaml_config,
transformer_config)`` (see train/model.py's module docstring and its ``create_model``
function) precisely because that wrapper's ``TtTntLlama`` accepts a null attention mask on
the causal path, same requirement SFTTrainer has (``attention_mask=None`` lets the model
build a causal mask on its own). That is the substitution used below.

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
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        p_ids = tok.encode(PROMPT)
        c_ids = tok.encode(COMPLETION, add_special_tokens=False)

        # -100 on prompt positions is what tells sft_collate_fn which tokens to mask.
        example = {"input_ids": p_ids + c_ids,
                   "labels": [-100] * len(p_ids) + c_ids}
        examples = [example] * 8

        collate = partial(sft_collate_fn, max_seq_len=512, pad_token_id=tok.pad_token_id or 0)
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

        trainer = SFTTrainer(
            model=model, train_dataloader=loader, eval_dataloader=None,
            config=SFTConfig(max_steps=4, learning_rate=1e-5, seed=5489,
                             max_seq_len=512, save_interval=0, eval_interval=0),
            optimizer={"type": "AdamW", "lr": 1e-5, "weight_decay": 0.01},
        )
        trainer.train()
        print("SFTTrainer completed 4 masked steps")
    finally:
        # Mirror image of open_device_mesh -- also clears ttml's global mesh state.
        # Bypassing device teardown entirely triggers an abort in
        # MetalContext::destroy_all_instances (see train/run.py's identical finally block).
        ttml.close_device_mesh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
