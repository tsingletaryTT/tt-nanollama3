#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Interactive completion REPL for the converted tt-tnt model. CPU only.

**This is a completion model, not a chat model.** It has no chat template and no
instruction tuning — it was trained to continue TinyStories-style prose, for 0.43 of one
epoch at ~22M parameters. Asking it questions will disappoint. Give it the opening of a
simple story and it does well:

    Once upon a time, there was a little
    Lily found a shiny red
    The dog wanted to

Runs against the local converted artifact (``artifacts/hf/``) on CPU — no Tenstorrent
hardware needed, since the conversion's whole point is that it is host-portable.

    python scripts/chat.py
    python scripts/chat.py --temperature 0.6 --max-new-tokens 80

Commands inside the REPL: ``/temp <float>``, ``/len <int>``, ``/greedy``, ``/quit``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "artifacts" / "hf"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="HF model directory.")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None, help="Set for reproducible samples.")
    args = p.parse_args()

    model_dir = Path(args.model)
    if not (model_dir / "config.json").is_file():
        print(f"ERROR: no model at {model_dir}. Run scripts/convert_checkpoint.py first.",
              file=sys.stderr)
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {model_dir} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).eval()
    n_params = sum(t.numel() for t in model.parameters())

    # max_position_embeddings is the real limit; tokenizer_config advertises a sentinel
    # ~1e18 that must never be used as a length bound.
    ctx = model.config.max_position_embeddings
    print(f"{type(model).__name__}  {n_params/1e6:.2f}M params  context {ctx}")
    print("completion model — give it a story opening, not a question. /quit to exit.\n")

    temperature, max_new = args.temperature, args.max_new_tokens
    greedy = False

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            return 0
        if prompt.startswith("/temp "):
            temperature = float(prompt.split()[1]); greedy = False
            print(f"  temperature = {temperature}"); continue
        if prompt.startswith("/len "):
            max_new = int(prompt.split()[1]); print(f"  max_new_tokens = {max_new}"); continue
        if prompt == "/greedy":
            greedy = True; print("  greedy decoding (temperature ignored)"); continue

        ids = tok(prompt, return_tensors="pt").input_ids
        if ids.shape[1] >= ctx:
            print(f"  prompt is {ids.shape[1]} tokens; context is {ctx}. Shorten it.")
            continue
        if args.seed is not None:
            torch.manual_seed(args.seed)

        room = min(max_new, ctx - ids.shape[1])
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=room,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
                top_p=None if greedy else args.top_p,
                pad_token_id=tok.pad_token_id,
            )
        # Print only the continuation, so the model's contribution is visible as its own.
        print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip(), "\n")


if __name__ == "__main__":
    raise SystemExit(main())
