#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Generate the frozen prompt set's completions from a checkpoint, for human judgment.

Loss cannot see "strangely satisfying", and a better-trained model in this project measured
FLATTER, not sharper. So the acceptance gate is a person reading these samples -- this
script only makes them, identically, every time.

No trained model exists for a fresh corpus the moment its tokenizer lands; the real
generation run happens after training. So this script validates its ``--model`` argument
as a filesystem check BEFORE importing torch/transformers, and before creating or writing
anything under ``docs/measurements/``. A missing or invalid model directory is a hard,
early, exit-1 failure with a clear message -- never a silently-written empty or partial
samples file that could be mistaken for a real result.

    python scripts/generate_samples.py --model artifacts/hf --label baseline-384
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPTS = ROOT / "docs" / "evaluation_prompts.json"


def load_prompts() -> list:
    return json.loads(PROMPTS.read_text())["prompts"]


def resolve_model_dir(model: str) -> Path:
    """Validate that ``model`` looks like a converted HF model directory.

    Deliberately a plain filesystem check with no torch/transformers import, so:
      (a) it runs before the expensive imports, giving a fast, clear failure instead of a
          transformers stack trace or (worse) a silently empty output file, and
      (b) it is testable in an environment where no model, and possibly no torch, is
          installed at all -- which is exactly the environment this test suite runs in
          before a model has been trained on the new corpus.

    Raises FileNotFoundError with an actionable message if ``model`` does not exist or does
    not look like a directory produced by ``scripts/convert_checkpoint.py`` (no config.json).
    """
    model_dir = Path(model)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"no such directory: {model_dir}. "
            "Point --model at a converted HF model directory, e.g. artifacts/hf/."
        )
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"{model_dir} has no config.json -- it does not look like a converted HF model "
            "directory. Run scripts/convert_checkpoint.py first."
        )
    return model_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model directory")
    p.add_argument("--label", required=True, help="Tag for this run, e.g. blend-v1-step20000")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Fail early and clearly on a missing/invalid model directory -- before importing torch,
    # before touching docs/measurements/ at all. See resolve_model_dir()'s docstring.
    try:
        model_dir = resolve_model_dir(args.model)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    print(f"loading {model_dir} ...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype="auto").eval()
    except (OSError, ValueError) as exc:
        # config.json existed but the directory is still broken in some other way (missing
        # weights, corrupt tokenizer files, ...). Same rule applies: report and stop, don't
        # write a partial or empty samples file.
        print(f"ERROR: could not load model from {model_dir}: {exc}", file=sys.stderr)
        return 1

    out_path = ROOT / "docs" / "measurements" / f"samples-{args.label}.md"
    lines = [f"# Samples — {args.label}", "",
             f"model: `{args.model}` · greedy · seed {args.seed} · "
             f"{args.max_new_tokens} new tokens", ""]

    for prompt in load_prompts():
        ids = tok(prompt["text"], return_tensors="pt").input_ids
        with torch.no_grad():
            got = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        completion = tok.decode(got[0][ids.shape[1]:], skip_special_tokens=True)
        lines += [f"### {prompt['id']} · _{prompt['probe']}_", "",
                  f"> {prompt['text']}**{completion}**", ""]
        print(f"  {prompt['id']:12} {completion[:60]!r}")

    # Only create the output directory and write the file once every prompt has generated
    # successfully -- a crash partway through must not leave a truncated file that looks
    # like a complete (if bad) result.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
