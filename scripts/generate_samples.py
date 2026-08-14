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

Decoding is GREEDY AND DETERMINISTIC by default -- unchanged from every prior run of this
script, so old samples files stay reproducible and comparable. Greedy decoding on a small,
undertrained model loops almost by construction ("I saw them" x12), which can hide what the
underlying distribution actually looks like. Pass ``--temperature``/``--top-p``/``--top-k``
to switch to sampling, and ``--num-samples`` to draw more than one completion per prompt:

    python scripts/generate_samples.py --model artifacts/hf-tt-tnt-v1 --label tt-tnt-v1-t0.8 \\
        --temperature 0.8 --top-p 0.95 --num-samples 2 --seed 0

Passing none of ``--temperature``/``--top-p``/``--top-k`` keeps ``do_sample=False`` and the
exact ``model.generate(...)`` call this script has always made -- that is the mechanism that
guarantees old output stays byte-identical. This project has been bitten before by
documentation that quietly stopped matching reality, so the output file's header always
spells out the exact decoding settings used (temperature/top_p/top_k/seed), whether the run
was greedy or sampled -- a samples file must never be mistakable for the other kind.
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


def validate_sampling_args(temperature, top_p, top_k, num_samples) -> None:
    """Reject nonsensical sampling flags before touching torch/transformers.

    A plain value check, same spirit as ``resolve_model_dir``: fail fast, with a message
    that says exactly what was wrong, before any expensive import or any write under
    ``docs/measurements/``. No bare ``assert`` -- these are user-facing CLI validation
    errors, not internal invariants, so they raise ``ValueError`` with an actionable message
    that ``main()`` turns into a clean ``ERROR: ...`` + exit 1 (see the pattern already used
    for a missing/invalid ``--model`` directory).
    """
    if num_samples < 1:
        raise ValueError(f"--num-samples must be >= 1, got {num_samples}")
    if temperature is not None and temperature <= 0:
        raise ValueError(f"--temperature must be > 0, got {temperature}")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"--top-p must be in (0, 1], got {top_p}")
    if top_k is not None and top_k < 0:
        raise ValueError(f"--top-k must be >= 0, got {top_k}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model directory")
    p.add_argument("--label", required=True, help="Tag for this run, e.g. blend-v1-step20000")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. Unset (default) => greedy, deterministic decoding, "
             "identical to every prior run of this script. Setting ANY of --temperature/"
             "--top-p/--top-k switches decoding to sampling (do_sample=True).",
    )
    p.add_argument(
        "--top-p", type=float, default=None,
        help="Nucleus sampling threshold in (0, 1]. Setting this (like --temperature/"
             "--top-k) switches decoding to sampling. Defaults to 1.0 (disabled) once "
             "sampling is on, if not given explicitly.",
    )
    p.add_argument(
        "--top-k", type=int, default=None,
        help="Top-k sampling cutoff (0 disables it). Setting this (like --temperature/"
             "--top-p) switches decoding to sampling. Defaults to 0 (disabled) once "
             "sampling is on, if not given explicitly.",
    )
    p.add_argument(
        "--num-samples", type=int, default=1,
        help="Completions to draw per prompt. Only meaningful when sampling -- greedy "
             "decoding always produces the same completion, so N>1 with no sampling flags "
             "just repeats it. Each sample is labeled in the output so a human can compare "
             "them side by side.",
    )
    args = p.parse_args()

    # Fail early and clearly on bad flags -- before importing torch, before touching
    # docs/measurements/ at all. Same rule as resolve_model_dir() below: a bad argument is a
    # hard, early, exit-1 failure with a clear message, never a partial or malformed run.
    try:
        validate_sampling_args(args.temperature, args.top_p, args.top_k, args.num_samples)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Fail early and clearly on a missing/invalid model directory -- before importing torch,
    # before touching docs/measurements/ at all. See resolve_model_dir()'s docstring.
    try:
        model_dir = resolve_model_dir(args.model)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Setting ANY of --temperature/--top-p/--top-k turns sampling on. This is the ONE branch
    # point in the whole script: when it's False, the exact model.generate(...) call below is
    # byte-for-byte what this script has always made (do_sample=False, no other decoding
    # kwargs), which is what keeps default-flag runs reproducing old samples files exactly.
    do_sample = args.temperature is not None or args.top_p is not None or args.top_k is not None
    # Effective values, computed once and used both for the actual generate() calls and for
    # the header text below, so the header can never drift from what was actually run. "Off"
    # defaults (temperature/top_p=1.0, top_k=0) are HF's own no-op values for each knob, not
    # borrowed from HF's GenerationConfig defaults (which can differ, and change over
    # versions) -- that keeps this script's behavior self-documenting and version-stable.
    eff_temperature = args.temperature if args.temperature is not None else 1.0
    eff_top_p = args.top_p if args.top_p is not None else 1.0
    eff_top_k = args.top_k if args.top_k is not None else 0

    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Seeded once, before any generation. Because every generate() call below runs in a
    # fixed, deterministic order (prompts in load_prompts() order, samples 1..N per prompt),
    # a single seed here makes the WHOLE run reproducible: same seed + same flags => same
    # sequence of RNG draws => same output. See test_sampling_is_reproducible_given_a_seed.
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

    # The header always states the decoding settings explicitly and unambiguously -- this
    # repo has been bitten before by documentation (including a samples header) that no
    # longer matched reality. "greedy" and "sampling" are spelled out AND every knob's value
    # is given, so a file can never be misread as the other kind.
    if do_sample:
        decoding_desc = (
            f"sampling (temperature={eff_temperature}, top_p={eff_top_p}, top_k={eff_top_k})"
        )
    else:
        decoding_desc = "greedy (temperature=n/a, top_p=n/a, top_k=n/a)"
    lines = [f"# Samples — {args.label}", "",
             f"model: `{args.model}` · {decoding_desc} · seed {args.seed} · "
             f"{args.max_new_tokens} new tokens · {args.num_samples} sample(s)/prompt", ""]

    generate_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": do_sample}
    if do_sample:
        generate_kwargs.update(
            temperature=eff_temperature, top_p=eff_top_p, top_k=eff_top_k,
        )

    for prompt in load_prompts():
        ids = tok(prompt["text"], return_tensors="pt").input_ids
        completions = []
        for _ in range(args.num_samples):
            with torch.no_grad():
                got = model.generate(input_ids=ids, **generate_kwargs)
            completions.append(tok.decode(got[0][ids.shape[1]:], skip_special_tokens=True))

        if args.num_samples == 1:
            # Single-sample formatting is UNCHANGED from every prior version of this script
            # -- this is what keeps a default (greedy, num-samples=1) run's body byte-for-
            # byte identical to old committed samples files; only the header above may differ.
            lines += [f"### {prompt['id']} · _{prompt['probe']}_", "",
                      f"> {prompt['text']}**{completions[0]}**", ""]
        else:
            lines += [f"### {prompt['id']} · _{prompt['probe']}_", ""]
            for i, completion in enumerate(completions, start=1):
                lines += [f"**sample {i}/{args.num_samples}:**", "",
                          f"> {prompt['text']}**{completion}**", ""]

        for i, completion in enumerate(completions, start=1):
            tag = prompt["id"] if args.num_samples == 1 else f"{prompt['id']}#{i}"
            print(f"  {tag:16} {completion[:60]!r}")

    # Only create the output directory and write the file once every prompt has generated
    # successfully -- a crash partway through must not leave a truncated file that looks
    # like a complete (if bad) result.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
