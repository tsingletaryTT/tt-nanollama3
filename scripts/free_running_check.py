#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Free-running generation check: the gate a decode PCC cannot be.

WHY THIS EXISTS
---------------
``models/tt_transformers/tests/test_model.py`` reports PCC 0.994-0.9998 over nine decode
steps, and this model still degenerates into repetition on the third token of real
generation. Those nine steps are **teacher forced** — the reference model picks every token
and both models are fed it (``test_model.py:393-401``, comment: *"Use the same token for TT
model (teacher forcing)"*). TT never consumes its own output, so per-step error cannot
accumulate. Nine teacher-forced steps are nine independent one-step checks.

Serving is free-running by definition. This script measures that directly: the same prompt
generated greedily on the device (through the OpenAI-compatible endpoint, i.e. the real
serving path) and on the CPU reference, compared token by token.

WHAT IT REPORTS
---------------
For each prompt, the number of leading tokens that agree, and where they first differ. The
headline is the **agreement length** distribution — how many tokens of real generation you
get before the paths separate. That is the number that matters for serving quality, and it
is the number no PCC in the tt-metal suite currently produces.

This is deliberately a *measurement*, not a pass/fail assertion with an invented threshold.
``--min-agree`` sets an exit-code gate when you want one in CI.

USAGE
-----
    # with a server already running on :8000
    python scripts/free_running_check.py --tokens 40
    python scripts/free_running_check.py --tokens 40 --json out.json --min-agree 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Prompts chosen to look like the corpus (TinyStories) rather than to flatter the model.
#: A model that only holds together on one hand-picked prompt is not working.
DEFAULT_PROMPTS = [
    "Once upon a time, there was a little",
    "One day, a small boy named Tom went to the park with his mother and",
    "The cat sat on the mat and looked at the",
    "Lily and her brother found a big red ball in the garden. They",
    "There was a dog who loved to run. Every morning he would",
    "The little girl opened the box and inside she saw a",
]


def tt_generate(prompt: str, n: int, url: str, model: str, timeout: float) -> list[str]:
    """Greedy generation through the real serving path, returning token strings."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": n,
            "temperature": 0,
            "logprobs": 1,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    choice = payload["choices"][0]
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens")
    if tokens is None:
        raise RuntimeError(
            "server returned no logprobs.tokens; token-level comparison needs them"
        )
    return tokens


def cpu_generate(model, tok, prompt: str, n: int) -> list[str]:
    """Greedy generation from the CPU reference, returning token strings."""
    import torch

    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False)
    return [tok.decode([t]) for t in out[0][ids.shape[1] :]]


def agreement(a: list[str], b: list[str]) -> int:
    """Number of leading tokens that match."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="episod/tt-tnt")
    p.add_argument("--hf-dir", default=None,
                   help="CPU reference model directory (default: the 384 size's artifacts).")
    p.add_argument("--tokens", type=int, default=40)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--json", type=Path, default=None, help="Write full results here.")
    p.add_argument("--min-agree", type=int, default=None,
                   help="Exit non-zero if the MEDIAN agreement is below this.")
    p.add_argument("--label", default="", help="Free-text tag recorded in the JSON output.")
    args = p.parse_args()

    if args.hf_dir is None:
        from train.paths import read_dir

        args.hf_dir = str(read_dir("384", "hf"))

    import warnings

    warnings.filterwarnings("ignore")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"CPU reference: {args.hf_dir}")
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    ref = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype="auto").eval()
    print(f"device server : {args.url} ({args.model})")
    print(f"tokens/prompt : {args.tokens}\n")

    rows = []
    for prompt in DEFAULT_PROMPTS:
        try:
            tt = tt_generate(prompt, args.tokens, args.url, args.model, args.timeout)
        except (urllib.error.URLError, RuntimeError, OSError) as exc:
            print(f"ERROR contacting server: {exc}", file=sys.stderr)
            return 2
        cpu = cpu_generate(ref, tok, prompt, args.tokens)
        n = agreement(cpu, tt)
        first = None
        if n < min(len(cpu), len(tt)):
            first = {"index": n, "cpu": cpu[n], "tt": tt[n]}
        rows.append(
            {"prompt": prompt, "agree": n, "of": args.tokens,
             "first_divergence": first, "cpu": cpu, "tt": tt}
        )
        shown = prompt if len(prompt) <= 44 else prompt[:41] + "..."
        detail = f"cpu={first['cpu']!r} tt={first['tt']!r}" if first else "(all match)"
        print(f"  {n:3}/{args.tokens}  {shown:<46} {detail}")

    agrees = [r["agree"] for r in rows]
    med = statistics.median(agrees)
    print()
    print(f"agreement over {len(rows)} prompts: "
          f"min {min(agrees)}, median {med:.1f}, max {max(agrees)}, of {args.tokens}")

    if args.json:
        args.json.write_text(json.dumps(
            {"label": args.label, "tokens": args.tokens, "model": args.model,
             "hf_dir": args.hf_dir, "median_agreement": med, "results": rows}, indent=2))
        print(f"wrote {args.json}")

    if args.min_agree is not None and med < args.min_agree:
        print(f"FAIL: median agreement {med:.1f} < --min-agree {args.min_agree}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
