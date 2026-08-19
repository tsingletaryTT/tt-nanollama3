#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does constraining generation to a REGION OF THE DIE change the register of the text?

THE QUESTION
============
``probe_die_regions.py`` established that corpus sources occupy distinct regions of
the Tensix grid: concentrated, separated, and pure well beyond a label-permutation
floor, with the effect strengthening under the frequency control. So "the poetry
region" names something real.

This asks whether that fact is USABLE. If sampling is restricted to the cells
within *k* NoC hops of a source's centroid, does the model write more like that
source? Measured with ``score_behaviour.RegisterProfile`` -- per-source unigram +
bigram language models fitted on the corpus, the same instrument the evaluation
suite already uses -- so the answer is in units this project has calibrated
rather than in a new number invented for the occasion.

WHY THIS IS NOT A GIMMICK
-------------------------
Gumbel-max already decomposes exactly: ``argmax_i(logit_i/T + g_i)`` taken per
core and then across cores is a draw from the softmax over the whole vocabulary,
provably, not approximately. Restricting which cores may win is therefore an
ordinary masked softmax that happens to be expressible as physical geography. The
mathematics was always per-core; the die only makes it visible.

WHY CPU FIRST
-------------
The hypothesis -- does restricting the candidate set move register -- is true or
false regardless of which silicon evaluates the logits. Running on the host makes
each condition seconds instead of minutes and keeps the device free. If register
moves, the on-device version becomes the demonstration; if it does not, no kernel
was written to find that out.

WHAT WOULD MAKE IT NULL
-----------------------
110 cells for 32,000 tokens is roughly 291 tokens per cell, and cell purity is
0.55 -- nearly half of a cell's characteristic tokens belong to some other source.
A neighbourhood may simply be too coarse to carry register even though the regions
are statistically real. That is the outcome this run is built to be able to report.

USAGE
-----
    python scripts/probe_die_steering.py \
        --hf-model artifacts/hf-tt-tnt-1024-dialogue \
        --regions docs/measurements/die-regions-tt-tnt-1024-dialogue.json \
        --hops 2 --samples 16 \
        --json-out docs/measurements/die-steering.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sample_topological import TokenCoreMap, neighbourhood  # noqa: E402
from scripts.score_behaviour import build_register_profile, words_of  # noqa: E402

#: Neutral prompts. Deliberately register-free -- an opener that already smells of
#: fairy tale would hand the result to the tinystories region for free. Each is a
#: bare continuation cue that any of the ten sources could plausibly continue.
PROMPTS = [
    "The first thing to say is",
    "It was there, and then",
    "Consider what happens when",
    "After that, the",
    "Nobody had told them that",
    "The next part is",
    "Somewhere between the",
    "What follows is",
]


def generate_constrained(model, tokenizer, prompt: str, *, allowed: Optional[np.ndarray],
                         max_new_tokens: int, temperature: float,
                         rng: np.random.Generator) -> tuple[str, List[int]]:
    """Sample a continuation, optionally restricted to a set of token ids.

    The restriction is a mask on the logits, which is the host-side equivalent of
    letting only certain cores compete in the per-core Gumbel-max. -inf rather than
    a filtered renormalisation so the temperature softmax below is unchanged in
    form; the two are identical in distribution and this one cannot silently
    reorder the vocabulary.
    """
    import torch

    ids = tokenizer.encode(prompt, return_tensors="pt")
    out_ids: List[int] = []
    mask = None
    if allowed is not None:
        mask = torch.full((model.config.vocab_size,), float("-inf"))
        mask[torch.as_tensor(allowed, dtype=torch.long)] = 0.0

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        if mask is not None:
            logits = logits + mask
        probs = torch.softmax(logits / temperature, dim=-1).numpy()
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            break
        probs = probs / total
        tok = int(rng.choice(len(probs), p=probs))
        out_ids.append(tok)
        ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
    return tokenizer.decode(out_ids), out_ids


def tokens_in_cells(layout: TokenCoreMap, cells: np.ndarray) -> np.ndarray:
    """Every token id whose cell is in ``cells``."""
    return np.flatnonzero(np.isin(layout.token_cell, cells))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", type=Path,
                   default=ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue")
    p.add_argument("--map", type=Path, default=ROOT / "artifacts" / "token_core_map.npz")
    p.add_argument("--regions", type=Path,
                   default=ROOT / "docs" / "measurements"
                          / "die-regions-tt-tnt-1024-dialogue.json")
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "artifacts" / "corpus")
    p.add_argument("--condition", default="content", choices=["all", "content"],
                   help="which characteristic-token condition's centroids to steer to")
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--samples", type=int, default=16, help="completions per prompt")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--permute-regions", action="store_true",
                   help="THE CONTROL. Steer each source to ANOTHER source's region, sizes "
                        "and everything else identical. Restricting the vocabulary to any "
                        "11.8% slice changes the text; this asks whether steering to the "
                        "MEASURED region raises the MATCHING register specifically. If the "
                        "lift survives here, the geography is doing nothing and mere "
                        "restriction explains the result.")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    layout = TokenCoreMap.load(args.map)
    regions = json.loads(args.regions.read_text())
    centroids: Dict[str, int] = regions["conditions"][args.condition]["centroid_cells"]
    sources = sorted(centroids)

    print(f"model    : {args.hf_model}")
    print(f"steering : {args.hops} hops around each source's centroid "
          f"({args.condition} condition)")
    print(f"sampling : {args.samples} completions x {len(PROMPTS)} prompts, "
          f"T={args.temperature}, {args.max_new_tokens} tokens\n")

    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model))
    model = AutoModelForCausalLM.from_pretrained(str(args.hf_model)).eval()
    profile = build_register_profile(args.corpus_dir, sources)

    # Candidate sets, one per condition, computed once.
    # In the control, source i steers to source (i+1)'s centroid: a derangement, so
    # no source keeps its own region and every region is still used exactly once.
    steer_to = dict(zip(sources, sources))
    if args.permute_regions:
        rot = sources[1:] + sources[:1]
        steer_to = dict(zip(sources, rot))
        print("CONTROL: regions permuted — " +
              ", ".join(f"{a}->{b}" for a, b in list(steer_to.items())[:3]) + ", ...\n")

    conditions: Dict[str, Optional[np.ndarray]] = {"unsteered": None}
    coverage: Dict[str, dict] = {}
    for name in sources:
        cells = neighbourhood(layout, centroids[steer_to[name]], args.hops)
        toks = tokens_in_cells(layout, cells)
        conditions[name] = toks
        coverage[name] = {
            "steered_to": steer_to[name],
            "centroid_cell": int(centroids[steer_to[name]]),
            "cells_in_neighbourhood": int(len(cells)),
            "tokens_available": int(len(toks)),
            "fraction_of_vocab": round(len(toks) / len(layout.token_cell), 4),
        }

    results: Dict[str, dict] = {}
    for cond, allowed in conditions.items():
        rng = np.random.default_rng(args.seed)
        nearest: Counter = Counter()
        n_words = 0
        texts: List[str] = []
        for prompt in PROMPTS:
            for _ in range(args.samples):
                text, _ids = generate_constrained(
                    model, tokenizer, prompt, allowed=allowed,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, rng=rng)
                texts.append(text)
                w = words_of(text)
                n_words += len(w)
                src = profile.nearest_source(w)
                if src:
                    nearest[src] += 1
        total = sum(nearest.values())
        share = {s: nearest.get(s, 0) / total for s in sources} if total else {}
        hit = share.get(cond, float("nan")) if cond != "unsteered" else float("nan")
        results[cond] = {
            "n_completions": total,
            "mean_words": round(n_words / max(len(texts), 1), 1),
            "nearest_source_share": {k: round(v, 4) for k, v in share.items()},
            "self_share": round(hit, 4) if hit == hit else None,
            "example": texts[0][:160] if texts else "",
        }
        if cond == "unsteered":
            print(f"  {cond:<20} top register: "
                  f"{', '.join(f'{s} {share.get(s,0):.2f}' for s in sorted(share, key=share.get, reverse=True)[:3])}")
        else:
            base = results["unsteered"]["nearest_source_share"].get(cond, 0.0)
            print(f"  steer->{cond:<13} own-register share {hit:.3f} "
                  f"(unsteered {base:.3f}, lift {hit - base:+.3f})  "
                  f"vocab {coverage[cond]['fraction_of_vocab']:.1%}")

    # The headline: does steering to a region raise that region's own register?
    lifts = {
        s: results[s]["nearest_source_share"].get(s, 0.0)
           - results["unsteered"]["nearest_source_share"].get(s, 0.0)
        for s in sources
    }
    mean_lift = float(np.mean(list(lifts.values())))
    n_pos = sum(1 for v in lifts.values() if v > 0)
    print(f"\nmean own-register lift {mean_lift:+.4f} across {len(lifts)} sources; "
          f"{n_pos}/{len(lifts)} positive")

    out = {
        "hf_model": str(args.hf_model), "map": str(args.map),
        "condition": args.condition, "hops": args.hops,
        "samples_per_prompt": args.samples, "prompts": PROMPTS,
        "temperature": args.temperature, "seed": args.seed,
        "permuted_regions": bool(args.permute_regions),
        "coverage": coverage, "results": results,
        "own_register_lift": {k: round(v, 4) for k, v in lifts.items()},
        "mean_own_register_lift": round(mean_lift, 4),
        "n_positive": n_pos, "n_sources": len(lifts),
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
