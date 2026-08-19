#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The permutation floor for die-region steering, computed without generating anything.

THE OBSERVATION THAT MAKES THIS FREE
====================================
``probe_die_steering.py`` generates completions once per REGION. Which source a
region is said to belong to is a label applied afterwards -- it changes the
statistic, not the text. So the full region x register matrix already on disk
answers every permutation at once:

    lift(source s, region r) = share_of_s_in_region_r - share_of_s_unsteered
    treatment  = mean over s of lift(s, s)
    permutation pi = mean over s of lift(s, pi(s))

The first control run generated a second complete set of completions for ONE
rotation. That was 40 minutes of CPU spent re-deriving a number already implied by
the first run's data, and it produced exactly one draw from a distribution this
computes in full.

With ten sources there are 10! = 3,628,800 permutations and 1,334,961 derangements
(no source keeping its own region). Both are enumerable by sampling, and the
derangement floor is the honest comparator: a permutation that leaves some sources
in place is partly treatment.

USAGE
-----
    python scripts/analyse_die_steering.py \
        --treatment docs/measurements/die-steering-treatment.json \
        --json-out docs/measurements/die-steering-floor.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def lift_matrix(data: dict, sources: list[str]) -> np.ndarray:
    """``L[s, r]`` = how much steering to region *r* raises source *s*'s register.

    Baseline is the unsteered arm, so the diagonal is the treatment effect and
    every off-diagonal entry is a counterfactual that needed no extra sampling.
    """
    base = data["results"]["unsteered"]["nearest_source_share"]
    L = np.zeros((len(sources), len(sources)))
    for si, s in enumerate(sources):
        for ri, r in enumerate(sources):
            L[si, ri] = data["results"][r]["nearest_source_share"].get(s, 0.0) - base.get(s, 0.0)
    return L


def sample_derangements(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Permutations with no fixed point. Rejection sampling; n=10 accepts ~37%."""
    out = []
    while len(out) < count:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            out.append(p)
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--treatment", type=Path,
                   default=Path("docs/measurements/die-steering-treatment.json"))
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    data = json.loads(args.treatment.read_text())
    sources = sorted(data["own_register_lift"])
    n = len(sources)
    L = lift_matrix(data, sources)

    observed = float(np.mean(np.diag(L)))
    rng = np.random.default_rng(args.seed)
    derangements = sample_derangements(n, args.permutations, rng)
    null = np.array([L[np.arange(n), pi].mean() for pi in derangements])

    ge = int((null >= observed).sum())
    p_val = (ge + 1) / (len(null) + 1)          # add-one: never report p = 0
    z = float((observed - null.mean()) / null.std()) if null.std() > 0 else float("nan")

    print(f"sources            {n}")
    print(f"observed (identity){observed:>+10.4f}   mean own-register lift")
    print(f"derangement floor  {null.mean():>+10.4f}   sd {null.std():.4f}  "
          f"(n={len(null):,} sampled of {round(math.factorial(n)/math.e):,})")
    print(f"z                  {z:>+10.2f}")
    print(f"p (one-sided)      {p_val:>10.5f}   "
          f"{ge:,} of {len(null):,} derangements scored >= observed")

    print(f"\npercentiles of the floor: "
          f"50th {np.percentile(null,50):+.4f}  "
          f"95th {np.percentile(null,95):+.4f}  "
          f"99th {np.percentile(null,99):+.4f}")

    # Per-source: does each source's own region beat the average wrong region?
    print(f"\n{'source':<20}{'own region':>12}{'mean other':>12}{'advantage':>11}")
    per_source = {}
    for i, s in enumerate(sources):
        own = float(L[i, i])
        other = float((L[i].sum() - own) / (n - 1))
        per_source[s] = {"own": round(own, 4), "mean_other": round(other, 4),
                         "advantage": round(own - other, 4)}
        print(f"  {s:<18}{own:>+12.4f}{other:>+12.4f}{own-other:>+11.4f}")
    n_pos = sum(1 for v in per_source.values() if v["advantage"] > 0)
    print(f"\n{n_pos}/{n} sources do better in their own region than in the average other")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "source": str(args.treatment),
            "method": ("Permutation floor computed from the region x register matrix the "
                       "treatment run already produced. Generation depends only on which "
                       "region is sampled, not on the label attached to it, so every "
                       "derangement is a relabelling of existing completions and needs no "
                       "new sampling."),
            "sources": sources,
            "observed_mean_own_register_lift": round(observed, 4),
            "derangement_floor": {
                "n_sampled": int(len(null)),
                "n_possible": int(round(math.factorial(n) / math.e)),
                "mean": round(float(null.mean()), 4),
                "sd": round(float(null.std()), 4),
                "p50": round(float(np.percentile(null, 50)), 4),
                "p95": round(float(np.percentile(null, 95)), 4),
                "p99": round(float(np.percentile(null, 99)), 4),
            },
            "z": round(z, 2),
            "p_one_sided": round(p_val, 5),
            "n_derangements_at_or_above_observed": ge,
            "per_source": per_source,
            "n_sources_favouring_own_region": n_pos,
            "why_derangements_not_permutations": ("A permutation with a fixed point leaves that "
                "source steered to its own region, which is treatment, not control. Derangements "
                "are the clean comparator."),
        }, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
