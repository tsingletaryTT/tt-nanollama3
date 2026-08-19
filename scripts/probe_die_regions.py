#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Do corpus sources occupy distinct REGIONS OF THE DIE, or only of embedding space?

WHY THIS EXISTS
===============
``probe_embedding_geography.py`` established that tokens characteristic of
different corpus sources occupy distinguishable regions of **embedding space** --
verdict SEPARATED, above a label-permutation floor and above a frequency-only
control. That is what makes a spatial sampler worth building at all: the layout is
*discovered* rather than imposed.

But embedding space is 1024-dimensional and the die is a 11x10 grid with 110
usable cells. The QAP placement in ``build_token_core_map.py`` projects one onto
the other, and a projection can destroy exactly the structure that justified it.
Separation upstairs does not entail separation downstairs.

This probe asks the downstairs question, and it is the precondition for everything
that follows. If sources do not occupy distinct die regions then "sample within k
hops of the poetry region" is a sentence with no referent, and a Mixture of
Enthusiasts pinned to die regions has nothing to pin to.

WHAT IT MEASURES
----------------
For each source, take its characteristic tokens (same statistic as the embedding
probe -- log-odds with an informative Dirichlet prior, z-scored) and look at where
they LAND on the grid:

* **concentration** -- mean pairwise NoC hop distance between a source's tokens'
  cells, against the same statistic for randomly relabelled tokens. Tight means
  the source occupies a place.
* **centroid separation** -- hop distance between sources' centroid cells, against
  a permutation floor. Distinct means the places are different places.
* **cell purity** -- for each cell, the fraction of characteristic tokens in it
  that belong to its plurality source, against chance at the realised class sizes.

Every headline is reported against a **label-permutation floor**: identical
computation with the source labels shuffled, which holds the layout, the class
sizes and the grid fixed and destroys only the correspondence. A number without
that floor would say nothing -- 110 cells and ten sources will always produce
*some* apparent structure.

WHAT A NULL RESULT WOULD MEAN
-----------------------------
That the geography is real in the embedding and inert on the die: the QAP
placement optimised something other than source coherence, and steering by hop
distance cannot work. That is a publishable answer and it is cheaper to find here
than after writing a kernel.

USAGE
-----
    python scripts/probe_die_regions.py \
        --hf-model artifacts/hf-tt-tnt-1024-dialogue \
        --map artifacts/token_core_map.npz \
        --out docs/measurements/die-regions.md \
        --json-out docs/measurements/die-regions.json

No device, no network. Reads the embedding matrix only to reuse the tokenizer, and
the corpus from ``artifacts/corpus``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_embedding_geography import (  # noqa: E402
    DEFAULT_PRIOR_STRENGTH,
    characteristic_tokens,
    count_tokens_by_source,
    log_odds_z,
)
from scripts.sample_topological import TokenCoreMap  # noqa: E402


def mean_pairwise_hops(cells: np.ndarray, distance: np.ndarray) -> float:
    """Mean NoC hop distance between every pair of the given cells.

    Cells repeat -- many tokens share one cell -- and the repetition is kept
    deliberately: a source whose tokens pile into one cell IS more concentrated
    than one spread over ten, and de-duplicating would erase that difference.
    """
    if len(cells) < 2:
        return 0.0
    sub = distance[np.ix_(cells, cells)]
    iu = np.triu_indices(len(cells), k=1)
    return float(sub[iu].mean())


def centroid_cell(cells: np.ndarray, distance: np.ndarray) -> int:
    """The cell minimising total hop distance to the given cells (a medoid).

    A medoid rather than a mean of coordinates: the grid is a torus for distance
    purposes and has holes where cores are harvested, so an averaged (x, y) can
    land on a cell that does not exist.
    """
    uniq = np.unique(cells)
    totals = distance[np.ix_(uniq, cells)].sum(axis=1)
    return int(uniq[int(np.argmin(totals))])


def analyse(token_cell: np.ndarray, distance: np.ndarray,
            labels: np.ndarray, tokens: np.ndarray,
            source_names: Sequence[str], rng: np.random.Generator,
            n_permutations: int) -> dict:
    """Concentration, centroid separation and cell purity, each against a floor."""
    cells = token_cell[tokens]

    per_source, centroids = {}, {}
    for i, name in enumerate(source_names):
        mine = cells[labels == i]
        if len(mine) < 2:
            continue
        per_source[name] = {
            "n_tokens": int(len(mine)),
            "n_distinct_cells": int(len(np.unique(mine))),
            "mean_pairwise_hops": mean_pairwise_hops(mine, distance),
        }
        centroids[name] = centroid_cell(mine, distance)

    names = list(per_source)
    cent_pairs = [
        float(distance[centroids[a], centroids[b]])
        for i, a in enumerate(names) for b in names[i + 1:]
    ]

    # Cell purity: per cell, the share held by its plurality source.
    pur_num = pur_den = 0
    for c in np.unique(cells):
        here = labels[cells == c]
        if len(here):
            pur_num += int(np.bincount(here, minlength=len(source_names)).max())
            pur_den += int(len(here))
    purity = pur_num / pur_den if pur_den else float("nan")

    # The floor. Shuffle labels only: layout, class sizes and grid all held fixed.
    null_conc, null_cent, null_pur = [], [], []
    for _ in range(n_permutations):
        perm = rng.permutation(labels)
        cs = [mean_pairwise_hops(cells[perm == i], distance)
              for i in range(len(source_names)) if (perm == i).sum() >= 2]
        null_conc.append(float(np.mean(cs)) if cs else np.nan)
        cen = {i: centroid_cell(cells[perm == i], distance)
               for i in range(len(source_names)) if (perm == i).sum() >= 1}
        ks = list(cen)
        pairs = [float(distance[cen[a], cen[b]])
                 for i, a in enumerate(ks) for b in ks[i + 1:]]
        null_cent.append(float(np.mean(pairs)) if pairs else np.nan)
        n = d = 0
        for c in np.unique(cells):
            here = perm[cells == c]
            if len(here):
                n += int(np.bincount(here, minlength=len(source_names)).max())
                d += int(len(here))
        null_pur.append(n / d if d else np.nan)

    obs_conc = float(np.mean([v["mean_pairwise_hops"] for v in per_source.values()]))
    obs_cent = float(np.mean(cent_pairs)) if cent_pairs else float("nan")

    def z(obs, null):
        arr = np.asarray(null, dtype=float)
        arr = arr[~np.isnan(arr)]
        sd = arr.std()
        return float((obs - arr.mean()) / sd) if sd > 0 else float("nan")

    return {
        "per_source": per_source,
        "centroid_cells": {k: int(v) for k, v in centroids.items()},
        "concentration": {
            "observed_mean_pairwise_hops": obs_conc,
            "permuted_mean": float(np.nanmean(null_conc)),
            "permuted_sd": float(np.nanstd(null_conc)),
            "z": z(obs_conc, null_conc),
            "reading": "LOWER than the floor means sources are concentrated in place",
        },
        "centroid_separation": {
            "observed_mean_pairwise_hops_between_centroids": obs_cent,
            "permuted_mean": float(np.nanmean(null_cent)),
            "permuted_sd": float(np.nanstd(null_cent)),
            "z": z(obs_cent, null_cent),
            "reading": "HIGHER than the floor means the places are different places",
        },
        "cell_purity": {
            "observed": purity,
            "permuted_mean": float(np.nanmean(null_pur)),
            "permuted_sd": float(np.nanstd(null_pur)),
            "z": z(purity, null_pur),
            "reading": "HIGHER than the floor means a cell tends to belong to one source",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", type=Path,
                   default=ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue")
    p.add_argument("--map", type=Path, default=ROOT / "artifacts" / "token_core_map.npz")
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "artifacts" / "corpus")
    p.add_argument("--words-per-source", type=int, default=1_000_000)
    p.add_argument("--per-source", type=int, default=150)
    p.add_argument("--exclude-top", type=int, default=500)
    p.add_argument("--min-count", type=int, default=25)
    p.add_argument("--n-permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    from transformers import AutoTokenizer

    layout = TokenCoreMap.load(args.map)
    tok = AutoTokenizer.from_pretrained(str(args.hf_model))
    sources = sorted(q.stem for q in args.corpus_dir.glob("*.txt")
                     if q.stem not in {"corpus", "blend"})
    print(f"map      : {args.map}  ({layout.n_cells} cells)")
    print(f"sources  : {', '.join(sources)}")

    counts = count_tokens_by_source(
        args.corpus_dir, sources, tok, len(tok),
        words_per_source=args.words_per_source,
        log=lambda m: print(f"  {m}"),
    )
    z = log_odds_z(counts, DEFAULT_PRIOR_STRENGTH)

    rng = np.random.default_rng(args.seed)
    out = {"map": str(args.map), "hf_model": str(args.hf_model), "sources": sources,
           "n_cells": int(layout.n_cells), "conditions": {}}

    for cond, excl in (("all", 0), ("content", args.exclude_top)):
        lab = characteristic_tokens(counts, z, sources, per_source=args.per_source,
                                    min_count=args.min_count, exclude_top=excl)
        res = analyse(layout.token_cell, layout.distance, lab.labels, lab.token_ids,
                      sources, np.random.default_rng(args.seed), args.n_permutations)
        out["conditions"][cond] = res
        c, s, u = res["concentration"], res["centroid_separation"], res["cell_purity"]
        print(f"\n[{cond}]  {len(lab.token_ids)} characteristic tokens")
        print(f"  concentration  {c['observed_mean_pairwise_hops']:.3f} hops "
              f"vs floor {c['permuted_mean']:.3f}  z={c['z']:+.2f}  (lower is concentrated)")
        print(f"  centroid sep   {s['observed_mean_pairwise_hops_between_centroids']:.3f} hops "
              f"vs floor {s['permuted_mean']:.3f}  z={s['z']:+.2f}  (higher is separated)")
        print(f"  cell purity    {u['observed']:.4f} vs floor {u['permuted_mean']:.4f}  "
              f"z={u['z']:+.2f}  (higher is pure)")

    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
