#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Freeze the measured token→core layout into a durable artifact.

What this is for
----------------
``probe_grid_layout.py`` answers a *question* — does placing vocabulary on the
NoC grid preserve enough semantic structure to be worth doing? It answered yes:
a core's bag is 86% as coherent as a cosine neighbourhood, and past ~200 cells it
beats one. But it recomputes the layout each run and keeps only the statistics.

The sampler needs the layout *itself*, fixed and quotable. This script produces
it, and that artifact is what makes the design claim true:

    A CPU can replay our sampler only by importing this file. The mapping is not
    an algorithm anyone derives — it is a measurement of this model's embedding
    geometry projected onto this box's harvested 11x10 die.

That is what "the hardware is the reference implementation" cashes out to. It is
deliberately NOT a claim that the mapping is secret or uncomputable: it is
seeded, deterministic and reproducible by re-running this script. What it is not
is *guessable* — you either measure it or you take ours.

Related measurement: the per-core PRNG carries no intrinsic identity
(``docs/measurements/core-prng-probe.json``, 0/16 cores differ under one seed),
so ALL of the hardware character in the sampler comes from this layout. That
makes this artifact load-bearing rather than decorative.

Pipeline (all three stages reused verbatim from the gate, not reimplemented)
---------------------------------------------------------------------------
1. balanced spherical k-means: vocabulary → ``n_cells`` capacity-capped clusters
2. spectral init: clusters dealt onto the grid by principal coordinates
3. simulated annealing on the QAP objective: similar clusters end up few hops apart

Output is an ``.npz`` plus a JSON sidecar carrying the provenance needed to tell
two layouts apart — model path, embedding digest, seed, grid spec, and the QAP
cost before and after annealing.

CPU only; no hardware required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_embedding_geography import load_embedding_matrix  # noqa: E402
from scripts.probe_grid_layout import (  # noqa: E402
    DEFAULT_GRIDS,
    GridSpec,
    anneal_placement,
    balanced_spherical_kmeans,
    grid_distance_matrix,
    qap_cost,
    spectral_grid_init,
    unit_rows,
)

DEFAULT_ANNEAL_STEPS = 200_000


def embedding_digest(embedding: np.ndarray) -> str:
    """Digest the exact float32 bytes the layout was derived from.

    Pinning the model *path* is not enough — artifacts get rebuilt in place. Two
    layouts are comparable only if this matches.
    """
    return hashlib.sha256(np.ascontiguousarray(embedding, dtype=np.float32)).hexdigest()


def resolve_grid(name: str) -> GridSpec:
    for spec in DEFAULT_GRIDS:
        if spec.name == name:
            return spec
    known = ", ".join(s.name for s in DEFAULT_GRIDS)
    raise SystemExit(f"unknown grid {name!r}; known: {known}")


def build(
    embedding: np.ndarray,
    spec: GridSpec,
    *,
    seed: int,
    anneal_steps: int,
    log=print,
) -> dict:
    """Run the three stages and return everything needed to reproduce and use them."""
    unit = unit_rows(embedding)
    log(f"  vocabulary {unit.shape[0]:,} x {unit.shape[1]} -> {spec.n_cells} cells "
        f"({spec.label} {spec.name})")

    assignment, centroids = balanced_spherical_kmeans(
        unit, spec.n_cells, seed=seed, log=log
    )

    similarity = centroids @ centroids.T
    distance = grid_distance_matrix(spec)

    placement = spectral_grid_init(centroids, spec)
    spectral_cost = qap_cost(similarity, distance, placement)
    placement, start_cost, end_cost = anneal_placement(
        similarity, distance, placement, n_steps=anneal_steps, seed=seed, log=log
    )

    # token -> grid cell index -> (x, y) on the die.
    token_cell = placement[assignment]
    xs = (token_cell % spec.width).astype(np.int16)
    ys = (token_cell // spec.width).astype(np.int16)

    occupancy = np.bincount(token_cell, minlength=spec.n_cells)

    return {
        "assignment": assignment.astype(np.int32),
        "placement": placement.astype(np.int32),
        "token_cell": token_cell.astype(np.int32),
        "token_xy": np.stack([xs, ys], axis=1),
        "occupancy": occupancy.astype(np.int32),
        "spectral_cost": float(spectral_cost),
        "start_cost": float(start_cost),
        "end_cost": float(end_cost),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", type=Path, default=None)
    parser.add_argument(
        "--grid",
        default="harvested-p300c",
        help="grid spec name; default is what this box actually exposes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anneal-steps", type=int, default=DEFAULT_ANNEAL_STEPS)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "token_core_map.npz",
    )
    args = parser.parse_args()

    if args.hf_model is None:
        from scripts.probe_grid_layout import _default_model

        args.hf_model = _default_model()

    spec = resolve_grid(args.grid)
    print(f"model: {args.hf_model}")
    embedding = load_embedding_matrix(args.hf_model)
    digest = embedding_digest(embedding)
    print(f"embedding digest: {digest[:16]}…")

    result = build(
        embedding, spec, seed=args.seed, anneal_steps=args.anneal_steps
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        token_cell=result["token_cell"],
        token_xy=result["token_xy"],
        placement=result["placement"],
        assignment=result["assignment"],
        occupancy=result["occupancy"],
    )

    occupancy = result["occupancy"]
    sidecar = {
        "hf_model": str(args.hf_model),
        "embedding_sha256": digest,
        "vocab_size": int(result["token_cell"].shape[0]),
        "grid": {
            "name": spec.name,
            "width": spec.width,
            "height": spec.height,
            "torus": spec.torus,
            "n_cells": spec.n_cells,
            "note": spec.note,
        },
        "seed": args.seed,
        "anneal_steps": args.anneal_steps,
        "qap_cost": {
            "spectral_init": result["spectral_cost"],
            "anneal_start": result["start_cost"],
            "anneal_end": result["end_cost"],
            # How much of the layout is annealing rather than the naive PCA squash.
            # A ratio near 1.0 would mean the anneal earned nothing.
            "improvement_ratio": result["start_cost"] / result["end_cost"]
            if result["end_cost"]
            else None,
        },
        "occupancy": {
            "min": int(occupancy.min()),
            "max": int(occupancy.max()),
            "mean": float(occupancy.mean()),
        },
        "artifact": str(args.out),
    }
    sidecar_path = args.out.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"\nQAP cost  spectral {result['spectral_cost']:,.0f}"
        f" -> annealed {result['end_cost']:,.0f}"
        f"  ({sidecar['qap_cost']['improvement_ratio']:.3f}x)"
    )
    print(
        f"tokens per core  min {occupancy.min()}  max {occupancy.max()}"
        f"  mean {occupancy.mean():.1f}"
    )
    print(f"\nwrote {args.out}\nwrote {sidecar_path}")


if __name__ == "__main__":
    main()
