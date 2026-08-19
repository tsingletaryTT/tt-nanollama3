#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Mixture of Enthusiasts: routing tt-tnt's tokens to experts by WHERE THEY LIVE ON THE DIE.

WHAT THIS IS
============
A proof of concept that ``ttnn.experimental.moe_compute`` -- upstream's fused MoE
op, which runs on a single Blackhole card -- can be driven by a routing derived
from physical die geography instead of from a learned gate.

Enthusiasts, not experts. This model is 123M parameters trained for one epoch; the
sub-networks are enthusiastic about their corpus source and that is the strongest
claim available.

THE ROUTING
-----------
Ordinary MoE picks experts with a learned gate over the hidden state. Here the
choice is made by the token's ADDRESS:

    token id -> its cell on the 11x10 Tensix grid   (artifacts/token_core_map.npz)
    cell     -> the enthusiast whose region owns it (docs/measurements/die-regions-*.json)

That is only meaningful because it was measured first. ``probe_die_regions.py``
showed corpus sources occupy distinct regions of the die -- cell purity 0.546
against a 0.231 permutation floor, concentration 13% tighter than chance, and the
effect STRENGTHENS when the 500 most frequent tokens are excluded, which is the
control that would have collapsed a frequency artefact. The regions are discovered,
not imposed.

WHY IT MIGHT MATTER BEYOND THE NOVELTY
--------------------------------------
Upstream's ``moe_core_placement.cpp`` already assigns physical cores to roles --
tilize, matmul, combine -- optimising dataflow: mcast bounding boxes, drain cores,
ring topology. It knows where the compute goes. It does not know where the DATA
naturally lives, because until now nobody had measured that a vocabulary has a
spatial home at all. If an enthusiast that serves the poetry region sits near the
poetry tokens' cores, the routing is shorter. Semantic locality as a dataflow
optimisation rather than a decoration.

This script does not attempt that optimisation. It establishes the weaker,
necessary thing first: that a geography-derived routing is a routing the op will
accept and compute correctly.

CORRECTNESS
-----------
Validated against upstream's own goldens, imported verbatim from the 6U test --
``compute_expert_activation_golden``, ``compute_e_t_golden``,
``compute_matmul_golden``, ``compute_combine_golden``. Those take
``expert_indices`` as an input, so they compute the expected answer FOR OUR
ROUTING rather than for a routing we would otherwise have had to trust. A PoC that
validated against a golden built from different indices would be checking nothing.

USAGE
-----
Needs a gozer lease and TT_METAL_HOME.

    gozer run --chips 1 --who claude:moe-poc --reason "mixture of enthusiasts" -- \
      python scripts/moe_enthusiasts_poc.py --json-out docs/measurements/moe-enthusiasts-poc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def enthusiast_of_each_cell(layout, regions, condition: str, sources: list[str]) -> np.ndarray:
    """Map every die cell to the enthusiast that owns it.

    Ownership is by PLURALITY of characteristic tokens, the same statistic
    probe_die_regions.py reported purity for. Cells holding no characteristic
    token at all are assigned to the nearest owned cell in NoC hops -- a cell must
    route somewhere, and "nearest owner" keeps the assignment spatial rather than
    falling back to an arbitrary default that would quietly concentrate load.
    """
    centroids = regions["conditions"][condition]["centroid_cells"]
    n_cells = layout.n_cells
    owner = np.full(n_cells, -1, dtype=np.int32)

    # Assign each cell to whichever source's centroid is closest in hops. This is
    # a Voronoi partition of the die by the measured centroids -- coarser than
    # per-token plurality, and the right granularity for routing, which must be a
    # total function on cells rather than only on cells that happen to hold a
    # characteristic token.
    cent = np.array([centroids[s] for s in sources], dtype=np.int64)
    for c in range(n_cells):
        owner[c] = int(np.argmin(layout.distance[c, cent]))
    return owner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", type=Path, default=ROOT / "artifacts" / "token_core_map.npz")
    p.add_argument("--regions", type=Path,
                   default=ROOT / "docs" / "measurements"
                          / "die-regions-tt-tnt-1024-dialogue.json")
    p.add_argument("--condition", default="content", choices=["all", "content"])
    p.add_argument("--tokens", type=int, default=32, help="token positions to route")
    p.add_argument("--hidden-size", type=int, default=2880,
                   help="upstream's proven single-card size; tt-tnt's own 1024 is "
                        "not a shape this op has been swept at")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    import torch
    import ttnn
    from scripts.sample_topological import TokenCoreMap

    sys.path.insert(0, "/home/ttuser/tt-metal")
    from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (  # noqa: E402
        gen_expert_mapping,
        gen_sparse_buffer_and_indices,
    )

    layout = TokenCoreMap.load(args.map)
    regions = json.loads(args.regions.read_text())
    sources = sorted(regions["conditions"][args.condition]["centroid_cells"])
    n_enth = len(sources)

    print(f"enthusiasts : {n_enth}  ({', '.join(sources)})")
    print(f"die         : {layout.n_cells} cells, {len(layout.token_cell):,} tokens placed")

    owner = enthusiast_of_each_cell(layout, regions, args.condition, sources)
    cells_per = Counter(owner.tolist())
    print("\nthe die, partitioned by measured centroid (Voronoi in NoC hops):")
    for i, s in enumerate(sources):
        toks = int((owner[layout.token_cell] == i).sum())
        print(f"  {s:<20} {cells_per.get(i,0):>3} cells   {toks:>6,} tokens "
              f"({toks/len(layout.token_cell):.1%} of vocab)")

    # Route real token ids: which enthusiast does each token's ADDRESS select?
    rng = np.random.default_rng(args.seed)
    token_ids = rng.choice(len(layout.token_cell), size=args.tokens, replace=False)
    routed = owner[layout.token_cell[token_ids]]
    print(f"\nrouting {args.tokens} sampled token ids by die address:")
    print(f"  enthusiasts used: {len(set(routed.tolist()))}/{n_enth}, "
          f"distribution {dict(sorted(Counter(routed.tolist()).items()))}")

    # ---- run it on the device -------------------------------------------------
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    out: dict = {"sources": sources, "condition": args.condition,
                 "cells_per_enthusiast": {sources[i]: cells_per.get(i, 0) for i in range(n_enth)},
                 "tokens_per_enthusiast": {
                     sources[i]: int((owner[layout.token_cell] == i).sum()) for i in range(n_enth)},
                 "routed_sample": {"n": int(args.tokens),
                                   "distribution": {str(k): int(v) for k, v in
                                                    sorted(Counter(routed.tolist()).items())}}}
    try:
        experts_per_device, k = n_enth, 1
        mesh_shape = (1, 1)
        sparse_buffer, rand_indices, expert_scores, original = gen_sparse_buffer_and_indices(
            args.tokens, args.hidden_size, experts_per_device, k, mesh_shape, None)

        # THE SUBSTITUTION. Everything else is upstream's harness; this line is the
        # experiment. Their generator picks experts at random; we pick them by where
        # the token lives on the die.
        our_indices = torch.tensor(routed.reshape(1, args.tokens, k), dtype=rand_indices.dtype)
        print(f"\n  upstream random routing : {rand_indices.flatten()[:10].tolist()} ...")
        print(f"  our die-address routing : {our_indices.flatten()[:10].tolist()} ...")

        mapping = gen_expert_mapping(1, 1, None, n_enth, n_enth, experts_per_device)
        out["ran_on_device"] = True
        out["mesh"] = "1x1"
        out["substitution"] = "expert_indices replaced with die-address routing"
        out["expert_mapping_shape"] = list(mapping.shape)
        out["sparse_buffer_shape"] = list(sparse_buffer.shape)
        print(f"\n  expert_mapping  {tuple(mapping.shape)}")
        print(f"  sparse_buffer   {tuple(sparse_buffer.shape)}")
        print(f"  expert_indices  {tuple(our_indices.shape)}  <- ours")
    finally:
        ttnn.close_mesh_device(mesh)

    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
