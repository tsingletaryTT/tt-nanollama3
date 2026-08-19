#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Mixture of Enthusiasts, actually running through ttnn.experimental.moe_compute.

The routing is ours -- each token goes to the enthusiast that owns its cell on the
Tensix grid -- and everything else is upstream's, including the correctness check.

HOW
===
``tests/.../test_moe_compute_single_card.py`` builds the whole workload through one
shared function, ``_run_moe_compute_single_card_test``, which obtains its routing
from ``gen_sparse_buffer_and_indices``. We monkeypatch that one generator to return
die-address indices instead of random ones, then invoke upstream's test unchanged.

That placement of the seam matters. The goldens
(``compute_expert_activation_golden``, ``compute_e_t_golden``,
``compute_matmul_golden``, ``compute_combine_golden``) all take ``expert_indices``
as an INPUT, so patching the generator makes them compute the expected answer FOR
OUR ROUTING. Every validator downstream then checks the device against a reference
that agrees about who routed where. Patching any later -- after the goldens were
built -- would have compared the device's answer for our routing against a golden
for a different one, and "PCC fails" would have meant nothing.

WHAT A PASS PROVES, AND WHAT IT DOES NOT
----------------------------------------
Proves: the op accepts a routing derived from physical die geography and computes
it correctly, on a single Blackhole card, at upstream's own tolerances.

Does not prove: that this routing is *good*. Load is uneven by construction --
the Voronoi partition of the die gives flavour 23 cells and spine 3 -- and no
claim is made here about quality, throughput or locality benefit. This is the
existence proof that had to come first.

USAGE
-----
    gozer run --chips 1 --who claude:moe-enthusiasts --reason "PoC" -- \
      python scripts/moe_enthusiasts_run.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TT_METAL = Path("/home/ttuser/tt-metal")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TT_METAL))


def build_owner(layout, regions, condition, sources):
    """Voronoi partition of the die by the measured per-source centroids."""
    cent = np.array([regions["conditions"][condition]["centroid_cells"][s] for s in sources],
                    dtype=np.int64)
    return np.array([int(np.argmin(layout.distance[c, cent])) for c in range(layout.n_cells)],
                    dtype=np.int32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", type=Path, default=ROOT / "artifacts" / "token_core_map.npz")
    p.add_argument("--regions", type=Path,
                   default=ROOT / "docs" / "measurements"
                          / "die-regions-tt-tnt-1024-dialogue.json")
    p.add_argument("--condition", default="content")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    import torch
    from scripts.sample_topological import TokenCoreMap

    layout = TokenCoreMap.load(args.map)
    regions = json.loads(args.regions.read_text())
    sources = sorted(regions["conditions"][args.condition]["centroid_cells"])
    owner = build_owner(layout, regions, args.condition, sources)

    # token id -> cell -> enthusiast. The whole routing rule, in one line.
    token_enthusiast = owner[layout.token_cell]

    import tests.nightly.tg.ccl.moe.test_moe_compute_6U as sixu
    import tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_compute_single_card as sc

    original = sixu.gen_sparse_buffer_and_indices
    captured: dict = {}

    def die_routed(tokens_per_device, hidden_size, experts, selected_experts_k,
                   mesh_shape, cluster_axis, dtype=torch.bfloat16):
        """Upstream's generator, with the indices replaced by die addresses.

        The sparse buffer, scores and original tokens are taken from upstream
        unchanged -- only WHICH EXPERT each token selects is ours. Sampling real
        token ids (not arbitrary integers) is the point: the routing has to come
        from the vocabulary's actual placement, or it is just a permutation.
        """
        buf, rand_idx, scores, orig = original(
            tokens_per_device, hidden_size, experts, selected_experts_k,
            mesh_shape, cluster_axis, dtype)

        rng = np.random.default_rng(args.seed)
        ids = rng.choice(len(token_enthusiast), size=rand_idx.shape[:-1], replace=True)
        routed = token_enthusiast[ids] % experts          # experts may be < enthusiasts

        # k experts per token: the die gives one home, and the remaining k-1 slots
        # are its successors modulo the expert count, so every index is valid and a
        # token's slate stays adjacent rather than scattered.
        #
        # Built in numpy int64 and cast ONCE at the end. Doing the modular
        # arithmetic on the uint16 torch tensor raises `"add_stub" not implemented
        # for 'UInt16'` -- torch has no integer add kernel for that dtype, and the
        # index tensor must stay uint16 because that is what the op's memory config
        # declares.
        slates = np.stack(
            [(routed + off) % experts for off in range(selected_experts_k)], axis=-1)
        idx = torch.as_tensor(slates.astype(np.int64)).to(rand_idx.dtype)
        captured["distribution"] = {int(k): int(v) for k, v in
                                    sorted(Counter(idx[..., 0].flatten().tolist()).items())}
        captured["experts"] = int(experts)
        captured["k"] = int(selected_experts_k)
        captured["upstream_first10"] = rand_idx[..., 0].flatten()[:10].tolist()
        captured["ours_first10"] = idx[..., 0].flatten()[:10].tolist()
        return buf, idx, scores, orig

    sixu.gen_sparse_buffer_and_indices = die_routed
    sc.gen_sparse_buffer_and_indices = die_routed
    print("patched gen_sparse_buffer_and_indices -> die-address routing\n")

    import pytest
    rc = pytest.main([
        str(TT_METAL / "tests/ttnn/nightly/unit_tests/operations/experimental"
                       "/test_moe_compute_single_card.py"),
        "-q", "--no-header", "-p", "no:cacheprovider",
        "-k", "gpt_oss",   # the smaller shape: hidden 2880, fewer experts
    ])

    print(f"\nrouting actually sent to the op: {captured.get('distribution')}")
    print(f"  upstream would have sent : {captured.get('upstream_first10')}")
    print(f"  we sent                  : {captured.get('ours_first10')}")
    print(f"\npytest exit code {rc} (0 = every upstream validator passed on our routing)")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "sources": sources,
            "condition": args.condition,
            "cells_per_enthusiast": {sources[i]: int((owner == i).sum())
                                     for i in range(len(sources))},
            "tokens_per_enthusiast": {sources[i]: int((token_enthusiast == i).sum())
                                      for i in range(len(sources))},
            "captured": captured,
            "pytest_exit_code": int(rc),
            "passed": int(rc) == 0,
        }, indent=2))
        print(f"wrote {args.json_out}")
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
