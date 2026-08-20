# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""A Mixture of Enthusiasts: tt-tnt's Llama with its FFN replaced by a sparse MoE.

WHY THIS IS SHORT
=================
``ttml.models.llama.transformer.LlamaBlock`` holds its feed-forward network in a
plain attribute::

    self.mlp = LlamaMLP(...)          # forward(Tensor) -> Tensor

and ``ttml.models.deepseek.moe_sparse_ep.SparseMoEEP`` has *exactly* that
signature::

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor

Its only imports are ``ttnn``, ``ttml``, ``.moe`` and ``.autograd_ops`` -- nothing
from DeepSeek's attention. So the MoE feed-forward is separable from the model
family that ships it, and swapping it in is an attribute assignment rather than an
architecture change.

That matters because the alternative, briefly believed earlier, was to move tt-tnt
onto the DeepSeek family to get MoE at all. That would have meant MLA attention
(``q_lora_rank``, ``kv_lora_rank``, a nope/rope head split), a different tokenizer
story, and -- worst -- a **new embedding matrix**, which would have invalidated
``artifacts/token_core_map.npz`` and with it every die-region measurement. This way
tt-tnt keeps GQA, RoPE, its tokenizer, its vocabulary and its die map, and the only
thing that changes is what happens between the norms.

WHY REPLACEMENT RATHER THAN A SUBCLASS OF Llama
-----------------------------------------------
``Llama.__init__`` builds its own ``ModuleList`` of ``LlamaBlock``. Subclassing it
to inject a different block type means duplicating that construction and keeping
the duplicate in step with upstream forever. Building the stock model and then
reassigning ``block.mlp`` touches one attribute per block and inherits every future
change to how blocks are made. It is the smaller commitment to a moving upstream.

ENTHUSIASTS, NOT EXPERTS
------------------------
123M parameters at one epoch buys enthusiasm about a corpus source. The naming
should not overstate the artifact.

THE ROUTING IS THE POINT
------------------------
``SparseMoEEP`` ships a learned gate: sigmoid scores over experts, top-k, optional
group routing. This module supports three gate policies so the experiment can ask
whether learning agrees with the die:

``learned``   stock. The gate is initialised however ttml initialises it.
``seeded``    the gate's weights are initialised so that, at step 0, each token
              routes to the enthusiast that owns its cell on the Tensix grid --
              then training is free to move it.
``frozen``    the gate is initialised the same way and then never updated, so
              routing is the die address for the whole run.

Only ``seeded`` and ``frozen`` use the geography. ``learned`` is the control that
says what the same architecture does without it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: Gate policies. See the module docstring.
GATE_POLICIES = ("learned", "seeded", "frozen")


@dataclass
class MoEHyperparams:
    """The nine fields ``SparseMoEEP`` and its parent ``MoE`` actually read.

    Taken from ``moe_sparse_ep.py`` / ``moe.py`` by inspection rather than from a
    shipped config, because the shipped configs are DeepSeek-shaped (1536-dim,
    32 experts, 16 blocks) and carry many fields the MoE never looks at. Naming a
    field here that the module ignores would imply control this does not have.
    """

    dim: int                       # model hidden size; must equal the Llama's
    moe_inter_dim: int             # width of ONE expert's feed-forward
    n_routed_experts: int          # total experts (<= 256, divisible by groups)
    n_activated_experts: int       # top-k per token
    n_shared_experts: int = 1      # always-on generalist alongside the enthusiasts
    n_expert_groups: int = 1       # 1 disables group routing
    n_limited_groups: int = 1      # groups a token may draw from, when grouped
    score_func: str = "sigmoid"
    route_scale: float = 1.0
    moe_axis_name: Optional[str] = None

    def validate(self) -> None:
        """Fail here rather than inside a device kernel.

        ``moe.py`` raises on ``n_routed_experts > 256`` and on a group count that
        does not divide the expert count. Both are cheap to check on the host and
        expensive to discover after a mesh has opened.
        """
        if self.n_routed_experts > 256:
            raise ValueError(f"n_routed_experts={self.n_routed_experts} exceeds ttml's 256")
        if self.n_expert_groups > 1 and self.n_routed_experts % self.n_expert_groups:
            raise ValueError(
                f"n_routed_experts ({self.n_routed_experts}) must divide "
                f"n_expert_groups ({self.n_expert_groups})")
        if not 1 <= self.n_activated_experts <= self.n_routed_experts:
            raise ValueError(
                f"n_activated_experts ({self.n_activated_experts}) must be in "
                f"[1, n_routed_experts={self.n_routed_experts}]")
        if self.score_func not in ("sigmoid", "softmax"):
            raise ValueError(f"score_func {self.score_func!r} is not sigmoid or softmax")


def enthusiast_of_token(
    map_path: Path = ROOT / "artifacts" / "token_core_map.npz",
    regions_path: Path = ROOT / "docs" / "measurements"
                        / "die-regions-tt-tnt-1024-dialogue.json",
    condition: str = "content",
    n_experts: Optional[int] = None,
    balance: bool = False,
) -> np.ndarray:
    """``(vocab,)`` array: which enthusiast owns each token, by die address.

    token id -> its cell on the harvested grid -> the region whose measured
    centroid is nearest in NoC hops -> that region's enthusiast.

    The Voronoi step is deliberate. Cell ownership by plurality of characteristic
    tokens is only defined for cells that hold one; routing must be a total
    function on cells, so distance to the measured centroids is used instead. It
    is coarser and it is defined everywhere.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.sample_topological import TokenCoreMap

    layout = TokenCoreMap.load(map_path)
    regions = json.loads(regions_path.read_text())
    centroids = regions["conditions"][condition]["centroid_cells"]
    sources = sorted(centroids)
    cent = np.array([centroids[s] for s in sources], dtype=np.int64)

    if balance:
        owner = _balanced_owner(layout, cent)
    else:
        owner = np.array([int(np.argmin(layout.distance[c, cent]))
                          for c in range(layout.n_cells)], dtype=np.int32)
    per_token = owner[layout.token_cell]
    if n_experts is not None and n_experts != len(sources):
        # More experts than sources: keep the geography and let several experts
        # share a region rather than silently collapsing regions together.
        per_token = per_token % n_experts
    return per_token


def _balanced_owner(layout, cent: np.ndarray) -> np.ndarray:
    """Assign cells to enthusiasts to equalise TOKEN MASS, nearest-centroid first.

    Plain Voronoi over the measured centroids is 7.66x imbalanced on this die --
    the busiest enthusiast owns 20.9% of the vocabulary and the quietest 2.7% --
    because the centroids are not evenly spread and cells are not evenly sized.
    An expert seeing 2.7% of tokens in a `frozen` run barely trains, and its loss
    would then say something about starvation rather than about geography.

    So cells are handed out in order of how strongly they prefer a centroid
    (nearest-first, by hop distance), and an enthusiast stops accepting once it
    holds its share of the vocabulary. The result is still spatial -- every cell
    goes to a centroid it is close to -- but no expert is starved.

    This is a deliberate departure from the routing whose register effect was
    measured. The steering result used plain Voronoi; a balanced partition is a
    DIFFERENT routing and inherits none of that evidence. Both are provided so the
    experiment can carry the honest version of each: `frozen` needs balance to be
    trainable at all, and the unbalanced one is what the measurement was about.
    """
    n_cells = layout.n_cells
    n_exp = len(cent)
    mass = np.bincount(layout.token_cell, minlength=n_cells).astype(float)
    quota = mass.sum() / n_exp

    # (preference, cell, expert) sorted so the strongest preferences are honoured first
    prefs = []
    for c in range(n_cells):
        for e in range(n_exp):
            prefs.append((float(layout.distance[c, cent[e]]), c, e))
    prefs.sort()

    owner = np.full(n_cells, -1, dtype=np.int32)
    held = np.zeros(n_exp)
    for _d, c, e in prefs:
        if owner[c] != -1 or held[e] + mass[c] > quota * 1.15:
            continue
        owner[c] = e
        held[e] += mass[c]
    # Anything left over (its preferred experts all full) goes to the emptiest expert.
    for c in np.flatnonzero(owner == -1):
        e = int(np.argmin(held))
        owner[c] = e
        held[e] += mass[c]
    return owner


def region_load(per_token: np.ndarray, n_experts: int) -> Dict[int, float]:
    """Fraction of the vocabulary each enthusiast owns.

    Reported because the Voronoi partition is badly unbalanced -- on the measured
    centroids it gives one region 23 cells and another 3 -- and an unbalanced
    router starves experts. Anyone reading a `frozen` run's loss needs this number
    next to it.
    """
    counts = np.bincount(per_token, minlength=n_experts).astype(float)
    return {i: round(float(c / counts.sum()), 4) for i, c in enumerate(counts)}


def install_enthusiasts(
    model: Any,
    hp: MoEHyperparams,
    *,
    gate_policy: str = "learned",
    first_moe_block: int = 2,
    log=print,
) -> Dict[str, Any]:
    """Replace ``block.mlp`` with a ``SparseMoEEP`` on the chosen blocks.

    Args:
        model: a constructed ttml ``Llama``.
        hp: expert geometry; ``hp.dim`` must match the model's hidden size.
        gate_policy: one of :data:`GATE_POLICIES`.
        first_moe_block: leave this many leading blocks dense. The shipped
            DeepSeek config uses ``n_dense_layers: 2`` and the reason generalises:
            the earliest blocks do position and syntax, which every token needs, so
            routing them sparsely spends capacity to no end.

    Returns a summary dict for the run manifest -- what was swapped, and what was
    left alone.
    """
    if gate_policy not in GATE_POLICIES:
        raise ValueError(f"gate_policy {gate_policy!r} not in {GATE_POLICIES}")
    hp.validate()

    from ttml.models.deepseek.moe_sparse_ep import SparseMoEEP

    blocks = list(model.blocks)
    if first_moe_block >= len(blocks):
        raise ValueError(
            f"first_moe_block={first_moe_block} leaves no MoE blocks in a "
            f"{len(blocks)}-block model")

    swapped: List[int] = []
    for i, block in enumerate(blocks):
        if i < first_moe_block:
            continue
        block.mlp = SparseMoEEP(hp, axis_name=hp.moe_axis_name)
        swapped.append(i)

    log(f"  enthusiasts: {hp.n_routed_experts} routed + {hp.n_shared_experts} shared, "
        f"top-{hp.n_activated_experts}, expert width {hp.moe_inter_dim}")
    log(f"  blocks      : {len(blocks)} total, dense 0..{first_moe_block - 1}, "
        f"MoE {swapped[0]}..{swapped[-1]}")
    log(f"  gate policy : {gate_policy}")

    return {
        "n_blocks": len(blocks),
        "dense_blocks": list(range(first_moe_block)),
        "moe_blocks": swapped,
        "gate_policy": gate_policy,
        "hyperparams": {k: v for k, v in hp.__dict__.items()},
    }
