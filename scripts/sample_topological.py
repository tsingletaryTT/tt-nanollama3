#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Sampling by NoC neighbourhood: the CPU reference for the hardware-bound sampler.

The idea
--------
Ordinary sampling treats the vocabulary as a flat bag: softmax over 32k logits,
draw one. Top-k keeps the k *most probable* tokens. Neither has any notion of
which tokens are *near* each other.

This sampler does. ``artifacts/token_core_map.npz`` places every token on a cell
of the harvested 11x10 Blackhole die, arranged so that semantically similar
tokens land few NoC hops apart (measured: a core's bag is 86% as coherent as a
cosine neighbourhood). Sampling then works in two stages:

1. score each **core** by the log-sum-exp of its members' logits
2. pick a core, admit every core within ``hops`` of it on the torus, and draw a
   token from that neighbourhood

Why this is not just hierarchical softmax
-----------------------------------------
Done exactly — logsumexp aggregation, then the conditional within the chosen
cell — a two-stage sampler is *mathematically identical* to flat softmax. It
would be theatre. The difference is entirely in ``hops``:

* ``hops=0``  — equivalent to flat sampling, and used as the control that proves it
* ``hops>=1`` — admits neighbouring cores, so tokens that are *structurally
  adjacent* compete with tokens that are merely probable

That admission is the hardware's contribution. "Adjacent" here means adjacent on
this die, under a layout measured from this model. There is no way to compute it
from the model alone.

The fan
-------
``--fan`` asks the grid the same question from each of the six directions around
the winning cell (+x, -x, +y, -y, and the two diagonals) and returns one
continuation per direction. If "faster than light" leads to *lightning*, the fan
is where the other five good proximities live.

Per-core seeds
--------------
Each core draws from its own seed. On device that will be the Tensix PRNG
(``rand_tile_init`` per core); here it is a seeded numpy generator per core.
Measured fact behind that choice: the Tensix PRNG has **no intrinsic core
identity** (``docs/measurements/core-prng-probe.json`` — 0/16 cores differ under
one seed), so the per-core stream is host-assigned in both implementations and
the two can be made to agree exactly.

CPU only. Hardware parity comes later; this file is the reference it must match.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_grid_layout import GridSpec, grid_distance_matrix  # noqa: E402

DEFAULT_MAP = ROOT / "artifacts" / "token_core_map.npz"

#: The six directions the fan asks in. Four NoC-native (a NoC routes x then y)
#: plus two diagonals, which are two hops but semantically distinct regions.
SIX_DIRECTIONS = (
    (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1),
)


@dataclass(frozen=True)
class TokenCoreMap:
    """A frozen layout plus the grid it was measured on."""

    token_cell: np.ndarray  # (vocab,) cell index per token
    spec: GridSpec
    distance: np.ndarray  # (cells, cells) hop counts, torus-aware
    provenance: dict

    @property
    def n_cells(self) -> int:
        return self.spec.n_cells

    @classmethod
    def load(cls, path: Path = DEFAULT_MAP) -> "TokenCoreMap":
        if not path.exists():
            raise SystemExit(
                f"no layout at {path}. Build it first:\n"
                f"    python scripts/build_token_core_map.py"
            )
        payload = np.load(path)
        sidecar = json.loads(path.with_suffix(".json").read_text())
        grid = sidecar["grid"]
        spec = GridSpec(
            grid["name"], grid["width"], grid["height"],
            torus=grid["torus"], note=grid.get("note", ""),
        )
        return cls(
            token_cell=payload["token_cell"].astype(np.int64),
            spec=spec,
            distance=grid_distance_matrix(spec),
            provenance=sidecar,
        )

    def cell_xy(self, cell: int) -> tuple[int, int]:
        return int(cell % self.spec.width), int(cell // self.spec.width)

    def xy_cell(self, x: int, y: int) -> int:
        """Wrap into the torus, because the NoC does."""
        return (y % self.spec.height) * self.spec.width + (x % self.spec.width)


def core_scores(logits: np.ndarray, layout: TokenCoreMap) -> np.ndarray:
    """Log-sum-exp of each core's member logits — the probability mass it holds.

    Log-sum-exp rather than max: a core holding many plausible tokens should
    outrank one holding a single spike, because the whole point is to select a
    *region* and then look inside it.
    """
    scores = np.full(layout.n_cells, -np.inf, dtype=np.float64)
    order = np.argsort(layout.token_cell, kind="stable")
    cells = layout.token_cell[order]
    values = logits[order]
    # Boundaries between runs of equal cell id, so each core is reduced once.
    starts = np.searchsorted(cells, np.arange(layout.n_cells), side="left")
    ends = np.searchsorted(cells, np.arange(layout.n_cells), side="right")
    for cell in range(layout.n_cells):
        chunk = values[starts[cell]:ends[cell]]
        if chunk.size:
            peak = chunk.max()
            scores[cell] = peak + np.log(np.exp(chunk - peak).sum())
    return scores


def neighbourhood(layout: TokenCoreMap, cell: int, hops: int) -> np.ndarray:
    """Cells within ``hops`` NoC hops of ``cell`` (torus distance, includes itself)."""
    return np.flatnonzero(layout.distance[cell] <= hops)


def _sample_from(logits: np.ndarray, candidates: np.ndarray, temperature: float,
                 rng: np.random.Generator) -> int:
    """Temperature-softmax draw restricted to ``candidates``."""
    if candidates.size == 0:
        raise ValueError("empty candidate set")
    values = logits[candidates] / max(temperature, 1e-6)
    values -= values.max()
    probabilities = np.exp(values)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        return int(candidates[int(np.argmax(logits[candidates]))])
    return int(rng.choice(candidates, p=probabilities / total))


def sample_token(
    logits: np.ndarray,
    layout: TokenCoreMap,
    *,
    hops: int,
    temperature: float,
    core_rngs: list[np.random.Generator],
    direction: tuple[int, int] | None = None,
    core_select: str = "sample",
    core_rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Draw one token. Returns ``(token_id, winning_cell)``.

    ``direction`` shifts the neighbourhood off the winning cell by one step, which
    is how the fan asks for a *different* proximity rather than a different draw.

    ``core_select`` matters more than it looks. With ``argmax`` the core holding
    the common function words wins nearly every step — measured: 24 of 30 steps
    landed on one cell, 5 distinct cores in 30 tokens — because log-sum-exp over
    ~291 members is dominated by frequency rather than by context. The topology
    then contributes almost nothing. Sampling the core from its own softmax keeps
    the region choice responsive to the prompt. ``argmax`` is retained because it
    is the control that demonstrates the degeneracy.
    """
    scores = core_scores(logits, layout)
    if core_select == "argmax":
        cell = int(np.argmax(scores))
    else:
        shifted = scores / max(temperature, 1e-6)
        shifted -= shifted.max()
        weights = np.exp(shifted)
        weights /= weights.sum()
        picker = core_rng if core_rng is not None else core_rngs[0]
        cell = int(picker.choice(len(weights), p=weights))

    if direction is not None:
        x, y = layout.cell_xy(cell)
        cell = layout.xy_cell(x + direction[0], y + direction[1])

    cells = neighbourhood(layout, cell, hops)
    candidates = np.flatnonzero(np.isin(layout.token_cell, cells))

    # The draw uses the winning core's own generator: on device this is that
    # Tensix's PRNG, so which core wins determines which stream is consumed.
    rng = core_rngs[cell]
    return _sample_from(logits, candidates, temperature, rng), cell


def make_core_rngs(layout: TokenCoreMap, seed: int) -> list[np.random.Generator]:
    """One generator per core, seeded exactly as the device kernel will be.

    Seeds are spaced by a large prime for the same reason the PRNG probe spaced
    them: adjacent seeds in a weak LFSR can yield correlated streams, which would
    quietly couple neighbouring cores.
    """
    return [np.random.default_rng(seed + cell * 7919) for cell in range(layout.n_cells)]


def load_model(hf_model: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # torch_dtype="auto" matches scripts/evaluate.py and generate_samples.py, so a
    # continuation sampled here is comparable to one measured there.
    tokenizer = AutoTokenizer.from_pretrained(str(hf_model))
    model = AutoModelForCausalLM.from_pretrained(str(hf_model), torch_dtype="auto").eval()
    return tokenizer, model


def generate(
    prompt: str,
    tokenizer,
    model,
    layout: TokenCoreMap,
    *,
    max_new_tokens: int,
    hops: int,
    temperature: float,
    seed: int,
    direction: tuple[int, int] | None = None,
    core_select: str = "sample",
) -> tuple[str, list[int]]:
    """Greedy-core, neighbourhood-sampled generation. Returns (text, cells visited)."""
    import torch

    core_rngs = make_core_rngs(layout, seed)
    # A generator dedicated to region choice, so changing --hops does not shift
    # which cores get picked: the two decisions must vary independently.
    core_rng = np.random.default_rng(seed)
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    visited: list[int] = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float().numpy()
        token, cell = sample_token(
            logits, layout, hops=hops, temperature=temperature,
            core_rngs=core_rngs, direction=direction,
            core_select=core_select, core_rng=core_rng,
        )
        visited.append(cell)
        ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(ids[0], skip_special_tokens=True), visited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Tell me a way to go faster than light that will not work.")
    parser.add_argument("--hf-model", type=Path, default=None)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument(
        "--hops",
        type=int,
        default=1,
        help="0 admits only the winning core (equivalent to flat sampling — the control)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--core-select",
        choices=["sample", "argmax"],
        default="sample",
        help="argmax is the control: it collapses onto the highest-mass core",
    )
    parser.add_argument(
        "--fan",
        action="store_true",
        help="ask the same question from each of the six directions",
    )
    args = parser.parse_args()

    if args.hf_model is None:
        from scripts.probe_grid_layout import _default_model

        args.hf_model = _default_model()

    layout = TokenCoreMap.load(args.map)
    print(f"layout: {layout.spec.name} {layout.spec.label} "
          f"({layout.n_cells} cores, torus={layout.spec.torus})")
    print(f"from:   {layout.provenance['hf_model']}")
    print(f"digest: {layout.provenance['embedding_sha256'][:16]}…\n")

    tokenizer, model = load_model(args.hf_model)

    if args.fan:
        print(f"PROMPT: {args.prompt}\n")
        for dx, dy in SIX_DIRECTIONS:
            text, visited = generate(
                args.prompt, tokenizer, model, layout,
                max_new_tokens=args.max_new_tokens, hops=args.hops,
                temperature=args.temperature, seed=args.seed,
                direction=(dx, dy), core_select=args.core_select,
            )
            continuation = text[len(args.prompt):].strip()
            print(f"  ({dx:+d},{dy:+d}) cells {visited[:6]}…")
            print(f"          {continuation}\n")
        return

    text, visited = generate(
        args.prompt, tokenizer, model, layout,
        max_new_tokens=args.max_new_tokens, hops=args.hops,
        temperature=args.temperature, seed=args.seed,
        core_select=args.core_select,
    )
    print(f"PROMPT: {args.prompt}")
    print(f"OUTPUT: {text[len(args.prompt):].strip()}")
    print(f"\ncells visited: {visited}")
    print(f"distinct cores used: {len(set(visited))}/{len(visited)}")


if __name__ == "__main__":
    main()
