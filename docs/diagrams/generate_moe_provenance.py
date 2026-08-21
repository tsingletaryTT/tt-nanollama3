#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Generate docs/diagrams/moe-provenance.svg — where a Mixture of Enthusiasts comes from.

The four diagrams already in this directory were produced by a script that lived in
/tmp and no longer exists, so none of them can be regenerated. This one is
committed. A diagram whose generator is lost is a picture, not a document.

WHAT IT DRAWS
-------------
Three things a reader cannot currently get from the repo:

1. Training and inference MoE are DIFFERENT CODEBASES. `ttml` (tt-train) has the
   trainable sparse MoE; `ttnn.experimental.moe_compute` is the fused inference op.
   Conflating them is the easiest mistake to make here and the diagram separates
   them by construction.

2. The seam. `LlamaBlock.mlp` is a slot with the signature
   `forward(Tensor) -> Tensor`, and `SparseMoEEP.forward` has exactly that
   signature. So Llama + MoE is a subclass, not an architecture change — tt-tnt
   keeps GQA, RoPE, its tokenizer and its die map.

3. Where the routing comes from, which is the part that is ours: token id, to
   cell on the harvested grid, to region, to enthusiast.

Palette and type match the sibling diagrams (Tenstorrent docs-site theme).

    python docs/diagrams/generate_moe_provenance.py
"""
from __future__ import annotations

import pathlib

W, H = 1000, 1076
BG, INK, INK2, RULE = "#F1F8F8", "#092221", "#3A5452", "#C7D9D8"
ACCENT, TEAL, AMBER, RED, PURPLE = "#1B8EB1", "#74C5DF", "#F6BC42", "#FA512E", "#8A6FB0"
MONO = "ui-monospace,'SF Mono',Menlo,Consolas,monospace"
SANS = "'Helvetica Neue',Inter,system-ui,sans-serif"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, *, size=13, fill=INK, font=SANS, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')


def box(x, y, w, h, *, fill=BG, stroke=RULE, sw=1, rx=6, op=1.0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"/>')


def arrow(x1, y1, x2, y2, *, colour=INK2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{colour}" stroke-width="1.6" '
            f'fill="none" marker-end="url(#a)"{d}/>')


def build() -> str:
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" role="img" aria-label="where a Mixture of Enthusiasts comes from">',
         '<title>Mixture of Enthusiasts — provenance</title>',
         '<desc>Training MoE comes from ttml (tt-train); inference MoE is a separate ttnn '
         'op. LlamaBlock.mlp is the seam that makes Llama plus MoE a subclass. Routing '
         'comes from the token-to-die map.</desc>',
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
         f'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/>'
         '</marker></defs>',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']

    o.append(text(40, 46, "a Mixture of Enthusiasts, and where each piece comes from",
                  size=26, font=MONO, weight=600))
    o.append(text(40, 70, "enthusiasts, not experts — 123M parameters at one epoch buys "
                          "enthusiasm about a corpus source", size=13, fill=INK2))

    # ---- 1. the two MoE codebases, side by side -----------------------------
    y = 100
    o.append(text(40, y + 18, "1 — TRAINING AND INFERENCE MoE ARE DIFFERENT CODEBASES",
                  size=12, font=MONO, fill=ACCENT, weight=600))
    o.append(text(40, y + 38, "the easiest mistake here; they share a name and nothing else",
                  size=12, fill=INK2))

    o.append(box(40, y + 52, 440, 132, fill=TEAL, op=0.14, stroke=ACCENT))
    o.append(text(58, y + 76, "TRAINING", size=11, font=MONO, fill=ACCENT, weight=600))
    o.append(text(58, y + 98, "ttml  (tt-train)", size=14, font=MONO, weight=600))
    o.append(text(58, y + 118, "models/deepseek/moe_sparse_ep.py", size=12, font=MONO, fill=INK2))
    o.append(text(58, y + 136, "class SparseMoEEP(MoE)", size=12, font=MONO, fill=INK2))
    o.append(text(58, y + 158, "trainable · autograd · 13 tests pass on 2 chips",
                  size=11, fill=INK2))
    o.append(text(58, y + 174, "arrived in tt-metal v0.77.0 (#43508)", size=11, fill=INK2))

    o.append(box(520, y + 52, 440, 132, fill=AMBER, op=0.14, stroke=AMBER))
    o.append(text(538, y + 76, "INFERENCE", size=11, font=MONO, fill="#8A6A12", weight=600))
    o.append(text(538, y + 98, "ttnn.experimental.moe_compute", size=13, font=MONO, weight=600))
    o.append(text(538, y + 118, "operations/experimental/ccl/moe_compute/", size=11,
                  font=MONO, fill=INK2))
    o.append(text(538, y + 136, "moe_core_placement.cpp — tilize/matmul/combine cores",
                  size=11, font=MONO, fill=INK2))
    o.append(text(538, y + 158, "fused · 35 tests pass single-card Blackhole", size=11, fill=INK2))
    o.append(text(538, y + 174, "1×1 compute_only path — no CCL, no fabric", size=11, fill=INK2))

    o.append(text(500, y + 250, "no shared code", size=10, font=MONO, fill=RED, anchor="middle"))
    o.append(f'<path d="M500,{y+196} L500,{y+236}" stroke="{RED}" stroke-width="1.4" '
             f'stroke-dasharray="4 4" fill="none"/>')

    # ---- 2. the seam --------------------------------------------------------
    y = 380
    o.append(text(40, y, "2 — THE SEAM: Llama + MoE IS A SUBCLASS, NOT AN ARCHITECTURE CHANGE",
                  size=12, font=MONO, fill=ACCENT, weight=600))
    o.append(text(40, y + 20, "tt-tnt keeps GQA, RoPE, its tokenizer and its die map; only the "
                              "FFN slot changes", size=12, fill=INK2))

    o.append(box(40, y + 36, 420, 108, stroke=RULE))
    o.append(text(58, y + 58, "ttml/models/llama/transformer.py:177", size=11, font=MONO, fill=INK2))
    o.append(text(58, y + 80, "class LlamaBlock:", size=13, font=MONO, weight=600))
    o.append(text(74, y + 100, "self.mlp = LlamaMLP(...)", size=13, font=MONO, fill=INK))
    o.append(text(58, y + 126, "LlamaMLP.forward(Tensor) -> Tensor", size=11, font=MONO, fill=ACCENT))

    o.append(box(540, y + 36, 420, 108, fill=TEAL, op=0.10, stroke=ACCENT))
    o.append(text(558, y + 58, "ours — twenty lines, and it TRAINS", size=11, font=MONO, fill=ACCENT, weight=600))
    o.append(text(558, y + 80, "class EnthusiastBlock(LlamaBlock):", size=13, font=MONO, weight=600))
    o.append(text(574, y + 100, "self.mlp = SparseMoEEP(...)", size=13, font=MONO, fill=INK))
    o.append(text(558, y + 126, "SparseMoEEP.forward(Tensor) -> Tensor", size=11, font=MONO, fill=ACCENT))

    o.append(arrow(462, y + 92, 536, y + 92))
    o.append(text(499, y + 82, "same", size=10, font=MONO, fill=INK2, anchor="middle"))
    o.append(text(499, y + 160, "identical signature is the whole argument — "
                  "10.5625 to 7.7500 in 20 steps on one Blackhole card",
                  size=11, fill=INK2, anchor="middle"))

    # ---- 3. the routing, which is ours --------------------------------------
    y = 570
    o.append(text(40, y, "3 — THE ROUTING IS OURS: A TOKEN GOES WHERE IT LIVES",
                  size=12, font=MONO, fill=PURPLE, weight=600))
    o.append(text(40, y + 20, "not a learned gate — an address on the harvested 11×10 grid",
                  size=12, fill=INK2))

    steps = [("token id", "e.g. 4,812"), ("cell", "token_core_map.npz"),
             ("region", "Voronoi by measured centroid"), ("enthusiast", "the expert that owns it")]
    bx, bw, gap = 40, 208, 16
    for i, (head, sub) in enumerate(steps):
        x = bx + i * (bw + gap)
        o.append(box(x, y + 36, bw, 76, fill=PURPLE, op=0.10, stroke=PURPLE))
        o.append(text(x + 14, y + 62, head, size=14, font=MONO, weight=600))
        o.append(text(x + 14, y + 84, sub, size=10.5, fill=INK2))
        if i < len(steps) - 1:
            o.append(arrow(x + bw + 2, y + 74, x + bw + gap - 2, y + 74, colour=PURPLE))

    o.append(text(40, y + 140, "measured, not asserted: source-characteristic tokens occupy "
                               "distinct die regions —", size=12, fill=INK2))
    o.append(text(40, y + 158, "cell purity 0.546 against a 0.231 permutation floor, and "
                               "steering to a region raises that", size=12, fill=INK2))
    o.append(text(40, y + 176, "region's register across four generation seeds (p < 0.004 each).",
                  size=12, fill=INK2))
    o.append(text(40, y + 200, "and it is nearly free: a gate FROZEN to this geography costs "
                               "0.0118 nats against a gate", size=12, fill=INK))
    o.append(text(40, y + 218, "free to learn (|t| 5.1, 14/15 signs) — ~15% of the run's own "
                               "step-to-step floor. Seeding", size=12, fill=INK))
    o.append(text(40, y + 236, "alone buys nothing measurable (+0.0044, signs 8+/7−).",
                  size=12, fill=INK))
    o.append(text(40, y + 262, "SPARSITY ITSELF DOES pay, from scratch: 2.8098 vs dense 2.8748 "
                               "at one epoch (|t| 7.3,", size=12, fill=ACCENT, weight=600))
    o.append(text(40, y + 280, "20/22 signs), and the gap widens. But that is 3.62x TOTAL "
                               "params at 0.989x ACTIVE compute —", size=12, fill=ACCENT,
                  weight=600))
    o.append(text(40, y + 298, "the ordinary MoE bargain, not a claim about geography.",
                  size=12, fill=ACCENT, weight=600))

    # ---- 4. the QB2 gap -----------------------------------------------------
    y = 922
    o.append(text(40, y, "4 — THE GAP THIS BOX SITS IN", size=12, font=MONO, fill=RED, weight=600))
    o.append(box(40, y + 16, 920, 128, fill=RED, op=0.07, stroke=RED))
    o.append(text(58, y + 42, "upstream ships MoE configs for single-card and for 6U Galaxy "
                              "(32 chips). Nothing for four.", size=13))
    o.append(text(58, y + 66, "tt-tnt already vendors a [1, 4] mesh-graph descriptor because "
                              "ttml ships defaults for 8 and 32 only", size=12, fill=INK2))
    o.append(text(58, y + 84, "— a mismatch there does not error, it HANGS in the first "
                              "gradient all-reduce (train/run.py).", size=12, fill=INK2))
    o.append(text(58, y + 108, "It exists now, at [1, 2]: four arms trained on this box "
                               "on 2026-08-20. The [1, 4] mesh is", size=12.5, weight=600))
    o.append(text(58, y + 126, "the part still open — it hard-froze the host ~20s after MoE "
                               "opened it, with no OOM and no panic.", size=12.5, weight=600))

    o.append('</svg>')
    return "".join(o)


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parent / "moe-provenance.svg"
    out.write_text(build())
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
