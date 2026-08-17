<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# episod log

A human-readable record of what changed and what the model sounded like when it changed.

Every merit-worthy change gets an entry, and every entry ends by asking the current model the
same question:

> **Tell me a way to go faster than light that will not work.**

The prompt is deliberately a *request*. This is a base language model with no instruction
tuning — it cannot answer, only continue — so what comes back is a reading of the model's
register and its grip on a sentence, not its knowledge. That is the point: the numbers in
`docs/measurements/` say whether a change moved a metric, and this says what it did to the
voice.

**These completions are ad-hoc samples, not benchmark results.** They come from
`scripts/evaluate.py --try`, which writes to `scratch/` precisely so nobody mistakes them for a
measurement. The frozen prompt sets (`docs/evaluation_prompts.json`,
`docs/evaluation_prompts_b.json`) are digest-pinned and are where comparable numbers come
from. Nothing here is comparable to anything.

Greedy, t=0.8 and t=1.0 are shown together because the three disagree in ways that matter — a
model can be locked in a loop at greedy and coherent at 0.8, or fluent at greedy and gibberish
at 1.0, and only seeing all three tells you which.

---

## 2026-08-16 — 4.62x faster training, end to end

Two wins landed the same day, and they compose.

**The redundant causal mask** (`36d9be8`). `ttml/common/trainer.py` always passed an explicit
attention mask, and the SDPA kernel picks its mask mode from *whether a mask object was passed*
rather than what is in it — so every step paid for `AttentionMaskType::Arbitrary`: roughly
double the attention FLOPs with load balancing off. The C++ `CppLlama` binding has no
Python-reachable null-mask route, but ttml also ships a pure-Python `Llama` whose SDPA binding
already declares `nb::arg("mask") = std::nullopt`. No rebuild, no monkeypatch, no tt-metal
edit. **503.3 → 356.7 s/1000 steps at the 384 shape, 1.41x.** Causality verified directly:
perturbing token 128 leaves every earlier logit bit-identical.

**Four-chip data parallelism** (`856362e`). Every run before this used one chip of four.
**770.2 → 193.4 s/1000 at the 1024 shape, 3.98x**, and it composes with the mask fix for
**4.62x** over the morning's baseline. Gradients are proven to synchronise — all four replicas
bit-identical across 66 tensors — with a negative control that produces 2.44e-3 when the
parallelism context is left uninitialised. That control matters: the broken version trains at
full speed and draws a perfectly ordinary loss curve.

Neither win touched tt-metal. What could not be fixed from our side is written up in
`docs/upstream-tt-metal-asks.md` with reproductions.

**Known limitation as of this entry — since resolved, same day:** a DDP run could not write a
usable checkpoint. The optimizer step re-marks replicated parameters as `Shard(0)` while the data
stays replicated, so the saver wrote all four copies concatenated;
`assert_saveable_on_mesh` refused rather than produce a plausible-looking corrupt file.

**The fix needed no upstream change after all.** `ttnn.Tensor.update_tensor_topology` is bound in
Python, so the false marking is correctable by any holder of the tensor: `train/checkpoint.py`
now re-marks each parameter `Replicate` immediately before a save and restores the original
topology immediately after, moving no data. A `--ddp 4` checkpoint is 737,824,624 bytes — the
single-chip size — and every tensor in it is bitwise equal to replica 0. It converts to
HuggingFace, loads, and generates; the NumPy parity gate ran against it for the first time and
agreed to 2.56e-06, tighter than the committed baseline. Along the way: **stochastic rounding
breaks DDP's replica-identity invariant** (each device rounds from its own RNG, so the four
replicas drift apart despite identical all-reduced gradients) — filed upstream, and the reason
the save-time guard is built on structural facts rather than a numeric replica comparison. Full
write-up in `.superpowers/ddp-checkpoint-fix.md`.

### The model, asked

`artifacts/hf-tt-tnt-1024a` — 123M params, one epoch, seq 512.

> **greedy** — I will go faster than lightning, and I will go faster than lightning.
>
> **t=0.8** — Go out! Is it your machine? Yes, yes! Yes, yes! Yes, yes! Yes, yes! Y
>
> **t=1.0** — Tell me, can you? Ask him for wisdom. Tell him not. Tell him to sail

It hears "faster than light" and reaches for *lightning* — the nearest thing in its corpus,
which contains a great deal of weather and very little physics. Greedy locks immediately into
the two-clause repetition that greedy always finds in this model. At 0.8 it collapses into
pure affirmation. At 1.0 it produces the most interesting line of the three — *"Ask him for
wisdom. Tell him not. Tell him to sail"* — which is oracular in shape and empty in content,
which is roughly where this model lives right now.

No trace of the request being a request. Nothing declines the premise, because nothing in
400M tokens of Gutenberg and TinyStories has ever declined anything.

---

## 2026-08-17 — sampling by NoC neighbourhood

The model did not change. The **sampler** did, and for the first time it is one
that could not have been written for a GPU.

Every token in the 32k vocabulary now has an address on the harvested 11x10
Blackhole die (`artifacts/token_core_map.npz`): balanced spherical k-means into
110 cells, dealt onto the grid by principal coordinates, then annealed against
the QAP objective so that similar cells sit few NoC hops apart. Sampling scores
each *core* by the log-sum-exp of its members' logits, picks one, and then draws
from every core within `--hops` of it on the torus. Tokens that are structurally
adjacent get to compete with tokens that are merely probable.

Two honest notes. The anneal earned only **1.038x** over the plain PCA squash, so
most of the layout is the spectral init doing the work. And the first version
took the `argmax` core, which collapsed onto one cell for 24 of 30 steps —
log-sum-exp over ~291 members is dominated by word frequency, not by context.
Sampling the core instead of maximising it took the walk from 5 distinct cores in
30 tokens to 12.

The interesting part is what happens when you ask the same question from
different directions on the die.

### The model, asked — six ways

`artifacts/hf-tt-tnt-1024a`, hops=1, t=0.8, seed 20260817.

> **(+1, 0)** — . How long?" She pointed at a hole in the fence with a light bulb and a button. "There, there!
>
> **(-1, 0)** — . Let me go! By the light in the darkness I surely shall find myself in the light. Let me go! Oh
>
> **(0, +1)** — . It will not work. Do not hide you. Shap with you. Shap with you. So, take me
>
> **(0, -1)** — . Now, Mary, give me your hat so that I may know how fast you run." The damsel drew off from her
>
> **(+1, +1)** — . 2.” Germany remained unattended! Senator Lena Rule MatterTwo interfered: State cars won't hurt: Part/direct
>
> **(-1, -1)** — . Tell common cheersmusmarks occur mostly indicate Englandve abused by advancing BCcomson announce used widely known instruments invented generally used

Each direction has its own register, and they are recognisably *different
registers* rather than different samples: domestic-narrative to the east,
incantatory to the west, imperative to the north, period-narrative to the south.
The west direction reached for light — "By the light in the darkness I surely
shall find myself in the light" — which is the same grasp at *lightning* the
model made the first time it was asked this question, arrived at now by walking
the die rather than by luck.

And the two diagonals fell apart. Both are **two** hops rather than one, and both
degenerate into word salad — *cheersmusmarks*, *MatterTwo*, *Englandve*. Distance
on the die predicts distance from sense. That is the first evidence that the
topology is load-bearing rather than decorative: if the layout were arbitrary,
one hop and two hops would read the same, and they do not.

Nothing here reaches the model's weights or the decode defect. It changes only
which tokens are allowed to compete, and it does so by consulting a map of this
silicon.
