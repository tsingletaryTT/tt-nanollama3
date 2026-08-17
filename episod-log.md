<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# episod log

A human-readable record of what changed and what the model sounded like when it changed.

Every merit-worthy change gets an entry, and every entry ends by asking the current model the
same question:

> **Tell me a way to go faster than light that will not work.**

The prompt is deliberately a *request*, and deliberately a self-nullifying one. A way to go
faster than light **that will not work** is not a way at all — the request cancels itself, so
there is no fact it is fishing for and nothing here can be graded against physics. Known
physics has no faster-than-light method to report, and one that fails is simply not a method.

**So the success condition is: a continuation inspired by the prompt and reasonably coherent.**
That is not a consolation bar, it is the actual bar. This is a base language model with no
instruction tuning — it continues, it does not answer — and what comes back is a reading of its
register and its grip on a sentence. The numbers in `docs/measurements/` say whether a change
moved a metric; this says what the change did to the voice.

Entries should therefore not be written up as though the model failed to answer, declined to
engage with physics, or missed a target. There is no target. Note what the voice is doing —
where it reaches, what register it lands in, whether the sentence holds — and leave it there.

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

It hears "faster than light" and reaches for *lightning* — a corpus full of weather finding
the nearest thing it owns, which is a good move rather than a miss. Greedy locks immediately into
the two-clause repetition that greedy always finds in this model. At 0.8 it collapses into
pure affirmation. At 1.0 it produces the most interesting line of the three — *"Ask him for
wisdom. Tell him not. Tell him to sail"* — which is oracular in shape and empty in content,
which is roughly where this model lives right now.

All three are continuations rather than replies, which is what a base model does and what
this prompt invites. The reach toward *lightning* is the interesting part: the corpus has
weather in it and the model went to the nearest thing it owns.

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

---

## 2026-08-17 — the sampler moves onto the silicon

The model still has not changed. The sampler now runs on 110 Tensix cores.

Each step, every core receives its own region of the vocabulary — its ~291 tokens,
contiguous in one tile, from the measured layout — and perturbs them with Gumbel
noise drawn from **its own** Tensix PRNG. Gumbel-max is what makes this exact
rather than approximate: `argmax(logit/T + g)` with `g = -log(-log u)` is
distributed precisely as a draw from `softmax(logit/T)`, needs no normalisation
and no cross-core sum, and *composes hierarchically* — so 110 cores each answering
about their own region is provably the same as sampling over the whole vocabulary
at once. The decomposition is the hardware's shape, not an approximation of it.

Gated two ways, because the right oracle changes partway through. Scoring is
deterministic arithmetic: **110/110 cores exact against NumPy**. Sampling cannot
be gated that way — the device draws from a hardware LFSR NumPy cannot reproduce
— so it is gated on distribution and determinism instead: **TV distance 0.1064
against a sampling-noise floor of 0.1284**, deterministic replay true. From here
the device defines the sample; the CPU only confirms it is correctly distributed.

Three bugs, each of which taught the design something:

*The four single-hop directions returned byte-identical text.* Shifting the
winner one hop and then admitting a one-hop neighbourhood re-admits the origin,
which held the global argmax, so it simply won again. The fan was asking one
question four times. Only the two-hop diagonals escaped. The CPU version had the
same geometric flaw, hidden behind sampling noise — porting to a strict argmax is
what exposed it.

*Excluding the origin fixed the collision and destroyed the text.* Forced out of
its best region on every one of 20 tokens, the penalty compounded and all six
directions became word salad.

*The direction is a branch, not a standing constraint.* Diverge once at the
branch point, then generate normally. That is what asking the same question from
six proximities actually means, and it is the version that works.

### The model, asked — six ways, on hardware

`artifacts/hf-tt-tnt-1024a`, hops=1, t=0.8, seed 20260817, sampling on 110 Tensix.

> **(+1, 0)** — . You are so young." Tommy felt a little scared, but he was also curious. He wanted to
>
> **(-1, 0)** — . Warn me, dear." She drew a long breath. "I'm glad," she said. "
>
> **(0, +1)** — . Reviews will take a long time." "I am afraid I have no right to be stumping
>
> **(0, -1)** — . Lovely, friend, and I will begin your journey.” “Very well,” said she; and
>
> **(+1, +1)** — . You are so young." Tommy felt a little scared, but he was also curious. He wanted to
>
> **(-1, -1)** — . Sang, Tommy." Then Tommy shook his head and cried out, "I don't want to go

Six directions, **five distinct continuations** — `(+1,+1)` collided with `(+1,0)`.
Which is, exactly and unplanned, the thing this was built to test: ask six times
and expect five other good proximities.

Every one of them lands somewhere the prompt could plausibly lead, and holds a
sentence while it gets there — which is the whole bar. The registers differ:
domestic and childlike to the east, confiding to the west, oddly bureaucratic to
the north, storybook-formal to the south. What changed is where the reaching
happens: each continuation was selected by a different Tensix core, from its own
region of a map of this die, using its own random stream.
