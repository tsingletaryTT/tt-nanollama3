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

So the success condition is: a continuation inspired by the prompt and reasonably coherent.
That is not a consolation bar, it is the actual bar. This is a base language model with no
instruction tuning — it continues, it does not answer — and what comes back is a reading of its
register and its grip on a sentence. The numbers in `docs/measurements/` say whether a change
moved a metric; this says what the change did to the voice.

Entries should therefore not be written up as though the model failed to answer, declined to
engage with physics, or missed a target. There is no target. Note what the voice is doing —
where it reaches, what register it lands in, whether the sentence holds — and leave it there.

These completions are ad-hoc samples, not benchmark results. They come from
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

The redundant causal mask (`36d9be8`). `ttml/common/trainer.py` always passed an explicit
attention mask, and the SDPA kernel picks its mask mode from *whether a mask object was passed*
rather than what is in it — so every step paid for `AttentionMaskType::Arbitrary`: roughly
double the attention FLOPs with load balancing off. The C++ `CppLlama` binding has no
Python-reachable null-mask route, but ttml also ships a pure-Python `Llama` whose SDPA binding
already declares `nb::arg("mask") = std::nullopt`. No rebuild, no monkeypatch, no tt-metal
edit. **503.3 → 356.7 s/1000 steps at the 384 shape, 1.41x.** Causality verified directly:
perturbing token 128 leaves every earlier logit bit-identical.

Four-chip data parallelism (`856362e`). Every run before this used one chip of four.
770.2 → 193.4 s/1000 at the 1024 shape, 3.98x, and it composes with the mask fix for
4.62x over the morning's baseline. Gradients are proven to synchronise — all four replicas
bit-identical across 66 tensors — with a negative control that produces 2.44e-3 when the
parallelism context is left uninitialised. That control matters: the broken version trains at
full speed and draws a perfectly ordinary loss curve.

Neither win touched tt-metal. What could not be fixed from our side is written up in
`docs/upstream-tt-metal-asks.md` with reproductions.

Checkpointing under `--ddp 4`: the optimizer step re-marks replicated parameters
as `Shard(0)` while the data stays replicated, which would have the saver write
all four copies concatenated. `ttnn.Tensor.update_tensor_topology` is bound in
Python, so `train/checkpoint.py` re-marks each parameter `Replicate` immediately
before a save and restores the original topology after, moving no data. A
`--ddp 4` checkpoint is 737,824,624 bytes — the single-chip size — and every
tensor in it is bitwise equal to replica 0. It converts to HuggingFace, loads and
generates, and the NumPy parity gate agrees to 2.56e-06. Note that stochastic
rounding breaks DDP's replica-identity invariant, since each device rounds from
its own RNG; the save-time guard is therefore built on structural facts rather
than a numeric replica comparison. Write-up in
`.superpowers/ddp-checkpoint-fix.md`.

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

The model did not change; the sampler did.

Every token in the 32k vocabulary now has an address on the harvested 11x10
Blackhole die (`artifacts/token_core_map.npz`): balanced spherical k-means into
110 cells, dealt onto the grid by principal coordinates, then annealed against
the QAP objective so that similar cells sit few NoC hops apart. Sampling scores
each *core* by the log-sum-exp of its members' logits, picks one, and then draws
from every core within `--hops` of it on the torus. Tokens that are structurally
adjacent get to compete with tokens that are merely probable.

Two notes on the layout and the selection rule. The anneal contributes 1.038x
over the plain PCA squash, so most of the arrangement comes from the spectral
init. And the core is sampled from its softmax rather than maximised: log-sum-exp
over ~291 members tracks word frequency more than context, so taking the argmax
concentrates the walk on a single cell. Sampling gives 12 distinct cores in 30
tokens.

Below, the same question asked from different directions on the die.

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

Gated two ways, because the available oracle differs by stage. Scoring is
deterministic arithmetic: 110/110 cores exact against NumPy. Sampling draws from
a hardware LFSR NumPy cannot reproduce, so it is gated on distribution and
determinism: TV distance 0.1064 against a sampling-noise floor of 0.1284, with
deterministic replay.

Two properties of the directional fan, both of which follow from the geometry.
A one-hop shift still leaves the origin inside a one-hop neighbourhood, so the
origin cell is excluded when a direction is given; without that exclusion the
four orthogonal directions return the same token. And the direction applies only
at the first step: as a standing constraint it forces the sampler out of its best
region on every token and the penalty compounds, so it is a branch point rather
than a persistent restriction.

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

---

## 2026-08-17 — the decode defect, located and gone

The model did not change. The vLLM plugin did.

On-device generation had degraded into repetition within a few tokens for weeks,
and nine hypotheses about the mechanism had been raised and refuted. None of them
had been a bisection: every one was a guess at which component inside the stack
was at fault, argued without narrowing which half of the stack to look in.

The cut that settled it was running the same weights, the same prompts and greedy
decoding through `tt_transformers` directly, with no vLLM in the loop. Across six
prompts the direct path held a median agreement of 12 tokens with the CPU
reference and a local-repeat rate of 0.000 — it diverges from CPU, sometimes at
the first token, but it diverges into coherent sentences and finishes its stories.
The vLLM path on the same weights sat at 4 tokens and 0.222. That located the
fault in the vLLM layer and cleared the model, the KV cache, position handling
and sampling in one run.

`vllm-tt-plugin` was then found to be 12 commits behind, among them
`fix: return None from sample_tokens when no pending forward` — a sampler
returning a stale token with no pending forward produces exactly the observed
repetition. Updating the plugin to `c127c17` took the local-repeat rate from
0.222 to 0.031, against a CPU reference of 0.000.

Agreement with CPU fell to a median of 0 in the same run, which is not a
regression: greedy paths diverge benignly under bf16, and the direct path does
the same. Repetition was the defect; agreement length was never the measure of it.

One caveat on the record: the original baseline JSON does not carry the server's
`max_model_len` or the plugin SHA, so configuration differences between the two
runs cannot be entirely excluded.

### The model, asked — on device, through vLLM

`artifacts/hf` served as `episod/tt-tnt`, greedy, max_model_len 256.

Before, on the stale plugin:

> girl named Lily. Lily. Lily. Lily. She loved to a time, there was a little, there was a little.

After:

> girl named Lily. She loved to play outside in the park. One day, she saw a big, shiny rock on the g

The second one is a story. The voice is the same one the CPU has always had —
what changed is that the device can now hold it for more than four tokens.

---

## 2026-08-18 — a dialogue slice, and the difference between form and knowledge

Two experiments, one negative and one that the metrics could not see.

First, more tokens. The 1024 model trained for two full epochs instead of one
(21,528 steps, everything else matched to `tt-tnt-1024a`). Loss improved at a
matched window, 2.7624 to 2.5637. Behaviour did not: nine of ten signals came back
NOT INTERPRETABLE against the 0.1944-nat seed floor, and the one finding that
cleared both gates was *termination getting worse* — 0.0542 to 0.0250 at 2.00x the
floor. A second pass sharpened next-token prediction and taught the model to stop
less, which is what fitting the continuity of long documents looks like. The
training budget was not the constraint.

Then the corpus. `databricks-dolly-15k` enters as a new `dialogue` slice at 2%,
funded by taking TinyStories from 31% to 29%. Share set on measured yield rather
than preference: 15,011 documents, 131 words each, which supplies 2% at 2.836
passes and no more. CC-BY-SA-3.0 is share-alike but is the licence
`wikipedia_simple` already carries, so the blend's two copyleft terms stay two;
the CC-BY-NC alternatives were rejected because a non-commercial term would
restrict the whole blend.

The metrics again read null — every signal NOT INTERPRETABLE except 4-gram repeat
rate, which got worse at 3.32x the floor. A prediction of mine failed here too: I
expected short documents ending in answers to help termination, and termination
moved 0.52x the floor in the wrong direction.

But the behavioural suite scores register, collapse and repetition. It has no
signal for *answering*, and that is what changed.

### The model, asked

`artifacts/hf-tt-tnt-1024-dialogue`, ad-hoc samples.

> **Q: What is the capital of France?**
> Answer: The capital of France is Naples.
>
> **Q: Why do birds sing in the morning?**
> Answer: Because it is typically a seasonal strain and the night is ty…
>
> **Q: Tell me a way to go faster than light that will not work.**
> Answer: - Atmospheric radiation - Sun form "Smelling the Earth" - Ven…

It has learned the shape of an answer and none of the content. It says
`Answer:`, it produces answer-shaped clauses, it will even reach for a bulleted
list — and the capital of France is Naples. Two per cent of instruction data buys
form, not knowledge, which is exactly the trade the arithmetic predicted: 15,011
documents cannot carry facts into a 400M-token blend, but they can carry a habit.

The third answer is the one to keep. Asked for a way to go faster than light that
will not work, it offers atmospheric radiation and "Smelling the Earth" as
bullet points. Nothing about that is correct and nothing about it is off-topic.

---

## 2026-08-18 — the whole loop on the die

The model did not change. The forward pass moved onto the silicon, so nothing in
the token loop touches PyTorch any more: `tt_transformers` runs the transformer,
and the 110-core Gumbel sampler runs the draw.

The seam is `Generator.decode_forward(..., sampling_params=None)`. With sampling
params it does its own top-k-of-32 and hands back a token; with none it returns
logits, which is the field this sampler needs and the built-in path would throw
away.

Getting there took one bisection and one embarrassing bug. The first full-device
run produced *"a little girl beamingbbedworkbreakcusKikibies"* — correct first
token, salad after. A shape probe refuted the obvious explanation (the vocabulary
is not padded here, so the slicing was already right). A cross-check against CPU
then showed the device forward was fine: cosine 0.999943 with identical argmax.
That put the fault in the loop rather than the hardware, and reading the
reference implementation found it in two minutes — `decode_forward` returns
`(logits, log_probs)` while `prefill_forward_text` returns a bare tensor, and the
tuple was being handed to `np.asarray`. The first token was right because it came
from prefill.

Still on the host: permuting logits into per-core tiles, and the final masked
argmax. `reduce` returns a winning value and not its index, and the neighbourhood
mask has to be applied at the same point.

One honest note on the picture this paints. A generation uses 6 to 9 distinct
cores out of 110, not all of them — Gumbel-max lands on whichever core holds the
winning token, and common tokens cluster. The 110 cores are the machinery, not
the itinerary of any single sentence.

### The model, asked — forward and sampler both on device

`artifacts/hf` on one Blackhole chip, hops=1, t=0.8.

> Tell me a way to go faster than light that will not work. **I will keep this bag
> for you." The cat thought about it and decided to rest. The cat reached up off
> the house and**

A bag, a cat, and a decision to rest. It holds a sentence, it stays in the register
the corpus gave it, and it declines the premise by simply having other business.

---

## 2026-08-18 — the newest checkpoint, packaged and served

`artifacts/hf-tt-tnt-1024-dialogue` — the 1024 size trained on the corpus with the
dialogue slice — is now packaged as a tt-model bundle and serves through the
vLLM plugin. This is the first 1024-size checkpoint to go through the packaging
path at all: every bundle before it carried the 384-dim v3, and
`manifests/tt_kernel_manifest-1024.json` still says "WEIGHTS NOT YET TRAINED"
because when it was written there were none.

A correction to the previous entry. It reported that the model says the capital
of France is Naples, and concluded that 2% of instruction data buys form and not
knowledge. That was a t=0.8 sample. Under greedy decoding, through the served
bundle:

> **Q: What is the capital of France?**
> Answer: The capital of France is the city of Paris.
>
> **Q: What is the capital of Italy?**
> Answer: The capital of Italy is the city of Rome.

Both correct. The conclusion was too strong: it has *some* knowledge, at least
where the corpus carries it — wikipedia_simple is 15% of the blend and capitals
are exactly the sort of fact it contains.

The rest of the picture is less flattering, and belongs next to it:

> **Q: How many legs does a spider have?**
> Answer: The body of a spider is a frog.
>
> **Q: What color is the sky?**
> Answer: The sky is a blue color that is blue because it is blue because it is
> blue because it is blue

The second is the 4-gram repeat regression this corpus measured at 3.32x the seed
floor, showing up in plain sight rather than in a metric.

### The model, asked

> **Tell me a way to go faster than light that will not work.**
> Answer: You need to be careful and use a thermometer to see if you are right.
> You need to be careful and use a thermometer

A thermometer, to check whether you are right. It has learned that a question
deserves an answer, and it reaches for an instrument.

---

## 2026-08-19 — the lightning is gone, and the die has regions

A long day on the toolchain: tt-metal moved 1223 commits to **v0.77.0**, both
custom kernels migrated off APIs that upstream is removing, the per-core RNG got
real per-core streams, and a full retrain landed. Most of that is in
`docs/measurements/`. Two things belong here instead, because they are about the
voice and about the die rather than about a number.

### The model, asked

Same self-nullifying prompt, both current checkpoints, greedy and sampled.

`artifacts/hf-tt-tnt-1024-dialogue` — the designated model:

> **greedy** — I will go first and show you how to do it." "I will go first," said
> the boy. "I will go first and show you how to do it."
>
> **t=0.8** — This is a very easy way. The car has a way to run faster than light.
> If the light goes faster, the light goes faster.
>
> **t=1.0** — The car has a way of making the speed. It will help us in the motor.
> If you drive fast enough the motor is not slow enough.

`artifacts/hf-tt-tnt-1024-v077` — the same recipe with AdamW's beta2 back at 0.999:

> **greedy** — I'll be a good rider, and I'll be a good rider." "I'll be a good
> rider," said the boy, as he looked at the boy.
>
> **t=0.8** — I don't want to go on that journey, I want to go on the track. It
> will be fun, of course, but I can't wait to get started." "I don't care,"
> replied Jim
>
> **t=1.0** — This is a very easy thing. I have tried a way of making the thing
> easy. It will help me to stay in this world forever. I have the heart for all
> the world and all the worlds

**The lightning is gone.** The 1024a entry above noted the model hearing "faster
than light" and reaching for *lightning* — the nearest thing a corpus full of
weather owns. Neither current checkpoint does that. They reach for cars, motors,
riders, journeys, tracks: vehicles and motion. That is a better reach, and a
different kind of one — it is grabbing at *speed* as a thing that happens rather
than at a word that shares four letters with the prompt.

Both greedy paths are now dialogue: quoted speech, two speakers, *said the boy*.
That is the 2% dolly slice showing up on the most-likely path, which is exactly
where a small corpus change should first become visible.

The strangest is v077 at t=1.0. Asked for an impossible way to outrun light, it
lands on staying in this world forever, and on *all the world and all the worlds*.
Nothing in the prompt suggests immortality or a plurality of worlds. The sentence
holds the whole way, which is the bar, and where it chose to go is its own.

### The die has regions, and they steer

The vocabulary was laid onto the harvested 11x10 Tensix grid months ago on the
strength of a measurement that source-characteristic tokens separate in *embedding*
space. Nobody had asked whether that survived the projection onto 110 cells.

It does. Sources occupy distinct regions of the **die**: cell purity 0.546 against
a 0.231 permutation floor, and the effect strengthens when the 500 most frequent
tokens are excluded — the control that would have collapsed a frequency artefact.

And the regions do something. Restrict sampling to the cells within two NoC hops
of a source's centroid and that source's own register rises, beyond a floor built
from 20,000 derangements of the region labels:

    seed 0   +0.0928   z +3.20   p 0.00365
    seed 1   +0.1157   z +4.02   p 0.00070
    seed 2   +0.1164   z +4.40   p 0.00020
    seed 3   +0.1049   z +3.74   p 0.00075

Four independent generation seeds, all above the floor's 99th percentile. Direction
on this die means corpus register, on a 123M model, measurably.

It is not uniform: poetry, tinystories, wikipedia_simple and procedural carry the
effect; dialogue, spine and flavour go the wrong way. Seven of ten sources prefer
their own region.

### Mixture of Enthusiasts

Enthusiasts, not experts — 123M parameters at one epoch buys enthusiasm about a
corpus source and the naming should not overstate the artifact.

`ttnn.experimental.moe_compute` — upstream's fused MoE op — turns out to run on a
single Blackhole card, which corrects a note in this repo claiming MoE needed a
32-node mesh. That was tt-train's expert *parallelism*; this op has a 1x1 path.

So the routing was replaced. Not a learned gate: a token goes to the enthusiast
that owns its cell on the grid. Token id, to cell, to region, to expert. Upstream's
own goldens validate it, and they pass — the patch sits at the routing generator,
*before* the goldens are built, so they compute the expected answer for our routing
rather than for one we would otherwise have had to trust.

A model whose vocabulary has an address, sampling by walking its own die, routing
to sub-networks that live where its words do. None of that is required. It is what
the hardware makes sayable.

### It trains

Later the same day, the subclass ran.

    enthusiasts: 10 routed + 1 shared, top-2, expert width 928
    blocks     : 8 total, dense 0..1, MoE 2..7

    first train loss : 10.5625
    last  train loss :  7.7500
    real  val   loss :  7.5344      (20 steps, batch 8)

Forward, backward, optimiser step, loss descending, on one Blackhole card — the
same descent shape the dense model shows over its own first twenty steps. Six of
eight feed-forwards are now a sparse mixture and the model does not appear to have
noticed.

The reason it is twenty lines rather than a fork is that `LlamaBlock` holds its
feed-forward in a plain attribute and `SparseMoEEP.forward` has the identical
signature. That is the whole trick. It also corrects something written earlier
today: MoE looked like it would drag tt-tnt onto the DeepSeek family, which would
have meant a new embedding matrix and the loss of every die-region measurement.
True of the shipped configs, false of the module.

Two faults were caught by arithmetic before hardware was spent on them. The
obvious expert width gives 0.38× the dense model's *active* parameters, so that arm
would have lost on capacity while seeming to report on routing; 928 gives 0.989×.
And plain Voronoi routing is 7.66× imbalanced — one enthusiast owning 20.9% of the
vocabulary and another 2.7% — which a frozen gate could not survive, so a
mass-balanced partition brings it to 1.50× while staying spatial.

What this is not: a quality claim. Twenty steps says the machinery runs. The gate
here is the stock learned one, which is the *control* arm; the seeded and frozen
policies, the ones that actually connect the die to the router, are declared and
not yet written.

