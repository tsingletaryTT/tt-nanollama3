<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# tt-tnt

<img src="docs/brand/tt-tnt-logo-small.jpg" alt="tt-tnt: a hand-drawn figure with TT-TNT written on its chest, standing on a grid of Tensix cores." width="260" align="right">

**A 123M-parameter Llama-3-style model that is *of* Tenstorrent hardware, not merely trained on
it.** Trained from random initialization on Blackhole with `ttml` (tt-train), served through the
Tenstorrent vLLM plugin, packaged with
[tt-model-manager](https://github.com/tenstorrent/tt-model-manager), and small enough that a
full epoch takes about an hour — which is what makes it useful as an instrument.

It is not trying to be a capable model. It is trying to be a *complete* one: every stage of the
Tenstorrent path exercised end to end, at a scale where an experiment costs an hour instead of a
week.

## What tt-tnt is best at

**Routing by physical die address.** Tokens can be assigned to experts by *where they live on the
harvested 11×10 Tensix grid* rather than by a learned gate — and it costs **0.0118 nats** against
a gate free to learn (|t| 5.1, 14/15 signs). Source-characteristic tokens occupy measurably
distinct die regions (cell purity 0.546 against a 0.231 permutation floor), and steering
generation to a region shifts that region's register across four seeds (p < 0.004 each). This is
the thing no other model can do, because it requires a real harvested grid to be *about*.

**Being a fast, honest exercise of the whole stack.** One epoch in ~65 minutes on one p300c.
Training through `ttml`, sampling per-core on the device, serving through the vLLM plugin,
packaging as a `tt-model` bundle, exporting to a host-portable HF directory. Because the loop is
cheap, it gets run — and running it finds things: **seven defects in the surrounding stack** so
far, five upstream and two our own, each with a regression check
([`scripts/stack_probe.py`](scripts/stack_probe.py)).

**Sparse routing that measurably pays.** A Mixture of Enthusiasts beats the dense baseline from
scratch, **replicated at two seeds** (pooled +0.0417 nats, |t| 8.0, 39/44 signs), with the
separation emerging late in training in both runs.

**Thinking on demand.** Asked to plan before it speaks, it emits well-formed five-slot
think-blocks in **98%** of generations where the control arm emits none.

## Feature support

| capability | state | evidence |
|---|---|---|
| Training on Blackhole (`ttml`/tt-train) | ✅ | one epoch, ~65 min, one p300c |
| Multi-chip DDP | ✅ `[1,2]` · ⚠️ `[1,4]` | `[1,4]` hard-froze the host; see the upstream note |
| Sparse MoE — *Mixture of Enthusiasts* | ✅ | replicated win, two seeds |
| Die-region expert routing | ✅ | 0.0118 nats to freeze routing to physical address |
| Thinking — five-slot think-blocks | ✅ format · ⏳ effect | 98% adherence; steers but does not yet govern |
| Per-core Gumbel sampling on device | ✅ | custom kernels, per-core RNG streams |
| vLLM serving via the TT plugin | ✅ | `tt-model serve`, OpenAI-compatible |
| `tt-model` bundle packaging | ✅ v4 · ⏳ v5 self-contained | v5 needs wheels assembled |
| Host-portable HF export | ✅ | `scripts/chat.py` runs it CPU-only |
| Tool calling / reasoning parsers | ➖ | plumbing exists; base model declares none |
| Chat template / instruction tuning | ➖ | base completion model by design |
| Skits — multi-turn improv | ⏳ next | design note in the spec |

## The science

Three findings, in descending order of how much they are *ours*.

### Routing is nearly free, and the die has regions

The die-region work is the part of this project that could not exist on other silicon. The
harvested Tensix grid is a real, measured object: source-characteristic tokens cluster into
distinct regions, and a gate *frozen* to that geometry — never allowed to learn — performs within
0.0118 nats of one that learns freely. Seeding a learnable gate from the die map, by contrast,
buys nothing measurable (+0.0044, signs 8+/7−) even though the seeding provably works as a
classifier (61.2% region recovery against a 10% floor). The geometry is real; the loss cares
where the gate may *end up*, not where it starts.

### Sparse routing pays, and the win is the ordinary one

From scratch, one epoch, paired on data order, MoE beats dense at both seeds tried:

| seed | mean Δ | \|t\| | signs | final-5 dense | final-5 MoE |
|---|---|---|---|---|---|
| 5489 | +0.0481 | 7.5 | 20/22 | 2.8748 | **2.8098** |
| 8191 | +0.0354 | 4.5 | 19/22 | 2.8213 | **2.7617** |
| pooled | **+0.0417** | **8.0** | 39/44 | | |

Both runs are indifferent early and separate late, which is the more convincing half — a seed
artefact would be unlikely to reproduce that shape twice. Read it as the ordinary MoE bargain:
this configuration carries **3.62× total parameters at 0.989× active compute**, so more
parameters for the same compute helped. It is not evidence about the die geography above.
([`docs/measurements/moe-from-scratch.json`](docs/measurements/moe-from-scratch.json))

### Thinking: the format is learned, the plan is not yet obeyed

Fine-tuned to emit `offer / accept / add / stakes / handback` before continuing a story, one slot
per improv failure mode, the model produces well-formed blocks **98%** of the time against a
**0%** control. Substituting another story's block changes **100%** of continuations, so the
block steers rather than decorates.

What it does not yet do is *govern*. It planned `add: dance` and wrote a scary dog. The next win
is already located: a single continuation gives `handback` nothing to hand back *to*, so the slot
encoding "make your partner look good" cannot pay off. **Skits** — two or more turns with a
partner who answers — turn every slot from a description into a falsifiable prediction about a
later turn, which is the thing that can actually be scored.
One scorer was rebuilt and the null survived it. `groundedness` was originally a boolean
co-occurrence test, saturated on a corpus where 80.1% of prefix words are hubs — mean 0.998,
99.25% of scores exactly 1.0 — so it *could not* have found an effect. Rebuilt on normalised PMI
it now has **zero ties** (89 up, 111 down across 200 pairs) and reports **no effect**, |t| 0.23.
Same headline as before, much stronger evidence behind it: four scorers that could find an
effect and didn't, rather than three plus a dead one.
([`docs/measurements/improv-stage1.json`](docs/measurements/improv-stage1.json))

## The art

The numbers are in `docs/measurements/`. What the model *sounds like* is in
[`episod-log.md`](episod-log.md), which is where this project keeps its close reading — the run
where the model heard "faster than light" and reached for *lightning*, the run where it stopped
doing that and reached for cars and motors and tracks instead, and the day it learned to write a
plan and then cheerfully ignored it.

That log is the reason the die-region result exists. "The poetry region" started as a joke about
a heat map and turned into a measurement.

## Working with tt-model-manager

[tt-model-manager](https://github.com/tenstorrent/tt-model-manager) (the `tt-model` CLI,
previously `tt-kernel-package-manager`) is how this model is packaged, published and served. The
relationship is deliberately adversarial in one direction: **tt-tnt is a hard consumer, and
reports what breaks.**

```bash
tt-model doctor episod/tt-tnt-1024     # is the surrounding toolchain adequate?
tt-model pull  episod/tt-tnt-1024     # kernels, runner, weights
tt-model serve episod/tt-tnt-1024     # through the TT vLLM plugin
```

Being a real consumer has found and fixed defects the manager's own tests could not: a version
resolver trusting frozen editable metadata over the source tree, a toolchain probe describing the
wrong interpreter, an unreachable instance report, a bundle pinning a temp-directory
`TT_METAL_HOME`, and a `--restore-card` that published the wrong model's card. Findings are
recorded in [`docs/tt-kernel-conformance.md`](docs/tt-kernel-conformance.md) and the repeatable
subset is automated in [`scripts/stack_probe.py`](scripts/stack_probe.py), which re-checks every
one of them in a single command.

## Architecture

![tt-tnt-1024 network architecture](docs/diagrams/model-architecture.svg)

The network: 122,962,944 parameters, 8 blocks, d_model 1024, 512 context. Two annotations
carry engineering decisions rather than description — the attention mask is passed as `null`
so the SDPA kernel picks its Causal mode rather than Arbitrary (worth 1.41x), and the FFN
width of 2816 is 88 tiles = 8 x 11, chosen to tile a grid that is 11 cores wide.

![the stack, bottom to top](docs/diagrams/stack-ownership.svg)

The stack. The amber band is everything this repository adds. The dashed band is
`tt-inference-server`, which exists on the development machine but is **not** in this
project's path: serving goes through vLLM and the Tenstorrent plugin directly, and no
manifest here references it.

![which stack layer implements which part](docs/diagrams/layer-to-stack-map.svg)

The overlay: every model component and its owner at each stage. The same model appears
twice — once as something `ttml` trains, once as something `tt_transformers` serves — which
is why a parity gate between the two exists at all. Amber rows are implemented here, and
four of them exist because the stock path did not fit a harvested die.

Diagrams are generated by a script rather than drawn, so they can be regenerated when the
model changes.


### A Mixture of Enthusiasts, and where each piece comes from

![where the MoE pieces come from](docs/diagrams/moe-provenance.svg)

Enthusiasts rather than experts: 123M parameters at one epoch buys enthusiasm about a corpus
source, and the naming should not overstate the artifact.

Three things the diagram exists to say. **Training MoE and inference MoE are different
codebases** — `ttml`'s `SparseMoEEP` is trainable and lives in tt-train; `ttnn.experimental.moe_compute`
is a fused inference op — and they share a name and nothing else. **The seam is a subclass, not an
architecture change**: `LlamaBlock.mlp` is a slot whose signature is `forward(Tensor) -> Tensor`,
and `SparseMoEEP.forward` has exactly that signature, so tt-tnt keeps GQA, RoPE, its tokenizer and
its die map while only the FFN changes. **The routing is ours** — a token goes to the enthusiast
that owns its cell on the harvested grid, which is measured rather than asserted: cell purity
0.546 against a 0.231 permutation floor, and steering to a region raises that region's register
across four generation seeds at p < 0.004 each.

Upstream ships MoE configs for a single card and for a 6U Galaxy. Nothing for four chips. This
repository already vendors a `[1, 4]` mesh-graph descriptor because ttml ships defaults for 8 and
32 only, and a mismatch there does not error — it hangs in the first gradient all-reduce.

## The model

The **designated current model is `tt-tnt-1024-dialogue`** — the 1024 size: 1024 wide, 8 blocks,
16 heads over 4 KV groups, a 2816-wide SwiGLU, a 512-token window, **122,962,944 parameters**,
one full epoch. It is the checkpoint every result on this page was measured on, and it is
published as [`episod/tt-tnt-1024`](https://huggingface.co/episod/tt-tnt-1024).

The designation, its reason, its evidence and — importantly — what it does *not* mean are
recorded in [`docs/current_model.json`](docs/current_model.json), which is data rather than a
claim of quality in the abstract. It supersedes `tt-tnt-1024a` as of 2026-08-18, chosen for a
capability (it answers questions in the shape of an answer, and sometimes in fact) at a known
cost. Note that [`episod/tt-tnt`](https://huggingface.co/episod/tt-tnt) on the Hub is a
*different, older* model — the 384-wide, 6-layer, 2048-context lineage.

The table below is `tt-tnt-v1`, the **384** size, kept for shape comparison. Both sizes are
declared in [`train/sizes.py`](train/sizes.py) and one YAML each under
[`train/configs/model/`](train/configs/model/).

| Property | Value |
|---|---|
| Architecture | Llama-3 style — RoPE (θ=500000), RMSNorm, SwiGLU, grouped-query attention |
| Embedding dim | 384 |
| Blocks | 6 |
| Heads / KV groups | 6 / 3 |
| Sequence length | 512 |
| Vocabulary | 32,000 (byte-level BPE, trained here) |
| Corpus | Nine-source blend, **as it stood before 2026-08-14** — 399,594,747 tokens emitted per the provenance manifest; 392,773,300 tokens when the finished file is tokenized as training data (353,495,970 train / 39,277,330 validation). That revision carried no document separators, which has since been fixed and the blend rebuilt; see [`docs/corpus_blend.md`](docs/corpus_blend.md) for the current figures, for both of these, and for why they differ |
| Hardware | Tenstorrent Blackhole — trained on **one** p300c (`mesh_shape [1, 1]`, no DDP/TP) |

This repository owns its architectures. They live in
[`train/configs/model/`](train/configs/model/), one YAML per size, described and validated by
the registry in [`train/sizes.py`](train/sizes.py). `train/run.py --size <name>` selects one.

The 384 config is a verbatim copy of tt-train's own `nanollama3.yaml`, vendored rather than
read out of `$TT_METAL_HOME` so the architecture cannot change under a tt-metal upgrade
without a signal — `tests/test_sizes.py` compares the two whenever tt-metal is present, and
holds the registry and the YAML to describing the same model.

Measured on the `tt-tnt-v1` run: 10,787 steps — one epoch over the blend's train split — at
batch 64, sequence length 512, finishing at train loss **3.3125** with a validation loss of
4.2203. The validation curve
(`artifacts/checkpoints-tt-tnt-v1/val_losses.jsonl`) falls from 6.7125 at step 500 to about
4.29 by step 10,000, then plateaus — oscillating 4.29–4.38 for the last ~2,300 steps, including
a rise at step 9000 — rather than continuing to improve. ~58 minutes wall clock on one p300c;
eleven checkpoints at steps 1000–10787.

One chip is the default. `train/config.py` sets `device_config` to
`mesh_shape: [1, 1]` with `enable_ddp` and `enable_tp` both false, so an unqualified
`train/run.py` opens a single device, and every checkpoint this project has published was
trained that way — including all of the ones measured on this page. `--ddp 4` opens four
instead; the next paragraph is what that took. The host this was
developed on is a **TT-QuietBox 2** — four Blackhole chips on two dual-chip p300 cards, wired
as a `P300_X2` 2×2 ring mesh, not four independent boards — and three of those four chips sit
idle during a run. During the v2 run the working chip drew **82 W at 73 °C** against
61–73 W / 63–68 °C on the idle three; idle Blackhole holds its clock at 1350 MHz, so the
power gap is a clearer signal than temperature.

Multi-chip data parallelism was future work when this model was trained; it is not any more.
`train/run.py --ddp 4` opens a `[1, 4]` mesh and runs **3.98x faster** at this shape. The
gradients are verifiably reduced rather than merely fast: with stochastic rounding off, after
DDP steps all four replicas are bit-identical (max `|replica0 − replica_i|` = 0.0 over 66
tensors), and the same instrument was shown to be capable of catching the failure — skip the
parallelism-context init deliberately and the replicas drift by 2.44e-3. A four-chip run's
validation loss also tracks a single-chip run at the same seed to within 0.048, against this
project's 0.1944-nat seed-noise floor. As of `dce5b43` a `--ddp N` run writes a
checkpoint of 737,824,624 bytes, matching the single-chip size. The optimizer step marks
replicated parameters `Shard(0)` while the data stays replicated;
`ttnn.Tensor.update_tensor_topology` is bound in Python, so the marking is correctable by any
holder of the tensor before a save. See
[`docs/multi-chip-notes.md`](docs/multi-chip-notes.md) for the details. (The blow-by-blow lives in this repo's session reports under
`.superpowers/`, which is gitignored and therefore absent from a clone — their durable
conclusions are in the tracked docs instead.) The v2 run described on this page predates all of it and did use a single
chip.

One caveat that travels with a `--ddp N` checkpoint. `stochastic_rounding: true` — which
this project's training config sets, and needs, because bf16 parameters at 1.0 otherwise round
every update away — breaks DDP's replica-identity invariant. Each device draws its own rounding
decisions, so the replicas perform independent random walks about a common trajectory: with it
off, 0 of 66 parameters' replicas differ; with it on, 66 of 66 do. The replicas remain four
coherent models rather than four broken ones, but the file holds **replica 0's** weights, not
"the" weights. Filed as ask 4 in
[`docs/upstream-tt-metal-asks.md`](docs/upstream-tt-metal-asks.md).

Validation finishing *above* training loss (4.2203 vs. 3.3125) is the ordinary generalization
gap for a full epoch, unlike the original 0.43-epoch checkpoint, where the two were a
near-noise-dominated tie. The plateau in the curve above — not the train/val gap — is the more
informative signal for this run: it says the model had largely stopped learning from this
corpus well before step 10,787, not that it needs a slightly longer run to keep closing the gap.

## Getting started

Requires a tt-metal source checkout with `ttml` built, and `TT_METAL_HOME` set. See
[Build TT-Metalium from Source](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/build-tt-metal/).

```bash
pip install -e .

# Build the nine-source corpus blend (CPU only; downloads several GB, needs ~45 GB free —
# scripts/check_disk_space.py refuses to start if the volume is too full)
python scripts/fetch_corpus.py       # pinned revisions, one file per source
python scripts/prepare_corpus.py     # normalise, strip Gutenberg packaging
python scripts/measure_corpus.py     # availability gate; writes docs/measurements/
python scripts/blend_corpus.py       # -> artifacts/corpus/blend.txt + its manifest

# Train the tokenizer on the blend (CPU only)
python scripts/build_tokenizer.py

# Tokenize the corpus into training arrays (CPU only)
python train/tokenization.py

# Confirm the training config resolves without touching hardware
python train/run.py --dry-run --steps 20

# Train on Blackhole
python train/run.py --steps 20 --batch-size 64
```

`measure_corpus.py` counts tokens with the trained tokenizer when one exists and falls back
to a word approximation when none does, so the first pass through this sequence on a fresh
clone bootstraps from the approximation. Re-running `measure_corpus.py` and `blend_corpus.py`
once a tokenizer exists settles the numbers against the real vocabulary — see
[`docs/corpus_blend.md`](docs/corpus_blend.md), which records what the shipped blend actually
contains and why that ordering is cut where it is.

TinyStories only, for a quick smoke test — a ~512 MB single-source corpus rather than the
blend, downloading ~2.2 GB. It is named `corpus.txt`, never `blend.txt`: `build_tokenizer.py`
refuses to write the blend's name from this path, because a TinyStories file called
`blend.txt` is indistinguishable from the real blend to every later step.

```bash
python scripts/build_tokenizer.py --corpus artifacts/corpus/corpus.txt --corpus-mb 512
python train/tokenization.py --corpus artifacts/corpus/corpus.txt
```

`--corpus-mb` applies only to this path. On a corpus that already exists it is a no-op:
the file is trained on as-is, since a head-truncating byte cap would amputate a blend
written one source at a time rather than sample it.

Run the tests with `python -m pytest`. They need no hardware.

## Packaging and serving

The published checkpoint is packaged as a **tt-model v4 bundle** — a manifest under
[`manifests/`](manifests/) plus [`bundle/tt_tnt_adapter.py`](bundle/tt_tnt_adapter.py), the
class the Tenstorrent vLLM plugin imports — and served through that plugin:

```bash
cd <the vLLM plugin's examples/ directory>          # the launch command is cwd-dependent
tt-model serve episod/tt-tnt --force --instance metal-src-vllm
```

[`docs/serving-with-tt-kernel.md`](docs/serving-with-tt-kernel.md) is the guide: what the
bundle is, the two runtime patches the adapter carries (a `find_grid` fix for a harvested
Blackhole grid, and a converted-weight-cache fingerprint) and what would delete them, the exact
serve procedure, how to verify you are serving the weights you think you are — and the traps.
The traps are the valuable part, and all of them were found by hitting them: the cwd
dependence above, why `TT_VISIBLE_DEVICES` must **not** be exported, why the newest ttnn wheel
cannot serve on this box, and three separate ways a stale cache or a stale flag will hand you a
confident wrong answer rather than an error.

On-device generation through vLLM produces coherent free-running text as of plugin
`c127c17`, at a local-repeat rate of 0.031 against a CPU reference of 0.000
(`docs/measurements/decode-defect-resolved.json`). Earlier plugin builds degraded into
repetition within a few tokens; the fault was in the vLLM layer, not the model. Serving with a
plugin older than `c127c17` will reproduce it.

[`docs/tt-kernel-conformance.md`](docs/tt-kernel-conformance.md) is the companion findings
report — what the v4 manifest schema can express, which fields are actually read, and what
authoring a real bundle against it turned up.

## Evaluating a checkpoint

`scripts/evaluate.py` is the single entry point. It does not measure anything itself — it
runs the existing instruments (`score_behaviour.py`, `probe_context_use.py`,
`eval_per_source.py`) as one benchmark and **joins their numbers correctly**, which is the
step every significant error in this project came from. All three modes are CPU-only.

```bash
# 1. Evaluate one model. Defaults to the model designated in docs/current_model.json.
python scripts/evaluate.py --model artifacts/hf-tt-tnt-1024a

# 2. Compare two, with the seed-only noise floor applied automatically.
python scripts/evaluate.py --model artifacts/hf-tt-tnt-1024a \
    --against artifacts/hf-tt-tnt-384s512

# 3. Try an arbitrary prompt. Scratch output; never a measurement.
python scripts/evaluate.py --try "The lighthouse keeper wrote in the log:"
```

Three things it will not let you do:

- **Compare losses measured at different windows.** `evaluate()` windows at `cfg.seq_len`,
  so window size rides along with the model unless something stops it. Mode 2 refuses, names
  both windows, and exits non-zero — the eval window is a fixed constant, never the model's
  own `max_position_embeddings`.
- **Read a delta without its noise floor.** Every delta is printed beside its ratio to the
  **seed-only control** (`tt-tnt-v3` vs `tt-tnt-v5`, derived at runtime, never hardcoded).
  Anything within ~1.2× of that floor is labelled `NOT INTERPRETABLE` regardless of its
  confidence interval — the rule that would have caught a published −0.041 register finding
  sitting at 1.03× the floor, which a later seed-only control refuted. For loss trajectories
  it prefers the **sign test** (capacity: negative at 22/22 checkpoints; the seed floor
  changes sign at 8/22) and reports the endpoint and the trajectory average separately,
  saying which is the headline.
- **Slip an ad-hoc sample into the measurement namespace.** `--try` writes to `scratch/`,
  outside `docs/` and outside git, banner-marked, and refuses any other destination. A
  prompt that proves genuinely diagnostic gets promoted into a **new** frozen set with new
  ids in a deliberate commit — never by editing an existing set, whose digest every
  committed measurement depends on.

The current model is designated in [`docs/current_model.json`](docs/current_model.json),
with its reason, its evidence, and — since `tt-tnt-1024a` is trained at a 512 context while
`tt-tnt-v3` is at 2048 — the qualification that "best" is not unqualified.

## External benchmarks

Everything above is self-referential. Validation loss is a tail of our own blend, the
per-source losses slice that same blend, the behaviour scores use two prompt sets we wrote,
and the noise floor comes from our own seed-only control. A model that had learned to imitate
this corpus and nothing else would score exactly as well on all of it.

`scripts/benchmark_external.py` runs the model against benchmarks **someone else built, on
data we did not choose**, through EleutherAI's `lm-evaluation-harness`, and writes one report
per model to `docs/measurements/external-<label>.{md,json}`.

`lm-eval` is **not** a dependency of this repository and must not become one — `pyproject.toml`
lists three runtime dependencies and this project has repeatedly declined to add a fourth. The
harness lives in a throwaway virtualenv under `scratch/` (gitignored) and the script shells out
to it, recording the venv path and the exact `lm_eval`/`torch`/`transformers`/`datasets`
versions in every report.

```bash
python3 -m venv scratch/lm-eval-venv
scratch/lm-eval-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
scratch/lm-eval-venv/bin/pip install lm-eval==0.4.9 'transformers<5'

# Benchmark the designated current model (CPU only; takes a couple of hours).
python scripts/benchmark_external.py --model artifacts/hf-tt-tnt-1024a
```

Three things it will not let you do:

- **Read a chance-level score as a number.** Every task carries an explicit chance baseline
  (0.25 for 4-way multiple choice, 0.50 for 2-way, ~0 for open-vocabulary LAMBADA). A score
  within 2 standard errors of it is labelled `AT CHANCE`, its `reportable_score` in the JSON
  is `null`, and the prose names the task rather than quoting the figure. This is the same
  move `scripts/evaluate.py` makes with its `NOT INTERPRETABLE` floor — different null, same
  rule. **At chance was the predicted outcome for a model this size, and the prediction was
  wrong** — see the results below. The rule stayed in anyway: it is what makes the rows that
  did clear it worth reading.
- **Quietly report a truncated task.** `tt-tnt-1024a`'s window is 512 tokens. Every request
  the harness actually issued is re-tokenized and checked against it, and any task with a
  request over the limit is flagged as not a fair score instead of being averaged in.
- **Touch a Tenstorrent device.** CPU only; `--device` refuses anything else.

The GPT-2-small column is measured, not quoted, when a reference run is available: benchmark
`gpt2` with `--reference-run` and pass the resulting JSON as `--reference-json`. Published
figures from the GPT-2 paper are used only as a fallback and are labelled as such, because the
paper's detokenizers and scoring code are not lm-eval's.

### GPT-2 through the same harness

That option is the methodological centre of this section, not a convenience. Quoting GPT-2's
published figures would have compared our scores, under lm-eval's task definitions and
detokenizers, against numbers produced by different code — and it would have hidden the most
useful thing the reference run found.

- **It validated the whole setup with one number.** Our measured GPT-2 WikiText word perplexity
  is **37.3695** against the paper's published **37.50** — **0.3% apart**. The harness version,
  the detokenizer, and the claim that lm-eval's `wikitext` task really is WikiText-103 test
  perplexity (verified at run time: the `wikitext-2-raw-v1` and `wikitext-103-raw-v1` test
  splits are byte-identical, 1,288,493 characters) are all confirmed by that agreement.
- **It showed where quoting would have misled.** The same cross-check disagrees by **29.2%** on
  LAMBADA accuracy — 0.3256 measured against 0.4599 published — because the paper filters
  stopwords on generation while lm-eval scores the exact continuation by loglikelihood. A
  published figure and a measured one are not the same kind of object, and the reports print
  both with the gap rather than asking to be believed.
- **It showed that three of the tasks are dead instruments at this scale, for either model.**
  GPT-2 small is itself `AT CHANCE` on WinoGrande (0.5162, +1.2 s.e.) and `BELOW CHANCE` on
  ARC-Challenge (0.1903) and MMLU (0.2292). Read against quoted figures, our three matching
  rows would have looked like our failures. They are the benchmarks' floors. A later model that
  moves WinoGrande or MMLU has done something GPT-2 small could not.

The reference run is committed at
[`docs/measurements/external-gpt2-small.md`](docs/measurements/external-gpt2-small.md) and is
the JSON any future `--reference-json` should use.

### What the benchmarks said

`tt-tnt-1024a`, 512-token window, CPU, fp32, eight tasks, 3,465 s. Full report with all 15
metric rows, the per-task truncation audit and the caveats:
[`docs/measurements/external-tt-tnt-1024a.md`](docs/measurements/external-tt-tnt-1024a.md).

| task | metric | score | chance | s.e. from chance | verdict | GPT-2 small (measured) |
|---|---|---:|---:|---:|---|---:|
| wikitext | word perplexity | 222.6627 | — | — | no chance baseline | 37.3695 |
| lambada_openai | last-word accuracy | 0.0980 | ~0 | +23.7 | ABOVE CHANCE | 0.3256 |
| arc_easy | accuracy | 0.3106 | 0.25 | +6.4 | ABOVE CHANCE | 0.4381 |
| piqa | accuracy | 0.5484 | 0.50 | +4.2 | ABOVE CHANCE | 0.6289 |
| hellaswag | accuracy | 0.2643 | 0.25 | +3.2 | ABOVE CHANCE | 0.2892 |
| winogrande | accuracy | 0.4996 | 0.50 | −0.0 | **AT CHANCE** | 0.5162 (also at chance) |
| arc_challenge | accuracy | 0.1783 | 0.25 | −6.4 | BELOW CHANCE | 0.1903 (also below) |
| mmlu | accuracy | 0.2295 | 0.25 | −5.8 | BELOW CHANCE | 0.2292 (also below) |

**The scores indicate the model has learned English, not just our
corpus.** That is the one thing no instrument in this repository could establish, because every
other one is scored against our own corpus. ARC-Easy at +6.4 standard errors and PIQA at
+4.2, on data we did not choose and scored by code we did not write, against a null we do not
control, are the evidence.

The size of the gap is one number: WikiText-103 perplexity **222.66** against GPT-2 small's
37.37, at **1.2% fewer parameters** (122,962,944 against 124,439,808) and **113x less
training data** (352,714,752 tokens seen — 10,764 steps × batch 64 × seq_len 512 — against
WebText's ~40 billion). Roughly 6x the perplexity for roughly 1/113th of the tokens is an
ordinary place to sit on the scaling curve. Reading 222 as "broken" is reading the parameter
row instead of the token row.

Four honest deflations belong with the above, and the reports carry all four:

- **MMLU is not a fair score on `tt-tnt-1024a` and is flagged as such.** 524 of its 56,168
  requests exceed the 512-token window (longest 1,112), and lm-eval drops tokens off the *front*
  of the context, so the model answered 524 questions it was shown only part of. `tt-tnt-v3` at
  a 2048-token window truncates nothing on any task, which is what makes the flag a measurement
  rather than a guess. Nothing else came close: the longest request in the whole run outside
  MMLU and WikiText is 238 tokens, in PIQA. WikiText is kept separate and deliberately not
  called truncation — it is a rolling-loglikelihood task whose 62 documents (up to 16,372
  tokens) are scored in consecutive 512-token windows, so nothing is dropped; the real caveat
  there is that every token sees at most 511 tokens of context against GPT-2's 1,023.
- **Eleven rows carry a chance baseline, so ~0.5 of them would clear a 2-s.e. gate by luck even
  if every one were null.** HellaSwag at +3.2 s.e. is the weakest of the four and the one to
  distrust first. ARC-Easy at +6.4, and four rows moving the same way, is not what 0.5 expected
  false positives looks like.
- **`tt-tnt-v3` is not far behind on any of this**, at 5.6x fewer parameters: WikiText 247.32
  against 222.66, LAMBADA 0.0798 against 0.0980. It also *clears* WinoGrande (+2.3 s.e.) where
  1024a does not — a single-row flip that a multiple-comparison correction would eat, and not to
  be read as v3 resolving pronouns better. It is still an awkward result for a designation that
  rests entirely on matched-window validation loss over our own blend.
- **LAMBADA's long-range signal and the context probe have not been reconciled.** LAMBADA needs
  the last word of a passage and clears chance by 23.7 standard errors, while
  `scripts/probe_context_use.py` finds per-token loss flattening out — around position ~128 on
  `tt-tnt-v4` at a 2048 window, and around ~64 on the older separator-less `tt-tnt-v1`. The two
  have never been run on the same checkpoint: there is no context probe for `tt-tnt-1024a`. So
  this is an open question rather than a contradiction, and it is recorded as one.

## Embedding geography

A proposed sampler would lay the 32,000-token vocabulary onto Blackhole's Tensix grid and sample
by spatial neighbourhood, so that *direction on the grid* means *corpus register*. That only
works if tokens characteristic of different sources already occupy distinguishable regions of
the embedding space — otherwise any layout is imposed, and the claim is decoration.

`scripts/probe_embedding_geography.py` decides which, before a kernel is written. It reads
`model.embed_tokens.weight` out of `model.safetensors` and nothing else — no forward pass, no
device, CPU only. Characteristic tokens are picked by log-odds ratio with an informative
Dirichlet prior, z-scored (Monroe et al. 2008), 150 per source, winner-take-all so the label
sets are disjoint. Report:
[`docs/measurements/embedding-geography-tt-tnt-1024a.md`](docs/measurements/embedding-geography-tt-tnt-1024a.md).

In the `content` condition — the same measurement after excluding the
500 globally most frequent tokens, so a frequency artefact would collapse rather than survive —
k-NN purity over cosine neighbours of 1,350 labelled tokens is **0.5458** against a
label-permutation floor of 0.1103 ± 0.0031 and a chance rate of 0.1111. That is 138.8 of the
floor's own standard deviations. A multinomial logistic probe recovers which of nine sources a
token belongs to, from its embedding alone, **77.98% ± 2.31%** of the time, against **0.1007**
with the labels permuted and **0.2099** from a frequency-only control that carries no
directional information at all. The probe is linear on purpose: a grid layout needs source
identity to be *linearly* legible, and a stronger classifier would answer an easier question.

It **strengthens** when the commonest tokens are removed — purity 0.4984 → 0.5458, probe 0.7373
→ 0.7798 — which is the opposite of what a frequency artefact does.

| source (`content`, nine-way, chance 0.111) | k-NN purity | probe recall |
|---|---:|---:|
| procedural | 0.732 | 0.867 |
| flavour | 0.666 | 0.756 |
| poetry | 0.613 | 0.853 |
| wikipedia_simple | 0.589 | 0.831 |
| spine | 0.549 | 0.818 |
| tinystories | 0.544 | 0.867 |
| weird | 0.431 | 0.751 |
| folklore | 0.409 | 0.644 |
| gutenberg_children | 0.377 | 0.631 |

Every source is far above chance, but the ordering splits cleanly: `procedural`, `flavour`,
`poetry` and `wikipedia_simple` occupy well-separated regions, while `folklore`, `weird` and
`gutenberg_children` share one — they are each other's largest off-diagonal neighbours (0.14 /
0.17 for folklore↔weird). Asking that region for three independent directions would return
three draws from the same neighbourhood.

Two deflations belong with the number, and both are in the report:

- **A "direction" means subject matter *and* register, not provenance.** `procedural` separates
  best because its characteristic tokens are food and kitchen words, and food words would
  cluster in any embedding of any corpus. That the clusters *align with* corpus sources is the
  finding; that the model represents "which file this came from" is not claimed and was not
  measured.
- **The geography is not two-dimensional.** The first two principal components of the labelled
  tokens explain **2.7%** and **1.8%** of their variance in the `content` condition (3.3% / 2.0%
  in `all`). The report prints a PCA scatter, labelled *for looking at only*; nothing rests on
  it. Cosine silhouette is 0.0209 against a permuted floor of −0.0070 ± 0.0004 — positive and
  far outside its floor, but small, which is what silhouette does in 1024 dimensions.

It also says nothing about generated text. This is a measurement of the embedding table, not of
behaviour; `scripts/score_behaviour.py` is where register in actual completions is measured.

## Lineage: from nanollama3 to tt-tnt

This project was originally named **tt-nanollama3**, and this section says plainly what
changed and what didn't, rather than quietly dropping the earlier name.

What it started as. The model began as a hand-rolled nanollama3-like model: a Llama-3
architecture, trained from random initialization with tt-train's `ttml` trainer, on a single
downloaded corpus (TinyStories). That is still true today in the parts that matter most for
correctness — the 384 config vendored at
[`train/configs/model/tt-tnt-384.yaml`](train/configs/model/tt-tnt-384.yaml) is a
verbatim copy of tt-train's own `nanollama3.yaml`, and every checkpoint this project has ever
produced is trained against that same Llama-3 architecture (RoPE, RMSNorm, SwiGLU, grouped-query
attention) through `ttml`. Nothing about the rename touched the model's shape or its trainer.

What changed with the new name. Two things this project now owns that it didn't at the start:
a **nine-source, licence-audited corpus** blended to a 400M-token budget (see
[`docs/corpus_blend.md`](docs/corpus_blend.md) and [`docs/corpus_licensing.md`](docs/corpus_licensing.md)),
in place of a single downloaded TinyStories dump; and a **32,000-token BPE tokenizer trained
on that blend**, rather than inherited from someone else's vocabulary. Those two changes are
what stopped "nanollama3-like" from being an accurate description of this project's identity,
even though the underlying architecture and trainer it sits on did not change.

Existing artifacts still carry the old name, on purpose. Checkpoints from before this
rename are named `nanollama3_step*.pkl` and are left untouched — they are evidence of runs
made under the old name, and renaming them would misrepresent when they were produced. New
checkpoints are written as `tt_tnt_step*.pkl`; `train/checkpoint.latest_checkpoint` reads
both naming schemes so an existing checkpoint directory keeps resolving correctly.

## History

This model comes out of the **"Build an LLM from Scratch"** lesson arc in
[tt-vscode-toolkit](https://github.com/tenstorrent/tt-vscode-toolkit), which builds a
Llama-3-style model TT-native from the first line of code rather than porting one afterwards.
The lessons are readable without installing anything:

- [Pick Your Altitude](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-00-intro/) — the arc's introduction
- [Tokenizer & Data](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-01-tokenizer/)
- [Embeddings](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-02-embeddings/)
- [Attention](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-03-attention/)
- [The Transformer Block & the Model](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-04-block-and-model/)
- [Train It & Run for Real](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/lfs-05-train-and-run/)

Browse the whole microsite at
[docs.tenstorrent.com/tt-vscode-toolkit](https://docs.tenstorrent.com/tt-vscode-toolkit/).

This repository takes that arc past where the lessons stop: it owns its training entrypoint,
its corpus and tokenizer pipeline, and the packaging work that turns a checkpoint into
something servable.

Why it has its own training entrypoint. tt-train's stock Python trainer does not work
against current tt-metal. `examples/python/transformers/training.py` imports a `trainer`
module that is not on its path, calls `train()` with a `val_ids` argument the signature does
not accept, and depends on a `TrainingConfig` that never defines the `seq_len` `train()`
requires. Its data loader also hardcodes `shakespeare.txt`, ignoring any configured path. We
reuse everything ttml genuinely provides — `create_optimizer`, `train()`, `checkpointing`,
and both of its Llama implementations — and replace only what is broken or hardcoded.

Which Llama, and why it matters for speed. ttml ships two: a C++ `CppLlama` (reached
through its `TransformerModelFactory`) and a pure-Python `ttml.models.llama.Llama`. They are
the same architecture over the same fused ops and cost the same per step — but only the
Python one can be handed a null attention mask, and a null mask is what puts the fused SDPA
kernel on its causal path instead of its arbitrary-mask path, roughly halving the attention
work. `train/model.py` wraps the Python model so its parameter names, checkpoints, and HF
conversion are byte-for-byte what they always were, and `train/run.py --model-impl` selects
between the two. Measured: **1.41x** faster per training step at `--size 384`, **1.15x** at
`--size 1024`, with the loss trajectory unchanged. `train/model.py`'s module docstring carries
the reasoning and the numbers; [`docs/upstream-tt-metal-asks.md`](docs/upstream-tt-metal-asks.md)
carries the two-line tt-metal fix that would make this workaround unnecessary.

## Provenance and licensing

This project's source code is Apache-2.0, matching tt-metal and tt-vscode-toolkit. Every
source file carries an SPDX header. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Apache-2.0 covers *our code*. It does not override the terms of what this project consumes,
and two of those inputs deserve stating plainly rather than being folded into a blanket claim:

Training corpus — a nine-source blend, two sources share-alike. The corpus (see
[`docs/corpus_blend.md`](docs/corpus_blend.md) and
[`docs/corpus_licensing.md`](docs/corpus_licensing.md), the latter generated from
`train/corpus.py`) mixes TinyStories, Simple English Wikipedia, a small instruction-dialogue
slice, and seven curated Project Gutenberg slices. **Three** sources carry share-alike data
licenses: `tinystories`
([`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories), 29% of the
blend) under the Community Data License Agreement – Sharing, v1.0; `wikipedia_simple`
([`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), 15% of the
blend) under CC-BY-SA-3.0; and `dialogue`
([`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
2% of the blend) also under CC-BY-SA-3.0. This repository **does not redistribute the corpus**; the fetch
scripts download each source from the Hugging Face Hub at a pinned revision. Whether model
weights trained on share-alike data constitute a "Data Derivative" (CDLA-Sharing-1.0) or an
"Adaptation" (CC-BY-SA-3.0) is not settled, and we do not assert that they don't. Anyone
publishing weights trained with this code should reach their own conclusion rather than
inheriting ours.

Architectural inspiration — Mini-LLM. The lesson arc credits
[Mini-LLM by Ashx098](https://github.com/Ashx098/Mini-LLM) for its component choices — RoPE,
RMSNorm, SwiGLU, GQA, subword BPE. That repository **declares no license**, so no rights are
granted by it. This is a credit, not a license inheritance: the components themselves come
from published papers, and this implementation derives from tt-train's `nanollama3` model
config and the `ttml` library, not from Mini-LLM's source. No code was copied from it.

Model weights. Published, and public, at
[`episod/tt-tnt`](https://huggingface.co/episod/tt-tnt) via `scripts/publish_to_hub.py`, which
creates the repo private by default and has no code path of its own that flips it public — the
repo's visibility was changed separately, as an explicitly-authorized action (2026-08-14), and
is expected to stay public. [`docs/model-card.md`](docs/model-card.md) is the source of truth
for the card pushed there; it states the corpus and its license explicitly and describes the
model honestly as a demonstration rather than a capable general model, per the standard set
above.

Runtime dependencies — tt-metal / `ttml` / `ttnn` (Apache-2.0), `transformers` and
`tokenizers` (Apache-2.0), numpy (BSD-3-Clause).

## Contributing

This repository follows the plan-then-execute workflow in
[`docs/superpowers/`](docs/superpowers/): a spec, then implementation plans whose steps are
executed and reviewed task by task. The plans record what was verified against source rather
than assumed — including several defects found in the plans themselves.
