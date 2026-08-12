<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# A diverse corpus for a deliberately strange NanoLlama3

**Goal:** a training corpus that produces a model which is *slightly mysterious* — dreamlike
and associative in words the way AnimateDiff output is in pictures — capable of perpendicular
suggestions and of *playing* at agency for absurd ends, while staying coherent enough to be
worth reading.

**Status:** design. No implementation. Supersedes nothing; the existing TinyStories-only
pipeline (`train/data.py`) is the thing being generalised.

## Why now

Three independent measurements say data is the binding constraint, not compute and not
numerics:

1. The 21,034-step run **plateaued after step 10,000** — the final 11,000 steps bought about
   0.02 nats. More steps on the same corpus stopped paying.
2. `ct8` found the same at a different scale: ~80M parameters on 100M tokens plateaus at eval
   loss ~1.4, while the Mini-LLM reference needed **361M tokens** for readable prose. This
   project has 114.9M.
3. The near-tie measurement (`docs/measurements/v2-cpu-findings.md`): the better-trained model
   is **flatter**, not sharper — 32% of positions within 0.5 logits against the baseline's
   21%. A sharper model would also be more robust to the numerical error behind the serving
   defect, so corpus work is plausibly a partial remedy for that too.

## What the model is for

Three roles, staged, cheapest to evaluate first:

1. **Provocation engine** — short, dense, strange material to feed the image and video models
   in `tt-local-generator`. Success is pictures you would not have thought of.
2. **Weird interlocutor** — answers perpendicular; associative, occasionally uncannily apt.
3. **Agentic theatre** — *roleplays* agency: narrates plans, invents tools, describes using
   them, reports results.

**Real tool use is out of scope and out of reach.** It does not emerge at 22M parameters, nor
at the registered 1024 size (~133M). A model that convincingly performs agency without having
it is closer to the intended quality than one that actually calls tools.

## The target voice

Held as the acceptance criterion, not as an illustration:

> The chimp chose the longest stick, then the one that had been rained on. Ants came up it in
> a thin line, and each one, being eaten, told the next where the mouth was.

Coherent sentences. An unexpected turn. The turn is the product.

## The dial: grounded weird

A large well-formed backbone with a strong strange minority. Reads mostly coherent, then goes
sideways.

This is a scale decision, not only a taste one. **At ~133M parameters, corpus diversity fights
coherence**: spread the budget across eight unrelated registers and the model learns none well
enough to be interesting, and "weird" reads as "broken". The backbone is what buys the right
to be strange.

Worth stating plainly because it is counter-intuitive and we measured it: the flat next-token
distribution that makes this model fragile in serving is the *same property* that produces
associative text. Near-ties are perpendicular thinking, numerically. Pushing coherence up
sharpens the distribution and sands off some of the strangeness. The mix below is a chosen
point on that trade, not an optimum.

## Composition, by function

Slices earn their place by what they do to the output, not by where they came from.

**Sourcing rule: prefer datasets already on Hugging Face.** One pinned dataset revision with
a deterministic filter beats N bespoke fetches — fewer moving parts, reproducible by revision,
and the fetch layer stays small.

| Function | Target share | HF dataset | Selection |
|---|---|---|---|
| **Narrative backbone** | ~45% | `roneneldan/TinyStories` (have it) + `sedthh/gutenberg_english` | bookshelf *Children's Literature*; Aesop, Andersen, Grimm, Nesbit, Kipling |
| **Grounding** | ~15% | `wikimedia/wikipedia`, config `20231101.simple` | all |
| **Observational-mystical spine** | ~12% | `sedthh/gutenberg_english` | authors: **Fabre**, Maeterlinck, **Charles Fort**, Hodgson |
| **Folklore & myth** | ~8% | `sedthh/gutenberg_english` | bookshelves/subjects: mythology, folklore; *Arabian Nights*, Kalevala, Frazer |
| **Weird fiction** | ~4% | `sedthh/gutenberg_english` | authors: Blackwood, Dunsany, Machen, Browne; bookshelves *Horror*, *Fantasy* |
| **Poetry** | ~1% | `biglam/gutenberg-poetry-corpus` (CC0) | all |
| **Agentic format** | ~13% | vetted IF transcripts (open) + `sedthh/gutenberg_english` | Fabre's experimental passages; PD procedural/how-to texts |
| **Flavour essences** (upsampled, capped) | ~2% | `sedthh/gutenberg_english` | authors: Gertrude Stein; I Ching (Legge 1882) |

Measured while writing this spec: `sedthh/gutenberg_english` holds **48,284 English books
(10.75 GB, roughly 2.5-3B tokens)** with a `METADATA` field carrying `authors`, `title`,
`subjects`, and Gutenberg's curated `bookshelves` — which map almost directly onto the slices
above. `TEXT` already has Project Gutenberg boilerplate stripped.
`biglam/gutenberg-poetry-corpus` is 3.09M lines, CC0.

**Not yet verified:** that each named author is present in that dataset. The HF search index
returns 500 for it, so per-author confirmation is deferred to the first task of the plan
(`measure_corpus.py`), which has to enumerate them anyway. Coverage of ~48k books makes their
presence very likely but it is not established.

### Why these

**Fabre is the spine.** Obsessive field observation of insects — the ant subject matter
directly — and structurally *already* agentic tool-use theatre: "I placed a straw across the
trench; the procession halted; I waited." Plan, act, observe, report, with no licensing
problem. He is counted in two slices deliberately; the same passages do both jobs.

**Charles Fort** compiles anomalous phenomena deadpan, as data, with wild associative leaps
delivered straight. Fabre's method applied to things that should not happen. He is natively
perpendicular in a way dread-based weird fiction is not: he lists, and the wrongness
assembles itself. He replaced Lovecraft, whose copyright status is contested rather than
clear — a distinction this project's licensing discipline should respect.

**Maeterlinck** is mystical about insect collectives; ants as a distributed mind is his
actual thesis, which is squarely on brief.

**Stein** keeps grammar intact while dissolving semantics. Rare and valuable: it teaches that
syntactic correctness and semantic convention are separable, which is exactly the licence a
model needs to make unexpected turns without falling apart.

**The I Ching** supplies a format nothing else does — terse oracular response to a query,
judgment and image. It models the *interaction shape* of the provocation-engine role.

### Two risks, tracked not assumed

**Scarcity.** The strange material may not exist in the volume the shares imply. Fabre's ten
volumes are perhaps 5M tokens; Maeterlinck under 1M; Fort a few million. Reaching a 25%
strange share of a ~400M-token corpus may require broadening selection or upsampling, and
upsampling at 133M parameters risks memorisation rather than style transfer. **The shares
above are targets to be revised against measured availability** — see `measure_corpus.py`
below, which exists to find this out before ratios are committed.

**Stein collides with a live defect.** This model already degenerates into `Lily. Lily. Lily.`
and the serving path amplifies repetition. Training on *a rose is a rose is a rose* teaches
repetition as style. Her share is capped and small, and the evaluation set must include a
probe that distinguishes "learned Stein" from "learned to stutter".

## Pipeline

Generalises the single hardcoded source in `train/data.py` into a registry, mirroring
`train/sizes.py` — a pattern that has already earned its place here.

```
train/corpus.py            # registry: one CorpusSource per source
                           #   name, slice, target_share, upsample
                           #   fetch spec (hf repo+revision | gutenberg id | local)
                           #   license (SPDX string, URL, attribution requirement)

scripts/fetch_corpus.py    # HF download at pinned revision -> artifacts/raw/<source>/
                           #   one dataset + a filter per slice, not N bespoke fetches
scripts/prepare_corpus.py  # normalise       -> artifacts/corpus/<source>.txt
scripts/measure_corpus.py  # per-source token counts   <-- the scarcity gate
scripts/blend_corpus.py    # deterministic mix -> artifacts/corpus/blend.txt
                           #                   + artifacts/corpus/blend_manifest.json
scripts/build_tokenizer.py # retrain 32k BPE on the blend (existing, retargeted)
train/tokenization.py      # blend -> arrays (existing, retargeted)
```

Three load-bearing properties:

**Measure before blending.** `measure_corpus.py` reports available tokens per source against
target share and **fails loudly** when a slice cannot be filled without exceeding a stated
upsample cap. The scarcity risk becomes a gate rather than a surprise discovered after a
training run.

**Blending is deterministic and recorded.** `blend_manifest.json` captures per-source token
counts, upsample factors, achieved shares, source revisions, and a hash of the output, and is
committed. "What was this model trained on" becomes exactly answerable — which is precisely
what is usually unanswerable about a model.

**Licensing is data, not prose.** Each source carries its license in the registry; the model
card's licensing section is *generated* from it. This project has already been bitten twice by
prose drifting from reality (the README and model card both went stale on the hardware claim,
found only by grep). A generated section cannot go stale.

The registry gets the same anti-drift test `sizes.py` has: every source must have a license, a
slice, and a resolvable fetch spec, and every file under `artifacts/corpus/` must belong to a
registered source.

## Tokenizer

**Retrained from scratch on the blend**, staying at 32,000 tokens so `vocab_size` and every
manifest keep their shape.

The current tokenizer saw only TinyStories and would fragment the new domains badly —
`>NORTH`, taxonomic names, archaic orthography — spending model capacity on spelling instead
of meaning. Retraining invalidates existing checkpoints, which is acceptable because this
corpus requires training from scratch regardless. The 384 baseline stays frozen as published
evidence.

## Licensing

Three tiers, kept separable rather than blended:

| Tier | Sources | Obligations |
|---|---|---|
| Public domain (texts) | All Gutenberg material via `sedthh/gutenberg_english` (MIT-licensed *packaging*, PD texts) and `biglam/gutenberg-poetry-corpus` (CC0) | None on the texts; note the packaging licenses |
| Share-alike data | TinyStories (CDLA-Sharing-1.0), Simple English Wikipedia (CC-BY-SA 3.0) | Attribution; derivative status of weights unsettled |
| Per-work vetted | IF transcripts | Allowlist only, each recorded individually |

**We do not redistribute the corpus.** Load-bearing: CC-BY-SA 3.0 and CDLA-Sharing-1.0 are
not obviously compatible as terms on a single redistributed work. Shipping a *recipe* that
reconstructs the blend locally from pinned sources means that question never arises, while
`blend_manifest.json` keeps reproduction exact.

**Gutenberg's texts are public domain; the packaging is separately licensed.** Sourcing via
`sedthh/gutenberg_english` (MIT) and `biglam/gutenberg-poetry-corpus` (CC0) means the
*aggregation* carries its own permissive terms while the underlying pre-1929 texts remain
public domain. Both are recorded in the registry. `prepare_corpus.py` still strips any
residual PG boilerplate, and the project does not use the Project Gutenberg trademark.

**Attribution is generated from the registry**, so it cannot be forgotten.

**The weights question stays open, explicitly.** The existing stance extends unchanged:
whether weights trained on share-alike data constitute a Data Derivative is unsettled, this
project does not assert that they do not, and anyone publishing weights trained with this
recipe should reach their own conclusion rather than inheriting one.

This is a stated position and a conservative design, not legal advice. The design's job is to
make the position easy to state accurately and hard to drift from.

## Evaluation

**A frozen prompt set, judged by a human.** Roughly 20 prompts spanning the three roles.
Every checkpoint generates the same completions with the same seed; outputs are committed
under `docs/measurements/`. Acceptance is a human reading them.

Loss is tracked as a diagnostic and is **never** the acceptance criterion. The v2 result is
the argument: it trained better by every numeric measure and became flatter, not sharper.
Lower loss means better probability assignment, not a more interesting model.

The prompt set must include:

- **The target-voice probe** — the chimp-and-ants prompt above.
- **A stutter probe** — distinguishes "learned Stein" from "learned to repeat", given the
  known repetition defect.
- **An oracular probe** — a question, to see whether the I Ching's response shape took.
- **An agentic probe** — a situation requiring plan → act → observe → report.
- **A grounding probe** — something factual, to check Wikipedia did its job and the model
  knows what an ant actually is.

## Scope boundaries

**In scope:** the corpus registry, fetch/prepare/measure/blend pipeline, retrained tokenizer,
generated licensing, frozen evaluation prompt set.

**Out of scope, deliberately:**

- *Which model size to train.* The corpus targets ~400M tokens, enough to serve the 1024 size;
  choosing what to train is the training plan's decision.
- *The serving decode defect.* Unresolved, and a better model may mask rather than fix it. The
  free-running measurement (`scripts/free_running_check.py`, baseline median 4/40) should be
  taken on the current model **before** the corpus changes, so the two are not confounded.
- *Real tool use.* Not reachable at these scales.
- *IF transcripts, if vetting stalls.* The only slice that can block progress. Everything else
  is fetchable today, and Fabre and Fort already carry the agentic structure implicitly. Ship
  without it and add it as a second corpus release rather than waiting.
