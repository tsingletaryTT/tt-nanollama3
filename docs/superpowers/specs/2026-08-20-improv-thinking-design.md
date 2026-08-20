<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Improv thinking for tt-tnt — design

**Status:** approved in conversation 2026-08-20. Stage 1 is specified to build; stage 2 is
outlined, not specified.

## Goal

Give tt-tnt a *think-block* that steers **how** it thinks, **what** it thinks about, and
**why**, in service of one thing: being a storytelling improv partner that a human co-author
wants to keep playing with.

The target is not correctness. It is a playful, willing participant that inspires its
partner. Three named failure modes, all drawn from real improv experience, define the
opposite:

1. **Worst place possible** — every offer escalates toward harm.
2. **Most boring next step** — a block wearing the costume of a reply.
3. **So far out nobody can follow** — escalation untethered from what exists.

## Non-goals

- **Not general reasoning.** 123M parameters, 8 layers, one epoch, and a corpus with zero
  reasoning-shaped data. Chain-of-thought for arithmetic or logic is out of scope and would
  fail.
- **Not a chat model.** No instruction tuning, no chat template. It completes prose.
- **Not context extension.** See "Measured constraints" — 512 tokens is already sufficient,
  and an earlier version of this design wrongly treated it as the binding constraint.
- **No craft prose in the corpus.** `train/corpus.py` stays at ten sources.

## Measured constraints (not assumed)

Measured 2026-08-20 with the model's own tokenizer against `artifacts/corpus/tinystories.txt`:

| quantity | value |
|---|---|
| story length, median / p99 / max | **199 / 288 / 303** tokens |
| think-block, prose-paraphrase form | 65 tokens |
| think-block, extractive form (this design) | **40** tokens typical, **54** worst case (12-token offer, two adds) |
| whole story + think + 80-token reply within 512 | **fits 100%** of stories |

Two corrections behind that table, recorded because both nearly shaped the design wrongly:

- The document separator in the corpus files is **`</s>`**, not a blank line. `\n\n`
  separates paragraphs *within* a story. Splitting on `\n\n` reports a median of 40 tokens
  and makes stories look four times shorter than they are.
- The 512-token window was called "the binding constraint" before it was measured. It is
  not. Nothing in this design extends context, and `rope_theta` (already 500000) is
  irrelevant here.

Model shape, from `artifacts/hf-tt-tnt-1024-dialogue/config.json`: 8 layers, dim 1024, 16
heads, 4 KV heads, vocab 32000, `max_position_embeddings` 512.

## Architecture

Staged deliberately: stage 1 proves the mechanism on the simpler task, stage 2 adds the
interaction shape only once the schema is known to work.

- **Stage 1 (this spec): continuation.** Model is given a story prefix, thinks once, and
  continues. Simpler to train and to score, and closest to what the model already does.
- **Stage 2 (outlined): turn-taking co-author.** Human writes a beat, model thinks, replies
  with one move, hands back. Same schema, different framing, and the `handback` slot becomes
  directly load-bearing.

### The schema

Five slots. Each is a steering point — the schema *is* the mechanism, not decoration:

```
<think>
offer: the lantern went out
accept: the lantern is dark
add: moths
stakes: up
handback: her hands
</think>
```

| slot | improv function | kills which failure |
|---|---|---|
| `offer` | name what you were handed | — |
| `accept` | the "yes": restate it as true | the boring block |
| `add` | the "and": exactly one new thing | far-out drift |
| `stakes` | one notch, not ten | worst-place escalation |
| `handback` | what you leave your partner | flatness; this is "make your partner look good" |

**Slots are filled with spans lifted from the text, never paraphrases.** We have no
validated generator to paraphrase with, and putting an unvalidated model inside the data
pipeline would make every downstream number unattributable. Extraction is deterministic and
auditable.

## Component 1 — trace derivation (`scripts/derive_traces.py`, `train/improv.py`)

Read `artifacts/corpus/tinystories.txt`, split on `</s>`. Sentence-split with
`re.split(r'(?<=[.!?"])\s+', text)` — the corpus is simple prose and does not need a parser.

For each story, choose cut point `k` uniformly from `[2, n-2]` so the prefix holds at least
two sentences and the continuation at least one. Require prefix length in `[24, 320]` tokens.
`continuation` is the next one or two sentences.

Slot rules, in order:

- **`offer`** — the prefix's final sentence, truncated to its first 12 content tokens.
- **`accept`** — the longest run of non-stopword tokens present in *both* the final prefix
  sentence and the continuation. Fall back to the most frequent prefix content word that
  reappears in the continuation. **If neither exists, DROP the example**: a continuation
  that carries nothing forward is a block, and blocks must not be training exemplars.
- **`add`** — content words in the continuation absent from the prefix, ranked by IDF over
  the blend, take the top one or two.
- **`stakes`** — sign of the intensity delta: `up` / `level` / `down` (intensity defined in
  Component 2).
- **`handback`** — the final content noun of the continuation *if* it was introduced there;
  else the last newly-introduced entity; else the literal `open`.

**The drop rules make the training set exemplary rather than representative, and that is a
deliberate choice**: we are teaching good moves, not average ones. The stated risk is that
filtering shifts the distribution away from the corpus the model already knows. Mitigation:
log the drop rate per rule and report it with the results; a drop rate above ~50% means the
filter, not the model, is choosing the behaviour.

Output: JSONL with `{story_id, cut_k, prefix, think, continuation, slots, drops}`.

## Component 2 — the four scorers (`scripts/score_improv.py`)

**Every scorer is a delta from prefix to continuation.** This is the design's central move:
the failure modes are *relative*. A grim folktale continuing grim is not "going to the worst
place"; a gentle story turning grim is. Absolute darkness scoring would flag half the
corpus, because `folklore` is legitimately dark.

| scorer | definition | failure signal |
|---|---|---|
| `novelty` | count of content words in continuation absent from prefix | **0** — the boring block |
| `escalation` | `intensity(cont) − intensity(prefix)`, where intensity = harm-lexicon hits per 100 content words; plus boolean `new_harm` | large positive |
| `groundedness` | fraction of new content words co-occurring with ≥1 prefix content word in a sentence-window co-occurrence table built once from the blend; plus `new_proper_nouns` (capitalised, non-sentence-initial, absent from prefix) | low fraction / many fresh proper nouns |
| `affordance` | 1 if the final sentence contains a continuation-introduced element or ends in `?`, else 0 | **0** — nothing handed back |

The harm lexicon is a **curated, committed data file with provenance** (~80 terms), not a
scrape. Committed at `train/data/harm_lexicon.txt`.

**Scorer validation is part of the deliverable, not an afterthought.** Each scorer must be
shown to discriminate before any result that uses it is believed:

- `escalation` must score known-grim folklore passages high in *absolute* intensity **and**
  near zero in *delta* for folklore→folklore continuations. That second condition is the one
  that proves the relative framing works.
- `novelty` must return 0 for a verbatim restatement and >0 for any real continuation.
- `groundedness` must rank a continuation about the established characters above one that
  introduces three unrelated proper nouns.
- `affordance` must separate a closed ending ("and they went to bed.") from an open one.

A scorer that has never been seen to discriminate is a claim, not a check.

## Component 3 — training (two paired arms)

Warm-start from `artifacts/checkpoints-v077-beta2-control/tt_tnt_step00010764.pkl` — the
checkpoint with verified provenance (tokens-v4, seed 5489, beta2 0.999,
`stochastic_rounding: true`). We want the storytelling retained, not relearned.

| setting | value |
|---|---|
| arms | **think** (`prefix → <think>…</think> → continuation`) and **no-think** control (`prefix → continuation`), identical cut points |
| seed | 5489, both arms — paired, so data order cancels in the delta |
| steps | 3000, `--val-every 250` |
| ddp | **2, one board** — the 4-chip mesh hard-froze this host on 2026-08-20 |
| config | `train/configs/tt-tnt-v077.yaml` — required; without it `stochastic_rounding` is off and RMSNorm gammas do not learn |

**Open question that may require work in `train/run.py`:** loss should be masked to
`think + continuation`, leaving the prefix unsupervised. Whether ttml's `train()` supports a
loss mask is **not yet established**. If it does not, the fallback is loss over the full
sequence — acceptable but less clean, since it re-teaches prefix text the model already
knows. Resolve this before writing the training code; it is the single largest unknown in
this spec.

## Component 4 — evaluation (`scripts/eval_improv.py`)

200 held-out story openings, never trained on. Generate from both arms at greedy and at
T=0.8 × 3 samples.

Report **two numbers separately**, because at 123M they will come apart:

1. **Schema adherence** — fraction of generations whose think-block parses into all five
   slots. Measures whether the format was learned.
2. **Failure-mode rates** — the four scorers, think arm vs no-think control.

Statistics follow the discipline `scripts/compare_runs.py` already enforces: paired, report
the within-run floor, and correct for multiplicity. Five tests (four scorers + adherence)
gives a two-sided Bonferroni threshold of **α = 0.01**. Any effect inside the floor is
reported NOT INTERPRETABLE rather than narrated as a trend.

### The swap test (the load-bearing control)

Take 50 openings; regenerate each continuation with **another story's think-block**
substituted. Measure token-level divergence (position of first difference) and score
divergence.

**If continuations barely move, the thinking is decorative and stage 1 has failed** — the
model learned to emit a plan and ignore it. This is the control that distinguishes thinking
from performing thinking, and it exists because of a result from the same day: die-region
gate seeding worked as a classifier (61.2% vs 10% chance) and bought **nothing** measurable
in loss. Skill at producing an artefact does not mean the artefact is used.

## Success criteria (stage 1)

- Schema adherence **≥ 80%** well-formed think-blocks.
- **At least 2 of 4** failure-mode rates improve against the no-think control at α = 0.01,
  paired.
- Swap test shows continuations **do** change materially.

Failing the swap test kills the premise regardless of the other two, and that is the correct
outcome to report rather than work around.

## Craft material — design input, not training data

Mined for the slot design and the rubric; no text enters the corpus. All public domain:

- Aristotle, *Poetics* — reversal (*peripeteia*) and recognition; the source of `stakes` as
  a turn rather than an increment.
- Freytag, *Technique of the Drama* (1863) — rising action; why `stakes: up` is one notch.
- Archer, *Play-Making: A Manual of Craftsmanship* (1912) — the obligatory scene; the debt a
  story owes its own setup, which is what `handback` tracks.
- Polti, *The Thirty-Six Dramatic Situations* (1916 trans.) — a ready-made taxonomy of
  escalation *kinds*, useful for making `escalation` more than a harm count later.
- Quiller-Couch, *On the Art of Writing* (1916) — concrete detail over abstraction, which is
  the antidote to far-out drift.

Improv's own principles — "yes, and", don't block, make your partner look good — are ideas
rather than expression. They inform the schema directly and are cited as rationale; no text
is used.

## Files

| file | responsibility |
|---|---|
| `train/improv.py` | schema definition, slot extraction, trace rendering |
| `train/data/harm_lexicon.txt` | curated harm terms, with provenance header |
| `scripts/derive_traces.py` | corpus → SFT JSONL, with drop-rate reporting |
| `scripts/score_improv.py` | the four scorers |
| `scripts/eval_improv.py` | A/B + swap test + statistics |
| `tests/test_improv.py` | extraction, scorer discrimination, schema parsing |

## Risks

1. **The trace may be ignored.** Derived post-hoc, so the model can learn to emit a vague
   plan and write independently of it. The swap test is the detector, and it runs first among
   the evals.
2. **Loss masking may not be supported by ttml.** Largest unknown; resolve before coding.
3. **Filtering shifts the distribution.** Drop rules select exemplary moves. Drop rate is
   logged per rule and reported.
4. **123M may not fill five slots reliably.** Expected, and why adherence is reported
   separately from quality. A model that learns the *shape* is still constrained toward
   playing well even when slots are thin — that is the mechanism being tested.
5. **The harm lexicon is a blunt instrument.** Mitigated by the relative framing, but a
   folklore-heavy stage 2 should revisit it.
