<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Skits: improv thinking, stage 2 — design

**Status:** approved in conversation 2026-08-21. Supersedes the "Stage 2, re-scoped" design note
in [`2026-08-20-improv-thinking-design.md`](2026-08-20-improv-thinking-design.md), which this
document expands.

## Why stage 2 exists, and what stage 1 could not ask

Stage 1 taught the model to emit a five-slot think-block before continuing a story. It worked, in
the narrow sense: **98%** schema adherence against a **0%** control, and substituting another
story's block changed **100%** of continuations. And it moved **none** of the four failure-mode
scorers at α = 0.01 — including, after the scorer was rebuilt on NPMI, a `groundedness` that
provably discriminates (zero ties, 89 up / 111 down) and still found nothing.

The diagnosis is that **the block is context the model conditions on, not an instruction it
obeys** — influence, not governance. It planned `add: dance` and wrote a scary dog. Change the
block and the output moves; ask it to mean something and it shrugs.

Two structural reasons stage 1 could not have found otherwise:

1. **`handback` had no target.** The slot encodes improv's "make your partner look good", and with
   a single continuation there is no partner. It was decorative by construction; no scorer could
   have rescued it.
2. **`stakes` was measured over the wrong interval.** Escalation in improv happens *across* an
   exchange. Measuring intensity delta inside one continuation measures something else.

And one epistemic reason: stage 1 could only ask whether a plan *described* the continuation it
was derived from. It could not distinguish **"bad plan"** from **"good plan, ignored"** — two
failures with entirely different fixes.

Skits fix all three, because a later turn exists to be predicted.

## What a skit is

A story with at least seven sentences yields one skit:

```
prefix    sentences[0:2]     setup, context only
turn 1    sentences[2]       MODEL   — preceded by think-block 1
turn 2    sentences[3]       PARTNER — real corpus text
turn 3    sentences[4]       MODEL   — preceded by think-block 2
turn 4    sentences[5]       PARTNER — real corpus text
turn 5    sentences[6]       MODEL   — preceded by think-block 3
```

Five turns: three model, two partner. Measured on 2,000 stories with the model's own tokenizer:

| quantity | value |
|---|---|
| stories with ≥7 sentences | **99.8%** |
| five-turn skit, total tokens | median **202**, p90 226, p99 257, max 327 |
| fits the 512 window | **100%** |
| fits after tile-alignment to a multiple of 32 | **100%** |

The window is not a constraint here, and nothing in this design compromises for it. (Stage 1
initially assumed it was the binding constraint; it was not, and the assumption was wrong before
it was measured.)

**Five turns rather than three** was chosen deliberately: it yields a stakes *trajectory* across a
scene rather than a single delta, and it gives `handback` two chances to be tested instead of one.
The cost is more room for the repeat-degeneration this model is known to fall into, which is why
§5 measures degeneration explicitly rather than hoping.

### The partner is the corpus, in this build

The partner's turns are the **actual next passages of the story**. Nothing generated enters the
pipeline, so every downstream number stays attributable — the same rule that kept stage 1 honest
and the reason its findings survived three reviews.

A **model-partner arm** (the model plays both sides) is a deliberate follow-up, not part of this
build. It measures a different quantity — influence rather than anticipation — and it puts an
unvalidated generator inside the measurement, so a bad score could be a bad offer or a bad
partner with no way to tell. It becomes worth running once the corpus-partner numbers exist as a
baseline.

## 1. Derivation (`train/skit.py`, `scripts/derive_skits.py`)

Deterministic and extractive, exactly as stage 1: slots hold spans lifted from the text, never
paraphrases, because there is no validated generator to paraphrase with and putting one in the
pipeline would make every downstream number unattributable.

Each model turn gets a think-block derived from the scene so far plus the turn it precedes. The
one change from stage 1: **`offer` is the preceding partner turn**, not the prefix's tail. For
turn 1 there is no partner yet, so `offer` is the prefix's final sentence. That change is what
gives `accept` something real to accept.

Reuses `train.improv`'s `split_sentences`, `content_words`, `extract_slots`, `render_think`, and
`Slots`, and `scripts.score_improv`'s `intensity` and `load_harm_lexicon` (injected as a callable,
so `train/` never imports from `scripts/`).

### The drop rule, and the risk it carries

If **any** of the three model turns fails derivation — nothing carried forward, or nothing added —
**the whole skit is dropped**. A partial skit would silently change what the model sees, and a
think-block that is present for two turns and absent for the third teaches the wrong thing.

The stated risk: stage 1 dropped **6.05%** of single-turn examples on exactly these two rules.
Three turns each needing carry-and-add compounds, and the drop rate could be far higher. The
implementation must report drop rate per rule *and per turn position* before any training, and
warn above 50% — past that the filter rather than the model is choosing the behaviour, and any
result must be reported with that fact attached.

Output: JSONL, one record per skit, carrying the prefix, the five turns with their roles, the
three think-blocks with their slot dicts, the story id, and the sentence indices used.

## 2. Prediction scoring (`scripts/score_skits.py`)

**This is the primary result, and it is the thing stage 1 could not measure.** At evaluation the
model generates both the think-block and the turn that follows it, so every slot becomes a
falsifiable claim about text that did not exist when the claim was made.

| slot | the prediction | HIT when | defined for |
|---|---|---|---|
| `accept` | the offer is carried forward | the named element appears in the model's own turn | all 3 blocks |
| `add` | this element enters the scene | it appears in the model's own turn | all 3 blocks |
| `stakes` | intensity rises / holds / falls | the sign of the delta from the previous turn matches, within `STAKES_EPSILON` | all 3 blocks |
| `handback_anticipation` | the scene will need this | it appears in the **next partner turn** | blocks 1 and 2 only |
| `offer` | — | bookkeeping; never scored | — |

`handback_anticipation` is named for what it measures. **The corpus partner cannot have heard the
model**, so a hit means the model correctly anticipated what the story was about to need — it left
open the thing that turned out to matter. That is a real and scoreable skill, and it is a
*different quantity* from influence. The name carries the limitation so no later reader has to
find it in a footnote. Block 3 has no following partner turn, so its `handback` is **undefined and
not counted** — never scored as a miss.

Reported **per slot**, not only pooled, so the result says *which* slots the model honours.

## 3. Controls

Three, and the new one is load-bearing.

**Shuffled-slot control (new).** Take the model's own generated think-block, substitute another
skit's slot *values*, regenerate the turn, and re-score prediction accuracy. **If accuracy is
unchanged, the model is producing scaffolding rather than plans.** This directly tests the
template-inflation risk that approach A creates by putting three blocks in one sequence: stage 1
showed the model learning template tokens very fast, and its healthy-looking 1.71 loss turned out
to be carried almost entirely by literal scaffolding.

**No-think arm.** The same skits with no think-blocks. Baseline for the secondary failure-mode
scorers, and the control that showed stage 1's adherence was real (it emitted blocks 0% of the
time).

**Swap test.** Carried over unchanged from stage 1, and it **runs first**. If substituting another
skit's block does not change the continuations, the thinking is decorative and the stage has
failed regardless of the other numbers.

## 4. Training (`scripts/train_skits.py`)

**Approach A: one sequence per skit, all model turns supervised.**

```
prefix → think₁ turn₁ → partner₁ → think₂ turn₂ → partner₂ → think₃ turn₃
```

Loss is masked to the **think-blocks and the model's turns**. The prefix and both partner turns
are context and are never supervised — the model must learn to *read* a partner turn, not to
produce one.

Chosen because it matches inference exactly: at generation time the model sees the scene so far,
which is what this trains on. It also teaches that a think-block *follows* a partner turn, which
is the behaviour wanted. The two rejected alternatives are recorded so nobody re-litigates them:
one-example-per-turn trains a different thing than we evaluate and triples token throughput for
the same signal; warm-starting from the stage-1 think checkpoint bakes in a model that learned the
format *without* governance, which is the habit being broken (it is worth a later arm, not the
first build).

Settings, all inherited from stage 1 where they were established the hard way:

| setting | value | why it is not negotiable |
|---|---|---|
| labels | pre-shifted, `labels[t] = input_ids[t+1]` | ttml expects this; the HF convention silently trained two arms against wrong targets |
| boundary | last prompt position supervised, final position masked | its target is the first completion token — the transition being trained |
| tile alignment | every example padded to a multiple of 32 | ttml's SDPA backward mismatches raw-T against tile-padded-T and dies with `TT_FATAL` |
| `stochastic_rounding` | `True`, with a read-back assertion | it defaults off on the SFT path, which froze all 17 RMSNorm gammas and produced a 0%-adherence result that was pure artefact |
| warm start | the one-epoch base checkpoint | keep the storytelling; do not inherit stage 1's format-without-governance |
| mesh | `--ddp 2`, one board | the 4-chip mesh hard-froze this host |
| arms | think and no-think, same seed, same skits, same order | paired, so per-step noise cancels in the delta |

## 5. Evaluation (`scripts/eval_skits.py`)

Held-out skits whose story ids do not appear in the training set, verified by a runtime check
rather than by construction.

**Order matters.** The swap test runs first and can fail the stage alone.

**Primary:** per-slot prediction accuracy against the shuffled-slot control.

**Secondary:** the four failure-mode scorers with their intervals redefined to span turns —
`escalation` across the exchange rather than within a reply, `groundedness` (NPMI) against the
scene so far, `novelty` and `affordance` per model turn.

**Two guards, both because five turns was chosen over three:**

- **Adherence per turn position.** Does turn 3 still emit a well-formed block, or has the model
  collapsed by then? A pooled adherence number would hide exactly that.
- **Degeneration.** Repeat rate across turns. Stage 1 produced `cold cold cold cold …` and opened a
  second think-block mid-sentence; a five-turn scene gives that more room. Reported for both arms,
  because the question is whether *thinking* makes it worse.

**Statistics.** Roughly eleven tests (four slots, pooled accuracy, four failure scorers, adherence,
degeneration), so the two-sided Bonferroni threshold tightens to **α ≈ 0.0045**. Anything inside
the run's own floor is reported NOT INTERPRETABLE rather than narrated as a trend. `paired_verdict`
already carries the zero-scatter guard and per-scorer direction map, both of which were bugs once.

**Success criteria.**

1. Schema adherence **≥ 0.80 at every turn position**, not pooled.
2. Prediction accuracy materially above the shuffled-slot control on **at least 2 of 4** slots.
3. Degeneration **no worse** than the no-think arm.

Criterion 2 is the one that matters. It is the first time this project can ask whether the plan
governs the prose, and a failure there is a real result: it would say the model writes plans it
does not use, which localises the next fix to conditioning rather than to derivation.

## Files

| file | responsibility |
|---|---|
| `train/skit.py` | skit schema, per-turn slot derivation, rendering |
| `scripts/derive_skits.py` | corpus → skit JSONL, with per-rule and per-turn drop reporting |
| `scripts/score_skits.py` | the four prediction scorers |
| `scripts/train_skits.py` | the two paired arms |
| `scripts/eval_skits.py` | swap test, prediction accuracy, shuffled-slot control, secondary scorers, guards |
| `tests/test_skit.py` | derivation and rendering |
| `tests/test_score_skits.py` | prediction scorers, including discrimination against the real corpus |
| `tests/test_eval_skits.py` | statistics, controls, and the guards |

## Risks

1. **The drop rate compounds.** Three turns needing carry-and-add each. Measured and reported per
   rule and per turn before training; above 50% the filter is choosing the behaviour.
2. **Template inflation.** Three blocks per sequence triples the scaffolding tokens, and stage 1
   showed those learned almost instantly. The shuffled-slot control exists for this.
3. **Degeneration across five turns.** Known behaviour, explicitly measured, reported per arm.
4. **`handback_anticipation` is not influence.** Stated in the metric's name, not a footnote.
5. **A test that passes against correct and incorrect code.** This happened four times in stage 1.
   Every new scorer needs a discrimination test against the **real** corpus, not a synthetic
   fixture — the saturated `groundedness` passed a toy discrimination test while being dead on real
   data, and the first replacement test was hollow for the same reason.
