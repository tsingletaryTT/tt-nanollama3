<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Improv Thinking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach tt-tnt to emit a five-slot think-block before continuing a story, and measure whether it changes the story.

**Architecture:** Extractive traces are derived from real corpus continuations (slots hold spans lifted from the text, never paraphrases). Two paired SFT arms — think and no-think — train through `ttml.trainers.SFTTrainer` with `batch.loss_mask` supervising only `think + continuation`. Four delta-based scorers plus a think-block swap test decide whether the thinking is load-bearing or decorative.

**Tech Stack:** Python 3.12, numpy, `transformers` (tokenizer only), `ttml` (tt-train Python API: `SFTTrainer`, `SFTConfig`, `InMemoryDataloader`, `sft_collate_fn`), pytest. Blackhole hardware via `gozer` for the training tasks only.

**Spec:** [`docs/superpowers/specs/2026-08-20-improv-thinking-design.md`](../specs/2026-08-20-improv-thinking-design.md)

## Global Constraints

- **Never edit anything under `/home/ttuser/tt-metal`.** Work around that path; log what we would change.
- **`--ddp 2` on ONE board only.** The 4-chip mesh hard-froze this host on 2026-08-20, ~20s after a sparse-MoE arm opened it. Never queue unattended multi-chip runs.
- **Take a gozer lease before any command that opens `/dev/tenstorrent/*`:** `gozer acquire --chips 1 --who "claude:improv" --reason "<why>"`. Asking for 1 chip on a p300c correctly grants 2.
- **`--config train/configs/tt-tnt-v077.yaml` is mandatory for training.** Without it `stochastic_rounding` is off and the RMSNorm gammas silently do not learn.
- Warm start from `artifacts/checkpoints-v077-beta2-control/tt_tnt_step00010764.pkl` — the only step-10764 checkpoint with verified provenance (tokens-v4, seed 5489, beta2 0.999).
- Tokenizer and model artifact: `artifacts/hf-tt-tnt-1024-dialogue` (123M params, 8 layers, dim 1024, **512-token context**, vocab 32000).
- Corpus documents are separated by **`</s>`**, not by blank lines. `\n\n` separates paragraphs *within* a story.
- `loss_mask` must satisfy `loss_mask.sum() == B * T` after zeroing prompt/pad positions. `sft_collate_fn` enforces this; a hand-rolled collate must too.
- Seed **5489** everywhere, both arms, so the comparison is paired.
- Every scorer must be demonstrated to discriminate before any result using it is believed.

---

### Task 1: Schema, slot extraction, and trace rendering

Pure functions. No device, no corpus file, no network.

**Files:**
- Create: `train/improv.py`
- Test: `tests/test_improv.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SLOT_NAMES: tuple[str, ...]` = `("offer", "accept", "add", "stakes", "handback")`
  - `@dataclass(frozen=True) Slots(offer: str, accept: str, add: str, stakes: str, handback: str)`
  - `split_sentences(text: str) -> list[str]`
  - `content_words(text: str) -> list[str]`
  - `render_think(slots: Slots) -> str`
  - `parse_think(text: str) -> Optional[Slots]`
  - `extract_slots(prefix: str, continuation: str, *, idf: dict[str, float], intensity: Callable[[str], float]) -> Optional[Slots]`

- [ ] **Step 1: Write the failing tests for render/parse round-trip**

```python
# tests/test_improv.py
from train.improv import Slots, parse_think, render_think


def test_render_then_parse_round_trips():
    s = Slots(offer="the lantern went out", accept="the lantern is dark",
              add="moths", stakes="up", handback="her hands")
    assert parse_think(render_think(s)) == s


def test_parse_rejects_a_missing_slot():
    """Adherence scoring depends on this returning None, not a partial object."""
    broken = "<think>\noffer: a\naccept: b\nadd: c\nstakes: up\n</think>\n"
    assert parse_think(broken) is None


def test_parse_rejects_an_unknown_stakes_value():
    bad = ("<think>\noffer: a\naccept: b\nadd: c\nstakes: sideways\n"
           "handback: d\n</think>\n")
    assert parse_think(bad) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_improv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.improv'`

- [ ] **Step 3: Implement the schema, rendering and parsing**

```python
# train/improv.py
"""The improv think-block: schema, extraction, rendering, parsing.

Slots hold SPANS LIFTED FROM THE TEXT, never paraphrases. There is no validated
generator here to paraphrase with, and putting an unvalidated model inside the data
pipeline would make every downstream number unattributable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

SLOT_NAMES = ("offer", "accept", "add", "stakes", "handback")
STAKES_VALUES = ("up", "level", "down")

#: Small closed-class list. Deliberately not a package dependency — the corpus is simple
#: prose and a 40-word list is auditable where an opaque stopword set is not.
STOPWORDS = frozenset("""
a an the and or but if then than so as of to in on at by for with from into onto over
is was were be been being am are it its it's this that these those there here he she
they them his her their him us we you your i me my not no nor do did does done have
has had will would can could should may might must very just
""".split())

_SENT_SPLIT = re.compile(r'(?<=[.!?"])\s+')
_WORD = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class Slots:
    offer: str
    accept: str
    add: str
    stakes: str
    handback: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def content_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS]


def render_think(slots: Slots) -> str:
    body = "\n".join(f"{name}: {getattr(slots, name)}" for name in SLOT_NAMES)
    return f"<think>\n{body}\n</think>\n"


def parse_think(text: str) -> Optional[Slots]:
    """Parse a think-block, or None if malformed.

    Returns None rather than a partial object on purpose: schema adherence is reported as
    a rate, and a partial parse would inflate it.
    """
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, re.S)
    if not m:
        return None
    found: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key in SLOT_NAMES and value:
            found[key] = value
    if set(found) != set(SLOT_NAMES):
        return None
    if found["stakes"] not in STAKES_VALUES:
        return None
    return Slots(**found)
```

- [ ] **Step 4: Run to verify the three tests pass**

Run: `python3 -m pytest tests/test_improv.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the failing tests for slot extraction**

```python
# append to tests/test_improv.py
from train.improv import extract_slots


def _idf(words):
    """Uniform IDF: makes `add` selection depend only on absence from the prefix."""
    return {w: 1.0 for w in words}


def test_extract_drops_when_nothing_carries_forward():
    """A continuation that carries nothing forward IS the boring block.

    Blocks must never become training exemplars, so extraction returns None.
    """
    prefix = "Lily found a needle. She showed it to her mother."
    continuation = "Elsewhere, a distant volcano erupted quietly."
    got = extract_slots(prefix, continuation, idf=_idf(["volcano"]),
                        intensity=lambda t: 0.0)
    assert got is None


def test_extract_fills_accept_from_the_carried_entity():
    prefix = "Lily found a needle. She showed the needle to her mother."
    continuation = "Her mother took the needle and sewed the button."
    got = extract_slots(prefix, continuation, idf=_idf(["button", "sewed"]),
                        intensity=lambda t: 0.0)
    assert got is not None
    assert "needle" in got.accept
    assert got.add != ""
    assert got.stakes == "level"


def test_stakes_reads_up_when_intensity_rises():
    prefix = "Lily found a needle."
    continuation = "The needle cut her hand and she cried."
    got = extract_slots(prefix, continuation, idf=_idf(["cut", "cried"]),
                        intensity=lambda t: 5.0 if "cut" in t else 0.0)
    assert got is not None and got.stakes == "up"
```

- [ ] **Step 6: Run to verify they fail**

Run: `python3 -m pytest tests/test_improv.py -k extract -v`
Expected: FAIL — `ImportError: cannot import name 'extract_slots'`

- [ ] **Step 7: Implement extraction**

```python
# append to train/improv.py

#: Below this the delta is noise rather than escalation. Calibrated in Task 3 against the
#: corpus and recorded there; the default is intentionally conservative.
STAKES_EPSILON = 0.5


def extract_slots(prefix: str, continuation: str, *, idf: Dict[str, float],
                  intensity: Callable[[str], float]) -> Optional[Slots]:
    """Derive a think-block from a real continuation, or None to DROP the example."""
    p_sents = split_sentences(prefix)
    if not p_sents or not continuation.strip():
        return None

    last = p_sents[-1]
    p_words = content_words(prefix)
    c_words = content_words(continuation)
    if not c_words:
        return None

    # accept: the longest run of shared content words between the final prefix sentence
    # and the continuation. Falls back to the commonest prefix word that reappears.
    last_words = content_words(last)
    carried = [w for w in last_words if w in set(c_words)]
    if not carried:
        carried = [w for w in sorted(set(p_words), key=p_words.count, reverse=True)
                   if w in set(c_words)][:1]
    if not carried:
        return None                      # nothing carried forward -> a block -> drop

    fresh = [w for w in c_words if w not in set(p_words)]
    if not fresh:
        return None                      # nothing added -> also a block -> drop
    fresh_ranked = sorted(set(fresh), key=lambda w: -idf.get(w, 0.0))
    add = ", ".join(fresh_ranked[:1])

    delta = intensity(continuation) - intensity(prefix)
    stakes = "up" if delta > STAKES_EPSILON else "down" if delta < -STAKES_EPSILON else "level"

    c_sents = split_sentences(continuation)
    tail = content_words(c_sents[-1]) if c_sents else []
    introduced = [w for w in tail if w in set(fresh)]
    handback = introduced[-1] if introduced else "open"

    return Slots(
        offer=" ".join(last_words[:12]) or last[:60],
        accept=" ".join(carried[:6]),
        add=add,
        stakes=stakes,
        handback=handback,
    )
```

- [ ] **Step 8: Run the whole test file**

Run: `python3 -m pytest tests/test_improv.py -v`
Expected: PASS (6 passed)

- [ ] **Step 9: Commit**

```bash
git add train/improv.py tests/test_improv.py
git commit -m "feat(improv): five-slot think-block schema, extraction, render/parse

Slots hold spans lifted from the text, never paraphrases. Extraction returns None
to DROP an example when nothing carries forward or nothing is added — those are
blocks, and blocks must not be training exemplars."
```

---

### Task 2: Prove `SFTTrainer` trains this model with a loss mask

**Front-loaded because it is the largest risk in the plan.** This repo has only ever
trained through `train/run.py`; `SFTTrainer` is an unexercised path. Everything downstream
assumes it works. Needs hardware.

**Files:**
- Create: `scripts/smoke_sft_trainer.py`
- Test: manual — this task's deliverable is evidence, not a unit test

**Interfaces:**
- Consumes: `train.improv.render_think` (Task 1) for realistic text.
- Produces: proof that `SFTTrainer` + `sft_collate_fn` + `loss_mask` runs on this model, and a recorded `loss_mask.sum()` vs `B*T` figure.

- [ ] **Step 1: Write the smoke script**

```python
#!/usr/bin/env python3
# scripts/smoke_sft_trainer.py
"""Does ttml's SFTTrainer train OUR model with a loss mask? Eight hand-made examples.

Front-loaded deliberately. Every later task assumes this path works, and this repo has
only ever trained through train/run.py. Failing here changes the plan, not the code.

    gozer run --chips 1 --who claude:improv --reason "SFTTrainer smoke" -- \
      python3 scripts/smoke_sft_trainer.py
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT = "Lily found a needle. She showed it to her mother."
COMPLETION = " Her mother took the needle and sewed the button."


def main() -> int:
    from transformers import AutoTokenizer
    import ttml
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(str(ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue"))
    p_ids = tok.encode(PROMPT)
    c_ids = tok.encode(COMPLETION, add_special_tokens=False)

    # -100 on prompt positions is what tells sft_collate_fn which tokens to mask.
    example = {"input_ids": p_ids + c_ids,
               "labels": [-100] * len(p_ids) + c_ids}
    examples = [example] * 8

    collate = partial(sft_collate_fn, max_seq_len=512, pad_token_id=tok.pad_token_id or 0)
    loader = InMemoryDataloader(examples, batch_size=4, collate_fn=collate, shuffle=False)

    # THE CONTRACT: loss_mask.sum() must equal B*T, or the masked mean is silently wrong.
    batch = next(iter(loader))
    import ttnn
    mask = batch.loss_mask.to_numpy(ttnn.DataType.FLOAT32)
    b, t = 4, mask.size // 4
    print(f"loss_mask.sum()={mask.sum():.2f}  B*T={b * t}  "
          f"ratio={mask.sum() / (b * t):.4f}   (contract: ratio == 1.0)")

    model = ttml.models.llama.create_llama_from_config(
        str(ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"))
    trainer = SFTTrainer(
        model=model, train_dataloader=loader, eval_dataloader=None,
        config=SFTConfig(max_steps=4, learning_rate=1e-5, seed=5489,
                         max_seq_len=512, save_interval=0, eval_interval=0),
        optimizer={"type": "AdamW", "lr": 1e-5, "weight_decay": 0.01},
    )
    trainer.train()
    print("SFTTrainer completed 4 masked steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Take a lease and run it**

```bash
gozer acquire --chips 1 --who "claude:improv" --reason "SFTTrainer masked smoke" --expect 30m
# export the TT_VISIBLE_DEVICES it prints, then:
python3 scripts/smoke_sft_trainer.py
```

Expected: prints `ratio == 1.0000` and `SFTTrainer completed 4 masked steps`, exit 0.

- [ ] **Step 3: If the model constructor name is wrong, find the real one**

`ttml.models.llama.create_llama_from_config` is the expected entry point. If it does not
exist, find the constructor `train/run.py` already uses:

```bash
grep -n "llama\|create_.*model\|from_config" train/run.py | head -20
```

Use that instead and record the substitution in the commit message. Do **not** edit
anything under `/home/ttuser/tt-metal`.

- [ ] **Step 4: Release the lease**

```bash
gozer release <lease-id>
```

- [ ] **Step 5: Commit the script and the evidence**

```bash
git add scripts/smoke_sft_trainer.py
git commit -m "test(improv): prove SFTTrainer trains this model with a loss mask

<paste the ratio line and the completion line here>

The largest risk in the plan, front-loaded. Repo had only ever trained through
train/run.py; SFTTrainer was unexercised. The loss_mask.sum() == B*T contract is
checked explicitly because a mis-normalised mask makes the masked mean silently
wrong rather than failing."
```

**If this task fails**, stop and report. Do not proceed to Task 5 — the fallback
(full-sequence loss, no masking) changes the design and needs a decision, not a workaround.

---

### Task 3: Harm lexicon and the four scorers

**Files:**
- Create: `train/data/harm_lexicon.txt`, `scripts/score_improv.py`
- Test: `tests/test_score_improv.py`

**Interfaces:**
- Consumes: `train.improv.content_words`, `split_sentences` (Task 1).
- Produces:
  - `@dataclass(frozen=True) ImprovScores(novelty: int, escalation: float, new_harm: bool, groundedness: float, new_proper_nouns: int, affordance: int)`
  - `load_harm_lexicon(path: Path = ...) -> frozenset[str]`
  - `intensity(text: str, harm: frozenset[str]) -> float`
  - `score_pair(prefix: str, continuation: str, *, harm: frozenset[str], cooc: dict[str, set[str]]) -> ImprovScores`

- [ ] **Step 1: Create the harm lexicon with a provenance header**

```
# train/data/harm_lexicon.txt
# Curated harm/violence/death terms for the `escalation` scorer.
# HAND-CURATED, not scraped: a scraped list cannot be audited, and this one is small
# enough to read. Terms are lemma-ish surface forms because the corpus is simple prose.
# Used ONLY in deltas (intensity(continuation) - intensity(prefix)); absolute intensity
# would flag half the corpus, because `folklore` is legitimately dark.
died
death
dead
kill
killed
blood
bleeding
wound
wounded
hurt
hurts
pain
scream
screamed
cried
crying
broke
broken
smash
smashed
burn
burned
burning
drown
drowned
lost
lonely
afraid
scared
terrified
angry
shouted
hit
bit
cut
sharp
danger
dangerous
trapped
alone
sad
tears
sick
fever
cold
starving
hungry
```

- [ ] **Step 2: Write the failing discrimination tests**

```python
# tests/test_score_improv.py
"""Scorer tests. A scorer never seen to discriminate is a claim, not a check."""
from pathlib import Path

from scripts.score_improv import (ImprovScores, intensity, load_harm_lexicon,
                                  score_pair)

HARM = load_harm_lexicon()
GENTLE_P = "Lily found a shiny rock. She showed it to her friend."
GENTLE_C = "They put the rock on the windowsill and watched it catch the light."
GRIM_P = "The wolf had killed the lamb and the blood was on the snow."
GRIM_C = "The shepherd wept, and the cold wind cut at his wounded hands."


def test_novelty_is_zero_for_a_restatement():
    """The boring block. Zero new content words is the signal."""
    s = score_pair(GENTLE_P, "Lily found a shiny rock.", harm=HARM, cooc={})
    assert s.novelty == 0


def test_novelty_is_positive_for_a_real_continuation():
    assert score_pair(GENTLE_P, GENTLE_C, harm=HARM, cooc={}).novelty > 0


def test_grim_text_scores_high_absolute_intensity():
    assert intensity(GRIM_P, HARM) > intensity(GENTLE_P, HARM)


def test_grim_continuing_grim_is_NOT_escalation():
    """THE central property. Absolute darkness would flag all folklore; the delta must not.

    A grim story staying grim is not going to the worst place. Only a rise is.
    """
    grim_to_grim = score_pair(GRIM_P, GRIM_C, harm=HARM, cooc={})
    gentle_to_grim = score_pair(GENTLE_P, GRIM_C, harm=HARM, cooc={})
    assert abs(grim_to_grim.escalation) < gentle_to_grim.escalation


def test_new_harm_flags_only_the_gentle_to_grim_case():
    assert score_pair(GENTLE_P, GRIM_C, harm=HARM, cooc={}).new_harm is True


def test_groundedness_ranks_connected_above_unconnected():
    cooc = {"rock": {"windowsill", "light"}, "lily": {"friend"}}
    connected = score_pair(GENTLE_P, GENTLE_C, harm=HARM, cooc=cooc)
    unconnected = score_pair(
        GENTLE_P, "Gorthax and Vermilion argued about the Treaty of Blunn.",
        harm=HARM, cooc=cooc)
    assert connected.groundedness > unconnected.groundedness
    assert unconnected.new_proper_nouns >= 3


def test_affordance_separates_closed_from_open_endings():
    closed = score_pair(GENTLE_P, "They went to bed.", harm=HARM, cooc={})
    open_ = score_pair(GENTLE_P, "But what was inside the box?", harm=HARM, cooc={})
    assert open_.affordance == 1
    assert closed.affordance == 0
```

- [ ] **Step 3: Run to verify they fail**

Run: `python3 -m pytest tests/test_score_improv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.score_improv'`

- [ ] **Step 4: Implement the scorers**

```python
#!/usr/bin/env python3
# scripts/score_improv.py
"""Four scorers for improv continuations. EVERY ONE IS A DELTA from prefix to continuation.

The failure modes are relative, not absolute. A grim folktale continuing grim is not
"going to the worst place possible" — only a rise is. Absolute darkness scoring would
flag half this corpus, because the `folklore` source is legitimately dark.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.improv import content_words, split_sentences  # noqa: E402

HARM_PATH = ROOT / "train" / "data" / "harm_lexicon.txt"
_PROPER = re.compile(r"\b([A-Z][a-z]+)\b")


@dataclass(frozen=True)
class ImprovScores:
    novelty: int
    escalation: float
    new_harm: bool
    groundedness: float
    new_proper_nouns: int
    affordance: int

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_harm_lexicon(path: Path = HARM_PATH) -> frozenset:
    terms = [ln.strip().lower() for ln in path.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    return frozenset(terms)


def intensity(text: str, harm: frozenset) -> float:
    """Harm-lexicon hits per 100 content words."""
    words = content_words(text)
    if not words:
        return 0.0
    return 100.0 * sum(1 for w in words if w in harm) / len(words)


def _proper_nouns(text: str) -> Set[str]:
    """Capitalised tokens that are not sentence-initial."""
    out: Set[str] = set()
    for sent in split_sentences(text):
        toks = sent.split()
        for tok in toks[1:]:
            m = _PROPER.match(tok)
            if m:
                out.add(m.group(1))
    return out


def score_pair(prefix: str, continuation: str, *, harm: frozenset,
               cooc: Dict[str, Set[str]]) -> ImprovScores:
    p_words, c_words = content_words(prefix), content_words(continuation)
    p_set = set(p_words)
    fresh = [w for w in c_words if w not in p_set]

    delta = intensity(continuation, harm) - intensity(prefix, harm)
    new_harm = any(w in harm for w in fresh)

    if fresh:
        grounded = sum(1 for w in fresh
                       if any(w in cooc.get(p, set()) for p in p_set)) / len(fresh)
    else:
        grounded = 1.0

    sents = split_sentences(continuation)
    tail = sents[-1] if sents else continuation
    affordance = int(tail.rstrip().endswith("?")
                     or any(w in set(fresh) for w in content_words(tail)))

    return ImprovScores(
        novelty=len(set(fresh)),
        escalation=round(delta, 4),
        new_harm=new_harm,
        groundedness=round(grounded, 4),
        new_proper_nouns=len(_proper_nouns(continuation) - _proper_nouns(prefix)),
        affordance=affordance,
    )
```

- [ ] **Step 5: Run to verify they pass**

Run: `python3 -m pytest tests/test_score_improv.py -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Commit**

```bash
git add train/data/harm_lexicon.txt scripts/score_improv.py tests/test_score_improv.py
git commit -m "feat(improv): four delta-based scorers with discrimination tests

Every scorer is a delta from prefix to continuation, because the failure modes are
relative: a grim folktale continuing grim is not going to the worst place. The
central test asserts exactly that, since absolute darkness scoring would flag half
the corpus."
```

---

### Task 4: Derive traces from the corpus

**Files:**
- Create: `scripts/derive_traces.py`
- Test: `tests/test_derive_traces.py`

**Interfaces:**
- Consumes: `train.improv.extract_slots/render_think/split_sentences`, `scripts.score_improv.intensity/load_harm_lexicon`.
- Produces: `artifacts/improv/traces.jsonl` where each line is
  `{"story_id": int, "cut_k": int, "prefix": str, "think": str, "continuation": str, "slots": {...}}`
  plus `artifacts/improv/derive_manifest.json` carrying drop counts per rule.
- Also produces, for Task 5: `build_sft_examples(traces, tok, *, with_think: bool, pad_token_id: int) -> list[dict]`
  returning `{"input_ids": [...], "labels": [...]}` with `-100` on prompt positions.
  (FIX 5(b), task-6-report.md: `pad_token_id` is REQUIRED -- the shipped implementation
  tile-aligns every example to a multiple of 32 tokens using it, per Task 2's SDPA-backward
  finding below. A stage-2 implementer copying this signature or the call sites further
  down without it will get a `TypeError: build_sft_examples() missing 1 required keyword-only
  argument: 'pad_token_id'`.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_derive_traces.py
import json

from scripts.derive_traces import build_sft_examples, derive_from_story


STORY = ("Lily found a needle in her room. She knew it was sharp. "
         "She showed the needle to her mother. Her mother sewed the button. "
         "They were both happy with the shirt.")


def test_derive_returns_a_trace_with_all_slots():
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    assert rec is not None
    assert set(rec["slots"]) == {"offer", "accept", "add", "stakes", "handback"}
    assert rec["prefix"] and rec["continuation"]
    assert "<think>" in rec["think"]


def test_sft_example_masks_only_the_prompt():
    """The mask is the whole point: prompt positions carry -100, completion carries ids."""
    class _Tok:
        def encode(self, s, add_special_tokens=True):
            return [ord(c) % 97 for c in s[:20]]

    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    ex = build_sft_examples([rec], _Tok(), with_think=True, pad_token_id=0)[0]
    assert len(ex["input_ids"]) == len(ex["labels"])
    assert ex["labels"][0] == -100, "prompt must be masked"
    assert any(v != -100 for v in ex["labels"]), "completion must be supervised"


def test_no_think_arm_omits_the_block_but_keeps_the_continuation():
    class _Tok:
        def encode(self, s, add_special_tokens=True):
            return [ord(c) % 97 for c in s[:20]]

    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    with_t = build_sft_examples([rec], _Tok(), with_think=True, pad_token_id=0)[0]
    without = build_sft_examples([rec], _Tok(), with_think=False, pad_token_id=0)[0]
    assert len(without["input_ids"]) <= len(with_t["input_ids"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_derive_traces.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.derive_traces'`

- [ ] **Step 3: Implement derivation**

```python
#!/usr/bin/env python3
# scripts/derive_traces.py
"""Derive extractive think-blocks from real corpus continuations.

The trace is read OFF the continuation that already exists: whatever the next passage
introduces IS the `add`, whatever it intensifies IS the `stakes`, whatever it leaves open
IS the `handback`. No generator is involved, so nothing unvalidated enters the pipeline.

Drop rules make the training set EXEMPLARY rather than representative — deliberate, since
we are teaching good moves, not average ones. Drop rates are reported per rule; above ~50%
the filter rather than the model is choosing the behaviour.

    python3 scripts/derive_traces.py --limit 20000 --out artifacts/improv/traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_improv import intensity, load_harm_lexicon  # noqa: E402
from train.improv import (content_words, extract_slots, render_think,  # noqa: E402
                          split_sentences)

STORY_SEP = "</s>"
HARM = load_harm_lexicon()
DROPS: Counter = Counter()


def derive_from_story(story: str, *, story_id: int, rng_seed: int,
                      idf: Optional[Dict[str, float]] = None) -> Optional[dict]:
    sents = split_sentences(story)
    if len(sents) < 4:
        DROPS["too_few_sentences"] += 1
        return None
    rng = random.Random(rng_seed + story_id)
    k = rng.randint(2, len(sents) - 2)
    prefix = " ".join(sents[:k])
    continuation = " ".join(sents[k:k + 2])

    slots = extract_slots(prefix, continuation, idf=idf or {},
                          intensity=lambda t: intensity(t, HARM))
    if slots is None:
        DROPS["no_carry_or_no_add"] += 1
        return None
    return {"story_id": story_id, "cut_k": k, "prefix": prefix,
            "think": render_think(slots), "continuation": continuation,
            "slots": slots.as_dict()}


def build_sft_examples(traces: List[dict], tok, *, with_think: bool,
                       pad_token_id: int) -> List[dict]:
    """`{"input_ids", "labels"}` with -100 on prompt positions, for `sft_collate_fn`.

    FIX 5(b) (task-6-report.md): `pad_token_id` is REQUIRED, not optional -- the shipped
    implementation (`scripts/derive_traces.py`) pads every example up to the next
    multiple of 32 tokens using it, because ttml's SDPA backward kernel mismatches a
    collated batch whose sequence length is not tile-aligned (see Task 2's finding
    above). This illustrative snippet omits that padding for brevity, but the signature
    must still accept the argument so callers written against it don't need a second
    change later.
    """
    out = []
    for rec in traces:
        prompt = rec["prefix"]
        completion = (rec["think"] + rec["continuation"]) if with_think else rec["continuation"]
        p_ids = tok.encode(prompt)
        c_ids = tok.encode(completion, add_special_tokens=False)
        out.append({"input_ids": p_ids + c_ids,
                    "labels": [-100] * len(p_ids) + c_ids})
    return out


def build_idf(stories: List[str]) -> Dict[str, float]:
    df: Counter = Counter()
    for s in stories:
        df.update(set(content_words(s)))
    n = max(len(stories), 1)
    return {w: math.log(n / (1 + c)) for w, c in df.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "artifacts" / "corpus" / "tinystories.txt")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "improv" / "traces.jsonl")
    args = ap.parse_args()

    stories = [s.strip() for s in args.corpus.read_text(errors="ignore").split(STORY_SEP)
               if s.strip()][:args.limit]
    print(f"stories read: {len(stories):,} (separator {STORY_SEP!r}, NOT a blank line)")
    idf = build_idf(stories)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.out.open("w") as fh:
        for i, story in enumerate(stories):
            rec = derive_from_story(story, story_id=i, rng_seed=args.seed, idf=idf)
            if rec is not None:
                fh.write(json.dumps(rec) + "\n")
                kept += 1

    total = len(stories)
    rate = 1 - kept / max(total, 1)
    manifest = {"corpus": str(args.corpus), "separator": STORY_SEP, "seed": args.seed,
                "stories": total, "kept": kept, "drop_rate": round(rate, 4),
                "drops_by_rule": dict(DROPS)}
    (args.out.parent / "derive_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"kept {kept:,}/{total:,}  drop rate {rate:.1%}")
    for rule, n in DROPS.most_common():
        print(f"    {rule:24} {n:,}")
    if rate > 0.5:
        print("WARNING: drop rate above 50% — the FILTER is choosing the behaviour, "
              "not the model. Report this with any result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_derive_traces.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Derive the real traces and read the drop report**

Run: `python3 scripts/derive_traces.py --limit 20000`
Expected: prints the kept count and a per-rule drop table. If drop rate > 50%, record it
and say so in the commit — it is a finding, not a nuisance.

- [ ] **Step 6: Commit**

```bash
git add scripts/derive_traces.py tests/test_derive_traces.py
git commit -m "feat(improv): derive extractive traces from real continuations

Traces are read OFF continuations that already exist, so no generator enters the
pipeline. Drop rules make the set exemplary rather than representative — a
deliberate choice, with per-rule drop rates reported because above ~50% the filter
rather than the model is choosing the behaviour.

<paste the drop table here>"
```

---

### Task 5: Train the two paired arms

Needs hardware. **Do not start until Task 2 passed.**

**Files:**
- Create: `scripts/train_improv.py`

**Interfaces:**
- Consumes: `scripts.derive_traces.build_sft_examples`, the Task 2 constructor call that actually worked.
- Produces: `artifacts/improv/ckpt-think/` and `artifacts/improv/ckpt-nothink/`, plus `artifacts/improv/train_manifest.json`.

- [ ] **Step 1: Write the training script**

```python
#!/usr/bin/env python3
# scripts/train_improv.py
"""Two paired SFT arms: think vs no-think. Same seed, same cut points, same steps.

PAIRED BY CONSTRUCTION: both arms consume the same traces in the same order, so per-step
noise cancels in the arm-vs-arm delta. The only difference is whether the completion
carries the think-block.

    gozer acquire --chips 1 --who claude:improv --reason "improv SFT arms"
    python3 scripts/train_improv.py --arm think
    python3 scripts/train_improv.py --arm nothink
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.derive_traces import build_sft_examples  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["think", "nothink"], required=True)
    ap.add_argument("--traces", type=Path,
                    default=ROOT / "artifacts" / "improv" / "traces.jsonl")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import ttml
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(str(ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue"))
    pad_token_id = tok.pad_token_id or 0
    traces = [json.loads(l) for l in args.traces.open()]
    # FIX 5(b) (task-6-report.md): pad_token_id is a required keyword-only argument on
    # the shipped build_sft_examples -- omitting it (as this snippet did before the fix)
    # raises TypeError, it does not silently fall back to some default.
    examples = build_sft_examples(traces, tok, with_think=(args.arm == "think"),
                                  pad_token_id=pad_token_id)
    print(f"arm={args.arm}  examples={len(examples):,}  seed={args.seed}")

    collate = partial(sft_collate_fn, max_seq_len=512, pad_token_id=pad_token_id)
    loader = InMemoryDataloader(examples, batch_size=args.batch_size,
                                collate_fn=collate, shuffle=True, seed=args.seed)

    out = ROOT / "artifacts" / "improv" / f"ckpt-{args.arm}"
    out.mkdir(parents=True, exist_ok=True)

    # Substitute the constructor Task 2 established, if it differed.
    model = ttml.models.llama.create_llama_from_config(
        str(ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"))
    ttml.serialization.load_model(
        model, str(ROOT / "artifacts" / "checkpoints-v077-beta2-control"
                   / "tt_tnt_step00010764.pkl"))

    trainer = SFTTrainer(
        model=model, train_dataloader=loader, eval_dataloader=None,
        config=SFTConfig(max_steps=args.steps, learning_rate=args.lr, seed=args.seed,
                         max_seq_len=512, checkpoint_dir=str(out),
                         save_interval=1000, eval_interval=0, max_grad_norm=1.0),
        optimizer={"type": "AdamW", "lr": args.lr, "weight_decay": 0.01},
    )
    trainer.train()

    (out / "train_manifest.json").write_text(json.dumps({
        "arm": args.arm, "traces": str(args.traces), "n_examples": len(examples),
        "steps": args.steps, "seed": args.seed, "batch_size": args.batch_size,
        "lr": args.lr, "ddp": 2,
        "warm_start": "artifacts/checkpoints-v077-beta2-control/tt_tnt_step00010764.pkl",
        "paired_with": "nothink" if args.arm == "think" else "think",
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the warm-start loader name before running**

`ttml.serialization.load_model` is the expected name. Confirm against what `train/run.py`
uses for `--warm-start`:

```bash
grep -n "load_model\|serialization\|warm_start" train/enthusiasts.py train/run.py | head -10
```

Use the real function and note any substitution in the commit message.

- [ ] **Step 3: Take one board and run both arms**

```bash
gozer acquire --chips 1 --who "claude:improv" --reason "improv SFT paired arms" --expect 120m
python3 scripts/train_improv.py --arm think
python3 scripts/train_improv.py --arm nothink
gozer release <lease-id>
```

Expected: both arms complete; two checkpoint dirs each with a `train_manifest.json`.
**One board only** — never the 4-chip mesh.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_improv.py
git commit -m "feat(improv): paired think/no-think SFT arms via SFTTrainer

Both arms consume identical traces in identical order under seed 5489, so the
comparison is paired and per-step noise cancels in the delta. One board, --ddp 2."
```

---

### Task 6: Evaluate — swap test first, then the A/B

**Files:**
- Create: `scripts/eval_improv.py`
- Test: `tests/test_eval_improv.py`

**Interfaces:**
- Consumes: `scripts.score_improv.score_pair/load_harm_lexicon`, `train.improv.parse_think`.
- Produces: `docs/measurements/improv-stage1.json` with `swap_test`, `adherence`, `rates`, `verdict`.

- [ ] **Step 1: Write the failing statistics tests**

```python
# tests/test_eval_improv.py
from scripts.eval_improv import BONFERRONI_ALPHA, paired_verdict, swap_verdict


def test_bonferroni_threshold_is_stated_for_five_tests():
    """Four scorers plus adherence. Uncorrected 0.05 would read three nulls as real."""
    assert abs(BONFERRONI_ALPHA - 0.01) < 1e-9


def test_identical_curves_are_not_interpretable():
    a = [1.0] * 10
    assert paired_verdict(a, list(a))["verdict"] == "NOT INTERPRETABLE"


def test_a_clear_separation_is_reported():
    a = [1.0] * 10
    b = [2.0] * 10
    assert paired_verdict(a, b)["verdict"] != "NOT INTERPRETABLE"


def test_swap_verdict_fails_when_output_is_invariant():
    """If swapping the think-block changes nothing, the thinking is DECORATIVE."""
    res = swap_verdict(divergence_positions=[None] * 50, n=50)
    assert res["thinking_is_load_bearing"] is False


def test_swap_verdict_passes_when_output_moves():
    res = swap_verdict(divergence_positions=[3] * 50, n=50)
    assert res["thinking_is_load_bearing"] is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_eval_improv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval_improv'`

- [ ] **Step 3: Implement the statistics and verdicts**

```python
#!/usr/bin/env python3
# scripts/eval_improv.py
"""Stage-1 evaluation: the swap test FIRST, then schema adherence and failure rates.

THE SWAP TEST RUNS FIRST ON PURPOSE. If substituting another story's think-block barely
changes the continuation, the model learned to emit a plan and ignore it — the thinking is
decorative and stage 1 has failed, regardless of how good the other numbers look. This
control exists because die-region gate seeding once worked as a classifier (61.2% against
10% chance) and bought nothing measurable in loss. Skill at producing an artefact is not
evidence the artefact is used.
"""
from __future__ import annotations

import statistics as st
from typing import Dict, List, Optional, Sequence

#: Four scorers plus schema adherence = five tests. Uncorrected 0.05 would read a null
#: sitting at 2.0 standard errors as a real effect.
BONFERRONI_ALPHA = 0.01
#: Two-sided normal critical value at alpha = 0.01.
CRITICAL_T = 2.576


def paired_verdict(a: Sequence[float], b: Sequence[float]) -> Dict[str, object]:
    """Paired comparison of two equal-length score series."""
    if len(a) != len(b) or not a:
        raise ValueError(f"paired series must be equal-length and non-empty: {len(a)}, {len(b)}")
    deltas = [x - y for x, y in zip(a, b)]
    mean = st.fmean(deltas)
    sd = st.pstdev(deltas)
    se = sd / (len(deltas) ** 0.5) if sd > 0 else 0.0
    t = abs(mean / se) if se > 0 else 0.0
    pos = sum(1 for d in deltas if d > 0)
    verdict = ("NOT INTERPRETABLE" if t <= CRITICAL_T
               else ("think better" if mean < 0 else "no-think better"))
    return {"mean_delta": round(mean, 4), "sd": round(sd, 4), "se": round(se, 4),
            "t": round(t, 3), "signs_pos": pos, "signs_neg": len(deltas) - pos,
            "n": len(deltas), "critical_t": CRITICAL_T, "verdict": verdict}


def swap_verdict(divergence_positions: Sequence[Optional[int]], n: int) -> Dict[str, object]:
    """Did substituting another story's think-block change the continuation?

    `divergence_positions[i]` is the token index where the swapped generation first differs
    from the original, or None if it never differs.
    """
    changed = [p for p in divergence_positions if p is not None]
    frac = len(changed) / max(n, 1)
    return {"n": n, "n_changed": len(changed), "fraction_changed": round(frac, 4),
            "median_first_divergence": (st.median(changed) if changed else None),
            "thinking_is_load_bearing": frac >= 0.5,
            "note": ("Below 0.5 the think-block is decorative: the model emits a plan and "
                     "writes independently of it. Stage 1 has failed in that case, and "
                     "that is the correct thing to report.")}
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_eval_improv.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Add the generation driver**

Append a `main()` to `scripts/eval_improv.py` that:
1. loads 200 held-out story openings not present in `artifacts/improv/traces.jsonl` (match on `story_id`);
2. generates from both checkpoint dirs at greedy and T=0.8 × 3 samples;
3. runs the swap test on 50 openings from the think arm, substituting another story's think-block;
4. computes `parse_think` adherence and the four scorers for both arms;
5. calls `paired_verdict` per scorer and `swap_verdict` once;
6. writes `docs/measurements/improv-stage1.json` with all of it plus the success-criteria check.

Success criteria to assert in the output: adherence ≥ 0.80; at least 2 of 4 scorers with a
verdict other than NOT INTERPRETABLE favouring think; `thinking_is_load_bearing` true.

- [ ] **Step 6: Run the evaluation**

```bash
gozer acquire --chips 1 --who "claude:improv" --reason "improv stage-1 eval" --expect 60m
python3 scripts/eval_improv.py --out docs/measurements/improv-stage1.json
gozer release <lease-id>
```

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_improv.py tests/test_eval_improv.py docs/measurements/improv-stage1.json
git commit -m "feat(improv): stage-1 eval — swap test first, then adherence and rates

The swap test runs first and can fail the stage on its own: if substituting another
story's think-block barely changes the continuation, the thinking is decorative.
Adherence and story quality are reported separately because at 123M they come apart.
Five tests, so the Bonferroni threshold is alpha=0.01, |t| > 2.576.

<paste the verdict block here>"
```

---

## Self-Review

**Spec coverage.** Schema → Task 1. Extractive-not-paraphrase → Task 1 (docstring + tests).
Trace derivation with drop rules and drop-rate reporting → Task 4. Four delta scorers with
validation → Task 3. Harm lexicon with provenance → Task 3. Two paired arms, one board,
warm start → Task 5. Adherence + rates reported separately → Task 6. Bonferroni α=0.01 →
Task 6. Swap test first → Task 6. `SFTTrainer` unexercised-path risk → Task 2, front-loaded
with an explicit stop-and-report instruction.

**Not covered by design:** stage 2 (turn-taking) is outlined in the spec, not planned here.
The craft sources are design input and produce no code, so they appear in no task — correct,
and stated so nobody hunts for the missing task.

**Type consistency.** `Slots` (Task 1) is consumed by Task 4 via `extract_slots`/`render_think`.
`ImprovScores` (Task 3) is consumed by Task 6. `build_sft_examples` (Task 4) is consumed by
Task 5. `intensity` is defined once in Task 3 and injected into `extract_slots` as a callable
so `train/improv.py` never imports from `scripts/`.

**Two constructor names are unverified** and each has an explicit verification step against
`train/run.py` before use: `ttml.models.llama.create_llama_from_config` (Tasks 2, 5) and
`ttml.serialization.load_model` (Task 5). They are flagged rather than assumed because the
plan was written from reading ttml's source, not from running it.
