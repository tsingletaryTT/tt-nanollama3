<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# What is actually in the blend

The corpus itself is ~1.7 GB and is not committed; `artifacts/` is gitignored. This page and
[`measurements/blend_manifest.json`](measurements/blend_manifest.json) are the in-repo record
of what `scripts/blend_corpus.py` built, so "what was this model trained on" is answerable
from a clone rather than only from the machine that ran the blend.

**The manifest is authoritative.** It is written by the blend itself. The figures below are
copied from it and `tests/test_corpus_blend_doc.py` holds them to it, so this page cannot
drift from the artifact it describes.

## The headline number

**399,594,747 tokens** against a **400,000,000** budget — **405,253 short, −0.101%**.

That is the real count from the trained tokenizer, not an estimate. Each source's emitted
text is counted as it is written, chunked into paragraphs exactly the way
`scripts/measure_corpus.py` chunks a source file, so `emitted_tokens` and `available_tokens`
are the same kind of number and can be divided by each other. (BPE merges do not cross an
`encode()` call, so a different chunking would produce a slightly different total and the two
would no longer be comparable.)

The shortfall is the truncation of each source's final pass landing a word or two early,
nine times over. It is not a share problem: every slice is within 0.065 points of its target.

## Per source

| Source | Emitted tokens | Achieved share | Target | Real repetition | Declared `upsample` | tokens/word |
|---|---:|---:|---:|---:|---:|---:|
| `flavour` | 1,979,776 | 0.495% | 0.5% | 3.476x | 4x | 1.412315 |
| `folklore` | 32,078,474 | 8.028% | 8% | 1.5041x | 2x | 1.357535 |
| `gutenberg_children` | 59,984,098 | 15.011% | 15% | 1.7516x | 2x | 1.322561 |
| `poetry` | 3,950,673 | 0.989% | 1% | 0.1309x | 1x | 1.391492 |
| `procedural` | 47,994,328 | 12.011% | 12% | 3.911x | 4x | 1.340914 |
| `spine` | 53,914,933 | 13.492% | 13.5% | 2.061x | 3x | 1.337552 |
| `tinystories` | 124,034,584 | 31.040% | 31% | 0.2795x | 1x | 1.193674 |
| `weird` | 15,977,977 | 3.999% | 4% | 2.2724x | 3x | 1.311573 |
| `wikipedia_simple` | 59,679,904 | 14.935% | 15% | 0.8825x | 1x | 1.558776 |

`blend.txt` SHA-256 `da3d1bea402aaf5b0182fbb235cd368f6dafde70894213c38be332cc02a1fcc7`
(reproduced byte-identical on a second run).

## A second count: tokenizing the finished file

The headline number above (399,594,747) is **not** the only token count this project has for
this corpus, and the other one was missing from this page until now — which is exactly how a
reviewer came to flag it as unverifiable. Recorded here so it stops being a number that only
exists in a training run's own header.

`train/tokenization.py` runs the retrained tokenizer once over the finished, concatenated
`artifacts/corpus/blend.txt` and splits the result into train/val arrays. That pass produced:

| | Tokens |
|---|---:|
| Total | **392,773,300** |
| Train split | **353,495,970** |
| Val split | **39,277,330** |
| Vocabulary | 32,000 |

This is directly checkable on disk: `artifacts/tokens/train_ids.npy` and
`artifacts/tokens/val_ids.npy` are `uint32` arrays of shape `(353495970,)` and `(39277330,)`
respectively, summing to 392,773,300. The same total also appears as `corpus_tokens` in the
header of every `tt-tnt-v1` checkpoint (e.g. `artifacts/checkpoints-tt-tnt-v1/tt_tnt_step00010787.pkl`),
since `train/run.py` records whatever `train/tokenization.py` measured at tokenization time.

**Why this differs from the manifest's 399,594,747.** They are two different measurements of
two different things, not two attempts at the same number:

- The manifest's figure is a **sum of nine separate tokenizer calls**, one per source, each
  over that source's own emitted text, chunked into paragraphs exactly the way
  `scripts/measure_corpus.py` chunks a source file for its availability check (see "The
  headline number" above).
- This page's new figure is **one tokenizer call over the single concatenated file** that
  `blend_corpus.py` writes all nine sources into.

BPE merges do not cross an `encode()` call, so tokenizing nine chunks separately and
tokenizing their concatenation as one string can legitimately merge (or fail to merge) a
different set of byte pairs at every join — the boundary between `flavour`'s last paragraph
and `folklore`'s first, for instance, is a real encode-time seam in one measurement and
invisible in the other. A 0.46% difference between 399,594,747 and 392,773,300 is consistent
with that many source-to-source boundaries, not a sign either count is wrong. **Treat
399,594,747 as the per-source provenance figure (how the blend was assembled) and
392,773,300/353,495,970/39,277,330 as the tokenized-training-data figure (what
`train/run.py` actually reads token-by-token)** — use whichever one answers the question being
asked, and do not average them or treat the smaller one as a correction to the larger one.

**This is also where the step budget for one epoch comes from.** At `batch_size=64`,
`seq_len=512`, one step consumes 64 × 512 = 32,768 tokens. The **train split**, not the
manifest total, is what a training run actually iterates over:
353,495,970 / 32,768 ≈ 10,787.98, i.e. **10,787 steps** covers one epoch (with the last
32,768-token step short by construction, since 10,787 × 32,768 = 353,468,416 is 27,554 tokens
less than the full split — the standard "drop the final partial batch" behavior, not a bug).

### Real repetition is not the declared `upsample`

`upsample` in `train/corpus.py` is a **ceiling** — the most repetition a source is allowed to
carry, which is what the availability gate in `scripts/measure_corpus.py` checks against. The
repetition actually applied is `required_tokens / available_tokens`, and it is fractional.

Read the two columns together:

- **`procedural` 3.911x against a 4x limit.** The tightest slice in the registry. Task 6 moved
  a whole share point (13% → 12%) to keep it there; this is the number that move was for.
- **`wikipedia_simple` 0.8825x.** Below 1.0: 88% of Simple Wikipedia is used once and the rest
  is not used at all. Nothing is duplicated.
- **`tinystories` 0.2795x and `poetry` 0.1309x.** Same thing, further out — these sources are
  large relative to their shares, so most of each file never enters the blend.
- **`flavour` 3.476x.** The whole file, three and a half times over. That is deliberate (see
  its rationale) and it is close to its ceiling: `flavour` has 0.075 points of headroom.

A source whose real repetition EXCEEDED its declared `upsample` would mean the blend repeats
material the registry says it does not. That is what shipped before this was fixed:
`wikipedia_simple` made 1.058 passes while declaring `upsample=1`, duplicating ~5.8% of Simple
Wikipedia undeclared, and `procedural` made 4.034 passes against the 4x limit. The cause was
`_emit` sizing its output with a flat 1.3 tokens/word while the gate used tokenizer-measured
availability; real tokens/word runs 1.194–1.559 across these nine sources, so it over-emitted
for eight of them. `blend_corpus.py` now derives each source's ratio from the measurement, and
`tests/test_blend_corpus.py` pins that shut.

## The tokenizer was trained on an earlier revision of this blend

The shipped tokenizer (`artifacts/tokenizer/`, 32,000-token BPE) was **not** trained on the
blend described above. It was trained on the blend as it stood at 14:33 on 2026-08-13; that
blend was then rebuilt at 14:48 by the Task 6 re-settle, and rebuilt again by the fix
described in the previous section. **This is known and accepted, not an oversight.**

The dependency is circular:

```
tokenizer -> token availability per source -> settled shares -> blend -> tokenizer
```

Each arrow is real. Availability is measured in tokens, which needs a tokenizer. Shares are
settled against availability. The blend realises the shares. Training a tokenizer on the blend
changes the vocabulary, which changes availability, which can move the shares. The loop does
not converge on its own and has to be cut somewhere; every choice of cut leaves the tokenizer
one revision behind the corpus it will be used on.

What the cut costs here is small and bounded:

- The vocabulary was trained on **the same nine sources at near-identical proportions** — the
  re-settle moved one share point from `procedural` to `tinystories` and raised two `upsample`
  factors; the C2 fix changed no share at all, only how much text is emitted to hit one.
- A BPE vocabulary is a compression table, not a claim about the data. Being trained on 12.011%
  procedural rather than 13% costs a little compression efficiency on that slice. It cannot
  make any text untokenizable: byte-level BPE has no out-of-vocabulary case.
- The effect is measurable and was measured: the retrain moved per-source availability by
  −0.5% (`tinystories`) to −23.8% (`wikipedia_simple`), and the shares were re-settled against
  those new numbers. A second retrain would move them again, by less.

Retraining the tokenizer on the current blend would restart the loop — new vocabulary, new
availability, new shares, new blend, and the same statement to write one revision later. It is
deliberately not being done.

**If you retrain it**, re-run `scripts/measure_corpus.py` and `scripts/blend_corpus.py`
afterwards and settle the shares against the new measurement, exactly as Task 6 did. Watch
`flavour` in particular: it sits 0.075 points under its arithmetic ceiling, and a further 13%
fall in its measured availability makes even a 0.5% share unreachable within the 4x cap.

## Rebuilding it

```bash
python scripts/check_disk_space.py     # refuses to start if the volume is too full
python scripts/fetch_corpus.py
python scripts/prepare_corpus.py
python scripts/measure_corpus.py       # -> docs/measurements/corpus_availability.json
python scripts/blend_corpus.py         # -> artifacts/corpus/blend.txt + both manifests
```

The blend is deterministic: same sources, same availability report, same bytes, same SHA-256.
On a fresh clone with no tokenizer yet, `measure_corpus.py` falls back to a word approximation
and says so in its report; the numbers settle on the second pass, once a tokenizer exists.
