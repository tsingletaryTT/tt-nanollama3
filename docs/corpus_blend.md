<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# What is actually in the blend

The corpus itself is ~1.7 GB and is not committed; `artifacts/` is gitignored. This page and
[`measurements/blend_manifest.json`](measurements/blend_manifest.json) are the in-repo record
of what `scripts/blend_corpus.py` built, so "what was this model trained on" is answerable
from a clone rather than only from the machine that ran the blend.

The manifest is authoritative. It is written by the blend itself. The figures below are
copied from it and `tests/test_corpus_blend_doc.py` holds them to it, so this page cannot
drift from the artifact it describes.

## The headline number

399,508,203 tokens against a **400,000,000** budget — **491,797 short, −0.123%**.

That is the real count from the trained tokenizer, not an estimate. Each source's emitted
text is counted as it is written, chunked into paragraphs exactly the way
`scripts/measure_corpus.py` chunks a source file, so `emitted_tokens` and `available_tokens`
are the same kind of number and can be divided by each other. (BPE merges do not cross an
`encode()` call, so a different chunking would produce a slightly different total and the two
would no longer be comparable.)

The shortfall is the truncation of each source's final pass landing a word or two early,
nine times over. It is not a share problem: every slice is within 0.083 points of its target.

## Per source

| Source | Emitted tokens | Achieved share | Target | Real repetition | Declared `upsample` | tokens/word |
|---|---:|---:|---:|---:|---:|---:|
| `flavour` | 1,979,789 | 0.496% | 0.5% | 3.4759x | 4x | 1.412325 |
| `folklore` | 32,078,464 | 8.029% | 8% | 1.5041x | 2x | 1.357543 |
| `gutenberg_children` | 59,984,104 | 15.014% | 15% | 1.7516x | 2x | 1.322577 |
| `poetry` | 3,950,536 | 0.989% | 1% | 0.1305x | 1x | 1.392825 |
| `procedural` | 47,994,272 | 12.013% | 12% | 3.9109x | 4x | 1.340927 |
| `spine` | 53,915,065 | 13.495% | 13.5% | 2.061x | 3x | 1.33756 |
| `tinystories` | 124,030,364 | 31.046% | 31% | 0.2768x | 1x | 1.198246 |
| `weird` | 15,977,960 | 3.999% | 4% | 2.2724x | 3x | 1.31158 |
| `wikipedia_simple` | 59,597,649 | 14.918% | 15% | 0.8763x | 1x | 1.561208 |

`blend.txt` SHA-256 `24f3d112e04696630ff6553dbd9440ce77b54a54204b17b589ee2ec4cfc9f4d1`.

These numbers moved slightly on 2026-08-14 when document separators were added (see the next
section). Nothing about the shares changed: the budget is fixed, so the ~800k separator
tokens displace ordinary text rather than adding to the total, and each source's measured
`tokens/word` rose by the two tokens per document the separator and its newline contribute.

## Document boundaries

Every document in the blend is terminated by a line holding exactly `</s>` — the trained
tokenizer's eos token, id 2. There are **798,771** of them, one per ~500 tokens, 0.200% of
the corpus. `scripts/prepare_corpus.py` writes them, because it is the only stage that can
see a document boundary at all: `scripts/fetch_corpus.py` writes one JSON object per
document, `prepare_corpus.py` consumes them one at a time, and everything downstream sees
nothing but concatenated text.

This was missing, and it mattered. Until 2026-08-14 a document was written as
`text + "\n\n"`, which spells a document boundary exactly the way a paragraph break *inside* a
document is spelled. Nothing downstream could tell the two apart. `train/tokenization.py`
compounded it: it encodes the corpus one line at a time and drops the newline, so blank lines
contribute no tokens whatsoever. The result was a corpus containing **zero** `</s>` and token
arrays containing **zero** occurrences of id 2 — while the legacy TinyStories-only
`artifacts/corpus/corpus.txt`, which the *published* model trained on, contains 662,878 of
them.

The consequence was measured, not assumed. A position-wise loss probe on `tt-tnt-v1` shows
per-token loss flat from position ~64 out to 511, on books as much as on short items. With
boundaries unmarked, distant context genuinely *is* unpredictable, so a model that ignores it
is behaving correctly. The mid-generation topic collapse in the sample sheets ("The stick
remembered being a good friend... Once upon a time, there was a little bird") is the same
fact from the other side: the model faithfully reproducing unmarked document transitions it
was trained on. And having never seen an eos token, it could not learn to stop. This is also
why `train/configs/model/tt-tnt-384.yaml` only raised `max_sequence_length` to 2048 *after*
this fix — a longer window buys nothing while distant context is noise.

### Why `</s>` specifically

It is already id 2 in `artifacts/tokenizer/`, added as a *special* token, so it encodes to
exactly one id and byte-level BPE can neither split it nor absorb a neighbouring character
into it. It is `special_tokens_map.json`'s `eos_token`, and `convert/to_hf.py` writes
`eos_token_id: 2` into both `config.json` and `generation_config.json` of every published
model directory. A model that learns to emit it therefore terminates cleanly under
`transformers` and vLLM with no additional plumbing — the serving path was already waiting for
a token the training data never contained.

### Poetry is not one document per row

`biglam/gutenberg-poetry-corpus` has one row per **line** of verse: 3,085,117 rows averaging
about seven words. Treating a row as a document would have fired an end-of-document token
every seven words and, at this slice's 1% share, put roughly a third of every `</s>` in the
whole blend inside it — teaching a seven-word prior for "stop", which is the opposite of what
a document separator is for.

`CorpusSource.rows_per_document` records the distinction: 64 for `poetry`, 1 for every other
source, whose rows really are documents. Sixty-four consecutive lines (~450 words, the scale
of a short story) become one document, so `poetry` contributes 48,205 documents rather than
3,085,114 and carries 6,002 of the blend's separators rather than ~400,000. Rows arrive in
dataset order and that corpus is ordered by Gutenberg id, so the grouped lines are consecutive
lines of the same poem, occasionally straddling a book boundary at the seam. Exact poem
boundaries would need the upstream `gid` column, which `scripts/fetch_corpus.py` does not
retain; the grouping is an approximation and is documented as one.

### The truncated tail

`_emit` truncates each source's final pass at word level to hit its token target, which lands
mid-document. That fragment is now closed with a separator (never doubled, if the truncation
happened to land on one). Nine extra separators against ~400M tokens costs nothing, whereas
nine *unmarked* transitions — source A's half-sentence running straight into source B's first
document — would be the same defect this whole change removes, just rarer.

### Where the separators actually landed

Counted directly, by walking `blend.txt` and splitting it at each source's `emitted_words`
boundary from the manifest (the walk also re-verifies those boundaries: every one lands
exactly, on a line boundary, at the exact declared word count).

| Source | Separators in the blend | Documents in the prepared file | Documents/file-documents | Word-based `repetition_factor` |
|---|---:|---:|---:|---:|
| `flavour` | 26 | 7 | 3.71x | 3.4759x |
| `folklore` | 308 | 199 | 1.55x | 1.5041x |
| `gutenberg_children` | 1,016 | 583 | 1.74x | 1.7516x |
| `poetry` | 6,002 | 48,205 | 0.12x | 0.1305x |
| `procedural` | 702 | 180 | 3.90x | 3.9109x |
| `spine` | 494 | 241 | 2.05x | 2.0610x |
| `tinystories` | 584,456 | 2,119,489 | 0.28x | 0.2768x |
| `weird` | 117 | 55 | 2.13x | 2.2724x |
| `wikipedia_simple` | 205,650 | 241,787 | 0.85x | 0.8763x |
| **total** | **798,771** | **2,410,746** | | |

The last two columns are close but not equal, and should not be expected to be: repetition is
measured in **words**, and documents are not uniform in length. A source used fractionally
emits the first N% of its *words*, which is the first N% of its *documents* only if document
length is independent of position in the file. For `wikipedia_simple` (articles ordered by id,
lengths varying by orders of magnitude) that assumption is visibly wrong, and for `weird` —
55 books total — the granularity alone explains it.

The count survives tokenization exactly. `artifacts/tokens-v3/` holds 734,978 occurrences
of id 2 in `train_ids.npy` and 63,793 in `val_ids.npy`: **798,771** together, equal to the
number of `</s>` lines in `blend.txt` to the token. None were added, none were lost, and none
were split into ordinary subwords.

## A second count: tokenizing the finished file

The headline number above (399,508,203) is **not** the only token count this project has for
this corpus, and the other one was missing from this page until now — which is exactly how a
reviewer came to flag it as unverifiable. Recorded here so it stops being a number that only
exists in a training run's own header.

`train/tokenization.py` runs the retrained tokenizer once over the finished, concatenated
`artifacts/corpus/blend.txt` and splits the result into train/val arrays. Over the corpus as
rebuilt on 2026-08-14 (with document separators), written to `artifacts/tokens-v3/`:

| | Tokens |
|---|---:|
| Total | **391,921,555** |
| Train split | **352,729,403** |
| Val split | **39,192,152** |
| of which id 2 (`</s>`) | **798,771** |
| Vocabulary | 32,000 |

This is directly checkable on disk: `artifacts/tokens-v3/train_ids.npy` and
`artifacts/tokens-v3/val_ids.npy` are `uint32` arrays of shape `(352729403,)` and
`(39192152,)`. The split is stratified per source — see `_tokenize_stratified` — and the
per-source token counts it reports are in the run's `TokenStats`.

The older arrays are a different corpus and are kept. `artifacts/tokens/` (353,495,970 /
39,277,330, total 392,773,300) and `artifacts/tokens-stratified/` (353,495,973 train) were
tokenized from the blend as it stood *before* separators existed, and 392,773,300 is the
`corpus_tokens` recorded in the header of every `tt-tnt-v1` checkpoint (e.g.
`artifacts/checkpoints-tt-tnt-v1/tt_tnt_step00010787.pkl`). They are what the published model
actually trained on, which is why `tokenize_corpus` refuses to overwrite them and why the
rebuild went to a new directory instead. Both contain **zero** occurrences of id 2 — that is
the regression, still visible on disk.

Why this differs from the manifest's 399,508,203. They are two different measurements of
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
invisible in the other. The difference is **7,586,648 tokens, or 1.90%** of 399,508,203.

That magnitude is the point, and an earlier version of this paragraph got it wrong twice — it
quoted 0.42% and attributed the gap to the eight *source-to-source* seams. Eight seams cannot
produce 7.6M tokens; they would produce dozens. The real mechanism is that
`scripts/measure_corpus.py` splits each prepared file on `\n\n` and tokenizes **chunk by chunk**,
so every chunk boundary — millions of them, one per document or paragraph, not eight — is an
encode-time seam where a merge can differ. Same phenomenon, five orders of magnitude more of it,
which is why 1.90% is unremarkable rather than alarming. Neither count is wrong. **Treat
399,508,203 as the per-source provenance figure (how the blend was assembled) and
391,921,555/352,729,403/39,192,152 as the tokenized-training-data figure (what
`train/run.py` actually reads token-by-token)** — use whichever one answers the question being
asked, and do not average them or treat the smaller one as a correction to the larger one.

This is also where the step budget for one epoch comes from. The **train split**, not the
manifest total, is what a training run iterates over, and the step size follows the context
length:

- The published `tt-tnt-v1` run used `seq_len=512`, so one step consumed 64 × 512 = 32,768
  tokens, and 353,495,970 / 32,768 ≈ 10,787.98 — the **10,787 steps** that run reports.
- `tt-tnt-384` now declares `max_sequence_length: 2048`, so one step consumes
  64 × 2048 = 131,072 tokens and 352,729,403 / 131,072 ≈ 2,690.99, i.e. **2,690 steps** covers
  one epoch of `artifacts/tokens-v3`.

In both cases the final step is short by construction (2,690 × 131,072 = 352,583,680 is
145,723 tokens less than the split) — the standard "drop the final partial batch" behaviour,
not a bug.

### Real repetition is not the declared `upsample`

`upsample` in `train/corpus.py` is a **ceiling** — the most repetition a source is allowed to
carry, which is what the availability gate in `scripts/measure_corpus.py` checks against. The
repetition actually applied is `required_tokens / available_tokens`, and it is fractional.

Read the two columns together:

- **`procedural` 3.9109x against a 4x limit.** The tightest slice in the registry. Task 6 moved
  a whole share point (13% → 12%) to keep it there; this is the number that move was for.
- **`wikipedia_simple` 0.8763x.** Below 1.0: 88% of Simple Wikipedia is used once and the rest
  is not used at all. Nothing is duplicated.
- **`tinystories` 0.2768x and `poetry` 0.1305x.** Same thing, further out — these sources are
  large relative to their shares, so most of each file never enters the blend.
- **`flavour` 3.4759x.** The whole file, three and a half times over. That is deliberate (see
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
blend was then rebuilt at 14:48 by the Task 6 re-settle, rebuilt again by the fix described in
the previous section, and rebuilt once more on 2026-08-14 to add document separators. **This
is known and accepted, not an oversight.**

The separator rebuild does not deepen the problem, because it did not change the vocabulary's
relationship to the text: `</s>` was already in that vocabulary as a special token (id 2) —
it is what the *previous* corpus, `artifacts/corpus/corpus.txt`, used — so the tokenizer
encodes the new separators to exactly one id each, and no retrain is needed to represent them.

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

If you retrain it, re-run `scripts/measure_corpus.py` and `scripts/blend_corpus.py`
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
