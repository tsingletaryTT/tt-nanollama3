#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Held-out (or honestly-labelled in-sample) cross-entropy, per corpus source.

THE PROBLEM THIS MEASURES
--------------------------
``train/tokenization.py``'s validation split is the last ``val_fraction`` of the WHOLE
token stream. ``scripts/blend_corpus.py`` writes the nine corpus sources concatenated in
**sorted name order**, and ``wikipedia_simple`` sorts last -- so on the shipped blend, the
entire validation split (``artifacts/tokens/val_ids.npy``) falls inside ``wikipedia_simple``
(15% of the training mixture). Every validation loss ever reported for this model has
therefore been measuring domain transfer (narrative/naturalist prose -> encyclopedic
Wikipedia text), not learning progress on the actual training mixture.

This script reports cross-entropy loss separately for each of the nine sources, so it is
possible to see which sources the model actually fits and which it does not -- the question
that decides whether the next move is a longer/decayed run, a corpus rebalance, or a bigger
model.

METHOD
------
1. Recompute each source's EXACT span in the token stream by re-tokenizing
   ``artifacts/corpus/blend.txt`` with the same line-batched, no-special-tokens procedure
   ``train/tokenization.py`` uses (``train.tokenization.encode_batch``), using
   ``blend_manifest.json``'s ``emitted_words`` (an exact count from the actual corpus
   assembly, not an estimate) to locate each source's line range. See
   :func:`tokenize_blend_by_source`. This is verified, not assumed: the result is checked
   against ``len(train_ids) + len(val_ids)`` and against ``val_ids.npy`` directly (see
   :func:`verify_against_disk_arrays`) before anything is reported.

   ``blend_manifest.json`` ALSO records an approximate ``emitted_tokens`` per source
   (paragraph-chunked, see ``blend_corpus.TokenMeter``), which is measurably different from
   what line-based tokenization actually produces (on the shipped blend: 399,594,747
   manifest tokens vs. 392,773,300 real train+val tokens, a ~1.7% gap that is NOT uniform
   across sources). This script never uses that number for boundaries -- only the exact
   ``emitted_words`` count, replayed through the real tokenizer in the real chunking.

2. A source's raw file was fully consumed by the blend (``repetition_factor >= 1`` in the
   manifest) if the blend contains at least one complete pass over it. For such a source,
   EVERY word of its raw file was shown to the model at least once -- there is no unseen
   text left to hold out, so this script reports only an in-sample TRAIN loss for it,
   labelled as such.

   A source with ``repetition_factor < 1`` (poetry, tinystories, wikipedia_simple) supplied
   only a PREFIX of its raw file; the remaining tail was never emitted into the corpus and
   so never seen in training -- genuinely held-out data, sampled directly from the raw
   source file (see :func:`unseen_tail_ids`). wikipedia_simple is a special case of this:
   its unseen tail already exists as ``artifacts/tokens/val_ids.npy`` (the pipeline's own
   validation split, which is entirely wikipedia_simple -- see the module docstring above),
   so this script uses that directly rather than re-deriving a second unseen slice.

3. Every source ALSO gets an in-sample TRAIN loss, sampled from its own span within
   ``artifacts/tokens/train_ids.npy`` -- so the report can show, source by source, how much
   (if any) train/held-out gap exists, not just an absolute number.

4. Cross-entropy matches ``tests/test_hf_parity.py``'s method: forward the HF model, and
   compute loss directly against ``logits`` (never through the ``labels=`` kwarg, which
   shifts a second time -- see that test's docstring). Standard error is computed over
   PER-WINDOW mean losses (the natural exchangeable sampling unit here), not over
   per-token losses, which would understate the real noise from within-window
   autocorrelation.

CONSTRAINTS THIS SCRIPT RESPECTS
---------------------------------
CPU only. Never imports ttml/ttnn and never opens a Tenstorrent device -- if the HF model
directory is unavailable, this script fails with an explanation rather than falling back
to the ttml/device path.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES  # noqa: E402
from train.tokenization import TOKEN_DTYPE, encode_batch  # noqa: E402

#: Sequence length windows are sampled at. Matches the model's trained/context length
#: (train/sizes.py's 384-size max_sequence_length) so a per-source loss is comparable to
#: the training run's own numbers, not an apples-to-oranges context length.
DEFAULT_SEQ_LEN = 512

#: Windows sampled per (source, condition) pair. 256 windows of 512 tokens is ~131,000
#: tokens of signal per number -- comfortably more than the 10-batches-of-32
#: (~163,840 tokens, see train/run.py's evaluate()) that the investigation flagged as
#: carrying ~1 standard error of noise at this project's loss scale, while still running
#: in a couple of minutes on CPU for a 22M-parameter model.
DEFAULT_N_WINDOWS = 256

#: Forward-pass batch size. Purely a memory/throughput knob -- does not change which
#: windows are sampled or their losses.
DEFAULT_BATCH_SIZE = 32

#: Words sampled from an unused source's raw-file tail for a held-out measurement. Far more
#: than DEFAULT_N_WINDOWS * DEFAULT_SEQ_LEN tokens' worth, so window sampling from it has
#: real spread rather than being forced to reuse the same few thousand tokens repeatedly.
DEFAULT_TAIL_WORD_CAP = 3_000_000

#: Corpus-assembly chunking knob, matching train/tokenization.py's default. A memory knob
#: only -- see encode_batch's docstring for why chunk size cannot change the tokens
#: produced.
DEFAULT_CHUNK_LINES = 50_000


# ---------------------------------------------------------------------------------------
# Manifest and exact per-source token spans
# ---------------------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"blend manifest not found at {path}; this script needs it to locate each "
            f"source's span in the corpus (see docs/measurements/blend_manifest.json)"
        )
    return json.loads(path.read_text())


def emission_order(manifest: dict) -> List[str]:
    """Source names in the order ``scripts/blend_corpus.py`` actually wrote them.

    ``blend_corpus.py``'s ``main()`` loops ``for name in sorted(plan)`` -- sorted-name
    order, not registry order -- so this must match that exactly, not ``train.corpus``'s
    dict order or any other convenient ordering.
    """
    return sorted(manifest["sources"])


def emitted_words(manifest: dict) -> Dict[str, int]:
    return {name: rec["emitted_words"] for name, rec in manifest["sources"].items()}


def tokenize_blend_by_source(
    blend_path: Path,
    order: List[str],
    words_by_source: Dict[str, int],
    tok,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
) -> Dict[str, np.ndarray]:
    """Re-tokenize ``blend_path`` once, splitting the resulting ids by source.

    Walks the corpus a single time, batching lines for the tokenizer exactly like
    ``train.tokenization.tokenize_corpus`` does, but flushes the batch early -- and starts
    a new one -- every time the running word count crosses a source's declared
    ``words_by_source[name]``. Flushing early cannot change any line's tokens (see
    ``encode_batch``'s docstring: each line is encoded independently of its neighbours), so
    the concatenation of every source's returned array, in ``order``, is BYTE-IDENTICAL to
    tokenizing the whole file in one pass -- this is what makes the source boundaries this
    function locates trustworthy against ``artifacts/tokens/{train,val}_ids.npy``, which
    were produced by exactly that one-pass tokenization.

    The boundary check is exact equality (``words_seen == target``), not ``>=``: unlike a
    raw (pre-blend) source file, ``blend.txt`` is guaranteed by ``scripts/blend_corpus.py``'s
    own emission logic to end each source's contribution on a complete line, at EXACTLY
    that source's ``emitted_words`` (``_emit`` truncates mid-line when necessary and writes
    only the truncated line, so the line as it exists IN ``blend.txt`` is always whole).
    Anything other than an exact hit means ``words_by_source`` does not describe this exact
    ``blend_path``, and this function raises rather than silently mis-drawing the boundary.
    """
    if not order:
        raise ValueError("order is empty; nothing to tokenize")

    per_source_pieces: Dict[str, List[np.ndarray]] = {name: [] for name in order}
    batch: List[str] = []

    idx = 0
    current: Optional[str] = order[0]
    target = words_by_source[current]
    words_seen = 0

    def flush() -> None:
        nonlocal batch
        if current is not None and batch:
            per_source_pieces[current].append(encode_batch(batch, tok))
        batch = []

    with blend_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if current is None:
                raise ValueError(
                    f"{blend_path} has more lines than the sources in {order} account for "
                    f"({sum(words_by_source.values()):,} words declared total) -- the "
                    f"manifest does not describe this corpus file"
                )
            stripped = line.rstrip("\n")
            batch.append(stripped)
            words_seen += len(stripped.split())
            if words_seen == target:
                flush()
                idx += 1
                current = order[idx] if idx < len(order) else None
                target = words_by_source[current] if current is not None else 0
                words_seen = 0
            elif words_seen > target:
                raise ValueError(
                    f"source {current!r} overshot its declared {target:,} words in "
                    f"{blend_path} (observed {words_seen:,} at a line boundary) -- "
                    f"words_by_source does not match how this corpus file was actually "
                    f"assembled; re-check blend_manifest.json against blend.txt"
                )
            elif len(batch) >= chunk_lines:
                flush()
    flush()

    if current is not None:
        raise ValueError(
            f"{blend_path} ended before source {current!r} reached its declared "
            f"{target:,} words (only {words_seen:,} seen) -- words_by_source does not "
            f"match this corpus file"
        )

    return {
        name: (np.concatenate(pieces) if pieces else np.zeros(0, dtype=TOKEN_DTYPE))
        for name, pieces in per_source_pieces.items()
    }


def find_word_boundary_line(path: Path, target_words: int) -> int:
    """0-based index of the first line whose cumulative word count reaches or exceeds
    ``target_words`` in a RAW (pre-blend) source file.

    Unlike ``tokenize_blend_by_source``'s boundary in ``blend.txt`` (always exact, always a
    complete line -- see its docstring), the corresponding word offset in the RAW source
    file need not fall on a line boundary at all: ``scripts/blend_corpus.py``'s ``_emit``
    truncates a source's FINAL emitted line to ``parts[:need]`` words when the target is
    reached mid-line, silently discarding the rest of that raw line rather than emitting it
    -- so the raw file's own line at this position may be split between "partially
    emitted" and "never emitted". ``>=`` finds that line; callers wanting a clean, entirely
    fresh tail should start reading at the line AFTER this one (see
    :func:`unseen_tail_ids`), never at this boundary line itself.
    """
    words = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            words += len(line.split())
            if words >= target_words:
                return i
    raise ValueError(
        f"{path} has fewer than {target_words:,} words; cannot locate the boundary this "
        f"source's blend emission stopped at"
    )


def unseen_tail_ids(
    raw_path: Path,
    emitted_words_for_source: int,
    tok,
    word_cap: int = DEFAULT_TAIL_WORD_CAP,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
) -> Tuple[np.ndarray, int]:
    """Tokenize a sample from the tail of ``raw_path`` beyond what the blend emitted.

    Only meaningful for a source whose ``repetition_factor < 1`` (the blend used a prefix
    of the raw file, never wrapping around to repeat it) -- see the module docstring.
    Returns ``(ids, words_sampled)``; ``words_sampled`` can be less than ``word_cap`` if the
    raw file's tail is shorter than the cap.
    """
    boundary_line = find_word_boundary_line(raw_path, emitted_words_for_source)
    pieces: List[np.ndarray] = []
    batch: List[str] = []
    words_sampled = 0

    def flush() -> None:
        nonlocal batch
        if batch:
            pieces.append(encode_batch(batch, tok))
        batch = []

    with raw_path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i <= boundary_line:
                # This line may contain emitted (trained-on) words -- see
                # find_word_boundary_line's docstring on why it, not just lines strictly
                # before it, must be excluded from a "never seen" claim.
                continue
            if words_sampled >= word_cap:
                break
            stripped = line.rstrip("\n")
            batch.append(stripped)
            words_sampled += len(stripped.split())
            if len(batch) >= chunk_lines:
                flush()
    flush()

    ids = np.concatenate(pieces) if pieces else np.zeros(0, dtype=TOKEN_DTYPE)
    return ids, words_sampled


def verify_against_disk_arrays(
    per_source_ids: Dict[str, np.ndarray],
    order: List[str],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
) -> Dict[str, Tuple[int, int]]:
    """Cross-check the reconstruction against the arrays actually used for training.

    Raises ``ValueError`` (never proceeds silently) unless:

    1. Concatenating every source's array, in ``order``, has exactly
       ``len(train_ids) + len(val_ids)`` tokens.
    2. Exactly the sources whose span crosses the ``len(train_ids)`` boundary have a
       nonempty overlap with ``val_ids``, and for a SINGLE straddling source, its own
       val-side slice is byte-identical to ``val_ids`` -- the direct, mechanical
       confirmation of "the validation split is entirely inside this one source".

    Returns ``{name: (train_token_count, val_token_count)}`` for each source -- the exact
    split point this script uses to keep "in-sample train" and "held-out val" measurements
    from ever touching the same tokens for a given source.
    """
    total = sum(len(a) for a in per_source_ids.values())
    if total != len(train_ids) + len(val_ids):
        raise ValueError(
            f"reconstructed {total:,} tokens across all sources, but "
            f"train_ids ({len(train_ids):,}) + val_ids ({len(val_ids):,}) = "
            f"{len(train_ids) + len(val_ids):,} -- the per-source reconstruction does not "
            f"match the token arrays actually used for training; do not trust any number "
            f"downstream of this until the mismatch is understood"
        )

    boundary = len(train_ids)
    offset = 0
    split: Dict[str, Tuple[int, int]] = {}
    straddlers: List[str] = []
    for name in order:
        arr = per_source_ids[name]
        start, end = offset, offset + len(arr)
        train_part = max(0, min(end, boundary) - start)
        val_part = len(arr) - train_part
        split[name] = (train_part, val_part)
        if train_part > 0 and val_part > 0:
            straddlers.append(name)
        offset = end

    if len(straddlers) > 1:
        raise ValueError(
            f"more than one source straddles the train/val boundary ({straddlers}) -- "
            f"expected at most one, since sources occupy contiguous, non-overlapping spans"
        )
    if straddlers:
        name = straddlers[0]
        start = sum(len(per_source_ids[n]) for n in order[:order.index(name)])
        val_part_ids = per_source_ids[name][boundary - start:]
        if not np.array_equal(val_part_ids, val_ids):
            raise ValueError(
                f"{name}'s reconstructed val-side slice does not match val_ids.npy "
                f"byte-for-byte -- the source-boundary reconstruction is wrong; do not "
                f"trust the per-source split downstream of this"
            )

    return split


# ---------------------------------------------------------------------------------------
# Windowed cross-entropy
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LossEstimate:
    mean: float
    sem: float
    n_windows: int
    n_tokens_available: int


def sample_windows(ids: np.ndarray, seq_len: int, n_windows: int,
                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """``n_windows`` random (with-replacement) ``(x, y)`` windows of length ``seq_len``.

    Matches ``tests/test_hf_parity.py::test_validation_loss_matches_the_training_run``'s
    sampling exactly (uniform random start positions, with replacement) so numbers from
    this script are comparable to that gate's methodology, not a different one.
    """
    n = len(ids) - seq_len - 1
    if n <= 0:
        raise ValueError(
            f"only {len(ids):,} tokens available, need more than {seq_len + 1:,} to "
            f"sample even one window of length {seq_len}"
        )
    ix = rng.integers(0, n, size=(n_windows,))
    x = np.stack([ids[i:i + seq_len] for i in ix], axis=0).astype("int64")
    y = np.stack([ids[i + 1:i + seq_len + 1] for i in ix], axis=0).astype("int64")
    return x, y


def per_window_losses(model, x: np.ndarray, y: np.ndarray, batch_size: int) -> List[float]:
    """Cross-entropy for each window in ``x``/``y``, one float per row.

    No ``labels=`` kwarg: ``LlamaForCausalLM``'s own loss function shifts labels
    internally, so passing already-aligned next-token labels through it would shift a
    SECOND time and report a near-uniform-ceiling number regardless of model quality --
    see ``tests/test_hf_parity.py``'s docstring for how this exact mistake produced a
    passing-looking but meaningless number before. Computing directly against ``logits``
    avoids that entirely.
    """
    import torch

    losses: List[float] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size])
            yb = torch.from_numpy(y[start:start + batch_size])
            logits = model(xb).logits.float()
            per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="none"
            ).reshape(yb.shape)
            losses.extend(per_token.mean(dim=1).tolist())
    return losses


def mean_and_sem(losses: List[float]) -> Tuple[float, float]:
    """Mean and standard error of the mean over per-window losses.

    ``ddof=1`` (sample standard deviation): these are a sample of windows drawn from a
    source, not the full population of every possible window. SEM is 0.0 (not NaN, and not
    silently skipped) when fewer than 2 windows are available -- no spread can be measured
    from one sample, which the report should show as an explicit "n=1" rather than crash.
    """
    arr = np.asarray(losses, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, 0.0
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean, sem


def evaluate_ids(model, ids: np.ndarray, seq_len: int, n_windows: int, batch_size: int,
                 rng: np.random.Generator) -> LossEstimate:
    x, y = sample_windows(ids, seq_len, n_windows, rng)
    losses = per_window_losses(model, x, y, batch_size)
    mean, sem = mean_and_sem(losses)
    return LossEstimate(mean=mean, sem=sem, n_windows=len(losses),
                        n_tokens_available=len(ids))


# ---------------------------------------------------------------------------------------
# Per-source report assembly
# ---------------------------------------------------------------------------------------


@dataclass
class SourceReport:
    name: str
    slice_name: str
    target_share: float
    repetition_factor: float
    train_tokens: int
    train_share_of_train_split: float
    train_sample: LossEstimate
    held_out: Optional[LossEstimate] = None
    held_out_kind: str = ""
    note: str = ""


def build_reports(
    manifest: dict,
    per_source_ids: Dict[str, np.ndarray],
    split: Dict[str, Tuple[int, int]],
    order: List[str],
    val_ids: np.ndarray,
    corpus_dir: Path,
    tok,
    model,
    *,
    seq_len: int,
    n_windows: int,
    batch_size: int,
    tail_word_cap: int,
    seed: int,
) -> List[SourceReport]:
    total_train_tokens = sum(t for t, _ in split.values())
    reports: List[SourceReport] = []
    # A fresh rng per source (seeded off the base seed + a stable per-source offset) so
    # results are reproducible regardless of what order sources are processed in, and so
    # one source's sampling can never perturb another's.
    for i, name in enumerate(order):
        rec = manifest["sources"][name]
        repetition_factor = float(rec["repetition_factor"])
        train_tokens, val_tokens = split[name]
        train_ids_for_source = per_source_ids[name][:train_tokens]

        rng = np.random.default_rng(seed + 1000 * i)
        train_sample = evaluate_ids(model, train_ids_for_source, seq_len, n_windows,
                                    batch_size, rng)

        held_out: Optional[LossEstimate] = None
        held_out_kind = ""
        note = ""
        if repetition_factor >= 1.0:
            note = (
                "no unseen tail exists: the blend contains at least one complete pass "
                "over this source's entire raw file, so every word was shown to the "
                "model at least once"
            )
        elif val_tokens > 0:
            # The straddling source (wikipedia_simple on the shipped blend) -- its
            # held-out slice IS val_ids, already verified byte-identical in
            # verify_against_disk_arrays.
            held_out_rng = np.random.default_rng(seed + 1000 * i + 1)
            held_out = evaluate_ids(model, val_ids, seq_len, n_windows, batch_size,
                                    held_out_rng)
            held_out_kind = (
                "artifacts/tokens/val_ids.npy (the training run's own held-out split; "
                "confirmed above to be entirely this source)"
            )
        else:
            raw_path = corpus_dir / f"{name}.txt"
            tail_ids, words_sampled = unseen_tail_ids(
                raw_path, rec["emitted_words"], tok, word_cap=tail_word_cap)
            held_out_rng = np.random.default_rng(seed + 1000 * i + 1)
            held_out = evaluate_ids(model, tail_ids, seq_len, n_windows, batch_size,
                                    held_out_rng)
            held_out_kind = (
                f"unused tail of artifacts/corpus/{name}.txt beyond word "
                f"{rec['emitted_words']:,} ({words_sampled:,} words sampled, never "
                f"included in any training corpus)"
            )

        reports.append(SourceReport(
            name=name,
            slice_name=SOURCES[name].slice if name in SOURCES else "?",
            target_share=float(rec["target_share"]),
            repetition_factor=repetition_factor,
            train_tokens=train_tokens,
            train_share_of_train_split=(train_tokens / total_train_tokens
                                        if total_train_tokens else 0.0),
            train_sample=train_sample,
            held_out=held_out,
            held_out_kind=held_out_kind,
            note=note,
        ))
    return reports


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


def _fmt_loss(est: Optional[LossEstimate]) -> str:
    if est is None:
        return "n/a"
    return f"{est.mean:.4f} ± {est.sem:.4f} (n={est.n_windows})"


def render_markdown(reports: List[SourceReport], *, checkpoint_step: int,
                    hf_model: Path, seq_len: int, headline_val_loss: float) -> str:
    lines: List[str] = []
    lines.append("<!-- SPDX-License-Identifier: Apache-2.0 -->")
    lines.append("<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->")
    lines.append("")
    lines.append("# Per-source loss — tt-tnt-v1")
    lines.append("")
    lines.append(
        f"Checkpoint step {checkpoint_step:,} (the final checkpoint of the tt-tnt-v1 run), "
        f"evaluated via the converted CPU model at `{hf_model}`. Sequence length "
        f"{seq_len}, generated by `scripts/eval_per_source.py`."
    )
    lines.append("")
    lines.append(
        "## Why this exists"
    )
    lines.append("")
    lines.append(
        f"The model's one reported validation loss (**{headline_val_loss}**) is measured "
        f"entirely on `wikipedia_simple`: `scripts/blend_corpus.py` writes the nine corpus "
        f"sources concatenated in sorted-name order, `wikipedia_simple` sorts last, and "
        f"`train/tokenization.py`'s validation split is the tail 10% of the whole token "
        f"stream -- which lands entirely inside it. That number describes domain transfer "
        f"(narrative/naturalist prose -> encyclopedic text), not learning progress on the "
        f"training mixture as a whole. This table is the fix: loss broken out per source."
    )
    lines.append("")
    lines.append("## Method, in brief")
    lines.append("")
    lines.append(
        "- **train-sample loss**: sampled from the source's own span of "
        "`artifacts/tokens/train_ids.npy` -- **in-distribution training data**, not "
        "held out. Reported for every source so the table has a consistent baseline, "
        "never confused with the held-out column."
    )
    lines.append(
        "- **held-out loss**: only reported where genuinely unseen text exists for that "
        "source (`repetition_factor < 1` in `blend_manifest.json`, meaning the blend used "
        "only a prefix of the source's raw file). `wikipedia_simple` uses the existing "
        "`val_ids.npy` split; `tinystories` and `poetry` use a sample from the unused tail "
        "of their own raw corpus file, confirmed never emitted into the training corpus. "
        "The other six sources (`flavour`, `folklore`, `gutenberg_children`, `procedural`, "
        "`spine`, `weird`) were each fully consumed by the blend at least once, so no "
        "unseen slice of them exists -- their held-out column is `n/a`, not a number "
        "standing in for one."
    )
    lines.append(
        "- Both columns report **mean ± standard error over independent windows**, not a "
        "bare mean -- see each source's `n=` for how many windows the estimate rests on."
    )
    lines.append("")
    lines.append("## Per-source loss")
    lines.append("")
    lines.append(
        "| source | slice | target share | train tokens | share of train | repetition | "
        "train-sample loss | held-out loss | held-out is |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|")
    for r in reports:
        held_out_kind = r.held_out_kind if r.held_out is not None else r.note
        lines.append(
            f"| {r.name} | {r.slice_name} | {r.target_share * 100:.3g}% | "
            f"{r.train_tokens:,} | {r.train_share_of_train_split * 100:.2f}% | "
            f"{r.repetition_factor:.3g}x | {_fmt_loss(r.train_sample)} | "
            f"{_fmt_loss(r.held_out)} | {held_out_kind} |"
        )
    lines.append("")
    return "\n".join(lines)


def reports_to_json(reports: List[SourceReport], *, checkpoint_step: int,
                    hf_model: str, seq_len: int) -> dict:
    def _est(e: Optional[LossEstimate]) -> Optional[dict]:
        if e is None:
            return None
        return {"mean": e.mean, "sem": e.sem, "n_windows": e.n_windows,
                "n_tokens_available": e.n_tokens_available}

    return {
        "checkpoint_step": checkpoint_step,
        "hf_model": hf_model,
        "seq_len": seq_len,
        "sources": {
            r.name: {
                "slice": r.slice_name,
                "target_share": r.target_share,
                "repetition_factor": r.repetition_factor,
                "train_tokens": r.train_tokens,
                "train_share_of_train_split": r.train_share_of_train_split,
                "train_sample_loss": _est(r.train_sample),
                "held_out_loss": _est(r.held_out),
                "held_out_kind": r.held_out_kind,
                "note": r.note,
            }
            for r in reports
        },
    }


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", type=Path, default=ROOT / "artifacts" / "hf-tt-tnt-v1",
                   help="Converted HF model directory to evaluate (CPU only, default: "
                        "%(default)s).")
    p.add_argument("--blend", type=Path, default=ROOT / "artifacts" / "corpus" / "blend.txt")
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "docs" / "measurements" / "blend_manifest.json")
    p.add_argument("--tokens-dir", type=Path, default=ROOT / "artifacts" / "tokens")
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "artifacts" / "corpus")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS,
                   help="Windows sampled per (source, condition) pair (default: "
                        "%(default)s).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--tail-word-cap", type=int, default=DEFAULT_TAIL_WORD_CAP)
    p.add_argument("--chunk-lines", type=int, default=DEFAULT_CHUNK_LINES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-step", type=int, default=10787,
                   help="Recorded in the report for provenance only (default: %(default)s, "
                        "tt-tnt-v1's final checkpoint step).")
    p.add_argument("--headline-val-loss", type=float, default=4.2203,
                   help="The model card's reported validation loss, quoted in the report's "
                        "explanation (default: %(default)s).")
    p.add_argument("--out", type=Path,
                   default=ROOT / "docs" / "measurements" / "per-source-loss-tt-tnt-v1.md")
    p.add_argument("--json-out", type=Path,
                   default=ROOT / "docs" / "measurements" / "per-source-loss-tt-tnt-v1.json")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()

    if not (args.hf_model / "config.json").is_file():
        print(f"ERROR: no converted model at {args.hf_model} "
              f"(config.json missing). Run scripts/convert_checkpoint.py first, or point "
              f"--hf-model at an existing converted directory. This script is CPU-only "
              f"and deliberately does not fall back to the ttml/device path.",
              file=sys.stderr)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading manifest {args.manifest} ...")
    manifest = load_manifest(args.manifest)
    order = emission_order(manifest)
    words_by_source = emitted_words(manifest)
    print(f"  {len(order)} sources, emission order: {order}")

    print(f"loading tokenizer from {args.hf_model} ...")
    tok = AutoTokenizer.from_pretrained(str(args.hf_model))

    print(f"re-tokenizing {args.blend} by source (this replays the entire blend through "
          f"the tokenizer once; it can take several minutes) ...")
    per_source_ids = tokenize_blend_by_source(args.blend, order, words_by_source, tok,
                                              chunk_lines=args.chunk_lines)
    for name in order:
        print(f"    {name:22} {len(per_source_ids[name]):>13,} tokens")

    train_path = args.tokens_dir / "train_ids.npy"
    val_path = args.tokens_dir / "val_ids.npy"
    if not train_path.is_file() or not val_path.is_file():
        print(f"ERROR: {train_path} / {val_path} not found.", file=sys.stderr)
        return 1
    train_ids = np.load(train_path)
    val_ids = np.load(val_path)
    print(f"  train_ids={len(train_ids):,} val_ids={len(val_ids):,}")

    print("verifying the reconstruction against the arrays the model was actually "
          "trained on ...")
    split = verify_against_disk_arrays(per_source_ids, order, train_ids, val_ids)
    for name in order:
        t, v = split[name]
        if v:
            print(f"    {name:22} train={t:>13,} val={v:>13,}  <- straddles the boundary")
    print("  OK: token counts and the straddling source's val slice both check out.")

    print(f"loading model {args.hf_model} ...")
    model = AutoModelForCausalLM.from_pretrained(str(args.hf_model)).eval()

    print(f"evaluating {args.n_windows} windows per (source, condition) ...")
    reports = build_reports(
        manifest, per_source_ids, split, order, val_ids, args.corpus_dir, tok, model,
        seq_len=args.seq_len, n_windows=args.n_windows, batch_size=args.batch_size,
        tail_word_cap=args.tail_word_cap, seed=args.seed,
    )
    for r in reports:
        print(f"    {r.name:22} train-sample={_fmt_loss(r.train_sample)}  "
              f"held-out={_fmt_loss(r.held_out)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(reports, checkpoint_step=args.checkpoint_step,
                         hf_model=args.hf_model, seq_len=args.seq_len,
                         headline_val_loss=args.headline_val_loss)
    args.out.write_text(md)
    print(f"wrote {args.out}")

    if args.json_out:
        payload = reports_to_json(reports, checkpoint_step=args.checkpoint_step,
                                  hf_model=str(args.hf_model), seq_len=args.seq_len)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
