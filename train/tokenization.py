# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Turn a prepared corpus into token-id arrays on disk.

tt-train's own ``prepare_data`` (``ttml/common/data.py:82``) encodes the whole corpus in
one ``encode(text)`` call and splits 90/10 in memory. Against our 536 MB corpus that is a
single enormous tokenizer invocation and a multi-gigabyte intermediate Python list, so we
encode in line batches instead and grow a numpy array.

Chunk size is a **memory knob, not a correctness knob** — tokenizing the same corpus with
any two chunk sizes must produce byte-identical output, which the tests assert.

No ttnn/ttml imports: this runs on any machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

#: What ``ttml.common.trainer.get_batch_ttml`` expects to receive.
TOKEN_DTYPE = np.uint32


class TokenArtifactExistsError(FileExistsError):
    """Raised when ``tokenize_corpus`` would silently overwrite existing token arrays.

    ``train_ids.npy``/``val_ids.npy`` are not disposable intermediates: a parity gate that
    trusts them (``tests/test_hf_parity.py``, ``tests/test_ttml_forward.py``) only means
    anything if the tokens it scores against are the exact ones a model was trained and
    held out on. A different corpus or a retrained tokenizer produces numerically different
    ids from the same filenames, with nothing on disk to tell the two apart -- that silent
    swap is exactly what invalidated the v2 model's parity gate (see CLAUDE.md's
    ``parity-gate-restore`` entry). Refusing by default forces whoever is about to
    regenerate these files to notice that something depends on the current ones and make
    an explicit choice: copy them aside first, write to a different ``--out``, or pass
    ``--force``/``overwrite=True`` because they've confirmed nothing does.
    """


@dataclass
class TokenStats:
    """What ``tokenize_corpus`` produced, for logging and the model card."""

    total_tokens: int
    train_tokens: int
    val_tokens: int
    vocab_size: int
    #: Per-source ``{"train_tokens": ..., "val_tokens": ..., "total_tokens": ...}`` when the
    #: STRATIFIED split ran (``source_word_counts`` was given to :func:`tokenize_corpus`),
    #: else ``None``. See :func:`_tokenize_stratified` for what "stratified" means here.
    source_splits: Optional[Dict[str, Dict[str, int]]] = None


def encode_batch(batch: List[str], tok) -> np.ndarray:
    """Tokenize a batch of already newline-stripped lines into one flat id array.

    Shared by both split strategies in this module so they can never tokenize the same
    text two different ways.

    ``add_special_tokens=False``. The old justification for this flag -- "the corpus
    already carries its own ``</s>`` separators, so letting the tokenizer inject more would
    double them at every story boundary" -- described the TinyStories-only pipeline
    (``train/data.py``) and was false for the nine-source one for as long as that pipeline
    emitted no separators at all. Both halves of it were wrong, and the second half is wrong
    even now that ``scripts/prepare_corpus.py`` writes a ``DOCUMENT_SEPARATOR`` line after
    every document: ``artifacts/tokenizer``'s post-processor is a plain ``ByteLevel``, not a
    ``TemplateProcessing``, so ``add_special_tokens=True`` would inject nothing and the two
    settings produce identical ids today (measured, not assumed).

    The flag is kept ``False`` for a different and still-live reason: **this function is
    called once per LINE, not once per document.** The corpus is encoded line by line (see
    the loops below), so if the tokenizer ever did gain a template post-processor -- a
    retrain, a config edit, a newer ``tokenizers`` default -- ``True`` would wrap every
    single line of the corpus in bos/eos rather than every document, burying the real
    document boundaries under millions of spurious ones. ``False`` makes the separators the
    corpus text carries the ONLY source of eos in the token stream, which is exactly the
    property the position-wise-loss fix depends on.

    The separators themselves survive this flag: ``</s>`` is an *added token* in
    ``tokenizer.json``, and added tokens are matched in the input text before
    pre-tokenization regardless of ``add_special_tokens``, so a line holding exactly
    ``"</s>"`` encodes to exactly ``[2]``.

    Each line is encoded independently of its neighbours (the HF tokenizer treats each
    element of ``batch`` as its own sequence), so grouping lines into batches of any size,
    or splitting a batch at an arbitrary line, never changes the ids a given line produces
    -- ``tests/test_tokenization.py::test_chunking_does_not_change_output`` is what pins
    this, and it is exactly what makes the per-source boundary logic in
    :func:`_tokenize_stratified` safe: flushing a batch early, right at a source boundary,
    cannot perturb the tokens on either side of that boundary.
    """
    if not batch:
        return np.zeros(0, dtype=TOKEN_DTYPE)
    encoded = tok(batch, add_special_tokens=False)["input_ids"]
    flat = [i for seq in encoded for i in seq]
    return np.asarray(flat, dtype=TOKEN_DTYPE) if flat else np.zeros(0, dtype=TOKEN_DTYPE)


def _tokenize_stratified(
    corpus: Path,
    tok,
    source_word_counts: Dict[str, int],
    val_fraction: float,
    chunk_lines: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, int]]]:
    """Encode ``corpus`` and split off a proportional tail from EACH source's own span.

    THE BUG THIS EXISTS TO FIX. The default tail-of-the-whole-stream split (see
    :func:`tokenize_corpus`) assumes the corpus is a homogeneous mixture. It is not:
    ``scripts/blend_corpus.py`` writes sources concatenated in sorted-name order, so the
    tail of the stream is just the tail of whichever source sorts last. For the shipped
    nine-source blend that source is ``wikipedia_simple`` (15% of the blend), and it is
    large enough that the entire validation split lands inside it -- the model's reported
    validation loss has been measuring domain transfer (Wikipedia after training mostly on
    narrative prose), not learning progress on the actual training mixture.

    ``source_word_counts`` maps source name -> exactly how many whitespace-delimited words
    of ``corpus`` that source contributed (``blend_manifest.json``'s ``emitted_words`` --
    the number ``scripts/blend_corpus.py`` itself produced while writing the corpus, so it
    is exact, not estimated). Sources are visited in **sorted name order**, matching
    ``blend_corpus.py``'s own ``for name in sorted(plan)`` -- the same order the corpus was
    actually written in. This function walks the corpus once, counting words the same way
    ``scripts/blend_corpus.py``'s ``_emit``/``_count_words`` do (``str.split()``), and
    flushes the tokenizer batch early whenever a source's word target is reached -- so a
    source's contribution to the token stream is measured directly, from the exact bytes
    it contributed, never estimated or back-computed from an approximate ratio.

    Held out per source is the LAST ``val_fraction`` of THAT source's own tokens -- the
    same tail convention as the whole-stream split, just applied per source instead of
    once globally. The final train/val arrays are the per-source train/val slices
    concatenated in the same sorted-name order, so token order within each split still
    matches corpus order.

    Raises ``ValueError`` (never silently mis-splits) if ``source_word_counts`` doesn't
    match the corpus: overshoot (a source's word count is reached mid-line where more text
    for it was expected), undershoot (the corpus runs out of lines before every declared
    source is satisfied), or leftover corpus lines after every declared source is
    satisfied all mean the caller's word counts do not describe THIS corpus file.
    """
    order = sorted(source_word_counts)
    per_source_pieces: Dict[str, List[np.ndarray]] = {name: [] for name in order}
    batch: List[str] = []

    idx = 0
    current: Optional[str] = order[0] if order else None
    target = source_word_counts[current] if current is not None else 0
    words_seen = 0

    def flush() -> None:
        nonlocal batch
        if current is not None and batch:
            per_source_pieces[current].append(encode_batch(batch, tok))
        batch = []

    with corpus.open("r", encoding="utf-8") as fin:
        for line in fin:
            if current is None:
                raise ValueError(
                    "corpus has more lines than source_word_counts accounts for -- every "
                    "declared source already reached its word target, but the corpus "
                    "continues. source_word_counts does not describe this corpus file "
                    "(e.g. it may be stale against a regenerated blend)."
                )
            stripped = line.rstrip("\n")
            batch.append(stripped)
            words_seen += len(stripped.split())
            if words_seen >= target:
                if words_seen > target:
                    raise ValueError(
                        f"source {current!r} overshot its declared word count while "
                        f"tokenizing {corpus}: expected exactly {target:,} words at the "
                        f"boundary, observed {words_seen:,}. source_word_counts must come "
                        f"from the manifest that describes exactly how THIS corpus file "
                        f"was assembled (blend_manifest.json's emitted_words), or the "
                        f"per-source boundaries it implies are wrong."
                    )
                flush()
                idx += 1
                current = order[idx] if idx < len(order) else None
                target = source_word_counts[current] if current is not None else 0
                words_seen = 0
            elif len(batch) >= chunk_lines:
                flush()
    flush()

    if current is not None:
        raise ValueError(
            f"corpus ended before source {current!r} reached its declared "
            f"{target:,} words (only {words_seen:,} seen); source_word_counts does not "
            f"match this corpus file"
        )

    train_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []
    source_splits: Dict[str, Dict[str, int]] = {}
    for name in order:
        pieces = per_source_pieces[name]
        arr = np.concatenate(pieces) if pieces else np.zeros(0, dtype=TOKEN_DTYPE)
        n_val = int(len(arr) * val_fraction)
        split_point = len(arr) - n_val
        train_chunks.append(arr[:split_point])
        val_chunks.append(arr[split_point:])
        source_splits[name] = {
            "total_tokens": int(len(arr)),
            "train_tokens": int(split_point),
            "val_tokens": int(n_val),
        }

    train_ids = np.concatenate(train_chunks) if train_chunks else np.zeros(0, dtype=TOKEN_DTYPE)
    val_ids = np.concatenate(val_chunks) if val_chunks else np.zeros(0, dtype=TOKEN_DTYPE)
    return train_ids, val_ids, source_splits


def tokenize_corpus(
    corpus: Path,
    tokenizer_dir: Path,
    out_dir: Path,
    val_fraction: float = 0.1,
    chunk_lines: int = 50_000,
    overwrite: bool = False,
    source_word_counts: Optional[Dict[str, int]] = None,
) -> TokenStats:
    """Encode ``corpus`` with the tokenizer in ``tokenizer_dir``; write train/val ``.npy``.

    Split strategy depends on ``source_word_counts``:

    - **Default (``None``)**: the split is taken at the end of the token stream (the last
      ``val_fraction`` of tokens become validation), matching tt-train's 90/10 tail split.
      Unchanged from before ``source_word_counts`` existed -- this is what produced the
      currently-shipped ``artifacts/tokens/{train,val}_ids.npy``, and this branch of the
      function is untouched so that lineage keeps meaning what it always has.
    - **Stratified (a ``{source_name: emitted_words}`` mapping)**: a proportional
      ``val_fraction`` tail is held out from EACH source's own span instead of once from
      the whole stream -- see :func:`_tokenize_stratified` for why the default split
      silently produces a single-source validation set on a multi-source corpus, and for
      exactly what "each source's own span" means. Pass
      ``{name: rec["emitted_words"] for name, rec in json.load(open(blend_manifest))
      ["sources"].items()}`` to drive this from a real blend manifest.

    Refuses to overwrite an existing ``train_ids.npy``/``val_ids.npy`` in ``out_dir`` unless
    ``overwrite=True`` (CLI: ``--force``) -- see :class:`TokenArtifactExistsError`. This is
    checked before any tokenization work happens, so a run that will be refused fails fast
    rather than after minutes of encoding a 500MB+ corpus.
    """
    from transformers import AutoTokenizer

    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be in [0.0, 1.0], got {val_fraction}")
    if chunk_lines <= 0:
        raise ValueError(f"chunk_lines must be > 0, got {chunk_lines}")

    corpus, tokenizer_dir, out_dir = Path(corpus), Path(tokenizer_dir), Path(out_dir)
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus}")

    train_path, val_path = out_dir / "train_ids.npy", out_dir / "val_ids.npy"
    if not overwrite:
        existing = [p for p in (train_path, val_path) if p.exists()]
        if existing:
            named = ", ".join(str(p) for p in existing)
            raise TokenArtifactExistsError(
                f"refusing to overwrite existing token artifact(s): {named}. These may be "
                f"the only tokens a trained model's parity gate was ever validated "
                f"against, and a different corpus or tokenizer produces different ids "
                f"under the same filenames with no way to tell after the fact. Pass "
                f"overwrite=True (CLI: --force) if you have confirmed nothing depends on "
                f"the current contents, or write to a different --out directory instead."
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

    source_splits: Optional[Dict[str, Dict[str, int]]] = None
    if source_word_counts is not None:
        train_ids, val_ids, source_splits = _tokenize_stratified(
            corpus, tok, source_word_counts, val_fraction, chunk_lines)
        total_tokens = len(train_ids) + len(val_ids)
    else:
        pieces: List[np.ndarray] = []
        batch: List[str] = []

        def _flush() -> None:
            if not batch:
                return
            pieces.append(encode_batch(batch, tok))
            batch.clear()

        with corpus.open("r", encoding="utf-8") as fin:
            for line in fin:
                batch.append(line.rstrip("\n"))
                if len(batch) >= chunk_lines:
                    _flush()
        _flush()

        ids = np.concatenate(pieces) if pieces else np.zeros(0, dtype=TOKEN_DTYPE)
        n_val = int(len(ids) * val_fraction)
        split = len(ids) - n_val
        train_ids, val_ids = ids[:split], ids[split:]
        total_tokens = len(ids)

    np.save(train_path, train_ids)
    np.save(val_path, val_ids)

    return TokenStats(
        total_tokens=int(total_tokens),
        train_tokens=int(len(train_ids)),
        val_tokens=int(len(val_ids)),
        vocab_size=int(tok.vocab_size),
        source_splits=source_splits,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="artifacts/corpus/blend.txt",
                    help="Path to the prepared corpus text file (default: %(default)s). "
                         "Matches scripts/build_tokenizer.py's default on purpose: the "
                         "documented sequence trains the tokenizer and then tokenizes "
                         "with it, and two different defaults meant step 2 looked for a "
                         "file step 1 never wrote. Pass artifacts/corpus/corpus.txt to "
                         "tokenize the legacy TinyStories-only corpus instead.")
    p.add_argument("--tokenizer", default="artifacts/tokenizer",
                    help="Directory holding the trained tokenizer.")
    p.add_argument("--out", default="artifacts/tokens",
                    help="Directory to write train_ids.npy / val_ids.npy into.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                    help="Fraction of the token stream (tail) held out for validation.")
    p.add_argument("--chunk-lines", type=int, default=50_000,
                    help="Corpus lines encoded per tokenizer call (a memory knob only).")
    p.add_argument("--force", "--overwrite", dest="force", action="store_true",
                    help="Overwrite an existing train_ids.npy/val_ids.npy in --out. Without "
                         "this flag, tokenize_corpus refuses to run if either file already "
                         "exists, since a model's parity gate may still be validated "
                         "against the current contents (see TokenArtifactExistsError).")
    p.add_argument("--blend-manifest", default=None, type=Path,
                    help="Path to a blend_manifest.json (e.g. "
                         "docs/measurements/blend_manifest.json) describing exactly how "
                         "--corpus was assembled. When given, the validation split is "
                         "STRATIFIED: a proportional --val-fraction tail is held out from "
                         "EACH source's own span (located via the manifest's "
                         "per-source emitted_words) instead of once from the tail of the "
                         "whole stream -- see _tokenize_stratified's docstring for why the "
                         "default split can silently produce a single-source validation "
                         "set. Omit this flag to keep the pre-existing tail-of-the-whole-"
                         "stream behaviour exactly as it was.")
    return p.parse_args(argv)


def source_word_counts_from_manifest(manifest_path: Path) -> Dict[str, int]:
    """``{source_name: emitted_words}`` from a ``blend_manifest.json``, for --blend-manifest.

    ``emitted_words`` (not ``emitted_tokens``) on purpose: it is an exact count from
    ``scripts/blend_corpus.py``'s own word-truncation arithmetic, so it locates a source's
    line boundary in the corpus file exactly. ``emitted_tokens`` is measured with a
    different chunking (paragraph-based, see ``blend_corpus.TokenMeter``) than this module
    uses (line-based), so the two do not agree token-for-token and would drift the
    boundary this function is used to find.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    return {name: rec["emitted_words"] for name, rec in manifest["sources"].items()}


def main() -> int:
    args = _parse_args()
    source_word_counts = None
    if args.blend_manifest is not None:
        source_word_counts = source_word_counts_from_manifest(args.blend_manifest)
    try:
        stats = tokenize_corpus(
            Path(args.corpus),
            Path(args.tokenizer),
            Path(args.out),
            val_fraction=args.val_fraction,
            chunk_lines=args.chunk_lines,
            overwrite=args.force,
            source_word_counts=source_word_counts,
        )
    except TokenArtifactExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
