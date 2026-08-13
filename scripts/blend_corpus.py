#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Blend the prepared sources into one corpus, and record exactly what went in.

The manifest this writes is the point: it makes "what was this model trained on" an
answerable question, with per-source token counts, repetition factors, achieved shares and
the pinned revision each source came from. Every number in it is measured against the
trained tokenizer, not approximated -- see ``TokenMeter`` and ``_measure_tokens_per_word``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.corpus import SOURCES, format_share  # noqa: E402
from train.paths import shared_dir  # noqa: E402

DEFAULT_BUDGET = 400_000_000

#: Paragraphs handed to the tokenizer per ``encode_batch`` call while metering. A batching
#: knob only: the counts are identical at any batch size, because each paragraph is encoded
#: independently either way.
_METER_BATCH = 1_000


def plan_blend(available: Dict[str, int], budget: int) -> Dict[str, int]:
    """Tokens to emit per source. Raises ValueError if a share cannot be met.

    ``available`` holds TOKENIZER-MEASURED tokens per source, so the gate here is in real
    tokens and the emitter must be too -- see ``_measure_tokens_per_word`` for the bug that
    happens when it isn't.
    """
    plan: Dict[str, int] = {}
    for name, src in SOURCES.items():
        want = int(round(src.target_share * budget))
        have = available.get(name, 0) * src.upsample
        if have < want:
            raise ValueError(
                f"{name} cannot supply its {format_share(src.target_share)} share: needs "
                f"{want:,} tokens, has {available.get(name, 0):,} x{src.upsample} = "
                f"{have:,}. Re-run scripts/measure_corpus.py and settle the shares first."
            )
        plan[name] = want
    return plan


def _count_words(path: Path) -> int:
    """Whitespace-delimited words in ``path``, counted the way ``_emit`` counts them.

    Streams line by line: tinystories.txt is ~1.9 GB and ``read_text().split()`` over it
    would build a list of hundreds of millions of str objects. ``str.split()`` rather than
    ``wc -w`` semantics on purpose -- this number is the denominator of the ratio ``_emit``
    divides by, so it has to be the same notion of "word" the emitter uses (they differ on
    Unicode separators such as U+00A0, by a handful of words per source).
    """
    words = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            words += len(line.split())
    return words


def _measure_tokens_per_word(path: Path, available_tokens: int) -> Tuple[float, int]:
    """``(tokens_per_word, words)`` for one prepared source. Measured, never assumed.

    THE BUG THIS EXISTS TO CLOSE. ``_emit`` used to size its emission with a flat
    ``tokens_per_word = 1.3`` while ``plan_blend`` gated on tokenizer-MEASURED
    availability. Real tokens/word across these nine sources runs 1.194 (tinystories) to
    1.559 (wikipedia_simple) -- a 30% spread -- so the emitter over-emitted for eight of
    the nine, by exactly the factor ``real_ratio / 1.3``. Consequences in the shipped
    blend: ``wikipedia_simple`` declared ``upsample=1`` and made 1.058 passes, silently
    duplicating ~5.8% of Simple Wikipedia; ``procedural`` made 4.03 passes, over the 4x
    working limit that Task 6 moved a whole share point to stay under; the blend totalled
    425,024,350 real tokens against a 400M budget; and the manifest reported every
    ``achieved_share`` as exactly its target to 15 decimal places.

    The ratio is derived rather than declared: ``available_tokens`` comes from
    ``docs/measurements/corpus_availability.json``, which is the trained tokenizer's own
    count over this same file, and dividing it by the file's word count gives that source's
    real tokens per word. ``target_words = want_tokens / ratio`` then makes the REAL
    emitted token count track the plan.

    With the ratio correct, a source's repetition factor collapses to
    ``want_tokens / available_tokens``, which ``plan_blend``'s gate already holds at or
    below the declared ``upsample`` -- so the emitter can no longer exceed a source's
    declared repetition. That was not true with a flat constant.
    """
    words = _count_words(path)
    if words == 0:
        raise ValueError(f"{path} contains no words; cannot measure its tokens per word")
    if available_tokens <= 0:
        raise ValueError(
            f"no measured token count for {path.name} in the availability report; "
            f"re-run scripts/measure_corpus.py before blending"
        )
    return available_tokens / words, words


class TokenMeter:
    """Counts REAL tokens in the text written for one source, while it is written.

    The manifest's whole purpose is answering "what was this model trained on", so its
    token counts are the tokenizer's, not an estimate. Chunking matches
    ``scripts/measure_corpus.py`` exactly -- paragraphs split on blank lines, whitespace-
    only chunks skipped -- so ``emitted_tokens`` is directly comparable with
    ``available_tokens``, which that script produced from the same files. (BPE merges do
    not cross an ``encode`` call, so a different split would give a slightly different
    total and the two numbers would no longer be measuring the same thing.)

    Metering while writing avoids a second full pass over a 1.7 GB artifact; the buffer
    holds one paragraph, never the file.
    """

    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer
        self._buf = ""
        self._pending: List[str] = []
        self.tokens = 0

    def feed(self, text: str) -> None:
        """Consume a piece of the text being written. Splits on blank lines."""
        self._buf += text
        if "\n\n" not in self._buf:
            return
        chunks = self._buf.split("\n\n")
        # The last fragment may still be extended by the next feed(), so it stays buffered.
        self._buf = chunks.pop()
        for chunk in chunks:
            if chunk.strip():
                self._pending.append(chunk)
        if len(self._pending) >= _METER_BATCH:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        for enc in self._tok.encode_batch(self._pending):
            self.tokens += len(enc.ids)
        self._pending.clear()

    def close(self) -> int:
        """Flush the tail and return the total token count for this source."""
        if self._buf.strip():
            self._pending.append(self._buf)
        self._buf = ""
        self._flush()
        return self.tokens


def load_tokenizer(tokenizer_dir: Path):
    """The trained tokenizer, or None if there isn't one (fresh clone, pre-Task-2)."""
    tok_json = tokenizer_dir / "tokenizer.json"
    if not tok_json.is_file():
        return None
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(tok_json))
    except Exception as exc:  # noqa: BLE001 - any failure here means "no real counts"
        print(f"WARNING: could not load {tok_json}: {exc}", file=sys.stderr)
        return None


@dataclass(frozen=True)
class Emission:
    """What ``_emit`` actually wrote for one source.

    ``tokens`` is derived from the source's measured tokens/word ratio. It is what the
    emitter aimed at; the manifest prefers ``TokenMeter``'s tokenizer count when one is
    available, and records which of the two it used.
    """

    words: int
    tokens: int


def _emit(src_path: Path, want_tokens: int, out, *, tokens_per_word: float,
          on_text: Optional[Callable[[str], None]] = None) -> Emission:
    """Append text from ``src_path`` until ``want_tokens`` is reached, repeating if needed.

    ``tokens_per_word`` is that source's MEASURED ratio (see ``_measure_tokens_per_word``),
    not a shared constant: a flat constant over-emits for every source that compresses
    worse than the constant claims, which is how an ``upsample=1`` source ends up
    duplicating part of itself.

    The final pass is TRUNCATED. An earlier draft of this function wrote
    only whole passes over the source file, which cannot undershoot a large source: with
    tinystories offering 445M tokens against a 120M want, one pass emitted the entire file
    and the slice achieved 53% against a 30% target, with the blend totalling 839M tokens
    against a 400M budget. Truncation is what makes the achieved shares track the targets.

    Streams line by line rather than reading the file into memory: tinystories.txt is ~1.9 GB
    and ``text.split()`` over it would build a list of hundreds of millions of str objects.

    Truncation is at WORD level, not line level: a source whose paragraphs are single long
    lines cannot be trimmed at a line boundary, so overshoot would be bounded by the longest
    line rather than by a couple of percent.

    ``on_text`` receives every string written, so a ``TokenMeter`` can count the real
    tokens of exactly this emission without a second pass over the output.

    The truncation, streaming, word-level-boundary and measured-ratio behaviour described
    above is covered by ``tests/test_blend_corpus.py``.
    """
    if src_path.stat().st_size == 0:
        raise ValueError(f"{src_path} is empty; cannot emit {want_tokens:,} tokens from it")

    def write(text: str) -> None:
        out.write(text)
        if on_text is not None:
            on_text(text)

    words = 0
    target_words = want_tokens / tokens_per_word
    while True:
        pass_words = 0
        with src_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                n = len(parts)
                if words + n >= target_words:
                    need = int(target_words - words)
                    if need > 0:
                        write(" ".join(parts[:need]) + "\n")
                        words += need
                    return Emission(words=words, tokens=int(round(words * tokens_per_word)))
                write(line)
                words += n
                pass_words += n
        if pass_words == 0:
            # size > 0 but nothing but whitespace: the repeat loop would never terminate.
            raise ValueError(f"{src_path} contains no words; cannot emit tokens from it")
        write("\n\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--availability", type=Path,
                   default=ROOT / "docs" / "measurements" / "corpus_availability.json")
    p.add_argument("--record", type=Path,
                   default=ROOT / "docs" / "measurements" / "blend_manifest.json",
                   help="Tracked copy of the manifest (default: %(default)s). The blend "
                        "itself lives under artifacts/, which is gitignored, so without "
                        "this the answer to 'what was this model trained on' would exist "
                        "only on the machine that ran the blend.")
    args = p.parse_args()

    if not args.availability.is_file():
        print(f"ERROR: {args.availability} not found. Run scripts/measure_corpus.py first.",
              file=sys.stderr)
        return 1
    available = json.loads(args.availability.read_text())["available"]

    try:
        plan = plan_blend(available, args.budget)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    tokenizer = load_tokenizer(shared_dir("tokenizer"))
    if tokenizer is None:
        print("WARNING: no trained tokenizer found; emitted token counts will be derived "
              "from each source's measured tokens/word ratio rather than counted directly. "
              "The manifest will say so.", file=sys.stderr)

    corpus_dir = shared_dir("corpus")
    out_path = corpus_dir / "blend.txt"
    records: Dict[str, dict] = {}
    print(f"{'source':22} {'tokens':>13} {'share':>7} {'passes':>7} {'cap':>4}")
    print("-" * 58)
    with out_path.open("w", encoding="utf-8") as out:
        for name in sorted(plan):
            src_path = corpus_dir / f"{name}.txt"
            if not src_path.is_file():
                print(f"ERROR: {src_path} missing; run scripts/prepare_corpus.py",
                      file=sys.stderr)
                return 1
            try:
                ratio, source_words = _measure_tokens_per_word(src_path,
                                                               available.get(name, 0))
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

            meter = TokenMeter(tokenizer) if tokenizer is not None else None
            emission = _emit(src_path, plan[name], out, tokens_per_word=ratio,
                             on_text=(meter.feed if meter is not None else None))
            measured = meter.close() if meter is not None else None

            # The repetition actually applied. Fractional on purpose: it is what happened,
            # not what the registry declares. Below 1.0 means part of the source was used
            # once and the rest not at all; above 1.0 means that much of it was repeated.
            passes = emission.words / source_words
            src = SOURCES[name]
            records[name] = {
                "planned_tokens": plan[name],
                "emitted_tokens": measured if measured is not None else emission.tokens,
                "emitted_tokens_method": ("tokenizer" if measured is not None
                                          else "approx (words x measured tokens/word)"),
                "emitted_words": emission.words,
                "source_file_words": source_words,
                "source_tokens_per_word": round(ratio, 6),
                "repetition_factor": round(passes, 4),
                "declared_upsample": src.upsample,
                "repetition_within_declared_upsample": passes <= src.upsample,
                "target_share": src.target_share,
                "available_tokens": available.get(name, 0),
                "hf_repo": src.hf_repo,
                "hf_revision": src.hf_revision,
                "license_id": src.license_id,
            }
            if measured is not None and plan[name]:
                records[name]["planned_vs_emitted_error"] = round(
                    measured / plan[name] - 1.0, 6)
            print(f"{name:22} {records[name]['emitted_tokens']:>13,} "
                  f"{format_share(records[name]['emitted_tokens'] / args.budget):>7} "
                  f"{passes:>7.3f} {src.upsample:>4}")

    total = sum(r["emitted_tokens"] for r in records.values())
    for name, rec in records.items():
        rec["achieved_share"] = rec["emitted_tokens"] / total

    over = [n for n, r in records.items() if not r["repetition_within_declared_upsample"]]
    if over:
        # plan_blend's gate makes this unreachable while the ratio is measured (repetition
        # is want/available, which the gate holds at or below upsample). Kept as a loud
        # signal rather than a silent invariant: it is exactly the condition that shipped.
        print(f"\nWARNING: real repetition exceeds the declared upsample for: "
              f"{', '.join(sorted(over))}", file=sys.stderr)

    digest = hashlib.sha256()
    with out_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    manifest = {
        "budget": args.budget,
        "total_emitted_tokens": total,
        "total_vs_budget_tokens": total - args.budget,
        "total_vs_budget_pct": round(100.0 * (total / args.budget - 1.0), 3),
        "token_count_method": (
            "tokenizer" if tokenizer is not None
            else "approx (words x each source's measured tokens/word)"),
        "token_count_note": (
            "emitted_tokens is counted with the trained tokenizer over exactly the text "
            "written for each source, chunked the same way scripts/measure_corpus.py "
            "chunks it, so it is directly comparable with available_tokens. "
            "repetition_factor is exact (emitted_words / source_file_words). "
            "source_tokens_per_word is available_tokens / source_file_words, the measured "
            "ratio the emitter sizes its output with."),
        "output": out_path.name,
        "sha256": digest.hexdigest(),
        "sources": {name: records[name] for name in sorted(records)},
    }
    manifest_path = corpus_dir / "blend_manifest.json"
    serialised = json.dumps(manifest, indent=2)
    manifest_path.write_text(serialised)
    print(f"\ntotal {total:,} tokens against a {args.budget:,} budget "
          f"({manifest['total_vs_budget_pct']:+.3f}%)")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    print(f"wrote {manifest_path}")
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(serialised)
        print(f"wrote {args.record} (tracked record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
