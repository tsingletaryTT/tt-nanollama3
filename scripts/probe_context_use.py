#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does this model use the context it already has? Mean cross-entropy by position, bucketed.

THE PROBLEM THIS MEASURES
--------------------------
``.superpowers/max-context-investigation.md`` §7 ran a one-off probe that turned out to be
the single most decisive measurement this project has made: per-token loss on
``artifacts/hf-tt-tnt-v1`` stops improving at position ~64 and stays flat (or gets fractionally
*worse*) all the way to 511, on a book source (``gutenberg_children``) as much as a short one
(``tinystories``). That reframed "how big should the context window be" into "the model
cannot use the window it already has" -- a conclusion strong enough to gate a retrain on.

That probe was ad-hoc: hand-picked spans "confirmed by decoding", a fixed 64-window sample,
no standard errors, and never committed. This script is the tool version -- the same
measurement, reproducible, with the honesty this project has repeatedly needed reminding of
(see ``scripts/eval_per_source.py``'s docstring on the same theme): a bare mean is not a
finding, a standard error is.

METHOD
------
1. Sample ``--n-windows`` random (with-replacement) windows of length ``--seq-len`` from a
   token array on disk (e.g. ``artifacts/tokens-stratified/val_ids.npy``), exactly the way
   ``scripts/eval_per_source.py::sample_windows`` does -- imported directly from there so the
   two scripts can never sample two subtly different ways.
2. Forward the model once per window batch and compute cross-entropy **per token position**
   (never through ``labels=``, which shifts a second time -- see
   ``scripts/eval_per_source.py::per_window_losses``'s docstring for how that mistake produces
   a passing-looking but meaningless number). This gives one loss value per
   ``(window, position)`` pair.
3. Bucket positions (default ``[0,32) [32,64) [64,128) [128,256) [256,512) ...``, doubling
   naturally past 512 -- see :func:`default_bucket_edges`). Within each bucket, first average
   over positions **within a window** (one number per window), THEN take the mean and standard
   error **over windows**. This is the same "the window is the exchangeable sampling unit, not
   the token" convention ``eval_per_source.py`` uses for its own SEM, for the same reason:
   positions in the same window share a document and a context, so treating every
   ``(window, position)`` pair as an independent draw would understate the real noise.

CONSTRAINTS THIS SCRIPT RESPECTS
---------------------------------
CPU only. Never imports ttml/ttnn and never opens a Tenstorrent device. A model whose
``max_position_embeddings`` is smaller than ``--seq-len`` is a **hard, explicit failure**
(:func:`require_context_capacity`) -- this script never silently truncates the window or lets
a model read positions it was never trained on (RoPE quietly zero-fills past its cache length
rather than erroring; see ``.superpowers/max-context-investigation.md`` §3.3(a)).

PER-SOURCE BREAKDOWN (--blend-manifest)
----------------------------------------
Books vs. short items is exactly the comparison that made the original finding convincing --
a model might genuinely use long context on 100,000-token books and not on 200-token
TinyStories, and a blended average would hide that split entirely.

``train/tokenization.py --blend-manifest`` (the ``_tokenize_stratified`` path) holds out a
proportional ``--val-fraction`` tail from EACH source's own span rather than once from the
tail of the whole stream, so this script's per-source reconstruction has to match that split
exactly (see :func:`stratified_source_spans`), not ``eval_per_source.py``'s
``verify_against_disk_arrays`` (written for the OLDER single-straddler split). Reuses
``eval_per_source.py``'s ``tokenize_blend_by_source`` for the part that IS shared: replaying
``--blend`` through the tokenizer once, split by source, using the manifest's exact
``emitted_words`` (never the estimated ``emitted_tokens`` -- see that function's docstring).
The reconstruction is verified byte-for-byte against ``--tokens`` before anything is reported
(:func:`verify_stratified_reconstruction`) -- a mismatch raises rather than silently
mis-attributing tokens to the wrong source.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_per_source import (  # noqa: E402
    emission_order,
    emitted_words,
    load_manifest,
    mean_and_sem,
    sample_windows,
    tokenize_blend_by_source,
)
from train.corpus import SOURCES  # noqa: E402

#: Sequence length windows are sampled at, by default. Matches
#: ``artifacts/hf-tt-tnt-v1/config.json``'s ``max_position_embeddings`` -- the model's actual
#: trained context -- so a default run measures exactly the window the model has, not an
#: arbitrary smaller or larger one.
DEFAULT_SEQ_LEN = 512

#: Windows sampled per (source or aggregate) probe. Chosen to finish in a few minutes on CPU
#: for a model this project's size (~22M params): ``eval_per_source.py`` uses 256 windows per
#: (source, condition) pair and reports "a couple of minutes" for up to 18 such pairs: a
#: single aggregate probe at this sample size is comfortably faster than that whole budget.
DEFAULT_N_WINDOWS = 256

#: Forward-pass batch size. A memory/throughput knob only -- does not change which windows are
#: sampled or their losses.
DEFAULT_BATCH_SIZE = 32

#: Corpus-assembly chunking knob, matching train/tokenization.py's default. Memory only.
DEFAULT_CHUNK_LINES = 50_000

#: Matches train/tokenization.py's own default -- must match whatever fraction actually
#: produced --tokens for the stratified reconstruction to check out byte-for-byte.
DEFAULT_VAL_FRACTION = 0.1

#: Bucket edges this project's own decisive measurement used, before doubling naturally into
#: longer windows. See :func:`default_bucket_edges`.
_BASE_BUCKET_EDGES = (0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


# ---------------------------------------------------------------------------------------
# Context-capacity gate: the loud failure this tool must never skip
# ---------------------------------------------------------------------------------------


def require_context_capacity(hf_model: Path, seq_len: int) -> int:
    """Raise clearly if ``seq_len`` exceeds ``hf_model``'s trained context.

    Reads ``config.json`` directly (no ``transformers`` import needed for this check, so it
    runs before the -- possibly slow -- model load). A model whose ``max_position_embeddings``
    is smaller than the requested window is refused outright: RoPE has no bounds check against
    the input's actual sequence length (only against its own cos/sin cache), so exceeding it
    does not raise on this project's Llama-family models -- it silently zero-fills positions
    past the cache and produces a confidently wrong loss (see
    ``.superpowers/max-context-investigation.md`` §3.3(a), which is exactly the class of
    silent-corruption bug this project's tooling is supposed to refuse rather than reproduce).

    Returns the model's ``max_position_embeddings`` for the caller to record as provenance.
    """
    config_path = hf_model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"no config.json at {hf_model}; cannot verify --seq-len {seq_len} fits inside "
            f"this model's trained context before doing any work"
        )
    config = json.loads(config_path.read_text())
    max_pos = config.get("max_position_embeddings")
    if max_pos is None:
        raise ValueError(
            f"{config_path} has no 'max_position_embeddings' field -- cannot verify "
            f"--seq-len {seq_len} is safe for this model. Refusing to guess."
        )
    if seq_len > max_pos:
        raise ValueError(
            f"--seq-len {seq_len} exceeds {hf_model}'s max_position_embeddings ({max_pos}). "
            f"This model was never trained on positions past {max_pos - 1}; forwarding a "
            f"longer window would not raise (RoPE silently zero-fills past its cache -- see "
            f".superpowers/max-context-investigation.md §3.3(a)) but the resulting loss would "
            f"be meaningless. Pass --seq-len <= {max_pos}, or point --hf-model at a model "
            f"converted with a longer trained context."
        )
    return int(max_pos)


# ---------------------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------------------


def default_bucket_edges(seq_len: int) -> List[int]:
    """Position-bucket boundaries for a window of length ``seq_len``.

    ``[0, 32, 64, 128, 256, 512, ...]``, doubling, truncated to whatever is ``< seq_len`` and
    always closed off with ``seq_len`` itself -- so at ``seq_len=512`` this is exactly
    ``[0, 32, 64, 128, 256, 512]``, the buckets ``[0,32) [32,64) [64,128) [128,256) [256,512)``
    that reproduced the original finding, and at any other ``seq_len`` it extends or truncates
    the same scheme naturally rather than needing a special case per window length.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    edges = [e for e in _BASE_BUCKET_EDGES if e < seq_len]
    edges.append(seq_len)
    return edges


@dataclass(frozen=True)
class BucketEstimate:
    """One position bucket's loss: mean ± SEM over ``n_windows`` independent windows."""

    lo: int
    hi: int
    mean: float
    sem: float
    n_windows: int


def bucket_position_losses(per_position: np.ndarray, edges: List[int]) -> List[BucketEstimate]:
    """Turn a ``(n_windows, seq_len)`` per-token-loss array into bucketed mean ± SEM.

    Within each bucket, first averages **across positions within a window** (one number per
    window), then computes mean/SEM **across windows** -- see the module docstring for why
    the window, not the token, is the sampling unit SEM is computed over.
    """
    if per_position.ndim != 2:
        raise ValueError(
            f"expected a (n_windows, seq_len) array, got shape {per_position.shape}"
        )
    n_windows, seq_len = per_position.shape
    if edges[0] != 0:
        raise ValueError(f"bucket edges must start at 0, got {edges}")
    if edges[-1] > seq_len:
        raise ValueError(
            f"bucket edges go up to {edges[-1]}, past the {seq_len}-token window available"
        )
    results: List[BucketEstimate] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        window_means = per_position[:, lo:hi].mean(axis=1)
        mean, sem = mean_and_sem(window_means.tolist())
        results.append(BucketEstimate(lo=lo, hi=hi, mean=mean, sem=sem,
                                      n_windows=len(window_means)))
    return results


# ---------------------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------------------


def per_window_position_losses(model, x: np.ndarray, y: np.ndarray,
                               batch_size: int) -> np.ndarray:
    """Cross-entropy for every ``(window, position)`` pair -- shape ``(n_windows, seq_len)``.

    Same no-``labels=``-kwarg convention as ``eval_per_source.py::per_window_losses`` (see its
    docstring): computed directly against ``logits``, never through the HF model's own
    ``labels=`` shift-again path. The only difference from that function is that this one
    keeps every position's loss instead of collapsing each window to its mean.
    """
    import torch

    n_windows, seq_len = x.shape
    out = np.empty((n_windows, seq_len), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            xb = torch.from_numpy(x[start:start + batch_size])
            yb = torch.from_numpy(y[start:start + batch_size])
            logits = model(xb).logits.float()
            per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="none"
            ).reshape(yb.shape)
            out[start:start + per_token.shape[0]] = per_token.numpy()
    return out


def probe_positions(model, ids: np.ndarray, seq_len: int, n_windows: int, batch_size: int,
                    rng: np.random.Generator,
                    edges: Optional[List[int]] = None) -> List[BucketEstimate]:
    """Sample windows from ``ids`` and return the bucketed position-wise loss.

    Raises ``ValueError`` (via ``sample_windows``) if ``ids`` does not hold enough tokens for
    even one window of length ``seq_len`` -- never silently samples fewer/shorter windows.
    """
    if edges is None:
        edges = default_bucket_edges(seq_len)
    x, y = sample_windows(ids, seq_len, n_windows, rng)
    per_position = per_window_position_losses(model, x, y, batch_size)
    return bucket_position_losses(per_position, edges)


# ---------------------------------------------------------------------------------------
# Per-source span reconstruction (stratified split -- train/tokenization.py's --blend-manifest)
# ---------------------------------------------------------------------------------------


def stratified_source_spans(per_source_ids: Dict[str, np.ndarray], val_fraction: float,
                            side: str) -> Dict[str, np.ndarray]:
    """One source's slice of ``--tokens``, replaying ``_tokenize_stratified``'s own split.

    ``train/tokenization.py::_tokenize_stratified`` holds out the LAST ``val_fraction`` of
    EACH source's own span (``n_val = int(len(arr) * val_fraction)``) -- this must match that
    arithmetic exactly (same floor-toward-zero truncation, same tail-not-head direction) or
    the spans this function returns will not line up with the real ``train_ids.npy``/
    ``val_ids.npy`` files token-for-token.
    """
    if side not in ("train", "val"):
        raise ValueError(f"side must be 'train' or 'val', got {side!r}")
    spans: Dict[str, np.ndarray] = {}
    for name, arr in per_source_ids.items():
        n_val = int(len(arr) * val_fraction)
        split_point = len(arr) - n_val
        spans[name] = arr[:split_point] if side == "train" else arr[split_point:]
    return spans


def verify_stratified_reconstruction(spans: Dict[str, np.ndarray], order: List[str],
                                     tokens: np.ndarray) -> None:
    """Raise unless concatenating ``spans`` in ``order`` reproduces ``tokens`` byte-for-byte.

    The mechanical confirmation that ``--blend``/``--blend-manifest``/``--val-fraction``/
    ``--split`` actually describe how ``--tokens`` was produced -- mirrors
    ``eval_per_source.py::verify_against_disk_arrays``'s philosophy (never trust a
    reconstruction that has not been checked against the real array on disk) for the
    stratified split's different shape (every source contributes to both sides, rather than
    at most one straddler).
    """
    pieces = [spans[name] for name in order]
    reconstructed = (np.concatenate(pieces) if pieces
                     else np.zeros(0, dtype=tokens.dtype))
    if len(reconstructed) != len(tokens) or not np.array_equal(reconstructed, tokens):
        raise ValueError(
            f"reconstructed {len(reconstructed):,} per-source tokens do not match the "
            f"{len(tokens):,} tokens in --tokens byte-for-byte. --blend, --blend-manifest, "
            f"--val-fraction, or --split do not describe how --tokens was actually produced "
            f"-- do not trust any per-source breakdown downstream of this until the mismatch "
            f"is understood."
        )


def infer_split_side(tokens_path: Path) -> Optional[str]:
    """Guess 'train' or 'val' from ``tokens_path``'s filename; ``None`` if ambiguous.

    A convenience default only -- ``--split`` always overrides it, and a caller relying on the
    guess for a file named neither ``*val*`` nor ``*train*`` gets an explicit error instead of
    a silent wrong guess (see ``_parse_args``/``main``).
    """
    name = tokens_path.name.lower()
    has_val, has_train = "val" in name, "train" in name
    if has_val and not has_train:
        return "val"
    if has_train and not has_val:
        return "train"
    return None


@dataclass
class SourcePositionReport:
    """One source's position-wise loss, or a note explaining why it has none."""

    name: str
    slice_name: str
    n_tokens_available: int
    buckets: Optional[List[BucketEstimate]]
    note: str = ""


def build_source_reports(manifest: dict, spans: Dict[str, np.ndarray], order: List[str],
                         model, *, seq_len: int, n_windows: int, batch_size: int, seed: int,
                         edges: List[int]) -> List[SourcePositionReport]:
    """Per-source bucketed loss, skipping (with a note, never a crash) a source too short."""
    reports: List[SourcePositionReport] = []
    min_tokens = seq_len + 1
    for i, name in enumerate(order):
        span = spans[name]
        slice_name = SOURCES[name].slice if name in SOURCES else "?"
        if len(span) < min_tokens:
            reports.append(SourcePositionReport(
                name=name, slice_name=slice_name, n_tokens_available=len(span), buckets=None,
                note=f"insufficient tokens for a {seq_len}-token window: only {len(span):,} "
                    f"available in this split, need > {min_tokens:,}",
            ))
            continue
        rng = np.random.default_rng(seed + 1000 * i)
        buckets = probe_positions(model, span, seq_len, n_windows, batch_size, rng, edges)
        reports.append(SourcePositionReport(name=name, slice_name=slice_name,
                                            n_tokens_available=len(span), buckets=buckets))
    return reports


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


def _fmt_bucket_row(b: BucketEstimate) -> str:
    return (f"| [{b.lo}, {b.hi}) | {b.n_windows} | {b.mean:.4f} | {b.sem:.4f} |")


def render_markdown(overall: List[BucketEstimate], *, hf_model: Path, tokens_path: Path,
                    seq_len: int, n_windows: int, batch_size: int,
                    source_reports: Optional[List[SourcePositionReport]] = None,
                    checkpoint_note: str = "") -> str:
    lines: List[str] = []
    lines.append("<!-- SPDX-License-Identifier: Apache-2.0 -->")
    lines.append("<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->")
    lines.append("")
    lines.append("# Position-wise loss — does this model use its context?")
    lines.append("")
    lines.append(
        f"Model `{hf_model}`, tokens `{tokens_path}`, window length {seq_len}, "
        f"{n_windows} windows per bucket (batch size {batch_size}), generated by "
        f"`scripts/probe_context_use.py`."
    )
    if checkpoint_note:
        lines.append("")
        lines.append(checkpoint_note)
    lines.append("")
    lines.append("## Why this exists")
    lines.append("")
    lines.append(
        "If a model's per-token loss keeps falling as position (context length) increases, "
        "it is using the context it has. If the loss goes flat, the model has stopped "
        "benefiting from additional tokens of context -- raising the context window further "
        "would not help it. This is the tool version of the probe that first found "
        "`artifacts/hf-tt-tnt-v1`'s loss goes flat at position ~64 "
        "(`.superpowers/max-context-investigation.md` §7), now reproducible with standard "
        "errors instead of a bare mean."
    )
    lines.append("")
    lines.append("## Method, in brief")
    lines.append("")
    lines.append(
        "- Windows are sampled uniformly at random (with replacement) from the token array, "
        "the same way `scripts/eval_per_source.py::sample_windows` does."
    )
    lines.append(
        "- Cross-entropy is computed directly against `logits`, never through the `labels=` "
        "kwarg (which shifts a second time -- see `eval_per_source.py`'s docstring)."
    )
    lines.append(
        "- Each bucket reports **mean ± standard error over independent windows** -- first "
        "averaged across positions within a window, then across windows, so within-window "
        "autocorrelation is not mistaken for independent samples. `n` is the number of "
        "windows the estimate rests on, never the number of tokens."
    )
    lines.append("")
    lines.append("## Position-wise loss (all sources combined)")
    lines.append("")
    lines.append("| position bucket | n windows | mean loss | SEM |")
    lines.append("|---|---:|---:|---:|")
    for b in overall:
        lines.append(_fmt_bucket_row(b))
    lines.append("")

    if source_reports is not None:
        lines.append("## Position-wise loss, per source")
        lines.append("")
        lines.append(
            "Per-source spans are reconstructed from `--blend-manifest` via the same "
            "stratified split `train/tokenization.py --blend-manifest` uses -- see "
            "`stratified_source_spans`'s docstring -- and checked byte-for-byte against "
            "`--tokens` before anything below is computed."
        )
        lines.append("")
        lines.append("| source | slice | tokens available | position bucket | n windows | "
                     "mean loss | SEM |")
        lines.append("|---|---|---:|---|---:|---:|---:|")
        for r in source_reports:
            if r.buckets is None:
                # Keep the table's column count fixed; the reason a source has no buckets
                # goes on its own line below the table row, not squeezed into a cell.
                lines.append(f"| {r.name} | {r.slice_name} | {r.n_tokens_available:,} | "
                            f"n/a | n/a | n/a | n/a |")
                lines.append(f"  - *{r.name}: {r.note}*")
                continue
            for b in r.buckets:
                lines.append(f"| {r.name} | {r.slice_name} | {r.n_tokens_available:,} | "
                            f"[{b.lo}, {b.hi}) | {b.n_windows} | {b.mean:.4f} | {b.sem:.4f} |")
        lines.append("")

    return "\n".join(lines)


def _bucket_to_json(b: BucketEstimate) -> dict:
    return {"lo": b.lo, "hi": b.hi, "mean": b.mean, "sem": b.sem, "n_windows": b.n_windows}


def report_to_json(overall: List[BucketEstimate], *, hf_model: str, tokens_path: str,
                   seq_len: int, n_windows: int, batch_size: int, seed: int,
                   bucket_edges: List[int],
                   source_reports: Optional[List[SourcePositionReport]] = None) -> dict:
    payload = {
        "hf_model": hf_model,
        "tokens": tokens_path,
        "seq_len": seq_len,
        "n_windows": n_windows,
        "batch_size": batch_size,
        "seed": seed,
        "bucket_edges": bucket_edges,
        "overall": [_bucket_to_json(b) for b in overall],
    }
    if source_reports is not None:
        payload["per_source"] = {
            r.name: {
                "slice": r.slice_name,
                "n_tokens_available": r.n_tokens_available,
                "buckets": [_bucket_to_json(b) for b in r.buckets] if r.buckets else None,
                "note": r.note,
            }
            for r in source_reports
        }
    return payload


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", type=Path, default=ROOT / "artifacts" / "hf-tt-tnt-v1",
                   help="Converted HF model directory to evaluate (CPU only, default: "
                        "%(default)s).")
    p.add_argument("--tokens", type=Path,
                   default=ROOT / "artifacts" / "tokens-stratified" / "val_ids.npy",
                   help="A .npy uint array of token ids to sample windows from (default: "
                        "%(default)s).")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                   help="Window length to probe up to (default: %(default)s). Must not "
                        "exceed --hf-model's max_position_embeddings.")
    p.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS,
                   help="Windows sampled per probe (aggregate, and per source if "
                        "--blend-manifest is given). Default %(default)s finishes in a few "
                        "minutes on CPU for a model this project's size.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blend-manifest", type=Path, default=None,
                   help="Path to a blend_manifest.json (e.g. "
                        "docs/measurements/blend_manifest.json). When given, also reports a "
                        "per-source breakdown -- see the module docstring's PER-SOURCE "
                        "BREAKDOWN section.")
    p.add_argument("--blend", type=Path, default=ROOT / "artifacts" / "corpus" / "blend.txt",
                   help="Corpus file --blend-manifest describes (default: %(default)s). "
                        "Only read when --blend-manifest is given.")
    p.add_argument("--split", choices=("train", "val"), default=None,
                   help="Which side of the stratified split --tokens is. Inferred from the "
                        "filename ('val'/'train' substring) when omitted; required "
                        "explicitly if that is ambiguous. Only used with --blend-manifest.")
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                   help="Must match whatever --val-fraction actually produced --tokens "
                        "(default: %(default)s, train/tokenization.py's own default).")
    p.add_argument("--chunk-lines", type=int, default=DEFAULT_CHUNK_LINES)
    p.add_argument("--checkpoint-note", type=str, default="",
                   help="Freeform provenance line recorded verbatim in the markdown report "
                        "(e.g. which checkpoint step --hf-model was converted from).")
    p.add_argument("--out", type=Path, default=None,
                   help="Markdown output path (default: derived from --hf-model's directory "
                        "name under docs/measurements/).")
    p.add_argument("--json-out", type=Path, default=None,
                   help="JSON output path (default: derived the same way as --out).")
    return p.parse_args(argv)


def _default_output_paths(hf_model: Path) -> tuple[Path, Path]:
    tag = hf_model.name
    if tag.startswith("hf-"):
        tag = tag[len("hf-"):]
    out_dir = ROOT / "docs" / "measurements"
    return (out_dir / f"context-use-{tag}.md", out_dir / f"context-use-{tag}.json")


def main() -> int:
    args = _parse_args()

    if not (args.hf_model / "config.json").is_file():
        print(f"ERROR: no converted model at {args.hf_model} (config.json missing). Run "
              f"scripts/convert_checkpoint.py first, or point --hf-model at an existing "
              f"converted directory. This script is CPU-only and deliberately does not fall "
              f"back to the ttml/device path.", file=sys.stderr)
        return 1

    try:
        max_pos = require_context_capacity(args.hf_model, args.seq_len)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"model max_position_embeddings={max_pos}, probing up to --seq-len {args.seq_len}")

    if not args.tokens.is_file():
        print(f"ERROR: --tokens {args.tokens} not found.", file=sys.stderr)
        return 1
    ids = np.load(args.tokens)
    print(f"loaded {len(ids):,} tokens from {args.tokens}")

    edges = default_bucket_edges(args.seq_len)
    print(f"bucket edges: {edges}")

    from transformers import AutoModelForCausalLM

    print(f"loading model {args.hf_model} ...")
    model = AutoModelForCausalLM.from_pretrained(str(args.hf_model)).eval()

    print(f"probing {args.n_windows} windows of length {args.seq_len} ...")
    rng = np.random.default_rng(args.seed)
    overall = probe_positions(model, ids, args.seq_len, args.n_windows, args.batch_size, rng,
                              edges)
    for b in overall:
        print(f"    [{b.lo:>5}, {b.hi:>5}) mean={b.mean:.4f} sem={b.sem:.4f} n={b.n_windows}")

    source_reports: Optional[List[SourcePositionReport]] = None
    if args.blend_manifest is not None:
        if not args.blend.is_file():
            print(f"ERROR: --blend {args.blend} not found (required by --blend-manifest).",
                  file=sys.stderr)
            return 1
        side = args.split or infer_split_side(args.tokens)
        if side is None:
            print(f"ERROR: cannot infer --split ('train' or 'val') from --tokens "
                  f"{args.tokens}'s filename; pass --split explicitly.", file=sys.stderr)
            return 1

        from transformers import AutoTokenizer

        print(f"loading manifest {args.blend_manifest} ...")
        manifest = load_manifest(args.blend_manifest)
        order = emission_order(manifest)
        words_by_source = emitted_words(manifest)
        print(f"  {len(order)} sources, emission order: {order}")

        print(f"loading tokenizer from {args.hf_model} ...")
        tok = AutoTokenizer.from_pretrained(str(args.hf_model))

        print(f"re-tokenizing {args.blend} by source (this replays the entire blend through "
              f"the tokenizer once; it can take several minutes) ...")
        per_source_ids = tokenize_blend_by_source(args.blend, order, words_by_source, tok,
                                                  chunk_lines=args.chunk_lines)

        spans = stratified_source_spans(per_source_ids, args.val_fraction, side)
        print(f"verifying the {side}-side reconstruction against {args.tokens} ...")
        verify_stratified_reconstruction(spans, order, ids)
        print("  OK: reconstructed per-source spans match --tokens byte-for-byte.")

        print(f"probing {args.n_windows} windows per source ...")
        source_reports = build_source_reports(
            manifest, spans, order, model, seq_len=args.seq_len, n_windows=args.n_windows,
            batch_size=args.batch_size, seed=args.seed, edges=edges)
        for r in source_reports:
            if r.buckets is None:
                print(f"    {r.name:22} SKIPPED: {r.note}")
            else:
                print(f"    {r.name:22} "
                      f"[{r.buckets[0].lo},{r.buckets[0].hi}) -> "
                      f"[{r.buckets[-1].lo},{r.buckets[-1].hi}) computed")

    default_out, default_json_out = _default_output_paths(args.hf_model)
    out = args.out or default_out
    json_out = args.json_out or default_json_out

    out.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(overall, hf_model=args.hf_model, tokens_path=args.tokens,
                         seq_len=args.seq_len, n_windows=args.n_windows,
                         batch_size=args.batch_size, source_reports=source_reports,
                         checkpoint_note=args.checkpoint_note)
    out.write_text(md)
    print(f"wrote {out}")

    json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = report_to_json(overall, hf_model=str(args.hf_model), tokens_path=str(args.tokens),
                             seq_len=args.seq_len, n_windows=args.n_windows,
                             batch_size=args.batch_size, seed=args.seed, bucket_edges=edges,
                             source_reports=source_reports)
    json_out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
