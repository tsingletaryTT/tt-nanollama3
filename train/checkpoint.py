# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Checkpoint header schema and thin wrappers over ``ttml.checkpointing``.

ttml already does the hard part — it streams tensors to disk one at a time and writes
atomically (temp file then rename), so a crash mid-write leaves the previous checkpoint
intact. We add two things it deliberately leaves open:

1. **A validated header schema.** ttml's header is an opaque dict, so nothing checks it. A
   checkpoint whose header omits ``vocab_size`` cannot be converted later without guessing,
   and guessing is how a converted model silently mismatches its tokenizer.
2. **Path conventions**, so checkpoints sort by step and a resume can find the newest.

Everything else is a pass-through. Do not reimplement ttml's storage.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Optional

from train.config import SEQ_LEN, VOCAB_SIZE

#: Our header schema version, independent of ttml's own on-disk FORMAT_VERSION.
#: Bump when a field's meaning changes, not when one is added.
CHECKPOINT_FORMAT = 1

#: Fields every checkpoint header must carry. `extra` may not shadow any of these.
#:
#: `corpus_tokens`/`batch_size`/`tokens_seen` replace the old single `total_tokens` field
#: (checkpoint format 1, pre-fix): `total_tokens` recorded the size of the whole corpus
#: (train + val split), not how many tokens this checkpoint's training actually consumed —
#: a model card reading it would silently overstate training volume by ~2.6x for these
#: checkpoints (127.6M corpus vs 49.2M actually trained on at step 3000). `batch_size` makes
#: `tokens_seen = step * batch_size * seq_len` derivable from the header alone, without
#: guessing at a value nothing else records.
_REQUIRED = (
    "format", "step", "vocab_size", "seq_len",
    "model_config_path", "tokenizer_dir",
    "corpus_tokens", "batch_size", "tokens_seen", "created_at",
)


def build_header(
    step: int,
    *,
    model_config_path: str,
    tokenizer_dir: str,
    corpus_tokens: int,
    batch_size: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the header stored alongside a checkpoint's tensors.

    ``vocab_size`` and ``seq_len`` are recorded from ``train.config`` rather than passed in:
    they must describe the model that produced these weights, and taking them from the
    single source of truth removes the chance of a caller recording something else.

    ``corpus_tokens`` is the size of the corpus split the checkpoint was trained against
    (train + val token count) — provenance, not a training-volume claim. ``batch_size`` plus
    ``step`` and ``seq_len`` (already in the header) let us record the number that actually
    matters, ``tokens_seen``, without the caller having to compute or pass it separately.

    ``extra`` is also where a caller should put facts that exist only as hardcoded defaults
    in ttml's C++ (e.g. ``intermediate_dim``, ``weight_tying``, ``rms_norm_eps``) and are not
    recoverable from any yaml or from the checkpoint's own tensors later — see
    ``train/run.py``'s call site for why those three specifically must be captured here, at
    write time.
    """
    header: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "model_config_path": str(model_config_path),
        "tokenizer_dir": str(tokenizer_dir),
        "corpus_tokens": int(corpus_tokens),
        "batch_size": int(batch_size),
        "tokens_seen": int(step) * int(batch_size) * SEQ_LEN,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if extra:
        clashes = sorted(set(extra) & set(_REQUIRED))
        if clashes:
            raise ValueError(f"extra may not override schema field(s): {', '.join(clashes)}")
        header.update(extra)
    return header


def validate_header(header: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``header`` is not a checkpoint header this code can read."""
    missing = [f for f in _REQUIRED if f not in header]
    if missing:
        raise ValueError(f"checkpoint header missing required field(s): {', '.join(missing)}")
    fmt = header["format"]
    if fmt > CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint header format {fmt} is newer than this code understands "
            f"({CHECKPOINT_FORMAT}); upgrade tt-nanollama3 to read it"
        )


def checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    """``<dir>/nanollama3_step<step>.pkl``, zero-padded so paths sort by step.

    Without padding, ``step10`` sorts before ``step9`` and "newest checkpoint" becomes wrong.
    """
    return Path(checkpoint_dir) / f"nanollama3_step{int(step):08d}.pkl"


def latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Highest-step checkpoint in ``checkpoint_dir``, or ``None`` if there are none.

    This is the highest **step**, not the most recently *written* file — with one directory
    shared across runs those can silently differ (an older run's step-5000 checkpoint
    outranks this run's fresh step-100 one, even though the step-100 file has the newer
    mtime). A caller that wants to know which run's weights this actually picked should
    inspect the returned checkpoint's header for ``created_at``, printed by ``--resume``.
    """
    paths = sorted(Path(checkpoint_dir).glob("nanollama3_step*.pkl"))
    return paths[-1] if paths else None


def save(path: Path, *, header: Dict[str, Any], model_params, optimizer,
         display_progress: bool = False) -> None:
    """Write a checkpoint. Pass-through to ttml, which handles atomicity and streaming."""
    from ttml.checkpointing import save_checkpoint

    validate_header(header)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(path), header=header, model_params=model_params,
                    optimizer=optimizer, display_progress=display_progress)


def load(path: Path, *, model_params=None, optimizer=None,
         display_progress: bool = False) -> Dict[str, Any]:
    """Restore a checkpoint in place and return its validated header.

    Validates the header *before* calling ``load_checkpoint``, not after: a bad or
    future-format header should fail fast, without first mutating the live model/optimizer
    with a multi-second tensor load whose result we'd have to discard anyway. ``read_header``
    only reads record 0 (no tensor data), so this costs nothing extra on the success path.
    """
    from ttml.checkpointing import load_checkpoint, read_header

    validate_header(read_header(str(path)))
    return load_checkpoint(str(path), model_params=model_params, optimizer=optimizer,
                           display_progress=display_progress)


def peek(path: Path) -> Dict[str, Any]:
    """Read a checkpoint's header without touching its tensors."""
    from ttml.checkpointing import read_header

    header = read_header(str(path))
    validate_header(header)
    return header
