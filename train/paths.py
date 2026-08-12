# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Where artifacts live, once there is more than one model size.

THE PROBLEM THIS SOLVES
-----------------------
Every artifact path in this repo was flat — ``artifacts/checkpoints``, ``artifacts/hf`` —
from when there was exactly one model. With two sizes registered that is no longer safe:
a second size would write its checkpoints on top of the first size's.

Worse, ``artifacts/checkpoints`` and ``artifacts/hf`` hold **the baseline evidence**: the
weights published to the private Hub repo, the checkpoint the NumPy parity gate pins
against, and the only copy of a training run that cost 47 minutes of hardware. Until now
``--checkpoint-dir`` *defaulted* to that directory. A plain ``python train/run.py`` with no
flags would have written into it.

THE RULE
--------
**Reads fall back to the legacy flat path; writes never do.**

- :func:`write_dir` always returns ``artifacts/<size>/<kind>`` and refuses, loudly, to
  return a protected path. New runs cannot land on the baseline however they are invoked.
- :func:`read_dir` prefers ``artifacts/<size>/<kind>`` but falls back to the flat legacy
  path when the per-size one does not exist — so the converter, the parity gate, and the
  publish scripts keep finding the baseline artifacts exactly where they are today.

That asymmetry is deliberate. It means this change protects the evidence without moving a
single existing byte, which is the safest possible migration for data that cannot be
regenerated without retraining.

SHARED VS PER-SIZE
------------------
Not everything is per-size. The tokenizer, the tokenised corpus, and the raw download are
**shared**: every size in this repo uses the same 32,000-token vocabulary, which is much of
the reason they live in one repository at all. Only trained outputs — checkpoints and the
converted HF directory — are per-size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

from train.sizes import ModelSize, get_size

#: Repository root (``train/`` -> parent).
ROOT = Path(__file__).resolve().parent.parent

#: Artifact kinds every size shares, because every size shares one tokenizer.
#: Keyed by kind name; the value is the flat directory under ``artifacts/``.
SHARED_KINDS = frozenset({"tokenizer", "tokens", "corpus", "raw"})

#: Artifact kinds that belong to a single trained model.
PER_SIZE_KINDS = frozenset({"checkpoints", "hf"})

#: The one size the flat pre-registry layout belonged to.
#:
#: ``artifacts/checkpoints`` and ``artifacts/hf`` were written before any size existed, by
#: and for the 384 model. The legacy fallback in :func:`read_dir` therefore applies to
#: **this size only**. Letting it apply to every size would be actively dangerous: asking
#: for the untrained 1024 model's checkpoints would silently hand back the 384 baseline,
#: and the converter would produce a confidently wrong model from it.
LEGACY_SIZE = "384"

#: Directories holding irreplaceable evidence, relative to ``artifacts/``.
#:
#: ``checkpoints`` is the 3000-step baseline the parity gate pins against; ``hf`` is the
#: converted artifact uploaded to the private Hub repo. Neither can be regenerated without
#: retraining, so :func:`write_dir` refuses to hand either out as a write target.
PROTECTED_RELATIVE = frozenset({"checkpoints", "hf"})


class ProtectedPathError(RuntimeError):
    """Raised when something asks to write into irreplaceable evidence."""


def _artifacts(root: Optional[Path] = None) -> Path:
    return (Path(root) if root else ROOT) / "artifacts"


def protected_dirs(root: Optional[Path] = None) -> Iterable[Path]:
    """The concrete paths :func:`write_dir` will refuse to return."""
    base = _artifacts(root)
    return tuple(base / name for name in sorted(PROTECTED_RELATIVE))


def assert_not_protected(path: Union[str, Path], root: Optional[Path] = None) -> Path:
    """Raise :class:`ProtectedPathError` if ``path`` is baseline evidence.

    Compared by resolved path so ``artifacts/../artifacts/hf`` and a symlink cannot slip
    through. Call this on any directory about to be written to, including one a user
    supplied on the command line — the point is that no invocation reaches the baseline,
    not merely that the default does not.
    """
    resolved = Path(path).expanduser().resolve()
    for guarded in protected_dirs(root):
        if resolved == guarded.resolve():
            raise ProtectedPathError(
                f"refusing to write to {resolved}: this is protected baseline evidence "
                f"(the published weights and the checkpoint the parity gate pins "
                f"against), and it cannot be regenerated without retraining. Write to a "
                f"per-size directory instead, e.g. "
                f"{_artifacts(root) / '<size>' / resolved.name}."
            )
    return Path(path)


def shared_dir(kind: str, root: Optional[Path] = None) -> Path:
    """Path for an artifact kind shared by every size (tokenizer, tokens, corpus, raw)."""
    if kind not in SHARED_KINDS:
        raise ValueError(
            f"{kind!r} is not a shared artifact kind; shared kinds are "
            f"{sorted(SHARED_KINDS)}. For per-size kinds use write_dir()/read_dir()."
        )
    return _artifacts(root) / kind


def write_dir(
    size: Union[str, ModelSize, None] = None,
    kind: str = "checkpoints",
    root: Optional[Path] = None,
) -> Path:
    """Where a training or conversion run for ``size`` should WRITE ``kind``.

    Always ``artifacts/<size>/<kind>``. Never the legacy flat path, so no invocation —
    default, explicit, or scripted — can land on the baseline.
    """
    if kind in SHARED_KINDS:
        return shared_dir(kind, root)
    if kind not in PER_SIZE_KINDS:
        raise ValueError(
            f"unknown artifact kind {kind!r}; expected one of "
            f"{sorted(PER_SIZE_KINDS | SHARED_KINDS)}"
        )
    resolved = size if isinstance(size, ModelSize) else get_size(size)
    target = _artifacts(root) / resolved.name / kind
    # Defence in depth: a size literally named "checkpoints" or "hf" would otherwise
    # produce artifacts/checkpoints/checkpoints, which is not protected but is confusing.
    return assert_not_protected(target, root)


def read_dir(
    size: Union[str, ModelSize, None] = None,
    kind: str = "checkpoints",
    root: Optional[Path] = None,
) -> Path:
    """Where to READ ``kind`` for ``size``, tolerating the pre-registry flat layout.

    Prefers ``artifacts/<size>/<kind>``. Falls back to the flat ``artifacts/<kind>`` when
    the per-size directory does not exist but the flat one does — which is how the baseline
    384 artifacts stay findable without being moved.

    **The fallback applies to :data:`LEGACY_SIZE` only.** The flat directories were written
    by and for that model; handing them to any other size would silently substitute the
    wrong weights — asking for the untrained 1024 checkpoints would return the 384 baseline
    and the converter would build a confidently wrong model from it.

    Returns the per-size path when nothing exists, so error messages point at where the
    artifact *should* be rather than at a legacy location.
    """
    if kind in SHARED_KINDS:
        return shared_dir(kind, root)
    resolved = size if isinstance(size, ModelSize) else get_size(size)
    per_size = _artifacts(root) / resolved.name / kind
    if per_size.exists():
        return per_size
    if resolved.name == LEGACY_SIZE:
        legacy = _artifacts(root) / kind
        if legacy.exists():
            return legacy
    return per_size


__all__ = [
    "LEGACY_SIZE",
    "PER_SIZE_KINDS",
    "PROTECTED_RELATIVE",
    "ProtectedPathError",
    "ROOT",
    "SHARED_KINDS",
    "assert_not_protected",
    "protected_dirs",
    "read_dir",
    "shared_dir",
    "write_dir",
]
