# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Read a ttml checkpoint's header and tensor manifest without touching ttml/ttnn.

A ttml checkpoint (``ttml.checkpointing.save_checkpoint``) is one pickle record —
``{"format": ..., "header": {...opaque caller header...}, "manifest": {...tensor
skeleton...}}`` — followed by one pickled tensor array per entry the manifest describes.
Reading record 0 needs nothing beyond stdlib ``pickle``: no ttml import, no ttnn import, no
Tenstorrent hardware, no tt-metal checkout on the machine doing the reading. ttml's own
``read_header`` (``ttml/checkpointing.py``) does the same read internally, but importing
that module drags in ``ttml``/``ttnn`` as a side effect even though the read itself doesn't
need them — this module gets the same information without paying that cost.

This is what ``scripts/backfill_checkpoint_headers.py`` uses to rewrite record 0 of the six
existing checkpoints in place, and what any future CPU-side conversion step (turning a
checkpoint into HF format) needs first: the set of tensor names and dtypes, before it can
plan how to read them back out of the file.

``convert/`` never imports ttnn or ttml — see the module purity rule in
``docs/superpowers/specs/2026-08-11-tt-nanollama3-design.md``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_record0(path: Path) -> Tuple[Dict[str, Any], int]:
    """Return (``{"format", "header", "manifest"}``, byte offset right after it).

    The offset is where the first tensor record begins in the file — useful to a caller
    (like the backfill script) that wants to copy the remaining tensor bytes through
    unchanged without ever deserializing them.
    """
    path = Path(path)
    with open(path, "rb") as f:
        try:
            record = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError) as e:
            raise ValueError(f"{path}: could not read checkpoint record 0: {e}") from e
        offset = f.tell()
    if not isinstance(record, dict) or "header" not in record or "manifest" not in record:
        raise ValueError(f"{path}: record 0 is not a ttml checkpoint (missing header/manifest)")
    return record, offset


def read_checkpoint_meta(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(header, manifest)`` for ``path``. No tensor data is read."""
    record, _offset = read_record0(path)
    return record["header"], record["manifest"]


def tensor_names(manifest: Dict[str, Any], group: str = "model") -> List[str]:
    """Flatten ``manifest[group]``'s named parameters into a sorted list of tensor names.

    ``group`` is ``"model"`` or ``"optimizer"`` — whichever top-level key
    ``ttml.checkpointing._walk`` wrote the skeleton under. A group is not always a flat
    ``named_parameters`` leaf (the optimizer nests sub-state, e.g. AdamW's
    ``exp_avg``/``exp_avg_sq``), so this recurses the same way ``_walk``/``_skip`` do —
    every ``named_parameters`` leaf found anywhere under ``group`` contributes its names,
    duplicates included (a tensor with per-state moments appears once per sub-state).
    """
    if group not in manifest:
        raise ValueError(f"checkpoint manifest has no {group!r} group (has {sorted(manifest)})")

    names: List[str] = []

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "named_parameters" in node:
            names.extend(node["named_parameters"].keys())
            return
        for sub in node.values():
            _walk(sub)

    _walk(manifest[group])
    return sorted(names)
