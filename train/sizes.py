# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The model-size registry — one entry per architecture this repository ships.

WHY THIS EXISTS
---------------
Until this module, the architecture came from ``$TT_METAL_HOME/tt-train/configs/
model_configs/nanollama3.yaml``: a file we did not own, could not vary, and which could
change under a tt-metal upgrade without any signal. That made a second model size
impossible and made the first one fragile.

Now every architecture lives in ``train/configs/model/`` and is described here. The YAML
remains the file ttml actually reads; this registry is the *typed* description of it, plus
the derived hardware facts that YAML cannot express. ``tests/test_sizes.py`` asserts the
two never drift apart.

WHAT THE DERIVED PROPERTIES ARE FOR
-----------------------------------
Choosing ``embedding_dim`` and ``num_groups`` for a Tenstorrent model is not a free choice
the way it is on a GPU — both decide, at design time, how much of the die the model can
use and which meshes it can be served on. Those consequences were discovered the expensive
way on the 384 model (see ``docs/multi-chip-notes.md`` and the 384 YAML's header) and are
encoded here so the next size is chosen with the numbers visible rather than by convention:

- ``tiles`` / ``best_core_grid`` / ``core_utilisation`` — how much of the compute grid a
  tensor of this width can actually occupy.
- ``servable_mesh_widths`` — which mesh shapes ``tt_transformers`` will accept, from its
  ``n_heads % cluster_shape[1] == 0 and n_kv_heads % cluster_shape[1] == 0`` assertion
  (``models/tt_transformers/tt/model_config.py:687-691``).
- ``tensor_parallel_capable`` — whether the hidden dimension splits into whole 32-wide
  tiles per chip.

Nothing here talks to hardware; it is pure arithmetic over the config, so it is testable
without a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

#: Directory holding the vendored ttml model configs, one per registered size.
MODEL_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "model"

#: A Tenstorrent tile is 32x32. Every width/height that wants to shard cleanly across
#: cores has to be counted in these, not in elements.
TILE = 32

#: Compute grid of the Blackhole p300c this project was developed on, as
#: ``(rows, cols)``, read from ``mesh_device.compute_with_storage_grid_size()``.
#:
#: **This is a harvested part: 11 columns, not the architectural 12+.** That single fact
#: is why ``ModelArgs.find_grid`` fails for a 384-wide model (it hardcodes ``max_cols=12``
#: and never asks the device) and why ``bundle/tt_nanollama3_adapter.py`` ships a shim.
#: Treat it as the default for utilisation arithmetic, not as a universal constant —
#: pass a different grid for a different part.
BLACKHOLE_P300C_GRID: Tuple[int, int] = (10, 11)


@dataclass(frozen=True)
class ModelSize:
    """One architecture: the ttml config fields, plus where its YAML lives.

    Field names deliberately match ttml's ``transformer_config`` keys so that comparing a
    registry entry against its YAML is a straight key-for-key check with no translation
    layer to get wrong.
    """

    #: Registry key and filename stem, e.g. ``"384"`` -> ``nanollama3-384.yaml``. Also the
    #: suffix used for per-size artifact directories and Hub repo ids.
    name: str

    embedding_dim: int
    num_blocks: int
    num_heads: int
    num_groups: int          # GQA key/value groups (HF: num_key_value_heads)
    vocab_size: int
    max_sequence_length: int
    theta: float
    dropout_prob: float = 0.0
    model_type: str = "llama"
    runner_type: str = "default"

    #: One-line note on why this size exists. Shown by ``describe()``.
    rationale: str = ""

    # -- paths ---------------------------------------------------------------------

    @property
    def config_filename(self) -> str:
        return f"nanollama3-{self.name}.yaml"

    @property
    def config_path(self) -> Path:
        """Absolute path to the ttml model config for this size."""
        return MODEL_CONFIG_DIR / self.config_filename

    def artifact_dir(self, root: Path, kind: str) -> Path:
        """Per-size artifact location, e.g. ``artifacts/384/checkpoints``.

        Keeping sizes in separate subtrees is what stops a second size from overwriting
        the first one's checkpoints — the failure this repo has explicitly protected
        against since the baseline run.
        """
        return Path(root) / "artifacts" / self.name / kind

    # -- shape arithmetic ----------------------------------------------------------

    @property
    def head_dim(self) -> int:
        if self.embedding_dim % self.num_heads:
            raise ValueError(
                f"{self.name}: embedding_dim {self.embedding_dim} is not divisible by "
                f"num_heads {self.num_heads}"
            )
        return self.embedding_dim // self.num_heads

    @property
    def tiles(self) -> int:
        """Hidden dimension measured in 32-wide tiles — the unit core sharding works in."""
        if self.embedding_dim % TILE:
            raise ValueError(
                f"{self.name}: embedding_dim {self.embedding_dim} is not a multiple of "
                f"the {TILE}-element tile width"
            )
        return self.embedding_dim // TILE

    # -- hardware consequences -----------------------------------------------------

    def best_core_grid(
        self, grid: Tuple[int, int] = BLACKHOLE_P300C_GRID
    ) -> Optional[Tuple[int, int, int]]:
        """Largest ``(cores, rows, cols)`` within ``grid`` that divides ``tiles`` evenly.

        This is the *utilisation-maximising* answer, deliberately different from
        ``ModelArgs.find_grid``, which sorts candidates by closeness to a hardcoded target
        of 32 cores and so leaves most of the die idle on any grid larger than that. At
        ``embedding_dim=3520`` (110 tiles) stock picks 22 cores where 110 are available.

        Returns None when no sub-grid divides the tile count.
        """
        max_rows, max_cols = grid
        best: Optional[Tuple[int, int, int]] = None
        for rows in range(1, max_rows + 1):
            for cols in range(1, max_cols + 1):
                cores = rows * cols
                if self.tiles % cores == 0 and (best is None or cores > best[0]):
                    best = (cores, rows, cols)
        return best

    def core_utilisation(self, grid: Tuple[int, int] = BLACKHOLE_P300C_GRID) -> float:
        """Fraction of the compute grid a hidden-dimension shard can occupy (0.0-1.0)."""
        best = self.best_core_grid(grid)
        if best is None:
            return 0.0
        return best[0] / (grid[0] * grid[1])

    def servable_mesh_widths(self, max_width: int = 8) -> List[int]:
        """Mesh column counts ``tt_transformers`` will accept for this architecture.

        Mirrors the assertion pair at ``model_config.py:687-691``: both the query heads and
        the KV groups must divide the mesh width. ``num_groups`` is the smaller number and
        therefore the binding one — ``num_groups=3`` restricts serving to widths {1, 3},
        which excludes every common multi-chip mesh.
        """
        return [
            w
            for w in range(1, max_width + 1)
            if self.num_heads % w == 0 and self.num_groups % w == 0
        ]

    def tensor_parallel_capable(self, chips: int) -> bool:
        """Whether the hidden dimension splits into whole tiles across ``chips``."""
        return chips in self.servable_mesh_widths(max(chips, 1)) and (
            self.embedding_dim % (TILE * chips) == 0
        )

    # -- serialisation -------------------------------------------------------------

    def to_transformer_config(self) -> dict:
        """The ``transformer_config`` block as ttml expects to read it from YAML."""
        return {
            "model_type": self.model_type,
            "num_heads": self.num_heads,
            "num_groups": self.num_groups,
            "embedding_dim": self.embedding_dim,
            "dropout_prob": self.dropout_prob,
            "num_blocks": self.num_blocks,
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "runner_type": self.runner_type,
            "theta": self.theta,
        }

    def load_yaml_transformer_config(self) -> dict:
        """Read this size's YAML and return its ``transformer_config`` block."""
        with self.config_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)["transformer_config"]

    def describe(self, grid: Tuple[int, int] = BLACKHOLE_P300C_GRID) -> str:
        """Human-readable summary, including the hardware consequences."""
        best = self.best_core_grid(grid)
        grid_s = f"{best[1]}x{best[2]} = {best[0]} cores" if best else "no fitting grid"
        return (
            f"{self.name}: dim={self.embedding_dim} ({self.tiles} tiles), "
            f"blocks={self.num_blocks}, heads={self.num_heads}/{self.num_groups} groups, "
            f"head_dim={self.head_dim}, seq={self.max_sequence_length}\n"
            f"  best core grid : {grid_s} "
            f"({100 * self.core_utilisation(grid):.0f}% of {grid[0]}x{grid[1]})\n"
            f"  servable meshes: widths {self.servable_mesh_widths()}\n"
            f"  {self.rationale}"
        )


#: Every architecture this repository ships, keyed by ``ModelSize.name``.
#:
#: Adding a size means: a new YAML in ``train/configs/model/``, a new entry here, and
#: nothing else — the tests parametrise over this dict, so coverage follows automatically.
SIZES: Dict[str, ModelSize] = {
    "384": ModelSize(
        name="384",
        embedding_dim=384,
        num_blocks=6,
        num_heads=6,
        num_groups=3,
        vocab_size=32000,
        max_sequence_length=256,
        theta=500000.0,
        rationale=(
            "The original: matches tt-train's shipped nanollama3 example. Trained, "
            "converted, and published. Single-chip only (num_groups=3 admits mesh widths "
            "{1,3}) and 11% core utilisation — both consequences of a shape chosen before "
            "either was measured. Kept as the baseline, not as a template."
        ),
    ),
}

#: The size used when nothing is specified. Deliberately the original model, so every
#: existing command line keeps its current meaning.
DEFAULT_SIZE = "384"


def get_size(name: Optional[str] = None) -> ModelSize:
    """Look up a registered size, defaulting to :data:`DEFAULT_SIZE`.

    Raises ``KeyError`` with the available names rather than a bare miss, because this is
    reached straight from a CLI flag.
    """
    key = name or DEFAULT_SIZE
    try:
        return SIZES[key]
    except KeyError:
        raise KeyError(
            f"unknown model size {key!r}; registered sizes: {sorted(SIZES)}"
        ) from None


__all__ = [
    "BLACKHOLE_P300C_GRID",
    "DEFAULT_SIZE",
    "MODEL_CONFIG_DIR",
    "ModelSize",
    "SIZES",
    "TILE",
    "get_size",
]
