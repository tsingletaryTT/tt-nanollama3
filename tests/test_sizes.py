# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the model-size registry.

The load-bearing test here is :func:`test_registry_matches_its_yaml`. The registry and the
YAML describe the same architecture in two places: the YAML is what ttml actually reads at
training time, the registry is what every other tool in this repo reasons about. If they
drift, training and packaging silently describe different models — the exact class of
defect this project keeps finding (a norm mis-mapping, a frozen gamma, a mis-declared
capability) where nothing crashes and the numbers merely become wrong.

Everything here is pure arithmetic and file reading. No hardware, no device.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from train.sizes import (
    BLACKHOLE_P300C_GRID,
    DEFAULT_SIZE,
    MODEL_CONFIG_DIR,
    SIZES,
    TILE,
    ModelSize,
    get_size,
)

ALL_SIZES = sorted(SIZES)


@pytest.mark.parametrize("name", ALL_SIZES)
def test_every_registered_size_has_a_config_file(name):
    """A registry entry with no YAML is a training run that dies at startup."""
    size = SIZES[name]
    assert size.config_path.is_file(), (
        f"{name} is registered but {size.config_path} does not exist"
    )


@pytest.mark.parametrize("name", ALL_SIZES)
def test_registry_matches_its_yaml(name):
    """THE anti-drift gate: the typed entry and the file ttml reads must agree.

    Compared key-for-key rather than as a whole-dict equality so a mismatch names the
    offending field instead of dumping two dicts.
    """
    size = SIZES[name]
    from_yaml = size.load_yaml_transformer_config()
    from_registry = size.to_transformer_config()

    assert set(from_yaml) == set(from_registry), (
        f"{name}: key mismatch between YAML and registry; "
        f"only in YAML: {set(from_yaml) - set(from_registry)}, "
        f"only in registry: {set(from_registry) - set(from_yaml)}"
    )
    for key in sorted(from_registry):
        assert from_yaml[key] == from_registry[key], (
            f"{name}: {key} is {from_yaml[key]!r} in {size.config_filename} but "
            f"{from_registry[key]!r} in train/sizes.py"
        )


@pytest.mark.parametrize("name", ALL_SIZES)
def test_no_stray_config_files(name):
    """Every YAML in the config dir is registered — an unregistered one is dead weight."""
    registered = {SIZES[n].config_filename for n in SIZES}
    on_disk = {p.name for p in MODEL_CONFIG_DIR.glob("*.yaml")}
    assert on_disk <= registered, (
        f"unregistered model config(s) present: {sorted(on_disk - registered)}"
    )


def test_vendored_384_is_faithful_to_tt_metal_original():
    """The 384 config claims to be a verbatim copy of tt-train's. Hold it to that.

    Skipped when tt-metal is not present (CI, a clone without TT_METAL_HOME). When it *is*
    present this catches both a bad transcription now and an upstream change later --
    upstream drifting is informative, not a failure of ours, so the message says so.
    """
    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if not tt_metal_home:
        pytest.skip("TT_METAL_HOME not set")
    upstream = (
        Path(tt_metal_home) / "tt-train/configs/model_configs/nanollama3.yaml"
    )
    if not upstream.is_file():
        pytest.skip(f"upstream config not found at {upstream}")

    with upstream.open("r", encoding="utf-8") as fh:
        upstream_cfg = yaml.safe_load(fh)["transformer_config"]
    ours = SIZES["384"].load_yaml_transformer_config()

    assert ours == upstream_cfg, (
        "train/configs/model/nanollama3-384.yaml no longer matches tt-train's "
        f"nanollama3.yaml at {upstream}. If upstream changed, that is worth knowing but "
        "is not automatically a bug in this repo: decide whether to follow it, then "
        "update the vendored copy and its provenance header together."
    )


# --------------------------------------------------------------------------------------
# Derived hardware arithmetic. The 384 values are pinned to what was MEASURED on the
# p300c this session, so a refactor of the arithmetic cannot quietly change them.
# --------------------------------------------------------------------------------------


def test_384_tiles_and_core_grid_match_measured_values():
    size = SIZES["384"]
    assert size.tiles == 12, "384 / 32 = 12 tiles"
    cores, rows, cols = size.best_core_grid(BLACKHOLE_P300C_GRID)
    assert (cores, rows, cols) == (12, 2, 6), (
        "measured on p300c: 12 tiles best-fit a 2x6 block within the 11x10 grid"
    )
    assert round(size.core_utilisation(BLACKHOLE_P300C_GRID), 3) == round(12 / 110, 3)


def test_384_is_single_chip_only():
    """num_groups=3 admits mesh widths {1,3}; every common multi-chip mesh is excluded.

    This is the constraint that decided the serving story for this model, so it is pinned.
    """
    assert SIZES["384"].servable_mesh_widths(max_width=8) == [1, 3]
    assert not SIZES["384"].tensor_parallel_capable(2)
    assert not SIZES["384"].tensor_parallel_capable(4)


def test_384_head_dim():
    assert SIZES["384"].head_dim == 64


@pytest.mark.parametrize("name", ALL_SIZES)
def test_dimensions_are_tile_aligned(name):
    """A hidden dimension that is not a multiple of 32 cannot shard onto cores at all."""
    size = SIZES[name]
    assert size.embedding_dim % TILE == 0
    assert size.tiles >= 1


@pytest.mark.parametrize("name", ALL_SIZES)
def test_heads_divide_the_hidden_dimension(name):
    size = SIZES[name]
    assert size.embedding_dim % size.num_heads == 0
    assert size.num_heads % size.num_groups == 0, (
        f"{name}: num_heads {size.num_heads} must be a whole multiple of num_groups "
        f"{size.num_groups} for grouped-query attention"
    )


# --------------------------------------------------------------------------------------
# Registry mechanics
# --------------------------------------------------------------------------------------


def test_default_size_is_registered_and_is_the_original():
    assert DEFAULT_SIZE in SIZES
    assert DEFAULT_SIZE == "384", (
        "the default must stay the originally-trained model so existing command lines "
        "keep their meaning"
    )


def test_get_size_defaults_and_rejects_unknown():
    assert get_size() is SIZES[DEFAULT_SIZE]
    assert get_size("384") is SIZES["384"]
    with pytest.raises(KeyError) as excinfo:
        get_size("does-not-exist")
    assert "registered sizes" in str(excinfo.value), (
        "the error should list what IS available; this is reached straight from a CLI flag"
    )


def test_artifact_dirs_are_separated_per_size(tmp_path):
    """Two sizes must not be able to overwrite each other's checkpoints."""
    a = ModelSize(
        name="a", embedding_dim=64, num_blocks=1, num_heads=2, num_groups=1,
        vocab_size=32, max_sequence_length=32, theta=1.0,
    )
    b = ModelSize(
        name="b", embedding_dim=64, num_blocks=1, num_heads=2, num_groups=1,
        vocab_size=32, max_sequence_length=32, theta=1.0,
    )
    assert a.artifact_dir(tmp_path, "checkpoints") != b.artifact_dir(tmp_path, "checkpoints")
    assert a.artifact_dir(tmp_path, "checkpoints").name == "checkpoints"
    assert a.artifact_dir(tmp_path, "checkpoints").parent.name == "a"


@pytest.mark.parametrize("name", ALL_SIZES)
def test_describe_mentions_the_measured_consequences(name):
    """describe() is what a human reads before picking a size; it must show the trade-offs."""
    text = SIZES[name].describe()
    assert "cores" in text
    assert "servable meshes" in text
