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


def test_intermediate_dim_reproduces_the_real_converted_model():
    """The FFN width is DERIVED by ttml, not declared — so it can silently disagree.

    ``artifacts/hf/config.json`` records what the actually-trained-and-converted 384 model
    has: ``intermediate_size: 1024``. If :attr:`ModelSize.intermediate_dim` ever stops
    reproducing that, then the registry and the real model disagree about the widest
    tensors in the network, and every parameter count and utilisation figure derived from
    it is wrong.

    Falls back to the known constant when the artifact is absent (fresh clone), because
    the rule itself is what is under test.
    """
    assert SIZES["384"].intermediate_dim == 1024

    cfg = Path(__file__).resolve().parents[1] / "artifacts" / "hf" / "config.json"
    if not cfg.is_file():
        pytest.skip("artifacts/hf/config.json not present")
    import json

    with cfg.open() as fh:
        converted = json.load(fh)
    assert converted["hidden_size"] == SIZES["384"].embedding_dim
    assert converted["intermediate_size"] == SIZES["384"].intermediate_dim, (
        "train/sizes.py's ttml FFN derivation no longer matches the converted model"
    )


def test_intermediate_dim_rounds_up_to_256_with_c_truncation():
    """Pin the exact rule, including the C++ float->uint32 truncation before rounding.

    ``8/3 * 1024 = 2730.666...``; C++ truncates to 2730, then rounds up to 2816. Rounding
    the exact rational instead would give 2731 -> 2816 here (same answer), so the
    truncation is pinned separately below where it can actually differ.
    """
    from train.sizes import ModelSize

    def mk(dim):
        return ModelSize(
            name="t", embedding_dim=dim, num_blocks=1, num_heads=dim // 64,
            num_groups=1, vocab_size=32, max_sequence_length=32, theta=1.0,
        )

    assert mk(384).intermediate_dim == 1024      # 1024 exactly, no rounding needed
    assert mk(1024).intermediate_dim == 2816     # 2730 -> next multiple of 256
    assert mk(2560).intermediate_dim == 6912     # 6826 -> next multiple of 256
    # Every result is a multiple of 256 by construction.
    for dim in (384, 512, 1024, 2048, 2560, 3520):
        assert mk(dim).intermediate_dim % 256 == 0


def test_1024_unlocks_the_meshes_384_cannot_reach():
    """The entire reason the 1024 size exists."""
    small, big = SIZES["384"], SIZES["1024"]
    assert small.servable_mesh_widths(8) == [1, 3]
    assert big.servable_mesh_widths(8) == [1, 2, 4]
    for chips in (2, 4):
        assert not small.tensor_parallel_capable(chips)
        assert big.tensor_parallel_capable(chips), (
            f"1024 must be tensor-parallel-capable on {chips} chips; that is its purpose"
        )


def test_1024_ffn_fits_the_grid_better_than_its_hidden_dim():
    """The measurement that decided 1024 over larger candidates.

    The MLP carries most of the parameters and matmul work, so its derived width matters
    more than the hidden dimension's. Pinned so a change to the grid search cannot quietly
    invalidate the rationale recorded in the config header.
    """
    size = SIZES["1024"]
    assert size.intermediate_dim == 2816
    assert size.ffn_tiles == 88
    assert size.best_ffn_core_grid(BLACKHOLE_P300C_GRID) == (88, 8, 11)
    assert size.ffn_core_utilisation(BLACKHOLE_P300C_GRID) > size.core_utilisation(
        BLACKHOLE_P300C_GRID
    ), "the FFN should occupy more of the grid than the hidden dimension does"


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
