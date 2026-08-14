# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Self-consistency gates for the v4 manifests in ``manifests/``.

WHY THESE EXIST
---------------
tt-kernel validates a manifest's *shape* — types, required fields, PEP 440 ranges — but not
its *coherence*. Several ways of writing a manifest that parses cleanly will serve the
wrong thing, silently:

- ``mesh.devices: 4`` with ``env.MESH_DEVICE`` naming a single-chip preset. These are two
  independent channels: the whole ``mesh`` block contributes **nothing** to the rendered
  ``vllm_metadata.json`` (verified: it has exactly four keys — ``arch``, ``hf_weights``,
  ``launch``, ``main_class``), and the mesh shape the plugin actually opens comes only from
  the ``MESH_DEVICE`` string. Nothing cross-checks them, so a bundle can advertise four
  chips and serve on one.
- ``resources.max_model_len`` disagreeing with the architecture's trained context. The
  tokenizer advertises ``model_max_length: 1000000000000000019884624838656``, so anything
  deriving a context from it gets a serving stack accepting 4k prompts from a model trained
  to 256 — degraded output, no error.
- An architecture whose ``num_groups`` cannot shard onto the mesh the manifest claims, which
  fails at ``model_config.py:687-691`` only once a device is open.

Each is caught here, against the size registry, with no hardware and no Hub access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.sizes import SIZES, get_size

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"

#: Mesh presets, mirroring ``vllm_tt_plugin/utils/dp_discovery.py::_MESH_GRID_PRESETS``.
#:
#: Vendored rather than imported because the plugin is not installed in the environment
#: these tests run in. If the plugin's table gains an entry this copy will not know about
#: it — the failure mode is a spurious "unknown preset" here, which is loud, not silent.
MESH_PRESETS = {
    "N150": 1, "P100": 1, "P150": 1,
    "N300": 2, "P300": 2, "P150x2": 2,
    "N150x4": 4, "P150x4": 4, "P300x2": 4,
    "T3K": 8, "P150x8": 8,
}


def _manifests():
    return sorted(MANIFEST_DIR.glob("tt_kernel_manifest-*.json"))


def _size_of(path: Path):
    """The registry entry a manifest is for, from its filename suffix."""
    return get_size(path.stem.rsplit("-", 1)[-1])


def _mesh_devices_from_env(env: dict) -> int | None:
    """Device count implied by ``MESH_DEVICE``, or None if unset/unparseable.

    Accepts the plugin's two accepted forms: a preset name, or a literal ``"(r, c)"``
    tuple (``_parse_mesh_grid`` ``ast.literal_eval``s the latter).
    """
    raw = (env or {}).get("MESH_DEVICE")
    if raw is None:
        return None
    if raw in MESH_PRESETS:
        return MESH_PRESETS[raw]
    try:
        import ast

        val = ast.literal_eval(raw)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return int(val[0]) * int(val[1])
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def test_manifest_directory_is_not_empty():
    assert _manifests(), f"no manifests found in {MANIFEST_DIR}"


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_manifest_is_valid_v4(path):
    """Parses as a v4 unified manifest once tt-kernel's push-time fields are supplied."""
    pytest.importorskip("tt_kernel", reason="tt-kernel not importable")
    from tt_kernel.manifest import Manifest

    raw = json.loads(path.read_text())
    # tt-kernel generates these at push from the detected environment; supply stand-ins so
    # the authored blocks can be validated on their own.
    generated = {
        "tt_metal_version": "test",
        "arch": "blackhole",
        "producer": {"tt_kernel_version": "0.0.0", "created_at": "1970-01-01T00:00:00Z"},
        "runner": {"backend": "vllm", "bundle_dir": "bundle"},
    }
    m = Manifest.model_validate({**raw, **generated})
    assert m.is_v4, f"{path.name} does not read as a v4 unified manifest"
    assert m.entrypoint is not None


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_mesh_devices_agrees_with_mesh_device_env(path):
    """THE GATE tt-kernel does not provide.

    ``mesh.devices`` and ``env.MESH_DEVICE`` are independent channels; only the latter
    reaches the plugin. A manifest claiming four chips while naming a one-chip preset
    parses cleanly and serves on one chip.
    """
    raw = json.loads(path.read_text())
    declared = (raw.get("mesh") or {}).get("devices")
    if declared is None:
        pytest.skip("manifest declares no mesh.devices")

    env_devices = _mesh_devices_from_env(raw.get("env"))
    assert env_devices is not None, (
        f"{path.name}: mesh.devices={declared} but env.MESH_DEVICE is unset or is not a "
        f"recognised preset/tuple. MESH_DEVICE is the ONLY channel that reaches the "
        f"plugin — without it the mesh defaults to every visible device."
    )
    assert env_devices == declared, (
        f"{path.name}: mesh.devices={declared} but env.MESH_DEVICE implies {env_devices} "
        f"device(s). Nothing in tt-kernel cross-checks these, so this would serve on "
        f"{env_devices} chip(s) while advertising {declared}."
    )


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_architecture_can_shard_onto_the_declared_mesh(path):
    """num_groups must divide the mesh width, or serving dies once a device is open.

    ``model_config.py:687-691`` asserts both ``n_heads`` and ``n_kv_heads`` divide
    ``cluster_shape[1]``. Catching it here costs nothing; catching it on hardware costs a
    device open and an opaque assertion.
    """
    raw = json.loads(path.read_text())
    size = _size_of(path)
    declared = (raw.get("mesh") or {}).get("devices", 1)
    assert declared in size.servable_mesh_widths(max(declared, 1)), (
        f"{path.name}: declares mesh.devices={declared}, but size {size.name} has "
        f"num_heads={size.num_heads}/num_groups={size.num_groups}, which admits only "
        f"widths {size.servable_mesh_widths(8)}"
    )


#: The context each manifest's *actually-published* weights were trained to, independent of
#: ``train/sizes.py``'s ``max_sequence_length``.
#:
#: Before 2026-08-13 these two numbers were always the same value, because
#: ``ModelSize.max_sequence_length`` had only one meaning: "what this architecture was
#: trained to." Raising ``seq_len`` to 512 for the next training run (see
#: ``.superpowers/seqlen-ddp-investigation.md``) gave it a second, forward-looking meaning
#: -- "what the vendored YAML currently targets for the NEXT run of this architecture" --
#: and for a day those two meanings legitimately disagreed for ``384``, so this constant
#: held 256 against a registry that already said 512.
#:
#: **That gap closed for 384 at 512, and has now closed again at 2048.** The size registry
#: moved ``384`` to ``max_sequence_length=2048``, and ``episod/tt-tnt`` was retrained at
#: that context as tt-tnt-v3 (``artifacts/checkpoints-tt-tnt-v3``, 10,764 steps, final
#: val_loss 2.939, against v1's 4.31 on a differently-composed val split) and republished.
#: Verified directly rather than taken on trust:
#: ``artifacts/hf-tt-tnt-v3/config.json`` reads ``max_position_embeddings: 2048``, and its
#: ``model.safetensors`` (sha256 ``97e19118...``) is what was uploaded -- distinct from
#: v1's ``dbc46211...``. The manifest that serves those weights must therefore declare
#: 2048, or serving silently truncates three quarters of the model's trained context --
#: the same trap this test exists to catch, from the other direction, and a bigger bite of
#: it than the 512 case was.
#:
#: ``1024`` still has no published weights at all (``episod/tt-tnt-1024`` does not exist on
#: the Hub -- checked, ``RepositoryNotFoundError``), so its entry is not a *published* fact
#: but a placeholder. It tracks the registry's design target, which is what that size will
#: actually be trained to when it is trained, so that the packaging path is exercised
#: against a coherent number rather than one left behind by an earlier architecture.
#:
#: Update a *published* entry here only once that size's weights are ACTUALLY retrained and
#: republished at the new context length -- not when the registry's design target changes.
_PUBLISHED_CONTEXT = {"384": 2048, "1024": 512}


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_max_model_len_matches_the_trained_context(path):
    """Guards the documented serving trap.

    ``tokenizer_config.json`` advertises a model_max_length of ~1e18, so any stack deriving
    the context from the tokenizer will happily accept prompts far beyond what the model was
    trained on and return degraded output with no error.

    Compared against ``_PUBLISHED_CONTEXT`` (what the manifest's ``weights.repo`` was
    ACTUALLY trained to), not ``size.max_sequence_length`` (the registry's current design
    target for the architecture) -- see that constant's docstring for why those can now
    differ without either one being wrong.
    """
    raw = json.loads(path.read_text())
    size = _size_of(path)
    declared = (raw.get("resources") or {}).get("max_model_len")
    expected = _PUBLISHED_CONTEXT[size.name]
    assert declared == expected, (
        f"{path.name}: resources.max_model_len={declared} but the published weights this "
        f"manifest serves were trained to {expected}"
    )


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_max_model_len_is_a_power_of_two(path):
    """``model_config.py:1150-1152`` requires capped_warmup_seq_len to be a power of two.

    256 is fine; 200 would hard-fail at startup with an unrelated-looking message.
    """
    declared = (json.loads(path.read_text()).get("resources") or {}).get("max_model_len")
    if declared is None:
        pytest.skip("no max_model_len declared")
    assert declared & (declared - 1) == 0, (
        f"{path.name}: max_model_len={declared} is not a power of two"
    )


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_weights_repo_and_hf_model_env_agree(path):
    """``HF_MODEL`` must be set, and must name the same repo as ``weights.repo``.

    tt_transformers reads the model id from ``HF_MODEL``, not from the ``--model`` argument
    tt-kernel composes; an unset value raises at engine init and a *mismatched* one would
    serve different weights than the bundle advertises. tt-kernel warns about neither.
    """
    raw = json.loads(path.read_text())
    repo = (raw.get("weights") or {}).get("repo")
    hf_model = (raw.get("env") or {}).get("HF_MODEL")
    assert repo, f"{path.name}: no weights.repo"
    assert hf_model, (
        f"{path.name}: env.HF_MODEL is unset. tt_transformers raises "
        f"'Please set HF_MODEL to a HuggingFace name' at engine init; the composed "
        f"--model argument does not satisfy it."
    )
    assert hf_model == repo, (
        f"{path.name}: weights.repo={repo!r} but env.HF_MODEL={hf_model!r}"
    )


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_entrypoint_points_at_our_bundle_adapter(path):
    """The entrypoint module must be the one the bundle folder actually ships."""
    raw = json.loads(path.read_text())
    cls = (raw.get("entrypoint") or {}).get("class", "")
    module, _, name = cls.partition(":")
    assert module == "tt_tnt_adapter", (
        f"{path.name}: entrypoint class {cls!r} does not name the bundled adapter module"
    )
    assert name, f"{path.name}: entrypoint class {cls!r} has no class name"
    bundle = MANIFEST_DIR.parent / "bundle" / f"{module}.py"
    assert bundle.is_file(), f"{path.name}: {bundle} does not exist"


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_every_manifest_has_a_registered_size(path):
    """A manifest for an unregistered size has no architecture to validate against."""
    size = _size_of(path)
    assert size.name in SIZES
