# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Checkpoint header schema. Pure dict work — no hardware, no ttml import."""

from pathlib import Path

import pytest

from train.checkpoint import (
    CHECKPOINT_FORMAT,
    build_header,
    checkpoint_path,
    latest_checkpoint,
    validate_header,
)


def _header(**kw):
    base = dict(
        step=100,
        model_config_path="/models/nanollama3.yaml",
        tokenizer_dir="artifacts/tokenizer",
        corpus_tokens=127_635_889,
        batch_size=64,
    )
    base.update(kw)
    return build_header(**base, seed=0, tokens_dir="artifacts/tokens-test", optimizer={"type": "AdamW"}, ddp=1)


def test_header_carries_format_version():
    assert _header()["format"] == CHECKPOINT_FORMAT


def test_header_carries_resume_and_conversion_fields():
    h = _header()
    for field in ("step", "vocab_size", "seq_len", "model_config_path", "tokenizer_dir",
                  "corpus_tokens", "batch_size", "tokens_seen", "created_at"):
        assert field in h, f"header missing {field}"


def test_header_records_vocab_and_seq_len_from_config():
    from train.config import SEQ_LEN, VOCAB_SIZE

    h = _header()
    assert h["vocab_size"] == VOCAB_SIZE
    assert h["seq_len"] == SEQ_LEN


def test_header_records_explicit_seq_len_not_the_module_default():
    """seq_len is now a CLI flag (train/run.py --seq-len), so build_header must record the
    seq_len THIS run actually used when the caller passes it explicitly -- not silently
    fall back to whatever train.config.SEQ_LEN happens to be. A run at a non-default
    seq_len whose checkpoint header lied about it would propagate a wrong
    max_position_embeddings into convert/to_hf.py with no error anywhere."""
    from train.config import SEQ_LEN

    h = _header(step=10, batch_size=4, seq_len=256)
    assert h["seq_len"] == 256
    assert h["seq_len"] != SEQ_LEN  # proves it didn't fall back to the default
    assert h["tokens_seen"] == 10 * 4 * 256


def test_header_computes_tokens_seen_from_step_batch_and_seq_len():
    """tokens_seen must be derived, not guessed -- batch_size isn't recorded anywhere
    else a model card could read it from, so this is the only source of truth."""
    from train.config import SEQ_LEN

    h = _header(step=3000, batch_size=64)
    assert h["tokens_seen"] == 3000 * 64 * SEQ_LEN
    # Also pinned to a literal (not just the formula above), so a regression that breaks
    # the multiplication itself (e.g. accidentally swapping in addition) still fails
    # loudly. This literal assumes SEQ_LEN == 512 (true today) and must be recomputed if
    # SEQ_LEN ever changes again -- exactly what happened here: the previous literal,
    # 49_152_000, was only correct while SEQ_LEN was 256.
    assert SEQ_LEN == 512 and h["tokens_seen"] == 98_304_000


def test_corpus_tokens_and_tokens_seen_are_independent_fields():
    """corpus_tokens (whole corpus) must not be confused with tokens_seen (this run's
    actual training volume) -- they differ by design for a partial-epoch run."""
    from train.config import SEQ_LEN

    h = _header(step=3000, batch_size=64, corpus_tokens=127_635_889)
    assert h["corpus_tokens"] == 127_635_889
    assert h["tokens_seen"] == 3000 * 64 * SEQ_LEN
    assert h["corpus_tokens"] != h["tokens_seen"]


def test_extra_carries_ttml_cpp_defaults_absent_from_any_yaml():
    """intermediate_dim, weight_tying, rms_norm_eps, and weights_dtype exist only as ttml
    C++ defaults/manifest facts -- they must survive build_header via extra and pass
    validate_header, or a converter has nowhere to read them from but a guess."""
    h = _header(extra={
        "transformer_config": {"embedding_dim": 384, "num_heads": 6, "num_groups": 3},
        "intermediate_dim": 1024,
        "weight_tying": True,
        "rms_norm_eps": 1e-5,
        "weights_dtype": "bfloat16",
    })
    validate_header(h)  # must not raise
    assert h["intermediate_dim"] == 1024
    assert h["weight_tying"] is True
    assert h["rms_norm_eps"] == 1e-5
    assert h["weights_dtype"] == "bfloat16"
    assert h["transformer_config"]["embedding_dim"] == 384


def test_extra_is_merged_without_clobbering_required_fields():
    h = _header(extra={"note": "smoke run"})
    assert h["note"] == "smoke run"
    assert h["step"] == 100  # extra must not overwrite schema fields


def test_extra_cannot_override_a_schema_field():
    with pytest.raises(ValueError, match="may not override"):
        _header(extra={"vocab_size": 999})


def test_validate_accepts_a_built_header():
    validate_header(_header())  # must not raise


def test_validate_rejects_missing_field():
    h = _header()
    del h["vocab_size"]
    with pytest.raises(ValueError, match="vocab_size"):
        validate_header(h)


def test_validate_rejects_future_format():
    h = _header()
    h["format"] = CHECKPOINT_FORMAT + 1
    with pytest.raises(ValueError, match="format"):
        validate_header(h)


def test_checkpoint_path_is_step_numbered():
    p = checkpoint_path(Path("/ckpt"), 2500)
    assert p == Path("/ckpt/tt_tnt_step00002500.pkl")


def test_checkpoint_paths_sort_lexicographically_by_step():
    """Zero-padding matters: without it, step10 sorts before step9."""
    paths = sorted(str(checkpoint_path(Path("/c"), s)) for s in (9, 10, 100))
    assert paths == [str(checkpoint_path(Path("/c"), s)) for s in (9, 10, 100)]


def test_latest_checkpoint_returns_none_for_empty_dir(tmp_path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_picks_highest_step_not_newest_file(tmp_path):
    """The docstring used to say "newest"; with one directory shared across runs,
    highest-step and most-recently-written are not the same file. Create the
    higher-step file *first* and the lower-step one *second* -- so the lower-step
    file has the newer mtime -- to prove the selection is by step, not by mtime."""
    high_step_path = checkpoint_path(tmp_path, 5000)
    low_step_path = checkpoint_path(tmp_path, 100)
    high_step_path.touch()
    low_step_path.touch()  # written after, so it has the newer mtime
    assert low_step_path.stat().st_mtime_ns >= high_step_path.stat().st_mtime_ns
    assert latest_checkpoint(tmp_path) == high_step_path


def test_latest_checkpoint_finds_pre_rename_nanollama3_files(tmp_path):
    """Checkpoints written before the tt-nanollama3 -> tt-tnt rename are never renamed on
    disk (they are evidence of a real run under the old name) -- a directory holding only
    those files, e.g. the real ``artifacts/checkpoints/``, must still resolve via
    ``--resume latest``."""
    legacy = tmp_path / "nanollama3_step00003000.pkl"
    legacy.touch()
    assert latest_checkpoint(tmp_path) == legacy


def test_latest_checkpoint_picks_the_higher_step_across_both_naming_schemes(tmp_path):
    """A directory that mixes pre-rename and post-rename checkpoints (e.g. an old baseline
    directory a new run resumed into) must pick the highest **step**, regardless of which
    naming scheme it happens to be written under."""
    old_low = tmp_path / "nanollama3_step00000500.pkl"
    old_low.touch()
    new_high = checkpoint_path(tmp_path, 3000)
    new_high.touch()
    assert latest_checkpoint(tmp_path) == new_high

    # And the reverse: an old-prefixed file can still be the higher step.
    tmp_path2 = tmp_path / "mixed2"
    tmp_path2.mkdir()
    old_high = tmp_path2 / "nanollama3_step00021034.pkl"
    old_high.touch()
    new_low = checkpoint_path(tmp_path2, 100)
    new_low.touch()
    assert latest_checkpoint(tmp_path2) == old_high


# ---------------------------------------------------------------------------
# The DDP checkpoint guard
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# The mesh guard. These run without hardware by standing in for the three ttml/ttnn surfaces
# train.checkpoint touches: Sharding.from_tensor (is this tensor recorded as sharded?), the
# parallelism context (which parallelisms are live?), and TensorTopology/update_tensor_topology
# (the metadata correction itself). Everything they assert was measured on four Blackhole
# p300c chips first -- see .superpowers/ddp-checkpoint-fix.md.
# ---------------------------------------------------------------------------------------

DDP_SIZE = 4


class _FakeSharding:
    """Stands in for ttml.sharding.Sharding without a device."""

    def __init__(self, replicated):
        self._replicated = replicated

    @property
    def is_fully_replicated(self):
        return self._replicated


class _FakeNamedParameters(dict):
    """ttml.NamedParameters: the leaf type ttml's own ``_walk`` stops recursing at. A distinct
    type from ``dict`` on purpose -- the traversal's whole job is telling the two apart."""


class _FakeTopology:
    """A TensorTopology: a distribution shape, one placement per axis, and its coordinates."""

    def __init__(self, dist_shape, placements, coords):
        self._dist_shape, self._placements, self._coords = dist_shape, placements, coords

    def distribution_shape(self):
        return list(self._dist_shape)

    def placements(self):
        return list(self._placements)

    def mesh_coords(self):
        return list(self._coords)


class _FakeValue:
    """The ttnn tensor behind a parameter: holds the live topology and records every write.

    ``update_tensor_topology`` replaces the topology *and* appends to ``history``, so a test
    can assert both the final state (restored) and the sequence (re-marked, then restored) --
    the second is what proves a save does not merely end up looking untouched by accident.
    """

    def __init__(self, topology):
        self.topology = topology
        self.history = []

    def tensor_topology(self):
        return self.topology

    def update_tensor_topology(self, topology):
        self.topology = topology
        self.history.append(topology)


class _FakeTensor:
    """A ttml autograd Tensor: all train.checkpoint wants from one is ``get_value``."""

    def __init__(self, *, sharded, dist_shape=(DDP_SIZE,), coords=(0, 1, 2, 3)):
        placements = ["Shard(0)" if sharded else "Replicate"]
        self.sharded = sharded
        self.value = _FakeValue(_FakeTopology(dist_shape, placements, coords))

    def get_value(self, _precision=None):
        return self.value


def _install_fakes(monkeypatch, *, ddp=DDP_SIZE, tp=0, context=True):
    """Point train.checkpoint's lazily-imported ttml/ttnn at fakes.

    ``ddp``/``tp`` are the sizes the parallelism context reports; ``context=False`` stands for
    a single-chip run, where no context has been initialised at all.
    """
    import sys
    import types

    sharding_mod = types.ModuleType("ttml.sharding")
    sharding_mod.Sharding = type(
        "Sharding",
        (),
        {"from_tensor": staticmethod(lambda t: _FakeSharding(not t.sharded))},
    )

    pctx = types.SimpleNamespace(
        is_ddp_enabled=lambda: ddp > 1,
        is_tp_enabled=lambda: tp > 1,
        get_ddp_size=lambda: ddp,
        get_tp_size=lambda: tp,
    )
    auto_ctx = types.SimpleNamespace(
        is_parallelism_context_initialized=lambda: context,
        get_parallelism_context=lambda: pctx,
    )
    ttml_mod = types.ModuleType("ttml")
    ttml_mod.autograd = types.SimpleNamespace(
        AutoContext=types.SimpleNamespace(get_instance=lambda: auto_ctx),
        PreferredPrecision=types.SimpleNamespace(NATIVE="NATIVE"),
    )
    ttml_mod.NamedParameters = _FakeNamedParameters
    ttml_mod.sharding = sharding_mod

    ttnn_mod = types.ModuleType("ttnn")
    ttnn_mod.MeshShape = list
    ttnn_mod.PlacementReplicate = lambda: "Replicate"
    ttnn_mod.TensorTopology = lambda *, distribution_shape, placements, mesh_coords: (
        _FakeTopology(distribution_shape, placements, mesh_coords)
    )

    monkeypatch.setitem(sys.modules, "ttml", ttml_mod)
    monkeypatch.setitem(sys.modules, "ttml.sharding", sharding_mod)
    monkeypatch.setitem(sys.modules, "ttnn", ttnn_mod)


def test_saveable_on_mesh_accepts_fully_replicated_params(monkeypatch):
    """The single-chip case: nothing is recorded as sharded, so the saver may proceed and the
    parallelism context is never even consulted."""
    from train.checkpoint import assert_saveable_on_mesh

    _install_fakes(monkeypatch, context=False)
    assert_saveable_on_mesh({"w1": _FakeTensor(sharded=False),
                             "w2": _FakeTensor(sharded=False)})  # must not raise


def test_saveable_on_mesh_refuses_sharding_with_no_parallelism_context(monkeypatch):
    """A tensor recorded as sharded when nothing distributed it is unexplained, and the whole
    point of the guard is that unexplained sharding is never written past: ttml's saver would
    concatenate the shards, which is either an oversized file convert/ cannot read or a
    silently partial model."""
    from train.checkpoint import assert_saveable_on_mesh

    _install_fakes(monkeypatch, context=False)
    with pytest.raises(RuntimeError, match="recorded as SHARDED"):
        assert_saveable_on_mesh({"ok": _FakeTensor(sharded=False),
                                 "llama/block0/q_linear/weight": _FakeTensor(sharded=True)})


def test_saveable_on_mesh_names_an_offender(monkeypatch):
    """The message has to name a parameter, or an operator cannot tell this apart from a
    generic mesh complaint."""
    from train.checkpoint import assert_saveable_on_mesh

    _install_fakes(monkeypatch, context=False)
    with pytest.raises(RuntimeError, match="llama/block0/q_linear/weight"):
        assert_saveable_on_mesh({"llama/block0/q_linear/weight": _FakeTensor(sharded=True)})


def test_saveable_on_mesh_admits_the_data_parallel_remark(monkeypatch):
    """THE NARROWING. Under DDP-and-only-DDP the model is replicated and only the batch is
    split, so a Shard(0) on a parameter distributed over exactly the DDP axis cannot be true --
    it is the measured metadata defect, and replicated_for_save corrects it. The gate must let
    that case through, or --ddp N can never checkpoint."""
    from train.checkpoint import assert_saveable_on_mesh

    _install_fakes(monkeypatch, ddp=DDP_SIZE)
    assert_saveable_on_mesh({"llama/fc/weight": _FakeTensor(sharded=True)})  # must not raise


def test_saveable_on_mesh_still_refuses_when_tensor_parallel_is_live(monkeypatch):
    """The case the guard must never stop catching. Tensor parallelism shards parameters for
    real, so a Shard placement may be the truth and re-marking it Replicate would write one
    chip's slice as if it were the whole weight -- wrong in a way nothing downstream detects."""
    from train.checkpoint import assert_saveable_on_mesh

    _install_fakes(monkeypatch, ddp=2, tp=2)
    with pytest.raises(RuntimeError, match="tensor parallelism is enabled"):
        assert_saveable_on_mesh({"llama/fc/weight": _FakeTensor(sharded=True)})


def test_saveable_on_mesh_refuses_a_distribution_that_is_not_the_ddp_axis(monkeypatch):
    """Second, independent condition: DDP being the only parallelism is not enough on its own
    if the tensor is laid out over something wider than the DDP axis. A distribution this code
    cannot attribute to that axis may describe a real split, and is refused rather than
    guessed at."""
    from train.checkpoint import assert_saveable_on_mesh

    _install_fakes(monkeypatch, ddp=DDP_SIZE)
    wider = _FakeTensor(sharded=True, dist_shape=(2, 4), coords=tuple(range(8)))
    with pytest.raises(RuntimeError, match="other than the 4-device data-parallel axis"):
        assert_saveable_on_mesh({"llama/fc/weight": wider})


def test_replicated_for_save_remarks_then_restores(monkeypatch):
    """The save-time correction, and the property that makes it safe to run mid-run.

    Inside the block the tensor must read Replicate, so ttml's composer takes ONE copy instead
    of concatenating four. After the block it must read exactly what it read before, so a run
    with --save-every is the same computation as a run without it."""
    from train.checkpoint import replicated_for_save

    _install_fakes(monkeypatch, ddp=DDP_SIZE)
    tensor = _FakeTensor(sharded=True)
    before = tensor.value.topology.placements()
    assert before == ["Shard(0)"]

    with replicated_for_save({"llama/fc/weight": tensor}) as remarked:
        assert remarked == 1
        assert tensor.value.topology.placements() == ["Replicate"]

    assert tensor.value.topology.placements() == before
    assert len(tensor.value.history) == 2, "expected exactly one re-mark and one restore"


def test_replicated_for_save_restores_even_when_the_save_raises(monkeypatch):
    """A failed write must not leave the run's parameters carrying metadata this code invented.
    ttml's save is atomic (temp file then rename) precisely so a crash mid-write is survivable;
    the topology correction has to be equally survivable or the surviving run is corrupted."""
    from train.checkpoint import replicated_for_save

    _install_fakes(monkeypatch, ddp=DDP_SIZE)
    tensor = _FakeTensor(sharded=True)
    with pytest.raises(ValueError):
        with replicated_for_save({"llama/fc/weight": tensor}):
            raise ValueError("disk full")
    assert tensor.value.topology.placements() == ["Shard(0)"]


def test_replicated_for_save_leaves_replicated_tensors_alone(monkeypatch):
    """A single-chip run, and every already-correct tensor in a DDP run: touch nothing."""
    from train.checkpoint import replicated_for_save

    _install_fakes(monkeypatch, ddp=DDP_SIZE)
    tensor = _FakeTensor(sharded=False)
    with replicated_for_save({"llama/fc/weight": tensor}) as remarked:
        assert remarked == 0
    assert tensor.value.history == []


def test_optimizer_tensors_walks_nested_state(monkeypatch):
    """save() gates and corrects the optimizer's tensors as well as the model's, because
    ttml's save_checkpoint gathers both through the same Sharding.gather call. No AdamW moment
    has ever been observed re-marked (0 of 132 measured), but the guard is not narrowed to only
    look where the bug has already been seen. Nested dicts matter: composite optimizers
    (MuonWithAdamW) nest sub-state, and ttml's own _walk recurses through them."""
    from train.checkpoint import _optimizer_tensors

    _install_fakes(monkeypatch, ddp=DDP_SIZE)
    exp_avg, nested = _FakeTensor(sharded=False), _FakeTensor(sharded=False)
    optimizer = type("Opt", (), {
        "get_state_dict": lambda self: {
            "exp_avg": _FakeNamedParameters({"llama/fc/weight": exp_avg}),
            "inner": {"exp_avg_sq": _FakeNamedParameters({"llama/fc/weight": nested})},
            "step": 42,  # scalars pass through, exactly as ttml's _walk allows
        }
    })()
    found = _optimizer_tensors(optimizer)
    assert found == {
        "optimizer/exp_avg/llama/fc/weight": exp_avg,
        "optimizer/inner/exp_avg_sq/llama/fc/weight": nested,
    }


# ---------------------------------------------------------------------------
# Checkpoint format 2: provenance that travels with the artifact
# ---------------------------------------------------------------------------
# Added 2026-08-19. Two training runs were compared against a baseline trained on
# a different corpus and read as a 1.3-nat optimizer regression. The baseline
# directory held val_losses.jsonl and nothing else; identifying its corpus took
# mtime forensics plus a throughput argument, and its seed was never recoverable.
# corpus_tokens ALMOST sufficed -- it is what finally proved the corpus, by
# summing to exactly one token set's train+val -- but a sum is an inference where
# a path is a fact.


def _v2_kwargs(**over):
    base = dict(
        model_config_path="m.yaml", tokenizer_dir="tok", corpus_tokens=391823393,
        batch_size=64, seq_len=512, seed=5489, tokens_dir="artifacts/tokens-v4",
        optimizer={"type": "AdamW", "lr": 3e-4, "beta2": 0.999}, ddp=4,
    )
    base.update(over)
    return base


def test_new_headers_carry_the_provenance_fields():
    h = build_header(2000, **_v2_kwargs())
    assert h["format"] == 2
    assert h["seed"] == 5489
    assert h["tokens_dir"] == "artifacts/tokens-v4"
    assert h["ddp"] == 4
    assert h["optimizer"]["beta2"] == 0.999
    validate_header(h)


def test_provenance_fields_are_required_not_optional():
    """A caller that forgets them must fail at WRITE time.

    Optional provenance is provenance that is missing exactly when it matters --
    the baseline that caused this was written by code that simply never recorded
    the seed.
    """
    kwargs = _v2_kwargs()
    for missing in ("seed", "tokens_dir", "optimizer", "ddp"):
        bad = {k: v for k, v in kwargs.items() if k != missing}
        with pytest.raises(TypeError):
            build_header(2000, **bad, seed=0, tokens_dir="artifacts/tokens-test", optimizer={"type": "AdamW"}, ddp=1)


def test_format_1_checkpoints_remain_readable():
    """The upgrade must not retroactively invalidate history.

    There are many format-1 checkpoints on disk and they are evidence of real
    runs. Requiring format-2 fields of them would make an improvement in
    record-keeping destroy the records.
    """
    v1 = {
        "format": 1, "step": 2000, "vocab_size": 32000, "seq_len": 512,
        "model_config_path": "m.yaml", "tokenizer_dir": "tok",
        "corpus_tokens": 391823393, "batch_size": 64, "tokens_seen": 65536000,
        "created_at": "2026-08-18T15:01:08+00:00",
    }
    validate_header(v1)  # must not raise
    assert all(f not in v1 for f in ("seed", "tokens_dir", "optimizer", "ddp"))


def test_a_format_2_header_missing_provenance_is_rejected():
    v1_fields = {
        "format": 2, "step": 2000, "vocab_size": 32000, "seq_len": 512,
        "model_config_path": "m.yaml", "tokenizer_dir": "tok",
        "corpus_tokens": 1, "batch_size": 64, "tokens_seen": 1,
        "created_at": "2026-08-19T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="seed"):
        validate_header(v1_fields)


def test_extra_still_cannot_shadow_the_new_required_fields():
    with pytest.raises(ValueError, match="seed"):
        build_header(2000, **_v2_kwargs(), extra={"seed": 999})
