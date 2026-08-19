# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""scripts/publish_corpus_recipe.py: no network required.

WHY THESE EXIST -- this module's own docstring explains it, but the short version: the
manifest-vs-tokenized gap percentage has shipped wrong TWICE (0.46% and 0.42%, for gaps that
were really ~1.71% and ~1.90%). ``test_gap_computation_*`` below pins the arithmetic against
those exact historical wrong/right pairs so it cannot happen a third time silently.

Every test here fakes the Hub (``huggingface_hub.HfApi`` / ``hf_hub_download``) rather than
skipping when the network is unavailable -- a test that quietly no-ops without network
would pass "vacuously" exactly the way the underlying bug went undetected for so long.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import needs_artifacts

ROOT = Path(__file__).resolve().parent.parent

# Loaded by file path, matching this repo's convention (see test_publish_to_hub.py /
# test_backfill_checkpoint_headers.py): avoids an unrelated `scripts` package on sys.path
# shadowing a bare `import scripts.publish_corpus_recipe`.
_SCRIPT_PATH = ROOT / "scripts" / "publish_corpus_recipe.py"
_spec = importlib.util.spec_from_file_location("publish_corpus_recipe", _SCRIPT_PATH)
pcr = importlib.util.module_from_spec(_spec)
# Registered in sys.modules BEFORE exec_module: the script's frozen @dataclass fields need
# `sys.modules[cls.__module__]` to resolve during class creation (Python 3.12's dataclasses
# looks the module up by name to check for `typing.KW_ONLY` sentinels), which fails with
# "'NoneType' object has no attribute '__dict__'" if the module isn't registered yet.
sys.modules[_spec.name] = pcr
_spec.loader.exec_module(pcr)


# -----------------------------------------------------------------------------------------
# The derived-figure computation -- especially the gap percentage.
# -----------------------------------------------------------------------------------------

def test_gap_computation_matches_the_2026_08_14_rebuild_exactly():
    """The real numbers from docs/measurements/blend_manifest.json and the real tokenized
    totals: manifest 399,508,203 vs tokenized 391,921,555 must compute to the documented
    7,586,648-token, 1.90% gap -- not the 0.42%/0.46% that shipped wrong twice."""
    gap = pcr.compute_manifest_vs_tokenized_gap(399_508_203, 391_921_555)
    assert gap.gap_tokens == 7_586_648
    assert round(gap.gap_pct, 2) == 1.90


def test_gap_computation_rejects_the_historically_wrong_answers():
    """0.42% and 0.46% were both shipped for gaps in this same ballpark. Neither is what
    the correct formula (gap_tokens / manifest_total * 100) produces for the real figures."""
    gap = pcr.compute_manifest_vs_tokenized_gap(399_508_203, 391_921_555)
    assert abs(gap.gap_pct - 0.42) > 1.0
    assert abs(gap.gap_pct - 0.46) > 1.0


def test_gap_computation_is_symmetric_in_which_count_is_larger():
    """The gap is a magnitude; it must not go negative or flip sign depending on which
    count -- manifest or tokenized -- happens to be larger."""
    gap_a = pcr.compute_manifest_vs_tokenized_gap(100_000, 90_000)
    gap_b = pcr.compute_manifest_vs_tokenized_gap(90_000, 100_000)
    assert gap_a.gap_tokens == 10_000
    # Percentage is always expressed against the FIRST (manifest) argument, so these two
    # differ -- only the raw token gap is symmetric, not the percentage.
    assert gap_a.gap_tokens == gap_b.gap_tokens
    assert gap_a.gap_pct == pytest.approx(10_000 / 100_000 * 100)
    assert gap_b.gap_pct == pytest.approx(10_000 / 90_000 * 100)


def test_gap_computation_zero_gap():
    gap = pcr.compute_manifest_vs_tokenized_gap(1_000_000, 1_000_000)
    assert gap.gap_tokens == 0
    assert gap.gap_pct == 0.0


def test_gap_computation_rejects_nonpositive_manifest_total():
    with pytest.raises(ValueError, match="positive"):
        pcr.compute_manifest_vs_tokenized_gap(0, 100)
    with pytest.raises(ValueError, match="positive"):
        pcr.compute_manifest_vs_tokenized_gap(-5, 100)


def test_render_honest_measurements_states_the_correct_gap_and_not_a_stale_one():
    """The rendered prose must quote the SAME gap the pure function computes -- catches a
    generator that computes correctly but then hardcodes or mis-substitutes the string."""
    manifest = {"total_emitted_tokens": 399_508_203}
    tokenized = pcr.TokenizedStats(
        total_tokens=391_921_555, train_tokens=352_729_403, val_tokens=39_192_152,
        eos_count=798_771,
    )
    out = pcr.render_honest_measurements(manifest, tokenized)
    assert "7,586,648 tokens, or 1.90%" in out
    assert "0.42%" not in out
    assert "0.46%" not in out


# -----------------------------------------------------------------------------------------
# compute_tokenized_stats: real numpy arrays, tiny fakes (no 1.4 GB fixture required).
# -----------------------------------------------------------------------------------------

def test_compute_tokenized_stats_counts_eos_and_lengths(tmp_path):
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    # id 2 is the fake eos token here; train has two, val has one.
    train = np.array([5, 2, 7, 2, 9], dtype=np.uint32)
    val = np.array([2, 3], dtype=np.uint32)
    np.save(tokens_dir / "train_ids.npy", train)
    np.save(tokens_dir / "val_ids.npy", val)

    stats = pcr.compute_tokenized_stats(tokens_dir, eos_token_id=2)
    assert stats.train_tokens == 5
    assert stats.val_tokens == 2
    assert stats.total_tokens == 7
    assert stats.eos_count == 3


def test_compute_tokenized_stats_missing_files_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="train/tokenization.py"):
        pcr.compute_tokenized_stats(tmp_path / "does-not-exist", eos_token_id=2)


def test_compute_tokenized_stats_against_the_real_current_blend():
    """If the real artifacts exist on this machine (they do in this environment -- a full
    corpus build has already run), the ground truth must match what docs/corpus_blend.md
    and the currently-published Hub card both state: 391,921,555 total, 798,771 </s>."""
    tokens_dir = pcr.TOKENS_DIR_DEFAULT
    if not (tokens_dir / "train_ids.npy").is_file():
        pytest.skip("artifacts/tokens-v3 not present on this machine")
    stats = pcr.compute_tokenized_stats(tokens_dir, eos_token_id=2)
    assert stats.total_tokens == 391_921_555
    assert stats.train_tokens == 352_729_403
    assert stats.val_tokens == 39_192_152
    assert stats.eos_count == 798_771


# -----------------------------------------------------------------------------------------
# The recipe file set: defined once, matches what's actually on disk.
# -----------------------------------------------------------------------------------------

def test_recipe_files_all_exist_on_disk():
    for rf in pcr.RECIPE_FILES:
        assert (ROOT / rf.path).is_file(), f"{rf.path} is in RECIPE_FILES but missing on disk"


def test_recipe_files_matches_current_live_hub_listing():
    """Regression pin: this is the exact file set already published, discovered by querying
    the live Hub once while building this script. A one-line addition to RECIPE_FILES is
    the correct way to grow this set; this test documents what "currently" means."""
    expected = {
        "train/corpus.py",
        "scripts/fetch_corpus.py",
        "scripts/prepare_corpus.py",
        "scripts/measure_corpus.py",
        "scripts/blend_corpus.py",
        "scripts/check_disk_space.py",
        "docs/corpus_licensing.md",
        "docs/corpus_blend.md",
        "docs/measurements/blend_manifest.json",
    }
    assert {rf.path for rf in pcr.RECIPE_FILES} == expected


def test_recipe_file_table_lists_every_recipe_file_and_nothing_else():
    out = pcr.render_recipe_file_table()
    for rf in pcr.RECIPE_FILES:
        assert f"`{rf.path}`" in out
    assert out.count("\n| `") == len(pcr.RECIPE_FILES)


# -----------------------------------------------------------------------------------------
# Never upload corpus text, tokenizer files, or weights.
# -----------------------------------------------------------------------------------------

def test_the_real_recipe_passes_the_redistributability_guard():
    pcr._assert_recipe_is_redistributable()


@pytest.mark.parametrize("bad_path", [
    "artifacts/corpus/blend.txt",
    "artifacts/tokenizer/tokenizer.json",
    "artifacts/hf-tt-tnt-v1/model.safetensors",
    "train/tokens_are_sneaky.npy",
    "somewhere/else/vocab.json",
    "somewhere/else/merges.txt",
    "somewhere/else/pytorch_model.bin",
])
def test_forbidden_paths_are_rejected(bad_path):
    fake = [pcr.RecipeFile(bad_path, "sneaked in")]
    with pytest.raises(ValueError):
        pcr._assert_recipe_is_redistributable(fake)


def test_oversized_file_is_rejected_even_with_an_innocent_name(tmp_path, monkeypatch):
    """Belt-and-suspenders: a recipe-shaped filename that is nonetheless huge (e.g. someone
    accidentally pointed a recipe entry at the real corpus file under a misleading name)
    must still be refused, on size alone."""
    monkeypatch.setattr(pcr, "ROOT", tmp_path)
    big = tmp_path / "docs" / "corpus_blend.md"
    big.parent.mkdir(parents=True)
    big.write_bytes(b"x" * (pcr._MAX_RECIPE_FILE_BYTES + 1))
    fake = [pcr.RecipeFile("docs/corpus_blend.md", "looks innocent, isn't")]
    with pytest.raises(ValueError, match="larger than any real recipe file"):
        pcr._assert_recipe_is_redistributable(fake)


def test_source_never_sets_visibility():
    """No code path in this script may flip repo visibility, in either direction."""
    source = _SCRIPT_PATH.read_text()
    assert "private=True" not in source
    assert "private=False" not in source
    assert "private = True" not in source
    assert "private = False" not in source


def _boom_hf_api(monkeypatch):
    """Patches huggingface_hub.HfApi to explode if constructed, so a test can prove a code
    path never reaches the Hub -- without blocking the ``huggingface_hub`` module wholesale
    (transformers imports it internally even for a purely local, ``local_files_only=True``
    tokenizer load, so blocking the module breaks unrelated local-only code paths too)."""
    import huggingface_hub

    def _boom(*a, **k):
        raise AssertionError("must not contact the Hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", _boom)


def test_dry_run_never_touches_the_hub(monkeypatch, capsys):
    _boom_hf_api(monkeypatch)
    rc = pcr.cmd_dry_run("episod/tt-tnt-corpus", pcr.TOKENS_DIR_DEFAULT)
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "generated card" in out


@needs_artifacts("artifacts/corpus")
def test_publish_without_yes_refuses_and_does_not_touch_the_hub(monkeypatch, capsys):
    _boom_hf_api(monkeypatch)
    rc = pcr.cmd_publish("episod/tt-tnt-corpus", yes=False, tokens_dir=pcr.TOKENS_DIR_DEFAULT)
    assert rc != 0
    assert "--yes" in capsys.readouterr().err


# -----------------------------------------------------------------------------------------
# --verify: fake the Hub, prove divergence is actually detected (not vacuously passing).
# -----------------------------------------------------------------------------------------

class _FakeDatasetInfo:
    def __init__(self, private):
        self.private = private


class _FakeHfApi:
    def __init__(self, private=False):
        self._private = private

    def dataset_info(self, repo_id):
        return _FakeDatasetInfo(self._private)


def _install_fake_hub(monkeypatch, tmp_path, hub_files: dict, private: bool = False):
    """Fakes huggingface_hub.HfApi and hf_hub_download with in-memory content, so --verify
    can be exercised with zero network access -- and, crucially, so a test can assert it
    actually flags a divergence rather than trivially passing because nothing was checked.
    """
    import huggingface_hub

    def _fake_download(repo_id, filename, repo_type=None):
        if filename not in hub_files:
            raise FileNotFoundError(f"fake hub has no {filename}")
        dest = tmp_path / "hub_dl" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = hub_files[filename]
        dest.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _FakeHfApi(private=private))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)


def _matching_hub_files(tmp_path, monkeypatch):
    """Build a fake Hub file set that EXACTLY matches the working tree + a real generated
    card, using tiny fake token arrays so this test never touches the real 1.4 GB npy
    files or the network."""
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    np.save(tokens_dir / "train_ids.npy", np.array([1, 2, 3], dtype=np.uint32))
    np.save(tokens_dir / "val_ids.npy", np.array([2], dtype=np.uint32))
    monkeypatch.setattr(pcr, "discover_eos_token_id", lambda *a, **k: 2)

    tokenized = pcr.compute_tokenized_stats(tokens_dir, eos_token_id=2)
    manifest = pcr._load_manifest()
    card = pcr.render_card(manifest, pcr.SOURCES, tokenized)

    hub_files = {rf.path: (ROOT / rf.path).read_bytes() for rf in pcr.RECIPE_FILES}
    hub_files["README.md"] = card
    return tokens_dir, hub_files


def test_verify_passes_when_hub_matches_working_tree_exactly(tmp_path, monkeypatch, capsys):
    tokens_dir, hub_files = _matching_hub_files(tmp_path, monkeypatch)
    _install_fake_hub(monkeypatch, tmp_path, hub_files, private=False)

    rc = pcr.cmd_verify("episod/tt-tnt-corpus", tokens_dir)
    out = capsys.readouterr().out
    assert rc == 0
    assert "no divergence" in out


def test_verify_detects_a_stale_recipe_file(tmp_path, monkeypatch, capsys):
    """The exact failure this script exists to catch: a recipe file on the Hub has drifted
    from the working tree (e.g. pre-separator scripts left next to a post-separator
    manifest). Must exit non-zero and name the differing file."""
    tokens_dir, hub_files = _matching_hub_files(tmp_path, monkeypatch)
    hub_files["train/corpus.py"] = b"# stale pre-rebuild copy\n"
    _install_fake_hub(monkeypatch, tmp_path, hub_files, private=False)

    rc = pcr.cmd_verify("episod/tt-tnt-corpus", tokens_dir)
    out = capsys.readouterr().out
    assert rc != 0
    assert "train/corpus.py" in out
    assert "[FAIL]" in out


def test_verify_detects_a_stale_card(tmp_path, monkeypatch, capsys):
    tokens_dir, hub_files = _matching_hub_files(tmp_path, monkeypatch)
    hub_files["README.md"] = "# an old card with the wrong gap: 0.42%\n"
    _install_fake_hub(monkeypatch, tmp_path, hub_files, private=False)

    rc = pcr.cmd_verify("episod/tt-tnt-corpus", tokens_dir)
    out = capsys.readouterr().out
    assert rc != 0
    assert "README.md" in out


def test_verify_detects_a_missing_hub_file(tmp_path, monkeypatch, capsys):
    """A file this script would publish that the Hub doesn't have at all -- also
    divergence, not a crash."""
    tokens_dir, hub_files = _matching_hub_files(tmp_path, monkeypatch)
    del hub_files["docs/measurements/blend_manifest.json"]
    _install_fake_hub(monkeypatch, tmp_path, hub_files, private=False)

    rc = pcr.cmd_verify("episod/tt-tnt-corpus", tokens_dir)
    out = capsys.readouterr().out
    assert rc != 0
    assert "docs/measurements/blend_manifest.json" in out
    assert "[FAIL]" in out


def test_verify_detects_visibility_drift(tmp_path, monkeypatch, capsys):
    tokens_dir, hub_files = _matching_hub_files(tmp_path, monkeypatch)
    _install_fake_hub(monkeypatch, tmp_path, hub_files, private=True)  # flipped!

    rc = pcr.cmd_verify("episod/tt-tnt-corpus", tokens_dir)
    out = capsys.readouterr().out
    assert rc != 0
    assert "visibility" in out.lower()


@needs_artifacts("artifacts/corpus")
def test_verify_does_not_pass_vacuously_with_an_empty_or_broken_fake():
    """Guards against a fake so weak it always reports success. If cmd_verify is called
    against a fake Hub with NOTHING on it, every file must be reported as a failure to
    download, not silently skipped."""
    import huggingface_hub as hh

    class _EmptyApi:
        def dataset_info(self, repo_id):
            return _FakeDatasetInfo(False)

    def _always_missing(repo_id, filename, repo_type=None):
        raise FileNotFoundError("nothing on this fake hub")

    orig_api, orig_dl = hh.HfApi, hh.hf_hub_download
    hh.HfApi = lambda: _EmptyApi()
    hh.hf_hub_download = _always_missing
    try:
        rc = pcr.cmd_verify("episod/tt-tnt-corpus", pcr.TOKENS_DIR_DEFAULT)
    finally:
        hh.HfApi, hh.hf_hub_download = orig_api, orig_dl
    assert rc != 0


# -----------------------------------------------------------------------------------------
# Card rendering: placeholder behavior when tokenized stats aren't available, and that a
# real publish refuses to ship that placeholder.
# -----------------------------------------------------------------------------------------

def test_render_card_with_no_tokenized_stats_shows_an_explicit_placeholder_not_a_number():
    manifest = pcr._load_manifest()
    out = pcr.render_card(manifest, pcr.SOURCES, tokenized=None)
    assert "not available" in out
    assert "391,921,555" not in out


def test_publish_refuses_when_tokenized_stats_are_unavailable(tmp_path, monkeypatch):
    """A real --publish must never ship the placeholder -- it should fail loudly instead."""
    with pytest.raises(FileNotFoundError):
        pcr.cmd_publish("episod/tt-tnt-corpus", yes=True, tokens_dir=tmp_path / "nope")


def test_render_share_table_reflects_manifest_achieved_share_not_target():
    manifest = pcr._load_manifest()
    out = pcr.render_share_table(pcr.SOURCES, manifest)
    # flavour's target is 0.5% but its achieved share (per the real manifest) is 0.496%.
    assert "0.496%" in out
    assert "3.4759x" in out
