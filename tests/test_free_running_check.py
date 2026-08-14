# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""scripts/free_running_check.py: the parts that do not need a server or a device.

What is worth testing here is not the comparison arithmetic (it is six lines) but the
**reference-selection guard**, because that is where this script can produce numbers that
look like evidence and are not.

The script's whole output is a token-by-token comparison of device generation against a
local CPU reference. If the reference is a different model from the one being served, every
row still fills in, the medians still print, and the result is meaningless -- it measures
the gap between two models rather than the gap between two execution paths. That happened
for real: the default reference resolved to ``artifacts/384/hf`` (the v2 model, 256
context) while the Hub had been republished at 512, so the default silently pointed at the
wrong model.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Loaded by file path, matching this repo's convention (see test_publish_to_hub.py): an
# unrelated project's own `scripts/__init__.py` earlier on sys.path would otherwise shadow
# a bare `import scripts.free_running_check`.
_SCRIPT_PATH = ROOT / "scripts" / "free_running_check.py"
_spec = importlib.util.spec_from_file_location("free_running_check", _SCRIPT_PATH)
frc = importlib.util.module_from_spec(_spec)
sys.modules["free_running_check"] = frc
_spec.loader.exec_module(frc)


def _write_config(directory: Path, ctx: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"max_position_embeddings": ctx}))
    return directory


def _fake_models_endpoint(monkeypatch, payload):
    """Stand in for urlopen so the guard can be tested with no server running."""
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(frc.urllib.request, "urlopen", lambda *a, **k: _Resp())


def test_default_reference_is_the_published_artifact_not_the_v2_baseline():
    """The default must resolve to the artifact matching the published weights.

    `artifacts/384/hf` is the 256-context v2 model. Defaulting there while the Hub serves
    512-context weights is the exact wrong-model comparison this guard exists to stop.
    """
    assert frc.DEFAULT_HF_DIR_NAME == "artifacts/hf-tt-tnt-v1"
    default = frc._default_hf_dir()
    assert default == ROOT / "artifacts" / "hf-tt-tnt-v1"
    assert default != ROOT / "artifacts" / "384" / "hf"
    config = json.loads((default / "config.json").read_text())
    assert config["max_position_embeddings"] == 512


def test_guard_accepts_a_matching_reference(tmp_path, monkeypatch):
    _fake_models_endpoint(monkeypatch, {"data": [{"id": "m", "max_model_len": 512}]})
    frc._check_reference_matches_server(_write_config(tmp_path / "hf", 512), "http://x", "m", 1.0)


def test_guard_refuses_a_shorter_context_reference(tmp_path, monkeypatch):
    """The real failure: a 256-context reference against 512-context served weights."""
    _fake_models_endpoint(monkeypatch, {"data": [{"id": "m", "max_model_len": 512}]})
    with pytest.raises(frc.ReferenceMismatch, match="different models"):
        frc._check_reference_matches_server(
            _write_config(tmp_path / "hf", 256), "http://x", "m", 1.0
        )


def test_guard_refuses_when_the_server_does_not_serve_the_named_model(tmp_path, monkeypatch):
    _fake_models_endpoint(monkeypatch, {"data": [{"id": "other", "max_model_len": 512}]})
    with pytest.raises(frc.ReferenceMismatch, match="does not serve"):
        frc._check_reference_matches_server(
            _write_config(tmp_path / "hf", 512), "http://x", "m", 1.0
        )


def test_guard_refuses_a_directory_that_is_not_a_model(tmp_path):
    with pytest.raises(frc.ReferenceMismatch, match="not an HF model dir"):
        frc._check_reference_matches_server(tmp_path / "nope", "http://x", "m", 1.0)


def test_agreement_counts_leading_matches_only():
    """A later coincidental match must not be credited after a divergence."""
    assert frc.agreement(["a", "b", "c"], ["a", "b", "c"]) == 3
    assert frc.agreement(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert frc.agreement(["a"], ["x"]) == 0
    assert frc.agreement([], []) == 0
