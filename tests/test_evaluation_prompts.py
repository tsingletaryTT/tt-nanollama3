# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The prompt set is FROZEN. These tests protect comparability across checkpoints."""
import json
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "docs" / "evaluation_prompts.json"
REQUIRED_PROBES = {"target-voice", "stutter", "oracular", "agentic",
                   "grounding", "perpendicular", "coherence"}


def test_prompt_file_parses():
    data = json.loads(PROMPTS.read_text())
    assert data["prompts"]


def test_ids_are_unique():
    ids = [p["id"] for p in json.loads(PROMPTS.read_text())["prompts"]]
    assert len(ids) == len(set(ids))


def test_every_required_probe_is_present():
    probes = {p["probe"] for p in json.loads(PROMPTS.read_text())["prompts"]}
    missing = REQUIRED_PROBES - probes
    assert not missing, f"prompt set is missing probes: {sorted(missing)}"


def test_stutter_probe_exists_because_this_model_has_a_repetition_defect():
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    assert [p for p in prompts if p["probe"] == "stutter"], (
        "the stutter probe distinguishes 'learned Stein' from 'learned to repeat'"
    )


def test_no_prompt_is_empty_or_whitespace():
    for p in json.loads(PROMPTS.read_text())["prompts"]:
        assert p["text"].strip()
