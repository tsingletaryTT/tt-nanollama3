# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Wiring tests for scripts/derive_skits.py's build_skit_example.

STORY is deliberately NOT the dialogue-with-attribution shape ('"..." said X.'): that
shape is the documented split_sentences over-split (train/skit.py:31-38) and derive_skit
genuinely returns None on it (verified by hand against the merged train/skit.py before
writing this fixture) -- turn 1 becomes just the quoted line and turn 2 becomes just
"said her friend.", which share no content word. That is real corpus drop-rate pressure,
correctly reported in the derivation stats below, but it must not also break this fixture:
a fixture that can't even produce a Skit tests nothing. Every turn here instead carries a
plain declarative sentence with a word bridging it to the next, so accept/add succeed at
every one of the three model turns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.derive_skits import TILE, build_skit_example
from train.skit import MODEL_TURNS, PARTNER_TURNS, derive_skit

STORY = ("Lily found a shiny rock. She showed it to her friend. "
         "The rock sparkled on the windowsill. "
         "Her friend loved the windowsill shine. "
         "The shine made a rainbow appear. "
         "They admired the rainbow glow. "
         "The glow lasted through the evening.")

_IDS = {}


class _Tok:
    """Faithful, deterministic, and honours add_special_tokens.

    Deterministic on purpose: builtins hash() is randomised per process, and a mock that
    ignores add_special_tokens let a spurious-BOS bug pass every stage-1 test.
    """
    pad_token_id = 0
    BOS = 1

    def encode(self, s, add_special_tokens=True):
        ids = [_IDS.setdefault(w, len(_IDS) + 2) for w in s.split()]
        return ([self.BOS] + ids) if add_special_tokens else ids


def _skit():
    s = derive_skit(STORY, story_id=0, idf={"windowsill": 1.0, "rainbow": 1.0},
                    intensity=lambda t: 0.0)
    assert s is not None
    return s


def test_labels_are_pre_shifted_at_every_supervised_position():
    """ttml compares logits[t] to labels[t] with no internal shift. The HF convention
    silently trained two arms against wrong targets in stage 1."""
    ex = build_skit_example(_skit(), _Tok(), with_think=True, pad_token_id=0)
    ids, labs = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labs)
    for t, v in enumerate(labs):
        if v != -100:
            assert t + 1 < len(ids), "a supervised position must have a next token"
            assert v == ids[t + 1], f"position {t} is not pre-shifted"


def test_partner_turns_are_never_supervised():
    """The model must learn to READ a partner turn, not produce one."""
    skit = _skit()
    tok = _Tok()
    ex = build_skit_example(skit, tok, with_think=True, pad_token_id=0)
    ids, labs = ex["input_ids"], ex["labels"]
    supervised_targets = [ids[t + 1] for t, v in enumerate(labs) if v != -100]
    for p in PARTNER_TURNS:
        partner_ids = tok.encode(skit.turns[p], add_special_tokens=False)
        # no contiguous run of the partner's ids may appear among supervised targets
        n = len(partner_ids)
        assert not any(supervised_targets[i:i + n] == partner_ids
                       for i in range(len(supervised_targets) - n + 1)), (
            f"partner turn {p} leaked into the supervised region")


def test_every_example_is_tile_aligned():
    """ttml's SDPA backward mismatches raw-T against tile-padded-T and dies with TT_FATAL."""
    for arm in (True, False):
        ex = build_skit_example(_skit(), _Tok(), with_think=arm, pad_token_id=0)
        assert len(ex["input_ids"]) % TILE == 0
        assert len(ex["labels"]) == len(ex["input_ids"])


def test_think_blocks_appear_only_in_the_think_arm():
    """Mutation guard: a build that ignored with_think would pass a length-only check."""
    skit, tok = _skit(), _Tok()
    with_t = build_skit_example(skit, tok, with_think=True, pad_token_id=0)
    without = build_skit_example(skit, tok, with_think=False, pad_token_id=0)
    from train.improv import render_think
    block_ids = tok.encode(render_think(skit.blocks[0]), add_special_tokens=False)
    n = len(block_ids)

    def contains(hay, needle):
        return any(hay[i:i + len(needle)] == needle
                   for i in range(len(hay) - len(needle) + 1))

    assert contains(with_t["input_ids"], block_ids), "think arm must carry the block"
    assert not contains(without["input_ids"], block_ids), (
        "no-think arm must not leak the block")
