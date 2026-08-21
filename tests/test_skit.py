"""Skit derivation. No hardware; `intensity` is injected so train/ never imports scripts/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.improv import content_words
from train.skit import (MIN_SENTENCES, MODEL_TURNS, PARTNER_TURNS, SKIT_ROLES,
                        derive_skit, skit_segments)

# Story with turn-unique vocabulary to enable identity-based assertions.
# Turns carry words forward to satisfy accept/add constraints, but each turn has
# mostly distinct content words to distinguish correct sourcing from incorrect.
STORY = ("Aardvark exists. Aardvark runs. "
         "Aardvark jumps. Bear arrives. "
         "Bear sleeps. Eagle flies. "
         "Eagle sings.")


def _idf(words):
    return {w: 1.0 for w in words}


def test_a_skit_has_five_turns_and_three_blocks():
    s = derive_skit(STORY, story_id=0, idf=_idf(["aardvark", "bear", "eagle"]),
                    intensity=lambda t: 0.0)
    assert s is not None
    assert len(s.turns) == 5
    assert len(s.blocks) == len(MODEL_TURNS) == 3
    assert SKIT_ROLES == ("model", "partner", "model", "partner", "model")


def test_offer_of_a_later_block_comes_from_the_preceding_partner_turn():
    """Offer must come from immediately preceding PARTNER turn, not from scene-so-far.

    Mutation: source offer from prefix+all-prior-turns instead of just preceding turn.
    This test catches that by asserting offer is a subset of the specific partner turn
    AND disjoint from the preceding model turn (which has different vocabulary).
    """
    s = derive_skit(STORY, story_id=0, idf=_idf(["bear", "eagle"]),
                    intensity=lambda t: 0.0)
    assert s is not None

    # Block 1: offer must come from PARTNER_TURNS[0] = turn 1 ("Bear arrives")
    # Turn 0 ("Aardvark jumps") has disjoint vocabulary.
    partner_1_words = set(content_words(s.turns[PARTNER_TURNS[0]]))
    turn_0_words = set(content_words(s.turns[0]))
    block_1_offer_words = set(s.blocks[1].offer.split())

    # Offer must be drawn from partner turn 1 only
    assert block_1_offer_words.issubset(partner_1_words), (
        f"block 1's offer {block_1_offer_words} must come from turn {PARTNER_TURNS[0]} "
        f"{partner_1_words}, not from scene-so-far")

    # Offer must NOT include words unique to the preceding model turn (identity check)
    assert block_1_offer_words.isdisjoint(turn_0_words), (
        f"block 1's offer {block_1_offer_words} should not contain turn 0's words {turn_0_words}")


def test_stakes_is_measured_across_turns_not_within_one():
    """Stakes computed as intensity(turn) - intensity(prev_turn), not vs the scene.

    Mutation: always diff against prefix instead of against the immediately preceding turn.
    This test catches that by recording all intensity() calls and asserting the exact
    (current_turn, previous_turn) pairs passed to intensity() for each block.
    """
    seen = []

    def spy(text):
        seen.append(text)
        return 5.0 if "eagle" in text.lower() else 0.0

    s = derive_skit(STORY, story_id=0, idf=_idf(["eagle"]), intensity=spy)
    assert s is not None

    # intensity() must be called once per block on current turn, once on previous turn.
    # Expected call sequence (order of subtraction is: turn - prev):
    # Block 0: intensity(turn 0), intensity(prefix)
    # Block 1: intensity(turn 2), intensity(turn 1)
    # Block 2: intensity(turn 4), intensity(turn 3)
    expected_calls = [
        s.turns[0],   # Block 0: current turn
        s.prefix,     # Block 0: previous turn (prefix)
        s.turns[2],   # Block 1: current turn
        s.turns[1],   # Block 1: previous turn (partner)
        s.turns[4],   # Block 2: current turn
        s.turns[3],   # Block 2: previous turn (partner)
    ]

    assert seen == expected_calls, (
        f"intensity() calls {seen} do not match expected block deltas {expected_calls}")


def test_story_with_too_few_sentences_returns_none():
    """Stories with fewer than MIN_SENTENCES sentences are dropped.

    Note: the MIN_SENTENCES gate is an explicit precondition. The downstream
    len(sents[2:7]) != 5 check in derive_skit is the structural enforcer, but the gate
    makes the intent explicit and is less fragile than relying on slice-width semantics.
    """
    assert derive_skit("One. Two. Three.", story_id=0, idf={},
                       intensity=lambda t: 0.0) is None


def test_segments_supervise_only_think_blocks_and_model_turns():
    s = derive_skit(STORY, story_id=0, idf=_idf(["aardvark", "bear"]),
                    intensity=lambda t: 0.0)
    assert s is not None
    segs = skit_segments(s)
    # prefix unsupervised, then (think, turn) supervised pairs with partners between
    assert segs[0][1] is False, "the prefix must never be supervised"
    supervised = [text for text, flag in segs if flag]
    assert len(supervised) == 6, "3 think-blocks + 3 model turns"
    for partner_idx in PARTNER_TURNS:
        assert all(s.turns[partner_idx] not in text for text in supervised), (
            "a partner turn leaked into the supervised region; the model must learn to "
            "READ a partner turn, not produce one")
