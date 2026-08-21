"""Skit derivation. No hardware; `intensity` is injected so train/ never imports scripts/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.skit import (MIN_SENTENCES, MODEL_TURNS, PARTNER_TURNS, SKIT_ROLES,
                        derive_skit, skit_segments)

STORY = ("There was a cat. The cat slept all day. "
         "The cat stretched and yawned. "
         "The cat was sleepy. "
         "The sleepy cat curled up. "
         "The cat was purring. "
         "The purring was very loud.")


def _idf(words):
    return {w: 1.0 for w in words}


def test_a_skit_has_five_turns_and_three_blocks():
    s = derive_skit(STORY, story_id=0, idf=_idf(["sleepy", "purring", "loud"]),
                    intensity=lambda t: 0.0)
    assert s is not None
    assert len(s.turns) == 5
    assert len(s.blocks) == len(MODEL_TURNS) == 3
    assert SKIT_ROLES == ("model", "partner", "model", "partner", "model")


def test_offer_of_a_later_block_comes_from_the_preceding_partner_turn():
    """This is the change that makes `accept` mean something: turn 3 accepts what the
    PARTNER said, not what the prefix said."""
    s = derive_skit(STORY, story_id=0, idf=_idf(["sleepy", "purring"]),
                    intensity=lambda t: 0.0)
    assert s is not None
    partner_words = set(s.turns[PARTNER_TURNS[0]].lower().split())
    offer_words = set(s.blocks[1].offer.split())
    assert offer_words & {w.strip('.,!"') for w in partner_words}, (
        f"block 1's offer {s.blocks[1].offer!r} does not come from partner turn "
        f"{s.turns[PARTNER_TURNS[0]]!r}")


def test_stakes_is_measured_across_turns_not_within_one():
    """Stage 1 measured intensity delta inside a single continuation, which is the wrong
    interval — escalation in improv happens across an exchange."""
    seen = []

    def spy(text):
        seen.append(text)
        return 5.0 if "purring" in text else 0.0

    s = derive_skit(STORY, story_id=0, idf=_idf(["purring"]), intensity=spy)
    assert s is not None
    # the whole scene-so-far must never be handed to intensity(); only single turns
    assert all(t.count(".") <= 2 for t in seen), (
        f"intensity() was called on a multi-sentence span: {seen}")


def test_a_story_with_too_few_sentences_is_dropped():
    assert derive_skit("One. Two. Three.", story_id=0, idf={},
                       intensity=lambda t: 0.0) is None


def test_segments_supervise_only_think_blocks_and_model_turns():
    s = derive_skit(STORY, story_id=0, idf=_idf(["sleepy", "purring"]),
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
