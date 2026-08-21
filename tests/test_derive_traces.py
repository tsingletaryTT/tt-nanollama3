import json

from scripts.derive_traces import build_sft_examples, derive_from_story


STORY = ("Lily found a needle in her room. She knew it was sharp. "
         "She showed the needle to her mother. Her mother sewed the button. "
         "They were both happy with the shirt.")


def test_derive_returns_a_trace_with_all_slots():
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    assert rec is not None
    assert set(rec["slots"]) == {"offer", "accept", "add", "stakes", "handback"}
    assert rec["prefix"] and rec["continuation"]
    assert "<think>" in rec["think"]


def test_sft_example_masks_only_the_prompt():
    """The mask is the whole point: prompt positions carry -100, completion carries ids."""
    class _Tok:
        def encode(self, s, add_special_tokens=True):
            return [ord(c) % 97 for c in s[:20]]

    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    ex = build_sft_examples([rec], _Tok(), with_think=True)[0]
    assert len(ex["input_ids"]) == len(ex["labels"])
    assert ex["labels"][0] == -100, "prompt must be masked"
    assert any(v != -100 for v in ex["labels"]), "completion must be supervised"


def test_no_think_arm_omits_the_block_but_keeps_the_continuation():
    class _Tok:
        def encode(self, s, add_special_tokens=True):
            return [ord(c) % 97 for c in s[:20]]

    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    with_t = build_sft_examples([rec], _Tok(), with_think=True)[0]
    without = build_sft_examples([rec], _Tok(), with_think=False)[0]
    assert len(without["input_ids"]) <= len(with_t["input_ids"])
