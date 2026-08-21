import json

from scripts.derive_traces import build_sft_examples, derive_from_story


STORY = ("Lily found a needle in her room. She knew it was sharp. "
         "She showed the needle to her mother. Her mother sewed the button. "
         "They were both happy with the shirt.")


class _Tok:
    """A faithful (non-lossy) stand-in tokenizer: one id per whitespace-separated word.

    Deliberately NOT `ord(c) % 97 for c in s[:20]` (the original brief's mock) — that
    truncates to 20 chars, so any two completions of >=20 chars collapse to the same
    token count regardless of content, making length-based assertions unable to fail.

    Deliberately NOT `hash(w)` either — Python's string hash is randomized per process
    (PYTHONHASHSEED), and this project's determinism guarantee (Task 1) depends on
    nothing in the pipeline depending on hash randomization. `sum(ord(c) for c in w)` is
    a stable, process-independent function of the word.
    """

    def encode(self, s, add_special_tokens=True):
        return [sum(ord(c) for c in w) % 31_000 for w in s.split()]


def test_derive_returns_a_trace_with_all_slots():
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    assert rec is not None
    assert set(rec["slots"]) == {"offer", "accept", "add", "stakes", "handback"}
    assert rec["prefix"] and rec["continuation"]
    assert "<think>" in rec["think"]


def test_sft_example_masks_only_the_prompt():
    """The mask is the whole point: prompt positions carry -100, completion carries ids.

    A hollow version of this test (asserting only labels[0] == -100 and "something is
    supervised") would still pass if the think-block were concatenated onto the PROMPT
    side instead of the completion side — which is exactly the worst-case bug for this
    project: it would train the model to never emit a think block. So this pins two
    exact facts instead: (1) the masked prefix is exactly as long as the prompt's own
    token count, no more, no less; and (2) the think-block's own token sequence appears
    verbatim inside the *supervised* (non -100) region of labels.
    """
    tok = _Tok()
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    ex = build_sft_examples([rec], tok, with_think=True)[0]
    assert len(ex["input_ids"]) == len(ex["labels"])

    prompt_ids = tok.encode(rec["prefix"])
    masked = [v for v in ex["labels"] if v == -100]
    assert len(masked) == len(prompt_ids), "masked region must be exactly the prompt"
    assert ex["labels"][:len(prompt_ids)] == [-100] * len(prompt_ids)

    supervised = ex["labels"][len(prompt_ids):]
    assert all(v != -100 for v in supervised), "nothing past the prompt should be masked"

    think_ids = tok.encode(rec["think"], add_special_tokens=False)
    assert supervised[:len(think_ids)] == think_ids, (
        "the think-block's tokens must land in the supervised region, not the prompt")


def test_no_think_arm_omits_the_block_but_keeps_the_continuation():
    """Assert the EXACT length delta, not just an inequality.

    `len(without) <= len(with_t)` is satisfied by equality, so it can't fail even if
    `with_think` is ignored entirely. The delta must equal precisely the think-block's
    own token count — that fails immediately if the flag is ignored, or if the block is
    concatenated in the wrong place, or duplicated.
    """
    tok = _Tok()
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    with_t = build_sft_examples([rec], tok, with_think=True)[0]
    without = build_sft_examples([rec], tok, with_think=False)[0]

    think_len = len(tok.encode(rec["think"], add_special_tokens=False))
    assert len(with_t["input_ids"]) - len(without["input_ids"]) == think_len
