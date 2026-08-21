import json

from scripts.derive_traces import (_sft_example_unaligned, build_sft_examples,
                                    derive_from_story)
from train.improv import Slots, render_think


STORY = ("Lily found a needle in her room. She knew it was sharp. "
         "She showed the needle to her mother. Her mother sewed the button. "
         "They were both happy with the shirt.")

PAD_TOKEN_ID = 999  # arbitrary; distinct from any id _Tok's checksum could plausibly emit


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
    project: it would train the model to never emit a think block. So this pins three
    exact facts instead: (1) the masked prefix is exactly as long as the prompt's own
    token count, no more, no less; (2) the think-block's own token sequence appears
    verbatim inside the *supervised* (non -100) region of labels; and (3) the ONLY other
    masked region is the trailing tile-alignment pad, which is legitimately -100 (padding
    must not contribute to the loss) rather than a sign the completion got clipped.
    """
    tok = _Tok()
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    ex = build_sft_examples([rec], tok, with_think=True, pad_token_id=PAD_TOKEN_ID)[0]
    assert len(ex["input_ids"]) == len(ex["labels"])
    assert len(ex["input_ids"]) % 32 == 0, "build_sft_examples must tile-align its output"

    prompt_ids = tok.encode(rec["prefix"])
    assert ex["labels"][:len(prompt_ids)] == [-100] * len(prompt_ids)

    completion_ids = tok.encode(rec["think"] + rec["continuation"], add_special_tokens=False)
    completion_region = ex["labels"][len(prompt_ids):len(prompt_ids) + len(completion_ids)]
    assert all(v != -100 for v in completion_region), (
        "nothing inside the actual completion should be masked")

    pad_region = ex["labels"][len(prompt_ids) + len(completion_ids):]
    assert pad_region == [-100] * len(pad_region), (
        "the tile-alignment pad tail must be masked too, or it would pull the loss "
        "toward predicting pad_token_id")
    assert ex["input_ids"][len(prompt_ids) + len(completion_ids):] == (
        [PAD_TOKEN_ID] * len(pad_region)), "pad tail on input_ids must use pad_token_id"

    think_ids = tok.encode(rec["think"], add_special_tokens=False)
    assert completion_region[:len(think_ids)] == think_ids, (
        "the think-block's tokens must land in the supervised region, not the prompt")


def test_no_think_arm_omits_the_block_but_keeps_the_continuation():
    """Assert the EXACT length delta, not just an inequality.

    `len(without) <= len(with_t)` is satisfied by equality, so it can't fail even if
    `with_think` is ignored entirely. The delta must equal precisely the think-block's
    own token count — that fails immediately if the flag is ignored, or if the block is
    concatenated in the wrong place, or duplicated.

    Asserted on `_sft_example_unaligned` (pre tile-alignment), not `build_sft_examples`'s
    public output: `build_sft_examples` pads each arm to the next multiple of 32
    independently, and that padding depends on each arm's own raw length mod 32 — so the
    *aligned* delta can differ from think_len by anywhere in [-31, +31] depending on the
    input, which would make this assertion either flaky or (worse) satisfiable by luck
    even with `with_think` ignored. The guarantee this test exists to pin — with_think
    adds exactly the think-block's own tokens, no more, no less — is a property of the
    unaligned construction; alignment is a separate, orthogonal concern covered by
    test_build_sft_examples_is_tile_aligned below.
    """
    tok = _Tok()
    rec = derive_from_story(STORY, story_id=0, rng_seed=5489)
    with_t = _sft_example_unaligned(rec, tok, with_think=True)
    without = _sft_example_unaligned(rec, tok, with_think=False)

    think_len = len(tok.encode(rec["think"], add_special_tokens=False))
    assert len(with_t["input_ids"]) - len(without["input_ids"]) == think_len


def _fake_trace(*, prefix: str, continuation: str) -> dict:
    """A trace dict with the same shape `derive_from_story` produces (prefix/think/
    continuation), built directly rather than via `derive_from_story` + `extract_slots`.

    Used only to get several DIFFERENT raw (pre-alignment) token lengths deterministically
    — `extract_slots` requires shared content words between prefix and continuation to
    return anything at all (see test_improv.py), which arbitrary short fixture text is not
    guaranteed to satisfy, and this test cares about exercising several remainders mod 32,
    not about `extract_slots`'s own selection logic (that is test_improv.py's job).
    """
    slots = Slots(offer="offer", accept="accept", add="add", stakes="level",
                  handback="handback")
    return {"prefix": prefix, "think": render_think(slots), "continuation": continuation}


def test_build_sft_examples_is_tile_aligned():
    """Every example build_sft_examples returns is a multiple of 32 tokens, both arms.

    This is the fix for the Task 2 SFTTrainer finding (task-2-report.md): ttml's SDPA
    backward kernel mismatches when a collated batch's sequence length is not a multiple
    of 32. sft_collate_fn pads a batch to its longest example, so tile-aligning every
    example here guarantees the batch max is aligned too. Uses several traces of
    DIFFERENT raw lengths (rather than one, repeated) so the raw pre-padding lengths land
    on different remainders mod 32 — a version that only right-pads to SOME fixed length
    regardless of input, or that is off by one tile boundary, would not reliably be caught
    by a single input whose raw length happens to already land on a multiple of 32.
    """
    tok = _Tok()
    recs = [
        _fake_trace(prefix="one two three", continuation="four five six seven"),
        _fake_trace(prefix=" ".join(f"w{i}" for i in range(20)),
                    continuation=" ".join(f"c{i}" for i in range(15))),
        _fake_trace(prefix="a", continuation="b"),
        derive_from_story(STORY, story_id=0, rng_seed=5489),
    ]
    raw_lengths = {len(_sft_example_unaligned(r, tok, with_think=True)["input_ids"]) % 32
                  for r in recs}
    assert len(raw_lengths) > 1, "fixture traces should span more than one remainder mod 32"

    for with_think in (True, False):
        examples = build_sft_examples(recs, tok, with_think=with_think,
                                      pad_token_id=PAD_TOKEN_ID)
        assert len(examples) == len(recs)
        for ex in examples:
            assert len(ex["input_ids"]) % 32 == 0
            assert len(ex["labels"]) == len(ex["input_ids"])
