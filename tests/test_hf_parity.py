# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Numerical verification of the converted model. CPU only."""

import hashlib
from pathlib import Path
from typing import Optional

import pytest

HF = Path("artifacts/hf")

#: The exact (model, tokenizer, tokens) triple the loss-comparison test below is meaningful
#: against. ``artifacts/hf`` is per-model *protected* evidence (train/paths.py's
#: PROTECTED_RELATIVE) -- it never changes out from under this test. The tokenizer and the
#: validation tokens are a different story: ``artifacts/tokenizer`` and ``artifacts/tokens``
#: are the *current*, shared, freely-regenerated locations (train/paths.py's SHARED_KINDS) --
#: whatever the newest corpus/tokenizer run happens to have left there, not necessarily what
#: this specific HF model was trained and held out on. Pointing this gate at those mutable
#: paths is exactly what silently broke it: the tokenizer was retrained (new nine-source
#: blend) and `artifacts/tokens` re-tokenized with it, while `artifacts/hf` kept the older
#: model those new tokens were never validated against -- and the gate could not tell,
#: because "whatever is currently there" isn't a claim about lineage. So this test pins the
#: triple explicitly instead, to the preserved v2 tokenizer and its own regenerated tokens
#: (see CLAUDE.md's ``parity-gate-restore`` entry for how these were restored).
TOKENIZER = Path("artifacts/tokenizer-tinystories-v2")
VAL_IDS = Path("artifacts/tokens-tinystories-v2/val_ids.npy")

#: sha256 of the preserved v2 tokenizer's tokenizer.json, pinned so a future accidental
#: overwrite of TOKENIZER (not just of the shared `artifacts/tokenizer`) is caught here
#: rather than silently invalidating VAL_IDS's lineage claim a second time.
_TOKENIZER_SHA256 = "09a98a92834eaeac502e77e8717f3f28aec7c523719752479230ee0ca85078b5"

pytestmark = pytest.mark.skipif(
    not (HF / "config.json").is_file(),
    reason="no converted model; run scripts/convert_checkpoint.py first",
)


def _v2_triple_skip_reason() -> Optional[str]:
    """Why the v2 parity gate can't run, or None if the pinned triple is present and intact."""
    if not VAL_IDS.is_file():
        return (
            f"no v2 validation tokens at {VAL_IDS}; regenerate with: python -m "
            f"train.tokenization --corpus artifacts/corpus/corpus.txt "
            f"--tokenizer {TOKENIZER} --out {VAL_IDS.parent}"
        )
    tok_file = TOKENIZER / "tokenizer.json"
    if not tok_file.is_file():
        return (
            f"the v2 tokenizer is missing at {tok_file}; {VAL_IDS} cannot be trusted to "
            f"match a tokenizer that isn't there to check against"
        )
    digest = hashlib.sha256(tok_file.read_bytes()).hexdigest()
    if digest != _TOKENIZER_SHA256:
        return (
            f"{tok_file} does not match the known-good v2 tokenizer digest "
            f"({_TOKENIZER_SHA256[:12]}..., got {digest[:12]}...) -- the preserved v2 "
            f"tokenizer appears to have changed, so {VAL_IDS}'s lineage can no longer be "
            f"trusted"
        )
    return None


#: The loss-comparison test additionally needs the pinned v2 (tokenizer, tokens) pair; skip
#: just that test (not the whole module) when it's absent or doesn't check out, so a machine
#: with a converted model but no v2 token cache still runs the structural/entropy tests above.
_v2_skip_reason = _v2_triple_skip_reason()
_no_val_ids = pytest.mark.skipif(
    _v2_skip_reason is not None,
    reason=_v2_skip_reason or "",
)

#: The training run's real held-out validation loss (ttml's own evaluate(), 10 batches of
#: 32 randomly-sampled 256-token windows). See CLAUDE.md's "numerical verification finds a
#: real bug" section for how this number was derived and cross-checked.
TRAINING_VAL_LOSS = 1.8781
#: Tasks 1-2 produce a directory that loads cleanly whether or not the conversion is
#: correct; this tolerance is what actually discriminates. The residual ~0.05-nat gap at
#: 1.9271 vs. 1.8781 is *not* fp32-CPU-vs-bf16-device precision -- measured directly, same
#: seed and windows, bf16 gives 1.9315 and fp32 gives 1.9314, a ~1e-4-nat difference. It's
#: sampling: three different seeds (0/1/2) give 1.9314 / 1.9208 / 1.8856, a seed-to-seed sd
#: of 0.024 nats, which puts the ~0.049-nat gap at z ~= 1.2 -- unremarkable noise, not a
#: signal. A gap of 1+ nats (as the straight-copied RoPE layout produced, before it was
#: fixed -- see convert.hf_mapping.permute_rope_qk) means a real layout bug; this tolerance
#: exists to catch that, not to police normal seed-to-seed variance.
LOSS_TOLERANCE = 0.2


def test_loads_with_automodel():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(HF))
    assert m.config.vocab_size == 32000
    assert m.config.num_key_value_heads == 3


def test_embedding_and_lm_head_are_tied():
    from transformers import AutoModelForCausalLM
    import torch

    m = AutoModelForCausalLM.from_pretrained(str(HF))
    assert torch.equal(m.model.embed_tokens.weight, m.lm_head.weight)


def test_forward_pass_produces_finite_logits():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(str(HF))
    m = AutoModelForCausalLM.from_pretrained(str(HF)).eval()
    ids = tok("Once upon a time", return_tensors="pt").input_ids
    with torch.no_grad():
        out = m(ids).logits
    assert out.shape[-1] == 32000
    assert torch.isfinite(out).all()


def test_next_token_distribution_is_not_uniform():
    """A mis-mapped model often yields near-uniform logits — entropy catches it.

    ln(32000) = 10.37 nats is the uniform ceiling. A model trained to ~1.88 loss must
    be far below that on in-distribution text.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(str(HF))
    m = AutoModelForCausalLM.from_pretrained(str(HF)).eval()
    ids = tok("Once upon a time there was a little girl named", return_tensors="pt").input_ids
    with torch.no_grad():
        logits = m(ids).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum().item()
    assert entropy < 7.0, f"next-token entropy {entropy:.2f} nats is near-uniform (10.37)"


@_no_val_ids
def test_validation_loss_matches_the_training_run():
    """The check every other test in this file is structurally incapable of doing.

    A wrong RoPE layout, a backwards K/V split, or swapped gate/up all produce a model
    that loads cleanly, ties its embedding, emits finite non-uniform logits, and *generates
    plausible fluent text* -- every test above this one would pass. This is the one that
    actually caught it: before ``convert.hf_mapping.permute_rope_qk`` existed, this
    computation returned 3.20 nats against a target of 1.8781 (a 1.32-nat gap) despite every
    structural and entropy check passing.

    Two things about this computation matter and are easy to get backwards:

    1. **No double shift.** ``LlamaForCausalLM``'s own loss function shifts ``labels``
       internally (``transformers.loss.loss_utils.ForCausalLMLoss``). Passing
       already-shifted ``input_ids``/``labels`` through the ``labels=`` kwarg -- as an
       earlier, incorrect version of this check did -- shifts a second time and reports a
       number close to the *uniform* ceiling regardless of whether the model is any good.
       This computes cross-entropy directly against ``logits`` instead, taking the
       ``labels=`` kwarg out of the picture entirely.
    2. **Random windows, not one contiguous block.** The training run's own validation loss
       (``train.run.evaluate``) averages 10 batches of 32 windows sampled uniformly across
       the *whole* validation set (``ttml.common.data.get_batch``), not one block from the
       front -- matched here so the two numbers are comparable.

    The seed below (``np.random.default_rng(0)``) is fixed deliberately, not incidentally:
    seed-to-seed variance on this exact computation is real and measured (seeds 0/1/2 give
    1.9314 / 1.9208 / 1.8856 nats, sd 0.024), so leaving the seed unpinned would make a
    0.2-nat gate flakier from one run to the next for no corresponding benefit.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(HF)).eval()
    val = np.load(VAL_IDS)
    seq_len, batch_size, num_batches = 256, 32, 10
    n = len(val) - seq_len - 1
    rng = np.random.default_rng(0)
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            ix = rng.integers(0, n, size=(batch_size,))
            x = np.stack([val[i:i + seq_len] for i in ix], axis=0).astype("int64")
            y = np.stack([val[i + 1:i + seq_len + 1] for i in ix], axis=0).astype("int64")
            logits = m(torch.from_numpy(x)).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), torch.from_numpy(y).reshape(-1)
            )
            losses.append(float(loss))
    mean_loss = sum(losses) / len(losses)
    gap = abs(mean_loss - TRAINING_VAL_LOSS)
    assert gap <= LOSS_TOLERANCE, (
        f"HF-side val loss {mean_loss:.4f} nats is {gap:.4f} nats from the training run's "
        f"{TRAINING_VAL_LOSS} (tolerance {LOSS_TOLERANCE}) -- suspect a layout bug "
        f"(RoPE interleaving, K/V split, or gate/up swap), not sampling noise."
    )
