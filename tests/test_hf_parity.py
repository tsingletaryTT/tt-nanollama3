# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Numerical verification of the converted model. CPU only."""

from pathlib import Path

import pytest

HF = Path("artifacts/hf")

pytestmark = pytest.mark.skipif(
    not (HF / "config.json").is_file(),
    reason="no converted model; run scripts/convert_checkpoint.py first",
)


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
