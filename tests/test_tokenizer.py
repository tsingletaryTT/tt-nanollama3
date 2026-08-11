# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tokenizer training, export, and ttml load-path compatibility.

The two load paths asserted here mirror ttml's loader exactly
(tt-metal/tt-train/sources/ttml/ttml/common/data.py:93 and :96).
"""

from pathlib import Path

import pytest

from convert.tokenizer import SPECIAL_TOKENS, train_bpe

# A small vocab keeps these tests fast; the production value is 32000.
TEST_VOCAB = 500


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    """A small but varied corpus — enough for BPE to learn real merges."""
    d = tmp_path_factory.mktemp("corpus")
    p = d / "corpus.txt"
    sentences = [
        "Once upon a time there was a little girl named Lily.",
        "She liked to play in the garden with her red ball.",
        "One day the ball rolled under a big tree.",
        "Lily looked and looked but she could not find it.",
        "Her friend Tom came to help her search for the ball.",
    ]
    p.write_text("\n".join(sentences * 200) + "\n", encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def exported(corpus: Path, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tok") / "tokenizer"
    return train_bpe(corpus, out, vocab_size=TEST_VOCAB)


def test_export_writes_tokenizer_json(exported: Path):
    assert (exported / "tokenizer.json").is_file()


def test_vocab_size_respects_the_cap(exported: Path):
    """`vocab_size` is a ceiling, not a promise.

    BPE stops when it runs out of merges to learn. This fixture corpus is five
    sentences repeated, so it exhausts around 378 tokens and never reaches 500 —
    asserting equality here would fail a perfectly correct implementation. What must
    hold is that the cap is respected and that real merges were learned on top of the
    256-token byte alphabet. Exactness against the production 32000 is asserted in
    Task 3, against the real corpus, where it is actually reachable.
    """
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(exported / "tokenizer.json"))
    assert tok.vocab_size <= TEST_VOCAB
    # 256 byte-alphabet tokens + 4 specials; anything above proves merges were learned.
    assert tok.vocab_size > 260


def test_special_tokens_present(exported: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(exported), local_files_only=True)
    vocab = tok.get_vocab()
    for special in SPECIAL_TOKENS:
        assert special in vocab, f"{special} missing from vocab"


def test_roundtrip_preserves_text(exported: Path):
    """Round-trips modulo the leading space ``add_prefix_space=True`` injects.

    Byte-level BPE with ``add_prefix_space=True`` treats the start of a sequence as if
    it followed a space, so the first word tokenizes identically to a mid-line word
    (see the tokenizer.py comment on ``pre_tokenizers.ByteLevel``). Decoding faithfully
    reproduces that injected leading space when the source text doesn't already start
    with whitespace — this is standard GPT-2/RoBERTa-style tokenizer behavior, not data
    loss, so ``.strip()`` here removes exactly that one injected character and nothing
    from the interior of the text.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(exported), local_files_only=True)
    text = "Lily looked for the red ball."
    assert tok.decode(tok.encode(text), skip_special_tokens=True).strip() == text


def test_ttml_directory_load_path(exported: Path):
    """ttml data.py:93 — AutoTokenizer.from_pretrained(dir, local_files_only=True)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(exported), local_files_only=True)
    assert len(tok.encode("a story about a ball")) > 0


def test_ttml_file_load_path(exported: Path):
    """ttml data.py:96 — PreTrainedTokenizerFast(tokenizer_file=path)."""
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(exported / "tokenizer.json"))
    assert len(tok.encode("a story about a ball")) > 0


def test_convert_module_imports_no_tenstorrent():
    """convert/ must run on a machine with no hardware and no tt-metal checkout.

    Checked in a subprocess: this test session has already imported plenty, so
    inspecting our own sys.modules would prove nothing. torch is deliberately not
    banned — transformers imports it transitively and CPU torch runs anywhere.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import convert.tokenizer; "
        "bad=[m for m in ('ttnn','ttml') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True, cwd=str(Path(__file__).parent.parent),
    )
    assert out.stdout.strip() == "", f"convert.tokenizer pulled in: {out.stdout.strip()}"
