# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Train and export the NanoLlama3 BPE tokenizer.

The export is written with ``PreTrainedTokenizerFast.save_pretrained()``, which produces
both a standalone ``tokenizer.json`` and the sidecar config a directory load needs. That
matters because ttml accepts either form — a directory via
``AutoTokenizer.from_pretrained`` or a bare file via ``PreTrainedTokenizerFast`` (see
tt-train ``sources/ttml/ttml/common/data.py:93,96``) — and the same directory is what
vLLM loads later. One artifact, three consumers.

Byte-level BPE is used so the tokenizer can never emit an out-of-vocabulary byte; every
input is representable, which removes an entire class of serving-time failure.
"""

from __future__ import annotations

from pathlib import Path

#: Must match ``vocab_size`` in tt-train's ``model_configs/nanollama3.yaml``.
VOCAB_SIZE = 32000

#: Special tokens, in the order they are assigned ids 0..3. ``<unk>`` is first so an
#: unknown id decodes visibly rather than silently vanishing.
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]


def train_bpe(corpus: Path, out_dir: Path, vocab_size: int = VOCAB_SIZE) -> Path:
    """Train a byte-level BPE over ``corpus`` and export it to ``out_dir``.

    ``vocab_size`` is the **total** including ``SPECIAL_TOKENS`` — the trainer is given
    the full target and reserves the specials itself, so the exported vocabulary is
    exactly ``vocab_size`` and matches what the model config declares.
    """
    from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    corpus, out_dir = Path(corpus), Path(out_dir)
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus}")

    tokenizer = Tokenizer(BPE(unk_token=None))
    # add_prefix_space keeps the first word of a line consistent with mid-line words,
    # so "ball" tokenizes identically at either position.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train([str(corpus)], trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out_dir))
    return out_dir


def load_exported(out_dir: Path):
    """Load an exported tokenizer the way ttml loads a directory. Verification helper."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(Path(out_dir)), local_files_only=True)
