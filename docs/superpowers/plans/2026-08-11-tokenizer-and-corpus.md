<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Tokenizer & Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a 32K-vocabulary BPE tokenizer and a prepared training corpus for NanoLlama3, exported in a format that both tt-train (`ttml`) and Hugging Face / vLLM load without modification.

**Architecture:** Two independent modules with no Tenstorrent dependencies. `train/data.py` fetches and prepares a plain-text corpus from TinyStories. `convert/tokenizer.py` trains a byte-level BPE over that corpus and exports it via `PreTrainedTokenizerFast.save_pretrained()`, which writes both a standalone `tokenizer.json` and a directory loadable by `AutoTokenizer` — the two forms `ttml`'s loader accepts. Neither module imports `ttnn`, so both run on any machine.

**Tech Stack:** Python 3.10+, `tokenizers` 0.21.4, `transformers` 4.52.4, `huggingface_hub` 0.36.2, pytest

## Global Constraints

- All new files carry an Apache-2.0 SPDX header: `# SPDX-License-Identifier: Apache-2.0` and `# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC`
- Python 3.10+
- **Vocabulary size is exactly 32000**, to match `~/tt-metal/tt-train/configs/model_configs/nanollama3.yaml` (`vocab_size: 32000`). Special tokens count toward this total.
- **Neither `train/` nor `convert/` may import `ttnn`, `ttml`, or `torch`.** These modules must run on a machine with no Tenstorrent hardware.
- `ttml` resolves `tokenizer_path` **relative to `$TT_METAL_HOME/tt-train`** (`sources/ttml/ttml/common/data.py:91`). Our exports live under `artifacts/tokenizer/` in this repo; wiring the path into a training config is Plan 2's job, not this plan's.
- `ttml` accepts a tokenizer as **either** a directory (`AutoTokenizer.from_pretrained(path, local_files_only=True)`, `data.py:93`) **or** a single file (`PreTrainedTokenizerFast(tokenizer_file=path)`, `data.py:96`). Both forms must work.
- Corpus: `roneneldan/TinyStories`, file `TinyStoriesV2-GPT4-train.txt` (and `TinyStoriesV2-GPT4-valid.txt` for validation). Verified present on the Hub.
- Model sequence length is 256 (`max_sequence_length` in `nanollama3.yaml`) — relevant later, not enforced here.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, pytest config, dev dependencies |
| `train/__init__.py` | Package marker |
| `train/data.py` | Fetch and prepare the plain-text corpus |
| `convert/__init__.py` | Package marker |
| `convert/tokenizer.py` | Train the BPE and export in both ttml-compatible forms |
| `tests/test_data.py` | Corpus preparation behavior |
| `tests/test_tokenizer.py` | Tokenizer training, export, round-trip, and ttml load-path compatibility |

`train/` and `convert/` are separate packages because they have different lifetimes: corpus preparation is a one-time setup step, while tokenizer export is re-run whenever the vocabulary changes.

---

## Task 1: Corpus fetch and preparation

**Files:**
- Create: `pyproject.toml`
- Create: `train/__init__.py`
- Create: `train/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `fetch_corpus(dest_dir: Path, split: str = "train") -> Path` — downloads the TinyStories file for `split` (`"train"` or `"valid"`) into `dest_dir`, returns the local path. Cached: returns immediately if the file already exists.
  - `prepare_corpus(src: Path, dest: Path, max_bytes: int | None = None) -> CorpusStats` — normalizes and optionally truncates, writes to `dest`, returns stats.
  - `CorpusStats` — dataclass with fields `bytes_written: int`, `line_count: int`, `truncated: bool`

- [ ] **Step 1: Create the package scaffolding**

`pyproject.toml`:

```toml
[project]
name = "tt-nanollama3"
version = "0.1.0"
description = "NanoLlama3 — a Tenstorrent-first model, trained, packaged, and served on TT tooling"
requires-python = ">=3.10"
dependencies = [
    "tokenizers>=0.21",
    "transformers>=4.52",
    "huggingface_hub>=0.36",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.setuptools]
packages = ["train", "convert"]
```

`train/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Corpus preparation and training launchers for NanoLlama3."""
```

- [ ] **Step 2: Write the failing tests**

`tests/test_data.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Corpus preparation behavior. No network, no Tenstorrent dependencies."""

from pathlib import Path

from train.data import CorpusStats, prepare_corpus


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_prepare_normalizes_crlf(tmp_path: Path):
    src = _write(tmp_path, "src.txt", "alpha\r\nbeta\r\n")
    dest = tmp_path / "out.txt"
    prepare_corpus(src, dest)
    assert "\r" not in dest.read_text(encoding="utf-8")


def test_prepare_drops_blank_lines(tmp_path: Path):
    src = _write(tmp_path, "src.txt", "alpha\n\n\nbeta\n")
    dest = tmp_path / "out.txt"
    stats = prepare_corpus(src, dest)
    assert dest.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert stats.line_count == 2


def test_prepare_reports_stats(tmp_path: Path):
    src = _write(tmp_path, "src.txt", "alpha\nbeta\n")
    dest = tmp_path / "out.txt"
    stats = prepare_corpus(src, dest)
    assert isinstance(stats, CorpusStats)
    assert stats.bytes_written == dest.stat().st_size
    assert stats.truncated is False


def test_prepare_truncates_at_max_bytes(tmp_path: Path):
    # Ten lines of 10 bytes each ("xxxxxxxxx\n"); cap at 25 bytes -> 2 whole lines.
    src = _write(tmp_path, "src.txt", "".join("x" * 9 + "\n" for _ in range(10)))
    dest = tmp_path / "out.txt"
    stats = prepare_corpus(src, dest, max_bytes=25)
    assert stats.truncated is True
    assert stats.line_count == 2
    assert stats.bytes_written <= 25


def test_prepare_never_splits_a_line(tmp_path: Path):
    src = _write(tmp_path, "src.txt", "short\n" + "y" * 100 + "\n")
    dest = tmp_path / "out.txt"
    prepare_corpus(src, dest, max_bytes=20)
    # Every written line must be complete.
    for line in dest.read_text(encoding="utf-8").splitlines():
        assert line in ("short", "y" * 100)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.data'`

- [ ] **Step 4: Implement `train/data.py`**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch and prepare the NanoLlama3 training corpus.

The corpus is TinyStories (``roneneldan/TinyStories``) — small, clean, and known to
produce coherent output at ~22M parameters, which is the scale this model targets.
We use the V2/GPT-4 variant, the higher-quality regeneration of the original.

This module deliberately imports nothing from Tenstorrent: corpus prep must run on any
machine, including one with no hardware and no tt-metal checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: The Hub dataset holding the corpus.
CORPUS_REPO = "roneneldan/TinyStories"

#: Filenames per split, verified present in the dataset repo.
CORPUS_FILES = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}


@dataclass
class CorpusStats:
    """What ``prepare_corpus`` produced, for logging and for the model card."""

    bytes_written: int
    line_count: int
    truncated: bool


def fetch_corpus(dest_dir: Path, split: str = "train") -> Path:
    """Download the TinyStories file for ``split`` into ``dest_dir``.

    Returns the local path. If the file is already present it is returned unchanged —
    the corpus is ~2 GB and re-downloading it is never what the caller wants.
    """
    if split not in CORPUS_FILES:
        raise ValueError(f"split must be one of {sorted(CORPUS_FILES)}, not {split!r}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / CORPUS_FILES[split]
    if local.exists():
        return local

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=CORPUS_REPO,
        filename=CORPUS_FILES[split],
        repo_type="dataset",
        local_dir=str(dest_dir),
    )
    return Path(downloaded)


def prepare_corpus(src: Path, dest: Path, max_bytes: Optional[int] = None) -> CorpusStats:
    """Normalize ``src`` into ``dest`` and report what was written.

    Normalization is deliberately minimal — the tokenizer should see text close to what
    the model will be served: CRLF collapsed to LF, blank lines dropped (they carry no
    signal and inflate the corpus), trailing whitespace stripped.

    ``max_bytes`` caps the output. The cap is applied on **whole lines only**: a partial
    final line would introduce a token boundary that never occurs in real text.
    """
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    lines = 0
    truncated = False

    with src.open("r", encoding="utf-8", errors="replace") as fin, \
            dest.open("w", encoding="utf-8", newline="\n") as fout:
        for raw in fin:
            line = raw.replace("\r\n", "\n").rstrip()
            if not line:
                continue
            encoded_len = len(line.encode("utf-8")) + 1  # +1 for the newline
            if max_bytes is not None and written + encoded_len > max_bytes:
                truncated = True
                break
            fout.write(line + "\n")
            written += encoded_len
            lines += 1

    return CorpusStats(bytes_written=written, line_count=lines, truncated=truncated)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_data.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
cd ~/code/tt-nanollama3
git add pyproject.toml train/ tests/test_data.py
git commit -m "feat(data): TinyStories corpus fetch and preparation

Whole-line truncation, CRLF normalization, blank-line removal. No
Tenstorrent imports so this runs on any machine."
```

---

## Task 2: BPE tokenizer training and export

**Files:**
- Create: `convert/__init__.py`
- Create: `convert/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Consumes: `train.data.prepare_corpus` (tests use a prepared corpus file)
- Produces:
  - `VOCAB_SIZE: int = 32000` — module constant, matches `nanollama3.yaml`
  - `SPECIAL_TOKENS: list[str]` — `["<unk>", "<s>", "</s>", "<pad>"]`
  - `train_bpe(corpus: Path, out_dir: Path, vocab_size: int = VOCAB_SIZE) -> Path` — trains a byte-level BPE and writes the export directory; returns `out_dir`
  - `load_exported(out_dir: Path)` — returns the tokenizer via the directory path, for verification

- [ ] **Step 1: Write the failing tests**

`tests/test_tokenizer.py`:

```python
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


def test_vocab_size_is_exact(exported: Path):
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(exported / "tokenizer.json"))
    assert tok.vocab_size == TEST_VOCAB


def test_special_tokens_present(exported: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(exported), local_files_only=True)
    vocab = tok.get_vocab()
    for special in SPECIAL_TOKENS:
        assert special in vocab, f"{special} missing from vocab"


def test_roundtrip_preserves_text(exported: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(exported), local_files_only=True)
    text = "Lily looked for the red ball."
    assert tok.decode(tok.encode(text), skip_special_tokens=True) == text


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
    """convert/ must run on a machine with no hardware."""
    import sys

    import convert.tokenizer  # noqa: F401

    for banned in ("ttnn", "ttml", "torch"):
        assert banned not in sys.modules, f"convert.tokenizer pulled in {banned}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_tokenizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'convert.tokenizer'`

- [ ] **Step 3: Implement `convert/tokenizer.py`**

`convert/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Conversion utilities: tokenizer export and checkpoint conversion.

Nothing here may import ttnn, ttml, or torch — these run anywhere.
"""
```

`convert/tokenizer.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/code/tt-nanollama3 && python -m pytest tests/test_tokenizer.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the whole suite**

Run: `cd ~/code/tt-nanollama3 && python -m pytest -v`
Expected: 12 passed

- [ ] **Step 6: Commit**

```bash
cd ~/code/tt-nanollama3
git add convert/ tests/test_tokenizer.py
git commit -m "feat(tokenizer): 32K byte-level BPE with ttml-compatible export

save_pretrained() writes both the standalone tokenizer.json and the
directory form, covering both of ttml's load paths (data.py:93,96) and
vLLM's later. Byte-level so no input is unrepresentable."
```

---

## Task 3: Produce the real artifacts

**Files:**
- Create: `scripts/build_tokenizer.py`
- Create: `artifacts/.gitignore`

**Interfaces:**
- Consumes: `train.data.fetch_corpus`, `train.data.prepare_corpus`, `convert.tokenizer.train_bpe`
- Produces: `artifacts/corpus/corpus.txt` and `artifacts/tokenizer/` on disk — the inputs Plan 2 trains against. No new Python API.

- [ ] **Step 1: Keep large artifacts out of git**

`artifacts/.gitignore`:

```gitignore
# Corpus and tokenizer artifacts are built, not committed. The tokenizer is small
# enough to publish with the model bundle; the corpus is ~2 GB and never belongs in git.
*
!.gitignore
```

- [ ] **Step 2: Write the build script**

`scripts/build_tokenizer.py`:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Build the corpus and tokenizer artifacts NanoLlama3 trains against.

    python scripts/build_tokenizer.py --corpus-mb 512

Downloading the full TinyStories train split takes a while and most of it is not needed
to fit a 32K vocabulary, so ``--corpus-mb`` caps what the tokenizer trains on. The cap
applies to whole lines only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convert.tokenizer import VOCAB_SIZE, train_bpe  # noqa: E402
from train.data import fetch_corpus, prepare_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-mb", type=int, default=512,
                        help="Megabytes of corpus to keep for tokenizer training "
                             "(default: 512). Whole lines only.")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help=f"Total vocabulary including specials (default: {VOCAB_SIZE}).")
    args = parser.parse_args()

    raw_dir = ARTIFACTS / "raw"
    corpus_out = ARTIFACTS / "corpus" / "corpus.txt"
    tok_out = ARTIFACTS / "tokenizer"

    print(f">> Fetching TinyStories into {raw_dir}")
    raw = fetch_corpus(raw_dir, split="train")

    print(f">> Preparing corpus (cap {args.corpus_mb} MB) -> {corpus_out}")
    stats = prepare_corpus(raw, corpus_out, max_bytes=args.corpus_mb * 1024 * 1024)
    print(f"   {stats.line_count:,} lines, {stats.bytes_written:,} bytes, "
          f"truncated={stats.truncated}")

    print(f">> Training {args.vocab_size}-token BPE -> {tok_out}")
    train_bpe(corpus_out, tok_out, vocab_size=args.vocab_size)

    print("\nDone. Artifacts:")
    print(f"  corpus:    {corpus_out}")
    print(f"  tokenizer: {tok_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run it for real**

Run: `cd ~/code/tt-nanollama3 && python scripts/build_tokenizer.py --corpus-mb 512`

Expected: the corpus downloads (several minutes; ~2 GB raw), preparation reports a line count and `truncated=True`, and BPE training completes. Final output lists both artifact paths.

- [ ] **Step 4: Verify the artifacts load both ways**

Run:

```bash
cd ~/code/tt-nanollama3 && python -c "
from transformers import AutoTokenizer, PreTrainedTokenizerFast
d = AutoTokenizer.from_pretrained('artifacts/tokenizer', local_files_only=True)
f = PreTrainedTokenizerFast(tokenizer_file='artifacts/tokenizer/tokenizer.json')
assert d.vocab_size == 32000, d.vocab_size
assert f.vocab_size == 32000, f.vocab_size
t = 'Once upon a time there was a little girl.'
assert d.decode(d.encode(t), skip_special_tokens=True) == t
print('OK — vocab', d.vocab_size, '| tokens for sample:', len(d.encode(t)))
"
```

Expected: `OK — vocab 32000 | tokens for sample: <n>`

- [ ] **Step 5: Commit**

```bash
cd ~/code/tt-nanollama3
git add scripts/build_tokenizer.py artifacts/.gitignore
git commit -m "feat(scripts): build_tokenizer.py produces corpus + tokenizer artifacts

Artifacts are gitignored — the corpus is ~2 GB. The tokenizer directory
is small and ships with the model bundle later."
```

---

## Self-Review

**Spec coverage.** This plan implements the spec's Stage 1 (Tokenizer and data) in full: the 32K BPE trained with the `tokenizers` library and exported in HF format, TinyStories as the corpus, and the explicit decision not to ship the template's pure-Python `tokenizer_bpe.py`. Stage 2 (training) and everything downstream are out of scope here and blocked on the open question below.

**Placeholders.** None. Every step carries runnable code or an exact command with expected output.

**Type consistency.** `CorpusStats` fields (`bytes_written`, `line_count`, `truncated`) are used consistently in Task 1's tests, its implementation, and Task 3's script. `VOCAB_SIZE` and `SPECIAL_TOKENS` are defined in Task 2 and consumed by name in Task 3. `train_bpe(corpus, out_dir, vocab_size)` has one signature throughout.

**Scope.** One subsystem, three tasks, each independently testable and committable.

## Findings that shape Plan 2

The trainer question is **resolved, and no tt-metal upgrade is required.**

**The trainer exists, at a new path.** `~/tt-metal/tt-train/sources/examples/python/transformers/training.py`
(72 lines) is the current Python trainer. It takes `-c/--config`, resolved by
`load_config` relative to `$TT_METAL_RUNTIME_ROOT/tt-train/configs/` (`common/config.py:209-220`),
and drives `prepare_data` → `TransformerModelFactory` → `create_optimizer` → `TrainingConfig`,
with DDP and TP support. It supersedes the `train_nanogpt.py` that the lesson template's
`train_nano_from_scratch.py` still shells out to — that path exists only in the much older
`~/code/tt-metal` (`v0.66.0-dev20260123`), so **the lesson launcher is broken against current
tt-metal**. Worth reporting back to `tt-vscode-toolkit`.

**Vocabulary is taken from the tokenizer, not the config.** `training.py:40-43` overwrites
`transformer_config["vocab_size"]` with the value `prepare_data` returns. So our 32K export
determines the model's vocabulary automatically, and `nanollama3.yaml`'s declared `32000` only
needs to agree — it is not the source of truth.

**The one real obstacle: the corpus path is hardcoded.** `prepare_data` (`common/data.py:82-83`)
calls `load_shakespeare_text()`, which reads
`$TT_METAL_HOME/tt-train/data/shakespeare.txt` with **no parameter and no config key**
(`data.py:27-33`). It ignores `data_path` entirely. Training on TinyStories therefore requires
either overwriting that file in the tt-metal tree, or supplying our own data loading.

**Recommendation for Plan 2: write our own training entrypoint in this repo.** It reuses
everything ttml genuinely provides — `TransformerModelFactory`, `create_optimizer`,
`TrainingConfig`, `DeviceConfig`, the trainer loop — and replaces only the hardcoded
Shakespeare loader with our prepared corpus. That respects the spec's reuse policy (never
reimplement what ttml provides) while removing the one hardcoded assumption that blocks us,
and it avoids mutating the tt-metal checkout. For a repo whose purpose is to be the reference
example, owning the training entrypoint is also the honest choice.

**Note for the BPE path:** ttml's `PAD_TOKEN`/`BEGIN_TOKEN`/`END_TOKEN` constants
(`data.py:36-38`) are `<PAD>`/`<BEG>`/`<END>` and belong to its `CharTokenizer`. They do not
apply to our BPE export, whose specials are defined in `convert/tokenizer.py`.
