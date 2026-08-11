# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Conversion utilities: tokenizer export and checkpoint conversion.

Nothing here may import ttnn or ttml — these must run on a machine with no
Tenstorrent hardware and no tt-metal checkout. (torch arrives transitively via
transformers and is fine; it is CPU-only here.)
"""
