<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# tt-kernel v4 conformance: what this model exercises, and what it found

This project exists partly to be a **hard consumer** of
[tt-kernel](https://github.com/tenstorrent/tt-kernel-package-manager) — to author a real v4
bundle, serve it, and report what breaks. This file is that report.

Everything below was measured against tt-kernel at `00dba42` by authoring manifests and
running tt-kernel's own rendering code, plus one live serving run. It is written from the
position of a model author reading the schema and expecting the documented behaviour.

## Coverage: what our manifest exercises

Of the 26 fields a v4 model author can write, our shipped
[`tt_kernel_manifest.json`](../tt_kernel_manifest.json) sets 13:

| Exercised | Not exercised (and why) |
|---|---|
| `platform.ttnn`, `runtime.kind` | `runtime.version`, `runtime.plugin_version` — vLLM exposes no dist version in our env |
| `entrypoint.class`, `entrypoint.arch_name` | `capabilities.*` — this is a base model, no tool/reasoning parsing |
| `weights.repo` | `weights.revision`/`allow_patterns`/`ignore_patterns`/`repo_type` |
| `mesh.devices`, `mesh.topology` | `mesh.fabric` — single chip, no fabric |
| `resources.max_model_len`, `max_num_seqs` | `resources.block_size`, `trace_region_bytes`, `extra_args`, `command_override` |
| `target`, `env`, `name`, `description` | |

A **maximal** manifest setting all 26 was rendered through `render_vllm_metadata` to check
the unexercised ones compose correctly. Most do — see below.

## What works, verified

- **Launch composition is faithful.** `block_size`, `trace_region_bytes`, `tool_parser`,
  `reasoning_parser`, and `extra_args` all appear in the rendered command, in a sane order
  (composed args first, `extra_args` appended last).
- **`command_override` replaces per machine.** A `"p300"` key produces its own launch entry
  while `default` keeps the composed command. Both inherit the composed env.
- **`weights.revision` / `allow_patterns` / `ignore_patterns` are honoured** at weights
  download (`runtime.py:68-70`). Worth stating because the rendered `vllm_metadata.json`
  shows only a bare repo id in `hf_weights`, which *looks* like the pin was dropped. It
  isn't — tt-kernel fetches the pinned revision into the local cache itself.
- **`mesh.devices` wins over the pusher's box** (`cli.py:332-334`), so a 4-chip model
  authored on a 1-chip dev host publishes `device_count: 4`. This is a good design
  decision and the comment says so explicitly.
- **Arch normalisation on push** (`cli.py:330`) turns `--arch bh` into `blackhole`, which
  prevents a bundle that is invisible to `search --arch blackhole`.

## Findings

### 1. `mesh.topology` and `mesh.fabric` are declared but never read

`Mesh.topology` and `Mesh.fabric` have **zero consumers** anywhere in `src/tt_kernel/`
outside their own definition. `mesh.devices` is consumed; these two are not.

This matters because `Mesh`'s own docstring says the opposite:

> Structured (rather than buried in an opaque launch command's env) so the launch renderer
> can compose `MESH_DEVICE`/fabric env and so search can reason about topology.

An author reading that will set `fabric: FABRIC_1D_RING` and expect it applied. It is
silently ignored. For multi-chip Blackhole this is not hypothetical: fabric configuration
is exactly what a 2/4-chip run needs, and getting it wrong produces
`Fabric Router Sync: Timeout` rather than a clear error.

**Suggested fix:** either compose them into the launch env, or drop the promise from the
docstring and mark both as search-only metadata. The current state is the worst of the
three, because it reads as implemented.

### 2. `HF_MODEL` is required for serving but nothing says so

tt-kernel composes `--model <repo>` into the launch command from `weights.repo`. That is
not sufficient: `tt_transformers` reads the model id from the **`HF_MODEL` environment
variable** and raises

```
ValueError: Please set HF_MODEL to a HuggingFace name e.g. meta-llama/Llama-3.1-8B-Instruct
```

if it is unset. We found this by serving and hitting it, not by reading anything.

The author must know to duplicate the repo id into `manifest.env`, and nothing validates
that they did. Since tt-kernel already knows `weights.repo`, it could either default
`HF_MODEL` from it or warn when a vLLM-backend manifest omits it.

**Impact:** every v4 vLLM bundle needs this, so every author hits it once.

### 3. `tag_repo` destroys model-card front matter

Recorded previously and unchanged: `hub.tag_repo` (`hub.py:56-66`) replaces `card.data`
wholesale with `ModelCardData(tags=...)`, discarding `license`, `pipeline_tag`,
`library_name`, `datasets`, and `base_model`. It runs on **every** push
(`cli.py:288`, `cli.py:503`) and on `set_catalog_listing`.

The prose body survives; only front matter is lost. Since the repo-level license setting is
independent of card front matter, the workaround is to set the license there and re-apply
front matter after any tt-kernel operation — which is what
[`scripts/publish_to_hub.py --restore-card`](../scripts/publish_to_hub.py) exists to do.

### 4. Not a tt-kernel defect, but it bit us: the manifest cannot express a runtime patch

Our model needs a tt-metal change to run at all on a harvested Blackhole (see
[`bundle/tt_nanollama3_adapter.py`](../bundle/tt_nanollama3_adapter.py)). The v4 schema has
no field for "this model requires a patch", and it does not need one — the adapter folder
(`runner.bundle_dir`) is the right vehicle, and it worked: the plugin imported our module,
the patch applied at import time, and the model ran.

Recording it here because it is the interesting positive result. **A model can carry the
tt-metal change it needs without that change landing upstream first**, and the v4
bundle-folder design is what makes that possible. Worth stating in tt-kernel's own docs as
a supported pattern rather than leaving authors to discover it.

## Still unproven

- **Serving correctness.** The bundle registers, launches, and generates, but generated
  text is wrong (a decode-path defect, unrelated to tt-kernel). Nothing in this file should
  be read as "the model serves correctly" — only that the *packaging* path works.
- **Multi-chip anything.** `num_groups=3` restricts this model to single-chip serving, so
  `mesh.devices > 1`, fabric, and multi-machine `command_override` are untested against
  real hardware.
- **`tt-kernel push` / `pull` / catalog.** Not yet run for this model; the serving test laid
  the bundle down by hand.
