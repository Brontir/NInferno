# Weight conversion

The converter creates `.ninferno` files from the original Hugging Face checkpoint (or a local
checkpoint directory, or a GGUF file) and a recipe. NInferno publishes no prebuilt model files, so
conversion is the only way to get a model. The output holds the configuration, encoded weights,
logical bindings and frontend resources. A converted file runs on any supported GPU (Volta `sm_70`
through Blackwell `sm_120`). NVFP4 weights run natively on `sm_120` (RTX 50); older GPUs run them on slower generic kernels, where a GGUF or FP8 checkpoint is the measured choice.

Run the commands below from the repository root.

## Quick start

**`ninferno-serve <org/name[@rev]>` converts automatically.** On first start it downloads the hub
model and converts it with the official recipe. To convert without serving, use
`ninferno-serve --prepare-only <org/name>` ([serving guide](serving.md)).

For explicit control, use `ninferno-convert`:

```bash
# fetch from the hub and convert with the official recipe for the architecture
./build/apps/ninferno-convert --fetch org/name --out models/name.ninferno

# a published GGUF quantization: --quant picks the file, --with-drafters adds the drafters
./build/apps/ninferno-convert --fetch unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL --with-drafters \
  --out models/qwen3_6_35b_a3b.ninferno

# convert a local checkpoint directory or a GGUF file
./build/apps/ninferno-convert --model /path/to/checkpoint --out models/name.ninferno
```

For Python recipes (custom recipes, overrides, custom sources), use `python3 -m tools.convert`.
It needs Python 3.11 with PyTorch and NumPy, uses CUDA by default (`--device cpu` selects CPU), and
requires `--model`, `--recipe` and `--out`:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --proposal \
  --name qwen3.6-27b \
  --out models/qwen3_6_27b.ninferno
```

## What `ninferno-convert` does

- `--model` takes a Hugging Face checkpoint directory, a `.gguf` file (the first part of a split
  set), or a directory holding one GGUF model.
- `--fetch` downloads one instead: `org/name[@rev]` from the hub (only the files a conversion
  reads), or an http(s) URL of one file. For a GGUF repository, `--quant` picks one quantization;
  the default preference is NVFP4, MXFP4, Q8_0, BF16, ... Files go into the Hugging Face cache
  layout (`$HF_HUB_CACHE`, `$HF_HOME/hub`, `$XDG_CACHE_HOME/huggingface/hub`, `~/.cache/huggingface/hub`). `$HF_TOKEN` (else
  `$HUGGING_FACE_HUB_TOKEN`) authorizes gated repositories, `$HF_ENDPOINT` points at another hub
  (`$MODELSCOPE_ENDPOINT` and `$MODELSCOPE_API_TOKEN` for ModelScope). `$HF_HUB_OFFLINE` (`1`,
  `true`, `yes` or `on`) uses the cached snapshot only.
- Before downloading weights, `config.json` alone decides whether the architecture converts, and the
  cache's free space is checked.
- `--fetch` without `--out` writes the artifact to
  `<cache>/models--org--name/ninferno/<commit>/<stem>.<recipe>.ninferno`, and running the same
  command again reuses it. The kept artifact is found by commit, quant and recipe, so a flag
  that changes what is written (`--components`, `--layout`, `--vocab`, `--drafter-quant`,
  `--proposal`, `--include`, `--no-drafters`, `--max-file-bytes`, ...) converts again and
  replaces it, and an explicit `--out` is never reused. `--model` needs `--out`.
- Without `--recipe`, the official recipe for the checkpoint's architecture is chosen (with its
  quantized source when it needs one).
- A model runs as published. A quantized checkpoint keeps its encoding (NVFP4, MXFP4, FP8 with
  per-tensor, per-channel or 128x128 block scales, AWQ, GPTQ with or without act-order, AutoRound,
  compressed-tensors INT4/INT8). An unquantized one keeps its projections in BF16 where its family
  has a BF16 recipe (dense decoders, MoE decoders, Qwen3.5 Dense, Gemma 4). An unquantized
  Qwen3.5 MoE (Qwen3.6-35B-A3B) has none: convert a quantized checkpoint of it (a GGUF, or one
  `qwen3_5_published` keeps), or quantize it with `--maintainer-quantize`.
- A published quantization the engine cannot run as published is refused, with the reason and the
  alternatives. It is never dequantized and quantized again.
- A recipe that would quantize an unquantized checkpoint is refused too (`dense_q8`, `dense_fp8`,
  `dense_nvfp4`, `moe_q8`, `qwen3_5_q8`, `qwen3_6_27b`, `qwen3_8_27b`, `qwen3_6_35b_a3b`, or an
  `_nvfp4` recipe on a checkpoint without NVFP4). `--maintainer-quantize` allows it; the artifact is
  then NInferno's own quantization.
- Conversion rejects missing logical coverage, invalid source geometry, unsupported encodings and
  invalid method output.

### Official recipes

Recipe names select conversion choices only. Runtime execution is selected from the architecture,
configuration and bindings stored in the artifact. `--name` sets the public model name; it does not
select kernels. Recipes are Python functions in
[`official_recipes.py`](../tools/convert/official_recipes.py) and the recipe registry.

| Architecture | Recipe | Formats | Drafters |
|---|---|---|---|
| Qwen3.6-27B | `qwen3_6_27b` | Q4/Q5 projections, Q6 vocabulary | MTP, DFlash |
| Qwen3.8-27B | `qwen3_8_27b` | Q4/Q5 projections, Q8 vocabulary | MTP, DFlash2 |
| Qwen3.6-35B-A3B | `qwen3_6_35b_a3b` | Q4 experts, Q5/Q6 expert down, Q8 shared/projection | MTP, DFlash |
| Qwen3.6-35B-A3B NVFP4 | `qwen3_6_35b_a3b_nvfp4` | Imported NVFP4 routed and shared experts, Q8 projections, Q8/Q6 vocabulary; needs `--source quantized=PATH` | MTP, DFlash |
| Qwen3.6-27B NVFP4 | `qwen3_6_27b_nvfp4` | Imported NVFP4, selected BF16 projections, Q8 vocabulary; needs `--source quantized=PATH` | MTP, DFlash |
| Qwen3.8-27B NVFP4 | `qwen3_8_27b_nvfp4` | Imported NVFP4/FP8, FP8 embedding generated from BF16; needs `--source quantized=PATH` | MTP, DFlash2 |
| Qwen3.5 Dense, any size | `qwen3_5_bf16`, `qwen3_5_q8` | BF16 / Q8 | MTP |
| Qwen3.5 Dense or MoE, any size | `qwen3_5_published` | Encoded projections and experts as published; quantizes nothing | MTP, DFlash |
| Gemma 4 text | `gemma4_published` (default) | NVFP4 and FP8 projections kept, rest BF16 | Gemma 4 assistant |
| Gemma 4 26B-A4B / 31B | `gemma4_26b_a4b_nvfp4`, `gemma4_31b_nvfp4` | Sites the checkpoint encodes are kept; the rest Q8 | Gemma 4 assistant |
| Gemma 4 assistant | `gemma4_assistant_published` (default), `gemma4_assistant_q8` | BF16 as is / GGUF Q8_0 kept; `_q8` is NInferno's own Q8, only when named | (is a drafter) |
| Gemma 4 DFlash drafter | `gemma4_dflash_published` (default), `gemma4_dflash_q8` | BF16 as is; `_q8` only when named | (a drafter for the `ninferno-gemma4` tool; the server runs Gemma 4 with its assistant) |
| Dense decoders (Llama, Mistral, Granite, OLMo, Phi-3, SmolLM3, Gemma 3, Seed-OSS, Qwen2/Qwen3, ...) | `dense_published`, `dense_encoded`, `dense_bf16`, `dense_q8`, `dense_fp8`, `dense_nvfp4` | `dense_published` quantizes nothing; `dense_encoded` (the default for a quantized checkpoint) keeps its NVFP4 or FP8 projections and takes Q8 for the projections the checkpoint left unquantized; the last three are NInferno's own quantization | DFlash |
| MoE decoders (gpt-oss, Qwen3-MoE, Nemotron-H) | `moe_published`, `moe_q8` | `moe_published` keeps MXFP4 and NVFP4 experts as published | DFlash |
| NemotronH MTP sidecar | `nemotron_h_mtp_published` (default) | GGUF Q4_0/Q8_0 blocks kept word for word, rest as stored | (is a drafter) |
| Block-diffusion drafter (Muse Glimmer assistant, `dflash` GGUF) | `dflash_published` | Quantizes nothing | (is a drafter) |

Notes on the table:

- The "Drafters" column lists the kinds the engine supports. Which ones exist for a given model
  comes from the official registry, `src/spec/data/drafter_registry.json` (see [Drafters](#drafters)).
- Python `tools.convert` always needs an explicit `--recipe`; `ninferno-convert` can choose one. The `qwen3_5_*`, `dense_*`, `moe_*`, `dflash_published` and `nemotron_h_mtp_published` names are listed by `ninferno-convert --help`.

### Qwen3.8-27B NVFP4/FP8 with DFlash2

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_nvfp4.ninferno
```

## Options

These are `ninferno-convert`'s options. The Python converter (`python3 -m tools.convert`) takes
`--model`, `--recipe`, `--source`, `--components`, `--resource`, `--proposal`, `--proposal-rows`,
`--ranking`, `--name`, `--out`, `--device`, `--rows-per-chunk` and `--max-file-bytes`, plus the
ones marked (Python); it refuses every other.

### Input and output

| Option | Default | Meaning |
|---|---|---|
| `--version` | | Print the version and exit. |
| `--model PATH` | required (or `--fetch`) | Checkpoint directory, `.gguf` file or directory with one GGUF. Supplies the main config, default resources and the source `base`. |
| `--fetch SPEC` | none | `org/name[@rev]` or an http(s) URL of one file. |
| `--recipe NAME` | official recipe for the architecture | Official name, or `file.py[:function]` (Python converter). |
| `--out PATH` | with `--fetch`, `<cache>/models--org--name/ninferno/<commit>/<stem>.<recipe>.ninferno` (reused by a rerun); required with `--model` and in Python | Output artifact. Existing files are not overwritten. |
| `--name NAME` | none | Public model name saved in metadata. |
| `--components text[,vision,mtp,dflash,dflash2]` | see below | Components to write. Must not repeat. |
| `--source NAME=PATH` | none | Extra named source, e.g. `quantized`, `dflash`, `dflash2`. Repeatable. `base` is reserved (use `--model`). |
| `--resource ROLE=PATH` | from `--model` | Replace a frontend resource. Repeatable. |
| `--override FILE` | none | (Python) Configuration applied after the recipe and proposal-head setup. |
| `--quiet` | off | Less output. |

Components: the Python converter defaults to `text`. A Qwen3.5-family conversion without
`--components` keeps the text model plus the MTP layers the checkpoint or GGUF carries and every
drafter named by `--source dflash=` / `--source dflash2=` (`--no-drafters` keeps the text alone).
Name `--components` to choose exactly. Include only the optional components you want to distribute.
MTP and Vision use the main source. DFlash and DFlash2 use their named source. An artifact may hold
several optional components; the Engine loads only those selected at startup, with at most one
speculative backend.

### Quantization and layout

| Option | Default | Meaning |
|---|---|---|
| `--maintainer-quantize` | off | Allow recipes that quantize an unquantized checkpoint. |
| `--round-tiny-scales` | off | Keep an integer matrix whose scales are below binary16's normal range (\|s\| < 2^-14) with those scales rounded to binary16. Changes those weights slightly. The report names the count. |
| `--vocab native\|q8` | `native`; `q8` for `dense_q8`, `dense_fp8`, `dense_nvfp4`, `moe_q8` | Token embedding and output head (dense and MoE recipes). See [Vocabulary](#vocabulary). |
| `--drafter-quant q8\|q4\|nvfp4` | none | Quantize a drafter itself. See [Drafters](#drafters). |
| `--layout auto\|published\|frag` | `auto` | Storage of GGUF routed-expert banks. See [Expert layout](#expert-bank-layout). |
| `--relayout auto\|published\|frag` | none | Move an existing artifact between layouts. Needs `--target-artifact`; takes no conversion flag. |
| `--proposal` | off | Add the indexed proposal head for speculative decoding. The ordinary output head is kept. |
| `--proposal-rows N` | `131072` | Rows of the proposal head. |
| `--ranking PATH` | repository token ranking | Token ranking used by the proposal head. |

### Performance and files

| Option | Default | Meaning |
|---|---|---|
| `--device DEV` | `cuda` (Python) | Conversion device of `tools.convert`; `cpu` for CPU. `ninferno-convert` accepts it and ignores the value: it always converts on the CPU. |
| `--rows-per-chunk N` | `512` | (Python) Rows per chunk. `ninferno-convert` records it in the report; its chunking is its own. |
| `--max-file-bytes N` | `32000000000` | Maximum file size, including framing. |
| `--threads N` | cores, at most 16 | Worker threads. |
| `--keep-source-cache` | off | Leave the checkpoint's pages in the operating system's page cache after reading them. |
| `--consume-source` | off | Release the checkpoint's disk space while converting. **The checkpoint is destroyed.** Progress is journaled next to `--out`; rerun the same command to resume. |
| `--consume-window BYTES` | 1 GiB | Window for `--consume-source`. |
| `--assume-free-bytes N` | free space of the disk | Override the free-space figure the checks use (for tests; on a real disk a wrong figure lets a conversion fill it). |

### Fetching (`--fetch`)

| Option | Default | Meaning |
|---|---|---|
| `--hub huggingface\|modelscope` (`hf`, `ms`) | `huggingface` | Hub to use. |
| `--revision REV` | `main` (Hugging Face), `master` (ModelScope) | Revision; `org/name@REV` in the spec overrides it. |
| `--quant LABEL` | NVFP4, MXFP4, Q8_0, BF16, ... by preference | GGUF quantization to download (a case-insensitive part of the file name), or a width (`4bit`, `8-bit`): the largest GGUF model of that width. |
| `--include GLOB` | what the conversion reads | Download only the files matching GLOB, instead of the converter's own choice. Repeatable. |
| `--cache DIR` | `$HF_HUB_CACHE`, `$HF_HOME/hub`, `$XDG_CACHE_HOME/huggingface/hub`, `~/.cache/huggingface/hub` | Cache directory. |
| `--streams N`, `--max-streams N` | `0` (adapt), `8` | Parallel ranges per file. |
| `--min-speed BYTES_PER_S`, `--stall-seconds S`, `--retries N` | 64 KiB/s, `20`, `10` per chunk | A stream slower than the minimum over that many seconds is retried. |
| `--chunk-bytes N` | 32 MiB | Download chunk size. |
| `--no-verify` | off | Skip download verification. |
| `--fetch-only` | off | Download, do not convert. |
| `--no-base-tokenizer` | off | Do not take tokenizer files from a GGUF's base-model repository. |

### Drafter options

| Option | Meaning |
|---|---|
| `--no-drafters` | Do not attach drafters. |
| `--with-drafters` | Attach drafters for a GGUF model and for `--model` too (`--fetch` does it by default). |
| `--model-id REPO` | Model id used to look up drafters in the official registry. |
| `--drafter-artifact NAME=PATH` | Attach an already converted drafter instead of fetching. A `NAME` the model does not declare adds it (e.g. a NemotronH MTP sidecar: `mtp=PATH`). Repeatable. |
| `--plan-drafters (--model DIR \| --target-artifact PATH)` | Print the plan; download nothing. |
| `--attach-drafters --target-artifact PATH [--model-id REPO]` | Add drafters to an existing artifact. Without `--model-id` only the drafters the artifact itself carries or declares, and those given with `--drafter-artifact`, are attached. |

`ninferno-convert` exits with 0 on success, 2 on an argument or conversion error, and 1 when an
attachment has a failed or incompatible drafter (the plan is printed); `--plan-drafters` only
prints the plan and exits with 0.

## Details

### Vocabulary

`--vocab` applies to dense and MoE recipes.

- `native` keeps the vocabulary as the checkpoint publishes it: BF16 stays BF16 next to AWQ/GPTQ
  int4 or FP8 projections; a GGUF's Q8_0 or K-quant table stays that table.
- `q8` quantizes an unquantized vocabulary to `q8_g32_fp16`.
- A vocabulary the publisher already quantized is kept as published either way, an NVFP4 or FP8
  output head included. ModelOpt quantizes `lm_head`: nvidia/Muse-Glimmer-30B-NVFP4 keeps its W4A16
  NVFP4 head of 202048 rows, whose last 128-row scale tile the layout completes.
- The choice is recorded as `provenance.vocabulary` in `<out>.conversion.json`.

### Published integer quantization

A published integer quantization (compressed-tensors, GPTQ, AWQ) is kept as published when every
scale is exact in binary16. Scales below binary16's normal range (|s| < 2^-14, a few in some 3-bit
checkpoints) are not exact; `--round-tiny-scales` handles them as described above.

### GGUF input

A GGUF converts the same way. Its tensors and metadata are read back into the Hugging Face config and
tensor names that llama.cpp's converter started from.

Supported GGUF architectures:

| GGUF architecture | Models and notes |
|---|---|
| `qwen35`, `qwen35moe` | Qwen3.5 family. |
| `gemma4` | Gemma 4. |
| `llama` | Llama, SmolLM2, TinyLlama, Mistral 7B. |
| `qwen2`, `qwen3`, `smollm3` | Dense decoders. |
| `granite` | Dense Granite 3.x/4.x. |
| `olmo2` | OLMo 2; OLMo 3 when the file declares a sliding window. |
| `mistral3` | Ministral 3. |
| `phi3` | Phi-3.x, Phi-4-mini. Fused `attn_qkv` and `ffn_up` are split by rows. |
| `muse-glimmer` | Muse Glimmer. The 1 + w norms, Q/K permutation and the `qk_scale_factor` held in `attn_q_norm` are read back. |
| `dflash` | llama.cpp's DFlash drafter (the Muse Glimmer drafter of the official GGUF): no table and no head; `target_layers` read one past as `target_layer_ids`; block mask bidirectional on every layer. A file with DFlash2's convolutions and candidate selector is read as DFlash2. |
| `gpt-oss` | gpt-oss-20b/120b as ggml-org publishes them. MXFP4 experts are kept as mxfp4 (each block's nibbles put back in natural order; gate and up banks interleaved again as transformers stores them). The Q8_0 attention, embedding and output head are kept as `q8_g32_fp16`. The sliding window on even layers, `swiglu_limit` 7 and the unrounded YaRN range come from the architecture, not the file. |

Formats that run as published: Q8_0; the K-quants (Q2_K to Q6_K); the 32-value blocks Q4_0, Q4_1,
Q5_0, Q5_1 and IQ4_NL; the i-quants IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M;
the ternary TQ1_0 and TQ2_0; Q1_0 and Q2_0.

A GGUF without tokenizer files next to it gets them rebuilt from `tokenizer.ggml`:

- byte-level BPE for the pre-tokenizers `qwen35`, `qwen2`, `llama-bpe`, `dbrx`, `smaug-bpe`,
  `tekken`, `pixtral` (the Llama 3 regex; tekken's for Nemotron 3), `gpt-4o`, `llama4`, `smollm`
  and `refact`;
- the Gemma 4 vocabulary;
- SentencePiece BPE, with merges taken from the piece scores. A file whose scores are all equal no
  longer holds the merge order; it needs the Hugging Face tokenizer files.

With `--fetch`, the tokenizer files come from the GGUF's base-model repository when it names one,
else are rebuilt from its metadata.

### Expert bank layout

`--layout` sets how a GGUF's routed-expert banks are stored. The bytes are the published ones either
way: `ggml_frag_v1` only reorders each row's code bytes into the order the grouped MoE kernel reads
them ([storage layouts](maintainer/storage-layouts.md), Section 7).

| Value | Behavior |
|---|---|
| `auto` (default) | Uses `ggml_frag_v1` where the kernel measured not slower: Q4_0 banks with K >= 1024 (Nemotron 3.5 Lightning's up and down, Gemma 4 26B's gate/up; its K = 704 down stays published), Q4_K, IQ3_XXS and IQ2_S banks. |
| `published` | Keeps every bank published (for A/B runs). |
| `frag` | Uses `ggml_frag_v1` wherever a format has it. |

The conversion log says how many banks were stored that way.

Move an existing artifact between layouts without its checkpoint:

```bash
./build/apps/ninferno-convert --relayout auto --target-artifact model.ninferno              # in place
./build/apps/ninferno-convert --relayout published --target-artifact model.ninferno --out ab/model.ninferno
```

- In place, only the permuted code planes and the directory's layout names are rewritten, under an
  undo journal next to the artifact (`model.ninferno.relayout-journal`, at most 64 MiB plus the
  directory). After an interruption, rerun the same command: it restores the chunk it was writing
  and finishes. Until then the engines refuse to load the artifact.
- Stop any server that has the artifact open first. It maps the payload, and an in-place relayout
  changes the bytes under it. A lock keeps only two relayouts apart.
- A directory whose reserve cannot hold the longer published layout names is refused before
  anything is written; use `--out` instead.
- `--out` writes a new artifact (a new artifact id) and needs the artifact's size in free space.
  The source is only read. Its `.speculation.json` and `.spec_profile.json` are carried over with
  the new id.
- `<artifact>.conversion.json` records each relayout under `relayouts`.
- `--relayout` accepts only `--target-artifact`, `--out`, `--max-file-bytes` and `--quiet`.

### Drafters

Speculative drafters come with the model. `--fetch` attaches a checkpoint's drafters
(`--no-drafters` skips them; `--with-drafters` does it for a GGUF model and for `--model`). In
order, they are:

1. the MTP layers inside the checkpoint (kept as the `mtp` component);
2. the drafters the checkpoint declares in `ninferno_speculation.json`;
3. else those the official registry names for the model id, or for a base model its model card
   declares. There is never a search.

Each is checked against the target (vocabulary, geometry, taps) before it is fetched and converted.

- A separate drafter (Gemma 4 assistant, DFlash) becomes its own artifact.
- A companion (Qwen DFlash/DFlash2) becomes a component of the target.
- The drafters and the published speculation profiles go to `<target>.ninferno.speculation.json`,
  which the engines read at start.
- `--drafter-quant` quantizes a drafter itself. Every projection its recipe leaves in BF16 becomes
  `q8_g32_fp16` (`q8`), `q4_g64_fp16` (`q4`) or weight-only NVFP4 (`nvfp4`; q8 where a matrix is
  not whole 128-row tiles). The target verifies every drafted token, so the output is unchanged;
  only the speed differs. A published encoding is kept, never quantized again. Of a target, only
  its companions (DFlash, DFlash2) are quantized, and a target without one is refused. A
  companion the registry declares with a `quant` is converted so unless the flag says otherwise.

A block-diffusion drafter converts on its own and is paired with its target at load time: Meta's
Muse Glimmer assistant (`MuseGlimmerAssistantModel`, BF16) or the `dflash` GGUF, with the
`dflash_published` recipe, which quantizes nothing.

- The assistant's repository has no tokenizer; pass the target's with `--resource tokenizer.json=...`
  (and its other tokenizer files).
- Neither names the target's depth, and the assistant names no vocabulary. The loader takes both from
  the target and still checks the width, the tapped layers and the mask token against it.

```bash
./build/apps/ninferno-convert --model meta-models/Muse-Glimmer-30B-assistant --recipe dflash_published \
  --resource tokenizer.json=/path/to/Muse-Glimmer-30B/tokenizer.json \
  --resource tokenizer_config.json=/path/to/Muse-Glimmer-30B/tokenizer_config.json \
  --out models/muse30b_assistant.ninferno
./build/apps/ninferno-dense models/muse30b_nvfp4.ninferno spec --dflash models/muse30b_assistant.ninferno \
  --tokens prompt.i32 --decode 256
```

## Custom recipes

### Change part of a recipe

Save as `my_recipe.py`:

```python
from tools.convert.official_recipes import qwen3_6_27b


def configure(model, recipe, sources):
    qwen3_6_27b(model, recipe, sources)
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method="grouped_absmax",
    )
```

Run it with the same source and component selection:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe my_recipe.py \
  --components text,vision,mtp \
  --proposal \
  --out models/my_qwen.ninferno
```

- The default entry function is `configure`; `--recipe my_recipe.py:customize` selects another.
- Alternatively, use an official `--recipe` and put only the changes in an `--override` file.
  Overrides run after the base recipe and the optional proposal-head setup.
- `model.parameters` maps logical names to shape, source and mathematical inputs. To list the names
  for your components:

```python
for name, parameter in model.parameters.items():
    print(name, parameter.shape, parameter.inputs)
```

`recipe.assign` takes one name, a list, or shell-style patterns such as `text/layers/*/mlp/down`. A
selector that matches nothing fails. Assignments run in Python order; later ones replace only the
choices they supply. Changing `format` does not change `method`. Use `layout="auto"` to select the
registered layout for the format when overriding an earlier explicit layout.

| Argument | Meaning |
|---|---|
| `format` | Persistent numeric format |
| `layout` | Physical encoding; inferred from format unless explicitly set |
| `method` | Built-in method name or Python callable |
| `source` | Logical values or encoded rows from the selected source |
| `parameters` | JSON-serializable numerical parameters passed to the method |
| `rows=(begin, end)` | Override complete leading-axis rows in a half-open range |
| `activation_policy` | Permission for activation precision at the parameter's mathematical inputs |

`rows=(0, 128)` can give the first 128 rows another format. That creates several physical parts when
needed. The container can represent it, but the Op must also support consuming those parts. Current
native projections generally need a contiguous parent region, so arbitrary splits of one projection
are not automatically executable.

### Formats and methods

| Format | Built-in method for floating-point input | Import of already encoded input |
|---|---|---|
| `bf16`, `fp32`, `int32` | `cast_direct` | Direct words through the source reader |
| `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | `grouped_absmax` | Supply a custom method/source if needed |
| `fp8_e4m3fn_row_bf16` | `fp8_row_maxabs` | `import_encoded` |
| `nvfp4` | Supply a custom quantizer | `import_encoded` |

- `grouped_absmax` stores one FP16 scale per group and signed integer codes.
- `fp8_row_maxabs` rounds input values to BF16, then produces E4M3FN codes and one BF16 multiplier
  per row.
- `import_encoded` preserves compatible code and scale words, including NVFP4's matrix weight
  divisor. It does not dequantize and requantize.

Exact numeric and packing rules: [numeric formats](maintainer/tensor-formats.md) and
[storage layouts](maintainer/storage-layouts.md). Source format names alone do not establish
compatibility: scale direction, granularity, code meaning and axis order must also match.

### Activation precision

Activation permissions are independent of the stored weight format.

| Policy | Permitted activation paths |
|---|---|
| `A16Only` | A16 |
| `AllowA8` | A16, A8 |
| `AllowA4` | A16, A8, A4 |

- They permit choices; they do not force the lowest precision.
- A fused operation that shares one activation across several projections respects the intersection
  of their permissions.
- `recipe.use(parameter, input_name, ...)` sets one mathematical input independently; names are in
  `parameter.inputs`.
- An NVFP4 A4 input needs a positive finite activation divisor. `import_encoded` takes it from the
  selected source, or a recipe supplies it with
  `recipe.use(..., auxiliaries={"activation_input_divisor": value})`. Shared weights keep separate
  Use records; sharing weights does not share calibration.

A kept block encoding whose every 32 values are int8-exact codes times a scale (and a min; Q6_K,
Q3_K, Q2_K, IQ2_XS and IQ2_S: per 16 values) records `AllowA8` (`kept_block_policy`):

- GGUF Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K, Q3_K, Q2_K and Q8_0;
- the i-quants IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS and IQ2_S (signed grid values) kept as published;
- u4 with zero points per 128, and MXFP4 (doubled E2M1 codes).

The output head keeps the recipe's policy. The W4A8 route computes these on int8 tensor cores
against activations quantized to int8 with one scale per 32 values, on calls of 17 tokens or more.
The R8 route (dp4a) serves decode widths for the formats that have one. `--activations exact` keeps
BF16 activations for them. Other kept encodings keep the recipe's policy or record their own (FP8:
`AllowA8`; NVFP4 with a divisor: `AllowA4`). A kept NVFP4 or FP8 output head keeps BF16 activations
(`A16Only`), so its logits take one route at every token count; a kept integer head keeps its own
policy.

### Fused parents and logical projections

- The Qwen adapter exposes Q, K, gate and V separately, even from fused source tensors. It also
  supplies finite packing groups for attention, GDN, MLP and MoE. Compatible selections are packed
  into a shared parent automatically by the built-in methods.
- For the Dense groupwise recipe, attention Q/K form one Q4 parent and gate/V one Q5 parent. The
  native fused Op receives two weights. To use the supported single-parent FP8 form, an override can
  assign all four projections together:

```python
def configure(model, recipe, sources):
    for layer, kind in enumerate(model.config["layer_types"]):
        if kind != "full_attention":
            continue
        prefix = f"text/layers/{layer}/attention/"
        recipe.assign(
            [prefix + role for role in ("query", "key", "gate", "value")],
            format="fp8_e4m3fn_row_bf16",
            layout="auto",
            method="fp8_row_maxabs",
            activation_policy="AllowA8",
        )
```

Use it as `--override` after `qwen3_6_27b` or `qwen3_8_27b`; other parameters stay under that
recipe. The adapter handles source Q/gate row order. Changing representation can change numerical
results and the physical kernels.

Explicit organization:

- `recipe.group([names...])` concatenates compatible selections in the given order.
- `recipe.separate(names)` disables automatic grouping for those parameters.
- `recipe.share(parameter, target)` binds equal-shaped parameters to the same physical data.
- Explicit groups must be disjoint and use unsplit selections with matching format, layout, method
  and method parameters.
- NVFP4 parents also need a common weight divisor. The exception is an explicit group of
  equal-shaped members, which keeps one divisor per member when each member is whole 128-row scale
  tiles. The native converter refuses a published expert bank whose NVFP4 experts end inside a
  scale tile and differ in divisor, and names the bank.
- Automatic grouping is limited to built-in methods; custom methods can request explicit groups.

Grouping chooses storage. Model execution code chooses the supported fused implementation. The
loader uploads the stored representation and does not repack it.

### Read another source

`--model` supplies the main config, default resources and the source `base`. Add other Safetensors
sources with repeated `--source NAME=PATH`; they are opened when used. Single-file Safetensors and
indexed shards work. Extra tensor-only sources can omit the model config; sources with a config are
checked against the model geometry.

Replace one logical parameter from another compatible checkpoint (pass
`--source alternate=/path/to/alternate-checkpoint`):

```python
name = "text/layers/0/mlp/down"
recipe.assign(
    name,
    source=model.source(name, sources["alternate"]),
    format="q6_g64_fp16",
    method="grouped_absmax",
)
```

`model.source` applies the architecture's source-name and axis mapping, including Q/gate extraction.
The built-in compressed-tensors reader understands the implemented per-row FP8 and NVFP4 code/scale
conventions. It can expose decoded values for another quantizer or encoded rows for exact import.

For another file format or quantization convention, provide a `LogicalSource`. Its value reader takes
flat C-order element bounds and returns exactly that range. Example: read a logical matrix from a
NumPy file beside the recipe (use it as an override):

```python
from pathlib import Path
import numpy as np
import torch
from tools.convert.sources.logical import LogicalSource


def configure(model, recipe, sources):
    name = "text/layers/0/mlp/down"
    path = Path(__file__).with_name("mlp-down.npy")
    data = np.load(path, mmap_mode="r")
    if tuple(data.shape) != model.parameters[name].shape or not data.flags.c_contiguous:
        raise ValueError("mlp-down.npy must have the logical shape and C-order storage")

    def read_values(begin, end):
        values = data.reshape(-1)[begin:end].astype(np.float32, copy=True)
        return torch.from_numpy(values)

    source = LogicalSource(tuple(data.shape), str(path), read_values)
    recipe.assign(name, source=source, format="q6_g64_fp16", method="grouped_absmax")
```

The NumPy file must already follow the logical row/column order. For an unfamiliar quantized source,
its reader does the decoding before returning values. To preserve existing compatible encoded words,
also provide `read_encoded` returning `EncodedRows`, and the format's required divisor accessors.
See [`sources/logical.py`](../tools/convert/sources/logical.py).

### Write a conversion method

A method receives a `PrepareRequest` and returns `request.job(produce=...)`. Preparation validates the
target and determines auxiliary values. `produce` reads bounded source regions and writes values or
codes/scales through `TensorOutput`; the writer owns placement and file I/O.

This example adds explicit clipping before the grouped quantizer. The threshold is the recipe
author's numerical choice. Use it as an override.

```python
import math
import torch
from tools.artifact.formats import QuantFormat, get_format
from tools.convert.quantization.groupwise import quantize_matrix


def clipped_grouped(request):
    if len(request.target.shape) != 2 or not isinstance(
        get_format(request.target.format), QuantFormat
    ):
        raise ValueError("clipped_grouped requires a grouped-integer matrix")
    limit = float(request.parameters["clip"])
    if not math.isfinite(limit) or limit <= 0 or request.rows_per_chunk <= 0:
        raise ValueError("clip and rows_per_chunk must be positive")
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("source contains non-finite values")
            encoded = quantize_matrix(
                values.clamp(-limit, limit),
                request.target.format,
                device=request.device,
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def configure(model, recipe, sources):
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method=clipped_grouped,
        parameters={"clip": 1.0},
    )
```

- `request.values` traverses the prepared logical inputs in parent order, including explicit groups.
- `output.write_codes` does the registered packing and validates codes/scales. Do not duplicate that
  byte-layout logic.
- Direct output uses `output.write_values`.
- Keep source blocks and temporary device tensors bounded to the method's working set.

## Resources, files and inspection

Text resources: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`. Vision adds its image and video processor configs. They come from
`--model`; `--resource ROLE=PATH` replaces one, e.g.
`--resource chat_template.jinja=/path/to/chat_template.jinja`.

Maintained templates used by the official examples:

| Model | Template | Defaults |
|---|---|---|
| Qwen3.6 Dense/MoE | [qwen3_6.jinja](../tools/chat_templates/qwen3_6.jinja) | thinking on; closed-turn reasoning omitted |
| Qwen3.8 | [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja) | thinking on; effort `xhigh`; closed-turn reasoning retained |

- Use your own Jinja file to change the artifact's default template. A startup
  [`--chat-template FILE`](cli.md#thinking-controls) overrides the stored template.
- `generation_config.json` is preserved. Sampling presets stay determined by the architecture and
  explicit application/request settings.

Files:

- The default maximum file size is 32,000,000,000 bytes, including framing. Smaller artifacts stay
  one file. Larger ones use an entry such as `models/my_qwen.ninferno` plus
  `my_qwen.ninferno.part-0001`, `my_qwen.ninferno.part-0002`, and so on in the same directory.
  Pass only the entry path to NInferno and keep the parts together. `--max-file-bytes` changes the
  limit.
- Conversion writes `models/my_qwen.ninferno.conversion.json` beside the artifact: sources, methods,
  formats, component configs, files and timing. It helps reproduce a recipe; the Engine reads the
  artifact itself. Existing output files are not overwritten.

Inspect an artifact without running inference:

```bash
python3 -m tools.artifact.inspect models/my_qwen.ninferno --objects --bindings
python3 -m tools.artifact.inspect models/my_qwen.ninferno --json
```

Actual Op support is checked by consumers during preparation, resource queries, warmup or execution.
A valid file may need more Op support before your chosen combination runs. Test the phases and
optional components you plan to use through the [CLI](cli.md) or [serving](serving.md).
