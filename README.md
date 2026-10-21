# NInferno

**Fast LLM inference on the NVIDIA GPU you already own.**

A well-tuned launch can run a model more than twice as fast as a plain one
([measured](docs/performance/README.md)). NInferno comes
already measured on nearly every GPU you can buy: it knows the best settings for each
architecture and model and starts with them, and its GUI walks you through fitting them to your
task. The aim: fast, cheap inference for everyone.

<!-- perf-summary:begin -->
<!-- perf-summary:end -->

- **GPUs:** NVIDIA Volta, Turing, Ampere, Ada and Blackwell: GeForce RTX 20 to 50, V100, T4, A10,
  A100, L4, L40S, RTX PRO 6000.
- **Not the target:** ✗ H100, H200, B200, B300. The project is fast inference on affordable cards;
  these run, but get no tuning and no measurements.
- **Models:** Qwen3.5 to 3.8, Gemma 4, gpt-oss, Nemotron 3.5 and Muse Glimmer, **and every
  fine-tune, fork and version built on their architectures** (support is per architecture), as
  published on Hugging Face: GGUF, NVFP4, MXFP4, FP8, AWQ, GPTQ.
- **Speed:** speculative decoding (MTP, DFlash, DFlash2) and kernels measured per GPU, one to eight
  concurrent streams.
- **Long context:** YaRN and LongRoPE as the checkpoint defines them (Qwen3.5/3.6 stretch their window
  with `--rope-yarn-factor`), KV cache in FP8 (NVFP4 for Qwen).
- **Prompt cache:** a repeated prefix (system prompt, chat history, a document) is not computed
  again: its KV state stays in GPU memory, then in pinned host RAM. Never on an SSD†.
- **API:** OpenAI (`/v1/chat/completions`, `/v1/responses`) and Anthropic (`/v1/messages`) on one
  server, with tool calls, structured output, vision and PDF documents.
- **Where it goes:** the [roadmap](#roadmap).

<sub>† A KV cache writes the drive all the time: under load it wears out a consumer SSD in one to
two months, an enterprise one in six to eight. NInferno will not offer one.</sub>

## Contents

- [Supported models](#supported-models)
- [Performance](#performance)
- [Quick start](#quick-start)
- [How it works](#how-it-works) · [Limits](#limits)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Supported models

| Family | Models | Speculative decoding | Input |
|---|---|---|---|
| Qwen3.5 | Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B | MTP | text, image, video |
| Qwen3.6 | Qwen3.6-27B, Qwen3.6-35B-A3B (MoE) | MTP, DFlash | text, image, video |
| Qwen3.8 | Qwen3.8-27B | MTP, DFlash2 | text, image, video |
| Gemma 4 | Gemma 4 12B, Gemma 4 26B-A4B (MoE), Gemma 4 31B | Gemma 4 assistant | text, image |
| gpt-oss | gpt-oss-20b (MoE) | DFlash | text |
| Nemotron 3.5 | Nemotron 3.5 Lightning 30B-A3B (MoE) | MTP | text |
| Muse Glimmer | Muse Glimmer 30B | DFlash | text |

- **Weights:** each model runs as published on Hugging Face (GGUF, NVFP4, MXFP4, FP8, AWQ,
  GPTQ), and so do its fine-tunes. A quantization the engine cannot read is refused at start,
  with the reason. [Weight conversion](docs/weight-conversion.md) lists each family's formats.
- **Image and video:** start with `--vision`, for models that have a vision tower (Gemma 4 12B has
  none); video is Qwen's. A GGUF needs its `mmproj` file beside it (converted with the model).
- **Speculative decoding:** the full model checks every drafted token. `--activations exact`
  makes the output identical whatever the load, batch or draft, at some speed
  ([quality checks](docs/perplexity.md)).
- **KV cache:** BF16 or FP8 for every family; Qwen also INT8, NVFP4 and K8V4 (8-bit keys, 4-bit
  values).

### By GPU generation

| | Blackwell<br>RTX 50, PRO 6000 | Ada<br>RTX 40, L4, L40S | Ampere<br>RTX 30, A10, A100 | Turing<br>RTX 20, T4 | Volta<br>V100 |
|---|---|---|---|---|---|
| GGUF, MXFP4, AWQ, GPTQ | ✓ | ✓ | ✓ | ✓ | ✓ |
| NVFP4 | ✓ native | slower path¹ | slower path¹ | slower path¹ | slower path¹ |
| FP8 weights | ✓ | ✓ | ✓ 16-bit math | ✓ 16-bit math | ✓ 16-bit math |
| BF16 weights | ✓ | ✓ | ✓ | ✓ via FP16² | ✓ via FP32 |
| FP8 KV cache | ✓ | ✓ | ✓ emulated | ✓ emulated | ✓ emulated |
| Driver | 580+ | 580+ | 580+ | 580+ | 570+ (CUDA 12.8 build) |

¹ Runs on generic kernels, not yet measured; the model's GGUF is the measured choice.
² FP16 range: values under 2⁻²⁴ round to zero.

## Performance

<!-- perf:begin -->
Tokens per second, best settings: **one stream / all streams**, superscript = concurrent streams. Click a GPU, model or number for details ([all GPUs and models](docs/performance/README.md)).

| GPU | [Qwen3.8<br>27B](docs/performance/models/qwen3.8-27b.md "Qwen3.5 dense, 27B") | [Qwen3.6<br>35B-A3B](docs/performance/models/qwen3.6-35b-a3b.md "Qwen3.5 MoE, 35B total, 3B active") | [Gemma 4<br>31B](docs/performance/models/gemma-4-31b-it.md "Gemma 4 dense, 31B") | [Gemma 4<br>26B-A4B](docs/performance/models/gemma-4-26b-a4b-it.md "Gemma 4 MoE, 26B total, 4B active") | [gpt-oss<br>20b](docs/performance/models/gpt-oss-20b.md "MoE decoder, 21B total, 3.6B active") | [Muse Glimmer<br>30B](docs/performance/models/muse-glimmer-30b.md "dense decoder, 30B") | [Nemotron 3.5<br>30B-A3B](docs/performance/models/nemotron-3.5-lightning-30b-a3b.md "Nemotron-H hybrid MoE, 30B total, 3B active") |
|---|---:|---:|---:|---:|---:|---:|---:|
| [**RTX&nbsp;5090**](docs/performance/gpus/rtx5090.md "RTX 5090 32 GB (GB202, sm_120)") | [193](docs/performance/runs/rtx5090/qwen3.8-27b.md "one stream: 193 tok/s, UD-Q4_K_XL, dflash2 ×7, measured 2026-10-19") / [557<sup>8</sup>](docs/performance/runs/rtx5090/qwen3.8-27b.md "all streams: 557 tok/s at 8 streams, UD-Q4_K_XL, dflash2 ×7, measured 2026-10-19") | [588](docs/performance/runs/rtx5090/qwen3.6-35b-a3b.md "one stream: 588 tok/s, UD-Q4_K_M, dflash ×7, measured 2026-10-19") / [1,526<sup>8</sup>](docs/performance/runs/rtx5090/qwen3.6-35b-a3b.md "all streams: 1,526 tok/s at 8 streams, UD-Q4_K_M, dflash ×7, measured 2026-10-19") | [151](docs/performance/runs/rtx5090/gemma-4-31b-it.md "one stream: 151 tok/s, UD-Q4_K_XL, mtp ×10, measured 2026-10-19") / [416<sup>8</sup>](docs/performance/runs/rtx5090/gemma-4-31b-it.md "all streams: 416 tok/s at 8 streams, UD-Q4_K_XL, mtp ×10, measured 2026-10-19") | [464](docs/performance/runs/rtx5090/gemma-4-26b-a4b-it.md "one stream: 464 tok/s, UD-Q4_K_XL, mtp ×6, measured 2026-10-19") / [1,349<sup>8</sup>](docs/performance/runs/rtx5090/gemma-4-26b-a4b-it.md "all streams: 1,349 tok/s at 8 streams, UD-Q4_K_XL, mtp ×6, measured 2026-10-19") | [409](docs/performance/runs/rtx5090/gpt-oss-20b.md "one stream: 409 tok/s, gpt-oss-20b, dflash ×7, measured 2026-10-20") / [1,281<sup>8</sup>](docs/performance/runs/rtx5090/gpt-oss-20b.md "all streams: 1,281 tok/s at 8 streams, gpt-oss-20b, dflash ×7, measured 2026-10-20") | [193](docs/performance/runs/rtx5090/muse-glimmer-30b.md "one stream: 193 tok/s, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") / [668<sup>8</sup>](docs/performance/runs/rtx5090/muse-glimmer-30b.md "all streams: 668 tok/s at 8 streams, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") | [488](docs/performance/runs/rtx5090/nemotron-3.5-lightning-30b-a3b.md "one stream: 488 tok/s, UD-Q4_K_XL, mtp ×3, measured 2026-10-21") / [1,077<sup>8</sup>](docs/performance/runs/rtx5090/nemotron-3.5-lightning-30b-a3b.md "all streams: 1,077 tok/s at 8 streams, UD-Q4_K_XL, mtp ×3, measured 2026-10-21") |
| [**RTX&nbsp;5070&nbsp;Ti**](docs/performance/gpus/rtx5070ti.md "RTX 5070 Ti 16 GB (GB203, sm_120)") | [120](docs/performance/runs/rtx5070ti/qwen3.8-27b.md "one stream: 120 tok/s, UD-IQ3_XXS, dflash2 ×7, measured 2026-10-19") / [261<sup>8</sup>](docs/performance/runs/rtx5070ti/qwen3.8-27b.md "all streams: 261 tok/s at 8 streams, UD-IQ3_XXS, no speculation, measured 2026-10-19") | [365](docs/performance/runs/rtx5070ti/qwen3.6-35b-a3b.md "one stream: 365 tok/s, UD-IQ3_XXS, dflash ×7, measured 2026-10-19") / [755<sup>8</sup>](docs/performance/runs/rtx5070ti/qwen3.6-35b-a3b.md "all streams: 755 tok/s at 8 streams, UD-IQ3_XXS, no speculation, measured 2026-10-19") |  | [301](docs/performance/runs/rtx5070ti/gemma-4-26b-a4b-it.md "one stream: 301 tok/s, UD-IQ3_XXS, mtp ×6, measured 2026-10-19") / [638<sup>4</sup>](docs/performance/runs/rtx5070ti/gemma-4-26b-a4b-it.md "all streams: 638 tok/s at 4 streams, UD-IQ3_XXS, mtp ×6, measured 2026-10-19") | [225](docs/performance/runs/rtx5070ti/gpt-oss-20b.md "one stream: 225 tok/s, gpt-oss-20b, dflash ×6, measured 2026-10-19") / [568<sup>8</sup>](docs/performance/runs/rtx5070ti/gpt-oss-20b.md "all streams: 568 tok/s at 8 streams, gpt-oss-20b, no speculation, measured 2026-10-19") | [101](docs/performance/runs/rtx5070ti/muse-glimmer-30b.md "one stream: 101 tok/s, UD-IQ3_XXS, dflash ×3, measured 2026-10-20") / [268<sup>8</sup>](docs/performance/runs/rtx5070ti/muse-glimmer-30b.md "all streams: 268 tok/s at 8 streams, UD-IQ3_XXS, no speculation, measured 2026-10-20") |  |
| [**RTX&nbsp;4090**](docs/performance/gpus/rtx4090.md "RTX 4090 24 GB (AD102, sm_89)") | [122](docs/performance/runs/rtx4090/qwen3.8-27b.md "one stream: 122 tok/s, UD-Q4_K_XL, dflash2 ×7, measured 2026-10-19") / [366<sup>8</sup>](docs/performance/runs/rtx4090/qwen3.8-27b.md "all streams: 366 tok/s at 8 streams, UD-Q4_K_XL, mtp ×3, measured 2026-10-19") | [398](docs/performance/runs/rtx4090/qwen3.6-35b-a3b.md "one stream: 398 tok/s, UD-Q4_K_M, dflash ×7, measured 2026-10-19") / [843<sup>4</sup>](docs/performance/runs/rtx4090/qwen3.6-35b-a3b.md "all streams: 843 tok/s at 4 streams, UD-Q4_K_M, dflash ×7, measured 2026-10-19") | [114](docs/performance/runs/rtx4090/gemma-4-31b-it.md "one stream: 114 tok/s, UD-Q4_K_XL, mtp ×10, measured 2026-10-19") / [143<sup>4</sup>](docs/performance/runs/rtx4090/gemma-4-31b-it.md "all streams: 143 tok/s at 4 streams, UD-Q4_K_XL, no speculation, measured 2026-10-19") | [321](docs/performance/runs/rtx4090/gemma-4-26b-a4b-it.md "one stream: 321 tok/s, UD-Q4_K_XL, mtp ×5, measured 2026-10-19") / [892<sup>8</sup>](docs/performance/runs/rtx4090/gemma-4-26b-a4b-it.md "all streams: 892 tok/s at 8 streams, UD-Q4_K_XL, mtp ×5, measured 2026-10-19") | [261](docs/performance/runs/rtx4090/gpt-oss-20b.md "one stream: 261 tok/s, gpt-oss-20b, dflash ×7, measured 2026-10-19") / [865<sup>8</sup>](docs/performance/runs/rtx4090/gpt-oss-20b.md "all streams: 865 tok/s at 8 streams, gpt-oss-20b, dflash ×7, measured 2026-10-19") | [123](docs/performance/runs/rtx4090/muse-glimmer-30b.md "one stream: 123 tok/s, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") / [402<sup>8</sup>](docs/performance/runs/rtx4090/muse-glimmer-30b.md "all streams: 402 tok/s at 8 streams, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") | [365](docs/performance/runs/rtx4090/nemotron-3.5-lightning-30b-a3b.md "one stream: 365 tok/s, UD-IQ3_XXS, mtp ×3, measured 2026-10-20") / [905<sup>8</sup>](docs/performance/runs/rtx4090/nemotron-3.5-lightning-30b-a3b.md "all streams: 905 tok/s at 8 streams, UD-IQ3_XXS, mtp ×3, measured 2026-10-20") |
| [**RTX&nbsp;4070&nbsp;Ti**](docs/performance/gpus/rtx4070ti.md "RTX 4070 Ti 12 GB (AD104, sm_89)") | [104](docs/performance/runs/rtx4070ti/qwen3.8-27b.md "one stream: 104 tok/s, UD-IQ2_XXS, dflash2 ×7, measured 2026-10-19") / [216<sup>8</sup>](docs/performance/runs/rtx4070ti/qwen3.8-27b.md "all streams: 216 tok/s at 8 streams, UD-IQ2_XXS, no speculation, measured 2026-10-19") |  |  |  |  |  |  |
| [**L4**](docs/performance/gpus/l4.md "L4 24 GB (AD104, sm_89)") | [32.8](docs/performance/runs/l4/qwen3.8-27b.md "one stream: 32.8 tok/s, UD-Q4_K_XL, dflash2 ×7, measured 2026-10-19") / [67.1<sup>8</sup>](docs/performance/runs/l4/qwen3.8-27b.md "all streams: 67.1 tok/s at 8 streams, UD-Q4_K_XL, no speculation, measured 2026-10-19") | [114](docs/performance/runs/l4/qwen3.6-35b-a3b.md "one stream: 114 tok/s, UD-Q4_K_M, dflash ×11, measured 2026-10-19") / [247<sup>8</sup>](docs/performance/runs/l4/qwen3.6-35b-a3b.md "all streams: 247 tok/s at 8 streams, UD-Q4_K_M, no speculation, measured 2026-10-19") | [33.0](docs/performance/runs/l4/gemma-4-31b-it.md "one stream: 33.0 tok/s, UD-Q4_K_XL, mtp ×10, measured 2026-10-20") / – | [92.9](docs/performance/runs/l4/gemma-4-26b-a4b-it.md "one stream: 92.9 tok/s, UD-Q4_K_XL, mtp ×5, measured 2026-10-20") / [205<sup>4</sup>](docs/performance/runs/l4/gemma-4-26b-a4b-it.md "all streams: 205 tok/s at 4 streams, UD-Q4_K_XL, mtp ×5, measured 2026-10-20") | [70.8](docs/performance/runs/l4/gpt-oss-20b.md "one stream: 70.8 tok/s, gpt-oss-20b, dflash ×7, measured 2026-10-20") / [174<sup>8</sup>](docs/performance/runs/l4/gpt-oss-20b.md "all streams: 174 tok/s at 8 streams, gpt-oss-20b, no speculation, measured 2026-10-20") | [34.3](docs/performance/runs/l4/muse-glimmer-30b.md "one stream: 34.3 tok/s, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") / [87.8<sup>8</sup>](docs/performance/runs/l4/muse-glimmer-30b.md "all streams: 87.8 tok/s at 8 streams, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") | [79.3](docs/performance/runs/l4/nemotron-3.5-lightning-30b-a3b.md "one stream: 79.3 tok/s, UD-IQ3_XXS, no speculation, measured 2026-10-20") / [197<sup>8</sup>](docs/performance/runs/l4/nemotron-3.5-lightning-30b-a3b.md "all streams: 197 tok/s at 8 streams, UD-IQ3_XXS, no speculation, measured 2026-10-20") |
| [**RTX&nbsp;4060&nbsp;Ti**](docs/performance/gpus/rtx4060ti.md "RTX 4060 Ti 16 GB (AD106, sm_89)") | [49.0](docs/performance/runs/rtx4060ti/qwen3.8-27b.md "one stream: 49.0 tok/s, UD-IQ3_XXS, dflash2 ×7, measured 2026-10-19") / [115<sup>8</sup>](docs/performance/runs/rtx4060ti/qwen3.8-27b.md "all streams: 115 tok/s at 8 streams, UD-IQ3_XXS, no speculation, measured 2026-10-19") | [155](docs/performance/runs/rtx4060ti/qwen3.6-35b-a3b.md "one stream: 155 tok/s, UD-IQ3_XXS, dflash ×11, measured 2026-10-19") / [360<sup>8</sup>](docs/performance/runs/rtx4060ti/qwen3.6-35b-a3b.md "all streams: 360 tok/s at 8 streams, UD-IQ3_XXS, no speculation, measured 2026-10-19") |  | [127](docs/performance/runs/rtx4060ti/gemma-4-26b-a4b-it.md "one stream: 127 tok/s, UD-IQ3_XXS, mtp ×5, measured 2026-10-19") / [294<sup>4</sup>](docs/performance/runs/rtx4060ti/gemma-4-26b-a4b-it.md "all streams: 294 tok/s at 4 streams, UD-IQ3_XXS, mtp ×5, measured 2026-10-19") | [86.6](docs/performance/runs/rtx4060ti/gpt-oss-20b.md "one stream: 86.6 tok/s, gpt-oss-20b, dflash ×7, measured 2026-10-19") / [208<sup>8</sup>](docs/performance/runs/rtx4060ti/gpt-oss-20b.md "all streams: 208 tok/s at 8 streams, gpt-oss-20b, no speculation, measured 2026-10-19") | [45.9](docs/performance/runs/rtx4060ti/muse-glimmer-30b.md "one stream: 45.9 tok/s, UD-IQ3_XXS, dflash ×7, measured 2026-10-20") / [120<sup>8</sup>](docs/performance/runs/rtx4060ti/muse-glimmer-30b.md "all streams: 120 tok/s at 8 streams, UD-IQ3_XXS, no speculation, measured 2026-10-20") |  |
| [**RTX&nbsp;3090**](docs/performance/gpus/rtx3090.md "RTX 3090 24 GB (GA102, sm_86)") | [94.6](docs/performance/runs/rtx3090/qwen3.8-27b.md "one stream: 94.6 tok/s, UD-Q4_K_XL, dflash2 ×7, measured 2026-10-19") / [264<sup>8</sup>](docs/performance/runs/rtx3090/qwen3.8-27b.md "all streams: 264 tok/s at 8 streams, UD-Q4_K_XL, mtp ×3, measured 2026-10-19") | [291](docs/performance/runs/rtx3090/qwen3.6-35b-a3b.md "one stream: 291 tok/s, UD-Q4_K_M, dflash ×7, measured 2026-10-19") / [595<sup>8</sup>](docs/performance/runs/rtx3090/qwen3.6-35b-a3b.md "all streams: 595 tok/s at 8 streams, UD-Q4_K_M, no speculation, measured 2026-10-19") | [89.8](docs/performance/runs/rtx3090/gemma-4-31b-it.md "one stream: 89.8 tok/s, UD-Q4_K_XL, mtp ×6, measured 2026-10-19") / – | [234](docs/performance/runs/rtx3090/gemma-4-26b-a4b-it.md "one stream: 234 tok/s, UD-Q4_K_XL, mtp ×6, measured 2026-10-19") / [497<sup>4</sup>](docs/performance/runs/rtx3090/gemma-4-26b-a4b-it.md "all streams: 497 tok/s at 4 streams, UD-Q4_K_XL, mtp ×6, measured 2026-10-19") | [201](docs/performance/runs/rtx3090/gpt-oss-20b.md "one stream: 201 tok/s, gpt-oss-20b, dflash ×7, measured 2026-10-19") / [453<sup>8</sup>](docs/performance/runs/rtx3090/gpt-oss-20b.md "all streams: 453 tok/s at 8 streams, gpt-oss-20b, no speculation, measured 2026-10-19") | [86.1](docs/performance/runs/rtx3090/muse-glimmer-30b.md "one stream: 86.1 tok/s, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") / [262<sup>8</sup>](docs/performance/runs/rtx3090/muse-glimmer-30b.md "all streams: 262 tok/s at 8 streams, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") | [279](docs/performance/runs/rtx3090/nemotron-3.5-lightning-30b-a3b.md "one stream: 279 tok/s, UD-IQ3_XXS, mtp ×3, measured 2026-10-20") / [642<sup>8</sup>](docs/performance/runs/rtx3090/nemotron-3.5-lightning-30b-a3b.md "all streams: 642 tok/s at 8 streams, UD-IQ3_XXS, mtp ×3, measured 2026-10-20") |
| [**A10**](docs/performance/gpus/a10.md "A10 24 GB (GA102, sm_86)") | [49.9](docs/performance/runs/a10/qwen3.8-27b.md "one stream: 49.9 tok/s, UD-Q4_K_XL, dflash2 ×7, measured 2026-10-19") / [104<sup>8</sup>](docs/performance/runs/a10/qwen3.8-27b.md "all streams: 104 tok/s at 8 streams, UD-Q4_K_XL, no speculation, measured 2026-10-19") | [193](docs/performance/runs/a10/qwen3.6-35b-a3b.md "one stream: 193 tok/s, UD-Q4_K_M, dflash ×7, measured 2026-10-19") / [386<sup>8</sup>](docs/performance/runs/a10/qwen3.6-35b-a3b.md "all streams: 386 tok/s at 8 streams, UD-Q4_K_M, no speculation, measured 2026-10-19") | [55.6](docs/performance/runs/a10/gemma-4-31b-it.md "one stream: 55.6 tok/s, UD-Q4_K_XL, mtp ×6, measured 2026-10-19") / – | [146](docs/performance/runs/a10/gemma-4-26b-a4b-it.md "one stream: 146 tok/s, UD-Q4_K_XL, mtp ×6, measured 2026-10-19") / [316<sup>4</sup>](docs/performance/runs/a10/gemma-4-26b-a4b-it.md "all streams: 316 tok/s at 4 streams, UD-Q4_K_XL, mtp ×6, measured 2026-10-19") | [120](docs/performance/runs/a10/gpt-oss-20b.md "one stream: 120 tok/s, gpt-oss-20b, dflash ×7, measured 2026-10-19") / [274<sup>8</sup>](docs/performance/runs/a10/gpt-oss-20b.md "all streams: 274 tok/s at 8 streams, gpt-oss-20b, no speculation, measured 2026-10-19") | [53.2](docs/performance/runs/a10/muse-glimmer-30b.md "one stream: 53.2 tok/s, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") / [136<sup>8</sup>](docs/performance/runs/a10/muse-glimmer-30b.md "all streams: 136 tok/s at 8 streams, UD-Q4_K_XL, no speculation, measured 2026-10-20") | [131](docs/performance/runs/a10/nemotron-3.5-lightning-30b-a3b.md "one stream: 131 tok/s, UD-IQ3_XXS, no speculation, measured 2026-10-20") / [320<sup>8</sup>](docs/performance/runs/a10/nemotron-3.5-lightning-30b-a3b.md "all streams: 320 tok/s at 8 streams, UD-IQ3_XXS, no speculation, measured 2026-10-20") |
| [**A100**](docs/performance/gpus/a100-40gb.md "A100 40 GB (GA100, sm_80)") | [87.8](docs/performance/runs/a100-40gb/qwen3.8-27b.md "one stream: 87.8 tok/s, UD-Q4_K_XL, dflash2 ×7, measured 2026-10-19") / [211<sup>8</sup>](docs/performance/runs/a100-40gb/qwen3.8-27b.md "all streams: 211 tok/s at 8 streams, UD-Q4_K_XL, dflash2 ×7, measured 2026-10-19") | [298](docs/performance/runs/a100-40gb/qwen3.6-35b-a3b.md "one stream: 298 tok/s, UD-Q4_K_M, dflash ×7, measured 2026-10-20") / [773<sup>8</sup>](docs/performance/runs/a100-40gb/qwen3.6-35b-a3b.md "all streams: 773 tok/s at 8 streams, UD-Q4_K_M, dflash ×7, measured 2026-10-20") | [106](docs/performance/runs/a100-40gb/gemma-4-31b-it.md "one stream: 106 tok/s, UD-Q4_K_XL, mtp ×6, measured 2026-10-20") / [247<sup>8</sup>](docs/performance/runs/a100-40gb/gemma-4-31b-it.md "all streams: 247 tok/s at 8 streams, UD-Q4_K_XL, mtp ×6, measured 2026-10-20") | [268](docs/performance/runs/a100-40gb/gemma-4-26b-a4b-it.md "one stream: 268 tok/s, UD-Q4_K_XL, mtp ×6, measured 2026-10-20") / [683<sup>8</sup>](docs/performance/runs/a100-40gb/gemma-4-26b-a4b-it.md "all streams: 683 tok/s at 8 streams, UD-Q4_K_XL, mtp ×6, measured 2026-10-20") | [225](docs/performance/runs/a100-40gb/gpt-oss-20b.md "one stream: 225 tok/s, gpt-oss-20b, dflash ×7, measured 2026-10-20") / [627<sup>8</sup>](docs/performance/runs/a100-40gb/gpt-oss-20b.md "all streams: 627 tok/s at 8 streams, gpt-oss-20b, dflash ×7, measured 2026-10-20") | [93.0](docs/performance/runs/a100-40gb/muse-glimmer-30b.md "one stream: 93.0 tok/s, UD-Q4_K_XL, dflash ×7, measured 2026-10-20") / [244<sup>8</sup>](docs/performance/runs/a100-40gb/muse-glimmer-30b.md "all streams: 244 tok/s at 8 streams, UD-Q4_K_XL, no speculation, measured 2026-10-20") | [270](docs/performance/runs/a100-40gb/nemotron-3.5-lightning-30b-a3b.md "one stream: 270 tok/s, UD-Q4_K_XL, mtp ×3, measured 2026-10-21") / [612<sup>8</sup>](docs/performance/runs/a100-40gb/nemotron-3.5-lightning-30b-a3b.md "all streams: 612 tok/s at 8 streams, UD-Q4_K_XL, mtp ×3, measured 2026-10-21") |
| [**RTX&nbsp;2080&nbsp;Ti**](docs/performance/gpus/rtx2080ti.md "RTX 2080 Ti 11 GB (TU102, sm_75)") | [52.1](docs/performance/runs/rtx2080ti/qwen3.8-27b.md "one stream: 52.1 tok/s, UD-IQ2_XXS, dflash2 ×7, measured 2026-10-19") / [120<sup>8</sup>](docs/performance/runs/rtx2080ti/qwen3.8-27b.md "all streams: 120 tok/s at 8 streams, UD-IQ2_XXS, no speculation, measured 2026-10-19") |  |  |  |  |  |  |
| [**Tesla&nbsp;V100**](docs/performance/gpus/v100-32gb.md "Tesla V100 32 GB (GV100, sm_70)") | [33.0](docs/performance/runs/v100-32gb/qwen3.8-27b.md "one stream: 33.0 tok/s, UD-Q4_K_XL, no speculation, measured 2026-10-19") / [82.9<sup>4</sup>](docs/performance/runs/v100-32gb/qwen3.8-27b.md "all streams: 82.9 tok/s at 4 streams, UD-Q4_K_XL, no speculation, measured 2026-10-19") | [111](docs/performance/runs/v100-32gb/qwen3.6-35b-a3b.md "one stream: 111 tok/s, UD-Q4_K_M, dflash ×3, measured 2026-10-19") / [193<sup>8</sup>](docs/performance/runs/v100-32gb/qwen3.6-35b-a3b.md "all streams: 193 tok/s at 8 streams, UD-Q4_K_M, dflash ×3, measured 2026-10-19") | [57.6](docs/performance/runs/v100-32gb/gemma-4-31b-it.md "one stream: 57.6 tok/s, UD-Q4_K_XL, mtp ×3, measured 2026-10-20") / [73.9<sup>4</sup>](docs/performance/runs/v100-32gb/gemma-4-31b-it.md "all streams: 73.9 tok/s at 4 streams, UD-Q4_K_XL, no speculation, measured 2026-10-20") | [106](docs/performance/runs/v100-32gb/gemma-4-26b-a4b-it.md "one stream: 106 tok/s, UD-Q4_K_XL, mtp ×3, measured 2026-10-20") / [141<sup>8</sup>](docs/performance/runs/v100-32gb/gemma-4-26b-a4b-it.md "all streams: 141 tok/s at 8 streams, UD-Q4_K_XL, mtp ×3, measured 2026-10-20") | [93.6](docs/performance/runs/v100-32gb/gpt-oss-20b.md "one stream: 93.6 tok/s, gpt-oss-20b, no speculation, measured 2026-10-20") / [188<sup>8</sup>](docs/performance/runs/v100-32gb/gpt-oss-20b.md "all streams: 188 tok/s at 8 streams, gpt-oss-20b, no speculation, measured 2026-10-20") | [38.9](docs/performance/runs/v100-32gb/muse-glimmer-30b.md "one stream: 38.9 tok/s, UD-Q4_K_XL, no speculation, measured 2026-10-20") / [115<sup>8</sup>](docs/performance/runs/v100-32gb/muse-glimmer-30b.md "all streams: 115 tok/s at 8 streams, UD-Q4_K_XL, no speculation, measured 2026-10-20") | [88.7](docs/performance/runs/v100-32gb/nemotron-3.5-lightning-30b-a3b.md "one stream: 88.7 tok/s, UD-Q4_K_XL, mtp ×3, measured 2026-10-20") / [134<sup>8</sup>](docs/performance/runs/v100-32gb/nemotron-3.5-lightning-30b-a3b.md "all streams: 134 tok/s at 8 streams, UD-Q4_K_XL, no speculation, measured 2026-10-20") |

<sub>Measured 2026-10-19 – 2026-10-21 · NInferno 0.1.0 · [methodology](docs/performance/methodology.md)</sub>
<!-- perf:end -->

## Quick start

### 1. Install

One release for every NVIDIA GPU from the RTX 20 series and V100 on, home, workstation and datacenter
cards alike, several of them in one machine too; no compiler needed. It wants NVIDIA driver 580 or
newer (570 for a V100).

**Windows:** download
[NInferno-Setup.exe](https://github.com/Brontir/NInferno/releases/latest/download/NInferno-Setup.exe) and
run it. It needs no administrator rights, installs the native engine, and adds NInferno to the Start
menu; it updates itself. [Windows](docs/windows.md) has the details.

**Linux** (or WSL 2):

```bash
curl -fsSL https://raw.githubusercontent.com/Brontir/NInferno/master/tools/release/install.sh | sh
```

The installer checks your GPUs, then installs only what the release manifest names, and only with a
valid signature of the release key it carries ([how releases are signed](tools/release/server/README.md#signing)).
To read it before running it, download [`install.sh`](tools/release/install.sh) and run it with `sh`. It unpacks the release into
`~/.local/share/ninferno` and links the commands into `~/.local/bin`; `NINFERNO_HOME` and
`NINFERNO_BIN` change those, `NINFERNO_UPDATE_URL` names another manifest (still checked), and
`NINFERNO_ARCHIVE=FILE` installs an archive you already have. Besides `ninferno-ui` and
`ninferno-serve` it installs `ninferno-start`, which lists the models measured on your GPU and starts
one at its measured settings (`--list` only lists them, `--multi` takes the setting for several
users, `NINFERNO_DEVICE` picks the GPU), and `ninferno-update`, which installs a newer release
(`--check` only says whether there is one, `--yes` skips the question, `--rollback` returns to the
one before).

### 2. Run with the GUI

```bash
ninferno-ui
```

On Windows, start **NInferno** from the Start menu instead. The page opens in your default
browser at `http://localhost:8090`, and from there you can:

- see your GPUs and the models measured on them, with the speed to expect;
- search Hugging Face, compare quantizations by size, context and speed, and download one;
- start it with the measured settings, then chat with it or point any OpenAI-compatible app at it.

The page is reachable from your own machine only. On a server with no display, `ninferno-ui`
prints the address instead of opening a browser (`--no-browser` asks for that anywhere); over SSH,
forward port 8090.

### 3. Or run from the command line

Name a model on Hugging Face. The first start downloads it and converts it once for your GPU:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL --with-drafters
```

The server answers the OpenAI API (`/v1/chat/completions`, `/v1/responses`) and the Anthropic API
(`/v1/messages`) on `http://127.0.0.1:8080`; an Anthropic client takes that address as its base URL.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "unsloth/Qwen3.6-35B-A3B-GGUF",
       "messages": [{"role": "user", "content": "Reply with one short sentence."}],
       "max_tokens": 64}'
```

- A local checkpoint directory or `.gguf` file works too.
- `HF_TOKEN` authorizes gated models.
- [Serving](docs/serving.md), [CLI](docs/cli.md) and [weight conversion](docs/weight-conversion.md)
  list every option; the [performance pages](docs/performance/README.md) show what fits and how
  fast it runs.

### 4. Manual settings

`ninferno-serve` chooses every setting for the model and the GPU (`ninferno-start` starts the
settings measured for yours), and the startup log prints each choice. To fix one, pass its flag;
the others stay automatic, except speculation: an explicit `--spec` runs without the measured load
table, so give `--spec-load` too.

| Flag | Automatic | Set it to |
|---|---|---|
| `--max-context N` | the model's context if it fits, else the longest that fits | a shorter context leaves memory for more requests |
| `--max-concurrency N` | `1` | requests generated at once, `1..8` |
| `--spec auto\|none\|mtp\|dflash\|dflash2` | `auto`: measured for this model and GPU | `none` turns speculation off |
| `--draft-tokens N` | with `--spec`: the measured depth, else `3` (MTP) or `7` (DFlash) | draft depth, `1..15` |
| `--spec-load L:D,...\|none` | with `auto`: the measured table | from `L` requests at once, drafts at most `D` deep (`0`: none) |
| `--kv-dtype bf16\|fp8` | per model family and GPU | a smaller KV cache holds a longer context (`fp8` before RTX 40 is emulated, slower); Qwen also `int8\|nvfp4\|k8v4` |
| `--activations exact\|fast` | `fast` | `exact`: the same output whatever the load and the drafts |
| `--device N` | `0` | the GPU |
| `--host H` `--port N` `--api-key-file F` | `127.0.0.1:8080`, no key (or `$NINFERNO_API_KEY`) | `--host 0.0.0.0` with a key opens it to the network |

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL --with-drafters \
  --max-context 32768 --max-concurrency 4 \
  --spec dflash --draft-tokens 5 --spec-load 3:3 \
  --host 0.0.0.0 --api-key-file ~/.ninferno-key
```

[Serving](docs/serving.md#important-options) lists every flag.

### Build it yourself

You need Linux, a CUDA toolkit (12.8, 13.2 or 13.4; 12.8 for a V100), CMake 3.28+, a C++20 compiler and Ninja,
plus a few development libraries ([building from source](docs/building.md) lists them).

```bash
git clone https://github.com/Brontir/NInferno.git && cd NInferno
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU:

| `120a` | `89` | `86` | `80` | `75` | `70` |
|---|---|---|---|---|---|
| RTX 50, PRO 6000 | RTX 40, L4, L40S | RTX 30, A10 | A100 | RTX 20, T4 | V100 |

The binaries land in `build/apps/`. [Building from source](docs/building.md) also covers Docker
images and release archives.

## How it works

- **One GPU, one resident model, fixed capacity.** GPU memory is planned once at startup: weights,
  KV pool, CUDA graphs and drafter state. Nothing is allocated while requests run.
- **Kernels chosen per GPU class.** Every linear, attention and MoE shape takes a route decided
  from the GPU's properties, refined by the tables measured on real cards where there is one for
  its class ([`routes/`](routes/)).
- **Speculation measured per GPU.** Draft depth and the drafter come from profiles measured on each
  GPU class, and the depth shrinks as more requests run at once
  ([drafter registry](src/spec/data/drafter_registry.json)).
- **Long contexts reuse their prefix.** A retained checkpoint of KV and model state resumes an
  exact prompt prefix on the GPU or from pinned host memory; the planner weighs restore cost
  against later reuse ([resource scheduling](docs/maintainer/resource-scheduling-and-context-cache.md)).

## Limits

- One GPU and one resident model per process; no multi-GPU or distributed serving.
- One to eight concurrent requests, fixed at startup, with a bounded FIFO queue; a running request
  gives up its lane only when the KV pool runs short (`--kv-lease-window`), and resumes.
- Model architectures and weight formats run on explicitly implemented paths only.
- Tool calls are parsed and returned to the client; NInferno does not execute tools.

## Roadmap

1. **Proven on real consumer GPUs** *(now)*. Run every supported model on every GPU generation
   people actually own, fix what breaks, and keep the published numbers current.
2. **New models first** *(ongoing)*. Support the new open models that matter soon after their
   release, starting with those that fit in one GPU's memory.
3. **Bigger models and longer contexts on one GPU** *(started: `--moe-offload`)*. Compute part of a
   MoE model's experts outside the GPU, so larger MoE models and contexts fit, while keeping the
   speed of one to four concurrent users.
4. **Several GPUs** *(planned)*. Pipeline parallelism across two to six cards for models that do not
   fit one. Consumer GPUs lack the interconnect tensor parallelism needs, so NInferno will not go
   that way.

## Contributing

NInferno is fully open (Apache 2.0) and gets better with every GPU and model people run it on. The
goal is fast, affordable inference for everyone, and there is a lot to do:

- **Run it on your GPU** and report what breaks or runs slower than the tables say, with
  `ninferno-serve --version` and the startup log.
- **Measure a GPU** that has no page yet: [`tools/bench/perf_matrix.py`](tools/bench/perf_matrix.py)
  and the [methodology](docs/performance/methodology.md); send the rows in a pull request.
- **Add a model, a format or a kernel.** [Build from source](docs/building.md), read the
  [maintainer documentation](docs/maintainer/), and open an issue first: [CONTRIBUTING](CONTRIBUTING.md)
  explains how changes are agreed, tested and measured before they are merged.

## Documentation

- [Documentation index](docs/README.md)
- [Performance by GPU and model](docs/performance/README.md)
- [The local page (GUI)](docs/ui.md) · [HTTP serving](docs/serving.md) · [CLI](docs/cli.md) · [Weight conversion](docs/weight-conversion.md)
- [Building from source](docs/building.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Maintainer documentation](docs/maintainer/) · [Contributing](CONTRIBUTING.md)

## Acknowledgements

NInferno started from [ninfer](https://github.com/Neroued/ninfer) by Neroued and grew from there.

It includes or builds with code from:

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp): the GGUF block-quant formats and their
  integer matrix kernel, the Jinja template engine and the chat window of ninferno-ui;
- [ninfer-all](https://github.com/iamwavecut/ninfer-all): the small-T tensor-core kernels for
  verify widths, the integer-activation prefill GEMMs, the ggml matrix-kernel bridge and the int8
  prompt attention;
- [ninfer-v100-sm70](https://github.com/Flo5k5/ninfer-v100-sm70), through which
  [sm70-attn](https://github.com/fishlikeX/sm70-attn) (1Cat-vLLM's kernel on a FlashAttention base
  layer) came: Volta attention for head size 256;
- [CUTLASS](https://github.com/NVIDIA/cutlass): Volta tensor-core GEMM for prefill;
- [xgrammar](https://github.com/mlc-ai/xgrammar): constrained output for tool calls and JSON;
- [cpp-httplib](https://github.com/yhirose/cpp-httplib), [nlohmann/json](https://github.com/nlohmann/json),
  [spdlog](https://github.com/gabime/spdlog) and [utf8proc](https://github.com/JuliaStrings/utf8proc).

Ideas and techniques came from [FlashInfer](https://github.com/flashinfer-ai/flashinfer) (linear attention),
[FlashAttention](https://github.com/Dao-AILab/flash-attention), [vLLM](https://github.com/vllm-project/vllm)
and [SGLang](https://github.com/sgl-project/sglang) (serving and API conventions),
[z-lab](https://huggingface.co/z-lab)'s DFlash (block-diffusion drafters), and the other forks of ninfer. [NOTICE](NOTICE) lists
every file taken from another project.

## License

NInferno is licensed under the [Apache License 2.0](LICENSE). It is derived from
[ninfer](https://github.com/Neroued/ninfer) by Neroued, under the same license; see [NOTICE](NOTICE).
Vendored dependencies keep their own licenses under `third_party/`. Model weights are subject to the
licenses of their original checkpoints.
