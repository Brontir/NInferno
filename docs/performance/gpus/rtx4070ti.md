# RTX 4070 Ti

AD104, sm_89, 12 GB · [all GPUs and models](../README.md)

Each model on this GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| Model | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [Qwen3.8-27B](../models/qwen3.8-27b.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-IQ2_XXS | 104 | 216 (C=8) | 1,688 | 112 | [all settings](../runs/rtx4070ti/qwen3.8-27b.md) |
| [Qwen3.5-9B](../models/qwen3.5-9b.md) | [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) Q4_K_M | 85.4 | 373 (C=8) | 5,700 | 80.5 | [all settings](../runs/rtx4070ti/qwen3.5-9b.md) |
| [Qwen3.5-4B](../models/qwen3.5-4b.md) | [`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) Q4_K_M | 149 | 628 (C=8) | 9,512 | 134 | [all settings](../runs/rtx4070ti/qwen3.5-4b.md) |
| [Qwen3-8B](../models/qwen3-8b.md) | [`Qwen/Qwen3-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-8B-GGUF) Q4_K_M | 93.2 | 500 (C=8) | 3,751 | 69.7 | [all settings](../runs/rtx4070ti/qwen3-8b.md) |

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · [methodology](../methodology.md)</sub>
