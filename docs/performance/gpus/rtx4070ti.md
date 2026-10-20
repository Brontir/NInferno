# RTX 4070 Ti

AD104, sm_89, 12 GB · [all GPUs and models](../README.md)

Each model on this GPU, tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| Model | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [Qwen3.8-27B](../models/qwen3.8-27b.md) | [`UD-IQ2_XXS`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 104 | 216<sup>8</sup> | 1,688 | 112 | [details](../runs/rtx4070ti/qwen3.8-27b.md) |
| [Qwen3.5-9B](../models/qwen3.5-9b.md) | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF "unsloth/Qwen3.5-9B-GGUF") | 85.4 | 373<sup>8</sup> | 5,700 | 80.5 | [details](../runs/rtx4070ti/qwen3.5-9b.md) |
| [Qwen3.5-4B](../models/qwen3.5-4b.md) | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF "unsloth/Qwen3.5-4B-GGUF") | 149 | 628<sup>8</sup> | 9,512 | 134 | [details](../runs/rtx4070ti/qwen3.5-4b.md) |
| [Qwen3-8B](../models/qwen3-8b.md) | [`Q4_K_M`](https://huggingface.co/Qwen/Qwen3-8B-GGUF "Qwen/Qwen3-8B-GGUF") | 93.2 | 500<sup>8</sup> | 3,751 | 69.7 | [details](../runs/rtx4070ti/qwen3-8b.md) |

Weights from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), [`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), [`Qwen/Qwen3-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-8B-GGUF).

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · [methodology](../methodology.md)</sub>
