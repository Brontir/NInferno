# RTX 2080 Ti

TU102, sm_75, 11 GB · [all GPUs and models](../README.md)

Each model on this GPU, tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| Model | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [Qwen3.8-27B](../models/qwen3.8-27b.md) | [`UD-IQ2_XXS`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 52.1 | 133<sup>8</sup> | 674 | 56.5 | [details](../runs/rtx2080ti/qwen3.8-27b.md) |
| [Qwen3.5-9B](../models/qwen3.5-9b.md) | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF "unsloth/Qwen3.5-9B-GGUF") | 165 | 397<sup>8</sup> | 2,363 | 151 | [details](../runs/rtx2080ti/qwen3.5-9b.md) |
| [Qwen3.5-4B](../models/qwen3.5-4b.md) | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF "unsloth/Qwen3.5-4B-GGUF") | 254 | 631<sup>8</sup> | 3,657 | 220 | [details](../runs/rtx2080ti/qwen3.5-4b.md) |
| [Qwen3-8B](../models/qwen3-8b.md) | [`Q4_K_M`](https://huggingface.co/Qwen/Qwen3-8B-GGUF "Qwen/Qwen3-8B-GGUF") | 92.9 | 386<sup>8</sup> | 1,436 | 42.1 | [details](../runs/rtx2080ti/qwen3-8b.md) |

Weights from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), [`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), [`Qwen/Qwen3-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-8B-GGUF).

<sub>Measured 2026-10-19 to 2026-11-08 · NInferno 0.1.0 · NVIDIA GeForce RTX 2080 Ti, driver 580.178.04, 240 W · NVIDIA GeForce RTX 2080 Ti, driver 615.71.09, 260 W · [methodology](../methodology.md)</sub>
