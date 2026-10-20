# RTX 2080 Ti

TU102, sm_75, 11 GB · [all GPUs and models](../README.md)

Each model on this GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| Model | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [Qwen3.8-27B](../models/qwen3.8-27b.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-IQ2_XXS | 52.1 | 120 (C=8) | 644 | 56.5 | [all settings](../runs/rtx2080ti/qwen3.8-27b.md) |
| [Qwen3.5-9B](../models/qwen3.5-9b.md) | [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) Q4_K_M | 91.7 | 334 (C=8) | 2,322 | 83.6 | [all settings](../runs/rtx2080ti/qwen3.5-9b.md) |
| [Qwen3.5-4B](../models/qwen3.5-4b.md) | [`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) Q4_K_M | 145 | 506 (C=8) | 3,581 | 124 | [all settings](../runs/rtx2080ti/qwen3.5-4b.md) |
| [Qwen3-8B](../models/qwen3-8b.md) | [`Qwen/Qwen3-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-8B-GGUF) Q4_K_M | 92.9 | 386 (C=8) | 1,436 | 42.1 | [all settings](../runs/rtx2080ti/qwen3-8b.md) |

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 2080 Ti, driver 580.178.04, 240 W · [methodology](../methodology.md)</sub>
