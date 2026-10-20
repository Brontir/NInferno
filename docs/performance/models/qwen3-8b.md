# Qwen3-8B

[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) · dense decoder, 8B · [all GPUs and models](../README.md)

The model on each GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX 4070 Ti](../gpus/rtx4070ti.md) | [`Qwen/Qwen3-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-8B-GGUF) Q4_K_M | 93.2 | 500 (C=8) | 3,751 | 69.7 | [all settings](../runs/rtx4070ti/qwen3-8b.md) |
| [RTX 2080 Ti](../gpus/rtx2080ti.md) | [`Qwen/Qwen3-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-8B-GGUF) Q4_K_M | 92.9 | 386 (C=8) | 1,436 | 42.1 | [all settings](../runs/rtx2080ti/qwen3-8b.md) |

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 2080 Ti, driver 580.178.04, 240 W · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · [methodology](../methodology.md)</sub>
