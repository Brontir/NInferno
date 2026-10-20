# Qwen3.5-4B

[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) · Qwen3.5 dense, 4B · [all GPUs and models](../README.md)

The model on each GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX 4070 Ti](../gpus/rtx4070ti.md) | [`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) Q4_K_M | 149 | 628 (C=8) | 9,512 | 134 | [all settings](../runs/rtx4070ti/qwen3.5-4b.md) |
| [RTX 2080 Ti](../gpus/rtx2080ti.md) | [`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) Q4_K_M | 145 | 506 (C=8) | 3,581 | 124 | [all settings](../runs/rtx2080ti/qwen3.5-4b.md) |

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 2080 Ti, driver 580.178.04, 240 W · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · [methodology](../methodology.md)</sub>
