# Qwen3.5-9B

[`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) · Qwen3.5 dense, 9B · [all GPUs and models](../README.md)

The model on each GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX 4070 Ti](../gpus/rtx4070ti.md) | [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) Q4_K_M | 85.4 | 373 (C=8) | 5,700 | 80.5 | [all settings](../runs/rtx4070ti/qwen3.5-9b.md) |
| [RTX 2080 Ti](../gpus/rtx2080ti.md) | [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) Q4_K_M | 91.7 | 334 (C=8) | 2,322 | 83.6 | [all settings](../runs/rtx2080ti/qwen3.5-9b.md) |

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 2080 Ti, driver 580.178.04, 240 W · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · [methodology](../methodology.md)</sub>
