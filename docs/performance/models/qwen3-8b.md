# Qwen3-8B

[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) · dense decoder, 8B · [all GPUs and models](../README.md)

Measured with [`Qwen/Qwen3-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-8B-GGUF). Fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;4070&nbsp;Ti](../gpus/rtx4070ti.md "RTX 4070 Ti 12 GB (AD104, sm_89)") | [`Q4_K_M`](https://huggingface.co/Qwen/Qwen3-8B-GGUF "Qwen/Qwen3-8B-GGUF") | 93.2 | 500<sup>8</sup> | 3,751 | 69.7 | [details](../runs/rtx4070ti/qwen3-8b.md) |
| [RTX&nbsp;2080&nbsp;Ti](../gpus/rtx2080ti.md "RTX 2080 Ti 11 GB (TU102, sm_75)") | [`Q4_K_M`](https://huggingface.co/Qwen/Qwen3-8B-GGUF "Qwen/Qwen3-8B-GGUF") | 92.9 | 386<sup>8</sup> | 1,436 | 42.1 | [details](../runs/rtx2080ti/qwen3-8b.md) |

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 2080 Ti, driver 580.178.04, 240 W · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · [methodology](../methodology.md)</sub>
