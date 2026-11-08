# Qwen3.5-9B

[`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) · Qwen3.5 dense, 9B · [all GPUs and models](../README.md)

Measured with [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF). Fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;4070&nbsp;Ti](../gpus/rtx4070ti.md "RTX 4070 Ti 12 GB (AD104, sm_89)") | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF "unsloth/Qwen3.5-9B-GGUF") | 176 | 398<sup>8</sup> | 5,718 | 187 | [details](../runs/rtx4070ti/qwen3.5-9b.md) |
| [RTX&nbsp;3060](../gpus/rtx3060.md "RTX 3060 12 GB (GA106, sm_86)") | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF "unsloth/Qwen3.5-9B-GGUF") | 120 | 264<sup>8</sup> | 1,999 | 118 | [details](../runs/rtx3060/qwen3.5-9b.md) |
| [RTX&nbsp;2080&nbsp;Ti](../gpus/rtx2080ti.md "RTX 2080 Ti 11 GB (TU102, sm_75)") | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF "unsloth/Qwen3.5-9B-GGUF") | 165 | 397<sup>8</sup> | 2,363 | 151 | [details](../runs/rtx2080ti/qwen3.5-9b.md) |

<sub>Measured 2026-11-08 · NInferno 0.1.0 · NVIDIA GeForce RTX 2080 Ti, driver 615.71.09, 260 W · NVIDIA GeForce RTX 3060, driver 615.71.09, 170 W · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · [methodology](../methodology.md)</sub>
