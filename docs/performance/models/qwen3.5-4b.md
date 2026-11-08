# Qwen3.5-4B

[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) · Qwen3.5 dense, 4B · [all GPUs and models](../README.md)

Measured with [`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF). Fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;4070&nbsp;Ti](../gpus/rtx4070ti.md "RTX 4070 Ti 12 GB (AD104, sm_89)") | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF "unsloth/Qwen3.5-4B-GGUF") | 286 | 649<sup>8</sup> | 9,627 | 275 | [details](../runs/rtx4070ti/qwen3.5-4b.md) |
| [RTX&nbsp;3060](../gpus/rtx3060.md "RTX 3060 12 GB (GA106, sm_86)") | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF "unsloth/Qwen3.5-4B-GGUF") | 185 | 442<sup>8</sup> | 3,303 | 173 | [details](../runs/rtx3060/qwen3.5-4b.md) |
| [RTX&nbsp;2080&nbsp;Ti](../gpus/rtx2080ti.md "RTX 2080 Ti 11 GB (TU102, sm_75)") | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF "unsloth/Qwen3.5-4B-GGUF") | 254 | 631<sup>8</sup> | 3,657 | 220 | [details](../runs/rtx2080ti/qwen3.5-4b.md) |

<sub>Measured 2026-11-08 · NInferno 0.1.0 · NVIDIA GeForce RTX 2080 Ti, driver 615.71.09, 260 W · NVIDIA GeForce RTX 3060, driver 615.71.09, 170 W · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · [methodology](../methodology.md)</sub>
