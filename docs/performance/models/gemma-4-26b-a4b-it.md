# Gemma 4 26B-A4B

[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) · Gemma 4 MoE, 26B total, 4B active · [all GPUs and models](../README.md)

The model on each GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX 5090](../gpus/rtx5090.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-Q4_K_XL | 464 | 1,349 (C=8) | 15,102 | 431 | [all settings](../runs/rtx5090/gemma-4-26b-a4b-it.md) |
| [RTX 5070 Ti](../gpus/rtx5070ti.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-IQ3_XXS | 301 | 638 (C=4) | 6,516 | 283 | [all settings](../runs/rtx5070ti/gemma-4-26b-a4b-it.md) |
| [RTX 4090](../gpus/rtx4090.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-Q4_K_XL | 321 | 892 (C=8) | 11,833 | 302 | [all settings](../runs/rtx4090/gemma-4-26b-a4b-it.md) |
| [L4](../gpus/l4.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-Q4_K_XL | 92.9 | 205 (C=4) | 3,092 | 86.1 | [all settings](../runs/l4/gemma-4-26b-a4b-it.md) |
| [RTX 4060 Ti 16 GB](../gpus/rtx4060ti.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-IQ3_XXS | 127 | 294 (C=4) | 3,158 | 119 | [all settings](../runs/rtx4060ti/gemma-4-26b-a4b-it.md) |
| [RTX 3090](../gpus/rtx3090.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-Q4_K_XL | 234 | 497 (C=4) | 5,210 | 223 | [all settings](../runs/rtx3090/gemma-4-26b-a4b-it.md) |
| [A10](../gpus/a10.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-Q4_K_XL | 146 | 316 (C=4) | 3,443 | 137 | [all settings](../runs/a10/gemma-4-26b-a4b-it.md) |
| [A100 40 GB](../gpus/a100-40gb.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-Q4_K_XL | 268 | 683 (C=8) | 5,948 | 252 | [all settings](../runs/a100-40gb/gemma-4-26b-a4b-it.md) |
| [Tesla V100 32 GB](../gpus/v100-32gb.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-Q4_K_XL | 106 | 141 (C=8) | 861 | 90.6 | [all settings](../runs/v100-32gb/gemma-4-26b-a4b-it.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
