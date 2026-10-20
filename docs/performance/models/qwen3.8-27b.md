# Qwen3.8-27B

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) · Qwen3.5 dense, 27B · [all GPUs and models](../README.md)

The model on each GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX 5090](../gpus/rtx5090.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_XL | 193 | 557 (C=8) | 3,996 | 195 | [all settings](../runs/rtx5090/qwen3.8-27b.md) |
| [RTX 5070 Ti](../gpus/rtx5070ti.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-IQ3_XXS | 120 | 261 (C=8) | 2,051 | 122 | [all settings](../runs/rtx5070ti/qwen3.8-27b.md) |
| [RTX 4090](../gpus/rtx4090.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_XL | 122 | 366 (C=8) | 3,345 | 127 | [all settings](../runs/rtx4090/qwen3.8-27b.md) |
| [RTX 4070 Ti](../gpus/rtx4070ti.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-IQ2_XXS | 104 | 216 (C=8) | 1,688 | 112 | [all settings](../runs/rtx4070ti/qwen3.8-27b.md) |
| [L4](../gpus/l4.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_XL | 32.8 | 67.1 (C=8) | 750 | 33.4 | [all settings](../runs/l4/qwen3.8-27b.md) |
| [RTX 4060 Ti 16 GB](../gpus/rtx4060ti.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-IQ3_XXS | 49.0 | 115 (C=8) | 869 | 47.5 | [all settings](../runs/rtx4060ti/qwen3.8-27b.md) |
| [RTX 3090](../gpus/rtx3090.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_XL | 94.6 | 264 (C=8) | 1,557 | 91.8 | [all settings](../runs/rtx3090/qwen3.8-27b.md) |
| [A10](../gpus/a10.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_XL | 49.9 | 104 (C=8) | 906 | 47.1 | [all settings](../runs/a10/qwen3.8-27b.md) |
| [A100 40 GB](../gpus/a100-40gb.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_XL | 87.8 | 211 (C=8) | 1,268 | 89.6 | [all settings](../runs/a100-40gb/qwen3.8-27b.md) |
| [RTX 2080 Ti](../gpus/rtx2080ti.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-IQ2_XXS | 52.1 | 120 (C=8) | 644 | 56.5 | [all settings](../runs/rtx2080ti/qwen3.8-27b.md) |
| [Tesla V100 32 GB](../gpus/v100-32gb.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_XL | 33.0 | 82.9 (C=4) | 856 | 33.2 | [all settings](../runs/v100-32gb/qwen3.8-27b.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 2080 Ti, driver 580.178.04, 240 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
