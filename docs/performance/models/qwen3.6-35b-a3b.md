# Qwen3.6-35B-A3B

[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · Qwen3.5 MoE, 35B total, 3B active · [all GPUs and models](../README.md)

The model on each GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX 5090](../gpus/rtx5090.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-Q4_K_M | 588 | 1,526 (C=8) | 18,032 | 593 | [all settings](../runs/rtx5090/qwen3.6-35b-a3b.md) |
| [RTX 5070 Ti](../gpus/rtx5070ti.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-IQ3_XXS | 365 | 755 (C=8) | 7,224 | 361 | [all settings](../runs/rtx5070ti/qwen3.6-35b-a3b.md) |
| [RTX 4090](../gpus/rtx4090.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-Q4_K_M | 398 | 843 (C=4) | 13,550 | 412 | [all settings](../runs/rtx4090/qwen3.6-35b-a3b.md) |
| [L4](../gpus/l4.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-Q4_K_M | 114 | 247 (C=8) | 3,278 | 115 | [all settings](../runs/l4/qwen3.6-35b-a3b.md) |
| [RTX 4060 Ti 16 GB](../gpus/rtx4060ti.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-IQ3_XXS | 155 | 360 (C=8) | 3,665 | 158 | [all settings](../runs/rtx4060ti/qwen3.6-35b-a3b.md) |
| [RTX 3090](../gpus/rtx3090.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-Q4_K_M | 291 | 595 (C=8) | 6,399 | 291 | [all settings](../runs/rtx3090/qwen3.6-35b-a3b.md) |
| [A10](../gpus/a10.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-Q4_K_M | 193 | 386 (C=8) | 3,875 | 188 | [all settings](../runs/a10/qwen3.6-35b-a3b.md) |
| [A100 40 GB](../gpus/a100-40gb.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-Q4_K_M | 298 | 773 (C=8) | 6,311 | 310 | [all settings](../runs/a100-40gb/qwen3.6-35b-a3b.md) |
| [Tesla V100 32 GB](../gpus/v100-32gb.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-Q4_K_M | 111 | 193 (C=8) | 1,750 | 107 | [all settings](../runs/v100-32gb/qwen3.6-35b-a3b.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
