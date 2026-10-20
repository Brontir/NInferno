# Gemma 4 31B

[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) · Gemma 4 dense, 31B · [all GPUs and models](../README.md)

The model on each GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX 5090](../gpus/rtx5090.md) | [`unsloth/gemma-4-31B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) UD-Q4_K_XL | 151 | 416 (C=8) | 3,078 | 133 | [all settings](../runs/rtx5090/gemma-4-31b-it.md) |
| [RTX 4090](../gpus/rtx4090.md) | [`unsloth/gemma-4-31B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) UD-Q4_K_XL | 114 | 143 (C=4) | 2,428 | 97.5 | [all settings](../runs/rtx4090/gemma-4-31b-it.md) |
| [L4](../gpus/l4.md) | [`unsloth/gemma-4-31B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) UD-Q4_K_XL | 33.0 | – | 612 | 30.0 | [all settings](../runs/l4/gemma-4-31b-it.md) |
| [RTX 3090](../gpus/rtx3090.md) | [`unsloth/gemma-4-31B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) UD-Q4_K_XL | 89.8 | – | 1,051 | 79.8 | [all settings](../runs/rtx3090/gemma-4-31b-it.md) |
| [A10](../gpus/a10.md) | [`unsloth/gemma-4-31B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) UD-Q4_K_XL | 55.6 | – | 648 | 49.5 | [all settings](../runs/a10/gemma-4-31b-it.md) |
| [A100 40 GB](../gpus/a100-40gb.md) | [`unsloth/gemma-4-31B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) UD-Q4_K_XL | 106 | 247 (C=8) | 1,010 | 97.4 | [all settings](../runs/a100-40gb/gemma-4-31b-it.md) |
| [Tesla V100 32 GB](../gpus/v100-32gb.md) | [`unsloth/gemma-4-31B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) UD-Q4_K_XL | 57.6 | 73.9 (C=4) | 266 | 43.1 | [all settings](../runs/v100-32gb/gemma-4-31b-it.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
