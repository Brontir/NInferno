# Qwen3.8-27B

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) · Qwen3.5 dense, 27B · [all GPUs and models](../README.md)

Measured with [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;5090](../gpus/rtx5090.md "RTX 5090 32 GB (GB202, sm_120)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 193 | 557<sup>8</sup> | 4,154 | 195 | [details](../runs/rtx5090/qwen3.8-27b.md) |
| [RTX&nbsp;5070&nbsp;Ti](../gpus/rtx5070ti.md "RTX 5070 Ti 16 GB (GB203, sm_120)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 120 | 261<sup>8</sup> | 2,070 | 122 | [details](../runs/rtx5070ti/qwen3.8-27b.md) |
| [RTX&nbsp;4090](../gpus/rtx4090.md "RTX 4090 24 GB (AD102, sm_89)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 122 | 366<sup>8</sup> | 3,376 | 127 | [details](../runs/rtx4090/qwen3.8-27b.md) |
| [RTX&nbsp;4070&nbsp;Ti](../gpus/rtx4070ti.md "RTX 4070 Ti 12 GB (AD104, sm_89)") | [`UD-IQ2_XXS`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 104 | 216<sup>8</sup> | 1,709 | 112 | [details](../runs/rtx4070ti/qwen3.8-27b.md) |
| [L4](../gpus/l4.md "L4 24 GB (AD104, sm_89)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 32.8 | 67.1<sup>8</sup> | 816 | 33.4 | [details](../runs/l4/qwen3.8-27b.md) |
| [RTX&nbsp;4060&nbsp;Ti&nbsp;16&nbsp;GB](../gpus/rtx4060ti.md "RTX 4060 Ti 16 GB (AD106, sm_89)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 49.0 | 115<sup>8</sup> | 986 | 47.5 | [details](../runs/rtx4060ti/qwen3.8-27b.md) |
| [RTX&nbsp;3090](../gpus/rtx3090.md "RTX 3090 24 GB (GA102, sm_86)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 94.6 | 264<sup>8</sup> | 1,572 | 91.8 | [details](../runs/rtx3090/qwen3.8-27b.md) |
| [A10](../gpus/a10.md "A10 24 GB (GA102, sm_86)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 49.9 | 104<sup>8</sup> | 919 | 47.1 | [details](../runs/a10/qwen3.8-27b.md) |
| [A100&nbsp;40&nbsp;GB](../gpus/a100-40gb.md "A100 40 GB (GA100, sm_80)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 87.8 | 211<sup>8</sup> | 1,268 | 89.6 | [details](../runs/a100-40gb/qwen3.8-27b.md) |
| [RTX&nbsp;2080&nbsp;Ti](../gpus/rtx2080ti.md "RTX 2080 Ti 11 GB (TU102, sm_75)") | [`UD-IQ2_XXS`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 52.1 | 120<sup>8</sup> | 660 | 56.5 | [details](../runs/rtx2080ti/qwen3.8-27b.md) |
| [Tesla&nbsp;V100&nbsp;32&nbsp;GB](../gpus/v100-32gb.md "Tesla V100 32 GB (GV100, sm_70)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 33.0 | 82.9<sup>4</sup> | 856 | 33.2 | [details](../runs/v100-32gb/qwen3.8-27b.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 2080 Ti, driver 580.178.04, 240 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · NVIDIA GeForce RTX 4070 Ti, driver 615.71.09, 285 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
