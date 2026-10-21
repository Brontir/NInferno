# Gemma 4 26B-A4B

[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) · Gemma 4 MoE, 26B total, 4B active · [all GPUs and models](../README.md)

Measured with [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF). Fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;5090](../gpus/rtx5090.md "RTX 5090 32 GB (GB202, sm_120)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 464 | 1,349<sup>8</sup> | 15,102 | 431 | [details](../runs/rtx5090/gemma-4-26b-a4b-it.md) |
| [RTX&nbsp;5070&nbsp;Ti](../gpus/rtx5070ti.md "RTX 5070 Ti 16 GB (GB203, sm_120)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 301 | 638<sup>4</sup> | 6,516 | 283 | [details](../runs/rtx5070ti/gemma-4-26b-a4b-it.md) |
| [RTX&nbsp;4090](../gpus/rtx4090.md "RTX 4090 24 GB (AD102, sm_89)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 321 | 892<sup>8</sup> | 11,901 | 302 | [details](../runs/rtx4090/gemma-4-26b-a4b-it.md) |
| [L4](../gpus/l4.md "L4 24 GB (AD104, sm_89)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 92.9 | 205<sup>4</sup> | 3,092 | 86.1 | [details](../runs/l4/gemma-4-26b-a4b-it.md) |
| [RTX&nbsp;4060&nbsp;Ti&nbsp;16&nbsp;GB](../gpus/rtx4060ti.md "RTX 4060 Ti 16 GB (AD106, sm_89)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 127 | 294<sup>4</sup> | 3,161 | 119 | [details](../runs/rtx4060ti/gemma-4-26b-a4b-it.md) |
| [RTX&nbsp;3090](../gpus/rtx3090.md "RTX 3090 24 GB (GA102, sm_86)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 234 | 497<sup>4</sup> | 5,210 | 223 | [details](../runs/rtx3090/gemma-4-26b-a4b-it.md) |
| [A10](../gpus/a10.md "A10 24 GB (GA102, sm_86)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 146 | 316<sup>4</sup> | 3,443 | 137 | [details](../runs/a10/gemma-4-26b-a4b-it.md) |
| [A100&nbsp;40&nbsp;GB](../gpus/a100-40gb.md "A100 40 GB (GA100, sm_80)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 268 | 683<sup>8</sup> | 5,948 | 252 | [details](../runs/a100-40gb/gemma-4-26b-a4b-it.md) |
| [Tesla&nbsp;V100&nbsp;32&nbsp;GB](../gpus/v100-32gb.md "Tesla V100 32 GB (GV100, sm_70)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 106 | 141<sup>8</sup> | 862 | 90.6 | [details](../runs/v100-32gb/gemma-4-26b-a4b-it.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
