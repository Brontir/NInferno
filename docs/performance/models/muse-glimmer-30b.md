# Muse Glimmer 30B

[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) · dense decoder, 30B · [all GPUs and models](../README.md)

Measured with [`unsloth/Muse-Glimmer-30B-GGUF`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF). Fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;5090](../gpus/rtx5090.md "RTX 5090 32 GB (GB202, sm_120)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 193 | 668<sup>8</sup> | 4,405 | 195 | [details](../runs/rtx5090/muse-glimmer-30b.md) |
| [RTX&nbsp;5070&nbsp;Ti](../gpus/rtx5070ti.md "RTX 5070 Ti 16 GB (GB203, sm_120)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 101 | 268<sup>8</sup> | 2,275 | 103 | [details](../runs/rtx5070ti/muse-glimmer-30b.md) |
| [RTX&nbsp;4090](../gpus/rtx4090.md "RTX 4090 24 GB (AD102, sm_89)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 123 | 402<sup>8</sup> | 3,522 | 124 | [details](../runs/rtx4090/muse-glimmer-30b.md) |
| [L4](../gpus/l4.md "L4 24 GB (AD104, sm_89)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 34.3 | 87.8<sup>8</sup> | 827 | 31.9 | [details](../runs/l4/muse-glimmer-30b.md) |
| [RTX&nbsp;4060&nbsp;Ti&nbsp;16&nbsp;GB](../gpus/rtx4060ti.md "RTX 4060 Ti 16 GB (AD106, sm_89)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 45.9 | 120<sup>8</sup> | 957 | 44.7 | [details](../runs/rtx4060ti/muse-glimmer-30b.md) |
| [RTX&nbsp;3090](../gpus/rtx3090.md "RTX 3090 24 GB (GA102, sm_86)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 86.1 | 262<sup>8</sup> | 1,563 | 89.4 | [details](../runs/rtx3090/muse-glimmer-30b.md) |
| [A10](../gpus/a10.md "A10 24 GB (GA102, sm_86)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 53.2 | 136<sup>8</sup> | 1,018 | 57.1 | [details](../runs/a10/muse-glimmer-30b.md) |
| [A100&nbsp;40&nbsp;GB](../gpus/a100-40gb.md "A100 40 GB (GA100, sm_80)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 93.0 | 244<sup>8</sup> | 1,461 | 99.7 | [details](../runs/a100-40gb/muse-glimmer-30b.md) |
| [Tesla&nbsp;V100&nbsp;32&nbsp;GB](../gpus/v100-32gb.md "Tesla V100 32 GB (GV100, sm_70)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 38.9 | 115<sup>8</sup> | 552 | 29.9 | [details](../runs/v100-32gb/muse-glimmer-30b.md) |

<sub>Measured 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
