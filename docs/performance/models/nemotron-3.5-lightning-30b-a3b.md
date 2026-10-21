# Nemotron 3.5 Lightning 30B-A3B

[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) · Nemotron-H hybrid MoE, 30B total, 3B active · [all GPUs and models](../README.md)

Measured with [`unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF). Fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;5090](../gpus/rtx5090.md "RTX 5090 32 GB (GB202, sm_120)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF") | 488 | 1,077<sup>8</sup> | 15,404 | 547 | [details](../runs/rtx5090/nemotron-3.5-lightning-30b-a3b.md) |
| [RTX&nbsp;4090](../gpus/rtx4090.md "RTX 4090 24 GB (AD102, sm_89)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF") | 365 | 905<sup>8</sup> | 13,173 | 394 | [details](../runs/rtx4090/nemotron-3.5-lightning-30b-a3b.md) |
| [L4](../gpus/l4.md "L4 24 GB (AD104, sm_89)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF") | 79.3 | 197<sup>8</sup> | 3,271 | 76.4 | [details](../runs/l4/nemotron-3.5-lightning-30b-a3b.md) |
| [RTX&nbsp;3090](../gpus/rtx3090.md "RTX 3090 24 GB (GA102, sm_86)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF") | 279 | 642<sup>8</sup> | 6,150 | 304 | [details](../runs/rtx3090/nemotron-3.5-lightning-30b-a3b.md) |
| [A10](../gpus/a10.md "A10 24 GB (GA102, sm_86)") | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF") | 131 | 320<sup>8</sup> | 3,808 | 128 | [details](../runs/a10/nemotron-3.5-lightning-30b-a3b.md) |
| [A100&nbsp;40&nbsp;GB](../gpus/a100-40gb.md "A100 40 GB (GA100, sm_80)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF") | 270 | 612<sup>8</sup> | 5,808 | 305 | [details](../runs/a100-40gb/nemotron-3.5-lightning-30b-a3b.md) |
| [Tesla&nbsp;V100&nbsp;32&nbsp;GB](../gpus/v100-32gb.md "Tesla V100 32 GB (GV100, sm_70)") | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF") | 88.7 | 134<sup>8</sup> | 1,275 | 89.1 | [details](../runs/v100-32gb/nemotron-3.5-lightning-30b-a3b.md) |

<sub>Measured 2026-10-20 to 2026-10-21 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
