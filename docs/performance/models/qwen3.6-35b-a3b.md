# Qwen3.6-35B-A3B

[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · Qwen3.5 MoE, 35B total, 3B active · [all GPUs and models](../README.md)

Measured with [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF). Its fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;5090](../gpus/rtx5090.md "RTX 5090, 32 GB (GB202, sm_120)") | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 588 | 1,526<sup>8</sup> | 18,032 | 593 | [details](../runs/rtx5090/qwen3.6-35b-a3b.md) |
| [RTX&nbsp;5070&nbsp;Ti](../gpus/rtx5070ti.md "RTX 5070 Ti, 16 GB (GB203, sm_120)") | [UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 365 | 755<sup>8</sup> | 7,224 | 361 | [details](../runs/rtx5070ti/qwen3.6-35b-a3b.md) |
| [RTX&nbsp;4090](../gpus/rtx4090.md "RTX 4090, 24 GB (AD102, sm_89)") | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 398 | 843<sup>4</sup> | 13,550 | 412 | [details](../runs/rtx4090/qwen3.6-35b-a3b.md) |
| [L4](../gpus/l4.md "L4, 24 GB (AD104, sm_89)") | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 114 | 247<sup>8</sup> | 3,278 | 115 | [details](../runs/l4/qwen3.6-35b-a3b.md) |
| [RTX&nbsp;4060&nbsp;Ti](../gpus/rtx4060ti.md "RTX 4060 Ti 16 GB (AD106, sm_89)") | [UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 155 | 360<sup>8</sup> | 3,665 | 158 | [details](../runs/rtx4060ti/qwen3.6-35b-a3b.md) |
| [RTX&nbsp;3090](../gpus/rtx3090.md "RTX 3090, 24 GB (GA102, sm_86)") | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 291 | 595<sup>8</sup> | 6,399 | 291 | [details](../runs/rtx3090/qwen3.6-35b-a3b.md) |
| [A10](../gpus/a10.md "A10, 24 GB (GA102, sm_86)") | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 193 | 386<sup>8</sup> | 3,875 | 188 | [details](../runs/a10/qwen3.6-35b-a3b.md) |
| [A100](../gpus/a100-40gb.md "A100 40 GB (GA100, sm_80)") | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 298 | 773<sup>8</sup> | 6,311 | 310 | [details](../runs/a100-40gb/qwen3.6-35b-a3b.md) |
| [Tesla&nbsp;V100](../gpus/v100-32gb.md "Tesla V100 32 GB (GV100, sm_70)") | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 111 | 193<sup>8</sup> | 1,750 | 107 | [details](../runs/v100-32gb/qwen3.6-35b-a3b.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
