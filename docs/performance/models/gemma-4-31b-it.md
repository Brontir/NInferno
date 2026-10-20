# Gemma 4 31B

[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) · Gemma 4 dense, 31B · [all GPUs and models](../README.md)

Measured with [`unsloth/gemma-4-31B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF). Its fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;5090](../gpus/rtx5090.md "RTX 5090, 32 GB (GB202, sm_120)") | [UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF "unsloth/gemma-4-31B-it-GGUF") | 151 | 416<sup>8</sup> | 3,078 | 133 | [details](../runs/rtx5090/gemma-4-31b-it.md) |
| [RTX&nbsp;4090](../gpus/rtx4090.md "RTX 4090, 24 GB (AD102, sm_89)") | [UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF "unsloth/gemma-4-31B-it-GGUF") | 114 | 143<sup>4</sup> | 2,428 | 97.5 | [details](../runs/rtx4090/gemma-4-31b-it.md) |
| [L4](../gpus/l4.md "L4, 24 GB (AD104, sm_89)") | [UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF "unsloth/gemma-4-31B-it-GGUF") | 33.0 | – | 612 | 30.0 | [details](../runs/l4/gemma-4-31b-it.md) |
| [RTX&nbsp;3090](../gpus/rtx3090.md "RTX 3090, 24 GB (GA102, sm_86)") | [UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF "unsloth/gemma-4-31B-it-GGUF") | 89.8 | – | 1,051 | 79.8 | [details](../runs/rtx3090/gemma-4-31b-it.md) |
| [A10](../gpus/a10.md "A10, 24 GB (GA102, sm_86)") | [UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF "unsloth/gemma-4-31B-it-GGUF") | 55.6 | – | 648 | 49.5 | [details](../runs/a10/gemma-4-31b-it.md) |
| [A100](../gpus/a100-40gb.md "A100 40 GB (GA100, sm_80)") | [UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF "unsloth/gemma-4-31B-it-GGUF") | 106 | 247<sup>8</sup> | 1,010 | 97.4 | [details](../runs/a100-40gb/gemma-4-31b-it.md) |
| [Tesla&nbsp;V100](../gpus/v100-32gb.md "Tesla V100 32 GB (GV100, sm_70)") | [UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF "unsloth/gemma-4-31B-it-GGUF") | 57.6 | 73.9<sup>4</sup> | 266 | 43.1 | [details](../runs/v100-32gb/gemma-4-31b-it.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
