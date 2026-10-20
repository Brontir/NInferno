# gpt-oss-20b

[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) · MoE decoder, 21B total, 3.6B active · [all GPUs and models](../README.md)

The model on each GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX 5090](../gpus/rtx5090.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 409 | 1,281 (C=8) | 20,164 | 317 | [all settings](../runs/rtx5090/gpt-oss-20b.md) |
| [RTX 5070 Ti](../gpus/rtx5070ti.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 225 | 568 (C=8) | 10,696 | 177 | [all settings](../runs/rtx5070ti/gpt-oss-20b.md) |
| [RTX 4090](../gpus/rtx4090.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 261 | 865 (C=8) | 14,131 | 197 | [all settings](../runs/rtx4090/gpt-oss-20b.md) |
| [L4](../gpus/l4.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 70.8 | 174 (C=8) | 3,735 | 56.0 | [all settings](../runs/l4/gpt-oss-20b.md) |
| [RTX 4060 Ti 16 GB](../gpus/rtx4060ti.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 86.6 | 208 (C=8) | 4,279 | 66.0 | [all settings](../runs/rtx4060ti/gpt-oss-20b.md) |
| [RTX 3090](../gpus/rtx3090.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 201 | 453 (C=8) | 6,476 | 160 | [all settings](../runs/rtx3090/gpt-oss-20b.md) |
| [A10](../gpus/a10.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 120 | 274 (C=8) | 4,231 | 94.4 | [all settings](../runs/a10/gpt-oss-20b.md) |
| [A100 40 GB](../gpus/a100-40gb.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 225 | 627 (C=8) | 7,081 | 186 | [all settings](../runs/a100-40gb/gpt-oss-20b.md) |
| [Tesla V100 32 GB](../gpus/v100-32gb.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 93.6 | 188 (C=8) | 1,000 | 68.2 | [all settings](../runs/v100-32gb/gpt-oss-20b.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
