# gpt-oss-20b

[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) · MoE decoder, 21B total, 3.6B active · [all GPUs and models](../README.md)

Measured with [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b). Its fine-tunes and other checkpoints of the same architecture run the same way.

Tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| GPU | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [RTX&nbsp;5090](../gpus/rtx5090.md "RTX 5090, 32 GB (GB202, sm_120)") | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 409 | 1,281<sup>8</sup> | 20,164 | 317 | [details](../runs/rtx5090/gpt-oss-20b.md) |
| [RTX&nbsp;5070&nbsp;Ti](../gpus/rtx5070ti.md "RTX 5070 Ti, 16 GB (GB203, sm_120)") | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 225 | 568<sup>8</sup> | 10,696 | 177 | [details](../runs/rtx5070ti/gpt-oss-20b.md) |
| [RTX&nbsp;4090](../gpus/rtx4090.md "RTX 4090, 24 GB (AD102, sm_89)") | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 261 | 865<sup>8</sup> | 14,131 | 197 | [details](../runs/rtx4090/gpt-oss-20b.md) |
| [L4](../gpus/l4.md "L4, 24 GB (AD104, sm_89)") | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 70.8 | 174<sup>8</sup> | 3,735 | 56.0 | [details](../runs/l4/gpt-oss-20b.md) |
| [RTX&nbsp;4060&nbsp;Ti](../gpus/rtx4060ti.md "RTX 4060 Ti 16 GB (AD106, sm_89)") | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 86.6 | 208<sup>8</sup> | 4,279 | 66.0 | [details](../runs/rtx4060ti/gpt-oss-20b.md) |
| [RTX&nbsp;3090](../gpus/rtx3090.md "RTX 3090, 24 GB (GA102, sm_86)") | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 201 | 453<sup>8</sup> | 6,476 | 160 | [details](../runs/rtx3090/gpt-oss-20b.md) |
| [A10](../gpus/a10.md "A10, 24 GB (GA102, sm_86)") | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 120 | 274<sup>8</sup> | 4,231 | 94.4 | [details](../runs/a10/gpt-oss-20b.md) |
| [A100](../gpus/a100-40gb.md "A100 40 GB (GA100, sm_80)") | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 225 | 627<sup>8</sup> | 7,081 | 186 | [details](../runs/a100-40gb/gpt-oss-20b.md) |
| [Tesla&nbsp;V100](../gpus/v100-32gb.md "Tesla V100 32 GB (GV100, sm_70)") | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 93.6 | 188<sup>8</sup> | 1,000 | 68.2 | [details](../runs/v100-32gb/gpt-oss-20b.md) |

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA A10, driver 570.144, 150 W · NVIDIA A100-SXM4-40GB, driver 595.99.02, 300 W · NVIDIA GeForce RTX 3090, driver 580.159.03, 350 W · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · NVIDIA GeForce RTX 4090, driver 580.95.05, 450 W · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · NVIDIA GeForce RTX 5090, driver 595.84, 575 W · NVIDIA L4, driver 615.71.09, 72 W · Tesla V100-SXM2-32GB, driver 580.159.03, 250 W · [methodology](../methodology.md)</sub>
