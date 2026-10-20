# RTX 4060 Ti 16 GB

AD106, sm_89, 16 GB · [all GPUs and models](../README.md)

Each model on this GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| Model | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [Qwen3.8-27B](../models/qwen3.8-27b.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-IQ3_XXS | 49.0 | 115 (C=8) | 869 | 47.5 | [all settings](../runs/rtx4060ti/qwen3.8-27b.md) |
| [Qwen3.6-35B-A3B](../models/qwen3.6-35b-a3b.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-IQ3_XXS | 155 | 360 (C=8) | 3,665 | 158 | [all settings](../runs/rtx4060ti/qwen3.6-35b-a3b.md) |
| [Gemma 4 26B-A4B](../models/gemma-4-26b-a4b-it.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-IQ3_XXS | 127 | 294 (C=4) | 3,158 | 119 | [all settings](../runs/rtx4060ti/gemma-4-26b-a4b-it.md) |
| [gpt-oss-20b](../models/gpt-oss-20b.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 86.6 | 208 (C=8) | 4,279 | 66.0 | [all settings](../runs/rtx4060ti/gpt-oss-20b.md) |

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · [methodology](../methodology.md)</sub>
