# RTX 5070 Ti

GB203, sm_120, 16 GB · [all GPUs and models](../README.md)

Each model on this GPU, tokens per second, each column at its best setting: one stream, all streams, prefill of a 24K-token prompt, one stream after it.

| Model | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [Qwen3.8-27B](../models/qwen3.8-27b.md) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-IQ3_XXS | 120 | 261 (C=8) | 2,051 | 122 | [all settings](../runs/rtx5070ti/qwen3.8-27b.md) |
| [Qwen3.6-35B-A3B](../models/qwen3.6-35b-a3b.md) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) UD-IQ3_XXS | 365 | 755 (C=8) | 7,224 | 361 | [all settings](../runs/rtx5070ti/qwen3.6-35b-a3b.md) |
| [Gemma 4 26B-A4B](../models/gemma-4-26b-a4b-it.md) | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-IQ3_XXS | 301 | 638 (C=4) | 6,516 | 283 | [all settings](../runs/rtx5070ti/gemma-4-26b-a4b-it.md) |
| [gpt-oss-20b](../models/gpt-oss-20b.md) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 225 | 568 (C=8) | 10,696 | 177 | [all settings](../runs/rtx5070ti/gpt-oss-20b.md) |

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · [methodology](../methodology.md)</sub>
