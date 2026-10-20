# RTX 4060 Ti 16 GB

AD106, sm_89, 16 GB · [all GPUs and models](../README.md)

Each model on this GPU, tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| Model | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [Qwen3.8-27B](../models/qwen3.8-27b.md) | [UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 49.0 | 115<sup>8</sup> | 869 | 47.5 | [details](../runs/rtx4060ti/qwen3.8-27b.md) |
| [Qwen3.6-35B-A3B](../models/qwen3.6-35b-a3b.md) | [UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 155 | 360<sup>8</sup> | 3,665 | 158 | [details](../runs/rtx4060ti/qwen3.6-35b-a3b.md) |
| [Gemma&nbsp;4&nbsp;26B-A4B](../models/gemma-4-26b-a4b-it.md) | [UD-IQ3_XXS](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 127 | 294<sup>4</sup> | 3,158 | 119 | [details](../runs/rtx4060ti/gemma-4-26b-a4b-it.md) |
| [gpt-oss-20b](../models/gpt-oss-20b.md) | [original](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 86.6 | 208<sup>8</sup> | 4,279 | 66.0 | [details](../runs/rtx4060ti/gpt-oss-20b.md) |

<sub>Measured 2026-10-19 · NInferno 0.1.0 · NVIDIA GeForce RTX 4060 Ti, driver 570.211.01, 165 W · [methodology](../methodology.md)</sub>
