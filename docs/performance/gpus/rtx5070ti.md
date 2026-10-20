# RTX 5070 Ti

GB203, sm_120, 16 GB · [all GPUs and models](../README.md)

Each model on this GPU, tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| Model | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [Qwen3.8-27B](../models/qwen3.8-27b.md) | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 120 | 261<sup>8</sup> | 2,051 | 122 | [details](../runs/rtx5070ti/qwen3.8-27b.md) |
| [Qwen3.6-35B-A3B](../models/qwen3.6-35b-a3b.md) | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 365 | 755<sup>8</sup> | 7,224 | 361 | [details](../runs/rtx5070ti/qwen3.6-35b-a3b.md) |
| [Gemma&nbsp;4&nbsp;26B-A4B](../models/gemma-4-26b-a4b-it.md) | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 301 | 638<sup>4</sup> | 6,516 | 283 | [details](../runs/rtx5070ti/gemma-4-26b-a4b-it.md) |
| [gpt-oss-20b](../models/gpt-oss-20b.md) | [`MXFP4`](https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b") | 225 | 568<sup>8</sup> | 10,696 | 177 | [details](../runs/rtx5070ti/gpt-oss-20b.md) |
| [Muse&nbsp;Glimmer&nbsp;30B](../models/muse-glimmer-30b.md) | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 101 | 268<sup>8</sup> | 2,230 | 103 | [details](../runs/rtx5070ti/muse-glimmer-30b.md) |

Weights from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF), [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b), [`unsloth/Muse-Glimmer-30B-GGUF`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF).

<sub>Measured 2026-10-19 to 2026-10-20 · NInferno 0.1.0 · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · [methodology](../methodology.md)</sub>
