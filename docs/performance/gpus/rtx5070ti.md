# RTX 5070 Ti

GB203, sm_120, 16 GB · [all GPUs and models](../README.md)

Each model on this GPU, tokens per second, each column at its best setting: one stream, all streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.

| Model | Weights | One stream | All streams | Prefill 24K | Decode @24K | |
|---|---|---:|---:|---:|---:|---|
| [Qwen3.8-27B](../models/qwen3.8-27b.md) | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF "unsloth/Qwen3.8-27B-GGUF") | 120 | 261<sup>8</sup> | 2,070 | 122 | [details](../runs/rtx5070ti/qwen3.8-27b.md) |
| [Qwen3.6-35B-A3B](../models/qwen3.6-35b-a3b.md) | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF "unsloth/Qwen3.6-35B-A3B-GGUF") | 399 | 812<sup>8</sup> | 7,581 | 383 | [details](../runs/rtx5070ti/qwen3.6-35b-a3b.md) |
| [Gemma&nbsp;4&nbsp;26B-A4B](../models/gemma-4-26b-a4b-it.md) | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF "unsloth/gemma-4-26B-A4B-it-GGUF") | 359 | 906<sup>8</sup> | 6,982 | 340 | [details](../runs/rtx5070ti/gemma-4-26b-a4b-it.md) |
| [gpt-oss-20b](../models/gpt-oss-20b.md) | [`original`](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF "ggml-org/gpt-oss-20b-GGUF") | 299 | 654<sup>8</sup> | 11,179 | 239 | [details](../runs/rtx5070ti/gpt-oss-20b.md) |
| [Muse&nbsp;Glimmer&nbsp;30B](../models/muse-glimmer-30b.md) | [`UD-IQ3_XXS`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF "unsloth/Muse-Glimmer-30B-GGUF") | 101 | 268<sup>8</sup> | 2,275 | 103 | [details](../runs/rtx5070ti/muse-glimmer-30b.md) |

Weights from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF), [`ggml-org/gpt-oss-20b-GGUF`](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF), [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b), [`unsloth/Muse-Glimmer-30B-GGUF`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF).

<sub>Measured 2026-10-19 to 2026-11-08 · NInferno 0.1.0 · NVIDIA GeForce RTX 5070 Ti, driver 615.71.09, 300 W · [methodology](../methodology.md)</sub>
