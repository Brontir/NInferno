# Gemma 4 26B-A4B on the RTX 5070 Ti

[RTX 5070 Ti](../../gpus/rtx5070ti.md) · [Gemma 4 26B-A4B](../../models/gemma-4-26b-a4b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 359 tok/s:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-IQ3_XXS --with-drafters --spec mtp --draft-tokens 4
```

8 concurrent chats, 906 tok/s over all streams:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-IQ3_XXS --with-drafters --spec mtp --draft-tokens 4 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-IQ3_XXS | mtp ×4 | auto | fast | 6,978 | 340 | 359 | 711 | 906 | 2026-11-08 |
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-IQ3_XXS | off | auto | fast | 6,982 | 231 | 248 | 535 | 637 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | UD-IQ3_XXS | off | Prefill 24K | 6,264 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ3_XXS | off | Decode @24K | 153 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ3_XXS | off | C=1 | 185 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ3_XXS | off | C=4 | 405 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ3_XXS | off | C=8 | 0.0 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 5070 Ti: driver 615.71.09, CUDA None, power limit 300 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none 
--max-context 32768 --max-concurrency 8 
--max-context 32768 --max-concurrency 8 --spec none 
```
