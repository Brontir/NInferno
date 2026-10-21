# Gemma 4 26B-A4B on the RTX 4060 Ti 16 GB

[RTX 4060 Ti 16 GB](../../gpus/rtx4060ti.md) · [Gemma 4 26B-A4B](../../models/gemma-4-26b-a4b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 127 tok/s:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-IQ3_XXS --with-drafters --spec mtp --draft-tokens 5
```

4 concurrent chats, 294 tok/s over all streams:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-IQ3_XXS --with-drafters --spec mtp --draft-tokens 5 --max-concurrency 4
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-IQ3_XXS | mtp ×5 | auto | fast | 3,158 | 119 | 127 | 294 | – | 2026-10-19 |
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-IQ3_XXS | off | auto | fast | 3,161 | 83.3 | 92.5 | 211 | 263 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 4060 Ti: driver 570.211.01, CUDA 12.8, power limit 165 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
