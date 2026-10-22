# Gemma 4 26B-A4B on the RTX 5070 Ti

[RTX 5070 Ti](../../gpus/rtx5070ti.md) · [Gemma 4 26B-A4B](../../models/gemma-4-26b-a4b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 301 tok/s:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-IQ3_XXS --with-drafters --spec mtp --draft-tokens 6
```

4 concurrent chats, 638 tok/s over all streams:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-IQ3_XXS --with-drafters --spec mtp --draft-tokens 6 --max-concurrency 4
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-IQ3_XXS | mtp ×6 | auto | auto | 6,516 | 283 | 301 | 638 | – | 2026-10-19 |
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-IQ3_XXS | off | auto | auto | 6,508 | 220 | 240 | 485 | 604 | 2026-10-19 |

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
--max-context 32768 --max-concurrency 8 --spec none
```
