# Gemma 4 31B on the RTX 3090

[RTX 3090](../../gpus/rtx3090.md) · [Gemma 4 31B](../../models/gemma-4-31b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 89.8 tok/s:

```bash
ninferno-serve unsloth/gemma-4-31B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 6
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | Measured |
|---|---|---|---|---:|---:|---:|---|
| `unsloth/gemma-4-31B-it-GGUF` UD-Q4_K_XL | mtp ×6 | auto | auto | 1,051 | 79.8 | 89.8 | 2026-10-19 |
| `unsloth/gemma-4-31B-it-GGUF` UD-Q4_K_XL | off | auto | auto | 1,049 | 34.8 | 38.2 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 3090: driver 580.159.03, CUDA 13.0, power limit 350 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
```
