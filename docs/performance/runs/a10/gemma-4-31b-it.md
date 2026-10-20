# Gemma 4 31B on the A10

[A10](../../gpus/a10.md) · [Gemma 4 31B](../../models/gemma-4-31b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 55.6 tok/s:

```bash
ninferno-serve unsloth/gemma-4-31B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 6
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | Measured |
|---|---|---|---:|---:|---:|---|
| `unsloth/gemma-4-31B-it-GGUF` UD-Q4_K_XL | mtp ×6 | auto | 648 | 49.5 | 55.6 | 2026-10-19 |
| `unsloth/gemma-4-31B-it-GGUF` UD-Q4_K_XL | off | auto | 666 | 20.8 | 22.9 | 2026-10-19 |

## Measurement

- NVIDIA A10: driver 570.144, CUDA 12.8, power limit 150 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
```
