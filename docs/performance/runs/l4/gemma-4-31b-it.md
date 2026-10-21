# Gemma 4 31B on the L4

[L4](../../gpus/l4.md) · [Gemma 4 31B](../../models/gemma-4-31b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 33.0 tok/s:

```bash
ninferno-serve unsloth/gemma-4-31B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 10
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | Measured |
|---|---|---|---|---:|---:|---:|---|
| `unsloth/gemma-4-31B-it-GGUF` UD-Q4_K_XL | mtp ×10 | auto | fast | 612 | 30.0 | 33.0 | 2026-10-20 |
| `unsloth/gemma-4-31B-it-GGUF` UD-Q4_K_XL | off | auto | fast | 611 | 11.9 | 12.6 | 2026-10-20 |

## Measurement

- NVIDIA L4: driver 615.71.09, CUDA None, power limit 72 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
```
