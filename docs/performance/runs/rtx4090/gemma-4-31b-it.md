# Gemma 4 31B on the RTX 4090

[RTX 4090](../../gpus/rtx4090.md) · [Gemma 4 31B](../../models/gemma-4-31b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 114 tok/s:

```bash
ninferno-serve unsloth/gemma-4-31B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 10
```

4 concurrent chats, 143 tok/s over all streams:

```bash
ninferno-serve unsloth/gemma-4-31B-it-GGUF --quant UD-Q4_K_XL --spec none --max-concurrency 4
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | Measured |
|---|---|---|---|---:|---:|---:|---:|---|
| `unsloth/gemma-4-31B-it-GGUF` UD-Q4_K_XL | mtp ×10 | auto | fast | 2,428 | 97.5 | 114 | – | 2026-10-19 |
| `unsloth/gemma-4-31B-it-GGUF` UD-Q4_K_XL | off | auto | fast | 2,428 | 42.5 | 45.5 | 143 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 4090: driver 580.95.05, CUDA 13.0, power limit 450 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
```
