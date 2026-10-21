# Gemma 4 26B-A4B on the RTX 4090

[RTX 4090](../../gpus/rtx4090.md) · [Gemma 4 26B-A4B](../../models/gemma-4-26b-a4b-it.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 321 tok/s:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 5
```

8 concurrent chats, 892 tok/s over all streams:

```bash
ninferno-serve unsloth/gemma-4-26B-A4B-it-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 5 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_XL | mtp ×5 | auto | fast | 11,833 | 302 | 321 | 694 | 892 | 2026-10-19 |
| `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_XL | off | auto | fast | 11,901 | 188 | 206 | 490 | 635 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 4090: driver 580.95.05, CUDA 13.0, power limit 450 W.
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
