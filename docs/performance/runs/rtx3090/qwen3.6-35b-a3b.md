# Qwen3.6-35B-A3B on the RTX 3090

[RTX 3090](../../gpus/rtx3090.md) · [Qwen3.6-35B-A3B](../../models/qwen3.6-35b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 323 tok/s:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 631 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | dflash ×7 | auto | fast | 6,723 | 312 | 323 | 558 | 562 | 2026-11-08 |
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | off | auto | fast | 6,917 | 204 | 219 | 519 | 631 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | Prefill 24K | 5,429 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | Prefill 24K | 5,701 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | Decode @24K | 202 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | Decode @24K | 241 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | C=1 | 199 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | C=1 | 226 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | C=4 | 389 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | C=4 | 328 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | C=8 | 361 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | C=8 | 373 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | Prefill 24K | 6,387 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | Decode @24K | 149 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | C=1 | 160 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | C=4 | 344 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | C=8 | 356 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 3090: driver 615.71.09, CUDA None, power limit 390 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 --spec dflash --draft-tokens 7 
--max-context 32768 --max-concurrency 4 --spec none 
--max-context 32768 --max-concurrency 8 --spec dflash --draft-tokens 7 
--max-context 32768 --max-concurrency 8 --spec none 
```
