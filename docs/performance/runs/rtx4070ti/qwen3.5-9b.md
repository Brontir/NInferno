# Qwen3.5-9B on the RTX 4070 Ti

[RTX 4070 Ti](../../gpus/rtx4070ti.md) · [Qwen3.5-9B](../../models/qwen3.5-9b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 176 tok/s:

```bash
ninferno-serve unsloth/Qwen3.5-9B-GGUF --quant Q4_K_M --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 398 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.5-9B-GGUF --quant Q4_K_M --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.5-9B-GGUF` Q4_K_M | dflash ×7 | auto | fast | 5,544 | 187 | 176 | – | – | 2026-11-08 |
| `unsloth/Qwen3.5-9B-GGUF` Q4_K_M | off | auto | fast | 5,718 | 81.4 | 86.6 | 280 | 398 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | Q4_K_M | mtp ×4 | Prefill 24K | 4,250 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | Decode @24K | 133 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | C=1 | 130 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | C=4 | 351 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | C=8 | 421 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | Prefill 24K | 4,864 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | Decode @24K | 70.8 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | C=1 | 79.2 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | C=4 | 257 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | C=8 | 296 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 4070 Ti: driver 615.71.09, CUDA None, power limit 285 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 8 --spec none 
```
