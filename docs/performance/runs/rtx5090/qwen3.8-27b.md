# Qwen3.8-27B on the RTX 5090

[RTX 5090](../../gpus/rtx5090.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 190 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 5
```

8 concurrent chats, 593 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 5 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | mtp ×5 | auto | fast | 4,318 | 202 | 190 | 437 | 593 | 2026-11-08 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | off | auto | fast | 4,328 | 85.9 | 86.9 | 264 | 410 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | UD-Q4_K_XL | off | Prefill 24K | 3,718 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_XL | off | Decode @24K | 73.1 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_XL | off | C=1 | 76.5 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_XL | off | C=4 | 178 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_XL | off | C=8 | 195 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 5090: driver 615.71.09, power limit 575 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 --spec mtp --draft-tokens 5 
--max-context 32768 --max-concurrency 4 --spec none 
--max-context 32768 --max-concurrency 8 --spec mtp --draft-tokens 5 
--max-context 32768 --max-concurrency 8 --spec none 
```

llama.cpp command-line flags of the reference rows (after `llama-server -m <GGUF>`):

```
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 131072 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 65536 -b 2048 -ub 2048    # llama.cpp b12198
```
