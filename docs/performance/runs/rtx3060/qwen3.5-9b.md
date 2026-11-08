# Qwen3.5-9B on the RTX 3060

[RTX 3060](../../gpus/rtx3060.md) · [Qwen3.5-9B](../../models/qwen3.5-9b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 120 tok/s:

```bash
ninferno-serve unsloth/Qwen3.5-9B-GGUF --quant Q4_K_M --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 264 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.5-9B-GGUF --quant Q4_K_M --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.5-9B-GGUF` Q4_K_M | dflash ×7 | auto | fast | 1,934 | 118 | 120 | 179 | – | 2026-11-08 |
| `unsloth/Qwen3.5-9B-GGUF` Q4_K_M | off | auto | fast | 1,999 | 57.3 | 60.9 | 179 | 264 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | Q4_K_M | mtp ×3 | Prefill 24K | 1,693 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | Prefill 24K | 1,694 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×3 | Decode @24K | 84.5 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | Decode @24K | 85.9 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×3 | C=1 | 83.6 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | C=1 | 79.2 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×3 | C=4 | 220 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | C=4 | 210 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×3 | C=8 | 235 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | C=8 | 248 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | Prefill 24K | 1,849 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | Decode @24K | 50.6 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | C=1 | 56.4 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | C=4 | 150 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | C=8 | 154 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 3060: driver 615.71.09, power limit 170 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none 
--max-context 32768 --max-concurrency 8 
```

llama.cpp command-line flags of the reference rows (after `llama-server -m <GGUF>`):

```
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3    # llama.cpp b12198
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 32768 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 32768 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 32768 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 65536 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 65536 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 65536 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4    # llama.cpp b12198
```
