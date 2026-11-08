# Qwen3.6-35B-A3B on the RTX 5090

[RTX 5090](../../gpus/rtx5090.md) · [Qwen3.6-35B-A3B](../../models/qwen3.6-35b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 680 tok/s:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 1,412 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --with-drafters --spec dflash --draft-tokens 7 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | dflash ×7 | auto | fast | 18,950 | 633 | 680 | 1,177 | 1,412 | 2026-11-08 |
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | off | auto | fast | 19,395 | 381 | 410 | 1,053 | 1,361 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | Prefill 24K | 11,902 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | Prefill 24K | 11,904 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | Decode @24K | 387 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | Decode @24K | 368 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | C=1 | 388 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | C=1 | 351 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | C=4 | 591 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | C=4 | 605 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×2 | C=8 | 687 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | mtp ×3 | C=8 | 679 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | Prefill 24K | 15,187 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | Decode @24K | 246 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | C=1 | 264 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | C=4 | 548 | 2026-11-08 |
| llama.cpp b12198 | UD-Q4_K_M | off | C=8 | 607 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 5090: driver 615.71.09, power limit 575 W.
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

llama.cpp command-line flags of the reference rows (after `llama-server -m <GGUF>`):

```
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 2    # llama.cpp b12198
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 131072 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 131072 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 32768 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 2    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 262144 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 262144 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 65536 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 2    # llama.cpp b12198
```
