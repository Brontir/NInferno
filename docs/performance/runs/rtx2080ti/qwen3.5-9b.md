# Qwen3.5-9B on the RTX 2080 Ti

[RTX 2080 Ti](../../gpus/rtx2080ti.md) · [Qwen3.5-9B](../../models/qwen3.5-9b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 165 tok/s:

```bash
ninferno-serve unsloth/Qwen3.5-9B-GGUF --quant Q4_K_M --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 397 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.5-9B-GGUF --quant Q4_K_M --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.5-9B-GGUF` Q4_K_M | dflash ×7 | auto | fast | 2,127 | 151 | 165 | 279 | 189‡ | 2026-11-08 |
| `unsloth/Qwen3.5-9B-GGUF` Q4_K_M | off | auto | fast | 2,363 | 87.0 | 95.0 | 279 | 397 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | Q4_K_M | mtp ×4 | Prefill 24K | 1,915 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | Decode @24K | 111 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | C=1 | 105 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | C=4 | 245 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | mtp ×4 | C=8 | 258 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | Prefill 24K | 2,134 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | Decode @24K | 73.6 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | C=1 | 80.7 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | C=4 | 196 | 2026-11-08 |
| llama.cpp b12198 | Q4_K_M | off | C=8 | 199 | 2026-11-08 |

‡ explicit --spec dflash bypasses the load table; on this 11 GB card eight contexts do not fit beside the drafter (KV pool 88,704 of 262,144 tokens), so requests wait for space; --spec auto serves plain here

## Measurement

- NVIDIA GeForce RTX 2080 Ti: driver 615.71.09, power limit 260 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none 
--max-context 32768 --max-concurrency 8 --spec dflash --draft-tokens 7 
--max-context 32768 --max-concurrency 8 --spec none 
```

llama.cpp command-line flags of the reference rows (after `llama-server -m <GGUF>`):

```
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 32768 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 32768 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 65536 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 65536 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4    # llama.cpp b12198
```
