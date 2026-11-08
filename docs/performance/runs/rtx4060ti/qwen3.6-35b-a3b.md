# Qwen3.6-35B-A3B on the RTX 4060 Ti 16 GB

[RTX 4060 Ti 16 GB](../../gpus/rtx4060ti.md) · [Qwen3.6-35B-A3B](../../models/qwen3.6-35b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 180 tok/s:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-IQ3_XXS --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 366 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-IQ3_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-IQ3_XXS | dflash ×7 | auto | fast | 4,031 | 164 | 180 | 339 | 325 | 2026-11-08 |
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-IQ3_XXS | off | auto | fast | 4,112 | 96.2 | 114 | 290 | 366 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | UD-IQ3_XXS | off | Prefill 24K | 3,478 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ3_XXS | off | Decode @24K | 73.9 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ3_XXS | off | C=1 | 84.1 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ3_XXS | off | C=4 | 212 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ3_XXS | off | C=8 | 223 | 2026-11-08 |

## Measurement

- NVIDIA GeForce RTX 4060 Ti: driver 615.71.09, power limit 165 W.
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
-ngl 99 -fa on -np 4 -c 131072 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 65536 -b 2048 -ub 2048    # llama.cpp b12198
```
