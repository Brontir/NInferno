# Qwen3.8-27B on the RTX 4070 Ti

[RTX 4070 Ti](../../gpus/rtx4070ti.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 104 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-IQ2_XXS --with-drafters --spec dflash2 --draft-tokens 7
```

8 concurrent chats, 241 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-IQ2_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ2_XXS | dflash2 ×7 | auto | fast | 1,688 | 112 | 104 | – | – | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ2_XXS | off | auto | fast | 1,713 | 52.5 | 59.2 | 187 | 241 | 2026-11-08 |

## Against llama.cpp

| Engine | Weights | Speculation | Measurement | tok/s | Measured |
|---|---|---|---|---:|---|
| llama.cpp b12198 | UD-IQ2_XXS | off | Prefill 24K | 1,456 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ2_XXS | off | Decode @24K | 44.7 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ2_XXS | off | C=1 | 51.1 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ2_XXS | off | C=4 | 122 | 2026-11-08 |
| llama.cpp b12198 | UD-IQ2_XXS | off | C=8‡ | 116 | 2026-11-08 |

‡ llama.cpp ran out of memory at the default context; 2048 tokens per slot

## Measurement

- NVIDIA GeForce RTX 4070 Ti: driver 615.71.09, power limit 285 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none 
--max-context 32768 --max-concurrency 4 --spec none 
--max-context 32768 --max-concurrency 8 --spec none 
```

llama.cpp command-line flags of the reference rows (after `llama-server -m <GGUF>`):

```
-ngl 99 -fa on -np 1 -c 32768 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 4 -c 32768 -b 2048 -ub 2048    # llama.cpp b12198
-ngl 99 -fa on -np 8 -c 16384 -b 2048 -ub 2048    # llama.cpp b12198
```
