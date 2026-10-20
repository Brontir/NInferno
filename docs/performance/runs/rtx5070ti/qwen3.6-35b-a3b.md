# Qwen3.6-35B-A3B on the RTX 5070 Ti

[RTX 5070 Ti](../../gpus/rtx5070ti.md) · [Qwen3.6-35B-A3B](../../models/qwen3.6-35b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 365 tok/s:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-IQ3_XXS --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 755 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-IQ3_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-IQ3_XXS | dflash ×7 | auto | 7,224 | 361 | 365 | 739 | – | 2026-10-19 |
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-IQ3_XXS | off | auto | 7,355 | 248 | 266 | 604 | 755 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 5070 Ti: driver 615.71.09, CUDA None, power limit 300 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
