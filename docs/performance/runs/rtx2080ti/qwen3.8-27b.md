# Qwen3.8-27B on the RTX 2080 Ti

[RTX 2080 Ti](../../gpus/rtx2080ti.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 52.1 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-IQ2_XXS --with-drafters --spec dflash2 --draft-tokens 7
```

8 concurrent chats, 120 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-IQ2_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ2_XXS | dflash2 ×7 | auto | fast | 644 | 56.5 | 52.1 | – | – | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ2_XXS | off | auto | fast | 660 | 40.4 | 43.7 | 102 | 120 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 2080 Ti: driver 580.178.04, CUDA 13.0, power limit 240 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
