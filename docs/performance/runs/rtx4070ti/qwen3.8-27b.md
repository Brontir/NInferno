# Qwen3.8-27B on the RTX 4070 Ti

[RTX 4070 Ti](../../gpus/rtx4070ti.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 104 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-IQ2_XXS --with-drafters --spec dflash2 --draft-tokens 7
```

8 concurrent chats, 216 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-IQ2_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ2_XXS | dflash2 ×7 | auto | 1,688 | 112 | 104 | – | – | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-IQ2_XXS | off | auto | 1,709 | 51.6 | 55.8 | 179 | 216 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 4070 Ti: driver 615.71.09, CUDA None, power limit 285 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
