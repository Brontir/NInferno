# Qwen3.6-35B-A3B on the RTX 4060 Ti 16 GB

[RTX 4060 Ti 16 GB](../../gpus/rtx4060ti.md) · [Qwen3.6-35B-A3B](../../models/qwen3.6-35b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 155 tok/s:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-IQ3_XXS --with-drafters --spec dflash --draft-tokens 11
```

8 concurrent chats, 360 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-IQ3_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-IQ3_XXS | dflash ×11 | auto | auto | 3,665 | 158 | 155 | 254 | – | 2026-10-19 |
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-IQ3_XXS | off | auto | auto | 3,730 | 103 | 112 | 287 | 360 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 4060 Ti: driver 570.211.01, CUDA 12.8, power limit 165 W.
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
