# Qwen3-8B on the RTX 4070 Ti

[RTX 4070 Ti](../../gpus/rtx4070ti.md) · [Qwen3-8B](../../models/qwen3-8b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 93.2 tok/s:

```bash
ninferno-serve Qwen/Qwen3-8B-GGUF --quant Q4_K_M --spec none
```

8 concurrent chats, 500 tok/s over all streams:

```bash
ninferno-serve Qwen/Qwen3-8B-GGUF --quant Q4_K_M --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3-8B-GGUF` Q4_K_M | off | auto | auto | 3,751 | 69.7 | 93.2 | 334 | 500 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 4070 Ti: driver 615.71.09, CUDA None, power limit 285 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
