# Qwen3.5-4B on the RTX 2080 Ti

[RTX 2080 Ti](../../gpus/rtx2080ti.md) · [Qwen3.5-4B](../../models/qwen3.5-4b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 145 tok/s:

```bash
ninferno-serve unsloth/Qwen3.5-4B-GGUF --quant Q4_K_M --spec none
```

8 concurrent chats, 506 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.5-4B-GGUF --quant Q4_K_M --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.5-4B-GGUF` Q4_K_M | off | auto | 3,581 | 124 | 145 | 385 | 506 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 2080 Ti: driver 580.178.04, CUDA 13.0, power limit 240 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
