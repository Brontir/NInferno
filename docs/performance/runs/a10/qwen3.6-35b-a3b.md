# Qwen3.6-35B-A3B on the A10

[A10](../../gpus/a10.md) · [Qwen3.6-35B-A3B](../../models/qwen3.6-35b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 193 tok/s:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 386 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | dflash ×7 | auto | fast | 3,875 | 188 | 193 | – | – | 2026-10-19 |
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | off | auto | fast | 3,941 | 123 | 130 | 308 | 386 | 2026-10-19 |

## Measurement

- NVIDIA A10: driver 570.144, CUDA 12.8, power limit 150 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
