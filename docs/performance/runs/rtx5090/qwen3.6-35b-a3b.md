# Qwen3.6-35B-A3B on the RTX 5090

[RTX 5090](../../gpus/rtx5090.md) · [Qwen3.6-35B-A3B](../../models/qwen3.6-35b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 588 tok/s:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 1,526 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --with-drafters --spec dflash --draft-tokens 7 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | dflash ×7 | auto | auto | 18,032 | 593 | 588 | 1,249 | 1,526 | 2026-10-19 |
| `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M | off | auto | auto | 18,496 | 344 | 378 | 955 | 1,215 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 5090: driver 595.84, CUDA 13.2, power limit 575 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 
--max-context 32768 --max-concurrency 8 --spec none
```
