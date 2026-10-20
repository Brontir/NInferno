# Qwen3.8-27B on the RTX 4090

[RTX 4090](../../gpus/rtx4090.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 122 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec dflash2 --draft-tokens 7
```

8 concurrent chats, 366 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 3 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | dflash2 ×7 | auto | 3,345 | 127 | 122 | 277 | – | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | mtp ×3 | auto | – | – | – | – | 366 | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | off | auto | 3,376 | 48.7 | 50.5 | 156 | 234 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 4090: driver 580.95.05, CUDA 13.0, power limit 450 W.
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
