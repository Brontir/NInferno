# Qwen3.8-27B on the A100 40 GB

[A100 40 GB](../../gpus/a100-40gb.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 87.8 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec dflash2 --draft-tokens 7
```

8 concurrent chats, 211 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec dflash2 --draft-tokens 7 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | dflash2 ×7 | auto | fast | 1,268 | 89.6 | 87.8 | 180 | 211 | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | off | auto | fast | 1,249 | 45.2 | 46.3 | 130 | 185 | 2026-10-20 |

## Measurement

- NVIDIA A100-SXM4-40GB: driver 595.99.02, CUDA 13.2, power limit 300 W.
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
