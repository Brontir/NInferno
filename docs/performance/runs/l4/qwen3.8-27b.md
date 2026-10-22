# Qwen3.8-27B on the L4

[L4](../../gpus/l4.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 32.8 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec dflash2 --draft-tokens 7
```

8 concurrent chats, 67.1 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | dflash2 ×7 | auto | auto | 750 | 33.4 | 32.8 | – | – | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | off | auto | auto | 816 | 13.8 | 14.3 | 42.6 | 67.1 | 2026-10-19 |

## Measurement

- NVIDIA L4: driver 615.71.09, CUDA None, power limit 72 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
