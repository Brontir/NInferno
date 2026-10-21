# Qwen3.8-27B on the RTX 3090

[RTX 3090](../../gpus/rtx3090.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 94.6 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec dflash2 --draft-tokens 7
```

8 concurrent chats, 264 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 3 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | dflash2 ×7 | auto | fast | 1,557 | 91.8 | 94.6 | 189 | – | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | mtp ×3 | auto | fast | – | – | – | – | 264 | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | off | auto | fast | 1,572 | 41.1 | 42.3 | 120 | 177 | 2026-10-19 |

## Measurement

- NVIDIA GeForce RTX 3090: driver 580.159.03, CUDA 13.0, power limit 350 W.
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
