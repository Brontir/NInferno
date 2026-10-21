# Qwen3.8-27B on the Tesla V100 32 GB

[Tesla V100 32 GB](../../gpus/v100-32gb.md) · [Qwen3.8-27B](../../models/qwen3.8-27b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 33.0 tok/s:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --spec none
```

4 concurrent chats, 82.9 tok/s over all streams:

```bash
ninferno-serve unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --spec none --max-concurrency 4
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | dflash2 ×5 | auto | fast | 673 | 15.2 | 15.8 | 81.2 | 38.9 | 2026-10-19 |
| `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL | off | auto | fast | 856 | 33.2 | 33.0 | 82.9 | 38.9 | 2026-10-19 |

## Measurement

- Tesla V100-SXM2-32GB: driver 580.159.03, CUDA 13.0, power limit 250 W.
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
