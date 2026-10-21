# Muse Glimmer 30B on the Tesla V100 32 GB

[Tesla V100 32 GB](../../gpus/v100-32gb.md) · [Muse Glimmer 30B](../../models/muse-glimmer-30b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 38.9 tok/s:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-Q4_K_XL --spec none
```

8 concurrent chats, 115 tok/s over all streams:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-Q4_K_XL --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Muse-Glimmer-30B-GGUF` UD-Q4_K_XL | dflash ×7 | auto | fast | 551 | 29.0 | 35.8 | 54.7 | 112 | 2026-10-20 |
| `unsloth/Muse-Glimmer-30B-GGUF` UD-Q4_K_XL | off | auto | fast | 552 | 29.9 | 38.9 | 97.1 | 115 | 2026-10-20 |

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
