# Muse Glimmer 30B on the A10

[A10](../../gpus/a10.md) · [Muse Glimmer 30B](../../models/muse-glimmer-30b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 53.2 tok/s:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-Q4_K_XL --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 136 tok/s over all streams:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-Q4_K_XL --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Muse-Glimmer-30B-GGUF` UD-Q4_K_XL | dflash ×7 | auto | 990 | 57.1 | 53.2 | 131 | 121 | 2026-10-20 |
| `unsloth/Muse-Glimmer-30B-GGUF` UD-Q4_K_XL | off | auto | 1,018 | 28.7 | 29.3 | 84.8 | 136 | 2026-10-20 |

## Measurement

- NVIDIA A10: driver 570.144, CUDA 12.8, power limit 150 W.
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
