# Muse Glimmer 30B on the RTX 5090

[RTX 5090](../../gpus/rtx5090.md) · [Muse Glimmer 30B](../../models/muse-glimmer-30b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 193 tok/s:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-Q4_K_XL --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 668 tok/s over all streams:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-Q4_K_XL --with-drafters --spec dflash --draft-tokens 7 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Muse-Glimmer-30B-GGUF` UD-Q4_K_XL | dflash ×7 | auto | fast | 4,340 | 195 | 193 | 432 | 668 | 2026-10-20 |
| `unsloth/Muse-Glimmer-30B-GGUF` UD-Q4_K_XL | off | auto | fast | 4,405 | 91.0 | 93.9 | 300 | 485 | 2026-10-20 |

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
