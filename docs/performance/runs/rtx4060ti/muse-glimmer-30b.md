# Muse Glimmer 30B on the RTX 4060 Ti 16 GB

[RTX 4060 Ti 16 GB](../../gpus/rtx4060ti.md) · [Muse Glimmer 30B](../../models/muse-glimmer-30b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 45.9 tok/s:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-IQ3_XXS --with-drafters --spec dflash --draft-tokens 7
```

8 concurrent chats, 120 tok/s over all streams:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-IQ3_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Muse-Glimmer-30B-GGUF` UD-IQ3_XXS | dflash ×7 | auto | fast | 937 | 44.7 | 45.9 | – | – | 2026-10-20 |
| `unsloth/Muse-Glimmer-30B-GGUF` UD-IQ3_XXS | off | auto | fast | 957 | 21.1 | 21.3 | 77.5 | 120 | 2026-10-20 |

## Measurement

- NVIDIA GeForce RTX 4060 Ti: driver 570.211.01, CUDA 12.8, power limit 165 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
