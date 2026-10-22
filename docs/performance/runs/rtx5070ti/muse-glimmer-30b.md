# Muse Glimmer 30B on the RTX 5070 Ti

[RTX 5070 Ti](../../gpus/rtx5070ti.md) · [Muse Glimmer 30B](../../models/muse-glimmer-30b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 101 tok/s:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-IQ3_XXS --with-drafters --spec dflash --draft-tokens 3
```

8 concurrent chats, 268 tok/s over all streams:

```bash
ninferno-serve unsloth/Muse-Glimmer-30B-GGUF --quant UD-IQ3_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/Muse-Glimmer-30B-GGUF` UD-IQ3_XXS | dflash ×3 | auto | auto | 2,230 | 103 | 101 | – | – | 2026-10-20 |
| `unsloth/Muse-Glimmer-30B-GGUF` UD-IQ3_XXS | off | auto | auto | 2,275 | 57.3 | 58.0 | 182 | 268 | 2026-10-20 |

## Measurement

- NVIDIA GeForce RTX 5070 Ti: driver 615.71.09, CUDA None, power limit 300 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
