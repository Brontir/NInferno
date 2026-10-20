# Nemotron 3.5 Lightning 30B-A3B on the RTX 4090

[RTX 4090](../../gpus/rtx4090.md) · [Nemotron 3.5 Lightning 30B-A3B](../../models/nemotron-3.5-lightning-30b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 265 tok/s:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-IQ3_XXS --spec none
```

8 concurrent chats, 751 tok/s over all streams:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-IQ3_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` UD-IQ3_XXS | off | auto | 13,154 | 252 | 265 | 602 | 751 | 2026-10-20 |

## Measurement

- NVIDIA GeForce RTX 4090: driver 580.95.05, CUDA 13.0, power limit 450 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
