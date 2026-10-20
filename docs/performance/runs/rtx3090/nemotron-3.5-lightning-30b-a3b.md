# Nemotron 3.5 Lightning 30B-A3B on the RTX 3090

[RTX 3090](../../gpus/rtx3090.md) · [Nemotron 3.5 Lightning 30B-A3B](../../models/nemotron-3.5-lightning-30b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 208 tok/s:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-IQ3_XXS --spec none
```

8 concurrent chats, 560 tok/s over all streams:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-IQ3_XXS --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` UD-IQ3_XXS | off | auto | 6,128 | 210 | 208 | 451 | 560 | 2026-10-20 |

## Measurement

- NVIDIA GeForce RTX 3090: driver 580.159.03, CUDA 13.0, power limit 350 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
