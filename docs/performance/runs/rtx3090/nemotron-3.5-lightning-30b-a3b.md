# Nemotron 3.5 Lightning 30B-A3B on the RTX 3090

[RTX 3090](../../gpus/rtx3090.md) · [Nemotron 3.5 Lightning 30B-A3B](../../models/nemotron-3.5-lightning-30b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 279 tok/s:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-IQ3_XXS --with-drafters --spec mtp --draft-tokens 3
```

8 concurrent chats, 642 tok/s over all streams:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-IQ3_XXS --with-drafters --spec mtp --draft-tokens 3 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` UD-IQ3_XXS | mtp ×3 | auto | fast | 5,834 | 304 | 279 | 511 | 642 | 2026-10-20 |
| `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` UD-IQ3_XXS | off | auto | fast | 6,150 | 203 | 212 | 457 | 563 | 2026-10-20 |

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
