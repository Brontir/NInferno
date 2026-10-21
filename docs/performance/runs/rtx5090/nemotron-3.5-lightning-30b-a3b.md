# Nemotron 3.5 Lightning 30B-A3B on the RTX 5090

[RTX 5090](../../gpus/rtx5090.md) · [Nemotron 3.5 Lightning 30B-A3B](../../models/nemotron-3.5-lightning-30b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 488 tok/s:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 3
```

8 concurrent chats, 1,077 tok/s over all streams:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-Q4_K_XL --with-drafters --spec mtp --draft-tokens 3 --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Activations | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` UD-Q4_K_XL | mtp ×3 | auto | fast | 14,587 | 547 | 488 | 860 | 1,077 | 2026-10-21 |
| `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` UD-Q4_K_XL | off | auto | fast | 15,404 | 356 | 377 | 807 | 973 | 2026-10-21 |

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
