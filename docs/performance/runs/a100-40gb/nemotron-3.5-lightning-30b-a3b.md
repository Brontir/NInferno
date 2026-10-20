# Nemotron 3.5 Lightning 30B-A3B on the A100 40 GB

[A100 40 GB](../../gpus/a100-40gb.md) · [Nemotron 3.5 Lightning 30B-A3B](../../models/nemotron-3.5-lightning-30b-a3b.md) · [all GPUs and models](../../README.md)

## Best launch

One stream, 228 tok/s:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-Q4_K_XL --spec none
```

8 concurrent chats, 517 tok/s over all streams:

```bash
ninferno-serve unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --quant UD-Q4_K_XL --spec none --max-concurrency 8
```

## Every setting measured

Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* concurrent short chats, all streams together.

| Weights | Speculation | KV | Prefill 24K | Decode @24K | C=1 | C=4 | C=8 | Measured |
|---|---|---|---:|---:|---:|---:|---:|---|
| `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` UD-Q4_K_XL | off | auto | 5,836 | 218 | 228 | 431 | 517 | 2026-10-20 |

## Measurement

- NVIDIA A100-SXM4-40GB: driver 595.99.02, CUDA 13.2, power limit 300 W.
- NInferno 0.1.0.
- How each number is taken: [methodology](../../methodology.md).

Server flags of the measured settings:

```
--max-context 32768 --max-concurrency 1 --spec none
--max-context 32768 --max-concurrency 4 --spec none
--max-context 32768 --max-concurrency 8 --spec none
```
